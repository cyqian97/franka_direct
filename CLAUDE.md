# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What This Repo Is

Standalone teleoperation stack for the **Franka FR3** arm, extracted from the
[DROID](https://droid-dataset.github.io) data-collection framework.

**Why it exists:** The original DROID Polymetis backend calls `ControlUpdate`
gRPC *inside* the 1 kHz real-time callback (~0.88 ms of a 1 ms budget), causing
~50% `communication_constraints_violation` faults. This repo replaces that path
with a C++ gRPC server (`franka_server.cpp`) where gRPC lives entirely outside
the RT loop — 0% violation rate.

## Hardware

| Device | IP | Role |
|---|---|---|
| NUC | 192.168.1.6 | RT Ubuntu, runs Docker container with franka_server |
| Laptop | 192.168.1.1 | Runs VR controller + IK + gRPC client |
| Franka FR3 | 192.168.1.11 | EtherCAT connected to NUC |
| Oculus Quest 3 | — | VR controller, connected to laptop via USB/ADB |

All IPs and robot type live in `parameters.py` at repo root.

## Build Commands

### C++ servers (run inside NUC Docker container)
```bash
docker exec franka_direct_nuc bash /app/build.sh
```
Produces `build/franka_server` and `build/franka_server_cartesian`.
Rebuild required after any change to `.cpp` or `.proto` files.

### Python gRPC stubs (after any `.proto` change)
```bash
bash python/generate_stubs.sh
```
Generates `python/franka_control_pb2.py` and `python/franka_control_pb2_grpc.py`.
The generated files are committed for convenience.

### Python package
```bash
pip install -e .
# Then install dm-robotics with --no-deps (conflicting transitive deps):
pip install --no-deps dm-robotics-moma==0.5.0 dm-robotics-transformations==0.5.0 \
    dm-robotics-agentflow==0.5.0 dm-robotics-geometry==0.5.0 \
    dm-robotics-manipulation==0.5.0 dm-robotics-controllers==0.5.0
```

## Running the System

### 1. Start containers (daily)
```bash
bash scripts/setup/nuc_setup.sh     # on NUC
bash scripts/setup/laptop_setup.sh  # on laptop
```

### 2. Launch gRPC server on NUC
```bash
# Joint torque control (use with simple_teleop_direct_torque.py)
docker exec franka_direct_nuc bash /app/launch_server.sh

# Cartesian velocity control (use with simple_teleop_direct.py)
docker exec franka_direct_nuc bash /app/launch_server_cartesian.sh
```
Server env vars: `ROBOT_IP` (default `192.168.1.11`), `GRPC_ADDR` (default
`0.0.0.0:50052`), `POLICY_HZ` (default `25`, joint server only), `CONFIG_FILE`.

Do **not** run Polymetis `launch_robot.sh` at the same time — the launch scripts
`pkill` Polymetis automatically.

### 3. Teleoperation from laptop container
```bash
# Joint torque + IK (recommended)
docker exec -it franka_direct_laptop \
    python /app/scripts/simple_teleop_direct_torque.py [--left] [--no_reset] [--hz 15]

# Cartesian velocity
docker exec -it franka_direct_laptop \
    python /app/scripts/simple_teleop_direct.py [--left] [--no_reset] [--hz 15]

# With ZED cameras
docker exec -it franka_direct_laptop \
    python /app/scripts/simple_teleop_direct_torque.py --cam0 <sn> --cam1 <sn>
```

### Test scripts (no VR)
```bash
python scripts/simple_joint_direct.py --joints 3   # sinusoidal joint 3 motion
python scripts/simple_pose_direct.py --z_mm -50    # move EE 50 mm down
```

## Architecture

### Network
```
Laptop (192.168.1.1) ──── gRPC :50052 ──── NUC (192.168.1.6) ──── EtherCAT ──── FR3
Quest 3 ──── USB/ADB ──── Laptop
```

### Control path (joint torque)
```
Quest 3 → VRController → pos_delta (m), rot_delta (SO3)
  → absolute T_target (4×4 SE3)
  → pose_to_cartesian_velocity() → RobotIKSolver (dm_robotics MuJoCo FR3)
  → Cartesian velocity → joint velocity → q_target
  → FrankaDirectClient.set_joint_target(q_target)  [15 Hz gRPC]
  → franka_server.cpp SharedState (mutex)
  → 1 kHz RT callback: τ = Kp(interp_q − q) − Kd·dq + coriolis
  → franka::Torques → FR3
```

### Control path (Cartesian velocity)
```
Quest 3 → VRController → T_target (4×4)
  → FrankaDirectClient.set_ee_target(T_target)  [15 Hz gRPC]
  → franka_server_cartesian.cpp SharedState
  → 1 kHz RT callback: Cartesian PD → franka::CartesianVelocities → FR3
```

## Key Design Details

### Pose encoding
All poses are 4×4 homogeneous transforms stored as 16 doubles in **column-major**
order (matching libfranka `O_T_EE`):
```python
pose16_to_mat = lambda p: np.array(p).reshape(4, 4, order='F')
mat_to_pose16 = lambda T: T.flatten(order='F').tolist()
```

### Torque formula (matches Polymetis DefaultController)
```
τ[i] = Kp[i] × (interp_q[i] − q[i]) − Kd[i] × dq[i] + coriolis[i]
τ[i] = clamp(τ[i], −tau_limit[i], +tau_limit[i])
```
- `interp_q` linearly interpolates toward `goal_q` by `max_step` per tick
- `coriolis = model.coriolis(state)` — libfranka adds gravity automatically
- Default gains: Kp = [10, 7.5, 12.5, 6.25, 8.25, 6.25, 2.5] (softer than
  Polymetis defaults to reduce oscillation at 15 Hz command rate)
- Clamp: ±[87, 87, 87, 87, 12, 12, 12] Nm

### Why `franka::Torques` not `franka::JointPositions`
`JointPositions` always triggers `joint_motion_generator_velocity/acceleration_discontinuity`
at startup on FR3 (around tick 15), even returning `rs.q`. Torque control bypasses
the motion generator entirely.

### Startup / recovery pattern
1. Wait for first `SetJointTarget` **before** calling `robot.control()` — avoids
   entering the RT loop with no goal
2. After `ControlException`: `automaticErrorRecovery()`, then wait for a new
   `SetJointTarget` before re-entering `robot.control()`
3. Seed `interp_q` from `robot.readOnce()` before each `robot.control()` call

### VRController coordinate transform chain
```python
transformed = global_to_env_mat @ vr_to_global_mat @ raw
```
- `vr_to_global_mat`: set at joystick press — full `inv(raw)` to zero out the
  current controller pose; continuously updated at startup until grip is held
- `global_to_env_mat`: signed permutation matrix from `rmat_reorder` argument
  (default `(-2, -1, -3, 4)` — matches original DROID VRPolicy axis mapping)
- `spatial_coeff`: scalar multiplier on position delta (default 1.0)

### VRController origin reset
On grip toggle, `_reset_origin = True`. On the next `get_pose_delta()` call,
the current VR pose is saved as origin and `(None, None, None)` is returned,
which signals the teleop script to capture the current robot EEF pose as
`robot_origin`. Subsequent deltas are computed relative to both origins.

### IK solver (RobotIKSolver)
- dm_robotics `Cartesian6dVelocityEffector` with MuJoCo FR3 model (`robot_ik/franka/fr3.xml`)
- Jacobian-based velocity IK, Tikhonov regularisation λ=0.01
- Joint position limit avoidance (0.3 rad margin)
- Nullspace control toward q=0 (gain=0.025)
- Per-step limits: position 75 mm, rotation 0.15 rad, joint 0.2 rad

### ZED camera recording
`scripts/zed_utils.py` provides `CameraRecorder` (two-camera MP4, frame-locked
to control loop rate) and `list_cameras()` / `open_camera()` helpers.
- `ZED_AVAILABLE` flag: scripts degrade gracefully if `pyzed` is not installed
- `--cam0` / `--cam1` serial numbers optional; `None` = auto-detect first two cameras
- A button starts recording, B button stops and saves

### CMakeLists.txt libfranka search order
1. Polymetis third-party vendored build at
   `/app/droid/fairo/polymetis/polymetis/src/clients/franka_panda_client/third_party/libfranka/`
2. Conda `polymetis-local` env
3. System `/usr/local/lib`, `/usr/lib`

This means the build inside the existing NUC Docker container uses the same
libfranka that Polymetis was compiled against, ensuring ABI compatibility.

## Key Files

| File | Purpose |
|---|---|
| `src/franka_server.cpp` | Joint torque gRPC server (1 kHz RT loop) |
| `src/franka_server_cartesian.cpp` | Cartesian velocity gRPC server |
| `proto/franka_control.proto` | gRPC service definition (SetJointTarget, SetEETarget, GetRobotState, …) |
| `python/franka_direct_client.py` | Python gRPC client; `get_robot_state()` returns `target_q` — use this, not `q`, as the base for new commands |
| `config/controller.yaml` | Joint torque gains — hot-reloadable on server restart |
| `config/controller_cartesian.yaml` | Cartesian gains — hot-reloadable |
| `build.sh` | CMake build (run inside NUC container) |
| `launch_server.sh` | Start joint torque server |
| `launch_server_cartesian.sh` | Start Cartesian velocity server |
| `CMakeLists.txt` | C++ build; conda `polymetis-local` provides gRPC/Protobuf |
| `robot_ik/robot_ik_solver.py` | dm_robotics IK wrapper |
| `robot_ik/arm.py` | MuJoCo FrankaArm; reads `robot_type` from `parameters.py` |
| `robot_ik/franka/fr3.xml` | MuJoCo FR3 model (+ meshes) used by IK |
| `scripts/vr_controller.py` | `VRController` — Quest 3 → pose deltas via OculusReader |
| `scripts/simple_teleop_direct_torque.py` | Main teleoperation script (joint torque + IK) |
| `scripts/simple_teleop_direct.py` | Teleoperation via Cartesian server |
| `scripts/zed_utils.py` | ZED camera utilities (`list_cameras`, `CameraRecorder`) |
| `scripts/simple_joint_direct.py` | Test: sinusoidal joint motion |
| `scripts/simple_pose_direct.py` | Test: fixed EE delta move |
| `scripts/test_vr_readout.py` | Test: print raw Quest controller output |
| `parameters.py` | Hardware IPs, robot type, ZED serials — edit here to reconfigure |
| `oculus_reader/` | Git submodule: Quest ADB reader + APK (APK tracked via Git LFS) |

## Configuration

### parameters.py
The single source of truth for hardware configuration. Read by `robot_ik/arm.py`
(robot_type → selects FR3 or Panda MuJoCo XML) and by the setup scripts.
```python
nuc_ip    = "192.168.1.6"
laptop_ip = "192.168.1.1"
robot_ip  = "192.168.1.11"
robot_type        = "fr3"
libfranka_version = "0.14.0"
zed_cam0_serial   = None   # None = auto-detect
zed_cam1_serial   = None
grpc_port         = 50052
```

### Controller gains
`config/controller.yaml` and `config/controller_cartesian.yaml` are read at
server startup. Edit and restart the server — no rebuild needed.

Current joint gains are intentionally **softer than Polymetis defaults**
(Kp ÷4, Kd ÷2) to reduce oscillation when the Python command rate is 15 Hz
rather than Polymetis's ~2 kHz.

## Docker

Two containers, both using the existing DROID images:

| Container | Image | Role |
|---|---|---|
| `franka_direct_nuc` | `droid-nuc-test` | Runs franka_server, accesses robot via EtherCAT |
| `franka_direct_laptop` | `ghcr.io/droid-dataset/droid_laptop:fr3` | Runs Python scripts, VR + camera |

Docker Compose files in `.docker/nuc/` and `.docker/laptop/`. The NUC container
mounts `src/`, `proto/`, `python/`, `config/`, `build.sh`, and launch scripts so
edits on the host take effect without rebuilding the image.

## Submodules
```bash
git submodule update --init --recursive
```
- `oculus_reader/` — Quest ADB reader; APK binary tracked via Git LFS (`*.apk`)

## Protobuf Version Lock
`protobuf==3.20.1` is pinned in `pyproject.toml`. The C++ Protobuf version
(bundled with the conda gRPC install) must remain compatible. Do not upgrade
either side independently.
