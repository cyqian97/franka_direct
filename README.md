# franka_direct

Standalone teleoperation stack for the Franka FR3 arm using a Meta Quest 3
controller. Replaces the Polymetis gRPC backend with a direct libfranka C++
server, eliminating `communication_constraints_violation` faults caused by
blocking gRPC calls inside the 1 kHz real-time loop.

## Architecture

```
Quest 3 ──ADB──► Laptop
                  VRController
                  RobotIKSolver (dm_robotics MuJoCo)
                  FrankaDirectClient
                       │ gRPC :50052
                       ▼
                    NUC (RT Ubuntu)
                  franka_server (C++)
                       │ EtherCAT
                       ▼
                    Franka FR3
```

**Control path (joint torque):**
```
Quest 3 → pos_delta, rot_delta
  → absolute T_target (4×4 SE(3))
  → IK: Cartesian velocity → joint delta
  → q_target → gRPC → franka_server.cpp
  → τ = Kp(q_target − q) − Kd·dq + coriolis  (1 kHz RT)
```

**Why not Polymetis:** Polymetis calls `ControlUpdate` gRPC *inside* the 1 kHz
RT callback (~0.88 ms of a 1 ms budget), causing ~50% fault rate. Here gRPC
is entirely outside the RT loop.

## Hardware

| Device | IP | Role |
|---|---|---|
| Laptop | 192.168.1.1 | Runs VR controller + IK + gRPC client |
| NUC | 192.168.1.6 | Runs franka_server (RT Ubuntu, EtherCAT) |
| Franka FR3 | 192.168.1.11 | Robot arm |
| Oculus Quest 3 | — | VR teleoperation controller (USB/ADB) |

Edit `parameters.py` to change IPs or ZED camera serial numbers.

---

## File Structure

```
franka_direct/
├── src/                            C++ gRPC server source
│   ├── franka_server.cpp           Joint torque control server
│   └── franka_server_cartesian.cpp Cartesian velocity control server
├── proto/
│   └── franka_control.proto        gRPC service definition
├── python/
│   ├── franka_direct_client.py     Python gRPC client
│   ├── generate_stubs.sh           Regenerate pb2 stubs after proto changes
│   ├── franka_control_pb2.py       Generated (committed for convenience)
│   └── franka_control_pb2_grpc.py  Generated
├── config/
│   ├── controller.yaml             Joint torque gains (hot-reloadable)
│   └── controller_cartesian.yaml   Cartesian gains (hot-reloadable)
├── robot_ik/
│   ├── robot_ik_solver.py          dm_robotics Jacobian IK wrapper
│   ├── arm.py                      MuJoCo FrankaArm model
│   └── franka/                     FR3 / Panda MJCF XML + meshes
├── scripts/
│   ├── setup/
│   │   ├── nuc_setup.sh            First-time + daily NUC setup
│   │   └── laptop_setup.sh         First-time + daily laptop setup
│   ├── vr_controller.py            VRController: Quest 3 → pose deltas
│   ├── simple_teleop_direct.py     Teleoperation via Cartesian server
│   ├── simple_teleop_direct_torque.py  Teleoperation via joint torque + IK
│   ├── simple_joint_direct.py      Test: sinusoidal joint motion
│   ├── simple_pose_direct.py       Test: move EE by a fixed delta
│   ├── test_vr_readout.py          Test: print raw VR controller output
│   └── zed_utils.py                ZED camera recording utilities
├── oculus_reader/                  Git submodule (Quest ADB reader + APK)
├── .docker/
│   ├── nuc/
│   │   ├── Dockerfile.nuc
│   │   └── docker-compose-nuc.yaml
│   └── laptop/
│       ├── Dockerfile.laptop
│       └── docker-compose-laptop.yaml
├── build.sh                        CMake build (run inside NUC container)
├── launch_server.sh                Launch joint torque server
├── launch_server_cartesian.sh      Launch Cartesian velocity server
├── CMakeLists.txt
├── parameters.py                   Hardware config (IPs, ZED serials)
└── pyproject.toml
```

---

## Setup

### Prerequisites

- NUC: real-time Ubuntu (RT kernel), Docker, EtherCAT connection to FR3
- Laptop: Ubuntu 22.04, Docker, NVIDIA GPU + drivers, ADB

### 1. Clone the repo

```bash
git clone <repo_url> ~/franka_direct
cd ~/franka_direct
git lfs install
git submodule update --init --recursive
```

### 2. Edit parameters.py

```python
nuc_ip    = "192.168.1.6"
laptop_ip = "192.168.1.1"
robot_ip  = "192.168.1.11"
robot_type = "fr3"          # or "panda"
libfranka_version = "0.14.0"

# Set ZED serial numbers, or leave None for auto-detect
zed_cam0_serial = None
zed_cam1_serial = None
```

### 3. Run setup scripts (first time on each machine)

**On the NUC** (as root):
```bash
bash scripts/setup/nuc_setup.sh
```

This installs Docker, applies the RT kernel patch, sets the CPU governor,
configures a static IP, and starts the NUC container.

**On the laptop** (as root):
```bash
bash scripts/setup/laptop_setup.sh
```

This installs Docker + NVIDIA container toolkit, installs the Quest APK via
ADB, configures a static IP, starts the ADB server, and starts the laptop
container.

### 4. Build the C++ servers (inside NUC container)

```bash
docker exec franka_direct_nuc bash /app/build.sh
```

This compiles `franka_server` and `franka_server_cartesian` using CMake. Only
needed after cloning or after any C++ / `.proto` changes.

### 5. Regenerate Python gRPC stubs (after .proto changes only)

```bash
bash python/generate_stubs.sh
```

---

## Daily Use

### Start containers

```bash
# NUC
bash scripts/setup/nuc_setup.sh

# Laptop
bash scripts/setup/laptop_setup.sh
```

Or start containers directly if already set up:
```bash
# NUC
docker compose -f .docker/nuc/docker-compose-nuc.yaml up -d

# Laptop
ROOT_DIR=$(pwd) NUC_IP=192.168.1.6 LAPTOP_IP=192.168.1.1 ROBOT_IP=192.168.1.11 \
LIBFRANKA_VERSION=0.14.0 DISPLAY=$DISPLAY DOCKER_XAUTH=/tmp/.docker.xauth \
docker compose -f .docker/laptop/docker-compose-laptop.yaml up -d
```

### Launch the gRPC server on NUC

```bash
# Joint torque control (use with simple_teleop_direct_torque.py)
docker exec franka_direct_nuc bash /app/launch_server.sh

# Cartesian velocity control (use with simple_teleop_direct.py)
docker exec franka_direct_nuc bash /app/launch_server_cartesian.sh
```

Do **not** run Polymetis `launch_robot.sh` at the same time — the launch
scripts `pkill` Polymetis automatically.

### Teleoperation from laptop container

```bash
# Joint torque + IK (recommended)
docker exec -it franka_direct_laptop \
    python /app/scripts/simple_teleop_direct_torque.py [--left] [--no_reset] [--hz 15]

# Cartesian velocity
docker exec -it franka_direct_laptop \
    python /app/scripts/simple_teleop_direct.py [--left] [--no_reset] [--hz 15]
```

### VR Controller

| Input | Action |
|---|---|
| Hold **GRIP TRIGGER** | Enable robot movement |
| **INDEX TRIGGER** | Proportional gripper (squeeze = close) |
| **JOYSTICK press** | Recalibrate yaw orientation |
| **A** (right) / **X** (left) | Start recording |
| **B** (right) / **Y** (left) | Stop and save recording |
| `r` + Enter | Reset VR state |
| `q` + Enter | Quit |
| Ctrl+C | Emergency stop |

### ZED camera recording (optional)

```bash
docker exec -it franka_direct_laptop \
    python /app/scripts/simple_teleop_direct_torque.py \
    --cam0 <serial> --cam1 <serial> [--cam_fps 60] [--out_dir /app/recordings]
```

Omit `--cam0`/`--cam1` to run without cameras. Videos are saved as
`recordings/<timestamp>/cam0.mp4` and `cam1.mp4`.

### Test scripts (no VR required)

```bash
# Sinusoidal motion on joint 3
docker exec -it franka_direct_laptop python /app/scripts/simple_joint_direct.py --joints 3

# Move EE down 50 mm
docker exec -it franka_direct_laptop python /app/scripts/simple_pose_direct.py --z_mm -50

# Print raw VR controller output
docker exec -it franka_direct_laptop python /app/scripts/test_vr_readout.py
```

---

## Configuration

### Controller gains (`config/controller.yaml`)

Joint torque gains and limits. Editable without rebuild — the server re-reads
on restart.

```yaml
# Joint stiffness / damping (7 joints)
Kp: [40, 30, 50, 25, 35, 25, 10]
Kd: [4,  6,  5,  5,  3,  2,  1]
# Torque clamp [Nm]
tau_limit: [87, 87, 87, 87, 12, 12, 12]
```

### Cartesian gains (`config/controller_cartesian.yaml`)

Linear and rotational PD gains, velocity limits. Also hot-reloadable.

---

## Build Dependencies (C++)

| Library | Version | Notes |
|---|---|---|
| libfranka | 0.14.0+ | Built from source inside Docker; FR3 requires ≥0.14 |
| gRPC | ≥1.39 | Must match Python `grpcio` |
| Protobuf (C++) | matches `protobuf==3.20.1` | Do not upgrade independently |
| CMake | ≥3.16 | Install from Kitware apt repo |
| Eigen3 | any | `apt install libeigen3-dev` |
| spdlog | any | `apt install libspdlog-dev` |

---

## Python Dependencies

Core packages (see `pyproject.toml`):
```
grpcio  grpcio-tools  protobuf==3.20.1
numpy  scipy  opencv-python==4.6.0.66
dm-control==1.0.5  mujoco==2.3.2
pure-python-adb  pyyaml
```

dm-robotics packages must be installed with `--no-deps`:
```bash
pip install --no-deps \
    dm-robotics-moma==0.5.0 \
    dm-robotics-transformations==0.5.0 \
    dm-robotics-agentflow==0.5.0 \
    dm-robotics-geometry==0.5.0 \
    dm-robotics-manipulation==0.5.0 \
    dm-robotics-controllers==0.5.0
```

ZED SDK (`pyzed`) is installed via the Stereolabs SDK installer inside the
laptop Docker image. Scripts run without cameras if pyzed is not available.