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
│   ├── data_recorder.py            HDF5 episode recorder (used by teleop script)
│   ├── preprocess_franka_data.py   Resize images + split train/val
│   ├── rlds_dataset/
│   │   └── franka_fr3/             TFDS dataset builder (HDF5 → RLDS)
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

| Input | Without `--task` | With `--task` (data recording) |
|---|---|---|
| Hold **GRIP TRIGGER** | Enable robot movement | Enable robot movement |
| **INDEX TRIGGER** | Proportional gripper | Proportional gripper |
| **JOYSTICK press** | Recalibrate yaw orientation | Recalibrate yaw orientation |
| **A** (right) / **X** (left) | Start video recording | Start episode recording |
| **B** (right) / **Y** (left) | Stop and save video | Save episode + reset to home |
| `r` + Enter | Reset VR state | Reset VR state |
| `q` + Enter | Quit | Quit (saves in-progress episode) |
| Ctrl+C | Emergency stop | Emergency stop (saves in-progress episode) |

### Gripper Control

The gripper is controlled by the **index trigger** (front of controller), not the grip trigger (side).

**VR trigger types**

The OculusReader APK exposes two versions of each trigger:

| Key | Type | Used for |
|---|---|---|
| `RG` / `LG` | **bool** | Grip trigger — enables arm movement (on/off) |
| `rightGrip` / `leftGrip` | **float** 0–1 | Grip trigger analog — available but unused |
| `RTr` / `LTr` | **bool** | Index trigger threshold — unused |
| `rightTrig` / `leftTrig` | **float** 0–1 | Index trigger analog — controls gripper |

The index trigger returns a continuous float, so the gripper has proportional control:
- Trigger fully released → gripper fully open (80 mm)
- Trigger fully squeezed → gripper fully closed (0 mm)

**Control path**

```
index_trig ∈ [0, 1]  (float, 15 Hz)
  → gripper_target = (1 − index_trig) × 0.08 m
  → deadband filter: only send gRPC if |change| > 2 mm
  → SetGripperTarget gRPC  (returns immediately, non-blocking)
  → C++ gripper thread (separate from 1 kHz RT arm loop)
      width > 40 mm → gripper.move(width, speed)    — open/position mode
      width ≤ 40 mm → gripper.grasp(width, speed,   — grasp mode
                           force=20 N, ε=0.08 m)
  → Franka Hand fingers
```

The gripper runs in a dedicated thread completely independent of the 1 kHz arm RT loop. `gripper.move()` and `gripper.grasp()` are blocking calls — the thread waits until the fingers reach position before accepting a new command. If a new `SetGripperTarget` arrives while a move is in progress, it is queued and executed immediately after.

**move vs grasp**

Below 40 mm the server switches from `move()` to `grasp()`. `grasp()` applies a
configurable closing force (default 20 N) and succeeds even if an object stops the
fingers early (tolerances set to ±80 mm). `move()` targets an exact width with no
extra force. The thresholds and force are tunable in `config/controller.yaml`:

```yaml
gripper_force:   20.0   # closing force [N]
gripper_eps_in:   0.08  # grasp epsilon_inner [m]
gripper_eps_out:  0.08  # grasp epsilon_outer [m]
```

**Recorded data**

| Field | Value |
|---|---|
| `action[-1]` | `last_sent_command / 0.08` — commanded normalized width (policy action) |
| `observations/gripper` | `measured_width / 0.08` — actual normalized width (sensor) |

Both are normalized to [0 = closed, 1 = open]. They differ when the deadband
suppresses a command or when the gripper is blocked by an object.

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

## Data Recording for OpenVLA-OFT Fine-tuning

The teleop script can record imitation-learning episodes to HDF5 files while
you operate the robot. Each episode stores joint state, EEF pose, images (if
cameras are attached), and two action representations — absolute joint targets
and delta-EEF — so you can experiment with either action space without
re-recording.

### Step 1 — Record episodes

Pass `--task` to enable HDF5 recording. Optionally add `--cam0`/`--cam1` for
synchronized camera frames.

```bash
docker exec -it franka_direct_laptop \
    python /app/scripts/simple_teleop_direct_torque.py \
    --task "pick up the red cup" \
    --cam0 <serial0> --cam1 <serial1> \
    --data_dir /app/data/pick_up_cup
```

**Per-episode workflow:**

1. The robot resets to home at startup.
2. Press **A** to start recording an episode.
3. Hold the **GRIP TRIGGER** and move the robot to complete the task.
4. Press **B** to save the episode. The robot resets to home automatically.
5. Repeat from step 2 for each new episode.
6. Press `q` + Enter (or Ctrl+C) to quit. Any in-progress episode is saved.

Each saved file is `data/<task_dir>/episode_XXXXXX.hdf5`.

**HDF5 layout per episode:**

```
/observations/
    qpos        [T, 7]   joint positions (rad)
    qvel        [T, 7]   joint velocities (rad/s)
    ee_pose     [T, 16]  O_T_EE column-major (m, base frame)
    gripper     [T, 1]   normalized gripper [0=closed, 1=open]
    images/
        primary [T, H, W, 3]  RGB, 3rd-person camera
        wrist   [T, H, W, 3]  RGB, wrist/secondary camera
/action         [T, 8]   absolute joint targets (7) + gripper_norm (1)
/action_eef     [T, 7]   delta EEF: pos_delta (3, m) + rot_vec (3, rad) + gripper_norm (1)
@language_instruction    task description string
@sim                     False
```

### Step 2 — Preprocess (resize images + split train/val)

openvla-oft expects 256×256 images. This script resizes from the native ZED
resolution and splits into `train/` and `val/` sub-directories.

```bash
python scripts/preprocess_franka_data.py \
    --data_dir  data/pick_up_cup \
    --out_dir   data/pick_up_cup_256 \
    --percent_val 0.1 \
    --img_size  256
```

### Step 3 — Build the RLDS dataset

Convert the preprocessed HDF5 files to TensorFlow Datasets (RLDS) format,
which is what openvla-oft's data loader consumes.

```bash
# Point the builder at your preprocessed directory and desired output path
FRANKA_FR3_DATA_DIR=data/pick_up_cup_256 \
python scripts/rlds_dataset/franka_fr3/franka_fr3_dataset_builder.py \
    --preprocessed_dir data/pick_up_cup_256 \
    --out_dir          /path/to/rlds_datasets/
```

This writes a TFDS dataset named `franka_fr3` under `/path/to/rlds_datasets/`.
Run it once; subsequent fine-tuning runs load directly from RLDS.

---

## OpenVLA-OFT Fine-tuning

After the RLDS dataset is built, fine-tune using the
[openvla-oft](https://github.com/moojink/openvla-oft) repo. Both action spaces
recorded above are registered and ready to use.

| Dataset name | Action space | When to use |
|---|---|---|
| `franka_fr3` | Absolute joint targets (8D) | Default; no accumulation at inference |
| `franka_fr3_eef` | Delta EEF pos+rot (7D) | If you want EEF-space generalization |

The openvla-oft repo (`~/cyqian/openvla-oft`) already has the required entries
in `configs.py`, `transforms.py`, `mixtures.py`, `materialize.py`, and
`constants.py`. No further changes to that repo are needed.

### Fine-tune with joint-position actions (recommended)

```bash
cd ~/cyqian/openvla-oft

torchrun --standalone --nnodes 1 --nproc-per-node <NUM_GPUS> \
    vla-scripts/finetune.py \
    --vla_path            openvla/openvla-7b \
    --data_root_dir       /path/to/rlds_datasets/ \
    --dataset_name        franka_fr3 \
    --run_root_dir        /path/to/checkpoints/ \
    --use_l1_regression   True \
    --use_diffusion       False \
    --use_film            True \
    --num_images_in_input 2 \
    --use_proprio         True \
    --batch_size          4 \
    --learning_rate       5e-4 \
    --num_steps_before_decay 50000 \
    --max_steps           100005 \
    --use_val_set         True \
    --val_freq            10000 \
    --save_freq           10000 \
    --image_aug           True \
    --lora_rank           32 \
    --wandb_entity        "<YOUR_ENTITY>" \
    --wandb_project       "<YOUR_PROJECT>"
```

`--num_images_in_input 2` uses both the primary and wrist camera images. Set to
`1` if you recorded without cameras (state-only) or want to use only the
primary view.

### Fine-tune with delta-EEF actions

Same command, replace `--dataset_name franka_fr3` with `--dataset_name franka_fr3_eef`.
The normalization scheme switches automatically to `BOUNDS_Q99` (relative
actions benefit from outlier clipping).

### Deployment

After fine-tuning, deploy the model as a gRPC server and query it from the
laptop container. The `unnorm_key` must match the dataset name used during
training:

```bash
# On the GPU machine — serve the fine-tuned checkpoint
python vla-scripts/deploy.py \
    --pretrained_checkpoint /path/to/checkpoints/<run_id>/... \
    --use_l1_regression     True \
    --use_film              True \
    --num_images_in_input   2 \
    --use_proprio           True \
    --center_crop           True \
    --unnorm_key            franka_fr3
```

Then run inference from the laptop container using
`vla-scripts/openvla_oft_franka_client.py` (see that script's docstring for
argument details).

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