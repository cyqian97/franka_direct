#!/usr/bin/env python3
"""
Teleoperation using VR controller + IK + joint torque control.

Pipeline:
  Quest 3 → VRController → pose delta → absolute EEF target T_target (SE(3))
  → pose_to_cartesian_velocity() → RobotIKSolver (dm_robotics, 15 Hz)
  → joint delta Δq → q_target → FrankaDirectClient (gRPC)
  → franka_server.cpp (1 kHz joint impedance torque loop)

Recording:
  All recording parameters are in config/teleop.yaml.

Controls:
  Hold GRIP TRIGGER    → move robot  +  auto-start episode recording
  Release GRIP         → auto-save episode (with label) + reset to home
  INDEX TRIGGER        → proportional gripper (squeeze = close)
  JOYSTICK press       → recalibrate controller orientation
  A / X                → mark current episode as SUCCESS
  B / Y                → mark current episode as FAIL
  r + Enter            → reset VR state
  q + Enter            → quit
  Ctrl+C               → emergency stop (in-progress episode discarded)
"""

import argparse
import os
import re
import select
import shutil
import signal
import sys
import time
from datetime import datetime

import numpy as np
import yaml

# ── Path setup ───────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

try:
    from franka_direct_client import FrankaDirectClient
except ImportError as e:
    print(f"[ERROR] Could not import FrankaDirectClient: {e}")
    print("Did you run:  bash python/generate_stubs.sh ?")
    sys.exit(1)

try:
    from robot_ik.robot_ik_solver import RobotIKSolver
except ImportError as e:
    print(f"[ERROR] Could not import RobotIKSolver: {e}")
    sys.exit(1)

from zed_utils import CameraRecorder, ZED_AVAILABLE
from data_recorder import DataRecorder
from vr_controller import VRController
from robot_reset import reset_to_home, HOME_Q


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load teleop YAML config and return a flat dict with defaults applied."""
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    cam = d.get("camera", {})
    ik  = d.get("ik", {})
    return {
        "task":             d.get("task", ""),
        "data_dir":         d.get("data_dir", "data"),
        "host":             d.get("host", "192.168.1.6"),
        "port":             int(d.get("port", 50052)),
        "hz":               int(d.get("hz", 15)),
        "left_controller":  bool(d.get("left_controller", False)),
        "spatial_coeff":    float(d.get("spatial_coeff", 1.0)),
        "no_reset":         bool(d.get("no_reset", False)),
        "ik_max_lin_delta":     float(ik.get("max_lin_delta",    0.075)),
        "ik_max_rot_delta":     float(ik.get("max_rot_delta",    0.15)),
        "ik_max_joint_delta":   float(ik.get("max_joint_delta",  0.2)),
        "ik_max_gripper_delta": float(ik.get("max_gripper_delta",0.25)),
        "ik_control_hz":        int(  ik.get("control_hz",       15)),
        "cam_enabled":      bool(cam.get("enabled", False)),
        "cam_serial0":      cam.get("serial0"),    # int or None
        "cam_serial1":      cam.get("serial1"),
        "cam_fps":          int(cam.get("fps", 15)),
        "cam_resolution":   str(cam.get("resolution", "HD720")),
        "cam_depth":        bool(cam.get("depth", False)),
        "cam_preview":      bool(cam.get("preview", True)),
        "gripper_open":     float(d.get("gripper", {}).get("open_m",   0.08)),
        "gripper_speed":    float(d.get("gripper", {}).get("speed",    0.1)),
        "gripper_deadband": float(d.get("gripper", {}).get("deadband", 0.002)),
        "reset_speed":      float(d.get("reset", {}).get("speed", 0.2)),
        "reset_randomize":  bool( d.get("reset", {}).get("randomize", False)),
        "reset_pos_low":    list( d.get("reset", {}).get("noise", {}).get("pos_low",  [-0.10, -0.20, -0.10])),
        "reset_pos_high":   list( d.get("reset", {}).get("noise", {}).get("pos_high", [ 0.10,  0.20,  0.10])),
        "reset_rot_low":    list( d.get("reset", {}).get("noise", {}).get("rot_low",  [-0.30, -0.30, -0.30])),
        "reset_rot_high":   list( d.get("reset", {}).get("noise", {}).get("rot_high", [ 0.30,  0.30,  0.30])),
    }


# ── Pose math helpers ─────────────────────────────────────────────────────────

def pose16_to_mat(pose16):
    return np.array(pose16).reshape(4, 4, order='F')


def mat_to_pose16(T):
    return T.flatten(order='F').tolist()


def rotation_error_vec(R_target, R_current):
    """Rotation error as axis-angle vector (ω ∈ ℝ³, ||ω|| = angle in rad)."""
    R_err = R_target @ R_current.T
    cos_a = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if angle < 1e-10:
        return np.zeros(3)
    k = angle / (2.0 * np.sin(angle))
    return k * np.array([R_err[2, 1] - R_err[1, 2],
                          R_err[0, 2] - R_err[2, 0],
                          R_err[1, 0] - R_err[0, 1]])


def pose_to_cartesian_velocity(T_target, T_current, ik):
    """Convert pose error to normalised 6D Cartesian velocity for RobotIKSolver."""
    p_err   = T_target[:3, 3] - T_current[:3, 3]
    rot_err = rotation_error_vec(T_target[:3, :3], T_current[:3, :3])
    lin_vel = p_err   / ik.max_lin_delta
    rot_vel = rot_err / ik.max_rot_delta
    lin_norm = np.linalg.norm(lin_vel)
    if lin_norm > 1.0:
        lin_vel /= lin_norm
    rot_norm = np.linalg.norm(rot_vel)
    if rot_norm > 1.0:
        rot_vel /= rot_norm
    return np.concatenate([lin_vel, rot_vel])


# ── Keyboard helper ───────────────────────────────────────────────────────────

def check_keyboard():
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.readline().strip().lower()
    return None


# ── Status printer ────────────────────────────────────────────────────────────

def print_status(step, enabled, recording, waiting, label, pos_err_mm, rot_err_deg, hz, gripper):
    if waiting:
        rec_str = "WAIT A=ok / B=fail "
    elif recording:
        rec_str = f"REC({label or 'unlabeled'})"
    else:
        rec_str = "                  "
    sys.stdout.write(
        f"\r[{step:>6}] {'MOVING' if enabled else 'PAUSED':<7} {rec_str:<19} | "
        f"pos={pos_err_mm:>6.1f}mm  rot={rot_err_deg:>5.1f}° | "
        f"Hz={hz:>5.1f} | grip={gripper*1000:>4.0f}mm  "
    )
    sys.stdout.flush()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Franka FR3 teleoperation via VR + IK + joint torque control")
    parser.add_argument("--config", default="config/teleop.yaml",
                        help="Path to YAML config (default: config/teleop.yaml)")
    cli = parser.parse_args()

    cfg = load_config(cli.config)
    right_controller = not cfg["left_controller"]
    loop_period      = 1.0 / cfg["hz"]

    # ── Ctrl+C handler ────────────────────────────────────────────────────────
    running = True

    def _sigint(sig, frame):
        nonlocal running
        if not running:
            os._exit(1)
        print("\n\nCtrl+C detected. Stopping...")
        running = False

    signal.signal(signal.SIGINT, _sigint)

    print("=" * 62)
    print("Teleoperation  —  VR + IK + joint torque control")
    print("=" * 62)
    print(f"  Config : {cli.config}")
    print(f"  Task   : \"{cfg['task']}\"" if cfg["task"] else "  Task   : (none — recording disabled)")

    # === Step 1: Connect to franka_server =====================================
    print(f"\nConnecting to franka_server at {cfg['host']}:{cfg['port']} ...")
    client = FrankaDirectClient(host=cfg["host"], port=cfg["port"])
    try:
        state = client.wait_until_ready(timeout=20.0)
        print(f"[OK] franka_server ready  (cmd_success_rate={state['cmd_success_rate']:.3f})")
    except (TimeoutError, RuntimeError) as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    # === Step 2: Initialize IK solver =========================================
    print("Initializing IK solver ...")
    try:
        ik = RobotIKSolver(
            max_lin_delta=cfg["ik_max_lin_delta"],
            max_rot_delta=cfg["ik_max_rot_delta"],
            max_joint_delta=cfg["ik_max_joint_delta"],
            max_gripper_delta=cfg["ik_max_gripper_delta"],
            control_hz=cfg["ik_control_hz"],
        )
        print(f"[OK] IK solver ready  "
              f"(lin={ik.max_lin_delta*1000:.0f} mm, "
              f"rot={np.degrees(ik.max_rot_delta):.1f}°, "
              f"joint={np.degrees(ik.max_joint_delta):.1f}°  per step)")
    except Exception as e:
        print(f"[FAIL] IK solver: {e}")
        sys.exit(1)

    # === Step 3: Initialize VR controller =====================================
    print("Initializing VR controller ...")
    try:
        vr = VRController(right_controller=right_controller)
        side = "right" if right_controller else "left"
        print(f"[OK] VR controller initialized ({side} hand, "
              f"spatial_coeff={cfg['spatial_coeff']:.2f})")
    except Exception as e:
        print(f"[FAIL] VR controller: {e}")
        sys.exit(1)

    # === Step 4: Initialize data recorder =====================================
    data_rec = None
    if cfg["task"]:
        task_slug   = re.sub(r"[^a-zA-Z0-9_-]", "_", cfg["task"].replace(" ", "_"))
        timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = os.path.join(cfg["data_dir"], task_slug, timestamp)
        os.makedirs(session_dir, exist_ok=True)

        # Snapshot the configs used for this session
        shutil.copy2(cli.config, session_dir)
        controller_yaml = os.path.join(REPO_ROOT, "config", "controller.yaml")
        if os.path.exists(controller_yaml):
            shutil.copy2(controller_yaml, session_dir)

        data_rec = DataRecorder(out_dir=session_dir, gripper_open_m=cfg["gripper_open"])
        print(f"[OK] Data recorder → {session_dir}/")
        print(f"     Task: \"{cfg['task']}\"")

    # === Step 5: Initialize cameras ===========================================
    recorder = None
    if cfg["cam_enabled"]:
        if not ZED_AVAILABLE:
            print("[WARN] pyzed not installed — camera disabled.")
        else:
            try:
                recorder = CameraRecorder(
                    serial0=cfg["cam_serial0"],
                    serial1=cfg["cam_serial1"],
                    fps=cfg["cam_fps"],
                    resolution=cfg["cam_resolution"],
                    out_dir=os.path.join(cfg["data_dir"], "videos"),
                    depth=cfg["cam_depth"],
                    preview=cfg["cam_preview"],
                )
                recorder.open()
            except RuntimeError as e:
                print(f"[WARN] Camera init failed: {e} — continuing without cameras.")
                recorder = None

    # === Step 6: Optionally reset robot to home ================================
    if not cfg["no_reset"]:
        print("Resetting robot to home position ...")
        ok, msg = reset_to_home(client, ik, cfg)
        print("[OK] At home" if ok else f"[WARN] Reset: {msg}")
    else:
        print("[SKIP] Robot reset skipped (no_reset=true in config)")

    # === Step 7: Print controls ================================================
    btn_a = "A" if right_controller else "X"
    btn_b = "B" if right_controller else "Y"
    print()
    print("=" * 62)
    print("TELEOPERATION READY")
    print("=" * 62)
    print(f"  Hold GRIP TRIGGER    → move robot" +
          ("  +  start episode recording" if data_rec else ""))
    print(f"  Release GRIP         → save episode + reset to home" if data_rec else
          f"  Release GRIP         → stop movement")
    print(f"  INDEX TRIGGER        → proportional gripper (squeeze = close)")
    print(f"  JOYSTICK press       → recalibrate orientation")
    if data_rec:
        print(f"  '{btn_a}'                  → mark episode as SUCCESS")
        print(f"  '{btn_b}'                  → mark episode as FAIL")
    print(f"  r + Enter            → reset VR state")
    print(f"  q + Enter            → quit")
    print(f"  Ctrl+C               → emergency stop")
    print(f"  Control frequency:   {cfg['hz']} Hz")
    if recorder:
        depth_tag = " + depth" if cfg["cam_depth"] else ""
        print(f"  Camera:              {cfg['cam_resolution']} @ {cfg['cam_fps']} fps{depth_tag}")
    print("=" * 62)
    print()

    # === Step 8: Main control loop ============================================
    step_count        = 0
    robot_origin      = None
    prev_grip         = False
    prev_btn_a        = False
    prev_btn_b        = False
    episode_label     = ""     # "success" | "fail" | "" (unlabeled)
    waiting_for_label = False  # True after grip release, before A/B pressed
    last_q_target     = None
    last_eef_delta    = None

    last_gripper_cmd = cfg["gripper_open"]

    while running:
        loop_start = time.time()
        step_count += 1

        # ── Camera: grab frame ───────────────────────────────────────────────
        if recorder is not None:
            recorder.grab()

        # ── VR controller ────────────────────────────────────────────────────
        info = vr.get_info()
        cur_grip  = info["movement_enabled"]
        cur_a     = info["success"]
        cur_b     = info["failure"]

        if not info["controller_on"]:
            sys.stdout.write("\r[WARN] VR controller lost. Waiting...               ")
            sys.stdout.flush()
            time.sleep(1)
            continue

        # ── Recording: grip press → start; grip release → wait for label ────────
        if data_rec is not None:
            if cur_grip and not prev_grip:
                # Grip just pressed → start new episode
                episode_label     = ""
                waiting_for_label = False
                data_rec.start_episode()
                print(f"\n[REC] Episode {data_rec._episode_count:06d} started")

            elif not cur_grip and prev_grip:
                # Grip just released → stop recording steps, wait for A/B
                if data_rec.is_recording:
                    if data_rec.num_steps > 0:
                        waiting_for_label = True
                        print(f"\n[REC] {data_rec.num_steps} steps captured — "
                              f"press A=success / B=fail ...")
                    else:
                        data_rec.discard_episode()
                        print("\n[REC] Discarded empty episode")

        # ── A/B: save + reset when waiting; otherwise pre-label during recording ─
        if data_rec is not None:
            if waiting_for_label:
                label_pressed = None
                if cur_a and not prev_btn_a:
                    label_pressed = "success"
                elif cur_b and not prev_btn_b:
                    label_pressed = "fail"
                if label_pressed is not None:
                    waiting_for_label = False
                    data_rec.save_episode(task=cfg["task"], label=label_pressed)
                    episode_label  = ""
                    print("      Resetting to home ...")
                    robot_origin   = None
                    last_q_target  = None
                    last_eef_delta = None
                    vr.reset_state()
                    ok, msg = reset_to_home(client, ik, cfg)
                    if not ok:
                        print(f"      [WARN] Reset: {msg}")
            elif data_rec.is_recording:
                # Pre-label while still holding grip (optional early hint)
                if cur_a and not prev_btn_a:
                    episode_label = "success"
                    print(f"\n[LABEL] → success (will confirm on grip release)")
                if cur_b and not prev_btn_b:
                    episode_label = "fail"
                    print(f"\n[LABEL] → fail (will confirm on grip release)")

        prev_grip  = cur_grip
        prev_btn_a = cur_a
        prev_btn_b = cur_b

        # ── Keyboard ─────────────────────────────────────────────────────────
        key = check_keyboard()
        if key == "q":
            print("\n\n[DONE] 'q' pressed — quitting")
            break
        elif key == "r":
            print("\n\nResetting VR controller state ...")
            vr.reset_state()
            robot_origin = None
            print("VR state reset. Hold grip trigger to start again.")
            continue

        # ── Get current robot state ───────────────────────────────────────────
        state = client.get_robot_state()
        if state["error"]:
            print(f"\n[ERROR] Robot error: {state['error']}")
            break

        T_current = pose16_to_mat(state["pose"])

        # ── Arm: move when grip trigger held ──────────────────────────────────
        pos_err_mm  = 0.0
        rot_err_deg = 0.0

        if info["movement_enabled"]:
            pos_delta, rot_delta, _ = vr.get_pose_delta()

            if vr.origin_just_reset:
                robot_origin = T_current.copy()

            elif pos_delta is not None and robot_origin is not None:
                # Apply spatial_coeff to position delta
                pos_delta_scaled = pos_delta * cfg["spatial_coeff"]

                T_target = np.eye(4)
                T_target[:3, :3] = rot_delta @ robot_origin[:3, :3]
                T_target[:3, 3]  = robot_origin[:3, 3] + pos_delta_scaled

                cart_vel = pose_to_cartesian_velocity(T_target, T_current, ik)

                robot_state_dict = {
                    "joint_positions":  state["q"],
                    "joint_velocities": state["dq"] if state["dq"] else [0.0] * 7,
                }
                joint_vel   = ik.cartesian_velocity_to_joint_velocity(cart_vel, robot_state_dict)
                joint_delta = ik.joint_velocity_to_delta(joint_vel)
                q_target    = (np.array(state["q"]) + joint_delta).tolist()
                client.set_joint_target(q_target)
                last_q_target = q_target

                rot_vec = rotation_error_vec(rot_delta, np.eye(3))
                last_eef_delta = np.concatenate([pos_delta_scaled, rot_vec])

                pos_err_mm  = np.linalg.norm(T_target[:3, 3] - T_current[:3, 3]) * 1000.0
                rot_err_deg = np.degrees(np.arccos(np.clip(
                    (np.trace(T_target[:3, :3] @ T_current[:3, :3].T) - 1.0) / 2.0,
                    -1.0, 1.0)))

        # ── Gripper ───────────────────────────────────────────────────────────
        trig_key   = "rightTrig" if right_controller else "leftTrig"
        index_trig = vr._state["buttons"].get(trig_key, (0.0,))
        if isinstance(index_trig, (tuple, list)):
            index_trig = index_trig[0]

        gripper_target = (1.0 - float(index_trig)) * cfg["gripper_open"]
        if abs(gripper_target - last_gripper_cmd) > cfg["gripper_deadband"]:
            client.set_gripper_target(gripper_target, cfg["gripper_speed"])
            last_gripper_cmd = gripper_target

        # ── Record step (only while grip is held, not while waiting for label) ──
        if data_rec is not None and data_rec.is_recording and not waiting_for_label:
            step_q_target  = last_q_target if last_q_target is not None else state["q"]
            step_grip_norm = last_gripper_cmd / cfg["gripper_open"]
            step_eef_delta = last_eef_delta if last_eef_delta is not None else np.zeros(6)
            img_p = img_w = depth_p = depth_w = None
            if recorder is not None:
                img_p   = recorder.last_frame0
                img_w   = recorder.last_frame1
                depth_p = getattr(recorder, "last_depth0", None)
                depth_w = getattr(recorder, "last_depth1", None)
            data_rec.record_step(state, step_q_target, step_grip_norm,
                                 step_eef_delta, img_p, img_w, depth_p, depth_w)
            last_eef_delta = None

        # ── Regulate frequency ────────────────────────────────────────────────
        elapsed = time.time() - loop_start
        sleep_t = loop_period - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

        actual_hz = 1.0 / max(time.time() - loop_start, 1e-6)
        print_status(step_count, info["movement_enabled"],
                     (data_rec.is_recording and not waiting_for_label) if data_rec else False,
                     waiting_for_label,
                     episode_label, pos_err_mm, rot_err_deg, actual_hz,
                     state["gripper_width"])

    # === Cleanup ==============================================================
    print("\nTeleoperation ended.")
    print(f"Total steps: {step_count}")
    # Discard any unsaved episode (auto-save disabled per user preference)
    if data_rec is not None and data_rec.is_recording:
        n = data_rec.discard_episode()
        if n > 0:
            state_str = "awaiting label" if waiting_for_label else "in-progress"
            print(f"[REC] {state_str.capitalize()} episode ({n} steps) discarded.")
    if recorder is not None:
        recorder.close()
    try:
        client.stop()
        client.close()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
