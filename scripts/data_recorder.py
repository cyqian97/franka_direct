"""
Episode data recorder for franka_direct teleoperation.

Records observations and actions to HDF5 files in a format compatible with
openvla-oft fine-tuning after RLDS conversion.

HDF5 layout per episode:
    /observations/
        qpos        [T, 7]   float32  measured joint positions (rad)
        qvel        [T, 7]   float32  measured joint velocities (rad/s)
        ee_pose     [T, 16]  float32  O_T_EE column-major (metres, base frame)
        gripper     [T, 1]   float32  normalized gripper [0=closed, 1=open]
        O_T_EE_d    [T, 16]  float32  desired EEF pose
        q_d         [T, 7]   float32  desired joint positions
        dq_d        [T, 7]   float32  desired joint velocities
        ddq_d       [T, 7]   float32  desired joint accelerations
        tau_J       [T, 7]   float32  measured link-side joint torques (Nm)
        tau_J_d     [T, 7]   float32  desired joint torques w/o gravity
        dtau_J      [T, 7]   float32  torque derivative
        tau_ext_hat_filtered [T, 7]  float32  filtered external torque estimate
        O_F_ext_hat_K [T, 6] float32  external wrench in base frame
        K_F_ext_hat_K [T, 6] float32  external wrench in stiffness frame
        joint_contact    [T, 7]  float32  contact level per joint
        cartesian_contact [T, 6] float32  contact level per Cartesian dim
        joint_collision    [T, 7]  float32  collision level per joint
        cartesian_collision [T, 6] float32 collision level per Cartesian dim
        theta       [T, 7]   float32  motor positions (rad)
        dtheta      [T, 7]   float32  motor velocities (rad/s)
        elbow       [T, 2]   float32  elbow config [joint3_pos, flip_dir]
        robot_mode  [T, 1]   int32    RobotMode enum value
        time_s      [T, 1]   float64  robot timestamp (s)
        images/
            primary         [T, H, W, 3]  uint8   RGB, 3rd-person camera
            primary_depth   [T, H, W]     float32 depth in metres (if recorded)
            wrist           [T, H, W, 3]  uint8   RGB, wrist/secondary camera
            wrist_depth     [T, H, W]     float32 depth in metres (if recorded)
    /action         [T, 8]   float32  [q_target (7), gripper_norm (1)]
    /action_eef     [T, 7]   float32  [pos_delta (3), rot_vec (3), gripper_norm (1)]
    @sim            False
    @language_instruction   str

Episode filename suffix:
    episode_000000_success.hdf5  (A pressed during recording)
    episode_000001_fail.hdf5     (B pressed during recording)
    episode_000002.hdf5          (no label)
"""

import json
import os
import time
from typing import Optional

import h5py
import numpy as np
from PIL import Image

_GRIPPER_OPEN_M_DEFAULT = 0.08   # Franka Hand max opening (metres)

# Extended libfranka fields: (key, default_shape, dtype)
_EXT_FIELDS = [
    ("O_T_EE_d",             (16,), np.float32),
    ("q_d",                  (7,),  np.float32),
    ("dq_d",                 (7,),  np.float32),
    ("ddq_d",                (7,),  np.float32),
    ("tau_J",                (7,),  np.float32),
    ("tau_J_d",              (7,),  np.float32),
    ("dtau_J",               (7,),  np.float32),
    ("tau_ext_hat_filtered", (7,),  np.float32),
    ("O_F_ext_hat_K",        (6,),  np.float32),
    ("K_F_ext_hat_K",        (6,),  np.float32),
    ("joint_contact",        (7,),  np.float32),
    ("cartesian_contact",    (6,),  np.float32),
    ("joint_collision",      (7,),  np.float32),
    ("cartesian_collision",  (6,),  np.float32),
    ("theta",                (7,),  np.float32),
    ("dtheta",               (7,),  np.float32),
    ("elbow",                (2,),  np.float32),
    ("robot_mode",           (1,),  np.int32),
    ("time_s",               (1,),  np.float64),
]


class DataRecorder:
    """Buffers one teleoperation episode and flushes to HDF5 on save.

    Usage:
        rec = DataRecorder(out_dir="data/my_task")
        rec.start_episode()

        for each step:
            rec.record_step(state, q_target, gripper_norm, ...)

        path = rec.save_episode(task="pick up the cup", label="success")
        # or: rec.discard_episode()
    """

    def __init__(self, out_dir: str = "data", gripper_open_m: float = _GRIPPER_OPEN_M_DEFAULT):
        self._out_dir = out_dir
        self._gripper_open_m = gripper_open_m
        self._buf: Optional[list] = None
        self._episode_count = self._count_existing_episodes()

    # ── Public API ────────────────────────────────────────────────────────────

    def start_episode(self) -> None:
        """Begin buffering a new episode (clears any previous buffer)."""
        self._buf = []

    def record_step(
        self,
        state: dict,
        q_target: list,
        gripper_norm: float,
        eef_delta_6d: Optional[np.ndarray] = None,
        img_primary: Optional[np.ndarray] = None,
        img_wrist: Optional[np.ndarray] = None,
        depth_primary: Optional[np.ndarray] = None,
        depth_wrist: Optional[np.ndarray] = None,
    ) -> None:
        """Append one step to the in-progress episode.

        Parameters
        ----------
        state : dict
            Return value of FrankaDirectClient.get_robot_state().
        q_target : list[7]
            Absolute joint target commanded this step (radians).
        gripper_norm : float
            Normalized gripper target [0=closed, 1=open].
        eef_delta_6d : np.ndarray[6] or None
            Delta EEF action: [pos_delta (3, metres), rot_vec (3, radians)].
        img_primary : np.ndarray[H, W, 3] or None
            BGR frame from primary camera; converted to RGB before storing.
        img_wrist : np.ndarray[H, W, 3] or None
            BGR frame from wrist camera; converted to RGB.
        depth_primary : np.ndarray[H, W] float32 or None
            Depth map from primary camera (metres).
        depth_wrist : np.ndarray[H, W] float32 or None
            Depth map from wrist camera (metres).
        """
        if self._buf is None:
            raise RuntimeError("Call start_episode() before record_step().")

        if eef_delta_6d is None:
            eef_delta_6d = np.zeros(6, dtype=np.float32)

        step = {
            "qpos":                 np.array(state["q"],    dtype=np.float32),
            "qvel":                 np.array(state["dq"] or [0.0] * 7, dtype=np.float32),
            "ee_pose":              np.array(state["pose"], dtype=np.float32),
            "gripper":              np.array([state["gripper_width"] / self._gripper_open_m],
                                             dtype=np.float32),
            "q_target":             np.array(q_target,     dtype=np.float32),
            "eef_delta_6d":         np.array(eef_delta_6d, dtype=np.float32),
            "gripper_norm":         float(gripper_norm),
        }

        # Extended libfranka fields (default to zeros if not in state dict)
        for key, shape, dtype in _EXT_FIELDS:
            default = np.zeros(shape, dtype=dtype)
            val = state.get(key)
            if val is not None:
                step[key] = np.array(val, dtype=dtype).reshape(shape)
            else:
                step[key] = default

        if img_primary is not None:
            step["img_primary"] = img_primary[:, :, ::-1].copy()   # BGR → RGB
        if img_wrist is not None:
            step["img_wrist"] = img_wrist[:, :, ::-1].copy()
        if depth_primary is not None:
            step["depth_primary"] = depth_primary.copy()
        if depth_wrist is not None:
            step["depth_wrist"] = depth_wrist.copy()

        self._buf.append(step)

    def save_episode(self, task: str = "", label: str = "") -> str:
        """Write the buffered episode to an HDF5 file and return its path.

        Parameters
        ----------
        task : str
            Language instruction stored as an HDF5 attribute.
        label : str
            Episode outcome label. Appended to filename as a suffix.
            Use "success", "fail", or "" (no suffix) for unlabeled.
        """
        if not self._buf:
            raise RuntimeError("No steps recorded — nothing to save.")

        os.makedirs(self._out_dir, exist_ok=True)
        suffix = f"_{label}" if label else ""
        path = os.path.join(self._out_dir,
                            f"episode_{self._episode_count:06d}{suffix}.hdf5")
        T = len(self._buf)

        has_primary       = "img_primary"    in self._buf[0]
        has_wrist         = "img_wrist"      in self._buf[0]
        has_depth_primary = "depth_primary"  in self._buf[0]
        has_depth_wrist   = "depth_wrist"    in self._buf[0]

        with h5py.File(path, "w") as f:
            f.attrs["sim"] = False
            f.attrs["language_instruction"] = task

            obs = f.create_group("observations")

            obs.create_dataset("qpos",    data=np.stack([s["qpos"]    for s in self._buf]),
                               dtype=np.float32)
            obs.create_dataset("qvel",    data=np.stack([s["qvel"]    for s in self._buf]),
                               dtype=np.float32)
            obs.create_dataset("ee_pose", data=np.stack([s["ee_pose"] for s in self._buf]),
                               dtype=np.float32)
            obs.create_dataset("gripper", data=np.stack([s["gripper"] for s in self._buf]),
                               dtype=np.float32)

            # Extended libfranka fields
            for key, _, dtype in _EXT_FIELDS:
                obs.create_dataset(key,
                                   data=np.stack([s[key] for s in self._buf]),
                                   dtype=dtype)

            if has_primary or has_wrist or has_depth_primary or has_depth_wrist:
                imgs = obs.create_group("images")
                if has_primary:
                    imgs.create_dataset("primary",
                                        data=np.stack([s["img_primary"] for s in self._buf]),
                                        dtype=np.uint8)
                if has_wrist:
                    imgs.create_dataset("wrist",
                                        data=np.stack([s["img_wrist"] for s in self._buf]),
                                        dtype=np.uint8)
                if has_depth_primary:
                    imgs.create_dataset("primary_depth",
                                        data=np.stack([s["depth_primary"] for s in self._buf]),
                                        dtype=np.float32)
                if has_depth_wrist:
                    imgs.create_dataset("wrist_depth",
                                        data=np.stack([s["depth_wrist"] for s in self._buf]),
                                        dtype=np.float32)

            gn_arr = np.array([[s["gripper_norm"]] for s in self._buf], dtype=np.float32)
            qt_arr = np.stack([s["q_target"]    for s in self._buf])          # [T, 7]
            f.create_dataset("action",
                             data=np.concatenate([qt_arr, gn_arr], axis=-1),
                             dtype=np.float32)

            ed_arr = np.stack([s["eef_delta_6d"] for s in self._buf])         # [T, 6]
            f.create_dataset("action_eef",
                             data=np.concatenate([ed_arr, gn_arr], axis=-1),
                             dtype=np.float32)

        # Duration from robot timestamps (time_s column, first field)
        t0 = float(self._buf[0]["time_s"][0])
        t1 = float(self._buf[-1]["time_s"][0])
        duration_s = round(t1 - t0, 4) if t1 > t0 else None

        status = {
            "episode":    self._episode_count,
            "label":      label or "",
            "task":       task,
            "num_steps":  T,
            "saved_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_s": duration_s,
            "hdf5_file":  os.path.basename(path),
        }
        status_path = path.replace(".hdf5", ".json")
        with open(status_path, "w") as sf:
            json.dump(status, sf, indent=2)

        if "img_primary" in self._buf[0]:
            cover_path = path.replace(".hdf5", "_cover.jpg")
            Image.fromarray(self._buf[0]["img_primary"]).save(cover_path, quality=90)

        lbl_str = f" [{label}]" if label else ""
        print(f"[REC] Saved {T} steps{lbl_str} → {path}")
        self._episode_count += 1
        self._buf = None
        return path

    def discard_episode(self) -> int:
        """Discard the buffered episode without saving. Returns step count."""
        n = len(self._buf) if self._buf else 0
        self._buf = None
        return n

    @property
    def is_recording(self) -> bool:
        return self._buf is not None

    @property
    def num_steps(self) -> int:
        return len(self._buf) if self._buf else 0

    # ── Internal ─────────────────────────────────────────────────────────────

    def _count_existing_episodes(self) -> int:
        if not os.path.isdir(self._out_dir):
            return 0
        return sum(
            1 for fn in os.listdir(self._out_dir)
            if fn.startswith("episode_") and fn.endswith(".hdf5")
        )
