"""
Episode data recorder for franka_direct teleoperation.

Records observations and actions to HDF5 files in a format compatible with
openvla-oft fine-tuning after RLDS conversion.

HDF5 layout per episode:
    /observations/
        qpos        [T, 7]   float32  measured joint positions (rad)
        qvel        [T, 7]   float32  measured joint velocities (rad/s)
        ee_pose     [T, 16]  float32  O_T_EE column-major (meters, base frame)
        gripper     [T, 1]   float32  normalized gripper [0=closed, 1=open]
        images/
            primary [T, H, W, 3]  uint8  RGB, 3rd-person camera
            wrist   [T, H, W, 3]  uint8  RGB, wrist/secondary camera
    /action         [T, 8]   float32  [q_target (7), gripper_norm (1)]
    /action_eef     [T, 7]   float32  [pos_delta (3), rot_vec (3), gripper_norm (1)]
    @sim            False
    @language_instruction   str

Two action representations are saved for flexibility:
  action     – absolute joint targets (7 DOF) + normalized gripper (0–1).
               Use with openvla-oft dataset name "franka_fr3".
  action_eef – delta EEF position (m) + rotation vector (rad) + gripper (0–1).
               Zero when the robot is holding position.
               Use with openvla-oft dataset name "franka_fr3_eef".
"""

import datetime
import os
from typing import Optional

import h5py
import numpy as np

GRIPPER_OPEN_M = 0.08   # Franka Hand max opening (meters)


class DataRecorder:
    """Buffers one teleoperation episode and flushes to HDF5 on save.

    Usage (inside the control loop):
        rec = DataRecorder(out_dir="data/my_task")
        rec.start_episode()

        for each step:
            rec.record_step(state, q_target, gripper_norm, img_primary, img_wrist)

        path = rec.save_episode(task="pick up the cup")   # or discard_episode()
    """

    def __init__(self, out_dir: str = "data"):
        self._out_dir = out_dir
        self._buf: Optional[list] = None      # list of per-step dicts while recording
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
    ) -> None:
        """Append one step to the in-progress episode.

        Parameters
        ----------
        state : dict
            Return value of FrankaDirectClient.get_robot_state().
            Required keys: 'q', 'dq', 'pose'.
        q_target : list[7]
            Absolute joint target commanded this step (radians).
        gripper_norm : float
            Normalized gripper target [0=closed, 1=open].
        eef_delta_6d : np.ndarray[6] or None
            Delta EEF action: [pos_delta (3, metres), rot_vec (3, radians)].
            Pass None or zeros(6) when the robot is holding position.
        img_primary : np.ndarray[H, W, 3] or None
            BGR frame from primary camera; converted to RGB before storing.
        img_wrist : np.ndarray[H, W, 3] or None
            BGR frame from wrist/secondary camera; converted to RGB.
        """
        if self._buf is None:
            raise RuntimeError("Call start_episode() before record_step().")

        if eef_delta_6d is None:
            eef_delta_6d = np.zeros(6, dtype=np.float32)

        step = {
            "qpos":                 np.array(state["q"],    dtype=np.float32),
            "qvel":                 np.array(state["dq"] or [0.0] * 7, dtype=np.float32),
            "ee_pose":              np.array(state["pose"], dtype=np.float32),
            "gripper":              np.array([state["gripper_width"] / GRIPPER_OPEN_M],
                                             dtype=np.float32),
            "q_target":             np.array(q_target,     dtype=np.float32),
            "eef_delta_6d":         np.array(eef_delta_6d, dtype=np.float32),
            "gripper_norm":         float(gripper_norm),
            # Extended libfranka RobotState fields
            "O_T_EE_d":             np.array(state.get("O_T_EE_d",             [0.0]*16), dtype=np.float32),
            "q_d":                  np.array(state.get("q_d",                  [0.0]*7),  dtype=np.float32),
            "dq_d":                 np.array(state.get("dq_d",                 [0.0]*7),  dtype=np.float32),
            "ddq_d":                np.array(state.get("ddq_d",                [0.0]*7),  dtype=np.float32),
            "tau_J":                np.array(state.get("tau_J",                [0.0]*7),  dtype=np.float32),
            "tau_J_d":              np.array(state.get("tau_J_d",              [0.0]*7),  dtype=np.float32),
            "dtau_J":               np.array(state.get("dtau_J",               [0.0]*7),  dtype=np.float32),
            "tau_ext_hat_filtered": np.array(state.get("tau_ext_hat_filtered", [0.0]*7),  dtype=np.float32),
            "O_F_ext_hat_K":        np.array(state.get("O_F_ext_hat_K",        [0.0]*6),  dtype=np.float32),
            "K_F_ext_hat_K":        np.array(state.get("K_F_ext_hat_K",        [0.0]*6),  dtype=np.float32),
            "joint_contact":        np.array(state.get("joint_contact",        [0.0]*7),  dtype=np.float32),
            "cartesian_contact":    np.array(state.get("cartesian_contact",    [0.0]*6),  dtype=np.float32),
            "joint_collision":      np.array(state.get("joint_collision",      [0.0]*7),  dtype=np.float32),
            "cartesian_collision":  np.array(state.get("cartesian_collision",  [0.0]*6),  dtype=np.float32),
            "theta":                np.array(state.get("theta",                [0.0]*7),  dtype=np.float32),
            "dtheta":               np.array(state.get("dtheta",               [0.0]*7),  dtype=np.float32),
            "elbow":                np.array(state.get("elbow",                [0.0]*2),  dtype=np.float32),
            "robot_mode":           np.array([state.get("robot_mode",          0)],       dtype=np.int32),
            "time_s":               np.array([state.get("time_s",              0.0)],     dtype=np.float64),
        }
        if img_primary is not None:
            step["img_primary"] = img_primary[:, :, ::-1].copy()   # BGR → RGB
        if img_wrist is not None:
            step["img_wrist"] = img_wrist[:, :, ::-1].copy()

        self._buf.append(step)

    def save_episode(self, task: str = "") -> str:
        """Write the buffered episode to an HDF5 file and return its path.

        Parameters
        ----------
        task : str
            Language instruction stored as a dataset attribute.

        Returns
        -------
        str
            Absolute path of the saved HDF5 file.
        """
        if not self._buf:
            raise RuntimeError("No steps recorded — nothing to save.")

        os.makedirs(self._out_dir, exist_ok=True)
        path = os.path.join(self._out_dir, f"episode_{self._episode_count:06d}.hdf5")
        T = len(self._buf)

        has_primary = "img_primary" in self._buf[0]
        has_wrist   = "img_wrist"   in self._buf[0]

        with h5py.File(path, "w") as f:
            f.attrs["sim"] = False
            f.attrs["language_instruction"] = task

            obs = f.create_group("observations")

            qpos_arr = np.stack([s["qpos"]    for s in self._buf])   # [T, 7]
            qvel_arr = np.stack([s["qvel"]    for s in self._buf])   # [T, 7]
            pose_arr = np.stack([s["ee_pose"] for s in self._buf])   # [T, 16]
            grip_arr = np.stack([s["gripper"] for s in self._buf])   # [T, 1]

            obs.create_dataset("qpos",    data=qpos_arr, dtype=np.float32)
            obs.create_dataset("qvel",    data=qvel_arr, dtype=np.float32)
            obs.create_dataset("ee_pose", data=pose_arr, dtype=np.float32)
            obs.create_dataset("gripper", data=grip_arr, dtype=np.float32)

            # Extended libfranka fields
            for key, dtype in [
                ("O_T_EE_d",             np.float32),
                ("q_d",                  np.float32),
                ("dq_d",                 np.float32),
                ("ddq_d",                np.float32),
                ("tau_J",                np.float32),
                ("tau_J_d",              np.float32),
                ("dtau_J",               np.float32),
                ("tau_ext_hat_filtered", np.float32),
                ("O_F_ext_hat_K",        np.float32),
                ("K_F_ext_hat_K",        np.float32),
                ("joint_contact",        np.float32),
                ("cartesian_contact",    np.float32),
                ("joint_collision",      np.float32),
                ("cartesian_collision",  np.float32),
                ("theta",                np.float32),
                ("dtheta",               np.float32),
                ("elbow",                np.float32),
                ("robot_mode",           np.int32),
                ("time_s",               np.float64),
            ]:
                obs.create_dataset(key, data=np.stack([s[key] for s in self._buf]), dtype=dtype)

            if has_primary or has_wrist:
                imgs = obs.create_group("images")
                if has_primary:
                    primary_arr = np.stack([s["img_primary"] for s in self._buf])
                    imgs.create_dataset("primary", data=primary_arr, dtype=np.uint8)
                if has_wrist:
                    wrist_arr = np.stack([s["img_wrist"] for s in self._buf])
                    imgs.create_dataset("wrist", data=wrist_arr, dtype=np.uint8)

            gn_arr    = np.array([[s["gripper_norm"]] for s in self._buf],
                                  dtype=np.float32)                              # [T, 1]

            qt_arr    = np.stack([s["q_target"]    for s in self._buf])          # [T, 7]
            action    = np.concatenate([qt_arr, gn_arr], axis=-1)                # [T, 8]
            f.create_dataset("action", data=action, dtype=np.float32)

            ed_arr    = np.stack([s["eef_delta_6d"] for s in self._buf])         # [T, 6]
            action_eef = np.concatenate([ed_arr, gn_arr], axis=-1)               # [T, 7]
            f.create_dataset("action_eef", data=action_eef, dtype=np.float32)

        print(f"[REC] Saved {T} steps → {path}")
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
        """Count existing episode HDF5 files to determine next episode index."""
        if not os.path.isdir(self._out_dir):
            return 0
        return sum(
            1 for f in os.listdir(self._out_dir)
            if f.startswith("episode_") and f.endswith(".hdf5")
        )
