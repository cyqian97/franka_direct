"""
robot_reset.py — reset helpers for franka_direct teleoperation.

Public API
----------
reset_to_home(client, ik, cfg)
    Move the robot to HOME_Q, optionally with a random EEF-space perturbation.

    cfg keys consumed (all produced by load_config() in simple_teleop_direct_torque.py):
        reset_speed      float  — joint-space motion speed [m/s equivalent]
        reset_randomize  bool   — whether to sample noise
        reset_pos_low    [3]    — position noise lower bound [m]
        reset_pos_high   [3]    — position noise upper bound [m]
        reset_rot_low    [3]    — rotation noise lower bound [rad], Euler XYZ
        reset_rot_high   [3]    — rotation noise upper bound [rad], Euler XYZ

Randomization strategy (mirrors DROID robot.py add_noise_to_joints)
---------------------------------------------------------------------
1. Forward-kinematics from HOME_Q → home EEF pose T_home.
2. Sample uniform noise (pos, euler) and build a perturbed target T_target.
3. Solve IK iteratively **in simulation** (offline, no real-robot step limits)
   starting from HOME_Q until convergence or max_iter.
4. Fall back to clean HOME_Q if IK diverges.
5. Call client.reset_to_joints(q_target, speed=...).
"""

import numpy as np
from scipy.spatial.transform import Rotation

HOME_Q = [0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0]

_IK_MAX_ITER = 400
_IK_POS_TOL  = 0.005   # m
_IK_ROT_TOL  = 0.02    # rad


# ── Forward kinematics ────────────────────────────────────────────────────────

def _fk(ik, q: np.ndarray) -> np.ndarray:
    """Return 4×4 SE3 for the wrist site given joint positions q."""
    ik._arm.update_state(ik._physics, q, np.zeros(7))
    ik._physics.forward()
    xpos = ik._physics.bind(ik._arm.wrist_site).xpos.copy()
    xmat = ik._physics.bind(ik._arm.wrist_site).xmat.copy().reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = xmat
    T[:3, 3]  = xpos
    return T


# ── Iterative IK ─────────────────────────────────────────────────────────────

def _rotation_error_vec(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Axis-angle error vector from R_current to R_target."""
    R_err = R_target @ R_current.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([R_err[2, 1] - R_err[1, 2],
                     R_err[0, 2] - R_err[2, 0],
                     R_err[1, 0] - R_err[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


def _ik_to_pose(ik, start_q: np.ndarray, target_T: np.ndarray) -> tuple[np.ndarray, bool]:
    """Iteratively solve IK toward target_T starting from start_q.

    Returns (q, converged).  Runs entirely in simulation — the real robot is
    not touched, so the per-step delta limits in RobotIKSolver are irrelevant
    (we just iterate many steps until we arrive).
    """
    q = start_q.copy()
    for _ in range(_IK_MAX_ITER):
        current_T = _fk(ik, q)

        p_err  = np.linalg.norm(target_T[:3, 3] - current_T[:3, 3])
        r_err  = np.linalg.norm(_rotation_error_vec(target_T[:3, :3], current_T[:3, :3]))
        if p_err < _IK_POS_TOL and r_err < _IK_ROT_TOL:
            return q, True

        # Normalised Cartesian velocity (same convention as pose_to_cartesian_velocity)
        lin_vel = (target_T[:3, 3] - current_T[:3, 3]) / ik.max_lin_delta
        rot_vel = _rotation_error_vec(target_T[:3, :3], current_T[:3, :3]) / ik.max_rot_delta
        if np.linalg.norm(lin_vel) > 1.0:
            lin_vel /= np.linalg.norm(lin_vel)
        if np.linalg.norm(rot_vel) > 1.0:
            rot_vel /= np.linalg.norm(rot_vel)
        cart_vel = np.concatenate([lin_vel, rot_vel])

        robot_state = {"joint_positions": q, "joint_velocities": np.zeros(7)}
        joint_vel   = ik.cartesian_velocity_to_joint_velocity(cart_vel, robot_state)
        joint_delta = ik.joint_velocity_to_delta(joint_vel)
        q = q + joint_delta

    return q, False


# ── Public API ────────────────────────────────────────────────────────────────

def reset_to_home(client, ik, cfg: dict) -> tuple[bool, str]:
    """Reset the robot to HOME_Q, with optional EEF noise.

    Parameters
    ----------
    client : FrankaDirectClient
    ik     : RobotIKSolver
    cfg    : flat config dict from load_config()

    Returns
    -------
    (success, message) from client.reset_to_joints
    """
    home_q = np.array(HOME_Q)
    target_q = home_q

    if cfg.get("reset_randomize", False):
        pos_noise = np.random.uniform(cfg["reset_pos_low"],  cfg["reset_pos_high"])
        rot_noise = np.random.uniform(cfg["reset_rot_low"],  cfg["reset_rot_high"])

        T_home = _fk(ik, home_q)
        T_target = T_home.copy()
        T_target[:3, 3]  = T_home[:3, 3] + pos_noise
        T_target[:3, :3] = Rotation.from_euler("xyz", rot_noise).as_matrix() @ T_home[:3, :3]

        q_noisy, converged = _ik_to_pose(ik, home_q, T_target)
        if converged:
            target_q = q_noisy
            print(f"[RESET] Randomized: pos_noise={np.round(pos_noise*1000).astype(int)} mm  "
                  f"rot_noise={np.round(np.degrees(rot_noise)).astype(int)} deg")
        else:
            print("[RESET] IK did not converge — using clean HOME_Q")

    return client.reset_to_joints(target_q.tolist(), speed=cfg.get("reset_speed", 0.2))
