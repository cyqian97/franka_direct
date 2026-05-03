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
3. Solve IK with scipy L-BFGS-B (joint-bounded, single-shot — same approach as
   DROID's Pinocchio IK).  Starting point: HOME_Q.
4. Validate result with FK, same way DROID does.
5. Fall back to clean HOME_Q if IK fails or error exceeds tolerance.
6. Call client.reset_to_joints(q_target, speed=...).
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple

HOME_Q = [0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0]

_IK_POS_TOL  = 0.005   # m
_IK_ROT_TOL  = 0.02    # rad
_CLIK_DT      = 0.1    # integration step (same as Pinocchio default)
_CLIK_DAMPING = 1e-6   # damped least-squares regularisation
_CLIK_MAX_ITER = 1000  # iterations (same as Pinocchio default)
_CLIK_JEPS     = 1e-5  # finite-difference step for numerical Jacobian


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


# ── Rotation error ────────────────────────────────────────────────────────────

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


# ── CLIK (Closed-Loop Inverse Kinematics) ─────────────────────────────────────
#
# Same algorithm as Pinocchio's inverse_kinematics:
#   1. Compute pose error in task space.
#   2. Compute numerical Jacobian J (6×7) via finite differences.
#   3. Damped least-squares step: Δq = Jᵀ(JJᵀ + λI)⁻¹ · e · dt
#   4. Hard-clamp q to joint limits (no soft avoidance zone unlike the
#      Cartesian6dVelocityEffector used in the old approach).
#   5. Repeat until convergence or max_iter.
#
# Why this is more robust than the old velocity-effector approach:
#   - The effector's 0.3 rad soft limit zone reduces commanded velocity near
#     joint limits, causing stalls when the IK solution lies near a limit.
#   - The effector's QP solver uses the teleoperation step sizes (max 0.075 m,
#     0.15 rad per step), which are conservative for offline use.
#   - Here we use dt=0.1 and hard clamping, matching Pinocchio's defaults.

def _ik_solve(ik, start_q: np.ndarray, target_T: np.ndarray) -> Tuple[np.ndarray, bool]:
    """CLIK: same algorithm as Pinocchio's inverse_kinematics. Returns (q, success)."""
    j_range = ik._physics.bind(ik._arm.joints).range  # (7, 2)
    q_min, q_max = j_range[:, 0], j_range[:, 1]

    q = np.clip(start_q.copy(), q_min, q_max)

    for _ in range(_CLIK_MAX_ITER):
        T0  = _fk(ik, q)
        dp  = target_T[:3, 3] - T0[:3, 3]
        dr  = _rotation_error_vec(target_T[:3, :3], T0[:3, :3])

        if np.linalg.norm(dp) < _IK_POS_TOL and np.linalg.norm(dr) < _IK_ROT_TOL:
            return q, True

        # Numerical Jacobian (6×7): perturb each joint by _CLIK_JEPS
        J = np.zeros((6, 7))
        for i in range(7):
            q_h = q.copy(); q_h[i] += _CLIK_JEPS
            T_h = _fk(ik, q_h)
            J[:3, i] = (T_h[:3, 3] - T0[:3, 3]) / _CLIK_JEPS
            J[3:, i] = _rotation_error_vec(T_h[:3, :3], T0[:3, :3]) / _CLIK_JEPS

        # Damped least-squares: Δq = Jᵀ(JJᵀ + λI)⁻¹ e · dt
        err = np.concatenate([dp, dr])
        dq  = J.T @ np.linalg.solve(J @ J.T + _CLIK_DAMPING * np.eye(6), err) * _CLIK_DT

        q = np.clip(q + dq, q_min, q_max)

    # Validate final result (mirrors DROID's post-IK FK check)
    current_T = _fk(ik, q)
    pos_err   = np.linalg.norm(current_T[:3, 3] - target_T[:3, 3])
    rot_err   = np.linalg.norm(_rotation_error_vec(target_T[:3, :3], current_T[:3, :3]))
    return q, (pos_err < _IK_POS_TOL and rot_err < _IK_ROT_TOL)


# ── Public API ────────────────────────────────────────────────────────────────

def reset_to_home(client, ik, cfg: dict) -> Tuple[bool, str]:
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
    home_q   = np.array(HOME_Q)
    target_q = home_q

    if cfg.get("reset_randomize", False):
        pos_noise = np.random.uniform(cfg["reset_pos_low"],  cfg["reset_pos_high"])
        rot_noise = np.random.uniform(cfg["reset_rot_low"],  cfg["reset_rot_high"])

        T_home   = _fk(ik, home_q)
        T_target = T_home.copy()
        T_target[:3, 3]  = T_home[:3, 3] + pos_noise
        T_target[:3, :3] = Rotation.from_euler("xyz", rot_noise).as_matrix() @ T_home[:3, :3]

        q_sol, success = _ik_solve(ik, home_q, T_target)

        if success:
            target_q = q_sol
            print(f"[RESET] Randomized: pos_noise={np.round(pos_noise*1000).astype(int)} mm  "
                  f"rot_noise={np.round(np.degrees(rot_noise)).astype(int)} deg  "
                  f"q={np.round(target_q, 3).tolist()}")
        else:
            print("[RESET] IK did not reach target — using clean HOME_Q")

    return client.reset_to_joints(target_q.tolist(), speed=cfg.get("reset_speed", 0.2))
