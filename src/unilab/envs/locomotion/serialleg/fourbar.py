from __future__ import annotations

import math

import numpy as np

ACTIVE_LOWER = 0.0
ACTIVE_UPPER = 1.469449651507

_KNEE_X = -0.17993464
_KNEE_Z = 0.00489576
_CALF_X = 0.05003347
_CALF_Z = 0.04149627
_DRIVE_X = 0.04009536
_DRIVE_Z = 0.04530576
_COUPLER_LEN = math.hypot(-0.16999653, 0.00108627)
_CALF_LEN = math.hypot(_CALF_X, _CALF_Z)
_CALF_ZERO_ANGLE = math.atan2(_CALF_Z, _CALF_X)
_LUT_SIZE = 8192
_NP_LUT_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None


def policy_to_output_pos_np(policy_pos: np.ndarray) -> np.ndarray:
    arr = np.asarray(policy_pos, dtype=np.float64)
    original_shape = arr.shape
    out = arr.reshape(-1, 4).copy()
    left_alpha = np.clip(out[:, 0] - out[:, 1], ACTIVE_LOWER, ACTIVE_UPPER)
    right_alpha = np.clip(out[:, 3] - out[:, 2], ACTIVE_LOWER, ACTIVE_UPPER)
    out[:, 1] = output_knee_from_active_angle_np_array(left_alpha)
    out[:, 3] = -output_knee_from_active_angle_np_array(right_alpha)
    return out.reshape(original_shape)


def output_to_policy_pos_np(output_pos: np.ndarray) -> np.ndarray:
    arr = np.asarray(output_pos, dtype=np.float64)
    original_shape = arr.shape
    out = arr.reshape(-1, 4).copy()
    left_alpha = active_angle_from_output_knee_np(out[:, 1], right_side=False)
    right_alpha = active_angle_from_output_knee_np(out[:, 3], right_side=True)
    out[:, 1] = out[:, 0] - left_alpha
    out[:, 3] = out[:, 2] + right_alpha
    return out.reshape(original_shape)


def output_to_policy_vel_np(output_pos: np.ndarray, output_vel: np.ndarray) -> np.ndarray:
    pos = np.asarray(output_pos, dtype=np.float64).reshape(-1, 4)
    vel_arr = np.asarray(output_vel, dtype=np.float64)
    original_shape = vel_arr.shape
    out = vel_arr.reshape(-1, 4).copy()
    left_alpha = active_angle_from_output_knee_np(pos[:, 1], right_side=False)
    right_alpha = active_angle_from_output_knee_np(pos[:, 3], right_side=True)
    left_j = output_knee_jacobian_np(left_alpha, right_side=False)
    right_j = output_knee_jacobian_np(right_alpha, right_side=True)
    out[:, 1] = out[:, 1] / _safe_denominator_np(left_j)
    out[:, 3] = out[:, 3] / _safe_denominator_np(right_j)
    out[:, 1] = vel_arr.reshape(-1, 4)[:, 0] - out[:, 1]
    out[:, 3] = vel_arr.reshape(-1, 4)[:, 2] + out[:, 3]
    return out.reshape(original_shape)


def policy_to_output_torque_np(policy_pos: np.ndarray, policy_torque: np.ndarray) -> np.ndarray:
    pos = np.asarray(policy_pos, dtype=np.float64).reshape(-1, 4)
    torque_arr = np.asarray(policy_torque, dtype=np.float64)
    original_shape = torque_arr.shape
    torque_rows = torque_arr.reshape(-1, 4)
    out = torque_rows.copy()
    left_alpha = np.clip(pos[:, 0] - pos[:, 1], ACTIVE_LOWER, ACTIVE_UPPER)
    right_alpha = np.clip(pos[:, 3] - pos[:, 2], ACTIVE_LOWER, ACTIVE_UPPER)
    left_j = output_knee_jacobian_np(left_alpha, right_side=False)
    right_j = output_knee_jacobian_np(right_alpha, right_side=True)
    out[:, 0] = torque_rows[:, 0] + torque_rows[:, 1]
    out[:, 1] = -torque_rows[:, 1] / _safe_denominator_np(left_j)
    out[:, 2] = torque_rows[:, 2] + torque_rows[:, 3]
    out[:, 3] = torque_rows[:, 3] / _safe_denominator_np(right_j)
    return out.reshape(original_shape)


def output_knee_from_active_angle_np_array(active_angle: np.ndarray) -> np.ndarray:
    alpha_grid, knee_grid, _, _, _ = _fourbar_lut_np()
    return _interp_lut_np(np.asarray(active_angle, dtype=np.float64), alpha_grid, knee_grid)


def active_angle_from_output_knee_np(output_knee: np.ndarray, *, right_side: bool) -> np.ndarray:
    target = np.asarray(output_knee, dtype=np.float64)
    target = -target if right_side else target
    _, _, inverse_knee_grid, inverse_alpha_grid, _ = _fourbar_lut_np()
    return _interp_lut_np(target, inverse_knee_grid, inverse_alpha_grid)


def output_knee_jacobian_np(active_angle: np.ndarray, *, right_side: bool) -> np.ndarray:
    alpha_grid, _, _, _, jacobian_grid = _fourbar_lut_np()
    value = _interp_lut_np(np.asarray(active_angle, dtype=np.float64), alpha_grid, jacobian_grid)
    return -value if right_side else value


def _fourbar_lut_np() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global _NP_LUT_CACHE

    if _NP_LUT_CACHE is not None:
        return _NP_LUT_CACHE

    alpha_grid = np.linspace(ACTIVE_LOWER, ACTIVE_UPPER, _LUT_SIZE, dtype=np.float64)
    knee_grid = _output_knee_from_active_angle_analytic_np_array(alpha_grid)
    inverse_knee_grid, inverse_alpha_grid = _inverse_lut_grids_np(knee_grid, alpha_grid)

    eps = 1.0e-3
    lo = np.clip(alpha_grid - eps, ACTIVE_LOWER, ACTIVE_UPPER)
    hi = np.clip(alpha_grid + eps, ACTIVE_LOWER, ACTIVE_UPPER)
    jacobian_grid = (
        _output_knee_from_active_angle_analytic_np_array(hi)
        - _output_knee_from_active_angle_analytic_np_array(lo)
    ) / np.maximum(hi - lo, 1.0e-6)

    _NP_LUT_CACHE = (alpha_grid, knee_grid, inverse_knee_grid, inverse_alpha_grid, jacobian_grid)
    return _NP_LUT_CACHE


def _output_knee_from_active_angle_analytic_np_array(active_angle: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.asarray(active_angle, dtype=np.float64), ACTIVE_LOWER, ACTIVE_UPPER)
    beta = -alpha
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    px = cos_b * _DRIVE_X + sin_b * _DRIVE_Z
    pz = -sin_b * _DRIVE_X + cos_b * _DRIVE_Z

    dx = px - _KNEE_X
    dz = pz - _KNEE_Z
    dist = np.sqrt(np.maximum(dx * dx + dz * dz, 1.0e-12))
    ex = dx / dist
    ez = dz / dist

    along = (_CALF_LEN**2 - _COUPLER_LEN**2 + dist * dist) / (2.0 * dist)
    height = np.sqrt(np.maximum(_CALF_LEN**2 - along * along, 0.0))
    cx = _KNEE_X + along * ex - height * ez
    cz = _KNEE_Z + along * ez + height * ex

    phi = np.arctan2(cz - _KNEE_Z, cx - _KNEE_X)
    return _wrap_angle_np_array(_CALF_ZERO_ANGLE - phi)


def _inverse_lut_grids_np(
    knee_grid: np.ndarray, alpha_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    increasing = bool(np.all(knee_grid[1:] >= knee_grid[:-1]))
    decreasing = bool(np.all(knee_grid[1:] <= knee_grid[:-1]))
    if increasing:
        return knee_grid, alpha_grid
    if decreasing:
        return np.flip(knee_grid), np.flip(alpha_grid)
    raise RuntimeError("fourbar knee-angle LUT is not monotonic")


def _interp_lut_np(query: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    q = np.clip(query, x_grid[0], x_grid[-1])
    return np.interp(q, x_grid, y_grid)


def _safe_denominator_np(value: np.ndarray) -> np.ndarray:
    sign = np.where(value < 0.0, -np.ones_like(value), np.ones_like(value))
    return np.where(np.abs(value) < 1.0e-6, sign * 1.0e-6, value)


def _wrap_angle_np_array(angle: np.ndarray) -> np.ndarray:
    return np.remainder(angle + math.pi, 2.0 * math.pi) - math.pi
