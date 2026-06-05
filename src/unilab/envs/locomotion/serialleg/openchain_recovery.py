from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dr import ResetPlan
from unilab.dr.dr_utils import zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_from_euler_xyz, np_quat_mul
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.serialleg.fourbar import ACTIVE_LOWER, ACTIVE_UPPER

from .openchain_flat import (
    DEFAULT_BASE_HEIGHT,
    DEFAULT_SERIALLEG_ANGLES,
    NUM_SERIALLEG_ACTIONS,
    NUM_SERIALLEG_LEG_ACTIONS,
    NUM_SERIALLEG_WHEEL_ACTIONS,
    SERIALLEG_LEG_INDICES,
    SERIALLEG_WHEEL_INDICES,
    SerialLegOpenChainDomainRandConfig,
    SerialLegOpenChainDomainRandomizationProvider,
    SerialLegOpenChainFlatCfg,
    SerialLegOpenChainFlatEnv,
    SerialLegOpenChainRewardConfig,
    build_serialleg_openchain_reset_randomization,
)

BASE_CONTACT_SENSOR_NAME = "base_contact"
WHEEL_CONTACT_SENSOR_NAMES: tuple[str, ...] = ("l_wheel_contact", "r_wheel_contact")
LEG_CONTACT_SENSOR_NAMES: tuple[str, ...] = (
    "lf0_contact",
    "lf1_contact",
    "rf0_contact",
    "rf1_contact",
)
CONTACT_SENSOR_FORCE_DIM = 3
CONTACT_FORCE_MAX_N = 5000.0
FULL_ANGLE_RESET_BBOX_MIN = np.asarray((-0.278, -0.242, -0.323), dtype=np.float64)
FULL_ANGLE_RESET_BBOX_MAX = np.asarray((0.278, 0.242, 0.111), dtype=np.float64)


@dataclass
class SerialLegOpenChainRecoveryResetConfig:
    pos_xy_range: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    height_offset_range: list[float] = field(default_factory=lambda: [0.0, 0.2])
    roll_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])
    pitch_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])
    yaw_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])
    lin_vel_range: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    ang_vel_range: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    clearance_range: list[float] = field(default_factory=lambda: [0.0, 0.05])


@dataclass
class SerialLegOpenChainRecoveryRewardConfig(SerialLegOpenChainRewardConfig):
    tracking_lin_vz_weight: float = 0.0
    contact_forces_threshold: float = 35.0
    collision_threshold: float = 0.1
    upright_contact_force_threshold: float = 1.0
    upright_contact_min_gate: float = 0.0
    wheel_contact_cmd_threshold: float = 0.1
    action_saturation_threshold: float = 0.95
    active_rod_margin_warning: float = 0.05


@dataclass
class SerialLegOpenChainRecoveryDomainRandConfig(SerialLegOpenChainDomainRandConfig):
    randomize_init_yaw: bool = False


@registry.envcfg("SerialLegOpenChainRecovery")
@dataclass
class SerialLegOpenChainRecoveryCfg(SerialLegOpenChainFlatCfg):
    reward_config: SerialLegOpenChainRecoveryRewardConfig | None = None
    recovery_reset: SerialLegOpenChainRecoveryResetConfig = field(
        default_factory=SerialLegOpenChainRecoveryResetConfig
    )
    domain_rand: SerialLegOpenChainRecoveryDomainRandConfig = field(
        default_factory=SerialLegOpenChainRecoveryDomainRandConfig
    )


def _range_bounds(name: str, values: list[float] | tuple[float, float]) -> tuple[float, float]:
    bounds = np.asarray(values, dtype=np.float64)
    if bounds.shape != (2,):
        raise ValueError(f"recovery_reset.{name} must have shape (2,), got {bounds.shape}")
    return float(min(bounds[0], bounds[1])), float(max(bounds[0], bounds[1]))


def _sample_range(name: str, values: list[float] | tuple[float, float], shape: tuple[int, ...]):
    low, high = _range_bounds(name, values)
    return np.random.uniform(low, high, size=shape)


def _quat_z_row(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.stack(
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=1,
    )


def _full_angle_safe_base_height(z_row: np.ndarray, clearance: np.ndarray) -> np.ndarray:
    min_z = np.minimum(z_row * FULL_ANGLE_RESET_BBOX_MIN, z_row * FULL_ANGLE_RESET_BBOX_MAX).sum(
        axis=1
    )
    return -min_z + clearance


class SerialLegOpenChainRecoveryDomainRandomizationProvider(
    SerialLegOpenChainDomainRandomizationProvider
):
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        cfg = env.cfg.recovery_reset
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))

        qpos[:, 0] += _sample_range("pos_xy_range", cfg.pos_xy_range, (num_reset,))
        qpos[:, 1] += _sample_range("pos_xy_range", cfg.pos_xy_range, (num_reset,))

        roll = _sample_range("roll_range", cfg.roll_range, (num_reset,))
        pitch = _sample_range("pitch_range", cfg.pitch_range, (num_reset,))
        yaw = _sample_range("yaw_range", cfg.yaw_range, (num_reset,))
        quat_delta = np_quat_from_euler_xyz(roll, pitch, yaw)
        new_quat = np_quat_mul(qpos[:, 3:7], quat_delta)

        z_row = _quat_z_row(new_quat)
        sampled_height = qpos[:, 2] + _sample_range(
            "height_offset_range", cfg.height_offset_range, (num_reset,)
        )
        safe_height = _full_angle_safe_base_height(
            z_row,
            _sample_range("clearance_range", cfg.clearance_range, (num_reset,)),
        )

        origins = env._spawn.origins_for(env_ids)
        qpos[:, 0:3] += origins
        qpos[:, 2] = np.maximum(sampled_height, safe_height) + origins[:, 2]
        qpos[:, 3:7] = new_quat
        qpos[:, 7:] = DEFAULT_SERIALLEG_ANGLES

        qvel[:, :] = 0.0
        qvel[:, 0:3] = _sample_range("lin_vel_range", cfg.lin_vel_range, (num_reset, 3))
        qvel[:, 3:6] = _sample_range("ang_vel_range", cfg.ang_vel_range, (num_reset, 3))

        commands = self._sample_commands(env, num_reset)
        standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0

        info_updates: dict[str, Any] = {
            "commands": commands,
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "current_ctrl": np.broadcast_to(
                DEFAULT_SERIALLEG_ANGLES, (num_reset, NUM_SERIALLEG_ACTIONS)
            ).astype(get_global_dtype()),
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "qacc": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "base_contact_force": np.zeros((num_reset,), dtype=get_global_dtype()),
            "wheel_contact_forces": np.zeros(
                (num_reset, NUM_SERIALLEG_WHEEL_ACTIONS), dtype=get_global_dtype()
            ),
            "leg_contact_forces": np.zeros(
                (num_reset, NUM_SERIALLEG_LEG_ACTIONS), dtype=get_global_dtype()
            ),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_serialleg_openchain_reset_randomization(env, num_reset),
        )


@registry.env("SerialLegOpenChainRecovery", sim_backend="mujoco")
class SerialLegOpenChainRecoveryEnv(SerialLegOpenChainFlatEnv):
    _cfg: SerialLegOpenChainRecoveryCfg

    def __init__(
        self, cfg: SerialLegOpenChainRecoveryCfg, num_envs: int = 1, backend_type: str = "mujoco"
    ):
        if isinstance(cfg.reward_config, dict):
            cfg.reward_config = SerialLegOpenChainRecoveryRewardConfig(**cfg.reward_config)
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._base_contact_force_buf = np.zeros((num_envs,), dtype=get_global_dtype())
        self._wheel_contact_force_buf = np.zeros(
            (num_envs, len(WHEEL_CONTACT_SENSOR_NAMES)), dtype=get_global_dtype()
        )
        self._leg_contact_force_buf = np.zeros(
            (num_envs, len(LEG_CONTACT_SENSOR_NAMES)), dtype=get_global_dtype()
        )
        self._zero_base_contact_force = np.zeros((num_envs,), dtype=get_global_dtype())
        self._zero_leg_contact_forces = np.zeros(
            (num_envs, len(LEG_CONTACT_SENSOR_NAMES)), dtype=get_global_dtype()
        )
        self._init_domain_randomization(SerialLegOpenChainRecoveryDomainRandomizationProvider())

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns.update(
            {
                "tracking_lin_vel": self._reward_tracking_lin_vel,
                "tracking_ang_vel": self._reward_tracking_ang_vel,
                "lin_vel_z": self._reward_lin_vel_z,
                "ang_vel_xy": self._reward_ang_vel_xy,
                "leg_torques": self._reward_joint_torques_l2,
                "leg_dof_acc": self._reward_dof_acc,
                "leg_power": self._reward_joint_power,
                "stand_still": self._reward_stand_still,
                "joint_pos_penalty": self._reward_joint_pos_penalty,
                "joint_mirror": self._reward_joint_mirror,
                "dof_pos_limits": rewards.joint_pos_limits,
                "collision": self._reward_collision,
                "contact_forces": self._reward_contact_forces,
                "upright_leg_contact": self._reward_upright_leg_contact,
                "wheel_contact_without_cmd": self._reward_wheel_contact_without_cmd,
                "diagnostics": self._reward_diagnostics,
            }
        )

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.gravity)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        state.info["torques"] = self._estimate_direct_actuator_torques(state.info, dof_pos, dof_vel)
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        base_contact, wheel_contact, leg_contact = self._contact_forces(
            include_base=True, include_leg=True
        )
        state.info["base_contact_force"] = base_contact
        state.info["wheel_contact_forces"] = wheel_contact
        state.info["leg_contact_forces"] = leg_contact
        terminated = self._compute_terminated(gravity)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        return np.zeros((gravity.shape[0],), dtype=bool)

    def _compute_reward(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        reward = super()._compute_reward(info, linvel, gyro, gravity, dof_pos, dof_vel)
        self._write_recovery_diagnostics(info, linvel, gyro, gravity, dof_pos)
        return reward

    def _upright_gate(self, gravity: np.ndarray | None, num_envs: int) -> np.ndarray:
        return rewards.upright_scale(gravity, num_envs)

    def _reward_tracking_lin_vel(self, ctx: RewardContext) -> np.ndarray:
        commands = ctx.info["commands"]
        gate = self._upright_gate(ctx.gravity, ctx.num_envs)
        vz_weight = float(self._reward_cfg.tracking_lin_vz_weight)
        error = np.square(commands[:, 0] - ctx.linvel[:, 0]) + vz_weight * np.square(
            ctx.linvel[:, 2]
        )
        return np.asarray(np.exp(-error / ctx.tracking_sigma) * gate, dtype=get_global_dtype())

    def _reward_tracking_ang_vel(self, ctx: RewardContext) -> np.ndarray:
        commands = ctx.info["commands"]
        gate = self._upright_gate(ctx.gravity, ctx.num_envs)
        error = np.square(commands[:, 2] - ctx.gyro[:, 2])
        return np.asarray(np.exp(-error / ctx.tracking_sigma) * gate, dtype=get_global_dtype())

    def _reward_lin_vel_z(self, ctx: RewardContext) -> np.ndarray:
        gate = self._upright_gate(ctx.gravity, ctx.num_envs)
        return np.asarray(np.square(ctx.linvel[:, 2]) * gate, dtype=get_global_dtype())

    def _reward_ang_vel_xy(self, ctx: RewardContext) -> np.ndarray:
        gate = self._upright_gate(ctx.gravity, ctx.num_envs)
        return np.asarray(
            np.sum(np.square(ctx.gyro[:, :2]), axis=1) * gate, dtype=get_global_dtype()
        )

    def _reward_stand_still(self, ctx: RewardContext) -> np.ndarray:
        gate = self._upright_gate(ctx.gravity, ctx.num_envs)
        commands = ctx.info["commands"]
        stopped = np.linalg.norm(commands[:, :2], axis=1) <= float(
            self._reward_cfg.wheel_contact_cmd_threshold
        )
        diff = ctx.dof_pos - ctx.default_angles
        penalty = np.sum(np.square(diff), axis=1)
        return np.asarray(
            penalty * stopped.astype(get_global_dtype()) * gate, dtype=get_global_dtype()
        )

    def _reward_joint_pos_penalty(self, ctx: RewardContext) -> np.ndarray:
        gate = self._upright_gate(ctx.gravity, ctx.num_envs)
        command_norm = np.linalg.norm(ctx.info["commands"][:, :2], axis=1)
        body_vel = np.linalg.norm(ctx.linvel[:, :2], axis=1)
        joint_error = np.linalg.norm(ctx.dof_pos - ctx.default_angles, axis=1)
        moving = (command_norm > self._reward_cfg.joint_pos_penalty_command_threshold) | (
            body_vel > self._reward_cfg.joint_pos_penalty_velocity_threshold
        )
        scale = np.where(moving, 1.0, self._reward_cfg.joint_pos_penalty_stand_still_scale)
        return np.asarray(joint_error * scale * gate, dtype=get_global_dtype())

    def _reward_joint_mirror(self, ctx: RewardContext) -> np.ndarray:
        gate = self._upright_gate(ctx.gravity, ctx.num_envs)
        hip_diff = ctx.dof_pos[:, 0] + ctx.dof_pos[:, 2]
        knee_diff = ctx.dof_pos[:, 1] + ctx.dof_pos[:, 3]
        return np.asarray(
            0.5 * (np.square(hip_diff) + np.square(knee_diff)) * gate,
            dtype=get_global_dtype(),
        )

    def _reward_collision(self, ctx: RewardContext) -> np.ndarray:
        gate = self._upright_gate(ctx.gravity, ctx.num_envs)
        base_contact = np.asarray(
            ctx.info.get("base_contact_force", np.zeros((ctx.num_envs,))), dtype=get_global_dtype()
        )
        return np.asarray(
            (base_contact > self._reward_cfg.collision_threshold).astype(get_global_dtype()) * gate,
            dtype=get_global_dtype(),
        )

    def _reward_contact_forces(self, ctx: RewardContext) -> np.ndarray:
        gate = self._upright_gate(ctx.gravity, ctx.num_envs)
        force = np.asarray(
            ctx.info.get("wheel_contact_forces", np.zeros((ctx.num_envs, 2))),
            dtype=get_global_dtype(),
        )
        excess = np.clip(force - self._reward_cfg.contact_forces_threshold, 0.0, None) / 100.0
        return np.asarray(np.sum(excess, axis=1) * gate, dtype=get_global_dtype())

    def _reward_upright_leg_contact(self, ctx: RewardContext) -> np.ndarray:
        gate = self._upright_gate(ctx.gravity, ctx.num_envs)
        active = gate >= self._reward_cfg.upright_contact_min_gate
        leg_contact = np.asarray(
            ctx.info.get("leg_contact_forces", np.zeros((ctx.num_envs, 4))),
            dtype=get_global_dtype(),
        )
        has_contact = np.any(leg_contact > self._reward_cfg.upright_contact_force_threshold, axis=1)
        return np.asarray(
            has_contact.astype(get_global_dtype()) * gate * active.astype(get_global_dtype()),
            dtype=get_global_dtype(),
        )

    def _reward_wheel_contact_without_cmd(self, ctx: RewardContext) -> np.ndarray:
        gate = self._upright_gate(ctx.gravity, ctx.num_envs)
        commands = ctx.info["commands"]
        stationary = (
            np.linalg.norm(commands[:, :2], axis=1) < self._reward_cfg.wheel_contact_cmd_threshold
        )
        wheel_contact = np.asarray(
            ctx.info.get("wheel_contact_forces", np.zeros((ctx.num_envs, 2))),
            dtype=get_global_dtype(),
        )
        has_contact = wheel_contact > self._reward_cfg.upright_contact_force_threshold
        return np.asarray(
            np.sum(has_contact.astype(get_global_dtype()), axis=1)
            * gate
            * stationary.astype(get_global_dtype()),
            dtype=get_global_dtype(),
        )

    def _reward_diagnostics(self, ctx: RewardContext) -> np.ndarray:
        return np.zeros((ctx.num_envs,), dtype=get_global_dtype())

    def _contact_forces(
        self, *, include_base: bool, include_leg: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        names = list(WHEEL_CONTACT_SENSOR_NAMES)
        base_index: int | None = None
        leg_start_index: int | None = None
        if include_base:
            base_index = len(names)
            names.append(BASE_CONTACT_SENSOR_NAME)
        if include_leg:
            leg_start_index = len(names)
            names.extend(LEG_CONTACT_SENSOR_NAMES)
        try:
            batch = np.asarray(self._backend.get_sensor_data_batch(names), dtype=np.float64)
        except (AttributeError, KeyError, NotImplementedError):
            base = (
                self._sensor_scalar(BASE_CONTACT_SENSOR_NAME)
                if include_base
                else self._zero_base_contact_force
            )
            leg = self._leg_contact_forces() if include_leg else self._zero_leg_contact_forces
            return base, self._wheel_contact_forces(), leg

        for column, sensor_index in enumerate(range(len(WHEEL_CONTACT_SENSOR_NAMES))):
            self._fill_contact_force(batch, sensor_index, self._wheel_contact_force_buf[:, column])
        base = self._zero_base_contact_force
        if base_index is not None:
            base = self._fill_contact_force(batch, base_index, self._base_contact_force_buf)
        leg = self._zero_leg_contact_forces
        if leg_start_index is not None:
            leg = self._leg_contact_force_buf
            for column, sensor_index in enumerate(
                range(leg_start_index, leg_start_index + len(LEG_CONTACT_SENSOR_NAMES))
            ):
                self._fill_contact_force(batch, sensor_index, leg[:, column])
        return base, self._wheel_contact_force_buf, leg

    def _wheel_contact_forces(self) -> np.ndarray:
        contact = self._wheel_contact_force_buf
        contact[:, 0] = self._sensor_scalar("l_wheel_contact")
        contact[:, 1] = self._sensor_scalar("r_wheel_contact")
        return contact

    def _leg_contact_forces(self) -> np.ndarray:
        contact = self._leg_contact_force_buf
        contact[:, 0] = self._sensor_scalar("lf0_contact")
        contact[:, 1] = self._sensor_scalar("lf1_contact")
        contact[:, 2] = self._sensor_scalar("rf0_contact")
        contact[:, 3] = self._sensor_scalar("rf1_contact")
        return contact

    def _fill_contact_force(
        self, batch: np.ndarray, sensor_index: int, out: np.ndarray
    ) -> np.ndarray:
        start = int(sensor_index) * CONTACT_SENSOR_FORCE_DIM
        force = batch[:, start : start + CONTACT_SENSOR_FORCE_DIM]
        magnitude = np.sqrt(np.sum(np.square(force), axis=1))
        return self._finite_contact_force_into(magnitude, out)

    def _sensor_scalar(self, name: str) -> np.ndarray:
        try:
            values = np.asarray(self._backend.get_sensor_data(name), dtype=get_global_dtype())
        except KeyError:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        flat = values.reshape(values.shape[0], -1)
        if flat.shape[1] >= CONTACT_SENSOR_FORCE_DIM:
            force_mag = np.linalg.norm(flat[:, :CONTACT_SENSOR_FORCE_DIM], axis=1)
            return self._finite_contact_force(force_mag)
        return self._finite_contact_force(flat[:, 0])

    def _finite_contact_force_into(self, force: np.ndarray, out: np.ndarray) -> np.ndarray:
        np.copyto(out, force, casting="same_kind")
        np.nan_to_num(
            out,
            copy=False,
            nan=CONTACT_FORCE_MAX_N,
            posinf=CONTACT_FORCE_MAX_N,
            neginf=0.0,
        )
        np.clip(out, 0.0, CONTACT_FORCE_MAX_N, out=out)
        return out

    def _finite_contact_force(self, force: np.ndarray) -> np.ndarray:
        finite = np.nan_to_num(
            np.asarray(force, dtype=np.float64),
            nan=CONTACT_FORCE_MAX_N,
            posinf=CONTACT_FORCE_MAX_N,
            neginf=0.0,
        )
        return np.asarray(np.clip(finite, 0.0, CONTACT_FORCE_MAX_N), dtype=get_global_dtype())

    def _write_recovery_diagnostics(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
    ) -> None:
        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        if not self._enable_reward_log or int(step_count[0]) % 4 != 0:
            return
        log = info.get("log", {})
        gate = self._upright_gate(gravity, gravity.shape[0])
        tilt_deg = np.rad2deg(np.arccos(np.clip(gravity[:, 2], -1.0, 1.0)))
        base_contact = np.asarray(
            info.get("base_contact_force", np.zeros((gravity.shape[0],))), dtype=get_global_dtype()
        )
        wheel_contact = np.asarray(
            info.get("wheel_contact_forces", np.zeros((gravity.shape[0], 2))),
            dtype=get_global_dtype(),
        )
        leg_contact = np.asarray(
            info.get("leg_contact_forces", np.zeros((gravity.shape[0], 4))),
            dtype=get_global_dtype(),
        )
        wheel_has_contact = wheel_contact > self._reward_cfg.upright_contact_force_threshold
        leg_has_contact = leg_contact > self._reward_cfg.upright_contact_force_threshold
        current_actions = np.asarray(
            info.get("current_actions", np.zeros((gravity.shape[0], self._num_action))),
            dtype=get_global_dtype(),
        )
        active_angle_left = dof_pos[:, 0] - dof_pos[:, 1]
        active_angle_right = dof_pos[:, 4] - dof_pos[:, 3]
        active_margin = np.minimum(
            np.minimum(active_angle_left - ACTIVE_LOWER, ACTIVE_UPPER - active_angle_left),
            np.minimum(active_angle_right - ACTIVE_LOWER, ACTIVE_UPPER - active_angle_right),
        )
        log.update(
            {
                "Recovery/tilt_deg": float(np.mean(tilt_deg)),
                "Recovery/upright_gate": float(np.mean(gate)),
                "Recovery/upright_15deg_rate": float(np.mean(tilt_deg < 15.0)),
                "Recovery/upright_30deg_rate": float(np.mean(tilt_deg < 30.0)),
                "Recovery/side_region_rate": float(
                    np.mean((tilt_deg >= 60.0) & (tilt_deg <= 120.0))
                ),
                "Recovery/inverted_130deg_rate": float(np.mean(tilt_deg > 130.0)),
                "Recovery/base_contact_rate": float(
                    np.mean(base_contact > self._reward_cfg.collision_threshold)
                ),
                "Recovery/wheel_contact_rate": float(np.mean(wheel_has_contact)),
                "Recovery/dual_wheel_contact_rate": float(
                    np.mean(np.all(wheel_has_contact, axis=1))
                ),
                "Recovery/leg_contact_rate": float(np.mean(np.any(leg_has_contact, axis=1))),
                "Recovery/contact_force_mean": float(np.mean(wheel_contact)),
                "Recovery/contact_force_max": float(np.max(wheel_contact)),
                "Recovery/base_vxy_mean": float(np.mean(np.linalg.norm(linvel[:, :2], axis=1))),
                "Recovery/base_vz_abs_mean": float(np.mean(np.abs(linvel[:, 2]))),
                "Recovery/base_ang_xy_mean": float(np.mean(np.linalg.norm(gyro[:, :2], axis=1))),
                "Recovery/action_saturation_rate": float(
                    np.mean(
                        np.max(np.abs(current_actions), axis=1)
                        > self._reward_cfg.action_saturation_threshold
                    )
                ),
                "Recovery/active_rod_margin_warning_rate": float(
                    np.mean(active_margin < self._reward_cfg.active_rod_margin_warning)
                ),
            }
        )
        info["log"] = log


registry.register_env(
    "SerialLegOpenChainRecovery", SerialLegOpenChainRecoveryEnv, sim_backend="motrix"
)
