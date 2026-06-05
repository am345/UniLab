from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import gymnasium as gym
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import DomainRandomizationCapabilities, ResetPlan, ResetRandomizationPayload
from unilab.dr.dr_utils import (
    build_interval_push_plan,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)
from unilab.envs.locomotion.common.commands import (
    Commands,
    apply_heading_yaw_feedback,
    sample_heading_commands,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext

SERIALLEG_JOINT_NAMES: tuple[str, ...] = (
    "lf0_Joint",
    "lf1_Joint",
    "l_wheel_Joint",
    "rf0_Joint",
    "rf1_Joint",
    "r_wheel_Joint",
)
SERIALLEG_LEG_INDICES = np.asarray([0, 1, 3, 4], dtype=np.int32)
SERIALLEG_WHEEL_INDICES = np.asarray([2, 5], dtype=np.int32)
NUM_SERIALLEG_ACTIONS = len(SERIALLEG_JOINT_NAMES)
NUM_SERIALLEG_LEG_ACTIONS = len(SERIALLEG_LEG_INDICES)
NUM_SERIALLEG_WHEEL_ACTIONS = len(SERIALLEG_WHEEL_INDICES)
SERIALLEG_OBS_DIM = 32
SERIALLEG_CRITIC_DIM = 41
SERIALLEG_COMMAND_SCALE = np.asarray([2.0, 0.25, 5.0, 5.0, 5.0], dtype=np.float64)
DEFAULT_SERIALLEG_ANGLES = np.asarray(
    [
        -0.275422946189,
        -1.242259649307,
        0.0,
        0.275422946189,
        1.242259649307,
        0.0,
    ],
    dtype=np.float64,
)
DEFAULT_BASE_HEIGHT = 0.22
SERIALLEG_FORCE_LOWER = np.asarray([-40.0, -40.0, -2.210526315789, -40.0, -40.0, -2.210526315789])
SERIALLEG_FORCE_UPPER = np.asarray([40.0, 40.0, 2.210526315789, 40.0, 40.0, 2.210526315789])


@dataclass
class InitState:
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, DEFAULT_BASE_HEIGHT])


@dataclass
class SerialLegOpenChainNoiseConfig(BaseNoiseConfig):
    level: float = 1.0
    scale_joint_angle: float = 0.01
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.0
    scale_wheel_vel: float = 0.5


@dataclass
class SerialLegOpenChainControlConfig(ControlConfigBase):
    action_scale: float = 0.25
    leg_action_scale: list[float] = field(default_factory=lambda: [0.35, 0.12, 0.35, 0.12])
    wheel_action_scale: float = 45.0
    Kp: float = 40.0  # noqa: N815
    Kd: float = 2.0  # noqa: N815
    wheel_Kd: float = 0.5  # noqa: N815
    clip_actions: float = 1.0


@dataclass
class SerialLegOpenChainDomainRandConfig(DomainRandConfig):
    randomize_init_yaw: bool = True
    init_yaw_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    push_body_name: str | None = "base_link"


@dataclass
class SerialLegOpenChainRewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float
    only_positive_rewards: bool = False
    joint_pos_penalty_stand_still_scale: float = 5.0
    joint_pos_penalty_velocity_threshold: float = 0.5
    joint_pos_penalty_command_threshold: float = 0.1


@dataclass
class SerialLegOpenChainSensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    gravity = "upvector"


@dataclass
class SerialLegOpenChainAsset:
    base_name: str = "base_link"
    ground: str = "floor"


@registry.envcfg("SerialLegOpenChainFlat")
@dataclass
class SerialLegOpenChainFlatCfg(LocomotionBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(
                ASSETS_ROOT_PATH
                / "robots"
                / "serialleg"
                / "serialleg_openchain_direct_actuator.xml"
            )
        )
    )
    max_episode_seconds: float = 20.0
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: SerialLegOpenChainRewardConfig | None = None
    sensor: SerialLegOpenChainSensor = field(default_factory=SerialLegOpenChainSensor)  # type: ignore[assignment]
    noise_config: SerialLegOpenChainNoiseConfig = field(
        default_factory=SerialLegOpenChainNoiseConfig
    )  # type: ignore[assignment]
    control_config: SerialLegOpenChainControlConfig = field(
        default_factory=SerialLegOpenChainControlConfig
    )  # type: ignore[assignment]
    domain_rand: SerialLegOpenChainDomainRandConfig = field(
        default_factory=SerialLegOpenChainDomainRandConfig
    )
    asset: SerialLegOpenChainAsset = field(default_factory=SerialLegOpenChainAsset)


def build_serialleg_openchain_reset_randomization(
    env: Any, num_reset: int
) -> ResetRandomizationPayload | None:
    domain_rand = getattr(env.cfg, "domain_rand", None)
    if domain_rand is None:
        return None

    payload = ResetRandomizationPayload()
    if getattr(domain_rand, "randomize_base_mass", False):
        low, high = domain_rand.added_mass_range
        payload.base_mass_delta = np.random.uniform(low, high, size=(num_reset,))

    if getattr(domain_rand, "random_com", False):
        low, high = domain_rand.com_offset_x
        base_com_offset = np.zeros((num_reset, 3), dtype=np.float64)
        base_com_offset[:, 0] = np.random.uniform(low, high, size=(num_reset,))
        payload.base_com_offset = base_com_offset

    if getattr(domain_rand, "randomize_gravity", False):
        gravity_range = np.asarray(domain_rand.gravity_range, dtype=np.float64)
        if gravity_range.shape != (2, 3):
            raise ValueError(
                f"domain_rand.gravity_range must have shape (2, 3), got {gravity_range.shape}"
            )
        low = np.minimum(gravity_range[0], gravity_range[1])
        high = np.maximum(gravity_range[0], gravity_range[1])
        payload.gravity = np.random.uniform(low=low, high=high, size=(num_reset, 3))

    return None if payload.is_empty() else payload


def sample_serialleg_reset_yaw(
    domain_rand: SerialLegOpenChainDomainRandConfig, num_reset: int
) -> np.ndarray:
    if not domain_rand.randomize_init_yaw:
        return np.zeros((num_reset,), dtype=get_global_dtype())
    yaw_range = np.asarray(domain_rand.init_yaw_range, dtype=np.float64)
    if yaw_range.shape != (2,):
        raise ValueError(f"domain_rand.init_yaw_range must have shape (2,), got {yaw_range.shape}")
    low, high = float(np.min(yaw_range)), float(np.max(yaw_range))
    return np.asarray(np.random.uniform(low, high, size=(num_reset,)), dtype=get_global_dtype())


class SerialLegOpenChainDomainRandomizationProvider(LocomotionDRProvider):
    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        payload = build_serialleg_openchain_reset_randomization(env, num_reset=1)
        if payload is not None:
            unsupported = capabilities.get_unsupported_reset_terms(payload.requested_terms())
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise NotImplementedError(
                    f"{env._backend.backend_type} backend does not support SerialLeg reset randomization terms: {names}"
                )
        validate_interval_push_support(env, capabilities)

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        return build_interval_push_plan(env, step_counter)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.1, 0.1, (num_reset, 2))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        yaw = sample_serialleg_reset_yaw(env.cfg.domain_rand, num_reset)
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        qpos[:, 7:] = DEFAULT_SERIALLEG_ANGLES
        qvel[:, :] = 0.0

        commands = self._sample_commands(env, num_reset)
        zero_small_xy_commands(commands)
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
        }
        if getattr(env.cfg.commands, "heading_command", False):
            info_updates["heading_commands"] = sample_heading_commands(env, num_reset)
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_serialleg_openchain_reset_randomization(env, num_reset),
        )

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: Any,
        info_updates: Any,
        linvel: Any,
        gyro: Any,
        gravity: Any,
        dof_pos: Any,
        dof_vel: Any,
    ) -> dict[str, np.ndarray]:
        del env_ids
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel),
        )


@registry.env("SerialLegOpenChainFlat", sim_backend="mujoco")
class SerialLegOpenChainFlatEnv(LocomotionBaseEnv):
    _cfg: SerialLegOpenChainFlatCfg

    def __init__(
        self, cfg: SerialLegOpenChainFlatCfg, num_envs: int = 1, backend_type: str = "mujoco"
    ):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        if isinstance(cfg.reward_config, dict):
            cfg.reward_config = SerialLegOpenChainRewardConfig(**cfg.reward_config)
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            add_body_sensors=False,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()
        self._leg_action_scale = self._build_leg_action_scale()
        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
        self._validate_motor_control_contract(ctrl_range)
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)
        self._force_lower = SERIALLEG_FORCE_LOWER.astype(self._np_dtype)
        self._force_upper = SERIALLEG_FORCE_UPPER.astype(self._np_dtype)
        joint_range = self._backend.get_joint_range()
        self._leg_joint_range = (
            np.asarray(joint_range[SERIALLEG_LEG_INDICES], dtype=get_global_dtype())
            if joint_range is not None
            else None
        )
        self._torque_estimate = np.zeros((num_envs, NUM_SERIALLEG_ACTIONS), dtype=self._np_dtype)
        self._last_dof_vel_for_acc = np.zeros(
            (num_envs, NUM_SERIALLEG_ACTIONS), dtype=get_global_dtype()
        )
        self._init_reward_functions()
        self._init_domain_randomization(SerialLegOpenChainDomainRandomizationProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": SERIALLEG_OBS_DIM, "critic": SERIALLEG_CRITIC_DIM}

    def _init_action_space(self) -> None:
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(NUM_SERIALLEG_ACTIONS,),
            dtype=np.float32,
        )

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.default_angles = np.asarray(DEFAULT_SERIALLEG_ANGLES, dtype=self.default_angles.dtype)

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        obs, info = super().reset(env_ids)
        dof_vel = self.get_dof_vel()
        if dof_vel.shape[0] == self._num_envs:
            self._last_dof_vel_for_acc[env_ids] = dof_vel[env_ids]
        return obs, info

    def _validate_motor_control_contract(self, ctrl_range: np.ndarray) -> None:
        if self._backend.num_actuators != NUM_SERIALLEG_ACTIONS:
            raise ValueError(
                f"SerialLeg requires {NUM_SERIALLEG_ACTIONS} direct actuators, got {self._backend.num_actuators}"
            )
        if ctrl_range.shape != (NUM_SERIALLEG_ACTIONS, 2):
            raise ValueError(
                f"SerialLeg actuator ctrl_range must have shape ({NUM_SERIALLEG_ACTIONS}, 2), got {ctrl_range.shape}"
            )
        pos_indices = self._backend.get_joint_dof_pos_indices(SERIALLEG_JOINT_NAMES)
        vel_indices = self._backend.get_joint_dof_vel_indices(SERIALLEG_JOINT_NAMES)
        expected = np.arange(NUM_SERIALLEG_ACTIONS, dtype=np.int32)
        if not np.array_equal(pos_indices, expected):
            raise ValueError("SerialLeg qpos order must match SERIALLEG_JOINT_NAMES")
        if not np.array_equal(vel_indices, expected):
            raise ValueError("SerialLeg qvel order must match SERIALLEG_JOINT_NAMES")

    def _build_leg_action_scale(self) -> np.ndarray:
        scale = np.asarray(self._cfg.control_config.leg_action_scale, dtype=self._np_dtype)
        if scale.shape != (NUM_SERIALLEG_LEG_ACTIONS,):
            raise ValueError(
                f"control_config.leg_action_scale must have shape ({NUM_SERIALLEG_LEG_ACTIONS},), got {scale.shape}"
            )
        return scale

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "action_rate": rewards.action_rate,
            "similar_to_default": rewards.similar_to_default,
            "orientation": rewards.orientation,
            "torques": self._reward_torques_l2,
            "joint_torques_l2": self._reward_joint_torques_l2,
            "energy": self._reward_energy,
            "dof_vel": self._reward_dof_vel,
            "dof_acc": self._reward_dof_acc,
            "joint_acc_l2": self._reward_dof_acc,
            "wheel_acc": self._reward_wheel_acc,
            "joint_acc_wheel_l2": self._reward_wheel_acc,
            "stand_still": self._reward_stand_still,
            "joint_pos_penalty": self._reward_joint_pos_penalty,
            "joint_power": self._reward_joint_power,
            "alive": rewards.alive,
            "upward": rewards.upward,
            "wheel_vel": self._reward_wheel_vel,
        }

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.asarray(
            np.clip(
                actions,
                -self._cfg.control_config.clip_actions,
                self._cfg.control_config.clip_actions,
            ),
            dtype=self._np_dtype,
        )
        state.info["last_actions"] = state.info.get(
            "current_actions", np.zeros_like(clipped_actions)
        )
        state.info["current_actions"] = clipped_actions
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else clipped_actions
        )

        ctrl = np.zeros((exec_actions.shape[0], NUM_SERIALLEG_ACTIONS), dtype=self._np_dtype)
        ctrl[:, SERIALLEG_LEG_INDICES] = (
            exec_actions[:, :NUM_SERIALLEG_LEG_ACTIONS] * self._leg_action_scale
            + self.default_angles[SERIALLEG_LEG_INDICES]
        )
        ctrl[:, SERIALLEG_WHEEL_INDICES] = (
            exec_actions[:, NUM_SERIALLEG_LEG_ACTIONS:]
            * self._cfg.control_config.wheel_action_scale
        )
        np.clip(ctrl, self._ctrl_lower, self._ctrl_upper, out=ctrl)
        state.info["current_ctrl"] = ctrl
        return ctrl

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.gravity)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        state.info["torques"] = self._estimate_direct_actuator_torques(state.info, dof_pos, dof_vel)
        state.info["qacc"] = self._estimate_dof_acc(dof_vel)
        terminated = self._compute_terminated(gravity)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        return gravity[:, 2] <= 0.5

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        leg_diff = dof_pos[:, SERIALLEG_LEG_INDICES] - self.default_angles[SERIALLEG_LEG_INDICES]
        leg_vel = dof_vel[:, SERIALLEG_LEG_INDICES]
        wheel_pos = dof_pos[:, SERIALLEG_WHEEL_INDICES]
        wheel_vel = dof_vel[:, SERIALLEG_WHEEL_INDICES]
        noisy_gyro = self._obs_noise(gyro * 0.25, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        noisy_leg_vel = self._obs_noise(leg_vel * 0.25, noise_cfg.scale_joint_vel)
        wheel_vel_scaled = wheel_vel * 0.05
        num_obs = gyro.shape[0]
        current_actions = np.asarray(
            info.get("current_actions", np.zeros((num_obs, self._num_action))),
            dtype=get_global_dtype(),
        )
        motor_ctrl = np.asarray(
            info.get("torques", np.zeros((num_obs, self._num_action), dtype=dof_pos.dtype)),
            dtype=get_global_dtype(),
        )
        commands = np.asarray(info["commands"], dtype=get_global_dtype())
        command_obs = np.zeros((num_obs, 5), dtype=get_global_dtype())
        command_obs[:, 0] = commands[:, 0]
        command_obs[:, 1] = commands[:, 2]
        command_obs[:, 4] = self._reward_cfg.base_height_target
        command_obs *= SERIALLEG_COMMAND_SCALE.astype(get_global_dtype())
        jump_commands = np.zeros((num_obs, 3), dtype=get_global_dtype())

        obs = np.concatenate(
            [
                noisy_gyro,
                -noisy_gravity,
                command_obs,
                noisy_leg_diff,
                noisy_leg_vel,
                wheel_pos,
                wheel_vel_scaled,
                current_actions,
                jump_commands,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate(
            [
                gyro * 0.25,
                -gravity,
                command_obs,
                leg_diff,
                leg_vel * 0.25,
                wheel_pos,
                wheel_vel_scaled,
                current_actions,
                jump_commands,
                linvel,
                motor_ctrl,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": critic}

    def _compute_reward(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, SERIALLEG_LEG_INDICES],
            dof_vel=dof_vel,
            num_envs=num_obs,
            default_angles=DEFAULT_SERIALLEG_ANGLES[SERIALLEG_LEG_INDICES].astype(dtype),
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._reward_base_height_values(num_obs),
            gravity=gravity,
            joint_range=self._leg_joint_range,
        )
        return rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._reward_cfg.only_positive_rewards,
        )

    def _update_commands(self, info: dict) -> None:
        commands = info.get("commands")
        if commands is None:
            return

        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        resampling_time = float(getattr(self._cfg.commands, "resampling_time", 0.0))
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / self._cfg.ctrl_dt)), 1)
            steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                low = np.asarray(self._cfg.commands.vel_limit[0], dtype=get_global_dtype())
                high = np.asarray(self._cfg.commands.vel_limit[1], dtype=get_global_dtype())
                sampled = np.random.uniform(low=low, high=high, size=(num_resample, 3)).astype(
                    get_global_dtype()
                )
                zero_small_xy_commands(sampled)
                standing_prob = float(getattr(self._cfg.commands, "rel_standing_envs", 0.0))
                if standing_prob > 0.0:
                    standing = np.random.uniform(size=(num_resample,)) < min(standing_prob, 1.0)
                    sampled[standing] = 0.0
                commands_arr[resample_mask] = sampled
                if getattr(self._cfg.commands, "heading_command", False):
                    heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
                    heading_commands[resample_mask] = sample_heading_commands(self, num_resample)
                    info["heading_commands"] = heading_commands

        if getattr(self._cfg.commands, "heading_command", False):
            heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
            base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
            if base_quat.shape[0] == commands_arr.shape[0]:
                stiffness = float(getattr(self._cfg.commands, "heading_control_stiffness", 0.5))
                apply_heading_yaw_feedback(
                    commands_arr, base_quat, heading_commands, stiffness=stiffness
                )
        info["commands"] = commands_arr

    def _ensure_heading_commands(self, info: dict, num_obs: int) -> np.ndarray:
        heading_commands = info.get("heading_commands")
        if heading_commands is None or np.asarray(heading_commands).shape != (num_obs,):
            heading_commands = sample_heading_commands(self, num_obs)
            info["heading_commands"] = heading_commands
        heading_commands = np.asarray(heading_commands, dtype=get_global_dtype())
        info["heading_commands"] = heading_commands
        return heading_commands

    def _estimate_dof_acc(self, dof_vel: np.ndarray) -> np.ndarray:
        qacc = np.asarray((dof_vel - self._last_dof_vel_for_acc) / self._cfg.ctrl_dt)
        self._last_dof_vel_for_acc[:] = dof_vel
        return np.asarray(qacc, dtype=get_global_dtype())

    def _estimate_direct_actuator_torques(
        self, info: dict, dof_pos: np.ndarray, dof_vel: np.ndarray
    ) -> np.ndarray:
        ctrl = np.asarray(
            info.get(
                "current_ctrl",
                np.broadcast_to(
                    DEFAULT_SERIALLEG_ANGLES, (dof_pos.shape[0], NUM_SERIALLEG_ACTIONS)
                ),
            ),
            dtype=self._np_dtype,
        )
        torque = self._torque_estimate
        torque[:, 0:2] = self._cfg.control_config.Kp * (ctrl[:, 0:2] - dof_pos[:, 0:2])
        torque[:, 0:2] -= self._cfg.control_config.Kd * dof_vel[:, 0:2]
        torque[:, 3:5] = self._cfg.control_config.Kp * (ctrl[:, 3:5] - dof_pos[:, 3:5])
        torque[:, 3:5] -= self._cfg.control_config.Kd * dof_vel[:, 3:5]
        torque[:, 2] = self._cfg.control_config.wheel_Kd * (ctrl[:, 2] - dof_vel[:, 2])
        torque[:, 5] = self._cfg.control_config.wheel_Kd * (ctrl[:, 5] - dof_vel[:, 5])
        np.clip(torque, self._force_lower, self._force_upper, out=torque)
        return torque.copy()

    def _reward_base_height_values(self, num_obs: int) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return np.zeros((num_obs,), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())

    def _reward_wheel_vel(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        wheel_vel = ctx.dof_vel[:, SERIALLEG_WHEEL_INDICES]
        return np.asarray(np.sum(np.square(wheel_vel), axis=1), dtype=get_global_dtype())

    def _reward_torques_l2(self, ctx: RewardContext) -> np.ndarray:
        torques = np.asarray(
            ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))),
            dtype=get_global_dtype(),
        )
        return np.asarray(np.sum(np.square(torques), axis=1), dtype=get_global_dtype())

    def _reward_joint_torques_l2(self, ctx: RewardContext) -> np.ndarray:
        torques = np.asarray(
            ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))),
            dtype=get_global_dtype(),
        )
        return np.asarray(
            np.sum(np.square(torques[:, SERIALLEG_LEG_INDICES]), axis=1),
            dtype=get_global_dtype(),
        )

    def _reward_energy(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        torques = np.asarray(
            ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))),
            dtype=get_global_dtype(),
        )
        return np.asarray(
            np.sum(np.abs(ctx.dof_vel * torques), axis=1),
            dtype=get_global_dtype(),
        )

    def _reward_dof_vel(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        return np.asarray(
            np.sum(np.square(ctx.dof_vel[:, SERIALLEG_LEG_INDICES]), axis=1),
            dtype=get_global_dtype(),
        )

    def _reward_dof_acc(self, ctx: RewardContext) -> np.ndarray:
        qacc = np.asarray(
            ctx.info.get("qacc", np.zeros((ctx.num_envs, NUM_SERIALLEG_ACTIONS))),
            dtype=get_global_dtype(),
        )
        return np.asarray(
            np.sum(np.square(qacc[:, SERIALLEG_LEG_INDICES]), axis=1), dtype=qacc.dtype
        )

    def _reward_wheel_acc(self, ctx: RewardContext) -> np.ndarray:
        qacc = np.asarray(
            ctx.info.get("qacc", np.zeros((ctx.num_envs, NUM_SERIALLEG_ACTIONS))),
            dtype=get_global_dtype(),
        )
        return np.asarray(
            np.sum(np.square(qacc[:, SERIALLEG_WHEEL_INDICES]), axis=1), dtype=qacc.dtype
        )

    def _reward_stand_still(self, ctx: RewardContext) -> np.ndarray:
        commands = ctx.info["commands"]
        stopped = np.linalg.norm(commands[:, :2], axis=1) < 0.1
        dof_error = np.sum(
            np.abs(ctx.dof_pos - DEFAULT_SERIALLEG_ANGLES[SERIALLEG_LEG_INDICES]), axis=1
        )
        return np.asarray(dof_error * stopped, dtype=get_global_dtype())

    def _reward_joint_pos_penalty(self, ctx: RewardContext) -> np.ndarray:
        return rewards.joint_pos_penalty(
            ctx,
            stand_still_scale=self._reward_cfg.joint_pos_penalty_stand_still_scale,
            velocity_threshold=self._reward_cfg.joint_pos_penalty_velocity_threshold,
            command_threshold=self._reward_cfg.joint_pos_penalty_command_threshold,
        )

    def _reward_joint_power(self, ctx: RewardContext) -> np.ndarray:
        assert ctx.dof_vel is not None
        torques = np.asarray(
            ctx.info.get("torques", np.zeros((ctx.num_envs, self._num_action))),
            dtype=get_global_dtype(),
        )
        return np.asarray(
            np.sum(
                np.abs(ctx.dof_vel[:, SERIALLEG_LEG_INDICES] * torques[:, SERIALLEG_LEG_INDICES]),
                axis=1,
            ),
            dtype=get_global_dtype(),
        )


registry.register_env("SerialLegOpenChainFlat", SerialLegOpenChainFlatEnv, sim_backend="motrix")
