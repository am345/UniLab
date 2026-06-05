from __future__ import annotations

import math
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
    build_common_reset_randomization,
    zero_actions,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import (
    np_matrix_from_quat,
    np_quat_apply_inverse,
    np_quat_mul,
    np_yaw_to_quat,
)
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.serialleg.fourbar import (
    ACTIVE_LOWER,
    ACTIVE_UPPER,
    output_to_policy_pos_vel_jacobian_np,
    output_to_policy_pos_vel_np,
    policy_to_output_pos_np,
    policy_to_output_torque_from_jacobian_np,
)

POLICY_JOINT_NAMES: tuple[str, ...] = (
    "lf0_Joint",
    "l_drive_bar_Joint",
    "rf0_Joint",
    "r_drive_bar_Joint",
    "l_wheel_Joint",
    "r_wheel_Joint",
)
NATIVE_JOINT_NAMES: tuple[str, ...] = (
    "lf0_Joint",
    "lf1_Joint",
    "l_wheel_Joint",
    "rf0_Joint",
    "rf1_Joint",
    "r_wheel_Joint",
)
OUTPUT_LEG_INDICES = np.asarray([0, 1, 3, 4], dtype=np.int32)
WHEEL_INDICES = np.asarray([2, 5], dtype=np.int32)
WHEEL_BODY_NAMES: tuple[str, ...] = ("l_wheel_Link", "r_wheel_Link")
BASE_CONTACT_SENSOR_NAME = "base_contact"
WHEEL_CONTACT_SENSOR_NAMES: tuple[str, ...] = ("l_wheel_contact", "r_wheel_contact")
LEG_CONTACT_SENSOR_NAMES: tuple[str, ...] = (
    "lf0_contact",
    "lf1_contact",
    "rf0_contact",
    "rf1_contact",
)
CONTACT_SENSOR_FORCE_DIM = 3

NUM_ACTIONS = 6
NUM_POLICY_LEG_ACTIONS = 4
NUM_WHEEL_ACTIONS = 2
ACTOR_OBS_DIM = 32
CRITIC_OBS_DIM = 38
FOURBAR_WHEEL_RADIUS = 0.060
RESET_WHEEL_CLEARANCE = 0.001

DEFAULT_POLICY_LEG_POS = np.asarray(
    [-0.275422946189, -1.592100148957, 0.275422946189, 1.592100148957],
    dtype=np.float64,
)
DEFAULT_POLICY_ACTION_POS = np.asarray(
    [-0.275422946189, -1.592100148957, 0.275422946189, 1.592100148957, 0.0, 0.0],
    dtype=np.float64,
)
DEFAULT_OUTPUT_LEG_POS = policy_to_output_pos_np(DEFAULT_POLICY_LEG_POS)
DEFAULT_NATIVE_DOF_POS = np.asarray(
    [
        DEFAULT_OUTPUT_LEG_POS[0],
        DEFAULT_OUTPUT_LEG_POS[1],
        0.0,
        DEFAULT_OUTPUT_LEG_POS[2],
        DEFAULT_OUTPUT_LEG_POS[3],
        0.0,
    ],
    dtype=np.float64,
)
DEFAULT_BASE_HEIGHT = 0.22
LEG_ACTION_SCALE = np.asarray([0.35, 0.25, 0.35, 0.25], dtype=np.float64)
WHEEL_ACTION_SCALE = 45.0
COMMAND_SCALE = np.asarray([2.0, 0.25, 5.0, 5.0, 5.0], dtype=np.float64)
CONTACT_FORCE_MAX_N = 5000.0
DM8009P_STALL_TORQUE = 40.0
DM8009P_NO_LOAD_SPEED = 160.0 * 2.0 * math.pi / 60.0
DM8009P_RATED_TORQUE = 20.0
M3508_C620_14_RATED_TORQUE = 3.0 * 14.0 / 19.0


@dataclass
class InitState:
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, DEFAULT_BASE_HEIGHT])


@dataclass
class SerialLegNoiseConfig(BaseNoiseConfig):
    level: float = 1.0
    scale_joint_angle: float = 0.01
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.0


@dataclass
class SerialLegControlConfig(ControlConfigBase):
    clip_actions: float | None = None
    leg_kp: float = 40.0
    leg_kd: float = 2.0
    wheel_kd: float = 0.5
    action_delay_enabled: bool = True
    action_delay_s: float = 0.005
    randomize_action_delay: bool = True
    min_action_delay_s: float = 0.004
    max_action_delay_s: float = 0.006


@dataclass
class SerialLegCommands:
    resampling_time: float = 5.0
    rel_standing_envs: float = 0.1
    pitch_range: list[float] = field(default_factory=lambda: [-0.2, 0.2])
    roll_range: list[float] = field(default_factory=lambda: [-0.1, 0.1])
    height_range: list[float] = field(default_factory=lambda: [0.20, 0.32])
    vx_deadband: float = 0.1
    yaw_deadband: float = 0.1
    steps_per_policy_iter: int = 32
    command_vel_schedule: list[list[float]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],
            [500.0, 0.5, 0.5],
            [1500.0, 1.0, 1.0],
            [2500.0, 1.5, 2.0],
            [3500.0, 2.0, 2.5],
            [4500.0, 2.5, 3.0],
        ]
    )


@dataclass
class SerialLegDomainRandConfig(DomainRandConfig):
    randomize_base_mass: bool = True
    added_mass_range: list[float] = field(default_factory=lambda: [-0.5, 1.5])
    randomize_ground_friction: bool = True
    ground_friction_multiplier_range: list[float] = field(default_factory=lambda: [0.25, 1.875])
    random_com: bool = True
    com_offset_x: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    com_offset_y: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    robot_friction_range: list[float] = field(default_factory=lambda: [0.2, 1.5])
    randomize_dof_armature: bool = False
    dof_armature_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    randomize_body_inertia: bool = True
    body_inertia_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    randomize_default_dof_pos: bool = True
    default_dof_pos_offset_range: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    push_robots: bool = True
    push_interval: int = 250
    push_interval_range_s: list[float] = field(default_factory=lambda: [5.0, 6.0])
    max_force: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.0])
    push_body_name: str | None = "base_link"


@dataclass
class SerialLegRewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 4.0,
            "tracking_ang_vel": 1.73,
            "tracking_orientation_l2": -12.0,
            "tracking_height": 2.49,
            "bad_tilt": -6.0,
            "ang_vel_xy": -0.146,
            "angular_momentum": -5.0e-5,
            "leg_torques": -2.0e-4,
            "wheel_torques": -1.0e-4,
            "stand_still": -1.0,
            "leg_dof_acc": -2.17e-7,
            "leg_power": -1.03e-4,
            "action_rate": -0.48,
            "joint_mirror": -0.179,
            "dof_pos_limits": -5.0,
            "collision": -16.0,
            "contact_forces": -1.07e-3,
            "upright_wheel_contact": -10.0,
            "upright_leg_contact": -25.0,
            "is_alive": 1.0,
        }
    )
    tracking_lin_vel_sigma_move: float = 0.08
    tracking_lin_vel_sigma_stand: float = 0.1
    tracking_lin_vel_vz_weight: float = 2.0
    tracking_ang_vel_sigma: float = 0.25
    tracking_height_sigma: float = 0.05
    bad_tilt_soft_limit_deg: float = 10.0
    bad_tilt_hard_limit_deg: float = 30.0
    bad_tilt_max_penalty: float = 4.0
    wheel_torques_max_torque: float = 3.0
    stand_still_command_threshold: float = 0.1
    stand_still_default_height: float = DEFAULT_BASE_HEIGHT
    stand_still_height_tolerance: float = 40.0
    contact_forces_threshold: float = 35.0
    collision_threshold: float = 0.1
    upright_contact_force_threshold: float = 1.0
    upright_contact_min_gate: float = 0.35
    only_positive_rewards: bool = False


@dataclass
class SerialLegAsset:
    base_name: str = "base_link"
    ground: str = "floor"


@dataclass
class SerialLegMujocoBackendConfig:
    nthread: int | str | None = None


@registry.envcfg("SerialLegFlatMLP")
@dataclass
class SerialLegFlatMLPCfg(LocomotionBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(
                ASSETS_ROOT_PATH / "robots" / "serialleg" / "serialleg_fourbar_surrogate_train.xml"
            )
        )
    )
    max_episode_seconds: float = 20.0
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    init_state: InitState = field(default_factory=InitState)
    commands: SerialLegCommands = field(default_factory=SerialLegCommands)
    control_config: SerialLegControlConfig = field(default_factory=SerialLegControlConfig)  # type: ignore[assignment]
    noise_config: SerialLegNoiseConfig = field(default_factory=SerialLegNoiseConfig)  # type: ignore[assignment]
    domain_rand: SerialLegDomainRandConfig = field(default_factory=SerialLegDomainRandConfig)
    reward_config: SerialLegRewardConfig | None = None
    asset: SerialLegAsset = field(default_factory=SerialLegAsset)
    mujoco_backend: SerialLegMujocoBackendConfig = field(
        default_factory=SerialLegMujocoBackendConfig
    )


class SerialLegFlatMLPDomainRandomizationProvider(LocomotionDRProvider):
    def _get_reset_randomization_baselines(
        self, env: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None, np.ndarray | None]:
        return (
            env._base_body_mass,
            env._base_geom_friction,
            env._ground_geom_id,
            env._base_dof_armature,
        )

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        payload = env.build_reset_randomization(np.asarray([0], dtype=np.int32))
        unsupported = (
            frozenset()
            if payload is None
            else capabilities.get_unsupported_reset_terms(payload.requested_terms())
        )
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                f"{env._backend.backend_type} backend does not support SerialLeg reset DR: {names}"
            )

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        env.update_push_curriculum()
        env.apply_velocity_push_if_due(step_counter)
        return None

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.1, 0.1, (num_reset, 2))
        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        qpos[:, 0:3] = env._spawn.apply_spawn(env_ids, qpos[:, 0:3], yaw=yaw)
        qvel[:, :] = 0.0

        leg_kp, leg_kd, default_policy_leg_pos = env.sample_reset_motor_params(env_ids)
        native_joint_pos = env.policy_default_to_native_qpos(default_policy_leg_pos)
        qpos[:, 7:] = native_joint_pos
        env.align_reset_qpos_to_wheel_clearance(env_ids, qpos, qvel)
        env.set_reset_runtime(env_ids, leg_kp, leg_kd, default_policy_leg_pos)

        info_updates: dict[str, Any] = {
            "commands": env.sample_commands(num_reset),
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "torques": np.zeros((num_reset, NUM_ACTIONS), dtype=get_global_dtype()),
            "policy_leg_torque": np.zeros(
                (num_reset, NUM_POLICY_LEG_ACTIONS), dtype=get_global_dtype()
            ),
            "policy_leg_vel": np.zeros(
                (num_reset, NUM_POLICY_LEG_ACTIONS), dtype=get_global_dtype()
            ),
            "policy_leg_acc": np.zeros(
                (num_reset, NUM_POLICY_LEG_ACTIONS), dtype=get_global_dtype()
            ),
            "wheel_contact_forces": np.zeros(
                (num_reset, NUM_WHEEL_ACTIONS), dtype=get_global_dtype()
            ),
        }
        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=env.build_reset_randomization(env_ids),
        )

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        del linvel, gyro, gravity
        base_pos, base_linvel, base_angvel, projected_gravity = env.base_state(env_ids=env_ids)
        return cast(
            dict[str, np.ndarray],
            env.compute_obs_from_arrays(
                info_updates,
                base_pos,
                base_linvel,
                base_angvel,
                projected_gravity,
                dof_pos,
                dof_vel,
                env_ids=env_ids,
            ),
        )


@registry.env("SerialLegFlatMLP", sim_backend="mujoco")
@registry.env("SerialLegFlatMLP", sim_backend="motrix")
class SerialLegFlatMLPEnv(LocomotionBaseEnv):
    _cfg: SerialLegFlatMLPCfg

    def __init__(self, cfg: SerialLegFlatMLPCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        if isinstance(cfg.reward_config, dict):
            cfg.reward_config = SerialLegRewardConfig(**cfg.reward_config)
        backend_kwargs: dict[str, Any] = {}
        if backend_type == "mujoco":
            backend_kwargs["nthread"] = cfg.mujoco_backend.nthread
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
            **backend_kwargs,
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()
        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True

        self._dof_pos_indices = self._backend.get_joint_dof_pos_indices(NATIVE_JOINT_NAMES)
        self._dof_vel_indices = self._backend.get_joint_dof_vel_indices(NATIVE_JOINT_NAMES)
        expected_indices = np.arange(NUM_ACTIONS, dtype=np.int32)
        if not np.array_equal(self._dof_pos_indices, expected_indices):
            raise ValueError(
                "SerialLeg native joint qpos order must match NATIVE_JOINT_NAMES for reset qpos"
            )
        if not np.array_equal(self._dof_vel_indices, expected_indices):
            raise ValueError(
                "SerialLeg native joint qvel order must match NATIVE_JOINT_NAMES for control"
            )
        self._dof_pos_indices = None
        self._dof_vel_indices = None
        if self._backend.num_actuators != NUM_ACTIONS:
            raise ValueError(
                f"SerialLeg requires {NUM_ACTIONS} motor actuators, got {self._backend.num_actuators}"
            )
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=np.float64)
        if ctrl_range.shape != (NUM_ACTIONS, 2):
            raise ValueError(
                f"SerialLeg actuator ctrl range shape must be (6, 2), got {ctrl_range.shape}"
            )
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)
        self._joint_range = self._backend.get_joint_range()

        self._base_leg_kp = np.full(
            (NUM_POLICY_LEG_ACTIONS,), cfg.control_config.leg_kp, dtype=np.float64
        )
        self._base_leg_kd = np.full(
            (NUM_POLICY_LEG_ACTIONS,), cfg.control_config.leg_kd, dtype=np.float64
        )
        self._leg_kp = np.broadcast_to(self._base_leg_kp, (num_envs, NUM_POLICY_LEG_ACTIONS)).copy()
        self._leg_kd = np.broadcast_to(self._base_leg_kd, (num_envs, NUM_POLICY_LEG_ACTIONS)).copy()
        self._default_policy_leg_pos = np.broadcast_to(
            DEFAULT_POLICY_LEG_POS, (num_envs, NUM_POLICY_LEG_ACTIONS)
        ).copy()
        self._policy_leg_torque = np.zeros((num_envs, NUM_POLICY_LEG_ACTIONS), dtype=self._np_dtype)
        self._policy_leg_vel = np.zeros((num_envs, NUM_POLICY_LEG_ACTIONS), dtype=self._np_dtype)
        self._policy_leg_pos = np.zeros((num_envs, NUM_POLICY_LEG_ACTIONS), dtype=self._np_dtype)
        self._last_policy_leg_vel = np.zeros_like(self._policy_leg_vel)
        self._policy_leg_acc = np.zeros_like(self._policy_leg_vel)
        self._policy_order_torque_buf = np.zeros((num_envs, NUM_ACTIONS), dtype=self._np_dtype)
        self._last_motor_ctrl = np.zeros((num_envs, NUM_ACTIONS), dtype=self._np_dtype)
        self._bad_orientation_steps = np.zeros((num_envs,), dtype=np.int32)
        self._base_contact_force_buf = np.zeros((num_envs,), dtype=self._np_dtype)
        self._zero_base_contact_force = np.zeros((num_envs,), dtype=self._np_dtype)
        self._wheel_contact_force_buf = np.zeros(
            (num_envs, NUM_WHEEL_ACTIONS), dtype=self._np_dtype
        )
        self._leg_contact_force_buf = np.zeros(
            (num_envs, NUM_POLICY_LEG_ACTIONS), dtype=self._np_dtype
        )
        self._zero_leg_contact_forces = np.zeros(
            (num_envs, NUM_POLICY_LEG_ACTIONS), dtype=self._np_dtype
        )
        self._actor_obs_buf = np.zeros((num_envs, ACTOR_OBS_DIM), dtype=self._np_dtype)
        self._critic_obs_buf = np.zeros((num_envs, CRITIC_OBS_DIM), dtype=self._np_dtype)
        self._init_action_delay_buffers(num_envs)
        self._next_push_step = self._sample_push_interval_steps()

        self._base_body_mass = self._backend.get_body_mass()
        self._base_geom_friction = self._resolve_base_geom_friction()
        self._base_dof_armature = self._resolve_base_dof_armature()
        self._base_body_inertia = self._resolve_base_body_inertia()
        self._base_body_id = self._backend.get_body_id(cfg.asset.base_name)
        self._robot_body_ids = self._backend.get_body_subtree_ids(self._base_body_id)
        self._robot_body_mass = self._base_body_mass[self._robot_body_ids]
        self._robot_body_inertia = (
            None
            if self._base_body_inertia is None
            else self._base_body_inertia[self._robot_body_ids]
        )
        self._ground_geom_id = self._backend.get_geom_id(cfg.asset.ground)
        geom_body_ids = self._backend.get_geom_body_ids()
        self._robot_geom_ids = np.flatnonzero(np.isin(geom_body_ids, self._robot_body_ids)).astype(
            np.int32
        )
        self._robot_friction_geom_ids = self._resolve_robot_friction_geom_ids()
        self._wheel_body_ids = self._backend.get_body_ids(WHEEL_BODY_NAMES)
        self._init_startup_randomization()
        self._backend.set_pre_step_control(self._pre_step_motor_control)
        self._init_reward_functions()
        self._init_domain_randomization(SerialLegFlatMLPDomainRandomizationProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": ACTOR_OBS_DIM, "critic": CRITIC_OBS_DIM}

    def _init_action_space(self) -> None:
        clip_actions = self._cfg.control_config.clip_actions
        low = -np.inf if clip_actions is None else -float(clip_actions)
        high = np.inf if clip_actions is None else float(clip_actions)
        self._action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(NUM_ACTIONS,),
            dtype=np.float32,
        )

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.default_angles = np.asarray(DEFAULT_NATIVE_DOF_POS, dtype=self.default_angles.dtype)

    def get_dof_pos(self) -> np.ndarray:
        dof_pos = self._backend.get_dof_pos()
        indices = getattr(self, "_dof_pos_indices", None)
        if indices is None:
            return dof_pos
        return dof_pos[:, indices]

    def get_dof_vel(self) -> np.ndarray:
        dof_vel = self._backend.get_dof_vel()
        indices = getattr(self, "_dof_vel_indices", None)
        if indices is None:
            return dof_vel
        return dof_vel[:, indices]

    def _init_action_delay_buffers(self, num_envs: int) -> None:
        delay_cfg = self._cfg.control_config
        min_steps = self._delay_seconds_to_steps(delay_cfg.min_action_delay_s)
        max_steps = self._delay_seconds_to_steps(delay_cfg.max_action_delay_s)
        nominal_steps = self._delay_seconds_to_steps(delay_cfg.action_delay_s)
        self._min_delay_steps = max(0, min(min_steps, max_steps))
        self._max_delay_steps = max(self._min_delay_steps, max_steps)
        self._nominal_delay_steps = int(
            np.clip(nominal_steps, self._min_delay_steps, self._max_delay_steps)
        )
        self._delay_steps = np.full((num_envs,), self._nominal_delay_steps, dtype=np.int32)
        self._delay_steps_intp = self._delay_steps.astype(np.intp)
        self._env_row_indices = np.arange(num_envs, dtype=np.intp)
        self._action_delay_head = 0
        self._delay_slot_indices = np.zeros((num_envs,), dtype=np.intp)
        self._action_delay_fifo = np.zeros(
            (self._max_delay_steps + 1, num_envs, NUM_ACTIONS), dtype=self._np_dtype
        )

    def _delay_seconds_to_steps(self, seconds: float) -> int:
        if seconds <= 0.0:
            return 0
        return max(int(round(float(seconds) / self._cfg.sim_dt)), 1)

    def _resolve_base_geom_friction(self) -> np.ndarray:
        try:
            return np.asarray(self._backend.get_geom_friction(), dtype=np.float64).copy()
        except NotImplementedError:
            if self._cfg.domain_rand.randomize_ground_friction:
                raise
            return np.zeros((0, 3), dtype=np.float64)

    def _resolve_base_dof_armature(self) -> np.ndarray:
        try:
            return np.asarray(self._backend.get_dof_armature(), dtype=np.float64).copy()
        except NotImplementedError:
            if self._cfg.domain_rand.randomize_dof_armature:
                raise
            return np.zeros((self._backend.num_dof_vel,), dtype=np.float64)

    def _resolve_base_body_inertia(self) -> np.ndarray | None:
        body_inertia = getattr(self._backend.model, "body_inertia", None)
        if body_inertia is not None:
            return np.asarray(body_inertia, dtype=np.float64).copy()
        if self._cfg.domain_rand.randomize_body_inertia:
            raise NotImplementedError(
                f"{self._backend.backend_type} backend does not expose body inertia"
            )
        return None

    def _resolve_robot_friction_geom_ids(self) -> np.ndarray:
        robot_geom_ids = np.asarray(self._robot_geom_ids, dtype=np.int32)
        try:
            contype, conaffinity = self._backend.get_geom_contact_masks()
        except NotImplementedError:
            return robot_geom_ids
        contact_geom_ids = np.flatnonzero(
            (np.asarray(contype, dtype=np.int32) != 0)
            | (np.asarray(conaffinity, dtype=np.int32) != 0)
        ).astype(np.int32)
        return np.intersect1d(robot_geom_ids, contact_geom_ids, assume_unique=False).astype(
            np.int32
        )

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        obs, info = super().reset(env_ids)
        self._last_policy_leg_vel[env_ids] = self._policy_leg_vel[env_ids]
        return obs, info

    def sample_commands(self, num_reset: int) -> np.ndarray:
        vx_limit, yaw_limit = self.current_command_limits()
        commands = np.zeros((num_reset, 5), dtype=get_global_dtype())
        commands[:, 0] = np.random.uniform(-vx_limit, vx_limit, size=(num_reset,))
        commands[:, 1] = np.random.uniform(-yaw_limit, yaw_limit, size=(num_reset,))
        commands[:, 2] = np.random.uniform(
            self._cfg.commands.pitch_range[0], self._cfg.commands.pitch_range[1], size=(num_reset,)
        )
        commands[:, 3] = np.random.uniform(
            self._cfg.commands.roll_range[0], self._cfg.commands.roll_range[1], size=(num_reset,)
        )
        commands[:, 4] = np.random.uniform(
            self._cfg.commands.height_range[0],
            self._cfg.commands.height_range[1],
            size=(num_reset,),
        )
        commands[np.abs(commands[:, 0]) < self._cfg.commands.vx_deadband, 0] = 0.0
        commands[np.abs(commands[:, 1]) < self._cfg.commands.yaw_deadband, 1] = 0.0
        standing_count = int(
            num_reset * max(0.0, min(float(self._cfg.commands.rel_standing_envs), 1.0))
        )
        if standing_count > 0:
            commands[:standing_count, 0:4] = 0.0
        return commands

    def current_command_limits(self) -> tuple[float, float]:
        steps_per_iter = max(int(self._cfg.commands.steps_per_policy_iter), 1)
        ppo_iter = self.step_counter // steps_per_iter
        vx_limit = 0.0
        yaw_limit = 0.0
        for stage in self._cfg.commands.command_vel_schedule:
            if len(stage) < 3:
                continue
            if ppo_iter >= int(stage[0]):
                vx_limit = float(stage[1])
                yaw_limit = float(stage[2])
        return vx_limit, yaw_limit

    def update_push_curriculum(self) -> None:
        steps_per_iter = max(int(self._cfg.commands.steps_per_policy_iter), 1)
        ppo_iter = self.step_counter // steps_per_iter
        limit = 0.0
        for stage_iter, stage_limit in (
            (0, 0.0),
            (2000, 0.3),
            (5000, 0.5),
            (10000, 1.0),
            (20000, 1.5),
            (40000, 2.0),
        ):
            if ppo_iter >= stage_iter:
                limit = stage_limit
        self._cfg.domain_rand.max_force = [limit, limit, 0.0]

    def _sample_push_interval_steps(self) -> int:
        low_s, high_s = self._cfg.domain_rand.push_interval_range_s
        low_steps = max(int(round(float(low_s) / self._cfg.ctrl_dt)), 1)
        high_steps = max(int(round(float(high_s) / self._cfg.ctrl_dt)), low_steps)
        return int(np.random.randint(low_steps, high_steps + 1))

    def apply_velocity_push_if_due(self, step_counter: int) -> None:
        domain_rand = self._cfg.domain_rand
        if not domain_rand.push_robots:
            return
        if step_counter < self._next_push_step:
            return
        self._next_push_step = step_counter + self._sample_push_interval_steps()
        velocity_limit = np.asarray(domain_rand.max_force, dtype=np.float64)
        if not np.any(velocity_limit):
            return
        base_lin_vel = getattr(self._backend, "_base_lin_vel_view", None)
        if base_lin_vel is None:
            return
        delta = np.random.uniform(-1.0, 1.0, size=(self._num_envs, 3)) * velocity_limit
        base_lin_vel[:] = np.asarray(base_lin_vel, dtype=np.float64) + delta

    def _init_startup_randomization(self) -> None:
        (
            self._startup_leg_kp,
            self._startup_leg_kd,
            self._startup_default_policy_leg_pos,
        ) = self._sample_startup_motor_params(self._num_envs)
        self._leg_kp[:] = self._startup_leg_kp
        self._leg_kd[:] = self._startup_leg_kd
        self._default_policy_leg_pos[:] = self._startup_default_policy_leg_pos
        self._startup_reset_randomization = self._sample_startup_reset_randomization(self._num_envs)

    def _sample_startup_motor_params(
        self, num_envs: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        domain_rand = self._cfg.domain_rand
        kp = np.broadcast_to(self._base_leg_kp, (num_envs, NUM_POLICY_LEG_ACTIONS)).copy()
        kd = np.broadcast_to(self._base_leg_kd, (num_envs, NUM_POLICY_LEG_ACTIONS)).copy()
        if domain_rand.randomize_kp:
            kp *= np.random.uniform(*domain_rand.kp_multiplier_range, size=(num_envs, 1))
        if domain_rand.randomize_kd:
            kd *= np.random.uniform(*domain_rand.kd_multiplier_range, size=(num_envs, 1))
        default_pos = np.broadcast_to(
            DEFAULT_POLICY_LEG_POS, (num_envs, NUM_POLICY_LEG_ACTIONS)
        ).copy()
        if domain_rand.randomize_default_dof_pos:
            low, high = domain_rand.default_dof_pos_offset_range
            default_pos += np.random.uniform(low, high, size=default_pos.shape)
            default_pos = self._clamp_active_rod_angles(default_pos)
        return kp, kd, default_pos

    def sample_reset_motor_params(
        self, env_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.asarray(env_ids, dtype=np.intp)
        return (
            self._startup_leg_kp[rows].copy(),
            self._startup_leg_kd[rows].copy(),
            self._startup_default_policy_leg_pos[rows].copy(),
        )

    def _legacy_build_reset_randomization(self, num_reset: int) -> ResetRandomizationPayload | None:
        domain_rand = self._cfg.domain_rand
        original_randomize_kp = domain_rand.randomize_kp
        original_randomize_kd = domain_rand.randomize_kd
        domain_rand.randomize_kp = False
        domain_rand.randomize_kd = False
        try:
            # 腿部 kp/kd 在 policy-space PD 中按 reset 采样，不能作为 backend actuator gain payload。
            payload = build_common_reset_randomization(
                self,
                num_reset,
                base_body_mass=self._base_body_mass,
                base_geom_friction=self._base_geom_friction,
                ground_geom_id=self._ground_geom_id,
                base_dof_armature=self._base_dof_armature,
            )
        finally:
            domain_rand.randomize_kp = original_randomize_kp
            domain_rand.randomize_kd = original_randomize_kd
        if domain_rand.randomize_body_inertia:
            if self._base_body_inertia is None:
                raise NotImplementedError(
                    f"{self._backend.backend_type} backend does not expose body inertia"
                )
            if payload is None:
                payload = ResetRandomizationPayload()
            inertia = np.broadcast_to(
                self._base_body_inertia, (num_reset, *self._base_body_inertia.shape)
            ).copy()
            randomized = self._base_body_inertia > 0.0
            low, high = domain_rand.body_inertia_multiplier_range
            inertia[:, randomized] *= np.random.uniform(
                low, high, size=(num_reset, int(np.count_nonzero(randomized)))
            )
            payload.body_inertia = inertia
        return None if payload is None or payload.is_empty() else payload

    def _sample_startup_reset_randomization(
        self, num_envs: int
    ) -> ResetRandomizationPayload | None:
        domain_rand = self._cfg.domain_rand
        payload = ResetRandomizationPayload()

        if domain_rand.randomize_base_mass:
            low, high = domain_rand.added_mass_range
            payload.base_mass_delta = np.random.uniform(low, high, size=(num_envs,))

        if domain_rand.random_com:
            base_com_offset = np.zeros((num_envs, 3), dtype=np.float64)
            low, high = domain_rand.com_offset_x
            base_com_offset[:, 0] = np.random.uniform(low, high, size=(num_envs,))
            low, high = domain_rand.com_offset_y
            base_com_offset[:, 1] = np.random.uniform(low, high, size=(num_envs,))
            low, high = domain_rand.com_offset_z
            base_com_offset[:, 2] = np.random.uniform(low, high, size=(num_envs,))
            payload.base_com_offset = base_com_offset

        if domain_rand.randomize_ground_friction:
            geom_friction = np.broadcast_to(
                self._base_geom_friction, (num_envs, *self._base_geom_friction.shape)
            ).copy()
            robot_geom_ids = np.asarray(
                getattr(self, "_robot_friction_geom_ids", self._robot_geom_ids), dtype=np.intp
            )
            if robot_geom_ids.size:
                low, high = domain_rand.robot_friction_range
                geom_friction[:, robot_geom_ids, 0] = np.random.uniform(
                    low, high, size=(num_envs, 1)
                )
            payload.geom_friction = geom_friction

        if domain_rand.randomize_dof_armature:
            dof_armature = np.broadcast_to(
                self._base_dof_armature, (num_envs, self._base_dof_armature.size)
            ).copy()
            randomized = self._base_dof_armature > 0.0
            low, high = domain_rand.dof_armature_multiplier_range
            dof_armature[:, randomized] *= np.random.uniform(
                low, high, size=(num_envs, int(np.count_nonzero(randomized)))
            )
            payload.dof_armature = dof_armature

        if domain_rand.randomize_body_inertia:
            if self._base_body_inertia is None:
                raise NotImplementedError(
                    f"{self._backend.backend_type} backend does not expose body inertia"
                )
            inertia = np.broadcast_to(
                self._base_body_inertia, (num_envs, *self._base_body_inertia.shape)
            ).copy()
            low, high = domain_rand.body_inertia_multiplier_range
            inertia[:, self._base_body_id, :] = self._base_body_inertia[
                self._base_body_id
            ] * np.random.uniform(low, high, size=(num_envs, 3))
            payload.body_inertia = inertia

        return None if payload.is_empty() else payload

    def build_reset_randomization(self, env_ids: np.ndarray) -> ResetRandomizationPayload | None:
        payload = self._startup_reset_randomization
        if payload is None or payload.is_empty():
            return None
        rows = np.asarray(env_ids, dtype=np.intp)
        sliced = ResetRandomizationPayload(
            base_mass_delta=self._slice_reset_field(payload.base_mass_delta, rows),
            base_com_offset=self._slice_reset_field(payload.base_com_offset, rows),
            gravity=self._slice_reset_field(payload.gravity, rows),
            body_iquat=self._slice_reset_field(payload.body_iquat, rows),
            body_inertia=self._slice_reset_field(payload.body_inertia, rows),
            body_ipos=self._slice_reset_field(payload.body_ipos, rows),
            body_mass=self._slice_reset_field(payload.body_mass, rows),
            dof_armature=self._slice_reset_field(payload.dof_armature, rows),
            geom_friction=self._slice_reset_field(payload.geom_friction, rows),
            kp=None,
            kd=None,
        )
        return None if sliced.is_empty() else sliced

    def _slice_reset_field(self, value: np.ndarray | None, rows: np.ndarray) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value[rows], dtype=np.float64).copy()

    def set_reset_runtime(
        self,
        env_ids: np.ndarray,
        leg_kp: np.ndarray,
        leg_kd: np.ndarray,
        default_policy_leg_pos: np.ndarray,
    ) -> None:
        self._leg_kp[env_ids] = np.asarray(leg_kp, dtype=np.float64)
        self._leg_kd[env_ids] = np.asarray(leg_kd, dtype=np.float64)
        self._default_policy_leg_pos[env_ids] = np.asarray(default_policy_leg_pos, dtype=np.float64)
        self._action_delay_fifo[:, env_ids, :] = 0.0
        delay_cfg = self._cfg.control_config
        if delay_cfg.action_delay_enabled and delay_cfg.randomize_action_delay:
            self._delay_steps[env_ids] = np.random.randint(
                self._min_delay_steps, self._max_delay_steps + 1, size=(len(env_ids),)
            )
        else:
            self._delay_steps[env_ids] = (
                self._nominal_delay_steps if delay_cfg.action_delay_enabled else 0
            )
        self._delay_steps_intp[env_ids] = self._delay_steps[env_ids]
        self._last_motor_ctrl[env_ids] = 0.0
        self._policy_leg_torque[env_ids] = 0.0
        self._policy_leg_vel[env_ids] = 0.0
        self._policy_leg_pos[env_ids] = default_policy_leg_pos
        self._policy_leg_acc[env_ids] = 0.0
        self._last_policy_leg_vel[env_ids] = 0.0
        self._bad_orientation_steps[env_ids] = 0

    def policy_default_to_native_qpos(self, policy_leg_pos: np.ndarray) -> np.ndarray:
        output = policy_to_output_pos_np(policy_leg_pos)
        native = np.zeros((output.shape[0], NUM_ACTIONS), dtype=np.float64)
        native[:, OUTPUT_LEG_INDICES] = output
        return native

    def align_reset_qpos_to_wheel_clearance(
        self, env_ids: np.ndarray, qpos: np.ndarray, qvel: np.ndarray
    ) -> None:
        if len(env_ids) == 0:
            return
        wheel_z = self._probe_reset_wheel_z(env_ids, qpos, qvel)
        if wheel_z is None:
            return
        ground_z = qpos[:, 2] - DEFAULT_BASE_HEIGHT
        wheel_bottom = wheel_z - ground_z[:, None] - FOURBAR_WHEEL_RADIUS
        min_wheel_bottom = np.min(wheel_bottom, axis=1)
        lift = np.clip(RESET_WHEEL_CLEARANCE - min_wheel_bottom, 0.0, None)
        qpos[:, 2] += lift

    def _probe_reset_wheel_z(
        self, env_ids: np.ndarray, qpos: np.ndarray, qvel: np.ndarray
    ) -> np.ndarray | None:
        backend = self._backend
        pool = getattr(backend, "_pool", None)
        physics_state = getattr(backend, "_physics_state", None)
        sensor_indices = getattr(backend, "_sensor_indices", {})
        if pool is None or physics_state is None:
            return None
        idx_qpos = getattr(backend, "_idx_qpos", None)
        idx_qvel = getattr(backend, "_idx_qvel", None)
        nq = getattr(backend, "nq", None)
        nv = getattr(backend, "nv", None)
        if None in (idx_qpos, idx_qvel, nq, nv):
            return None

        wheel_sensor_cols: list[int] = []
        for body_name in WHEEL_BODY_NAMES:
            indices = sensor_indices.get(f"track_pos_w_{body_name}")
            if indices is None or len(indices) < 3:
                return None
            wheel_sensor_cols.append(int(indices[2]))

        rows = np.asarray(env_ids, dtype=np.intp)
        probe_state = np.asarray(physics_state, dtype=np.float64).copy()
        probe_state[rows, int(idx_qpos) : int(idx_qpos) + int(nq)] = qpos
        probe_state[rows, int(idx_qvel) : int(idx_qvel) + int(nv)] = qvel
        sensor_data = pool.forward(probe_state)
        return np.asarray(sensor_data[rows[:, None], wheel_sensor_cols], dtype=np.float64)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        action = np.asarray(actions, dtype=self._np_dtype)
        clip_actions = self._cfg.control_config.clip_actions
        if clip_actions is not None:
            action = np.asarray(
                np.clip(action, -float(clip_actions), float(clip_actions)), dtype=self._np_dtype
            )
        previous_actions = state.info.get("current_actions")
        if previous_actions is None:
            previous_actions = np.zeros_like(action)
        state.info["last_actions"] = previous_actions
        state.info["current_actions"] = action
        return action

    def _pre_step_motor_control(self, backend: Any, policy_ctrl: np.ndarray) -> np.ndarray:
        delayed = self._select_delayed_actions(policy_ctrl)
        native_pos = self.get_dof_pos()
        native_vel = self.get_dof_vel()
        output_pos = native_pos[:, OUTPUT_LEG_INDICES]
        output_vel = native_vel[:, OUTPUT_LEG_INDICES]
        policy_pos, policy_vel, left_j, right_j = output_to_policy_pos_vel_jacobian_np(
            output_pos, output_vel
        )
        policy_pos = policy_pos.astype(self._np_dtype)
        policy_vel = policy_vel.astype(self._np_dtype)
        target = (
            delayed[:, :NUM_POLICY_LEG_ACTIONS] * LEG_ACTION_SCALE + self._default_policy_leg_pos
        )
        target = self._clamp_active_rod_angles(target)
        policy_torque = self._leg_kp * (target - policy_pos) - self._leg_kd * policy_vel
        policy_torque = self._clip_active_motor_torque(policy_torque, policy_vel)
        output_torque = policy_to_output_torque_from_jacobian_np(
            policy_torque, left_j, right_j
        ).astype(self._np_dtype)

        wheel_vel = native_vel[:, WHEEL_INDICES]
        wheel_target_vel = delayed[:, NUM_POLICY_LEG_ACTIONS:] * WHEEL_ACTION_SCALE
        wheel_torque = self._cfg.control_config.wheel_kd * (wheel_target_vel - wheel_vel)

        motor_ctrl = self._last_motor_ctrl
        motor_ctrl[:, OUTPUT_LEG_INDICES] = output_torque
        motor_ctrl[:, WHEEL_INDICES] = wheel_torque
        np.clip(motor_ctrl, self._ctrl_lower, self._ctrl_upper, out=motor_ctrl)

        self._policy_leg_pos[:] = policy_pos
        self._policy_leg_vel[:] = policy_vel
        self._policy_leg_torque[:] = policy_torque
        return motor_ctrl

    def _select_delayed_actions(self, policy_ctrl: np.ndarray) -> np.ndarray:
        head = self._action_delay_head
        self._action_delay_fifo[head] = np.asarray(policy_ctrl, dtype=self._np_dtype)
        self._action_delay_head = (head - 1) % self._action_delay_fifo.shape[0]
        if not self._cfg.control_config.action_delay_enabled:
            return self._action_delay_fifo[head]
        np.add(self._delay_steps_intp, head, out=self._delay_slot_indices)
        np.remainder(
            self._delay_slot_indices,
            self._action_delay_fifo.shape[0],
            out=self._delay_slot_indices,
        )
        return self._action_delay_fifo[self._delay_slot_indices, self._env_row_indices]

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        base_pos, base_linvel, base_angvel, projected_gravity = self.base_state()
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        base_contact_force, wheel_contact_forces, leg_contact_forces = self._contact_forces(
            include_base=self._reward_scale_enabled("collision"),
            include_leg=self._reward_scale_enabled("upright_leg_contact"),
        )
        output_leg_pos = dof_pos[:, OUTPUT_LEG_INDICES]
        output_leg_vel = dof_vel[:, OUTPUT_LEG_INDICES]
        policy_leg_pos, policy_leg_vel = output_to_policy_pos_vel_np(output_leg_pos, output_leg_vel)
        policy_leg_pos = policy_leg_pos.astype(get_global_dtype())
        policy_leg_vel = policy_leg_vel.astype(get_global_dtype())
        self._policy_leg_acc[:] = (
            self._policy_leg_vel - self._last_policy_leg_vel
        ) / self._cfg.ctrl_dt
        self._last_policy_leg_vel[:] = self._policy_leg_vel

        state.info["torques"] = self._policy_order_torques()
        state.info["policy_leg_torque"] = self._policy_leg_torque
        state.info["policy_leg_vel"] = self._policy_leg_vel
        state.info["policy_leg_acc"] = self._policy_leg_acc
        state.info["base_contact_force"] = base_contact_force
        state.info["wheel_contact_forces"] = wheel_contact_forces
        state.info["leg_contact_forces"] = leg_contact_forces
        terminated = self._compute_terminated(
            base_pos,
            base_linvel,
            base_angvel,
            projected_gravity,
            dof_pos,
            dof_vel,
            policy_leg_pos,
            policy_leg_vel,
        )
        reward = self._compute_reward(
            state.info,
            base_pos,
            base_linvel,
            base_angvel,
            projected_gravity,
            dof_pos,
            dof_vel,
            base_contact_force,
            wheel_contact_forces,
            leg_contact_forces,
            policy_leg_pos,
            policy_leg_vel,
        )
        obs = self.compute_obs_from_arrays(
            state.info,
            base_pos,
            base_linvel,
            base_angvel,
            projected_gravity,
            dof_pos,
            dof_vel,
            policy_leg_pos=policy_leg_pos,
            policy_leg_vel=policy_leg_vel,
        )
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def base_state(
        self, env_ids: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows = slice(None) if env_ids is None else np.asarray(env_ids, dtype=np.intp)
        base_pos = np.asarray(self._backend.get_base_pos()[rows], dtype=get_global_dtype())
        base_quat = np.asarray(self._backend.get_base_quat()[rows], dtype=get_global_dtype())
        world_linvel = np.asarray(self._backend.get_base_lin_vel()[rows], dtype=get_global_dtype())
        world_angvel = np.asarray(self._backend.get_base_ang_vel()[rows], dtype=get_global_dtype())
        gravity_w = np.broadcast_to(
            np.asarray([0.0, 0.0, -1.0], dtype=get_global_dtype()), world_linvel.shape
        )
        base_linvel = np.asarray(
            np_quat_apply_inverse(base_quat, world_linvel), dtype=get_global_dtype()
        )
        base_angvel = np.asarray(
            np_quat_apply_inverse(base_quat, world_angvel), dtype=get_global_dtype()
        )
        projected_gravity = np.asarray(
            np_quat_apply_inverse(base_quat, gravity_w), dtype=get_global_dtype()
        )
        return base_pos, base_linvel, base_angvel, projected_gravity

    def compute_obs_from_arrays(
        self,
        info: dict[str, Any],
        base_pos: np.ndarray,
        base_linvel: np.ndarray,
        base_angvel: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        *,
        env_ids: np.ndarray | None = None,
        policy_leg_pos: np.ndarray | None = None,
        policy_leg_vel: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        num_obs = base_pos.shape[0]
        if policy_leg_pos is None or policy_leg_vel is None:
            computed_policy_pos, computed_policy_vel = output_to_policy_pos_vel_np(
                dof_pos[:, OUTPUT_LEG_INDICES], dof_vel[:, OUTPUT_LEG_INDICES]
            )
            if policy_leg_pos is None:
                policy_leg_pos = computed_policy_pos.astype(get_global_dtype())
            if policy_leg_vel is None:
                policy_leg_vel = computed_policy_vel.astype(get_global_dtype())
        default_leg_pos = (
            self._default_policy_leg_pos
            if env_ids is None
            else self._default_policy_leg_pos[np.asarray(env_ids, dtype=np.intp)]
        )
        leg_pos_rel = policy_leg_pos - default_leg_pos
        wheel_pos = dof_pos[:, WHEEL_INDICES]
        wheel_vel = dof_vel[:, WHEEL_INDICES]
        commands = np.asarray(info["commands"], dtype=get_global_dtype())
        current_actions = info.get("current_actions")
        if current_actions is None:
            current_actions = np.zeros((num_obs, NUM_ACTIONS), dtype=get_global_dtype())
        else:
            current_actions = np.asarray(current_actions, dtype=get_global_dtype())
        wheel_contact_forces = info.get("wheel_contact_forces")
        if wheel_contact_forces is None:
            wheel_contact_forces = np.zeros((num_obs, NUM_WHEEL_ACTIONS), dtype=get_global_dtype())
        else:
            wheel_contact_forces = np.asarray(wheel_contact_forces, dtype=get_global_dtype())
        actor, critic = self._obs_output_arrays(num_obs, env_ids)
        self._fill_obs_arrays(
            actor,
            critic,
            base_pos,
            base_linvel,
            base_angvel,
            projected_gravity,
            leg_pos_rel,
            policy_leg_vel,
            wheel_pos,
            wheel_vel,
            commands,
            current_actions,
            wheel_contact_forces,
        )
        return {"obs": actor, "critic": critic}

    def _obs_output_arrays(
        self, num_obs: int, env_ids: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if env_ids is None and num_obs == self._num_envs:
            return self._actor_obs_buf, self._critic_obs_buf
        return (
            np.empty((num_obs, ACTOR_OBS_DIM), dtype=get_global_dtype()),
            np.empty((num_obs, CRITIC_OBS_DIM), dtype=get_global_dtype()),
        )

    def _fill_obs_arrays(
        self,
        actor: np.ndarray,
        critic: np.ndarray,
        base_pos: np.ndarray,
        base_linvel: np.ndarray,
        base_angvel: np.ndarray,
        projected_gravity: np.ndarray,
        leg_pos_rel: np.ndarray,
        policy_leg_vel: np.ndarray,
        wheel_pos: np.ndarray,
        wheel_vel: np.ndarray,
        commands: np.ndarray,
        current_actions: np.ndarray,
        wheel_contact_forces: np.ndarray,
    ) -> None:
        noise_cfg = self._cfg.noise_config
        actor[:, 0:3] = self._obs_noise(base_angvel * 0.25, noise_cfg.scale_gyro)
        actor[:, 3:6] = self._obs_noise(projected_gravity, noise_cfg.scale_gravity)
        np.multiply(commands, COMMAND_SCALE, out=actor[:, 6:11])
        actor[:, 11:15] = self._obs_noise(leg_pos_rel, noise_cfg.scale_joint_angle)
        actor[:, 15:19] = self._obs_noise(policy_leg_vel * 0.25, noise_cfg.scale_joint_vel)
        actor[:, 19:21] = wheel_pos
        np.multiply(wheel_vel, 0.05, out=actor[:, 21:23])
        actor[:, 23:29] = current_actions
        actor[:, 29:32] = 0.0

        np.multiply(base_angvel, 0.25, out=critic[:, 0:3])
        critic[:, 3:6] = projected_gravity
        critic[:, 6:11] = actor[:, 6:11]
        critic[:, 11:15] = leg_pos_rel
        np.multiply(policy_leg_vel, 0.25, out=critic[:, 15:19])
        critic[:, 19:21] = wheel_pos
        critic[:, 21:23] = actor[:, 21:23]
        critic[:, 23:29] = current_actions
        critic[:, 29:32] = 0.0
        critic[:, 32:35] = base_linvel
        critic[:, 35:37] = wheel_contact_forces
        critic[:, 37:38] = base_pos[:, 2:3]

    def _update_commands(self, info: dict[str, Any]) -> None:
        commands = info.get("commands")
        if commands is None:
            return
        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        interval_steps = max(int(round(self._cfg.commands.resampling_time / self._cfg.ctrl_dt)), 1)
        steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
        resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
        if np.any(resample_mask):
            commands_arr[resample_mask] = self.sample_commands(int(np.count_nonzero(resample_mask)))
        info["commands"] = commands_arr

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "tracking_orientation_l2": self._reward_tracking_orientation_l2,
            "tracking_height": self._reward_tracking_height,
            "bad_tilt": self._reward_bad_tilt,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "angular_momentum": self._reward_angular_momentum,
            "leg_torques": self._reward_leg_torques,
            "wheel_torques": self._reward_wheel_torques,
            "stand_still": self._reward_stand_still,
            "leg_dof_acc": self._reward_leg_dof_acc,
            "leg_power": self._reward_leg_power,
            "action_rate": self._reward_action_rate,
            "joint_mirror": self._reward_joint_mirror,
            "dof_pos_limits": self._reward_dof_pos_limits,
            "collision": self._reward_collision,
            "contact_forces": self._reward_contact_forces,
            "upright_wheel_contact": self._reward_upright_wheel_contact,
            "upright_leg_contact": self._reward_upright_leg_contact,
            "is_alive": self._reward_is_alive,
        }

    def _reward_scale_enabled(self, name: str) -> bool:
        return float(self._reward_cfg.scales.get(name, 0.0)) != 0.0

    def _compute_reward(
        self,
        info: dict[str, Any],
        base_pos: np.ndarray,
        base_linvel: np.ndarray,
        base_angvel: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        base_contact_force: np.ndarray,
        wheel_contact_forces: np.ndarray,
        leg_contact_forces: np.ndarray,
        policy_leg_pos: np.ndarray,
        policy_leg_vel: np.ndarray,
    ) -> np.ndarray:
        data = {
            "info": info,
            "base_pos": base_pos,
            "base_linvel": base_linvel,
            "base_angvel": base_angvel,
            "projected_gravity": projected_gravity,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "base_contact_force": base_contact_force,
            "upright_factor": self._upright_factor(projected_gravity),
            "wheel_contact_forces": wheel_contact_forces,
            "leg_contact_forces": leg_contact_forces,
            "policy_leg_pos": policy_leg_pos,
            "policy_leg_vel": policy_leg_vel,
        }
        reward = np.zeros((base_pos.shape[0],), dtype=get_global_dtype())
        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and int(step_count[0]) % 4 == 0
        log = {} if should_log else info.get("log", {})
        for name, scale in self._reward_cfg.scales.items():
            if scale == 0.0 or name not in self._reward_fns:
                continue
            term = self._reward_fns[name](data)
            weighted = term * float(scale)
            reward += weighted
            if should_log:
                log[f"reward/{name}"] = float(np.mean(weighted))
        info["log"] = log
        if self._reward_cfg.only_positive_rewards:
            np.maximum(reward, 0.0, out=reward)
        return reward * self._cfg.ctrl_dt

    def _compute_terminated(
        self,
        base_pos: np.ndarray,
        base_linvel: np.ndarray,
        base_angvel: np.ndarray,
        projected_gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        policy_leg_pos: np.ndarray,
        policy_leg_vel: np.ndarray,
    ) -> np.ndarray:
        finite = (
            np.isfinite(base_pos).all(axis=1)
            & np.isfinite(base_linvel).all(axis=1)
            & np.isfinite(base_angvel).all(axis=1)
            & np.isfinite(projected_gravity).all(axis=1)
            & np.isfinite(dof_pos).all(axis=1)
            & np.isfinite(dof_vel).all(axis=1)
        )
        leg_pos_bad = np.any(np.abs(policy_leg_pos - self._default_policy_leg_pos) > 3.0, axis=1)
        leg_vel_bad = np.any(np.abs(policy_leg_vel) > 120.0, axis=1)
        root_lin_bad = np.sum(np.square(base_linvel), axis=1) > 80.0 * 80.0
        root_ang_bad = np.sum(np.square(base_angvel), axis=1) > 500.0 * 500.0
        height_bad = (base_pos[:, 2] < -0.5) | (base_pos[:, 2] > 3.0)
        tilt = np.arccos(np.clip(-projected_gravity[:, 2], -1.0, 1.0))
        bad_orientation = tilt > 0.5236
        self._bad_orientation_steps[bad_orientation] += 1
        self._bad_orientation_steps[~bad_orientation] = 0
        delayed_bad_orientation = self._bad_orientation_steps > 100
        return (
            ~finite
            | leg_pos_bad
            | leg_vel_bad
            | root_lin_bad
            | root_ang_bad
            | height_bad
            | delayed_bad_orientation
        )

    def _upright_factor(self, projected_gravity: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.clip(-projected_gravity[:, 2], 0.0, 0.7) / 0.7, dtype=get_global_dtype()
        )

    def _reward_upright_factor(self, data: dict[str, Any]) -> np.ndarray:
        gate = data.get("upright_factor")
        if gate is not None:
            return cast(np.ndarray, gate)
        return self._upright_factor(data["projected_gravity"])

    def _reward_tracking_lin_vel(self, data: dict[str, Any]) -> np.ndarray:
        commands = data["info"]["commands"]
        linvel = data["base_linvel"]
        error_x = linvel[:, 0] - commands[:, 0]
        sigma = np.where(
            np.abs(commands[:, 0]) < 0.2,
            self._reward_cfg.tracking_lin_vel_sigma_stand,
            self._reward_cfg.tracking_lin_vel_sigma_move,
        )
        reward = np.exp(
            -(
                error_x * error_x
                + self._reward_cfg.tracking_lin_vel_vz_weight * linvel[:, 2] * linvel[:, 2]
            )
            / sigma
        )
        return np.asarray(reward * self._reward_upright_factor(data), dtype=get_global_dtype())

    def _reward_tracking_ang_vel(self, data: dict[str, Any]) -> np.ndarray:
        commands = data["info"]["commands"]
        error = data["base_angvel"][:, 2] - commands[:, 1]
        reward = np.exp(-(error * error) / self._reward_cfg.tracking_ang_vel_sigma)
        return np.asarray(reward * self._reward_upright_factor(data), dtype=get_global_dtype())

    def _reward_tracking_orientation_l2(self, data: dict[str, Any]) -> np.ndarray:
        pg = data["projected_gravity"]
        commands = data["info"]["commands"]
        current_pitch = np.arcsin(np.clip(pg[:, 0], -1.0, 1.0))
        current_roll = np.arcsin(np.clip(-pg[:, 1], -1.0, 1.0))
        return np.asarray(
            np.square(current_pitch - commands[:, 2]) + np.square(current_roll - commands[:, 3]),
            dtype=get_global_dtype(),
        )

    def _reward_tracking_height(self, data: dict[str, Any]) -> np.ndarray:
        target = data["info"]["commands"][:, 4]
        error = np.square(data["base_pos"][:, 2] - target)
        return np.asarray(
            np.exp(-error / self._reward_cfg.tracking_height_sigma), dtype=get_global_dtype()
        )

    def _reward_bad_tilt(self, data: dict[str, Any]) -> np.ndarray:
        pg = data["projected_gravity"]
        tilt = np.arccos(np.clip(-pg[:, 2], -1.0, 1.0))
        soft = math.radians(self._reward_cfg.bad_tilt_soft_limit_deg)
        hard = math.radians(self._reward_cfg.bad_tilt_hard_limit_deg)
        span = max(hard - soft, 1.0e-6)
        excess = np.clip((tilt - soft) / span, 0.0, None)
        return np.asarray(
            np.clip(excess * excess, None, self._reward_cfg.bad_tilt_max_penalty),
            dtype=get_global_dtype(),
        )

    def _reward_ang_vel_xy(self, data: dict[str, Any]) -> np.ndarray:
        gate = self._reward_upright_factor(data)
        gyro = data["base_angvel"]
        return np.asarray(np.sum(np.square(gyro[:, :2]), axis=1) * gate, dtype=get_global_dtype())

    def _reward_angular_momentum(self, data: dict[str, Any]) -> np.ndarray:
        gate = self._reward_upright_factor(data)
        return np.asarray(
            self._robot_angular_momentum_sq(data["base_angvel"]) * gate, dtype=get_global_dtype()
        )

    def _robot_angular_momentum_sq(self, fallback_angvel: np.ndarray) -> np.ndarray:
        try:
            angmom = np.asarray(
                self._backend.get_sensor_data("robot_subtree_angmom"), dtype=np.float64
            ).reshape(self._num_envs, 3)
            return np.sum(np.square(angmom), axis=1)
        except Exception:
            pass

        body_ids = getattr(self, "_robot_body_ids", None)
        body_mass = getattr(self, "_robot_body_mass", None)
        body_inertia = getattr(self, "_robot_body_inertia", None)
        if body_ids is None or body_mass is None or body_inertia is None:
            return np.sum(np.square(fallback_angvel), axis=1)
        try:
            pos_w, quat_w, lin_vel_w, ang_vel_w = self._backend.get_body_state_w(body_ids)
        except Exception:
            return np.sum(np.square(fallback_angvel), axis=1)

        mass = np.asarray(body_mass, dtype=np.float64)
        total_mass = max(float(np.sum(mass)), 1.0e-6)
        com_w = np.sum(pos_w * mass[None, :, None], axis=1) / total_mass
        orbital = np.sum(
            np.cross(pos_w - com_w[:, None, :], lin_vel_w * mass[None, :, None]), axis=1
        )

        rot = np_matrix_from_quat(quat_w.reshape(-1, 4)).reshape(*quat_w.shape[:2], 3, 3)
        local_ang_vel = np.einsum("ebji,ebj->ebi", rot, ang_vel_w)
        local_spin = local_ang_vel * np.asarray(body_inertia, dtype=np.float64)[None, :, :]
        world_spin = np.einsum("ebij,ebj->ebi", rot, local_spin)
        angular_momentum = orbital + np.sum(world_spin, axis=1)
        return np.sum(np.square(angular_momentum), axis=1)

    def _reward_leg_torques(self, data: dict[str, Any]) -> np.ndarray:
        del data
        return np.asarray(
            np.sum(np.square(self._policy_leg_torque), axis=1), dtype=get_global_dtype()
        )

    def _reward_wheel_torques(self, data: dict[str, Any]) -> np.ndarray:
        del data
        wheel_torque = self._last_motor_ctrl[:, WHEEL_INDICES]
        excess = np.clip(
            np.abs(wheel_torque) - self._reward_cfg.wheel_torques_max_torque, 0.0, None
        )
        return np.asarray(np.sum(np.square(excess), axis=1), dtype=get_global_dtype())

    def _reward_stand_still(self, data: dict[str, Any]) -> np.ndarray:
        commands = data["info"]["commands"]
        command_threshold_sq = self._reward_cfg.stand_still_command_threshold**2
        stopped = np.sum(np.square(commands[:, :2]), axis=1) <= command_threshold_sq
        diff = data["policy_leg_pos"] - self._default_policy_leg_pos
        height_scale = np.exp(
            -self._reward_cfg.stand_still_height_tolerance
            * np.square(commands[:, 4] - self._reward_cfg.stand_still_default_height)
        )
        return np.asarray(
            np.sum(np.square(diff), axis=1)
            * stopped.astype(get_global_dtype())
            * height_scale
            * self._reward_upright_factor(data),
            dtype=get_global_dtype(),
        )

    def _reward_leg_dof_acc(self, data: dict[str, Any]) -> np.ndarray:
        penalty = np.sum(np.square(self._policy_leg_acc), axis=1)
        steps = np.asarray(data["info"].get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
        penalty = np.where(steps <= 1, 0.0, penalty)
        return np.asarray(penalty, dtype=get_global_dtype())

    def _reward_leg_power(self, data: dict[str, Any]) -> np.ndarray:
        del data
        return np.asarray(
            np.sum(np.abs(self._policy_leg_torque * self._policy_leg_vel), axis=1),
            dtype=get_global_dtype(),
        )

    def _reward_action_rate(self, data: dict[str, Any]) -> np.ndarray:
        info = data["info"]
        return np.asarray(
            np.sum(np.square(info["current_actions"] - info["last_actions"]), axis=1),
            dtype=get_global_dtype(),
        )

    def _reward_joint_mirror(self, data: dict[str, Any]) -> np.ndarray:
        pos = data["policy_leg_pos"]
        hip_diff = pos[:, 0] + pos[:, 2]
        knee_diff = pos[:, 1] + pos[:, 3]
        gate = self._reward_upright_factor(data)
        return np.asarray(
            (hip_diff * hip_diff + knee_diff * knee_diff) * 0.5 * gate, dtype=get_global_dtype()
        )

    def _reward_dof_pos_limits(self, data: dict[str, Any]) -> np.ndarray:
        pos = data["policy_leg_pos"]
        left = pos[:, 0] - pos[:, 1]
        right = pos[:, 3] - pos[:, 2]
        penalty = (
            np.clip(ACTIVE_LOWER - left, 0.0, None)
            + np.clip(left - ACTIVE_UPPER, 0.0, None)
            + np.clip(ACTIVE_LOWER - right, 0.0, None)
            + np.clip(right - ACTIVE_UPPER, 0.0, None)
        )
        return np.asarray(penalty, dtype=get_global_dtype())

    def _reward_collision(self, data: dict[str, Any]) -> np.ndarray:
        base_contact = data["base_contact_force"]
        gate = self._reward_upright_factor(data)
        return np.asarray(
            (base_contact > self._reward_cfg.collision_threshold).astype(get_global_dtype()) * gate,
            dtype=get_global_dtype(),
        )

    def _reward_contact_forces(self, data: dict[str, Any]) -> np.ndarray:
        force = data["wheel_contact_forces"]
        excess = np.clip(force - self._reward_cfg.contact_forces_threshold, 0.0, None) / 100.0
        return np.asarray(
            np.sum(excess, axis=1) * self._reward_upright_factor(data),
            dtype=get_global_dtype(),
        )

    def _reward_upright_wheel_contact(self, data: dict[str, Any]) -> np.ndarray:
        gate = self._reward_upright_factor(data)
        active = gate >= self._reward_cfg.upright_contact_min_gate
        in_contact = data["wheel_contact_forces"] > self._reward_cfg.upright_contact_force_threshold
        contact_ratio = np.mean(in_contact.astype(get_global_dtype()), axis=1)
        return np.asarray(
            (1.0 - contact_ratio) * gate * active.astype(get_global_dtype()),
            dtype=get_global_dtype(),
        )

    def _reward_upright_leg_contact(self, data: dict[str, Any]) -> np.ndarray:
        gate = self._reward_upright_factor(data)
        active = gate >= self._reward_cfg.upright_contact_min_gate
        leg_contact = data["leg_contact_forces"]
        has_contact = np.any(leg_contact > self._reward_cfg.upright_contact_force_threshold, axis=1)
        return np.asarray(
            has_contact.astype(get_global_dtype()) * gate * active.astype(get_global_dtype()),
            dtype=get_global_dtype(),
        )

    def _reward_is_alive(self, data: dict[str, Any]) -> np.ndarray:
        return np.ones((data["base_pos"].shape[0],), dtype=get_global_dtype())

    def _policy_order_torques(self) -> np.ndarray:
        torques = self._policy_order_torque_buf
        torques[:, :NUM_POLICY_LEG_ACTIONS] = self._policy_leg_torque
        torques[:, NUM_POLICY_LEG_ACTIONS:] = self._last_motor_ctrl[:, WHEEL_INDICES]
        return torques

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
        if flat.shape[1] >= 3:
            force_mag = np.linalg.norm(flat[:, :3], axis=1)
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

    def _clamp_active_rod_angles(self, leg_target: np.ndarray) -> np.ndarray:
        target = np.asarray(leg_target, dtype=np.float64).copy()
        for front_idx, back_idx, front_coef, back_coef in (
            (0, 1, 1.0, -1.0),
            (2, 3, -1.0, 1.0),
        ):
            angle = np.clip(
                front_coef * target[:, front_idx] + back_coef * target[:, back_idx],
                ACTIVE_LOWER,
                ACTIVE_UPPER,
            )
            target[:, back_idx] = (angle - front_coef * target[:, front_idx]) / back_coef
        return target

    def _clip_active_motor_torque(self, torque: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        vel_at_effort_limit = DM8009P_NO_LOAD_SPEED * (
            1.0 + DM8009P_RATED_TORQUE / DM8009P_STALL_TORQUE
        )
        clipped_velocity = np.clip(velocity, -vel_at_effort_limit, vel_at_effort_limit)
        top = DM8009P_STALL_TORQUE * (1.0 - clipped_velocity / DM8009P_NO_LOAD_SPEED)
        bottom = DM8009P_STALL_TORQUE * (-1.0 - clipped_velocity / DM8009P_NO_LOAD_SPEED)
        max_effort = np.minimum(top, DM8009P_RATED_TORQUE)
        min_effort = np.maximum(bottom, -DM8009P_RATED_TORQUE)
        return np.clip(torque, min_effort, max_effort)
