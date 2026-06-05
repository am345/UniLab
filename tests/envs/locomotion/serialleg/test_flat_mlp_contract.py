from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab import cli
from unilab.base import registry
from unilab.base.registry import apply_cfg_overrides
from unilab.dr.types import RESET_TERM_KD, RESET_TERM_KP
from unilab.envs.locomotion.serialleg.flat_mlp import (
    ACTOR_OBS_DIM,
    COMMAND_SCALE,
    CRITIC_OBS_DIM,
    DEFAULT_BASE_HEIGHT,
    DEFAULT_OUTPUT_LEG_POS,
    DEFAULT_POLICY_LEG_POS,
    FOURBAR_WHEEL_RADIUS,
    LEG_ACTION_SCALE,
    NUM_ACTIONS,
    NUM_POLICY_LEG_ACTIONS,
    OUTPUT_LEG_INDICES,
    RESET_WHEEL_CLEARANCE,
    WHEEL_ACTION_SCALE,
    WHEEL_INDICES,
    SerialLegFlatMLPCfg,
    SerialLegFlatMLPEnv,
    SerialLegMujocoBackendConfig,
    SerialLegRewardConfig,
)
from unilab.envs.locomotion.serialleg.fourbar import (
    output_to_policy_pos_np,
    output_to_policy_pos_vel_jacobian_np,
    output_to_policy_pos_vel_np,
    output_to_policy_vel_np,
    policy_to_output_torque_from_jacobian_np,
    policy_to_output_torque_np,
)
from unilab.training.backend_adapter import BackendAdapter

ROOT_DIR = Path(__file__).resolve().parents[4]
CONF_DIR = ROOT_DIR / "conf"
XML_PATH = (
    ROOT_DIR
    / "src"
    / "unilab"
    / "assets"
    / "robots"
    / "serialleg"
    / "serialleg_fourbar_surrogate_train.xml"
)


def _serialleg_env_stub() -> Any:
    env = cast(Any, object.__new__(SerialLegFlatMLPEnv))
    env._cfg = SerialLegFlatMLPCfg(reward_config=SerialLegRewardConfig())
    env._num_envs = 2
    env._np_dtype = np.float32
    env._default_policy_leg_pos = np.broadcast_to(DEFAULT_POLICY_LEG_POS, (2, 4)).copy()
    return env


def test_serialleg_flat_mlp_registers_mujoco_and_motrix_backends() -> None:
    import unilab.envs.locomotion.serialleg  # noqa: F401

    assert registry.contains("SerialLegFlatMLP")
    registered = registry.list_registered_envs()["SerialLegFlatMLP"]
    assert registered["config_class"] == "SerialLegFlatMLPCfg"
    assert set(registered["available_backends"]) == {"mujoco", "motrix"}


def test_serialleg_flat_mlp_config_matches_se3_flat_mlp_contract() -> None:
    cfg = SerialLegFlatMLPCfg(reward_config=SerialLegRewardConfig())

    assert cfg.sim_dt == pytest.approx(0.005)
    assert cfg.ctrl_dt == pytest.approx(0.02)
    assert cfg.init_state.pos[2] == pytest.approx(DEFAULT_BASE_HEIGHT)
    assert cfg.commands.steps_per_policy_iter == 32
    assert cfg.commands.command_vel_schedule == [
        [0.0, 0.0, 0.0],
        [500.0, 0.5, 0.5],
        [1500.0, 1.0, 1.0],
        [2500.0, 1.5, 2.0],
        [3500.0, 2.0, 2.5],
        [4500.0, 2.5, 3.0],
    ]
    assert cfg.control_config.leg_kp == pytest.approx(40.0)
    assert cfg.control_config.leg_kd == pytest.approx(2.0)
    assert cfg.control_config.wheel_kd == pytest.approx(0.5)
    assert cfg.control_config.clip_actions is None
    assert cfg.control_config.min_action_delay_s == pytest.approx(0.004)
    assert cfg.control_config.max_action_delay_s == pytest.approx(0.006)
    assert cfg.domain_rand.randomize_dof_armature is False
    assert cfg.domain_rand.robot_friction_range == [0.2, 1.5]
    np.testing.assert_allclose(LEG_ACTION_SCALE, [0.35, 0.25, 0.35, 0.25])
    assert WHEEL_ACTION_SCALE == pytest.approx(45.0)
    np.testing.assert_allclose(COMMAND_SCALE, [2.0, 0.25, 5.0, 5.0, 5.0])


def test_serialleg_fourbar_default_pose_matches_se3_reference() -> None:
    np.testing.assert_allclose(
        DEFAULT_POLICY_LEG_POS,
        [-0.275422946189, -1.592100148957, 0.275422946189, 1.592100148957],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        DEFAULT_OUTPUT_LEG_POS,
        [-0.275422946189, -1.242259649307, 0.275422946189, 1.242259649307],
        rtol=0.0,
        atol=5.0e-6,
    )
    output_torque = policy_to_output_torque_np(
        DEFAULT_POLICY_LEG_POS.reshape(1, -1),
        np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float64),
    )
    assert output_torque.shape == (1, NUM_POLICY_LEG_ACTIONS)
    assert np.isfinite(output_torque).all()


def test_serialleg_fourbar_fused_helpers_match_reference_helpers() -> None:
    output_pos = np.stack(
        [
            DEFAULT_OUTPUT_LEG_POS,
            DEFAULT_OUTPUT_LEG_POS + np.array([0.05, -0.04, -0.05, 0.04], dtype=np.float64),
        ]
    )
    output_vel = np.array(
        [[0.1, -0.2, 0.3, -0.4], [-1.0, 2.0, -3.0, 4.0]],
        dtype=np.float64,
    )
    policy_torque = np.array(
        [[1.0, -2.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0]],
        dtype=np.float64,
    )

    ref_pos = output_to_policy_pos_np(output_pos)
    ref_vel = output_to_policy_vel_np(output_pos, output_vel)
    fused_pos, fused_vel = output_to_policy_pos_vel_np(output_pos, output_vel)
    jac_pos, jac_vel, left_j, right_j = output_to_policy_pos_vel_jacobian_np(output_pos, output_vel)
    ref_torque = policy_to_output_torque_np(ref_pos, policy_torque)
    fused_torque = policy_to_output_torque_from_jacobian_np(policy_torque, left_j, right_j)

    np.testing.assert_allclose(fused_pos, ref_pos, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(fused_vel, ref_vel, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(jac_pos, ref_pos, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(jac_vel, ref_vel, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(fused_torque, ref_torque, rtol=0.0, atol=1.0e-12)


def test_serialleg_obs_contract_uses_se3_actor_and_critic_layout() -> None:
    env = _serialleg_env_stub()
    base_pos = np.array([[0.0, 0.0, DEFAULT_BASE_HEIGHT], [0.0, 0.0, 0.25]], dtype=np.float32)
    base_linvel = np.array([[0.2, 0.0, 0.01], [0.0, 0.0, -0.02]], dtype=np.float32)
    base_angvel = np.array([[0.0, 0.1, 0.2], [0.3, 0.0, -0.1]], dtype=np.float32)
    projected_gravity = np.array([[0.0, 0.0, -1.0], [0.1, -0.1, -0.98]], dtype=np.float32)
    dof_pos = np.zeros((2, NUM_ACTIONS), dtype=np.float32)
    dof_vel = np.zeros((2, NUM_ACTIONS), dtype=np.float32)
    dof_pos[:, OUTPUT_LEG_INDICES] = DEFAULT_OUTPUT_LEG_POS.astype(np.float32)
    dof_pos[:, WHEEL_INDICES] = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)
    dof_vel[:, WHEEL_INDICES] = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    info = {
        "commands": np.array(
            [[0.5, -1.0, 0.1, -0.1, 0.22], [0.0, 0.0, 0.0, 0.0, 0.30]], dtype=np.float32
        ),
        "current_actions": np.array(
            [[0.1, -0.2, 0.3, -0.4, 0.5, -0.6], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "wheel_contact_forces": np.array([[12.0, 13.0], [0.0, 1.0]], dtype=np.float32),
    }

    obs = env.compute_obs_from_arrays(
        info, base_pos, base_linvel, base_angvel, projected_gravity, dof_pos, dof_vel
    )

    assert set(obs) == {"obs", "critic"}
    assert obs["obs"].shape == (2, ACTOR_OBS_DIM)
    assert obs["critic"].shape == (2, CRITIC_OBS_DIM)
    np.testing.assert_allclose(obs["obs"][:, 6:11], info["commands"] * COMMAND_SCALE)
    np.testing.assert_allclose(obs["obs"][:, 23:29], info["current_actions"])
    np.testing.assert_allclose(obs["obs"][:, 29:32], 0.0)
    np.testing.assert_allclose(obs["critic"][:, 32:35], base_linvel)
    np.testing.assert_allclose(obs["critic"][:, 35:37], info["wheel_contact_forces"])
    np.testing.assert_allclose(obs["critic"][:, 37:38], base_pos[:, 2:3])
    assert env.obs_groups_spec == {"obs": ACTOR_OBS_DIM, "critic": CRITIC_OBS_DIM}


def test_serialleg_identity_dof_order_avoids_numpy_index_copy() -> None:
    env = _serialleg_env_stub()
    dof_pos = np.zeros((2, NUM_ACTIONS), dtype=np.float32)
    dof_vel = np.ones((2, NUM_ACTIONS), dtype=np.float32)
    env._backend = SimpleNamespace(
        get_dof_pos=lambda: dof_pos,
        get_dof_vel=lambda: dof_vel,
    )
    env._dof_pos_indices = None
    env._dof_vel_indices = None

    assert env.get_dof_pos() is dof_pos
    assert env.get_dof_vel() is dof_vel


def test_serialleg_contact_rewards_use_precomputed_contact_arrays() -> None:
    env = _serialleg_env_stub()
    env._reward_cfg = env._cfg.reward_config
    env._backend = SimpleNamespace(
        get_sensor_data=lambda name: pytest.fail(f"unexpected sensor read: {name}")
    )
    data = {
        "projected_gravity": np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32),
        "base_contact_force": np.array([0.2, 0.0], dtype=np.float32),
        "leg_contact_forces": np.array(
            [[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32
        ),
    }

    np.testing.assert_allclose(env._reward_collision(data), [1.0, 0.0])
    np.testing.assert_allclose(env._reward_upright_leg_contact(data), [1.0, 0.0])


def test_serialleg_reset_alignment_lifts_root_to_wheel_clearance() -> None:
    class FakePool:
        def __init__(self, sensor_data: np.ndarray) -> None:
            self.sensor_data = sensor_data

        def forward(self, state: np.ndarray) -> np.ndarray:
            assert state.shape == (2, 26)
            return self.sensor_data

    env = _serialleg_env_stub()
    env._backend = SimpleNamespace(
        _pool=FakePool(
            np.array(
                [
                    [0.0, 0.0, FOURBAR_WHEEL_RADIUS - 0.005, 0.0, 0.0, 0.08],
                    [0.0, 0.0, 0.08, 0.0, 0.0, 0.09],
                ],
                dtype=np.float64,
            )
        ),
        _physics_state=np.zeros((2, 26), dtype=np.float64),
        _sensor_indices={
            "track_pos_w_l_wheel_Link": [0, 1, 2],
            "track_pos_w_r_wheel_Link": [3, 4, 5],
        },
        _idx_qpos=1,
        _idx_qvel=14,
        nq=13,
        nv=12,
    )
    qpos = np.zeros((2, 13), dtype=np.float64)
    qpos[:, 2] = DEFAULT_BASE_HEIGHT
    qvel = np.zeros((2, 12), dtype=np.float64)

    env.align_reset_qpos_to_wheel_clearance(np.array([0, 1], dtype=np.int32), qpos, qvel)

    assert qpos[0, 2] == pytest.approx(DEFAULT_BASE_HEIGHT + 0.005 + RESET_WHEEL_CLEARANCE)
    assert qpos[1, 2] == pytest.approx(DEFAULT_BASE_HEIGHT)


def test_serialleg_leg_dof_acc_ignores_first_two_episode_steps() -> None:
    env = _serialleg_env_stub()
    env._policy_leg_acc = np.full((2, NUM_POLICY_LEG_ACTIONS), 3.0, dtype=np.float32)
    data = {"info": {"steps": np.array([1, 2], dtype=np.uint32)}}

    penalty = env._reward_leg_dof_acc(data)

    np.testing.assert_allclose(penalty, [0.0, 36.0])


def test_serialleg_push_schedule_uses_se3_velocity_disturbance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    randint_calls: list[tuple[int, int]] = []

    def fake_randint(low: int, high: int) -> int:
        randint_calls.append((low, high))
        return low

    def fake_uniform(low: float, high: float, size: tuple[int, int]) -> np.ndarray:
        assert (low, high, size) == (-1.0, 1.0, (2, 3))
        return np.ones(size, dtype=np.float64)

    monkeypatch.setattr(np.random, "randint", fake_randint)
    monkeypatch.setattr(np.random, "uniform", fake_uniform)
    env = _serialleg_env_stub()
    env._backend = SimpleNamespace(_base_lin_vel_view=np.zeros((2, 3), dtype=np.float64))
    env._next_push_step = env._sample_push_interval_steps()

    assert randint_calls == [(250, 301)]
    env.step_counter = 32 * 2000
    env.update_push_curriculum()
    env._next_push_step = env.step_counter
    env.apply_velocity_push_if_due(env.step_counter)

    np.testing.assert_allclose(env._backend._base_lin_vel_view, [[0.3, 0.3, 0.0], [0.3, 0.3, 0.0]])
    assert randint_calls == [(250, 301), (250, 301)]
    assert env._next_push_step == env.step_counter + 250


def test_serialleg_angular_momentum_uses_robot_body_state_when_available() -> None:
    class FakeBackend:
        def get_body_state_w(self, body_ids: np.ndarray):
            assert body_ids.tolist() == [0, 1]
            pos = np.array(
                [
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                ],
                dtype=np.float64,
            )
            quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 2, 1))
            lin_vel = np.array(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ],
                dtype=np.float64,
            )
            ang_vel = np.zeros((2, 2, 3), dtype=np.float64)
            return pos, quat, lin_vel, ang_vel

    env = _serialleg_env_stub()
    env._backend = FakeBackend()
    env._robot_body_ids = np.array([0, 1], dtype=np.int32)
    env._robot_body_mass = np.array([1.0, 1.0], dtype=np.float64)
    env._robot_body_inertia = np.zeros((2, 3), dtype=np.float64)

    momentum_sq = env._robot_angular_momentum_sq(np.full((2, 3), 9.0, dtype=np.float64))

    np.testing.assert_allclose(momentum_sq, [0.25, 0.0])


def test_serialleg_angular_momentum_prefers_mujoco_subtree_sensor() -> None:
    class FakeBackend:
        def get_sensor_data(self, name: str) -> np.ndarray:
            assert name == "robot_subtree_angmom"
            return np.array([[1.0, 2.0, 2.0], [0.0, 3.0, 4.0]], dtype=np.float64)

    env = _serialleg_env_stub()
    env._backend = FakeBackend()

    momentum_sq = env._robot_angular_momentum_sq(np.full((2, 3), 9.0, dtype=np.float64))

    np.testing.assert_allclose(momentum_sq, [9.0, 25.0])


def test_serialleg_contact_sensor_reader_uses_netforce_magnitude() -> None:
    class FakeBackend:
        def get_sensor_data(self, name: str) -> np.ndarray:
            assert name == "l_wheel_contact"
            return np.array(
                [[3.0, 4.0, 0.0], [np.nan, 0.0, 0.0], [np.inf, 0.0, 0.0]],
                dtype=np.float64,
            )

    env = _serialleg_env_stub()
    env._num_envs = 3
    env._backend = FakeBackend()

    np.testing.assert_allclose(env._sensor_scalar("l_wheel_contact"), [5.0, 5000.0, 5000.0])


def test_serialleg_sample_commands_matches_se3_standing_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_uniform(low: float, high: float, size: tuple[int, ...]) -> np.ndarray:
        del low, high
        return np.full(size, 0.25, dtype=np.float64)

    monkeypatch.setattr(np.random, "uniform", fake_uniform)
    env = _serialleg_env_stub()
    env.step_counter = 32 * 5000
    env._cfg.commands.rel_standing_envs = 0.5
    env._cfg.commands.vx_deadband = 0.0
    env._cfg.commands.yaw_deadband = 0.0

    commands = env.sample_commands(5)

    np.testing.assert_allclose(commands[:2, :4], 0.0)
    np.testing.assert_allclose(commands[:2, 4], 0.25)
    np.testing.assert_allclose(commands[2:, :4], 0.25)


def test_serialleg_default_action_path_keeps_se3_unclipped_actions() -> None:
    env = _serialleg_env_stub()
    state = SimpleNamespace(
        info={
            "current_actions": np.ones((2, NUM_ACTIONS), dtype=np.float32),
        }
    )
    actions = np.array(
        [[2.5, -2.0, 0.5, -0.5, 1.25, -1.25], [-1.5, 1.5, 0.0, 0.0, 2.0, -2.0]],
        dtype=np.float32,
    )

    out = env.apply_action(actions, cast(Any, state))

    np.testing.assert_allclose(out, actions)
    np.testing.assert_allclose(state.info["current_actions"], actions)
    np.testing.assert_allclose(state.info["last_actions"], np.ones((2, NUM_ACTIONS)))

    env._cfg.control_config.clip_actions = 1.0
    clipped = env.apply_action(actions, cast(Any, state))
    np.testing.assert_allclose(clipped, np.clip(actions, -1.0, 1.0))


def test_serialleg_hydra_owner_config_feeds_env_override() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=serialleg_flat_mlp/mujoco"])

    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR)
    env_cfg_override = adapter.build_task_env_cfg_override()
    env_cfg = SerialLegFlatMLPCfg()
    apply_cfg_overrides(env_cfg, env_cfg_override)

    assert cfg.training.task_name == "SerialLegFlatMLP"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.num_steps_per_env == 32
    assert cfg.algo.policy.actor_hidden_dims == [512, 256, 128]
    assert cfg.algo.algorithm.learning_rate == pytest.approx(6.5e-4)
    assert isinstance(env_cfg.reward_config, SerialLegRewardConfig)
    assert env_cfg.reward_config.scales["tracking_lin_vel"] == pytest.approx(4.0)
    assert env_cfg.reward_config.scales["upright_leg_contact"] == pytest.approx(-25.0)
    assert env_cfg.control_config.action_delay_enabled is True


def test_serialleg_appo_owner_config_feeds_env_override() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "appo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=serialleg_flat_mlp/mujoco"])

    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="appo")
    env_cfg_override = adapter.build_task_env_cfg_override()
    env_cfg = SerialLegFlatMLPCfg()
    apply_cfg_overrides(env_cfg, env_cfg_override)

    assert cfg.training.task_name == "SerialLegFlatMLP"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.num_envs == 4096
    assert cfg.algo.num_workers == 4
    assert cfg.algo.rollouts_per_update == 4
    assert cfg.algo.min_rollouts_for_update is None
    assert cfg.algo.steps_per_env == 16
    assert cfg.algo.actor.hidden_dims == [512, 256, 128]
    assert cfg.algo.actor.distribution_cfg.init_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(6.5e-4)
    assert isinstance(env_cfg.reward_config, SerialLegRewardConfig)
    assert isinstance(env_cfg.mujoco_backend, SerialLegMujocoBackendConfig)
    assert env_cfg.mujoco_backend.nthread == 6
    assert env_cfg.reward_config.scales["contact_forces"] == pytest.approx(-1.07e-3)
    assert env_cfg.control_config.min_action_delay_s == pytest.approx(0.004)


def test_serialleg_appo_motrix_owner_is_training_mainline() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "appo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=serialleg_flat_mlp/motrix"])

    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="appo")
    env_cfg_override = adapter.build_task_env_cfg_override()
    env_cfg = SerialLegFlatMLPCfg()
    apply_cfg_overrides(env_cfg, env_cfg_override)

    assert cfg.training.task_name == "SerialLegFlatMLP"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_workers == 2
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.steps_per_env == 32
    assert env_cfg.motrix_max_iterations == 3
    assert env_cfg.domain_rand.randomize_ground_friction is False
    assert env_cfg.domain_rand.randomize_body_inertia is False
    assert env_cfg.domain_rand.randomize_dof_armature is False
    assert env_cfg.domain_rand.push_robots is False
    assert isinstance(env_cfg.reward_config, SerialLegRewardConfig)
    assert env_cfg.reward_config.scales["tracking_lin_vel"] == pytest.approx(4.0)


def test_serialleg_ppo_motrix_owner_is_available_for_sync_baseline() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=serialleg_flat_mlp/motrix"])

    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR)
    env_cfg_override = adapter.build_task_env_cfg_override()
    env_cfg = SerialLegFlatMLPCfg()
    apply_cfg_overrides(env_cfg, env_cfg_override)

    assert cfg.training.task_name == "SerialLegFlatMLP"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 2048
    assert cfg.algo.num_steps_per_env == 32
    assert env_cfg.domain_rand.randomize_ground_friction is False
    assert env_cfg.domain_rand.randomize_body_inertia is False
    assert env_cfg.domain_rand.randomize_dof_armature is False
    assert env_cfg.domain_rand.push_robots is False


def test_serialleg_robot_friction_ids_keep_only_contact_geoms() -> None:
    class FakeBackend:
        def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.array([0, 1, 0, 2, 0], dtype=np.int32),
                np.array([0, 0, 1, 0, 0], dtype=np.int32),
            )

    env = _serialleg_env_stub()
    env._backend = FakeBackend()
    env._robot_geom_ids = np.array([1, 2, 3, 4], dtype=np.int32)

    np.testing.assert_array_equal(env._resolve_robot_friction_geom_ids(), [1, 2, 3])


def test_serialleg_friction_randomization_only_changes_contact_geoms() -> None:
    env = cast(Any, object.__new__(SerialLegFlatMLPEnv))
    env._cfg = SerialLegFlatMLPCfg(reward_config=SerialLegRewardConfig())
    env._base_body_mass = np.ones(1, dtype=np.float64)
    env._base_geom_friction = np.full((5, 3), [0.8, 0.005, 0.0001], dtype=np.float64)
    env._base_dof_armature = np.ones(1, dtype=np.float64)
    env._base_body_inertia = np.ones((1, 3), dtype=np.float64)
    env._base_body_id = 0
    env._robot_geom_ids = np.array([1, 2, 3, 4], dtype=np.int32)
    env._robot_friction_geom_ids = np.array([1, 3], dtype=np.int32)
    env._cfg.domain_rand.randomize_base_mass = False
    env._cfg.domain_rand.random_com = False
    env._cfg.domain_rand.robot_friction_range = [1.4, 1.4]
    env._cfg.domain_rand.randomize_dof_armature = False
    env._cfg.domain_rand.randomize_body_inertia = False

    payload = env._sample_startup_reset_randomization(2)

    assert payload is not None
    assert payload.geom_friction is not None
    np.testing.assert_allclose(payload.geom_friction[:, [1, 3], 0], [[1.4, 1.4]] * 2)
    expected_unchanged = np.broadcast_to(env._base_geom_friction[[2, 4]], (2, 2, 3))
    np.testing.assert_allclose(payload.geom_friction[:, [2, 4], :], expected_unchanged)


def test_serialleg_backend_dr_payload_excludes_custom_policy_pd_gains() -> None:
    env = cast(Any, object.__new__(SerialLegFlatMLPEnv))
    env._cfg = SerialLegFlatMLPCfg(reward_config=SerialLegRewardConfig())
    env._num_action = NUM_ACTIONS
    env._base_body_mass = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    env._base_geom_friction = np.array([[0.8, 0.005, 0.0001]], dtype=np.float64)
    env._ground_geom_id = 0
    env._base_dof_armature = np.ones(12, dtype=np.float64)
    env._base_body_inertia = np.ones((3, 3), dtype=np.float64)
    env._base_body_id = 1
    env._robot_geom_ids = np.array([0], dtype=np.int32)
    env._startup_reset_randomization = env._sample_startup_reset_randomization(2)

    payload = env.build_reset_randomization(np.array([0, 1], dtype=np.int32))

    assert payload is not None
    assert RESET_TERM_KP not in payload.requested_terms()
    assert RESET_TERM_KD not in payload.requested_terms()
    assert env._cfg.domain_rand.randomize_kp is True
    assert env._cfg.domain_rand.randomize_kd is True


def test_serialleg_startup_randomization_is_reused_across_resets() -> None:
    env = cast(Any, object.__new__(SerialLegFlatMLPEnv))
    env._cfg = SerialLegFlatMLPCfg(reward_config=SerialLegRewardConfig())
    env._num_envs = 3
    env._num_action = NUM_ACTIONS
    env._base_leg_kp = np.full((NUM_POLICY_LEG_ACTIONS,), 40.0, dtype=np.float64)
    env._base_leg_kd = np.full((NUM_POLICY_LEG_ACTIONS,), 2.0, dtype=np.float64)
    env._leg_kp = np.zeros((3, NUM_POLICY_LEG_ACTIONS), dtype=np.float64)
    env._leg_kd = np.zeros((3, NUM_POLICY_LEG_ACTIONS), dtype=np.float64)
    env._default_policy_leg_pos = np.zeros((3, NUM_POLICY_LEG_ACTIONS), dtype=np.float64)
    env._base_body_mass = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    env._base_geom_friction = np.array(
        [[0.8, 0.005, 0.0001], [0.7, 0.005, 0.0001], [0.6, 0.005, 0.0001]],
        dtype=np.float64,
    )
    env._base_dof_armature = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    env._base_body_inertia = np.ones((3, 3), dtype=np.float64)
    env._base_body_id = 1
    env._robot_geom_ids = np.array([1, 2], dtype=np.int32)
    env._cfg.domain_rand.added_mass_range = [1.0, 1.0]
    env._cfg.domain_rand.com_offset_x = [0.01, 0.01]
    env._cfg.domain_rand.com_offset_y = [0.02, 0.02]
    env._cfg.domain_rand.com_offset_z = [0.03, 0.03]
    env._cfg.domain_rand.robot_friction_range = [1.4, 1.4]
    env._cfg.domain_rand.randomize_dof_armature = True
    env._cfg.domain_rand.dof_armature_multiplier_range = [1.5, 1.5]
    env._cfg.domain_rand.body_inertia_multiplier_range = [1.2, 1.2]
    env._cfg.domain_rand.kp_multiplier_range = [1.1, 1.1]
    env._cfg.domain_rand.kd_multiplier_range = [0.9, 0.9]
    env._cfg.domain_rand.default_dof_pos_offset_range = [0.02, 0.02]

    env._init_startup_randomization()
    env_ids = np.array([0, 2], dtype=np.int32)
    payload_a = env.build_reset_randomization(env_ids)
    payload_b = env.build_reset_randomization(env_ids)
    leg_kp_a, leg_kd_a, default_a = env.sample_reset_motor_params(env_ids)
    leg_kp_b, leg_kd_b, default_b = env.sample_reset_motor_params(env_ids)

    assert payload_a is not None
    assert payload_b is not None
    np.testing.assert_allclose(payload_a.base_mass_delta, payload_b.base_mass_delta)
    np.testing.assert_allclose(payload_a.base_com_offset, payload_b.base_com_offset)
    np.testing.assert_allclose(payload_a.geom_friction, payload_b.geom_friction)
    np.testing.assert_allclose(payload_a.dof_armature, payload_b.dof_armature)
    np.testing.assert_allclose(payload_a.body_inertia, payload_b.body_inertia)
    np.testing.assert_allclose(leg_kp_a, leg_kp_b)
    np.testing.assert_allclose(leg_kd_a, leg_kd_b)
    np.testing.assert_allclose(default_a, default_b)
    np.testing.assert_allclose(leg_kp_a, 44.0)
    np.testing.assert_allclose(leg_kd_a, 1.8)
    np.testing.assert_allclose(default_a, np.tile(DEFAULT_POLICY_LEG_POS + 0.02, (2, 1)))
    np.testing.assert_allclose(payload_a.base_mass_delta, [1.0, 1.0])
    np.testing.assert_allclose(payload_a.base_com_offset, [[0.01, 0.02, 0.03]] * 2)
    np.testing.assert_allclose(payload_a.geom_friction[:, [1, 2], 0], [[1.4, 1.4]] * 2)
    np.testing.assert_allclose(payload_a.dof_armature, [[0.0, 1.5, 3.0]] * 2)
    np.testing.assert_allclose(payload_a.body_inertia[:, 1, :], 1.2)
    np.testing.assert_allclose(payload_a.body_inertia[:, [0, 2], :], 1.0)


def test_serialleg_cli_routes_to_ppo_mujoco_owner_config() -> None:
    command = cli.build_command(
        mode="train",
        algo="ppo",
        task="serialleg_flat_mlp",
        sim="mujoco",
        overrides=["training.no_play=true"],
        root=ROOT_DIR,
    )

    assert command[1:] == [
        str(ROOT_DIR / "scripts" / "train_rsl_rl.py"),
        "task=serialleg_flat_mlp/mujoco",
        "training.no_play=true",
    ]


def test_serialleg_cli_routes_to_ppo_motrix_owner_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: object() if name == "motrixsim" else None)

    command = cli.build_command(
        mode="train",
        algo="ppo",
        task="serialleg_flat_mlp",
        sim="motrix",
        overrides=["training.no_play=true"],
        root=ROOT_DIR,
    )

    assert command[1:] == [
        str(ROOT_DIR / "scripts" / "train_rsl_rl.py"),
        "task=serialleg_flat_mlp/motrix",
        "training.no_play=true",
    ]


def test_serialleg_cli_routes_to_appo_mujoco_owner_config() -> None:
    command = cli.build_command(
        mode="train",
        algo="appo",
        task="serialleg_flat_mlp",
        sim="mujoco",
        overrides=["training.no_play=true"],
        root=ROOT_DIR,
    )

    assert command[1:] == [
        str(ROOT_DIR / "scripts" / "train_appo.py"),
        "task=serialleg_flat_mlp/mujoco",
        "training.no_play=true",
    ]


def test_serialleg_cli_routes_to_appo_motrix_owner_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: object() if name == "motrixsim" else None)

    command = cli.build_command(
        mode="train",
        algo="appo",
        task="serialleg_flat_mlp",
        sim="motrix",
        overrides=["training.no_play=true"],
        root=ROOT_DIR,
    )

    assert command[1:] == [
        str(ROOT_DIR / "scripts" / "train_appo.py"),
        "task=serialleg_flat_mlp/motrix",
        "training.no_play=true",
    ]


def test_serialleg_xml_keeps_policy_order_sensor_contract() -> None:
    root = ET.parse(XML_PATH).getroot()
    actuators = [
        actuator.attrib["joint"]
        for actuator in root.find("actuator") or []
        if actuator.tag == "motor"
    ]
    sensors = {
        sensor.attrib["name"] for sensor in root.find("sensor") or [] if "name" in sensor.attrib
    }
    contact_sensors = {
        sensor.attrib["name"]: sensor.attrib
        for sensor in root.find("sensor") or []
        if sensor.tag == "contact" and "name" in sensor.attrib
    }

    assert actuators == [
        "lf0_Joint",
        "lf1_Joint",
        "l_wheel_Joint",
        "rf0_Joint",
        "rf1_Joint",
        "r_wheel_Joint",
    ]
    assert {
        "gyro",
        "local_linvel",
        "upvector",
        "robot_subtree_angmom",
        "track_pos_w_l_wheel_Link",
        "track_pos_w_r_wheel_Link",
        "base_contact",
        "l_wheel_contact",
        "r_wheel_contact",
        "lf0_contact",
        "lf1_contact",
        "rf0_contact",
        "rf1_contact",
    }.issubset(sensors)
    for name in (
        "base_contact",
        "l_wheel_contact",
        "r_wheel_contact",
        "lf0_contact",
        "lf1_contact",
        "rf0_contact",
        "rf1_contact",
    ):
        sensor = contact_sensors[name]
        assert sensor["geom1"] == "floor"
        assert sensor["data"] == "force"
        assert sensor["reduce"] == "netforce"
