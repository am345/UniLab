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
    SerialLegRewardConfig,
)
from unilab.envs.locomotion.serialleg.fourbar import policy_to_output_torque_np
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
    env._default_policy_leg_pos = np.broadcast_to(DEFAULT_POLICY_LEG_POS, (2, 4)).copy()
    return env


def test_serialleg_flat_mlp_registers_mujoco_backend() -> None:
    import unilab.envs.locomotion.serialleg  # noqa: F401

    assert registry.contains("SerialLegFlatMLP")
    registered = registry.list_registered_envs()["SerialLegFlatMLP"]
    assert registered["config_class"] == "SerialLegFlatMLPCfg"
    assert registered["available_backends"] == ["mujoco"]


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
    assert cfg.control_config.min_action_delay_s == pytest.approx(0.004)
    assert cfg.control_config.max_action_delay_s == pytest.approx(0.006)
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


def test_serialleg_backend_dr_payload_excludes_custom_policy_pd_gains() -> None:
    env = cast(Any, object.__new__(SerialLegFlatMLPEnv))
    env._cfg = SerialLegFlatMLPCfg(reward_config=SerialLegRewardConfig())
    env._num_action = NUM_ACTIONS
    env._base_body_mass = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    env._base_geom_friction = np.array([[0.8, 0.005, 0.0001]], dtype=np.float64)
    env._ground_geom_id = 0
    env._base_dof_armature = np.ones(12, dtype=np.float64)
    env._base_body_inertia = np.ones((3, 3), dtype=np.float64)

    payload = env.build_reset_randomization(num_reset=2)

    assert payload is not None
    assert RESET_TERM_KP not in payload.requested_terms()
    assert RESET_TERM_KD not in payload.requested_terms()
    assert env._cfg.domain_rand.randomize_kp is True
    assert env._cfg.domain_rand.randomize_kd is True


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
        "base_contact",
        "l_wheel_contact",
        "r_wheel_contact",
        "lf0_contact",
        "lf1_contact",
        "rf0_contact",
        "rf1_contact",
    }.issubset(sensors)
