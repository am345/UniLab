from __future__ import annotations

import queue

import numpy as np
import pytest
import torch

import unilab.algos.torch.appo.runner as appo_runner_module
from unilab.algos.torch.appo.runner import APPORunner


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    _FakeLearner.last_instance = None
    _FakeRolloutRingBuffer.last_instance = None
    _FakeRolloutRingBuffer.instances = []
    _FakeRolloutRingBuffer.available_rollouts = 1
    _FakeRolloutRingBuffer.available_rollouts_by_instance = None
    _FakeLogger.last_instance = None


class _FakeModule:
    def __init__(self) -> None:
        self.loaded_state: dict[str, torch.Tensor] | None = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": torch.zeros(1)}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.loaded_state = state_dict


class _FakeOptimizer:
    def __init__(self) -> None:
        self.loaded_state: dict | None = None
        self.param_groups = [{"lr": 0.001}]

    def load_state_dict(self, state_dict: dict) -> None:
        self.loaded_state = state_dict
        self.param_groups = state_dict.get("param_groups", self.param_groups)


class _FakeLearner:
    last_instance: "_FakeLearner | None" = None

    def __init__(self) -> None:
        self.actor = _FakeModule()
        self.critic = _FakeModule()
        self.optimizer = _FakeOptimizer()
        self.learning_rate = 0.001
        self.num_learning_epochs = 1
        self.last_batch: dict[str, torch.Tensor] | None = None
        self.target_update_calls = 0
        _FakeLearner.last_instance = self

    def get_state_dict(self) -> dict[str, int]:
        return {"iteration": 0}

    def process_batch(self, batch: dict[str, torch.Tensor]) -> None:
        self.last_batch = batch

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        del batch
        return {"loss": 0.5}

    def update_target_network(self) -> None:
        self.target_update_calls += 1


class _FakeRolloutRingBuffer:
    last_instance: "_FakeRolloutRingBuffer | None" = None
    instances: list["_FakeRolloutRingBuffer"] = []
    available_rollouts: int = 1
    available_rollouts_by_instance: list[int] | None = None

    def __init__(
        self,
        *,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        critic_dim: int,
        num_slots: int,
        create: bool,
    ) -> None:
        self.index = len(_FakeRolloutRingBuffer.instances)
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic_dim = critic_dim
        self.num_slots = num_slots
        self.create = create
        self.name = f"fake-storage-{self.index}"
        self._write_ptr = object()
        self._read_ptr = object()
        self.wait_calls = 0
        self.advance_calls = 0
        if _FakeRolloutRingBuffer.available_rollouts_by_instance is None:
            self.available_rollouts = _FakeRolloutRingBuffer.available_rollouts
        else:
            self.available_rollouts = _FakeRolloutRingBuffer.available_rollouts_by_instance[
                self.index
            ]
        _FakeRolloutRingBuffer.instances.append(self)
        _FakeRolloutRingBuffer.last_instance = self

    @property
    def slot_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "obs": (self.num_envs, self.num_steps, self.obs_dim),
            "critic": (self.num_envs, self.num_steps, self.critic_dim),
            "actions": (self.num_envs, self.num_steps, self.action_dim),
            "log_probs": (self.num_envs, self.num_steps),
            "rewards": (self.num_envs, self.num_steps),
            "dones": (self.num_envs, self.num_steps),
            "truncated": (self.num_envs, self.num_steps),
            "last_obs": (self.num_envs, self.obs_dim),
            "last_critic": (self.num_envs, self.critic_dim),
        }

    def wait_for_data(self, timeout: float = 60.0) -> bool:
        del timeout
        self.wait_calls += 1
        return True

    def available(self) -> int:
        return max(self.available_rollouts - self.advance_calls, 0)

    def read_torch(self, device: str) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.zeros(
                self.num_envs,
                self.num_steps,
                self.obs_dim,
                device=device,
            ),
            "critic": torch.zeros(
                self.num_envs,
                self.num_steps,
                self.critic_dim,
                device=device,
            ),
            "actions": torch.zeros(
                self.num_envs,
                self.num_steps,
                self.action_dim,
                device=device,
            ),
            "log_probs": torch.zeros(self.num_envs, self.num_steps, device=device),
            "rewards": torch.zeros(self.num_envs, self.num_steps, device=device),
            "dones": torch.zeros(self.num_envs, self.num_steps, device=device),
            "truncated": torch.zeros(self.num_envs, self.num_steps, device=device),
            "last_obs": torch.zeros(self.num_envs, self.obs_dim, device=device),
            "last_critic": torch.zeros(self.num_envs, self.critic_dim, device=device),
        }

    def read_numpy_views(self) -> dict[str, np.ndarray]:
        value = float(self.index * 10 + self.advance_calls + 1)
        return {
            field: np.full(shape, value, dtype=np.float32)
            for field, shape in self.slot_shapes.items()
        }

    def advance_read(self) -> None:
        self.advance_calls += 1

    def cleanup(self) -> None:
        pass


class _FakeWeightSync:
    def __init__(self) -> None:
        self.name = "fake-weight-sync"
        self._lock = object()

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], create: bool = True
    ) -> "_FakeWeightSync":
        del state_dict, create
        return cls()

    def cleanup(self) -> None:
        pass

    def write_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        del state_dict


class _FakeLogger:
    last_instance: "_FakeLogger | None" = None

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self._total_steps = 0
        self._mean_ep_length = 0.0
        self.collection_sync_calls: list[tuple[bool, int]] = []
        self.step_calls: list[dict] = []
        _FakeLogger.last_instance = self

    def set_collection_sync(self, enabled: bool, env_steps_per_sync: int) -> None:
        self.collection_sync_calls.append((enabled, env_steps_per_sync))

    def start(self, *, status: str = "") -> None:
        del status

    def log_status(self, status: str) -> None:
        del status

    def log_save(self, ckpt_path: str) -> None:
        del ckpt_path

    def finish(self) -> None:
        pass

    def update_replay_queue(self, current_len: int, max_size: int) -> None:
        del current_len, max_size

    def update_staging_pool(self, current_len: int, max_size: int) -> None:
        del current_len, max_size

    def log_collector(self, total_steps: int, buffer_size: int, mean_reward: float = 0.0) -> None:
        del buffer_size, mean_reward
        self._total_steps = total_steps

    def update_ep_length(self, mean_ep_length: float) -> None:
        self._mean_ep_length = mean_ep_length

    def update_collector_timing(self, timing_ms: dict[str, float]) -> None:
        del timing_ms

    def update_done_rates(self, timeout_rate: float, terminated_rate: float) -> None:
        del timeout_rate, terminated_rate

    def log_step(self, **kwargs) -> None:
        self.step_calls.append(kwargs)


class _FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def time(self) -> float:
        return next(self._values)


def test_appo_runner_uses_explicit_runtime_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured_detect: dict[str, object] = {}
    captured_collector: dict[str, object] = {}

    def fake_detect_dims(self: APPORunner) -> tuple[int, int]:
        captured_detect["sim_backend"] = self.sim_backend
        self.critic_dim = 7
        self.critic_input_dim = 5
        return (4, 2)

    def capture_start_collector(*, target_fn, kwargs):
        del target_fn
        captured_collector.update(kwargs)

    monkeypatch.setattr(APPORunner, "_detect_dims", fake_detect_dims)
    monkeypatch.setattr(APPORunner, "_build_learner", lambda self: _FakeLearner())
    monkeypatch.setattr(appo_runner_module, "RolloutRingBuffer", _FakeRolloutRingBuffer)
    monkeypatch.setattr(appo_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(appo_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(appo_runner_module.torch, "save", lambda *args, **kwargs: None)

    runner = APPORunner(
        env_name="DummyEnv",
        env_cfg_overrides={"reward_config": {"scales": {"alive": 1.0}}},
        rl_cfg={"actor": {}, "critic": {}, "algorithm": {}},
        device="cpu",
        collector_device="cpu",
        sim_backend="motrix",
        num_envs=2,
        steps_per_env=4,
    )
    monkeypatch.setattr(runner, "_start_collector", capture_start_collector)

    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    assert captured_detect["sim_backend"] == "motrix"
    assert captured_collector["sim_backend"] == "motrix"
    assert captured_collector["env_cfg_override"] == {"reward_config": {"scales": {"alive": 1.0}}}


def test_appo_runner_restores_resume_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def fake_detect_dims(self: APPORunner) -> tuple[int, int]:
        self.critic_dim = 7
        self.critic_input_dim = 5
        return (4, 2)

    checkpoint = {
        "actor": {"weight": torch.ones(1)},
        "critic": {"weight": torch.full((1,), 2.0)},
        "optimizer": {"param_groups": [{"lr": 0.004}]},
    }

    monkeypatch.setattr(APPORunner, "_detect_dims", fake_detect_dims)
    monkeypatch.setattr(APPORunner, "_build_learner", lambda self: _FakeLearner())
    monkeypatch.setattr(appo_runner_module, "RolloutRingBuffer", _FakeRolloutRingBuffer)
    monkeypatch.setattr(appo_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(appo_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(appo_runner_module.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(appo_runner_module.torch, "save", lambda *args, **kwargs: None)

    runner = APPORunner(
        env_name="DummyEnv",
        env_cfg_overrides={},
        rl_cfg={"actor": {}, "critic": {}, "algorithm": {}},
        device="cpu",
        collector_device="cpu",
        sim_backend="mujoco",
        num_envs=2,
        steps_per_env=4,
        resume_path=str(tmp_path / "model_7.pt"),
    )
    monkeypatch.setattr(runner, "_start_collector", lambda *args, **kwargs: None)

    runner.learn(max_iterations=0, save_interval=0, log_dir=str(tmp_path))

    learner = _FakeLearner.last_instance
    assert learner is not None
    assert learner.actor.loaded_state == checkpoint["actor"]
    assert learner.critic.loaded_state == checkpoint["critic"]
    assert learner.optimizer.loaded_state == checkpoint["optimizer"]
    assert learner.learning_rate == pytest.approx(0.004)
    assert learner.target_update_calls == 1


def test_appo_runner_logs_learner_timing_for_fps_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def fake_detect_dims(self: APPORunner) -> tuple[int, int]:
        self.critic_dim = 7
        self.critic_input_dim = 5
        return (4, 2)

    monkeypatch.setattr(APPORunner, "_detect_dims", fake_detect_dims)
    monkeypatch.setattr(APPORunner, "_build_learner", lambda self: _FakeLearner())
    monkeypatch.setattr(APPORunner, "_check_collector_alive", lambda self: True)
    monkeypatch.setattr(appo_runner_module, "RolloutRingBuffer", _FakeRolloutRingBuffer)
    monkeypatch.setattr(appo_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(appo_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(appo_runner_module.mp, "get_context", lambda method: queue)
    monkeypatch.setattr(appo_runner_module.torch, "save", lambda *args, **kwargs: None)

    fake_clock = _FakeClock([100.0, 100.0, 110.0, 120.0, 120.5, 121.0])
    monkeypatch.setattr(appo_runner_module.time, "time", fake_clock.time)

    runner = APPORunner(
        env_name="DummyEnv",
        env_cfg_overrides={},
        rl_cfg={"actor": {}, "critic": {}, "algorithm": {}},
        device="cpu",
        collector_device="cpu",
        sim_backend="mujoco",
        num_envs=2,
        steps_per_env=4,
    )
    monkeypatch.setattr(runner, "_start_collector", lambda *args, **kwargs: None)

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    logger = _FakeLogger.last_instance
    storage = _FakeRolloutRingBuffer.last_instance
    assert logger is not None
    assert storage is not None
    assert storage.wait_calls == 1
    assert storage.advance_calls == 1
    assert logger.step_calls

    step = logger.step_calls[0]
    assert "collect_time" not in step
    assert step["wait_time"] == pytest.approx(10.0)
    assert step["train_time"] == pytest.approx(0.5)
    assert step["learner_incremental_h2d_time"] >= 0.0
    assert step["weight_sync_time"] >= 0.0
    assert step["extra_info"] == {"throughput_steps": 8}
    assert step["extra_info"]["throughput_steps"] == 8
    assert step["metrics"]["rollouts_read"] == 1.0
    assert step["metrics"]["staging_pool_len"] == 1.0


def test_appo_runner_num_workers_starts_isolated_collectors_and_counts_rollouts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def fake_detect_dims(self: APPORunner) -> tuple[int, int]:
        self.critic_dim = 7
        self.critic_input_dim = 5
        return (4, 2)

    start_calls: list[dict] = []
    _FakeRolloutRingBuffer.available_rollouts_by_instance = [2, 1]
    monkeypatch.setattr(APPORunner, "_detect_dims", fake_detect_dims)
    monkeypatch.setattr(APPORunner, "_build_learner", lambda self: _FakeLearner())
    monkeypatch.setattr(APPORunner, "_check_collector_alive", lambda self: True)
    monkeypatch.setattr(appo_runner_module, "RolloutRingBuffer", _FakeRolloutRingBuffer)
    monkeypatch.setattr(appo_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(appo_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(appo_runner_module.mp, "get_context", lambda method: queue)
    monkeypatch.setattr(appo_runner_module.torch, "save", lambda *args, **kwargs: None)

    fake_clock = _FakeClock([100.0, 100.0, 100.25, 100.25, 100.75, 101.0])
    monkeypatch.setattr(appo_runner_module.time, "time", fake_clock.time)

    runner = APPORunner(
        env_name="DummyEnv",
        env_cfg_overrides={},
        rl_cfg={"actor": {}, "critic": {}, "algorithm": {}},
        device="cpu",
        collector_device="cpu",
        sim_backend="mujoco",
        num_envs=2,
        steps_per_env=4,
        num_workers=2,
        replay_queue_size=3,
    )
    monkeypatch.setattr(
        runner,
        "_start_collector",
        lambda *, target_fn, kwargs: start_calls.append({"target_fn": target_fn, "kwargs": kwargs}),
    )

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    logger = _FakeLogger.last_instance
    learner = _FakeLearner.last_instance
    assert logger is not None
    assert learner is not None
    assert len(start_calls) == 2
    assert len(_FakeRolloutRingBuffer.instances) == 2
    assert logger.init_kwargs["num_envs"] == 4
    assert logger.collection_sync_calls == [(True, 16)]

    worker_kwargs = [call["kwargs"] for call in start_calls]
    assert [kwargs["worker_index"] for kwargs in worker_kwargs] == [0, 1]
    assert [kwargs["worker_name"] for kwargs in worker_kwargs] == [
        "APPOWorker-0",
        "APPOWorker-1",
    ]
    assert [kwargs["shm_rollout_ring_buffer_name"] for kwargs in worker_kwargs] == [
        "fake-storage-0",
        "fake-storage-1",
    ]

    first_ring, second_ring = _FakeRolloutRingBuffer.instances
    assert first_ring.advance_calls == 2
    assert second_ring.advance_calls == 1
    assert learner.last_batch is not None
    assert learner.last_batch["observations"].shape == (4, 6, 4)
    assert torch.equal(
        torch.unique(learner.last_batch["observations"]),
        torch.tensor([1.0, 2.0, 11.0]),
    )

    step = logger.step_calls[0]
    assert step["extra_info"] == {"throughput_steps": 24}
    assert step["metrics"]["rollouts_read"] == 3.0
    assert step["metrics"]["available_on_arrive"] == 3.0
    assert step["metrics"]["staging_pool_len"] == 3.0


def test_appo_runner_rollouts_per_update_caps_learner_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def fake_detect_dims(self: APPORunner) -> tuple[int, int]:
        self.critic_dim = 7
        self.critic_input_dim = 5
        return (4, 2)

    _FakeRolloutRingBuffer.available_rollouts_by_instance = [2, 1]
    monkeypatch.setattr(APPORunner, "_detect_dims", fake_detect_dims)
    monkeypatch.setattr(APPORunner, "_build_learner", lambda self: _FakeLearner())
    monkeypatch.setattr(APPORunner, "_check_collector_alive", lambda self: True)
    monkeypatch.setattr(appo_runner_module, "RolloutRingBuffer", _FakeRolloutRingBuffer)
    monkeypatch.setattr(appo_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(appo_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(appo_runner_module.mp, "get_context", lambda method: queue)
    monkeypatch.setattr(appo_runner_module.torch, "save", lambda *args, **kwargs: None)

    fake_clock = _FakeClock([100.0, 100.0, 100.25, 100.25, 100.75, 101.0])
    monkeypatch.setattr(appo_runner_module.time, "time", fake_clock.time)

    runner = APPORunner(
        env_name="DummyEnv",
        env_cfg_overrides={},
        rl_cfg={"actor": {}, "critic": {}, "algorithm": {}},
        device="cpu",
        collector_device="cpu",
        sim_backend="mujoco",
        num_envs=2,
        steps_per_env=4,
        num_workers=2,
        replay_queue_size=3,
        rollouts_per_update=2,
    )
    monkeypatch.setattr(runner, "_start_collector", lambda *args, **kwargs: None)

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    logger = _FakeLogger.last_instance
    learner = _FakeLearner.last_instance
    assert logger is not None
    assert learner is not None
    assert logger.collection_sync_calls == [(True, 16)]

    first_ring, second_ring = _FakeRolloutRingBuffer.instances
    assert first_ring.advance_calls == 2
    assert second_ring.advance_calls == 1
    assert learner.last_batch is not None
    assert learner.last_batch["observations"].shape == (4, 4, 4)
    assert torch.equal(
        torch.unique(learner.last_batch["observations"]),
        torch.tensor([2.0, 11.0]),
    )

    step = logger.step_calls[0]
    assert step["extra_info"] == {"throughput_steps": 24}
    assert step["metrics"]["rollouts_read"] == 3.0
    assert step["metrics"]["rollouts_per_update"] == 2.0
    assert step["metrics"]["rollouts_in_update"] == 2.0
    assert step["metrics"]["rollouts_overwritten"] == 1.0
    assert step["metrics"]["staging_pool_len"] == 2.0
    assert step["metrics"]["staging_pool_capacity"] == 2.0
    assert step["metrics"]["train_batch_env_steps"] == 16.0


def test_appo_runner_stages_multiple_rollouts_without_runner_cat(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def fake_detect_dims(self: APPORunner) -> tuple[int, int]:
        self.critic_dim = 7
        self.critic_input_dim = 5
        return (4, 2)

    def fail_cat(*args, **kwargs):
        del args, kwargs
        raise AssertionError("runner must not rebuild APPO batches with torch.cat")

    _FakeRolloutRingBuffer.available_rollouts = 2
    monkeypatch.setattr(APPORunner, "_detect_dims", fake_detect_dims)
    monkeypatch.setattr(APPORunner, "_build_learner", lambda self: _FakeLearner())
    monkeypatch.setattr(APPORunner, "_check_collector_alive", lambda self: True)
    monkeypatch.setattr(appo_runner_module, "RolloutRingBuffer", _FakeRolloutRingBuffer)
    monkeypatch.setattr(appo_runner_module, "SharedWeightSync", _FakeWeightSync)
    monkeypatch.setattr(appo_runner_module, "OffPolicyLogger", _FakeLogger)
    monkeypatch.setattr(appo_runner_module.mp, "get_context", lambda method: queue)
    monkeypatch.setattr(appo_runner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(appo_runner_module.torch, "cat", fail_cat)

    fake_clock = _FakeClock([100.0, 100.0, 110.0, 120.0, 120.5, 121.0])
    monkeypatch.setattr(appo_runner_module.time, "time", fake_clock.time)

    runner = APPORunner(
        env_name="DummyEnv",
        env_cfg_overrides={},
        rl_cfg={"actor": {}, "critic": {}, "algorithm": {}},
        device="cpu",
        collector_device="cpu",
        sim_backend="mujoco",
        num_envs=2,
        steps_per_env=4,
    )
    monkeypatch.setattr(runner, "_start_collector", lambda *args, **kwargs: None)

    runner.learn(max_iterations=1, save_interval=0, log_dir=str(tmp_path))

    storage = _FakeRolloutRingBuffer.last_instance
    learner = _FakeLearner.last_instance
    logger = _FakeLogger.last_instance
    assert storage is not None
    assert learner is not None
    assert learner.last_batch is not None
    assert logger is not None
    assert storage.advance_calls == 2

    batch = learner.last_batch
    assert batch["observations"].shape == (4, 4, 4)
    assert batch["actions"].shape == (4, 4, 2)
    assert batch["actions_log_prob"].shape == (4, 4)
    assert batch["last_obs"].shape == (4, 4)
    assert torch.equal(torch.unique(batch["observations"]), torch.tensor([1.0, 2.0]))
    assert logger.step_calls[0]["metrics"]["staging_pool_len"] == 2.0
    assert logger.step_calls[0]["metrics"]["rollouts_read"] == 2.0
