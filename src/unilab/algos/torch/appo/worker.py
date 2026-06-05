"""APPO Rollout Worker — runs in a subprocess.

Collects rollout payloads and writes them to RolloutRingBuffer.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import defaultdict
from queue import Empty, Full
from typing import Any, Dict

import numpy as np
import torch
from rsl_rl.utils import resolve_callable

from unilab.algos.torch.appo.learner import clamp_distribution_std
from unilab.base.final_observation import resolve_terminal_observation_contract
from unilab.base.observations import split_obs_dict
from unilab.base.registry import ensure_registries
from unilab.training.seed import apply_training_seed


def put_latest_metrics(metrics_queue: Any, msg: dict[str, Any], *, worker_name: str) -> None:
    """Best-effort metrics enqueue that keeps recent data under learner stalls."""
    try:
        metrics_queue.put_nowait(msg)
        return
    except Full:
        pass
    except Exception as e:
        print(f"[{worker_name}] metrics enqueue error: {type(e).__name__}: {e}", file=sys.stderr)
        return

    try:
        metrics_queue.get_nowait()
    except Empty:
        pass
    except Exception as e:
        print(
            f"[{worker_name}] metrics drop stale metrics error: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return

    try:
        metrics_queue.put_nowait(msg)
    except Full:
        pass
    except Exception as e:
        print(f"[{worker_name}] metrics enqueue error: {type(e).__name__}: {e}", file=sys.stderr)


def _record_timing_ms(
    timing_accum_ms: dict[str, float],
    timing_counts: dict[str, int],
    key: str,
    value: float,
) -> None:
    timing_accum_ms[key] += float(value)
    timing_counts[key] += 1


def _record_phase_ms(
    timing_accum_ms: dict[str, float],
    timing_counts: dict[str, int],
    key: str,
    start_ns: int,
) -> int:
    end_ns = time.perf_counter_ns()
    _record_timing_ms(timing_accum_ms, timing_counts, key, (end_ns - start_ns) / 1e6)
    return end_ns


def _average_timing_ms(
    timing_accum_ms: dict[str, float],
    timing_counts: dict[str, int],
) -> dict[str, float]:
    return {
        key: value / timing_counts[key]
        for key, value in timing_accum_ms.items()
        if timing_counts[key] > 0
    }


def _action_bounds_tensors(env: Any, action_dim: int, device: str) -> tuple[Any, Any] | None:
    action_space = getattr(env, "action_space", None)
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return None
    low_np = np.asarray(low, dtype=np.float32)
    high_np = np.asarray(high, dtype=np.float32)
    try:
        low_np = np.broadcast_to(low_np, (action_dim,)).copy()
        high_np = np.broadcast_to(high_np, (action_dim,)).copy()
    except ValueError:
        return None
    if not (np.all(np.isfinite(low_np)) and np.all(np.isfinite(high_np))):
        return None
    return torch.from_numpy(low_np).to(device), torch.from_numpy(high_np).to(device)


def _record_env_timing_ms(
    timing_accum_ms: dict[str, float],
    timing_counts: dict[str, int],
    env_timing: Any,
) -> None:
    if not isinstance(env_timing, dict):
        return
    for key, value in env_timing.items():
        if not isinstance(value, int | float | np.integer | np.floating):
            continue
        timing_key = "env_inner_step_total_ms" if key == "env_step_total_ms" else f"env_{key}"
        _record_timing_ms(timing_accum_ms, timing_counts, timing_key, float(value))


def compute_timeout_bootstrap_correction(
    critic: Any,
    collector_device: str,
    gamma: float,
    timeout_mask: np.ndarray,
    final_obs: np.ndarray,
    final_critic: np.ndarray,
) -> np.ndarray:
    """Compute gamma * V(final_observation) for current timeout envs."""
    corrections = np.zeros(timeout_mask.shape, dtype=np.float32)
    if not np.any(timeout_mask):
        return corrections

    from tensordict import TensorDict

    critic_input_np = final_critic
    critic_input = torch.from_numpy(critic_input_np[timeout_mask]).to(collector_device)
    critic_td = TensorDict(
        {"policy": critic_input},
        batch_size=critic_input.shape[0],
        device=collector_device,
    )
    with torch.no_grad():
        bootstrap = critic(critic_td).squeeze(-1).cpu().numpy().astype(np.float32, copy=False)
    corrections[timeout_mask] = float(gamma) * bootstrap
    return corrections


def appo_collector_fn(
    stop_event: Any,
    env_name: str,
    rl_cfg: dict,
    num_envs: int,
    steps_per_env: int,
    shm_rollout_ring_buffer_name: Dict[str, str],
    sync_primitives: tuple,
    obs_dim: int,
    action_dim: int,
    critic_dim: int,
    actor_weight_sync_name: str,
    actor_weight_param_shapes: dict,
    critic_weight_sync_name: str,
    critic_weight_param_shapes: dict,
    metrics_queue: Any,
    collector_device: str = "cpu",
    sim_backend: str = "mujoco",
    env_cfg_override: dict | None = None,
    seed: int | None = None,
    worker_index: int = 0,
    worker_name: str | None = None,
    actor_weight_sync_lock: Any | None = None,
    critic_weight_sync_lock: Any | None = None,
):
    """Entry point for the APPO collector subprocess.

    Creates environment + policy, collects rollouts, writes raw payloads
    to the IPC ring buffer. Error handling is provided by the
    ``_collector_entry_wrapper`` in ``async_runner.py``.
    """
    from copy import deepcopy

    from tensordict import TensorDict

    from unilab.base import registry
    from unilab.ipc import RolloutRingBuffer, SharedWeightSync

    worker_label = worker_name or f"APPOWorker-{worker_index}"
    ensure_registries()
    apply_training_seed(seed, torch_runtime=True, cuda=True)

    # Connect to shared memory
    ring_buffer = RolloutRingBuffer(
        num_envs=num_envs,
        num_steps=steps_per_env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        critic_dim=critic_dim,
        create=False,
        shm_name_prefix=shm_rollout_ring_buffer_name,
    )
    ring_buffer.attach_sync_primitives(*sync_primitives)  # (write_ptr, read_ptr)
    actor_weight_sync = SharedWeightSync(
        actor_weight_param_shapes,
        create=False,
        shm_name=actor_weight_sync_name,
        lock=actor_weight_sync_lock,
    )
    critic_weight_sync = SharedWeightSync(
        critic_weight_param_shapes,
        create=False,
        shm_name=critic_weight_sync_name,
        lock=critic_weight_sync_lock,
    )

    # Create environment
    env: Any = registry.make(
        env_name, num_envs=num_envs, sim_backend=sim_backend, env_cfg_override=env_cfg_override
    )
    action_bounds = _action_bounds_tensors(env, action_dim, collector_device)

    # Build actor (stochastic MLPModel — mirrors runner._build_learner)
    cfg = dict(rl_cfg)

    obs_example = torch.zeros((num_envs, obs_dim), device=collector_device)
    td_example = TensorDict({"policy": obs_example}, batch_size=num_envs)

    # deepcopy so MLPModel.__init__'s distribution_cfg.pop("class_name") doesn't
    # mutate the shared rl_cfg dict.
    actor_cfg = deepcopy(cfg["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    actor = actor_cls(
        td_example,
        cfg.get("obs_groups", {"actor": {"policy": obs_dim}}),
        "actor",
        action_dim,
        **actor_cfg,
    )
    actor = actor.to(collector_device)
    actor.eval()

    critic_obs_dim = critic_dim if critic_dim > 0 else obs_dim
    critic_obs_example = torch.zeros((num_envs, critic_obs_dim), device=collector_device)
    critic_td_example = TensorDict({"policy": critic_obs_example}, batch_size=num_envs)
    critic_cfg = deepcopy(cfg.get("critic") or cfg.get("actor") or {})
    critic_cls = resolve_callable(critic_cfg.pop("class_name", "rsl_rl.models.MLPModel"))
    critic_cfg.pop("num_actions", None)
    critic_cfg.pop("distribution_cfg", None)
    critic = critic_cls(
        critic_td_example,
        cfg.get("obs_groups", {"critic": {"policy": critic_obs_dim}}),
        "critic",
        1,
        **critic_cfg,
    )
    critic = critic.to(collector_device)
    critic.eval()
    # Load initial weights
    actor_sd = dict(actor.state_dict())
    actor_weight_sync.read_weights_into(actor_sd)
    actor.load_state_dict(actor_sd)
    clamp_distribution_std(actor)
    local_actor_weight_version = actor_weight_sync.version

    critic_sd = dict(critic.state_dict())
    critic_weight_sync.read_weights_into(critic_sd)
    critic.load_state_dict(critic_sd)
    local_critic_weight_version = critic_weight_sync.version

    # Reset environment
    env_indices = np.arange(num_envs, dtype=np.int32)
    try:
        obs_out, _ = env.reset(env_indices)
    except TypeError:
        obs_out, _ = env.reset()

    def to_float32_np(x):
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    obs_np, critic_np = split_obs_dict(obs_out)
    obs_np = to_float32_np(obs_np)
    critic_np = to_float32_np(critic_np)

    # Pre-allocate obs TensorDict once; update in-place each step to avoid
    # repeated TensorDict construction overhead in the hot loop.
    obs_torch = torch.zeros((num_envs, obs_dim), dtype=torch.float32, device=collector_device)
    obs_td = TensorDict({"policy": obs_torch}, batch_size=num_envs, device=collector_device)

    total_steps = 0
    ep_rewards = []
    ep_lengths = []
    current_ep_rewards = np.zeros(num_envs, dtype=np.float32)
    current_ep_lengths = np.zeros(num_envs, dtype=np.int32)
    ep_reward_components = defaultdict(list)

    # Episode completion mode counters (reset after each metrics report)
    ep_timeouts = 0
    ep_terminates = 0

    timing_accum_ms: dict[str, float] = defaultdict(float)
    timing_counts: dict[str, int] = defaultdict(int)

    try:
        while not stop_event.is_set():
            rollout_start_ns = time.perf_counter_ns()
            phase_start_ns = rollout_start_ns

            # Pull latest weights from learner
            if actor_weight_sync.version > local_actor_weight_version:
                actor_sd = dict(actor.state_dict())
                local_actor_weight_version = actor_weight_sync.read_weights_into(actor_sd)
                actor.load_state_dict(actor_sd)
                clamp_distribution_std(actor)
            if critic_weight_sync.version > local_critic_weight_version:
                critic_sd = dict(critic.state_dict())
                local_critic_weight_version = critic_weight_sync.read_weights_into(critic_sd)
                critic.load_state_dict(critic_sd)
            phase_start_ns = _record_phase_ms(
                timing_accum_ms,
                timing_counts,
                "weight_sync_ms",
                phase_start_ns,
            )

            # Collect one rollout of length steps_per_env
            write_buf = ring_buffer.write_buffer
            phase_start_ns = _record_phase_ms(
                timing_accum_ms,
                timing_counts,
                "ring_wait_ms",
                phase_start_ns,
            )
            for step in range(steps_per_env):
                # --- MLP inference (timed) ---
                phase_start_ns = time.perf_counter_ns()
                with torch.no_grad():
                    obs_torch.copy_(torch.from_numpy(obs_np))
                    raw_actions_torch = actor(obs_td, stochastic_output=True)
                    if action_bounds is None:
                        actions_torch = raw_actions_torch
                    else:
                        low_torch, high_torch = action_bounds
                        actions_torch = torch.clamp(raw_actions_torch, low_torch, high_torch)
                    log_probs_torch = actor.get_output_log_prob(actions_torch)
                    raw_actions_np = raw_actions_torch.cpu().numpy()
                    actions_np = actions_torch.cpu().numpy()
                phase_start_ns = _record_phase_ms(
                    timing_accum_ms,
                    timing_counts,
                    "mlp_infer_ms",
                    phase_start_ns,
                )

                write_buf["obs"][:, step, :] = obs_np
                if critic_np is not None:
                    write_buf["critic"][:, step, :] = critic_np
                write_buf["actions"][:, step, :] = actions_np
                write_buf["raw_actions"][:, step, :] = raw_actions_np
                write_buf["log_probs"][:, step] = log_probs_torch.cpu().numpy().ravel()
                phase_start_ns = _record_phase_ms(
                    timing_accum_ms,
                    timing_counts,
                    "ipc_write_ms",
                    phase_start_ns,
                )

                # --- Env step (timed) ---
                state = env.step(actions_np)
                phase_start_ns = _record_phase_ms(
                    timing_accum_ms,
                    timing_counts,
                    "env_step_total_ms",
                    phase_start_ns,
                )
                _record_env_timing_ms(
                    timing_accum_ms,
                    timing_counts,
                    state.info.get("timing", {}),
                )

                next_obs_raw = state.obs
                reward_raw = np.asarray(state.reward, dtype=np.float32).ravel()
                truncated_raw = state.truncated.astype(np.float32, copy=False).ravel()
                combined_done_raw = (
                    (state.terminated | state.truncated).astype(np.float32, copy=False).ravel()
                )

                next_actor_obs_np, next_critic_np = split_obs_dict(next_obs_raw)
                next_actor_obs_np = to_float32_np(next_actor_obs_np)
                next_critic_np = to_float32_np(next_critic_np)
                terminal_contract = resolve_terminal_observation_contract(
                    next_obs_batch_size=next_actor_obs_np.shape[0],
                    final_observation=state.final_observation,
                    done=combined_done_raw > 0.5,
                    info=state.info,
                    truncated=truncated_raw,
                )

                reward_raw += compute_timeout_bootstrap_correction(
                    critic=critic,
                    collector_device=collector_device,
                    gamma=float(cfg["algorithm"].get("gamma", 0.99)),
                    timeout_mask=terminal_contract.timeout_terminal_mask,
                    final_obs=(
                        terminal_contract.terminal_obs
                        if terminal_contract.terminal_obs is not None
                        else next_actor_obs_np
                    ),
                    final_critic=(
                        terminal_contract.terminal_critic
                        if terminal_contract.terminal_critic is not None
                        else next_critic_np
                    ),
                )

                write_buf["rewards"][:, step] = reward_raw
                write_buf["dones"][:, step] = combined_done_raw
                write_buf["truncated"][:, step] = truncated_raw

                # Episode tracking (vectorized)
                total_steps += num_envs
                current_ep_rewards += reward_raw
                current_ep_lengths += 1
                reset_indices = np.where(combined_done_raw > 0.5)[0]
                if len(reset_indices) > 0:
                    ep_rewards.extend(current_ep_rewards[reset_indices].tolist())
                    ep_lengths.extend(current_ep_lengths[reset_indices].tolist())
                    current_ep_rewards[reset_indices] = 0.0
                    current_ep_lengths[reset_indices] = 0
                    # Count episode completion modes for timeout/terminated rates
                    ep_timeouts += int(np.sum(truncated_raw[reset_indices] > 0.5))
                    ep_terminates += int(np.sum(truncated_raw[reset_indices] <= 0.5))

                log_info = state.info.get("log", {})
                for k, v in log_info.items():
                    if k.startswith("reward/"):
                        ep_reward_components[k].append(v)

                if metrics_queue is not None and total_steps % (num_envs * 10) == 0 and ep_rewards:
                    try:
                        msg = {
                            "worker_index": worker_index,
                            "worker_name": worker_label,
                            "total_steps": total_steps,
                            "mean_ep_reward": statistics.mean(ep_rewards[-100:]),
                            "mean_ep_length": statistics.mean(ep_lengths[-100:])
                            if ep_lengths
                            else 0.0,
                        }
                        # Episode completion mode rates
                        total_ep = ep_timeouts + ep_terminates
                        if total_ep > 0:
                            msg["timeout_rate"] = ep_timeouts / total_ep
                            msg["terminated_rate"] = ep_terminates / total_ep
                            ep_timeouts = 0
                            ep_terminates = 0
                        if ep_reward_components:
                            msg["reward_components"] = {
                                k: statistics.mean(v) for k, v in ep_reward_components.items() if v
                            }
                            ep_reward_components.clear()
                        put_latest_metrics(metrics_queue, msg, worker_name=worker_label)
                    except Exception as e:
                        print(
                            f"[{worker_label}] metrics build error: {type(e).__name__}: {e}",
                            file=sys.stderr,
                        )

                obs_np = next_actor_obs_np
                critic_np = next_critic_np
                phase_start_ns = _record_phase_ms(
                    timing_accum_ms,
                    timing_counts,
                    "postprocess_ms",
                    phase_start_ns,
                )

            phase_start_ns = time.perf_counter_ns()
            write_buf["last_obs"][:] = obs_np
            if critic_np is not None:
                write_buf["last_critic"][:] = critic_np
            ring_buffer.signal_write_done()  # atomic increment, non-blocking
            phase_start_ns = _record_phase_ms(
                timing_accum_ms,
                timing_counts,
                "rollout_finalize_ms",
                phase_start_ns,
            )
            _record_timing_ms(
                timing_accum_ms,
                timing_counts,
                "rollout_total_ms",
                (phase_start_ns - rollout_start_ns) / 1e6,
            )
            if metrics_queue is not None and timing_counts:
                put_latest_metrics(
                    metrics_queue,
                    {
                        "worker_index": worker_index,
                        "worker_name": worker_label,
                        "collector_timing_ms": _average_timing_ms(
                            timing_accum_ms,
                            timing_counts,
                        ),
                    },
                    worker_name=worker_label,
                )
                timing_accum_ms.clear()
                timing_counts.clear()

    except Exception:
        stop_event.set()
        raise

    ring_buffer.close()
    actor_weight_sync.close()
    critic_weight_sync.close()
    env.close()
