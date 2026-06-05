"""Train APPO agent — native multiprocessing."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from unilab.algos.torch.appo.learner import clamp_distribution_std
from unilab.algos.torch.appo.runtime import resolve_appo_runtime
from unilab.training import (
    BackendAdapter,
    apply_configured_training_seed,
    create_env,
    ensure_registries,
    get_log_root,
    log_playback_plan,
    should_run_playback,
)
from unilab.training.experiment import ExperimentTracker


def _training_resume_requested(load_run: Any) -> bool:
    if load_run is None:
        return False
    return str(load_run) not in {"", "-1"}


def build_appo_runner_kwargs(
    cfg: DictConfig,
    env_cfg_override: dict | None,
    collector_device: str | None,
    rl_cfg: dict[str, Any] | None = None,
) -> dict:
    if rl_cfg is None:
        rl_cfg_raw = OmegaConf.to_container(cfg.algo, resolve=True)
        if not isinstance(rl_cfg_raw, dict):
            raise TypeError("cfg.algo must resolve to a dict")
        rl_cfg = cast(dict[str, Any], rl_cfg_raw)

    runner_kwargs = {
        "env_name": cfg.training.task_name,
        "env_cfg_overrides": env_cfg_override,
        "rl_cfg": rl_cfg,
        "device": cfg.training.device,
        "collector_device": collector_device,
        "num_envs": cfg.algo.num_envs,
        "num_workers": int(OmegaConf.select(cfg, "algo.num_workers", default=1)),
        "rollouts_per_update": OmegaConf.select(cfg, "algo.rollouts_per_update", default=None),
        "min_rollouts_for_update": OmegaConf.select(
            cfg, "algo.min_rollouts_for_update", default=None
        ),
        "steps_per_env": cfg.algo.steps_per_env,
        "sim_backend": cfg.training.sim_backend,
        "seed": rl_cfg.get("seed"),
    }
    if cfg.training.replay_queue_size is not None:
        runner_kwargs["replay_queue_size"] = cfg.training.replay_queue_size
    load_run = OmegaConf.select(cfg, "algo.load_run", default="-1")
    if _training_resume_requested(load_run):
        resume_path, _ = resolve_appo_checkpoint_path(
            os.path.join(_get_log_root(cfg), cfg.training.task_name),
            str(load_run),
        )
        if resume_path is None:
            raise FileNotFoundError(f"Could not resolve APPO resume checkpoint: {load_run}")
        runner_kwargs["resume_path"] = resume_path
    return runner_kwargs


def apply_appo_runtime_flags(
    rl_cfg: dict[str, Any],
    cfg: DictConfig,
    *,
    training_enabled: bool,
) -> None:
    algorithm_cfg = rl_cfg.setdefault("algorithm", {})
    if not isinstance(algorithm_cfg, dict):
        return
    if not training_enabled:
        algorithm_cfg["enable_compile"] = False


def run_motrix_play_loop(
    env,
    actor,
    device: str,
    play_env_num: int,
    num_steps: int | None = None,
    stochastic_output: bool = False,
) -> None:
    import numpy as np
    from tensordict import TensorDict

    if env.state is None:
        env.init_state()

    with torch.inference_mode():
        env.run_playback(
            num_steps=num_steps,
            initialize=lambda: np.asarray(
                env.reset(np.arange(play_env_num, dtype=np.int32))[0]["obs"],
                dtype=np.float32,
            ),
            step=lambda obs_np: np.asarray(
                env.step(
                    actor(
                        TensorDict(
                            {"policy": torch.from_numpy(obs_np).to(device)}, batch_size=play_env_num
                        ),
                        stochastic_output=stochastic_output,
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                ).obs["obs"],
                dtype=np.float32,
            ),
        )


def _action_bounds_for_play(
    env: Any, action_dim: int, device: str
) -> tuple[torch.Tensor, torch.Tensor] | None:
    action_space = getattr(env, "action_space", None)
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return None
    import numpy as np

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


def _bounded_action_mean_for_play(
    mean: torch.Tensor,
    action_bounds: tuple[torch.Tensor, torch.Tensor] | None,
    action_bound_limit: float,
) -> torch.Tensor:
    if action_bounds is None:
        return float(action_bound_limit) * torch.tanh(mean / float(action_bound_limit))
    low_torch, high_torch = action_bounds
    center = 0.5 * (low_torch + high_torch)
    half_width = (0.5 * (high_torch - low_torch)).clamp_min(1e-6)
    return center + half_width * torch.tanh((mean - center) / half_width)


def _distribution_std_for_play(actor: Any, mean: torch.Tensor) -> torch.Tensor:
    distribution = actor.distribution
    if distribution.std_type == "scalar":
        std = distribution.std_param.expand_as(mean)
    else:
        std = torch.exp(distribution.log_std_param).expand_as(mean)
    return torch.nan_to_num(std, nan=0.05, posinf=2.0, neginf=0.05).clamp(min=0.05, max=2.0)


def resolve_appo_checkpoint_path(
    base_log_dir: str | Path,
    load_run: str | int,
) -> tuple[str | None, str | None]:
    from unilab.training import resolve_checkpoint_path

    checkpoint_path, checkpoint_dir = resolve_checkpoint_path(
        base_log_dir,
        str(load_run),
        suffix=".pt",
    )
    return (
        str(checkpoint_path) if checkpoint_path is not None else None,
        str(checkpoint_dir) if checkpoint_dir is not None else None,
    )


def _get_log_root(cfg: DictConfig) -> str:
    return str(get_log_root(ROOT_DIR, cfg))


def _current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _warn_if_play_commit_mismatch(load_path_dir: str | None) -> None:
    if load_path_dir is None:
        return
    run_config_path = Path(load_path_dir) / "run_config.json"
    if not run_config_path.is_file():
        return
    try:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: Could not read run_config.json for play provenance: {exc}")
        return
    run_commit = (
        run_config.get("run", {}).get("git", {}).get("commit")
        if isinstance(run_config, dict)
        else None
    )
    current_commit = _current_git_commit()
    if run_commit and current_commit and run_commit != current_commit:
        print(
            "WARNING: Play checkpoint was trained with a different git commit: "
            f"run_config={run_commit[:12]}, current={current_commit[:12]}. "
            "Use the original commit or regenerate the checkpoint before judging policy behavior."
        )


def play_appo(
    cfg: DictConfig,
    rl_cfg: dict[str, Any],
    *,
    root_dir: Path | None = None,
    resolve_checkpoint_path: Callable[[DictConfig], tuple[str | None, str | None]] | None = None,
) -> str | None:
    """Play mode for the default APPO runtime.

    Args:
        cfg: Resolved Hydra config for the current run.
        rl_cfg: Resolved algorithm config dictionary from Hydra composition.
        root_dir: Optional project root forwarded by generic runtime callers.
            The default APPO runtime does not need it and ignores the value.
        resolve_checkpoint_path: Optional checkpoint resolver injected by the
            generic script. When omitted, this function falls back to the
            default log-root based APPO checkpoint resolution.

    Returns:
        Output video path for offscreen rendering, or ``None`` when running the
        native Motrix viewer or when no checkpoint could be resolved.
    """
    del root_dir
    import numpy as np
    from rsl_rl.utils import resolve_callable
    from tensordict import TensorDict

    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="appo"
    ).build_task_env_cfg_override()

    device = cfg.training.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device for play: {device}")

    env = cast(
        Any,
        create_env(
            cfg,
            num_envs=cfg.training.play_env_num,
            env_cfg_override=env_cfg_override,
        ),
    )
    from unilab.base.observations import get_obs_dims

    obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
    action_shape = env.action_space.shape
    if action_shape is None:
        raise ValueError("env.action_space.shape must be defined")
    action_dim = int(action_shape[0])

    rl_cfg_dict = dict(rl_cfg)
    if "obs_groups" not in rl_cfg_dict:
        rl_cfg_dict["obs_groups"] = {
            "actor": {"policy": obs_dim},
            "critic": {"policy": critic_dim if critic_dim > 0 else obs_dim},
        }
    else:
        actor_group = rl_cfg_dict["obs_groups"].get(
            "actor", rl_cfg_dict["obs_groups"].get("policy", {})
        )
        if isinstance(actor_group, dict) and "policy" in actor_group:
            actor_group["policy"] = obs_dim
        critic_group = rl_cfg_dict["obs_groups"].get("critic")
        if critic_group is None:
            rl_cfg_dict["obs_groups"]["critic"] = {
                "policy": critic_dim if critic_dim > 0 else obs_dim
            }
        elif isinstance(critic_group, dict) and "policy" in critic_group:
            critic_group["policy"] = critic_dim if critic_dim > 0 else obs_dim

    from copy import deepcopy

    obs_example = torch.zeros((cfg.training.play_env_num, obs_dim), device=device)
    td_example = TensorDict({"policy": obs_example}, batch_size=cfg.training.play_env_num)

    actor_cfg = deepcopy(rl_cfg_dict["actor"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    actor_cfg.pop("num_actions", None)
    actor = actor_cls(td_example, rl_cfg_dict["obs_groups"], "actor", action_dim, **actor_cfg)
    actor = actor.to(device)
    actor.eval()

    if resolve_checkpoint_path is not None:
        load_path, load_path_dir = resolve_checkpoint_path(cfg)
    else:
        log_root = _get_log_root(cfg)
        base_log_dir = os.path.join(log_root, cfg.training.task_name)
        load_path, load_path_dir = resolve_appo_checkpoint_path(base_log_dir, cfg.algo.load_run)

    if not load_path or not os.path.exists(load_path):
        print(f"Could not find run to load. load_path={load_path}")
        return None

    print(f"Loading model: {load_path}")
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint["actor"])
    clamp_distribution_std(actor)
    _warn_if_play_commit_mismatch(load_path_dir)
    play_stochastic = bool(getattr(cfg.training, "play_stochastic", False))
    algorithm_cfg = rl_cfg_dict.get("algorithm", {})
    bounded_action_mean = bool(
        algorithm_cfg.get("bounded_action_mean", False)
        if isinstance(algorithm_cfg, dict)
        else False
    )
    action_bound_limit = float(
        algorithm_cfg.get("action_bound_limit", 1.0) if isinstance(algorithm_cfg, dict) else 1.0
    )
    if action_bound_limit <= 0.0:
        raise ValueError(f"action_bound_limit must be > 0, got {action_bound_limit}")
    action_bounds = _action_bounds_for_play(env, action_dim, device)
    print(f"Using stochastic play actions: {play_stochastic}")
    print(f"Using bounded action mean for play: {bounded_action_mean}")

    def actor_play_action(obs_tensor: torch.Tensor) -> torch.Tensor:
        if not bounded_action_mean:
            return actor(
                TensorDict({"policy": obs_tensor}, batch_size=obs_tensor.shape[0]),
                stochastic_output=play_stochastic,
            )
        mean = actor.mlp(actor.obs_normalizer(obs_tensor))
        mean = _bounded_action_mean_for_play(mean, action_bounds, action_bound_limit)
        if not play_stochastic:
            return mean
        std = _distribution_std_for_play(actor, mean)
        action = mean + std * torch.randn_like(mean)
        if action_bounds is None:
            return torch.clamp(action, -action_bound_limit, action_bound_limit)
        low_torch, high_torch = action_bounds
        return torch.clamp(action, low_torch, high_torch)

    # Export actor to ONNX
    if load_path_dir is not None:
        import numpy as np
        import torch.nn as nn

        class _DeterministicAPPOActor(nn.Module):
            def __init__(
                self,
                actor_model: nn.Module,
                use_bounded_mean: bool,
                limit: float,
                bounds: tuple[torch.Tensor, torch.Tensor] | None,
            ):
                super().__init__()
                self.actor_model = actor_model
                self.use_bounded_mean = use_bounded_mean
                self.limit = float(limit)
                if bounds is None:
                    self.low = None
                    self.high = None
                else:
                    self.register_buffer("low", bounds[0].reshape(1, -1))
                    self.register_buffer("high", bounds[1].reshape(1, -1))

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                action = self.actor_model.mlp(self.actor_model.obs_normalizer(obs))
                if not self.use_bounded_mean:
                    return action
                if self.low is None or self.high is None:
                    return self.limit * torch.tanh(action / self.limit)
                center = 0.5 * (self.low + self.high)
                half_width = (0.5 * (self.high - self.low)).clamp_min(1e-6)
                return center + half_width * torch.tanh((action - center) / half_width)

        export_module = _DeterministicAPPOActor(
            actor, bounded_action_mean, action_bound_limit, action_bounds
        )
        onnx_path = os.path.join(load_path_dir, "policy.onnx")
        dummy_input = torch.randn(1, obs_dim, device=device)
        with torch.inference_mode():
            torch.onnx.export(
                export_module,
                (dummy_input,),
                onnx_path,
                input_names=["obs"],
                output_names=["action"],
                opset_version=17,
            )
        print(f"Exported actor ONNX to {onnx_path}")

        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        verify_input = torch.randn(1, obs_dim, device=device)
        with torch.inference_mode():
            pt_output = export_module(verify_input).cpu().numpy()
        onnx_output = sess.run(None, {"obs": verify_input.cpu().numpy().astype(np.float32)})[0]
        max_diff = np.max(np.abs(pt_output - onnx_output))
        mean_diff = np.mean(np.abs(pt_output - onnx_output))
        print(f"ONNX vs PyTorch — max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e}")
        if max_diff > 1e-4:
            print("WARNING: ONNX output diverges from PyTorch!")
        else:
            print("ONNX export verified OK.")

    if env.state is None:
        env.init_state()

    with torch.inference_mode():
        play_video_path = env.run_playback_mode(
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
            play_steps=getattr(cfg.training, "play_steps", None),
            output_video=os.path.join(load_path_dir, "play_video.mp4") if load_path_dir else None,
            render_spacing=float(
                getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
            ),
            initialize=lambda: np.asarray(
                env.reset(np.arange(cfg.training.play_env_num, dtype=np.int32))[0]["obs"],
                dtype=np.float32,
            ),
            step=lambda obs_np: np.asarray(
                env.step(
                    actor_play_action(torch.from_numpy(obs_np).to(device))
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                ).obs["obs"],
                dtype=np.float32,
            ),
            camera_kwargs={
                "cam_distance": cfg.training.cam_distance,
                "cam_elevation": cfg.training.cam_elevation,
                "cam_azimuth": cfg.training.cam_azimuth,
                "cam_lookat": getattr(cfg.training, "cam_lookat", None),
                "cam_tracking": getattr(cfg.training, "cam_tracking", False),
                "cam_tracking_env_idx": getattr(cfg.training, "cam_tracking_env_idx", 0),
                "cam_tracking_extra_envs": getattr(cfg.training, "cam_tracking_extra_envs", 2),
            },
            on_plan=log_playback_plan,
        )
    if play_video_path is not None:
        print(f"Saving video to {play_video_path} with mediapy...")
    print("Done.")
    return play_video_path


@hydra.main(version_base="1.3", config_path="../conf/appo", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="appo"
    ).build_task_env_cfg_override()

    # Convert algo config to plain dict for APPORunner / RSL-RL internals
    rl_cfg_raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(rl_cfg_raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    rl_cfg = cast(dict[str, Any], rl_cfg_raw)
    apply_appo_runtime_flags(rl_cfg, cfg, training_enabled=not cfg.training.play_only)
    appo_runtime = resolve_appo_runtime(rl_cfg, default_play_fn=play_appo)

    if cfg.training.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root = _get_log_root(cfg)
        log_dir = os.path.join(
            log_root,
            cfg.training.task_name,
            f"{timestamp}_{cfg.training.sim_backend}",
        )
    else:
        log_dir = cfg.training.log_dir

    collector_device = cfg.training.collector_device
    if collector_device == "gpu":
        collector_device = "mps" if torch.backends.mps.is_available() else "cuda"

    learner_device = cfg.training.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    tracker = None
    if not cfg.training.play_only:
        tracker = ExperimentTracker(
            root_dir=ROOT_DIR,
            log_dir=log_dir,
            algo_name="appo",
            task_name=cfg.training.task_name,
            sim_backend=cfg.training.sim_backend,
            training_cfg=cfg.training,
            full_cfg=cfg,
            device=learner_device,
            collector_device=collector_device,
            seed_info=seed_info,
        )
        tracker.start()

    try:
        if not cfg.training.play_only:
            runner = appo_runtime.runner_cls(
                **build_appo_runner_kwargs(
                    cfg,
                    env_cfg_override=env_cfg_override,
                    collector_device=collector_device,
                    rl_cfg=rl_cfg,
                )
            )

            try:
                runner.learn(
                    max_iterations=cfg.algo.max_iterations,
                    save_interval=cfg.algo.save_interval,
                    log_dir=log_dir,
                    logger_type=cfg.training.logger,
                )
                if tracker is not None:
                    tracker.update_summary(getattr(runner, "last_run_summary", None))
            finally:
                runner.close()

        if should_run_playback(
            play_only=cfg.training.play_only,
            no_play=cfg.training.no_play,
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
        ):
            play_video_path = appo_runtime.play_fn(
                cfg,
                rl_cfg,
                root_dir=ROOT_DIR,
                resolve_checkpoint_path=lambda current_cfg: resolve_appo_checkpoint_path(
                    os.path.join(_get_log_root(current_cfg), current_cfg.training.task_name),
                    current_cfg.algo.load_run,
                ),
            )
            if tracker is not None:
                tracker.log_video(play_video_path)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
