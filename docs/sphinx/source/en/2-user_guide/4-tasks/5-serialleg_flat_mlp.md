# SerialLeg Flat-MLP

`serialleg_flat_mlp` is the SerialLeg flat-ground walking base policy. The task
keeps the SerialLeg 6D policy action contract and real contact/reward terms, but
uses UniLab owner YAMLs so the same environment can run on Motrix for training
and MuJoCo for validation.

## Mainline

Use **APPO + Motrix** as the training mainline:

```bash
uv sync --extra motrix
SE3_LOGGER=tensorboard uv run train \
  --algo appo \
  --task serialleg_flat_mlp \
  --sim motrix \
  training.no_play=true
```

This routes to `conf/appo/task/serialleg_flat_mlp/motrix.yaml`, registers
`SerialLegFlatMLP` with `sim_backend="motrix"`, and uses the asynchronous APPO
runtime in `scripts/train_appo.py`. The owner defaults are sized as a conservative
single-machine baseline:

| Field | Default |
| --- | --- |
| `algo.num_envs` | `2048` |
| `algo.num_workers` | `2` |
| `algo.steps_per_env` | `32` |
| `env.motrix_max_iterations` | `3` |

For a minimal launch check, shrink the rollout and save the first checkpoint:

```bash
SE3_LOGGER=tensorboard uv run train \
  --algo appo \
  --task serialleg_flat_mlp \
  --sim motrix \
  training.no_play=true \
  training.log_root=/tmp/unilab_serialleg_smoke \
  algo.num_envs=64 \
  algo.num_workers=1 \
  algo.steps_per_env=4 \
  algo.max_iterations=1 \
  algo.save_interval=1
```

## MuJoCo Validation

Use **APPO + MuJoCo** to validate a Motrix-trained checkpoint:

```bash
SE3_LOGGER=tensorboard uv run eval \
  --algo appo \
  --task serialleg_flat_mlp \
  --sim mujoco \
  --load-run <run_dir_name_or_-1> \
  --render-mode none \
  training.play_env_num=2 \
  training.play_steps=64
```

This routes to `conf/appo/task/serialleg_flat_mlp/mujoco.yaml` and loads the APPO
checkpoint from the APPO log root. Keep the task, reward, observation, action,
and control settings aligned between the Motrix and MuJoCo owner YAMLs; backend
differences should stay explicit in the owner files.

PPO owners also exist for synchronous baselines:

```bash
uv run train --algo ppo --task serialleg_flat_mlp --sim motrix training.no_play=true
uv run eval --algo ppo --task serialleg_flat_mlp --sim mujoco --load-run -1 --render-mode none
```

## Motrix Capability Boundary

The Motrix owner intentionally disables domain-randomization paths that are not
yet equivalent to MuJoCo for this robot:

- `env.domain_rand.randomize_ground_friction: false`
- `env.domain_rand.randomize_body_inertia: false`
- `env.domain_rand.randomize_dof_armature: false`
- `env.domain_rand.push_robots: false`

Base mass, base CoM, PD gains, default joint position, and action-delay
randomization remain active through the SerialLeg task config. The friction
randomization implementation now only writes robot collision geoms, so future
Motrix friction support can be enabled without modifying visual-only geoms.

Motrix scene materialization also performs temporary XML compatibility fixes:

- unsupported `site type="ellipsoid"` entries are converted to spheres in the
  generated Motrix XML;
- body-based contact sensors are converted to geom-based contact sensors because
  Motrix requires both sides of a contact sensor to resolve to geoms.

The original SerialLeg MJCF remains the source of truth. These conversions are
backend materialization details, not asset edits.

## Benchmark Notes

Do not compare this task directly against generic UniLab Go2/G1/Motrix
env-step-only benchmark numbers. SerialLeg Flat-MLP uses real reward terms,
contact sensors, four-bar surrogate mappings, action delay, and backend reset
randomization. Use the APPO Motrix command for training throughput and the MuJoCo
eval command for behavioral validation.

Recommended checks before a longer run:

```bash
uv run --no-sync python -m pytest tests/envs/locomotion/serialleg/test_flat_mlp_contract.py -q
uv run --no-sync --with ruff ruff check \
  src/unilab/envs/locomotion/serialleg/flat_mlp.py \
  src/unilab/base/backend/motrix/scene.py \
  tests/envs/locomotion/serialleg/test_flat_mlp_contract.py
```
