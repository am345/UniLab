# SerialLeg Flat-MLP

`serialleg_flat_mlp` 是 SerialLeg 平地行走 MLP 基模任务。它保留 SerialLeg
6 维 policy action、真实 contact sensor、真实 reward 项和四连杆 surrogate 映射，
同时通过 UniLab owner YAML 让同一个任务可以用 Motrix 训练、用 MuJoCo 验证。

## 训练主线

训练主线使用 **APPO + Motrix**：

```bash
uv sync --extra motrix
SE3_LOGGER=tensorboard uv run train \
  --algo appo \
  --task serialleg_flat_mlp \
  --sim motrix \
  training.no_play=true
```

这条命令会进入 `conf/appo/task/serialleg_flat_mlp/motrix.yaml`，注册名是
`SerialLegFlatMLP`，后端是 `sim_backend="motrix"`，训练 runtime 是
`scripts/train_appo.py` 中的异步 APPO。当前 owner 默认档位是保守的单机基线：

| 字段 | 默认值 |
| --- | --- |
| `algo.num_envs` | `2048` |
| `algo.num_workers` | `2` |
| `algo.steps_per_env` | `32` |
| `env.motrix_max_iterations` | `3` |

最小启动检查可以缩小 rollout，并在第一轮保存 checkpoint：

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

## MuJoCo 验证

Motrix 训练出的 APPO checkpoint 用 **APPO + MuJoCo** 验证：

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

这条命令进入 `conf/appo/task/serialleg_flat_mlp/mujoco.yaml`，并从 APPO log root
解析 checkpoint。Motrix 和 MuJoCo 两个 owner 应保持 task、reward、observation、
action 和 control 参数一致；后端差异只放在 owner YAML 中显式表达。

仓库也保留 PPO owner，便于做同步采样基线：

```bash
uv run train --algo ppo --task serialleg_flat_mlp --sim motrix training.no_play=true
uv run eval --algo ppo --task serialleg_flat_mlp --sim mujoco --load-run -1 --render-mode none
```

## Motrix 能力边界

Motrix owner 当前故意关闭以下 domain randomization 路径，因为这些能力在
SerialLeg 上还没有和 MuJoCo 达到等价：

- `env.domain_rand.randomize_ground_friction: false`
- `env.domain_rand.randomize_body_inertia: false`
- `env.domain_rand.randomize_dof_armature: false`
- `env.domain_rand.push_robots: false`

base mass、base CoM、PD gain、default joint position 和 action delay 随机化仍然由
SerialLeg 任务配置启用。摩擦随机化的实现现在只会写机器人 collision geom，后续
Motrix 摩擦能力补齐后，可以打开摩擦随机化而不会误改 visual-only geom。

Motrix scene materialization 还会做临时 XML 兼容：

- 不支持的 `site type="ellipsoid"` 会在生成的 Motrix XML 中转成 sphere；
- body-based contact sensor 会转成 geom-based contact sensor，因为 Motrix 要求
  contact sensor 两侧都能解析到 geom。

原始 SerialLeg MJCF 仍然是唯一真实资产。这些转换只是后端 materialization 细节，
不会改动资产源文件。

## Benchmark 口径

不要把这个任务直接和 UniLab 常见 Go2/G1/Motrix env-step-only benchmark 数字硬比。
SerialLeg Flat-MLP 有真实 reward、contact sensor、四连杆 surrogate、action delay
和 backend reset randomization。训练吞吐看 APPO Motrix 命令，行为正确性看 MuJoCo
eval 命令。

长训前建议先跑：

```bash
uv run --no-sync python -m pytest tests/envs/locomotion/serialleg/test_flat_mlp_contract.py -q
uv run --no-sync --with ruff ruff check \
  src/unilab/envs/locomotion/serialleg/flat_mlp.py \
  src/unilab/base/backend/motrix/scene.py \
  tests/envs/locomotion/serialleg/test_flat_mlp_contract.py
```
