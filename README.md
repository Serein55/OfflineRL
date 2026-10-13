# ARFM · JAX π₀

仅保留原2k专家数据上的两组JAX实验：Vanilla FM与ARFM。模型与优化器使用官方OpenPI JAX/Flax框架；ARFM是本仓库独立实现，并非论文作者代码。

## 保留结果

| 实验 | steps | Spatial | Object | Goal | Long | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla FM | 15k | 73.4% | 94.8% | 85.8% | 75.4% | 82.35% |
| Vanilla FM | 30k | 93.8% | 96.2% | 90.0% | 86.2% | **91.55%** |
| ARFM | 15k | 79.8% | 95.8% | 83.6% | 75.8% | 83.75% |
| ARFM | 30k | 91.0% | 97.2% | 92.0% | 85.4% | **91.40%** |

每次eval为40任务×50次，共2000条。30k ARFM比Vanilla少成功3次，当前单seed实验未显示总体优势。两组均保留15k/30k checkpoint（含优化器和EMA）、逐episode评估、配置、metrics与训练日志。

## 配置

原2000条专家轨迹、338575帧；π₀ base初始化；无图像增强；global batch32；seed42；四卡FSDP；EMA=0.99；flow timestep为Beta(1.5,1.0)×0.999+0.001；action horizon50；官方extra-delta与独立重算norm stats。15k恢复后续训到累计30k，恢复优化器与EMA，DataLoader重新初始化。

评估使用EMA、EGL、seed7、replan=5、原生128渲染后resize224，任务内复用环境。ARFM复用与数据索引对应的原reward/RTG/LOO advantage，不做z-score；lambda=5e-4，adaptive alpha范围[0.01,5]。详细说明见[运行与数据](docs/jax_experiments.md)。

## 目录

- `src/arfm_jax/`：JAX目标函数、训练步骤、奖励和索引校验。
- `scripts/jax_2k/`：数据转换、norm stats、Vanilla训练和续训评估。
- `scripts/jax_arfm/`：ARFM训练、奖励sidecar和续训评估。
- `scripts/openpi_*.py`：共用配置、策略服务与评估。
- `artifacts/jax_2k/`、`artifacts/jax_arfm/`：仅保留的两组实验及数据验证记录。
- `data/raw/`、`data/processed/`、`data/openpi_lerobot/local/libero_expert2k/`、`data/jax_2k/`：原专家数据、复用奖励、JAX数据及索引sidecar。
- `checkpoints/openpi_pi0_base/`、`checkpoints/openpi_cache/`：初始化权重和tokenizer缓存。
- `vendor/openpi_official/`：固定版本OpenPI与其LIBERO依赖。

## 环境与验证

训练使用`vendor/openpi_official/.venv/bin/python`；模拟器客户端使用`.venv/bin/python`。OpenPI的CPU数据加载仍依赖PyTorch/LeRobot，因此保留运行依赖；本项目旧PyTorch模型训练实现已删除。

```bash
source scripts/jax_arfm/env.sh
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  vendor/openpi_official/.venv/bin/python tests/jax_arfm/test_port.py
```

清理范围见[删除清单](docs/cleanup_manifest.json)。模型权重、训练数据、运行环境和日志不进入Git；实验配置、评估结果与有效诊断证据随代码保留。

## 动作质量验证

已完成Spatial 10任务的228条同状态对照和32条多种子续跑。5步事件分数未区分成功/失败动作；学习排序器在含人为扰动的保留集上正确率81.8%，尚未验证自然动作排序有效。成功概率advantage出现局部信号，但未通过独立验证门槛。详见[验证结果与边界](docs/action_advantage_validation.md)。候选分数尚未接入训练。
