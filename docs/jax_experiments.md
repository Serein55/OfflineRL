# 两组JAX实验的运行与数据

## 保留产物

| 内容 | Vanilla | ARFM |
|---|---|---|
| checkpoint根目录 | `artifacts/jax_2k/checkpoints/pi0_libero/vanilla_seed42/` | `artifacts/jax_arfm/checkpoints/pi0_libero/arfm_expert2k_noaug_seed42/` |
| 15k eval | `artifacts/jax_2k/eval/expert2k_step_015000/` | `artifacts/jax_arfm/eval/arfm_expert2k_noaug_seed42_step_015000/` |
| 30k eval | `artifacts/jax_2k/eval/expert2k_step_030000/` | `artifacts/jax_arfm/eval/arfm_expert2k_noaug_seed42_step_030000/` |

每个checkpoint根目录保留`15000/`和`30000/`，以及metrics；每个eval目录保留四个suite的JSONL与summary。Vanilla最终1831/2000成功，ARFM1828/2000成功。

训练的原始配置快照分别为`artifacts/jax_2k/vanilla_seed42_config*.txt`和`artifacts/jax_arfm/runs/arfm_expert2k_noaug_seed42/config*.txt`；ARFM的protocol记录sidecar哈希、lambda和数据身份。

两次训练都是先15k再恢复到30k。恢复会重建DataLoader，不恢复采样游标；Vanilla旧metrics中存在15001–15384未保存更新及恢复后的重复step，分析时按最后一次step回退分段。

## 后续动作质量验证（未训练策略）

在保留Vanilla 30k策略上完成Spatial 10任务、30个初始episode、57个同状态分支点，共228条对照；另完成同一快照下32条多种子续跑。下表是未参与拟合的7个任务上、包含人为扰动的22个成功/失败配对，指标是**动作排序正确率，不是策略成功率**。

| 候选分数 | 排序正确率 | 结论 |
|---|---:|---|
| 5步事件回报 | 50.0%（全打平） | 没有提供区分信号 |
| 事件＋距离变化 | 31.8% | 掉落也可能更接近目标，未通过 |
| 同状态配对学习排序器 | 81.8% | 能识别部分人为扰动；保留集唯一一个结果不同的自然动作对排错 |

成功概率候选使用`A(s,c)=Q(s,c)−mean_candidate_Q`。一个开发状态的两个动作在16个续跑种子下分别成功7/16、14/16；前8种子估计、后8种子独立验证的排序方向一致，但独立验证配对检验p=0.125，尚未达到预设0.05门槛。保守advantage因此置0，未接入训练。

这轮验证针对replan=5实际执行前缀；不代表已验证完整50步开环chunk。详见[方法、原始证据和排除批次](action_advantage_validation.md)。现有训练配置、checkpoint和专家reward sidecar均未修改。

## 数据

保留原HDF5、原reward NPZ和manifest，用于验证来源和重新转换。`scripts/jax_2k/convert.py`保持2000条轨迹全部帧、原action/state及episode/frame索引；原生128图像旋转180度后无损PNG存入LeRobot parquet。20Hz，action chunk50，episode末尾按官方reader重复padding。

转换所需schema与metadata模板独立保存在`configs/expert2k_schema.arrow`和`configs/expert2k_info_template.json`，不再依赖已删除的官方1693条数据。extra-delta在模型transform中执行一次；训练norm stats位于`artifacts/jax_2k/assets/pi0_libero/local/libero_expert2k/norm_stats.json`。原reward与原始action对齐，不用delta action重算。

`data/jax_2k/annotations.npz`保留reward、平均剩余回报RTG、同concrete task全帧LOO advantage及索引；未使用任务内z-score。数据来源、逐文件哈希与映射仍保留在`artifacts/jax_2k/task_*.json`和`episode_mapping.jsonl`。

## 训练入口

在仓库根目录：

```bash
source scripts/jax_arfm/env.sh
# 新实验必须使用新名称；现有checkpoint不覆盖。
vendor/openpi_official/.venv/bin/python scripts/jax_2k/train.py \
  --experiment vanilla_new_seed42 --steps 30000
vendor/openpi_official/.venv/bin/python scripts/jax_arfm/train.py \
  --dataset expert2k --annotations data/jax_2k/annotations.npz \
  --experiment arfm_new_seed42 --method arfm --steps 30000
```

默认GPU4–7。上述命令仅供复现，本次清理没有启动训练。原自动续训脚本保留为两组实验的执行记录，已有完成状态时拒绝自动覆盖。

## 完整评估

评估共用`openpi_pipeline.evaluate`，新输出label必须不存在。例如重评Vanilla30k：

```bash
source scripts/jax_arfm/env.sh
export ARFM_JAX_DATASET=expert2k OPENPI_EVAL_RESOLUTION=128
vendor/openpi_official/.venv/bin/python - <<'PYCODE'
import openpi_pipeline as p
p.evaluate(p.ROOT/'artifacts/jax_2k/checkpoints/pi0_libero/vanilla_seed42/30000', 'vanilla_30k_reeval')
PYCODE
```

ARFM评估时把`p.ART`设为`p.ROOT/'artifacts/jax_arfm'`并传对应checkpoint。每个GPU加载一套policy服务、负责一个suite；客户端环境使用本地`.venv`。固定replan5、seed7、128渲染和EGL；不混用其他协议的成功率。

## 清理边界

2026-10-13删除本项目PyTorch训练/评估代码、权重、旧实验日志，官方1693数据实验、混合rollout及旧离线advantage探索。随后新增的同状态动作质量验证单独记录于上述文档。保留两组JAX所需的数据与环境。上游OpenPI包中存在PyTorch兼容代码及CPU reader依赖，保留上游依赖完整性，没有卸载环境中的torch。

测试已移除对旧PyTorch目标函数的导入，使用NumPy数值参考验证JAX solver、loss与梯度；同时检查四逻辑CPU设备全局分片、RTG/LOO、索引校验、JIT训练及EMA。6项通过。
