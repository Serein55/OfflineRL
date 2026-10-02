# ARFM / π0 + LIBERO 独立复现

2026-10-03 advantage scaling 实验：新增 `--advantage-normalization task_zscore`，按 concrete task 的全部训练 chunk 计算均值和总体标准差，epsilon=1e-8。默认 `none` 保留原实验行为。离线真实 loss 诊断及实验取舍见 [scaling report](artifacts/scaling_diagnostic/report.md)。

```bash
source scripts/env.sh
# 仅前向诊断，不更新模型
.venv/bin/torchrun --standalone --nproc_per_node=4 scripts/diagnose_advantage_scaling.py
# 诊断通过后，依次训练 ARFM / RWR alpha=0.1 / RWR alpha=0.5，各40k步并以replan=5评估
bash scripts/run_scaling_experiments.sh
```

已有 vanilla replan=5 基线为 **80.3%**，原始 ARFM 为 **81.3%**。新实验从原始 π0 独立初始化；不修改 vanilla、action transform、camera 或 sampler。`artifacts/scaling_comparison.json` 记录新实验完成后的比较结果。相同输出目录恢复需要 `--resume`，且不能改变 advantage scaling 或方法。

当前实验部署在 `ganrenda/ARFM`，限定 GPU 0–3。仓库不包含权重、数据、虚拟环境和运行 checkpoint。

代码包括：冻结历史 LeRobot π0、13 项离线奖励重建、RTG / LOO、跨 GPU 全局 ARFM loss、全参数训练、checkpoint 续跑、LIBERO 固定初始状态评估。详细假设见 `artifacts/assumptions.md`。推送时的进度快照和实测结果见 `artifacts/status.md`（GitHub 不会自动同步后台进度）；本 README 不意味着训练或数值复现已经完成。

克隆后先准备依赖与第三方源码（Linux、Python 3.11；安装步骤未在另一台机器上重新验证）：

```bash
git clone https://github.com/Serein55/OfflineRL.git
cd OfflineRL
bash scripts/fetch_sources.sh
python3.11 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements-lock.txt
.venv/bin/pip install --no-deps -e vendor/lerobot
source scripts/env.sh
.venv/bin/python scripts/download_assets.py
# 指向官方 PaliGemma SentencePiece tokenizer.model；本次实验复用了已有 OpenPI 缓存。
export PALIGEMMA_TOKENIZER_MODEL=/path/to/paligemma_tokenizer.model
.venv/bin/python scripts/setup_local.py
```

`fetch_sources.sh` 固定第三方 commit 并应用保存的兼容补丁，不提交第三方仓库副本。`requirements-lock.txt` 是实际运行环境快照，LeRobot 单独以本地源码安装，避免其历史依赖声明覆盖实测兼容的 Transformers 版本。

```bash
cd OfflineRL
source scripts/env.sh
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/preprocess.py --workers 24
.venv/bin/torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --method arfm --time-sampler uniform --steps 40000 --output artifacts/arfm_uniform_seed42
```

同一入口 `--method vanilla` 是基线，`--method rwr --fixed-alpha 0.1` 是固定权重对照；`--time-sampler beta` 切换历史采样方式。所有实验使用相同数据和归一化。完整流程由 `scripts/run_pipeline.sh` 执行，正式任务启动信息见 status 文件。

```bash
.venv/bin/python scripts/evaluate.py \
  --checkpoint artifacts/arfm_uniform_seed42/latest.pt \
  --output artifacts/arfm_uniform_seed42/evaluation.jsonl
```

训练每步写 `metrics.jsonl`，每1000步保存模型、各卡优化器分片、RNG 和归一化统计，原子更新 `latest.pt` 链接，保留最近两个完整 checkpoint，恢复需要原来的4卡规模，添加 `--resume <latest.pt>`，`--steps` 为累计目标步数。

依赖固定在 `.venv`，版本快照 `artifacts/environment.lock.txt`；已有项目未被修改。模型及数据的不可变下载版本记录在各 `download.json`。`artifacts/lerobot.patch` 记录局部改动（允许指定本地 tokenizer）。当前 tokenizer 复用已有官方 OpenPI SentencePiece 文件，哈希已保存。

参考：[ARFM](https://arxiv.org/abs/2509.04063)、[LeRobot](https://github.com/huggingface/lerobot)、[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)、[ReinboT](https://github.com/COST-97/reinboT)。附件原文已留存于 `artifacts/ARFM_reproduction_guide.md`。

查看状态：`.venv/bin/python scripts/status.py`。后台会话启动后使用 `tmux -L arfm list-sessions` 查看；训练和评估日志均保存在 `logs/`。`artifacts/*/results.json` 只在完整的 4×500 次评估实际结束且验证无缺项时生成。

每种方法完成评估后，流程还会更新 `artifacts/report/results.md`、训练 loss/α/ESS 曲线及 advantage 分布图。也可手动运行 `.venv/bin/python scripts/report.py` 生成当前进度图；未完成评估的成功率保持 pending。
