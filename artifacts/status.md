# 推送时的实验状态快照

更新时间：2026-10-02T12:19:07.739693+08:00。此文件不是实时监控。

- π0 baseline：40,000/40,000步，40任务×50次评估完成，平均成功率57.8%。
- Goal 70.4%，Spatial 58.8%，Object 65.6%，Long 36.4%。详见 `../results/vanilla_uniform_seed42.json`。
- ARFM：28581/40,000步，仍在训练，尚无完整评估结果。
- RWR：尚未开始，后台流程在ARFM评估后继续执行。
- GPU 0–3，4×A800 80GB，40任务×50演示=2000轨迹，338,575 action chunks。
- 已通过单元测试、四卡梯度一致性、真实模型严格加载/反向传播、checkpoint保存/恢复、观测回放对齐及单次闭环运行检查。

当前baseline显著低于文档中的88.1%，尚未对齐论文，不能宣称复现了92.1%。ARFM初期及当前权重接近均匀；需要完整评估后分析差异。

本机实时状态：`.venv/bin/python scripts/status.py`；后台tmux server `arfm`、session `reproduction`。
实验假设见 `assumptions.md`，版本见 `provenance.json` 和 `environment.lock.txt`。
