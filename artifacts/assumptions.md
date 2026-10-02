# 复现假设与边界

这是独立复现，不是作者公开实现；附带文档的建议被视为技术参考，而不是额外用户指令。

- 固定 LeRobot `ce3b9f627e55223d6d1c449d348c6b351b35d082`，不是论文声明的 commit。
- 初始权重 `lerobot/pi0_old@e4ed526af508e58f6008b29e9e48f1098278fdb5`，基础 π0，没有用现成 LIBERO 微调权重替代训练。该模型与新版 `pi0_base` 不混用。
- LIBERO 官方原始演示：Goal / Spatial / Object / libero_10（Long），各 10 tasks × 50 demos。数据版本见下载 manifest。
- state = ee_pos(3) + ee_ori axis-angle(3) + gripper qpos(2)，action 7 维；数据全局 mean/std；原 LeRobot pad 到 32 维。沿用其时间 padding 清零后固定分母 mean（包括 padded action dimensions）的行为。
- 奖励不是作者原始代码：Table 8 的 13 项和权重，图像 128×128 原始 uint8；MSE exp(-.01 MSE)，SSIM exp(SSIM-1)，ORB exp(match-ratio-1)，ORB edgeThreshold=0 fastThreshold=40，Hamming cross-check。
- subgoal 取同一成功 demo 的下一 heuristic keypoint。抓手指令符号改变、关节逐步差分绝对值≤0.1且抓手连续稳定（4-step cooldown）或最后帧触发。修复参考 heuristic 的前两帧负索引问题。progress 使用当前时间已跨过的 keypoints 比例。这是 hindsight reconstruction，非机器人在线奖励。
- Joint MSE 使用 exp(-.01 mean squared error)，是未公开尺度的显式选择。关节速度/加速度用逐 control step 差分，没有除 dt；动作使用原始数据差分；第一步差分置零。
- Success：官方成功演示假设，只有末帧加 1；该定义与逐帧 trajectory success 是不同 ablation。
- RTG 是 reward suffix 的平均。LOO 在具体 task 的所有 chunk-start RTG 上求，不是 trajectory-only mean，也不是 suite 分组。无额外 z-score。均匀抽 task 后均匀抽 chunk。
- 主时间采样 Uniform；支持 --time-sampler beta 使用冻结原版 Beta(1.5,1.0)。保持 noise→action 的原版整套符号约定。
- 全局 16 samples 求 α 和 softmax；不做 microbatch 局部归一化。α 使用 E[R²]，不额外 batch 中心化；FP64 求根，平方 α bounds；边界 fallback；exp 大值只用于单调符号判定。
- Full finetune（视觉也解冻），FP32 主参数、BF16 compute，MLP activation checkpointing 和 optimizer state sharding，不是 LoRA。原实现有未参与 loss 的 LM head，不为它伪造梯度。
- 1000-step warmup 到 2.5e-5，再 30000-step cosine 到 2.5e-6，之后 hold。AdamW β=(.9,.95), eps=1e-8, wd=1e-10, clip=10。
- ColorJitter 顺序随机，每张相机独立采样；每次应用；sharpness 均匀 .5–1.5。论文未明确顺序/概率/跨相机同步。
- 评估：50 个官方固定 init states/task，seed 42，settle 10 steps，Spatial 220 / Goal、Object 280 / Long 520 步上限，10 flow steps，执行50-action chunk；相机保持原始方向，与本次 raw HDF5 stored demos 一致（通过同一 simulator state 像素对照验证）。最后 checkpoint，无 best-checkpoint 选择。多 seed 与其他执行 horizon 需另行实验。
- resume 保存优化器和各 rank RNG，但多 worker 的预取/增强 RNG 不恢复到逐位一致；采样索引按 step 可重放。
- 只有实际完成训练与评估后才能报告成功率；不能把论文 92.1% 写成这里的结果。
- 实测兼容性修正：历史 commit 的 pyproject 声明 Transformers <4.52，但 pi0 代码及下载权重已使用新 PaliGemma 结构；运行固定为 Transformers 4.53.2。旧 4.51.3 的严格权重加载失败，未用 strict=False 绕过。
- LIBERO 官方 init-state 文件包含 NumPy 对象，针对 PyTorch ≥2.6 将官方加载行显式设为 `weights_only=False`；补丁保存于 `artifacts/libero.patch`。
- checkpoint 不调用 ZeRO 的全量 `consolidate_state_dict`（实测其大 bytearray→ByteTensor 转换过慢）。模型与各卡本地 AdamW state 分开保存，所有分片提交后原子更新 latest 链接；保留最近两个完整步骤。续跑必须保持 world_size=4。
