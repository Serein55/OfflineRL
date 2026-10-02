# ARFM 完整复现指南：π0 + LIBERO Offline RL Post-Training

> 论文：**Balancing Signal and Variance: Adaptive Offline RL Post-Training for VLA Flow Models**（AAAI 2026）  
> 方法：**ARFM — Adaptive Reinforced Flow Matching**  
> 目标：在**没有真机**的条件下，尽可能忠实地复现论文的 **π0 + LIBERO** 主实验，并明确区分“论文明确给出”“论文存在矛盾”“论文没有交代”的实现细节。

---

## 0. 先给结论

ARFM 的**核心算法是可复现的**：它本质上是在 π0 的逐样本 flow-matching loss 上施加 `softmax(alpha * advantage)` 权重，而 `alpha` 每个 batch 通过一个闭式推导后的非线性方程用二分法自适应求解。

但如果目标是**严格复现论文 92.1% 的 LIBERO 平均成功率**，仅靠论文目前还不够。主要原因不是 ARFM loss 本身，而是下面几项关键细节没有完全公开：

1. LIBERO dense reward 的 13 个分量虽然给了形式和权重，但 `f_MSE / f_SSIM / f_ORB` 的具体缩放、subgoal 构造、参考帧选择等没有完整定义。
2. “按 task category 计算 Leave-One-Out advantage” 中的分组方式与 `K` 的具体构造仍然模糊。
3. 论文算法写 `tau ~ Uniform(0,1)`，但论文所基于的 LeRobot π0 在论文提交前已经使用 `Beta(1.5, 1.0)` 的 flow time sampling。
4. Algorithm 2 的伪代码存在明显的**二次指数（double exponential）不一致**；Algorithm 1 的 `x` 上下界也和正文推导不一致。
5. 超参表同时写了 `Learning Rate = 1e-4` 和 `Peak Learning Rate = 2.5e-5`，没有解释二者关系。
6. 论文没有给出 LeRobot commit、π0 初始 checkpoint、LIBERO 每任务实际使用 demo 数、evaluation seeds / rollouts 数。

因此建议把复现分成两层：

- **Level A：算法复现**——严格验证 ARFM 比 vanilla π0 / RWR 是否稳定提升。这是可做的。
- **Level B：数值复现**——追论文 Table 1 的 92.1%。需要对 reward、advantage grouping、time sampler 等不明确项做系统 ablation，不能保证一次对齐。

---

## 1. 论文到底做了什么

ARFM 从一个已经能进行 flow matching 的 VLA policy（论文主干是 π0）出发，把 offline 数据中不同 action chunk 的质量信息编码成 advantage：

$$
R^*(A_t,o_t).
$$

普通 π0 flow matching 对所有样本基本等权：

$$
L_{FM}=
\mathbb E\left[
\|v_\theta(A_t^\tau,o_t)-u(A_t^\tau\mid A_t)\|_2^2
\right].
$$

ARFM 把目标数据分布重加权为

$$
\pi(A_t\mid o_t)
\propto
p(A_t\mid o_t)\exp\bigl(\alpha R^*(A_t,o_t)\bigr),
$$

然后在一个 batch 内使用归一化权重

$$
w_i(\alpha)
=
\frac{\exp(\alpha R_i^*)}
{\sum_{j=1}^{B}\exp(\alpha R_j^*)}.
$$

训练目标变为

$$
L_{ARFM}
=
\sum_{i=1}^{B}w_i(\alpha)L^{(i)}_{FM}.
$$

当 $\alpha=0$ 时，$w_i=1/B$，所以 ARFM 退化为普通 batch mean flow-matching loss。

核心问题是：$\alpha$ 太小没有 RL 信号，太大又会让少数高 advantage 样本垄断梯度。因此论文设计

$$
J(\alpha)
=
\operatorname{Var}(\hat g(\alpha))
-
\lambda S(\alpha),
$$

用来折中 gradient variance 和 RL advantage signal。

在高斯近似下，论文得到

$$
J(\alpha)
=
\sigma_L^2
\left[
\exp(2\alpha^2\sigma_R^2)
-
\exp(\alpha^2\sigma_R^2)
\right]
-
\lambda\alpha\sigma_R^2.
$$

令

$$
x=\alpha^2\sigma_R^2,
$$

最优值满足

$$
F(x)=
4\sqrt{x}e^{2x}
-2\sqrt{x}e^x
-\frac{\lambda\sigma_R}{\sigma_L^2}
=0,
$$

然后

$$
\alpha^*=\frac{\sqrt{x^*}}{\sigma_R}.
$$

这就是整个 ARFM 的核心。

---

## 2. 复现范围：先只做论文主实验

### 2.1 推荐的第一阶段目标

只做：

$$
\boxed{\pi_0 + LIBERO + ARFM}
$$

不要一开始就换成 π0.5，也不要先搬到 RoboTwin。原因是：如果同时换 backbone、benchmark、reward，你无法判断失败来自 ARFM 实现还是迁移本身。

论文 LIBERO 主实验使用四个 suite：

- LIBERO-Goal
- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Long

每个 suite 10 个任务，总计 40 个任务。

论文的 Table 1 目标数值如下：

| Model | Goal | Spatial | Object | Long | Average |
|---|---:|---:|---:|---:|---:|
| π0 | 93.8 | 91.2 | 93.2 | 74.2 | 88.1 |
| ReinboT | 94.0 | 95.6 | 93.8 | 81.4 | 91.2 |
| RWR | 94.4 | 94.0 | 94.3 | 80.4 | 90.8 |
| **ARFM** | **94.9** | **95.8** | **95.0** | **82.6** | **92.1** |

你第一阶段不应该把“精确达到 92.1%”当唯一成功标准。更合理的复现判据是：

1. vanilla π0 baseline 和论文处于同一数量级；
2. ARFM 对 Long 等难任务有稳定改善；
3. ARFM 的 advantage weighting 不出现 collapse；
4. $\alpha$ 的 batch-wise 变化合理；
5. 和固定 $\alpha$ / RWR 相比，adaptive $\alpha$ 确实更稳定。

---

## 3. 环境与版本：必须先冻结

论文明确写道：π0、ReinboT、RWR 和 ARFM 都基于 **`huggingface/lerobot`** 实现，并使用 2 张 A100-SXM4-80GB 做全参数微调。

但论文**没有提供 LeRobot commit hash**，这是一个重要复现缺口。

### 3.1 建议的版本策略

保留两套环境：

```text
arfm-paper/
├── env_paper_literal/       # 按论文公式/Algorithm 2
└── env_lerobot_2025/        # 尽量贴近论文开发时 LeRobot
```

为了避免 2026 年之后 LeRobot 对 π0 代码的修改影响结果，建议至少记录：

```text
python version
pytorch version
cuda version
lerobot git commit
libero git commit
transformers version
numpy version
opencv version
scikit-image version
```

我核对了 LeRobot 的历史记录：在 **2025-08-07** 的 π0 实现中已经使用了 `Beta(1.5, 1.0)` timestep sampler，并把采样值变换为 `0.999 * t + 0.001`。因此论文的 `Uniform(0,1)` 并不自动等于作者实际代码。

一个可用于“接近论文提交时 LeRobot”的参考 commit 是：

```text
ce3b9f627e55223d6d1c449d348c6b351b35d082
```

注意：**这是复现建议，不是论文声称使用的 commit。**

---

## 4. 数据：LIBERO 的哪部分

官方 LIBERO 提供四套 demonstration dataset。标准 release 通常是每任务 50 条 human teleoperation demonstrations，但 ARFM 论文**没有明确写主 Table 1 实际用了每任务多少条**。

因此主复现建议：

```text
Goal:    10 tasks × 50 demos
Spatial: 10 tasks × 50 demos
Object:  10 tasks × 50 demos
Long:    10 tasks × 50 demos
```

总计约 2000 trajectories。

但要在实验记录中明确标注：

> “50 demos/task 是依据 LIBERO 标准数据设置采用的复现假设，不是 ARFM 论文正文明确声明。”

### 4.1 每个训练样本至少需要存什么

ARFM 一个样本不是单 action，而是 action chunk：

$$
A_t=[a_t,a_{t+1},\dots,a_{t+H-1}],\qquad H=50.
$$

建议离线预处理后每个 chunk 存：

```python
{
    "obs": ...,                  # 图像、语言、proprio
    "actions": A_t,             # [H, action_dim]
    "action_is_pad": ...,       # [H]
    "task_id": ...,
    "suite_id": ...,
    "episode_id": ...,
    "t": ...,
    "dense_reward": r_t,
    "rtg": G_t,
    "advantage": R_star,
}
```

reward / RTG / advantage **建议预计算并落盘**，不要每次训练动态计算。这样才能保证 ARFM、RWR、π0 使用完全相同的数据。

---

## 5. Dense reward：这是复现中最不清楚的一部分

### 5.1 论文明确给出的 13 个分量

arXiv 附录 Table 8 给出了四类 dense reward。

| 类别 | 分量 | 论文形式 | 权重 |
|---|---|---|---:|
| Sub-goal Achievement | Image MSE | $\exp(f_{MSE}(o_t,o_t^*))$ | $0.1/13$ |
| | Image SSIM | $\exp(f_{SSIM}(o_t,o_t^*))$ | $0.1/13$ |
| | Image ORB | $\exp(f_{ORB}(o_t,o_t^*))$ | $0.1/13$ |
| | Gripper Image MSE | 同上 | $0.1/13$ |
| | Gripper Image SSIM | 同上 | $0.1/13$ |
| | Gripper Image ORB | 同上 | $0.1/13$ |
| | Joint Position MSE | $\exp(f_{MSE}(s_t,s_t^*))$ | $0.1/13$ |
| Task Progress | Sub-goal Division | $n(s_t)/|\{s^*\}|$ | $0.1/13$ |
| Behavior Smoothness | Joint Velocity | $-\|\dot q\|^2$ | $0.1/13$ |
| | Joint Acceleration | $-\|\ddot q\|^2$ | $0.1/13$ |
| | Action Velocity | $-\|a_{t-1}-a_t\|^2$ | $0.01/13$ |
| | Action Acceleration | $-\|a_{t-2}-2a_{t-1}+a_t\|^2$ | $0.01/13$ |
| Task Completion | Success | $\mathbb I\{\tau\text{ successful}\}$ | $0.1/13$ |

### 5.2 论文没有交代清楚的 reward 细节

这是**Critical** 级别的不确定性。论文没有完整定义：

- `f_MSE` 的负号和尺度；
- `f_SSIM` 的尺度；
- `f_ORB` 的尺度与匹配算法；
- `o_t^*` / `s_t^*` 如何选；
- subgoal 如何自动发现；
- `n(s_t)` 如何判定已经完成几个 subgoal；
- 两个 camera view 的 resize / crop / normalization 是否在 reward 前进行；
- smoothness 项是对原始动作、归一化动作还是机器人控制空间动作计算；
- terminal success reward 是每 timestep 都加，还是只在 terminal timestep 加。

因此，**仅凭 ARFM 论文无法唯一恢复 dense reward。**

### 5.3 可用的 ReinboT 参考实现

ARFM 明确说 reward design 跟随 ReinboT。ReinboT 的公开代码给出了一个很有价值的参考：

```python
r_mse  = exp(-0.01 * MSE)
r_ssim = exp(SSIM - 1.0)
r_orb  = exp(ORB_similarity - 1.0)
```

其 ORB 参考实现大致使用：

```text
cv2.ORB_create(edgeThreshold=0, fastThreshold=40)
BFMatcher(NORM_HAMMING, crossCheck=True)
ORB similarity = #matches / min(#kp1, #kp2)
```

ReinboT 还使用基于 gripper change / robot stopping / episode end 的 heuristic keypoint discovery 来构造 subgoal。

**但是不要把 ReinboT reward_calculation.py 原样复制成 ARFM reward。** ARFM Table 8 的 13 项和 ReinboT 开源脚本中的 reward vector 并不完全相同。正确做法是：

> 用 ReinboT 代码补齐 `f_MSE / f_SSIM / f_ORB / subgoal discovery` 的缺失实现，但 reward 项与权重仍以 ARFM Table 8 为准。

### 5.4 建议的 reward 复现优先级

建议先做三个版本：

```text
Reward-A: ARFM Table 8 + ReinboT similarity/subgoal implementation
Reward-B: 仅 task success + subgoal progress（简化）
Reward-C: LIBERO 原生 sparse success reward
```

主复现报告用 Reward-A；B/C 是诊断 ablation。

如果 A 能提升而 C 不能，说明 ARFM 对 dense quality signal 很敏感；这本身就是有意义的复现结果。

---

## 6. RTG：论文最终版新增了一个关键说明

AAAI 最终版写明：使用 **undiscounted Return-To-Go**，并且 action chunk 使用其**第一个 action 对应的 RTG**，同时 RTG 要除以 remaining timesteps。

最合理的实现解释是，对 trajectory

$$
\tau=(s_0,a_0,r_0,\dots,s_{T-1},a_{T-1},r_{T-1})
$$

定义

$$
G_t
=
\frac{1}{T-t}
\sum_{k=t}^{T-1}r_k.
$$

然后 chunk

$$
A_t=[a_t,\dots,a_{t+H-1}]
$$

整体使用标签 $G_t$。

也就是：

```python
rtg[t] = rewards[t:].sum() / (T - t)
chunk_rtg[t] = rtg[t]
```

这比把 chunk 内 50 个 action 各自分配不同 RTG 更符合论文“corresponding to the first action”的表述。

### 6.1 仍存在的歧义

论文原句也可以被较弱地解释为“整条 trajectory 的 action chunks 分享一个归一化 RTG”。但这会损失 timestep-level quality variation，与 ReinboT 的 fine-grained RTG 动机不太一致。

因此本报告推荐上述 **per-chunk-start average RTG** 实现，并把另一种解释作为 ablation，而不是主实现。

---

## 7. Advantage：不要随手换成 critic

ARFM 的特点之一就是 **critic-free**。论文使用 Leave-One-Out (LOO) 形式：

$$
R^*(c,x_k)
=
R(c,x_k)
-
\frac{1}{K-1}\sum_{i\ne k}R(c,x_i),
$$

等价于

$$
R^*(c,x_k)
=
\frac{K}{K-1}
\left(
R(c,x_k)-\frac1K\sum_{i=1}^KR(c,x_i)
\right).
$$

### 7.1 “standardized” 不等于明确的 z-score

论文反复称 $R^*$ 为 “standardized RL advantage”，但给出的 Eq. 4 实际只是 Leave-One-Out centering/scaling，并**没有明确写**

$$
(R-\mu)/\sigma.
$$

而且 Algorithm 1 仍然计算 $\sigma_R$。如果已经强制 z-score 到方差 1，这一步就显得多余。

因此主复现建议：

```python
adv = K / (K - 1) * (rtg - rtg.mean())
```

**不要额外做除以 std 的 z-score**，除非作为 ablation。

### 7.2 最大的不清楚点：K 和 group 到底怎么定义

最终论文说：

> advantage is computed per task category; K depends on the number of tasks, not a hyperparameter.

但没有给出可直接编码的数据索引定义。

这里至少存在三种可能：

```text
解释 A：同一个具体 task 内，不同 trajectories 做 LOO
解释 B：同一个 suite/category 内，不同 task 做 LOO
解释 C：同一个 task category 内，对 trajectory/chunk sample 做 LOO
```

论文又强调“trajectory-level quality comparison”，这使 C/A 比纯 task-level B 更自然；但“`K` depends on the number of tasks” 又倾向 B。

这是**Critical 未决项**。

### 7.3 我建议的 canonical 实现

为了既保留 trajectory quality，又避免跨语义任务 reward scale 混用：

1. 先在**每个具体 LIBERO task 内**计算 trajectory/chunk RTG；
2. 对该 task 的所有 trajectory RTG 做 LOO；
3. 得到 chunk-level advantage；
4. batch sampler 对 40 个 tasks 做均衡采样。

即：

```python
for task_id in tasks:
    G = all_chunk_rtg_of_this_task
    K = len(G)
    R_star = K / (K - 1) * (G - G.mean())
```

这不是论文唯一确定的解释，但它在统计意义上最合理、实现最稳定。为了追数值，至少还要再跑一个 “suite-level LOO” ablation。

---

## 8. Flow-matching convention：论文和 LeRobot 不要混着改

### 8.1 AAAI 最终版论文

最终版使用

$$
A_t^\tau=\tau A_t+(1-\tau)\epsilon,
$$

$$
u=A_t-\epsilon,
$$

并从 $\tau=0$ 的 noise 积分到 $\tau=1$ 的 action。

### 8.2 LeRobot π0 的实现约定

论文提交前的 LeRobot π0 使用相反时间方向：

```python
x_t = t * noise + (1 - t) * actions
target = noise - actions
```

即：

- `t=1` 是 noise；
- `t=0` 是 action；
- inference 用负 `dt` 从 1 积分到 0。

两者通过

$$
\tau=1-t
$$

完全对应。

### 8.3 复现规则

**不要为了“看起来和论文公式一致”而只改 target 的符号。**

如果你使用 LeRobot π0，就保留 LeRobot 的整套 flow convention：

```python
x_t = t * noise + (1 - t) * actions
u_t = noise - actions
```

ARFM 只应该改变**per-sample loss 的权重**。

否则极容易做出：

```text
interpolation 用 LeRobot
velocity target 用论文
inference integration 仍用 LeRobot
```

这种符号不一致的错误。

### 8.4 arXiv v1 与 AAAI 最终版还有一次符号修正

arXiv v1 曾写过 $u=\epsilon-A_t$，AAAI 最终版改成 $u=A_t-\epsilon$。这进一步说明复现时应该以**最终 AAAI 公式 + 代码自身 convention 的整体一致性**为准，而不是逐行照搬 arXiv v1。

---

## 9. tau 的采样：论文与 LeRobot 的直接矛盾

Algorithm 2 明确写：

$$
\tau\sim \operatorname{Uniform}(0,1).
$$

但论文基于 LeRobot，而 2025 年论文提交前 LeRobot π0 已经使用：

$$
t\sim \operatorname{Beta}(1.5,1.0),
$$

随后

$$
t\leftarrow 0.999t+0.001.
$$

论文没有说明作者是否 override 了 LeRobot 默认 sampler。

因此建议做两条复现线：

### Paper-Literal

```python
t = torch.rand(B, device=device)
```

### LeRobot-Likely

```python
beta = Beta(1.5, 1.0)
t = beta.sample((B,)).to(device)
t = 0.999 * t + 0.001
```

如果目标是“算法符合论文”，优先报告 Uniform；如果目标是“追作者 Table 1 数字”，**Beta 版本很值得优先测试**。

---

## 10. Per-sample FM loss：ARFM 集成的真正改动点

历史 LeRobot π0 的 core model 已经返回 element-wise MSE：

```python
losses = F.mse_loss(u_t, v_t, reduction="none")
# [B, H, D]
```

vanilla policy 最后做

```python
loss = losses.mean()
```

ARFM 需要把它改成：

```python
per_sample_loss = losses.mean(dim=(1, 2))  # [B]
alpha = solve_alpha(advantage, per_sample_loss.detach())
weights = torch.softmax(alpha * advantage.detach(), dim=0)
loss = torch.sum(weights * per_sample_loss)
```

### 10.1 为什么推荐 `mean` 而不是数学公式里的 `||.||²` 求和

论文公式写的是 $\|\cdot\|^2$，严格来说可以理解为对 action chunk 全维求和。

但：

- 原始 LeRobot 最终使用全元素 mean；
- ARFM 的 $\sigma_L^2$ 会强烈依赖 loss scaling；
- $\lambda=5\times10^{-4}$ 也因此依赖你采用 sum 还是 mean。

所以为了和 π0 baseline 的 loss scale 一致，建议：

$$
L_i^{FM}
=
\operatorname{mean}_{h,d}\bigl[(u-v)^2\bigr].
$$

这是一个**非常重要但论文没写清楚的细节**。

### 10.2 padding 的处理

必须沿用你所冻结版本的 π0 baseline 行为。如果历史 LeRobot 是先把 padded timestep loss 置零、再对固定 $H\times D$ mean，那么 ARFM 的 per-sample loss 也应该使用同样 reduction，否则你已经改变了 base objective。

建议单元测试：当所有 advantage 都是 0 时，ARFM loss 必须和 vanilla π0 loss 数值一致到浮点误差。

---

## 11. 正确实现 alpha 二分法

### 11.1 论文推导

论文明确推导：

$$
x=\alpha^2\sigma_R^2.
$$

因此如果

$$
\alpha\in[\alpha_{min},\alpha_{max}],
$$

数学上一致的区间应为

$$
x_{low}=\alpha_{min}^2\sigma_R^2,
\qquad
x_{high}=\alpha_{max}^2\sigma_R^2.
$$

### 11.2 Algorithm 1 疑似笔误

论文 Algorithm 1 印成了

$$
x_{low}=\sigma_R^2\alpha_{min},
\quad
x_{high}=\sigma_R^2\alpha_{max},
$$

少了 $\alpha^2$。

而正文 Corollary 2 / 附录推导明确写 $x=\alpha^2\sigma_R^2$。

因此主复现建议使用**推导一致版本：平方上下界**。

Algorithm 1 还有一个符号笔误：恢复 $\alpha$ 时写了 $\sigma_A$，但全文对应量应该是 $\sigma_R$。

`clip(alpha*, alpha_max, alpha_min)` 的参数顺序也不符合常规 clamp 记法，实际应理解为：

```python
alpha = clamp(alpha, alpha_min, alpha_max)
```

### 11.3 推荐实现

```python
import math
import torch

@torch.no_grad()
def solve_arfm_alpha(
    advantage: torch.Tensor,
    per_sample_loss: torch.Tensor,
    lam: float = 5e-4,
    alpha_min: float = 0.01,
    alpha_max: float = 5.0,
    n_iter: int = 20,
    tol: float = 1e-5,
    eps: float = 1e-12,
) -> torch.Tensor:
    # 用 float64 做二分，避免 exp(2x) 数值误差
    r = advantage.detach().double()
    l = per_sample_loss.detach().double()

    # 论文假定 R* 已经 zero-mean，因此使用 E[R^2]
    sigma_r2 = torch.mean(r.square())
    sigma_r = torch.sqrt(torch.clamp(sigma_r2, min=eps))

    mu_l = torch.mean(l)
    sigma_l2 = torch.mean((l - mu_l).square())
    sigma_l2 = torch.clamp(sigma_l2, min=eps)

    if sigma_r2.item() < eps:
        return torch.tensor(alpha_min, device=advantage.device,
                            dtype=advantage.dtype)

    c = lam * sigma_r / sigma_l2

    def F(x):
        sx = torch.sqrt(torch.clamp(x, min=0.0))
        return 4.0 * sx * torch.exp(2.0 * x) \
             - 2.0 * sx * torch.exp(x) - c

    # 与 x = alpha^2 * sigma_R^2 一致
    x_lo = sigma_r2 * (alpha_min ** 2)
    x_hi = sigma_r2 * (alpha_max ** 2)

    f_lo = F(x_lo)
    f_hi = F(x_hi)

    # 论文没有说明 bracket 不住根时怎么办；这里做边界 fallback
    if f_lo >= 0:
        alpha = torch.tensor(alpha_min, dtype=torch.float64,
                             device=r.device)
    elif f_hi <= 0:
        alpha = torch.tensor(alpha_max, dtype=torch.float64,
                             device=r.device)
    else:
        for _ in range(n_iter):
            x_mid = 0.5 * (x_lo + x_hi)
            f_mid = F(x_mid)
            if torch.abs(f_mid) < tol:
                x_lo = x_hi = x_mid
                break
            if f_mid > 0:
                x_hi = x_mid
            else:
                x_lo = x_mid

        x_star = 0.5 * (x_lo + x_hi)
        alpha = torch.sqrt(x_star) / sigma_r
        alpha = torch.clamp(alpha, alpha_min, alpha_max)

    return alpha.to(device=advantage.device, dtype=advantage.dtype)
```

### 11.4 为什么 `alpha` 计算必须 detach

Algorithm 1 的流程是：

```text
先根据当前 batch statistics 求 alpha*
再固定 alpha* 做一次 gradient step
```

所以不要让 autograd 通过 $\sigma_L^2\to\alpha\to L$ 反传。

论文没有显式写 `stop_gradient(alpha)`，但从算法定义看这是最自然的实现。

---

## 12. Algorithm 2 的 double-exponential 明显不一致

正文明确给出：

$$
w_i
=
\operatorname{softmax}_i(\alpha R_i^*).
$$

但 Algorithm 2 又定义

$$
g_i=\exp(R_i^*)
$$

并随后写

$$
w_i
\propto
\exp(\alpha g_i),
$$

这实际上变成

$$
\exp\bigl(\alpha\exp(R_i^*)\bigr),
$$

即**二次指数**。

这和：

- 正文 policy distribution；
- practical loss；
- 后续 $\hat w_i=\exp(\alpha R_i^*)$ 推导；
- 高斯 moment derivation

全部不一致。

因此应该把 Algorithm 2 第 5/9 行视为伪代码笔误，主实现使用：

```python
weights = torch.softmax(alpha * advantage, dim=0)
```

而不是：

```python
weights = torch.softmax(alpha * torch.exp(advantage), dim=0)
```

建议把 double-exponential 版本只保留成一个“论文伪代码 literal ablation”，不要作为主复现。

---

## 13. 完整 ARFM training step

推荐主训练逻辑：

```python
# batch:
# obs, action chunk A, advantage R_star, masks

# 1. π0 原始 flow noise/time sampling
noise = model.sample_noise(actions.shape, actions.device)
time = model.sample_time(batch_size, actions.device)

# 2. 得到 element-wise FM loss，不做 batch reduction
loss_elem = model.forward_core(
    images, masks, language, state,
    actions, noise, time,
)  # [B, H, D]

# 3. 沿用 baseline 的 padding/action-dim mask
loss_elem = apply_same_mask_as_vanilla_pi0(loss_elem, batch)

# 4. 每个 sample 一个 scalar FM loss
per_sample_fm = loss_elem.mean(dim=(1, 2))   # [B]

# 5. 从离线数据中读取预计算的 R_star
adv = batch["advantage"].to(per_sample_fm.device)

# 6. 自适应 alpha
alpha = solve_arfm_alpha(
    adv,
    per_sample_fm,
    lam=5e-4,
    alpha_min=0.01,
    alpha_max=5.0,
    n_iter=20,
    tol=1e-5,
)

# 7. 正文一致的权重（不是 double exponential）
weights = torch.softmax(alpha * adv.detach(), dim=0)

# 8. ARFM loss
loss = torch.sum(weights * per_sample_fm)

# 9. optimizer
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
optimizer.step()
scheduler.step()
```

### 13.1 必须记录的 diagnostics

每一步或每 50/100 step 建议记录：

```text
train/loss
train/fm_loss_mean
train/fm_loss_std
train/adv_mean
train/adv_std
train/alpha
train/weight_max
train/weight_min
train/weight_entropy
train/effective_sample_size
train/grad_norm
```

其中

$$
ESS=\frac{1}{\sum_iw_i^2}.
$$

当 batch size = 16：

- ESS≈16：几乎等权；
- ESS≈1：一个样本垄断 batch，危险；
- ARFM 的目的就是避免长期落到极小 ESS。

---

## 14. 论文给出的训练超参

arXiv 附录 Table 7 给出：

| 参数 | 论文值 | 备注 |
|---|---:|---|
| LIBERO post-training steps | 40,000 | 明确 |
| Batch size | 16 | 明确 |
| Action horizon | 50 | 明确 |
| $\lambda$ | $5\times10^{-4}$ | 明确 |
| Bisection iterations $M$ | 20 | 明确 |
| $[\alpha_{min},\alpha_{max}]$ | [0.01, 5] | 明确 |
| Bisection tolerance | $10^{-5}$ | 明确 |
| Optimizer | AdamW | 明确 |
| “Learning Rate” | $10^{-4}$ | **与 peak LR 冲突/关系不明** |
| Adam $\beta$ | (0.9, 0.95) | 明确 |
| Adam epsilon | $10^{-8}$ | 明确 |
| Weight decay | $10^{-10}$ | 明确 |
| Gradient clip norm | 10 | 明确 |
| Scheduler | cosine decay + warmup | 明确 |
| Warmup steps | 1,000 | 明确 |
| Decay steps | 30,000 | 明确 |
| Peak LR | $2.5\times10^{-5}$ | 明确但与 LR row 不一致 |
| Decay LR | $2.5\times10^{-6}$ | 明确 |
| Brightness | [0.8, 1.2] | 明确范围，概率/顺序不明 |
| Contrast | [0.8, 1.2] | 同上 |
| Saturation | [0.5, 1.5] | 同上 |
| Hue | [-0.05, 0.05] | 同上 |
| Sharpness | [0.5, 1.5] | 同上 |

### 14.1 学习率冲突怎么处理

推荐主复现采用：

```text
warmup 1000 steps -> peak 2.5e-5
cosine decay 30000 steps -> 2.5e-6
剩余 steps 保持 2.5e-6
```

理由：`Peak Learning Rate` 是对 scheduler 实际最大值最直接的描述。

同时保留一个对照：

```text
max lr = 1e-4
```

如果你的 baseline 明显低于论文，再测试它。

**不要把“optimizer lr=1e-4”和“peak lr=2.5e-5”同时模糊地配置后假装已经复现。** 应在实验表中明确记录实际每一步 optimizer param_group 的 LR。

---

## 15. 论文硬件与耗时

论文 arXiv Code Appendix：

```text
2 × NVIDIA A100-SXM4-80GB
Intel Xeon Platinum 8358 @ 2.60 GHz
LIBERO: 40k full-parameter fine-tuning steps
约 11 小时
```

所以原实验不是 LoRA，而是 **full-parameter fine-tuning**。

如果你只有 4090 服务器：

- 可以做 gradient checkpointing；
- BF16；
- gradient accumulation 保持 **effective batch size 16**；
- 必要时 FSDP / ZeRO；
- 但不要为了省显存直接改成 LoRA 后仍称为“ARFM 原论文复现”。

LoRA 应单独标为 extension。

---

## 16. Batch sampler：论文没有写，但对 ARFM 很重要

因为权重是**batch 内 softmax**：

$$
w_i
=
\frac{e^{\alpha R_i^*}}{\sum_j e^{\alpha R_j^*}},
$$

所以 batch composition 会直接改变 objective。

如果一个 batch 全来自 Long，另一个 batch 混 Goal/Object/Spatial，权重含义完全不同。

论文没有说明 batch 是：

- random across all 40 tasks；
- suite-balanced；
- task-balanced；
- same-task batch。

建议主复现采用 **task-balanced mixed batch**，例如 batch=16 时先均匀抽 task，再从 task 内抽 chunk；并确保 advantage 在 task 内已中心化。

还应做一个 same-task batch ablation，因为 Eq. 4 的原始 LOO 语义更接近条件 $c$ 固定时比较样本。

---

## 17. Optimizer 与 scheduler 的推荐配置

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2.5e-5,        # 主实现把 peak LR 当真实 max LR
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=1e-10,
)
```

scheduler：

```text
step 0 → 1000: linear warmup to 2.5e-5
step 1000 → 31000: cosine decay to 2.5e-6
step 31000 → 40000: hold 2.5e-6
```

论文没有说明 decay steps 是否包含 warmup，也没有说明最后 9k step 的行为。上面是最可解释的 canonical choice。

---

## 18. 训练前必须先做的 6 个单元测试

### Test 1：alpha = 0 退化一致性

手动设置 $\alpha=0$：

```python
weights = [1/B, ..., 1/B]
```

应满足：

```text
ARFM weighted loss == vanilla π0 mean FM loss
```

### Test 2：advantage shift

因为 softmax 对整体平移不变：

$$
\operatorname{softmax}(\alpha(R+c))
=
\operatorname{softmax}(\alpha R),
$$

给所有 advantage 加同一个常数，weights 应不变。

### Test 3：advantage ordering

若

$$
R_1^*>R_2^*,
$$

且 $\alpha>0$，应有

$$
w_1>w_2.
$$

### Test 4：alpha solver residual

检查

```python
abs(F(alpha**2 * sigma_r**2)) < tolerance
```

或者根超出允许范围时落在边界。

### Test 5：flow sign consistency

固定 noise 和 action，用 LeRobot baseline 与 ARFM wrapper 在所有 advantage=0 时比较 elementwise loss，必须完全一致。

### Test 6：padding consistency

同一个真实 action chunk 加不同 dummy padded dimensions，不应该改变有效 action 部分的 loss 逻辑。

---

## 19. 训练时建议分三步，而不是直接跑 40k

### Phase 0：vanilla baseline 验证

先用完全相同 pipeline 训练/评估 π0：

```text
reward 不参与
advantage 不参与
ARFM weighting 不参与
```

如果 π0 baseline 和论文的 88.1% 差十几二十个百分点，不要继续怀疑 ARFM。

### Phase 1：固定 alpha 的 RWR-style sanity check

运行：

```text
alpha = 0.01
alpha = 0.1
alpha = 0.5
alpha = 1.0
```

确认高 advantage 样本能够得到更高权重，而且训练不会崩。

### Phase 2：ARFM adaptive alpha

启用二分法，先跑 1k–5k steps 看：

```text
alpha distribution
ESS
loss variance
grad norm
success rate
```

稳定后再做完整 40k。

---

## 20. Evaluation：论文明确和未明确的地方

### 20.1 Main benchmark

目标是 40 个 LIBERO tasks 的 success rate。

论文给出 suite-level SR，但**没有明确写：**

- 每 task rollout 几次；
- 随机种子数量；
- 是否用固定 LIBERO init states 的前 N 个；
- checkpoint selection 方法；
- 是最后 checkpoint 还是 best checkpoint；
- inference flow integration steps；
- action chunk execution horizon。

因此必须在你的报告里完整写自己的协议。例如：

```text
50 rollouts / task
固定 init-state protocol
3 random seeds
最后 checkpoint + 同时报告 best validation checkpoint
```

不能只写一个成功率数字。

### 20.2 Action perturbation

论文在 inference action 上加 Gaussian noise，noise level：

$$
0.1,0.15,0.2,0.25,0.3.
$$

论文 Table 2 的平均值：

| Model | Average SR under action noise |
|---|---:|
| π0 | 43.3 |
| ReinboT | 46.3 |
| RWR | 46.4 |
| ARFM | 48.2 |

这是很有价值的 secondary target，因为 ARFM 的论文主张之一就是 robustness。

但论文没有说 Gaussian noise 是加在 normalization 前还是后，也没有完整说明各 action dimension 是否同尺度；这也是一个复现缺口。

---

## 21. λ 和 M 的 ablation 可以作为 solver 正确性的检查

论文 Figure 4：

- $\lambda$ 测了大致 `0.05, 0.01, 0.005, 0.0005, 0.0001`；
- $M$ 测了 `5, 10, 20, 30`；
- $M\ge10$ 后性能基本稳定；
- 论文主配置 $M=20$；
- 主配置 $\lambda=5e-4$。

如果你的实现中 $M=10$ 和 $M=20$ 差异巨大，优先怀疑：

```text
x bounds
F(x)
loss scale
sigma_L^2
root bracketing
```

而不是认为论文结论失效。

---

## 22. 论文理论与实际 normalized weights 之间还有一个值得注意的缝隙

实际训练 loss 使用：

$$
w_i
=
\frac{e^{\alpha R_i^*}}
{\sum_j e^{\alpha R_j^*}}.
$$

但推导 gradient variance 时定义的是未归一化权重：

$$
\hat w_i=e^{\alpha R_i^*}.
$$

这使得 Eq. (7) 严格来说不是“实际 normalized batch loss 的精确 gradient variance”的闭式表达，而是一个近似/代理推导。

另外，推导把 advantage distribution 与 CFM loss/gradient 的统计关系简化得很强；论文没有显式处理它们之间的相关性。

**复现时不要自行“修理论”。** 正确策略是：

1. 按论文求 $\alpha$；
2. 实际训练仍用 normalized softmax weights；
3. 额外记录真实 batch 的 weight entropy、ESS、grad norm；
4. 如果写复现报告，可以把该理论近似作为 limitation。

---

## 23. 论文没有说明的数值稳定性问题

### 23.1 $\sigma_L^2 \approx 0$

$F(x)$ 包含

$$
\frac{\lambda\sigma_R}{\sigma_L^2}.
$$

当一个 batch 的 FM loss 很接近时会爆。

建议：

```python
sigma_l2 = max(sigma_l2, 1e-12)
```

并 log fallback 次数。

### 23.2 $\sigma_R\approx0$

说明 batch 中所有样本 advantage 近似相同，没有 RL preference。

推荐：

```python
alpha = alpha_min
```

此时 weights 几乎均匀。

### 23.3 bisection interval 不 bracket root

论文没有规定。

推荐：

```text
F(x_low) >= 0 -> alpha_min
F(x_high) <= 0 -> alpha_max
否则二分
```

### 23.4 `exp(2x)` overflow

用 float64 求 solver；必要时对 x 做合理上界保护。不要用 BF16/FP16 求 $F(x)$。

---

## 24. 论文交代不清/矛盾项总表

| 项目 | 状态 | 严重度 | 推荐处理 |
|---|---|---|---|
| ARFM core weighted loss | 清楚 | — | `softmax(alpha * R*)` |
| double exponential in Alg.2 | **和正文矛盾** | Critical | 以正文/推导为准，不二次指数 |
| $x$ bisection bounds | **和 $x=\alpha^2\sigma_R^2$ 矛盾** | Critical | 使用 squared-alpha bounds |
| `$sigma_A` vs `$sigma_R` | 疑似 typo | High | 使用 $\sigma_R$ |
| clamp 参数顺序 | 表述异常 | Medium | clamp to `[alpha_min, alpha_max]` |
| flow velocity sign | arXiv v1 与最终版不同 | High | 使用最终版；LeRobot 保持自身完整 convention |
| flow time sampler | Paper Uniform vs LeRobot Beta | **Critical** | 两种都跑，严格区分 |
| dense reward exact functions | **未完整公开** | Critical | Table8 + ReinboT reference reconstruction |
| subgoal extraction | **未公开 ARFM 具体实现** | Critical | 参考 ReinboT heuristic，并做 ablation |
| RTG 具体 chunk mapping | 有描述但仍可歧义 | High | chunk start 的 average future RTG |
| advantage “standardized” | 含义模糊 | High | Eq4 LOO，不额外 z-score；zscore 做 ablation |
| `K` / grouping | **不清楚** | Critical | task-level LOO 主实现 + suite-level ablation |
| per-sample loss sum/mean | 未写 | Critical | 沿用 LeRobot mean scale |
| task/batch sampler | 未写 | High | task-balanced，另做 same-task batch ablation |
| π0 exact checkpoint | 未写 | Critical for exact SR | 完整记录你使用 checkpoint |
| LeRobot commit | 未写 | Critical for exact SR | pin commit |
| LIBERO demos/task | 主实验未明说 | High | 默认标准 50 demos/task，明确是复现假设 |
| action normalization | 未写 | High | 沿用 frozen LeRobot pipeline |
| LR 1e-4 vs peak 2.5e-5 | **表内冲突/关系不明** | Critical | 2.5e-5 主跑，1e-4 对照 |
| 30k decay 后剩余 9k LR | 未写 | Medium | hold floor LR |
| augmentation probability/order | 未写 | Medium | 固定标准 ColorJitter 实现并记录 |
| alpha 求解的 epsilon/fallback | 未写 | High | 自己加数值保护并报告 |
| 是否对 alpha 反传 | 未写 | High | detach/no_grad |
| eval episodes/seeds | 未写 | Critical for exact SR | 自行固定并完整报告 |
| inference denoise steps | 未写在 ARFM 实验设置 | High | 沿用冻结 π0 baseline |
| official ARFM code | 截至本报告未找到 | Critical for exact replication | 只能论文复现 |

---

## 25. 我建议的“canonical reproduction config”

如果你现在直接开工，我会冻结成：

```yaml
experiment: arfm_libero_pi0_reproduction

model:
  backbone: pi0
  implementation: lerobot
  full_finetune: true
  action_horizon: 50

training:
  steps: 40000
  batch_size: 16
  grad_clip_norm: 10
  precision: bf16

optimizer:
  name: adamw
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 1.0e-10

scheduler:
  type: cosine_with_warmup
  warmup_steps: 1000
  decay_steps: 30000
  peak_lr: 2.5e-5
  floor_lr: 2.5e-6
  after_decay: hold_floor

arfm:
  lambda: 5.0e-4
  alpha_min: 0.01
  alpha_max: 5.0
  bisection_steps: 20
  bisection_tol: 1.0e-5
  weight_formula: softmax(alpha * advantage)
  x_bounds: squared_alpha   # 按正文推导，不按 Alg.1 疑似笔误
  detach_alpha_solver: true

advantage:
  return: undiscounted_average_rtg
  chunk_label: rtg_at_chunk_first_action
  estimator: leave_one_out
  zscore: false
  primary_grouping: per_task

flow_time:
  primary: uniform          # paper literal
  ablation: beta_1.5_1.0   # likely LeRobot behavior

data:
  suites: [goal, spatial, object, long]
  demos_per_task: 50       # LIBERO standard; NOT explicitly confirmed by ARFM paper

reward:
  primary: arfm_table8_with_reinbot_similarity_reference
```

为了追论文数值，我会把 `flow_time=beta_1.5_1.0` 也当作一级实验，而不是低优先级 ablation。

---

## 26. 推荐的实验矩阵

不要一次只跑一个 ARFM，然后成败全押在一个隐含实现选择上。

### Stage A：确认 baseline

| ID | Method | Reward | Time sampler | Purpose |
|---|---|---|---|---|
| A0 | π0 | none | LeRobot default | baseline |
| A1 | π0 | none | Uniform | 检查 timestep sampler 影响 |

### Stage B：确认 weighting

| ID | Method | alpha | Advantage | Purpose |
|---|---:|---|---|
| B0 | RWR-like | fixed 0.1 | LOO | weighting sanity |
| B1 | RWR-like | fixed 0.5 | LOO | weighting strength |
| B2 | ARFM | adaptive | LOO | core method |

### Stage C：解决论文歧义

| ID | Variant | Change |
|---|---|---|
| C0 | Paper-Literal | Uniform time |
| C1 | LeRobot-Likely | Beta(1.5,1.0) time |
| C2 | LOO-per-task | primary |
| C3 | LOO-per-suite | grouping ablation |
| C4 | no-zscore | primary |
| C5 | zscore | standardization ablation |
| C6 | lr=2.5e-5 | primary |
| C7 | lr=1e-4 | LR ambiguity |

这样做完，即使没有完全复现 92.1%，你也能非常明确地定位差异来源。



## 28. “复现成功”应输出哪些表和图

建议输出：

1. 四个 LIBERO suite 的 SR；
2. π0 vs fixed-RWR vs ARFM；
3. $\alpha$ 随 step 的均值/分布；
4. ESS / max weight 随 step；
5. advantage histogram；
6. per-sample FM loss histogram；
7. action noise robustness；
8. Uniform vs Beta timestep ablation；
9. per-task vs per-suite advantage grouping ablation；
10. 2.5e-5 vs 1e-4 LR ablation。

## 31. 本报告对“论文交代程度”的最终判断

### 可以直接根据论文实现的部分

ARFM 的主要思想、energy-weighted flow loss、adaptive $\alpha$ 目标、非线性求根方程、二分迭代、主要训练步数、batch size、action horizon、$\lambda$、$M$、$\alpha$ 范围、optimizer 大部分设置，都已经足够明确。

### 需要根据公式纠正伪代码的部分

最明显的是：

```text
Algorithm 2 double exponential
Algorithm 1 x bounds
sigma_A / sigma_R
clip 参数顺序
```

这些如果 literal implementation，反而很可能不是作者真正的 intended method。

### 目前无法仅从论文唯一恢复的部分

最关键的是：

```text
exact dense reward implementation
subgoal construction
advantage grouping/K
actual flow-time sampler
exact LeRobot version/checkpoint
loss reduction scale
exact LR schedule semantics
batch composition
evaluation protocol
```

这些会阻止“bit-for-bit / number-for-number”复现，但不阻止一个严谨的 ARFM independent reproduction。

---

# References

1. Zhang, H. et al. **Balancing Signal and Variance: Adaptive Offline RL Post-Training for VLA Flow Models.** AAAI 2026.  
   https://ojs.aaai.org/index.php/AAAI/article/view/38944

2. ARFM arXiv v1（包含 Data Appendix / Code Appendix / Table 7 / Table 8）.  
   https://arxiv.org/abs/2509.04063

3. Hugging Face LeRobot.  
   https://github.com/huggingface/lerobot

4. LeRobot π0 historical reference commit used in this report for code archaeology（不是论文声明版本）.  
   https://github.com/huggingface/lerobot/commit/ce3b9f627e55223d6d1c449d348c6b351b35d082

5. LIBERO official repository.  
   https://github.com/Lifelong-Robot-Learning/LIBERO

6. ReinboT public repository；本报告仅把它作为 ARFM 未公开 reward 细节的参考线索。  
   https://github.com/COST-97/reinboT

---

## 附：一句话实现摘要

如果已经有一个正确工作的 LeRobot π0 + LIBERO pipeline，ARFM 真正新增的训练代码可以概括为：

```python
per_sample_fm = vanilla_pi0_elementwise_fm_loss.mean((1, 2))
alpha = solve_alpha(advantage.detach(), per_sample_fm.detach())
w = softmax(alpha * advantage.detach())
loss = (w * per_sample_fm).sum()
```

**真正困难的不是这四行，而是把 reward → RTG → advantage，以及论文缺失的 grouping / timestep sampler / loss scale 复现正确。**
