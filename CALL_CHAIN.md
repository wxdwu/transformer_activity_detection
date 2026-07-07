# 调用链与文件说明

本文档记录当前 `main3` 分支的事件驱动随机接入方案，以及相关性如何进入 Transformer。

## 1. 当前目标

当前场景从独立 Bernoulli 活跃切换为事件驱动活跃：

```text
外部事件中心 -> 空间影响范围内的用户被触发 -> 相关活跃用户发起随机接入
```

这里不再强制保证整体活跃率等于 `activity_prob=0.1`。`activity_prob` 只保留给 `activity_mode="independent"` 的 baseline 使用；事件驱动模式下，活跃率由事件数量、事件影响半径、触发概率和背景活跃概率共同决定。

## 2. 数据生成调用链

入口：

- `network/data.py`
- `ActivityDataGenerator.sample_batch(batch_size, return_raw=False)`

主要流程：

1. 在半径 `cell_radius_m` 的圆形区域中采样用户位置 `positions: [B,N,2]`
2. 根据用户位置计算基站距离，用于大尺度衰落和功率控制
3. 根据用户间距离生成高斯相关矩阵 `corr_matrix: [B,N,N]`
4. 根据 `activity_mode` 生成活跃标签
5. 生成导频、信道、噪声和接收信号
6. 构造 Transformer 输入 `x_b, x_y`
7. 返回 `x_b, x_y, label, corr_matrix`

事件驱动模式：

```python
ACTIVITY_MODE = "event"
EVENT_LAMBDA = 1.0
EVENT_MAX_COUNT = 3
EVENT_SIGMA_M = 120.0
EVENT_TRIGGER_PROB = 0.9
BACKGROUND_ACTIVITY_PROB = 0.005
CORRELATION_LENGTH_M = 0.0
```

事件数量：

```text
K_event ~ Poisson(EVENT_LAMBDA), clipped to EVENT_MAX_COUNT
```

事件中心：

```text
e_k 在小区圆形区域内均匀采样
```

事件对用户的影响：

```text
g_ik = exp(-||u_i - e_k||^2 / (2 * EVENT_SIGMA_M^2))
```

多事件触发概率：

```text
p_event_i = 1 - Π_k (1 - EVENT_TRIGGER_PROB * g_ik)
```

加入背景活跃：

```text
p_i = 1 - (1 - BACKGROUND_ACTIVITY_PROB) * (1 - p_event_i)
```

最终标签：

```text
a_i ~ Bernoulli(p_i)
```

## 3. 相关矩阵

当前相关矩阵由用户间距离的高斯核生成：

```text
corr_ij = exp(-||u_i - u_j||^2 / (2 * ell_corr^2))
corr_ii = 0
```

如果 `CORRELATION_LENGTH_M <= 0`，默认使用：

```text
ell_corr = sqrt(2) * EVENT_SIGMA_M
```

直觉：距离越近的用户越可能被同一个事件共同触发，因此相关性越强。

## 4. Transformer 调用链

入口：

- `network/model.py`
- `HeterogeneousTransformer.forward(x_b, x_y, corr_matrix=None)`

基础输入保持论文形式：

```text
x_b: [B,N,2Lp]
x_y: [B,2Lp^2]
```

相关性不再作为平均标量拼到 `x_b` 中，而是通过完整 `corr_matrix` 进入网络。

## 5. Attention Bias

开关：

```python
USE_CORRELATION_ATTENTION_BIAS = True
CORR_ATTN_INIT = 1.0
```

在用户 token 之间的 attention score 上加入相关性 bias：

```text
score_ij = q_i k_j / sqrt(d) + alpha * corr_ij
```

其中 `alpha` 是可学习参数，初始值由 `CORR_ATTN_INIT` 控制。

## 6. Logits Refinement

开关：

```python
USE_CORRELATION_LOGIT_REFINEMENT = True
CORR_REFINE_INIT = 0.5
```

decoder 输出 logits 后，使用归一化相关矩阵做一次图传播：

```text
logits_refined = logits + gamma * CorrNorm @ logits
```

其中 `gamma` 是可学习参数，初始值由 `CORR_REFINE_INIT` 控制。

## 7. 训练调用链

入口：

- `network/train.py`

默认保存路径：

```python
SAVE_DIR = ROOT / "checkpoint/event_corrmatrix_260622"
```

训练前向：

```python
logits, _ = model(batch["x_b"], batch["x_y"], batch.get("corr_matrix"))
```

事件驱动模式下活跃率不固定，因此 loss 使用当前 batch 的真实活跃比例作为加权 BCE 的先验：

```python
activity_prob=float(batch["label"].mean().item())
```

并在 `network/losses.py` 中做了极小值裁剪，避免全负样本 batch 导致权重退化。

## 8. 评估和对照

以下脚本也会把 `corr_matrix` 传给模型：

- `network/evaluate.py`
- `network/compare_active_indices.py`
- `CE_methods/compare.py`
- `CE_methods/test_camp_genie_data.py`
- `plot/plot_amp_nmse_vs_iter.py`

推荐消融：

```text
A: ACTIVITY_MODE="event", USE_CORRELATION_ATTENTION_BIAS=False, USE_CORRELATION_LOGIT_REFINEMENT=False
B: ACTIVITY_MODE="event", USE_CORRELATION_ATTENTION_BIAS=True,  USE_CORRELATION_LOGIT_REFINEMENT=False
C: ACTIVITY_MODE="event", USE_CORRELATION_ATTENTION_BIAS=False, USE_CORRELATION_LOGIT_REFINEMENT=True
D: ACTIVITY_MODE="event", USE_CORRELATION_ATTENTION_BIAS=True,  USE_CORRELATION_LOGIT_REFINEMENT=True
```

## 9. 260630 相关性路径改进

本次改进不改训练 loss，不新增辅助 loss，也不改默认学习率策略。主 loss 仍是：

```python
weighted_activity_loss(
    logits=logits,
    targets=batch["label"],
    activity_prob=float(batch["label"].mean().item()),
)
```

改动只发生在使用相关性矩阵的模型路径中，目标是在已有增益基础上减少弱相关用户之间的噪声传播，并让相关性先验更直接进入 device token 表示。

### 9.1 稀疏局部相关图

原始实现直接使用完整 dense `corr_matrix`。当前新增：

```python
CORR_TOPK = 24
CORR_THRESHOLD = 0.10
```

模型内部会先对相关矩阵做筛选：

```text
corr_ij < CORR_THRESHOLD 的边置零
每个用户只保留 top-k 个最相关邻居
```

筛选后的相关图同时用于 attention bias、feature mixer 和 logits refinement。这样可以避免远距离弱相关用户在 dense 图里持续传播噪声。

### 9.2 Centered Logits Refinement

原始 logits refinement 是：

```text
logits_refined = logits + gamma * CorrNorm @ logits
```

该形式会直接叠加邻居 logits。如果某个 batch 中 logits 整体偏正，可能提高 `p_mean` 和 PF。当前默认改为 centered refinement：

```python
CORR_REFINE_MODE = "centered"
CORR_REFINE_INIT = 0.5
```

对应公式：

```text
neighbor_logits = CorrNorm @ logits
sample_center = mean(logits)
logits_refined = logits + gamma * (neighbor_logits - sample_center)
```

直觉：只利用“邻居相对本样本平均水平更活跃/更不活跃”的信息，而不是把邻居 logits 的绝对值直接加上去。

旧模式仍可恢复：

```python
CORR_REFINE_MODE = "additive"
```

也保留了图扩散模式：

```python
CORR_REFINE_MODE = "diffusion"
```

### 9.3 Correlation Feature Mixer

当前新增一个轻量图消息传递模块，位置在 Transformer encoder 之后、decoder 之前：

```python
USE_CORRELATION_FEATURE_MIXER = True
CORR_FEATURE_MIX_INIT = 0.1
```

计算方式：

```text
neighbor_h = CorrNorm @ h_b
h_b = h_b + tanh(eta) * MLP(neighbor_h - h_b)
```

它让 device token 在输出检测前吸收局部相关邻居的表示差异。该模块只在相关性路径打开时启用。

### 9.4 对照实验开关

不加相关性的对照仍只需要关闭原来的两个开关：

```python
USE_CORRELATION_ATTENTION_BIAS = False
USE_CORRELATION_LOGIT_REFINEMENT = False
```

即使 `USE_CORRELATION_FEATURE_MIXER = True` 保持默认，`build_model_config` 和 `build_model_from_config` 也会在上述两个开关都为 False 时自动禁用 feature mixer，保证对照模型不使用相关性矩阵。

## 10. 260707 噪声生成口径修正

### 10.1 Measured SNR Noise

当前将 `NOISE_MODE = "snr"` 改为与 MATLAB `awgn(x, SNR, "measured")` 对齐的口径。数据生成时先计算无噪声接收矩阵：

```text
BH = BAH
```

其中 `BH` 的尺寸为 `[batch, pilot_len, num_antennas]`。随后在每个样本的 `[pilot_len, num_antennas]` 维度上测量平均接收信号功率：

```text
signal_power = mean(|BH|^2)
noise_var = signal_power * 10^(-SNR/10)
```

最后生成单位功率复高斯噪声并叠加：

```text
W ~ CN(0, noise_var)
Y = BH + W
```

### 10.2 旧噪声公式保留方式

旧版 `NOISE_MODE = "snr"` 使用的是大尺度接收功率近似：

```text
noise_var = sum_k(p_k g_k a_k) / pilot_len * 10^(-SNR/10)
```

该公式现在保留为：

```python
NOISE_MODE = "large_scale_snr"
```

因此当前推荐实验配置仍写：

```python
NOISE_MODE = "snr"
```

表示在无噪声接收信号 `BH` 上测量功率后再按 SNR 加噪。
