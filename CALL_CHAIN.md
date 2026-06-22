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
