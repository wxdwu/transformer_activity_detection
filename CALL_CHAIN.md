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

## 11. 260707 Varying L 实验脚本

### 11.1 实验目的

当前新增 `network/run_varying_l.py`，用于自动跑导频长度变化实验。默认导频长度为：

```text
L = [4, 6, 8, ..., 30]
```

其他训练、数据、模型参数默认继承 `network/train.py` 的当前设置，只覆盖 `pilot_len` 和保存路径。

### 11.2 默认运行方式

默认运行 addcorr 和 noaddcorr 两组实验：

```bash
python network/run_varying_l.py
```

脚本会分别训练 addcorr/noaddcorr 下的每个 L 到 `epoch=100`，每个实验保存到：

```text
checkpoint/varying_L_260707/addcorr/Lxx/
```

其中 `Lxx` 表示具体导频长度，例如 `L04`、`L30`。
noaddcorr 对照会保存到：

```text
checkpoint/varying_L_260707/noaddcorr/Lxx/
```

### 11.3 输出结果

每个 L 的 epoch 100 最终 loss 会汇总到同一个 CSV，使用 `mode` 列区分 addcorr/noaddcorr：

```text
checkpoint/varying_L_260707/final_loss_by_L.csv
```

并绘制包含 addcorr/noaddcorr 两条曲线的折线图：

```text
checkpoint/varying_L_260707/final_loss_by_L.png
```

如果当前 Python 环境没有 `matplotlib`，脚本会自动生成无依赖 SVG 版本：

```text
checkpoint/varying_L_260707/final_loss_by_L.svg
```

图中横坐标为导频长度 `L`，范围 `[4, 30]`；纵坐标为最终 epoch 的训练 loss；两条折线分别表示 addcorr 和 noaddcorr。

### 11.4 对照和快速检查

如果只想单独跑一条曲线用于调试：

```bash
python network/run_varying_l.py --modes addcorr
python network/run_varying_l.py --modes noaddcorr
```

如果只想本地快速检查脚本是否能跑通：

```bash
python network/run_varying_l.py --pilot-lens 4,6 --epochs 1 --steps-per-epoch 1 --batch-size 2 --eval-batches 1 --device cpu
```

## 12. 260707 Varying SNR 实验脚本

### 12.1 实验目的

当前新增 `network/run_varing_SNR.py`，用于自动跑信噪比变化实验。默认 SNR 为：

```text
SNR = [10, 12, 14, 16, 18, 20] dB
```

其他训练、数据、模型参数默认继承 `network/train.py` 的当前设置，只覆盖 `snr_db` 和保存路径。

### 12.2 默认运行方式

默认运行 addcorr 和 noaddcorr 两组实验：

```bash
python network/run_varing_SNR.py
```

脚本会分别训练 addcorr/noaddcorr 下的每个 SNR 到 `epoch=100`，每个实验保存到：

```text
checkpoint/varying_SNR_260707/addcorr/SNRxx/
```

其中 `SNRxx` 表示具体信噪比，例如 `SNR10`、`SNR20`。
noaddcorr 对照会保存到：

```text
checkpoint/varying_SNR_260707/noaddcorr/SNRxx/
```

### 12.3 输出结果

每个 SNR 的 epoch 100 最终 loss 会汇总到同一个 CSV，使用 `mode` 列区分 addcorr/noaddcorr：

```text
checkpoint/varying_SNR_260707/final_loss_by_SNR.csv
```

并绘制包含 addcorr/noaddcorr 两条曲线的折线图：

```text
checkpoint/varying_SNR_260707/final_loss_by_SNR.png
```

如果当前 Python 环境没有 `matplotlib`，脚本会自动生成无依赖 SVG 版本：

```text
checkpoint/varying_SNR_260707/final_loss_by_SNR.svg
```

图中横坐标为信噪比 `SNR`，范围 `[10, 20]`；纵坐标为最终 epoch 的训练 loss；两条折线分别表示 addcorr 和 noaddcorr。

### 12.4 对照和快速检查

如果只想单独跑一条曲线用于调试：

```bash
python network/run_varing_SNR.py --modes addcorr
python network/run_varing_SNR.py --modes noaddcorr
```

如果只想本地快速检查脚本是否能跑通：

```bash
python network/run_varing_SNR.py --snrs 10,12 --epochs 1 --steps-per-epoch 1 --batch-size 2 --eval-batches 1 --device cpu
```

## 13. 260707 Varying N 实验脚本

### 13.1 实验目的

当前新增 `network/run_varing_number_of_users.py`，用于自动跑用户数量变化实验。默认用户数量为：

```text
N = [50, 100, 150, 200, 250]
```

其他训练、数据、模型参数默认继承 `network/train.py` 的当前设置，只覆盖 `num_devices` 和保存路径。

### 13.2 默认运行方式

默认运行 addcorr 和 noaddcorr 两组实验：

```bash
python network/run_varing_number_of_users.py
```

脚本会分别训练 addcorr/noaddcorr 下的每个 N 到 `epoch=100`，每个实验保存到：

```text
checkpoint/varying_N_260707/addcorr/Nxxx/
checkpoint/varying_N_260707/noaddcorr/Nxxx/
```

其中 `Nxxx` 表示具体用户数量，例如 `N050`、`N250`。

### 13.3 输出结果

每个 N 的 epoch 100 最终 loss 会汇总到同一个 CSV，使用 `mode` 列区分 addcorr/noaddcorr：

```text
checkpoint/varying_N_260707/final_loss_by_N.csv
```

并绘制包含 addcorr/noaddcorr 两条曲线的折线图：

```text
checkpoint/varying_N_260707/final_loss_by_N.png
```

如果当前 Python 环境没有 `matplotlib`，脚本会自动生成无依赖 SVG 版本：

```text
checkpoint/varying_N_260707/final_loss_by_N.svg
```

图中横坐标为用户数量 `N`，范围 `[50, 250]`；纵坐标为最终 epoch 的训练 loss；两条折线分别表示 addcorr 和 noaddcorr。

### 13.4 调试命令

如果只想单独跑一条曲线用于调试：

```bash
python network/run_varing_number_of_users.py --modes addcorr
python network/run_varing_number_of_users.py --modes noaddcorr
```

如果只想本地快速检查脚本是否能跑通：

```bash
python network/run_varing_number_of_users.py --num-users 50 --epochs 1 --steps-per-epoch 1 --batch-size 2 --eval-batches 1 --device cpu
```

## 14. 260720 多协方差行 Token 结构

### 14.1 修改原因

论文原始接收信号路径先构造：

```text
C = Y Y^H / M
x_y = [Re(vec(C)), Im(vec(C))]: [B, 2Lp^2]
```

随后使用一个线性层把完整协方差向量变成单个 signal token：

```text
[B, 2Lp^2] -> Linear(2Lp^2, D) -> [B, 1, D]
```

当前 `D=128`。当 `Lp=8` 时输入维度正好为 128；当 `Lp=30` 时，
1800 维输入会被一次性映射到 128 维。varying-L 实验中，大 L 模型出现明显的
优化平台和性能退化，因此本次从 signal token 结构上移除这一固定维度瓶颈，
不修改事件数据、噪声生成和训练 loss。

### 14.2 新增配置

`network/train.py` 和 `network/config.py` 新增：

```python
SIGNAL_TOKEN_MODE = "covariance_rows"
```

支持两个模式：

```text
flat             原论文结构，2Lp^2 -> D，生成 1 个 signal token
covariance_rows  新结构，按协方差矩阵行生成 Lp 个 signal tokens
```

默认使用 `covariance_rows`。需要复现原论文单 token 结构时设置：

```python
SIGNAL_TOKEN_MODE = "flat"
```

### 14.3 Covariance Row Tokens

数据生成仍返回原有 `x_y: [B,2Lp^2]`，不改变 `network/data.py` 的输出接口。
模型内部将实部和虚部分别恢复为 `[B,Lp,Lp]`，然后按行构造：

```text
r_i = [Re(C_i,:), Im(C_i,:)]
r: [B, Lp, 2Lp]
```

所有协方差行共享同一个投影：

```text
h_y = Linear(2Lp, D)(r) + row_position
h_y: [B, Lp, D]
```

在当前实验范围 `Lp<=30, D=128` 下，每个行 token 的输入维度最多为 60，
不会发生原来的 `2Lp^2 -> 128` 一次性维度压缩。整体 signal 表示随 Lp 增长为
`Lp*D`，新增导频观测会增加 signal token 数量。

### 14.4 异构 Transformer 调用链

异构 MHA 当前接收：

```text
device tokens: [B, N, D]
signal tokens: [B, Ty, D]
Ty = 1   when signal_token_mode="flat"
Ty = Lp  when signal_token_mode="covariance_rows"
```

两类 token 仍使用不同的 Q/K/V、输出投影、FFN 和归一化参数。相关性 attention
bias 仍只写入 device-device 的 `[N,N]` 区域，不作用于 signal token。

decoder 使用全部 signal tokens 作为 queries，并以所有 device/signal tokens 作为
keys 和 values。每个 signal query 得到一个 context，最后在 signal-token 维度求均值，
生成用于用户检测的全局 context vector。Feature Mixer 和 Centered Logits
Refinement 的位置和公式保持不变。

### 14.5 对照公平性与旧 Checkpoint

addcorr 和 noaddcorr 默认都使用相同的 `covariance_rows` signal 编码；两组实验的
区别仍只有相关性 attention bias、feature mixer 和 logits refinement，loss 不变。

旧 checkpoint 中没有 `signal_token_mode`。`build_model_from_config` 对缺失字段默认
使用 `flat`，并且 flat 路径没有新增模型参数，因此旧 checkpoint 可以按原结构严格加载。

### 14.6 Varying-L 验证方式

新实验默认输出到独立目录，避免覆盖 260707 单 token 结果：

```text
checkpoint/varying_L_260720_covrows/
```

直接运行新的多行 token 实验：

```bash
python network/run_varying_l.py
```

脚本新增 `--signal-token-mode`，并在 CSV 中记录该字段。可在相同设置下分别运行：

```bash
python network/run_varying_l.py --signal-token-mode covariance_rows --out-dir checkpoint/varying_L_260720_covrows
python network/run_varying_l.py --signal-token-mode flat --out-dir checkpoint/varying_L_260720_flat
```

新的默认训练、varying-SNR 和 varying-N 输出目录分别为：

```text
checkpoint/addcorr_260720_covrows/
checkpoint/varying_SNR_260720_covrows/
checkpoint/varying_N_260720_covrows/
```

多 signal-token attention 的 token 数从 `N+1` 增加到 `N+Lp`，计算量约随
`(N+Lp)^2` 增长。在当前 `N=200, Lp<=30` 的实验范围内增幅受控，但服务器运行时
仍应观察显存占用。

## 15. Loss 随 Epoch 变化实验（260720）

新增入口：

- `network/run_loss_vs_epoch.py`

脚本通过 `build_args()` 直接继承 `network/train.py` 的当前实验参数，包括
`LP=10`、`EPOCHS=100`、事件驱动数据、measured-SNR 噪声和
`signal_token_mode="covariance_rows"`，未修改训练流程或 loss 计算方式。

默认按相同随机种子依次运行两组实验：

```text
addcorr    使用 train.py 当前的相关性 attention bias、feature mixer 和 logits refinement
noaddcorr  关闭 correlation attention bias 和 logits refinement；feature mixer 随之自动关闭
```

服务器正式运行：

```bash
CUDA_VISIBLE_DEVICES=5 nohup python network/run_loss_vs_epoch.py > nohup_loss_vs_epoch.log 2>&1 &
```

默认输出目录：

```text
checkpoint/loss_vs_epoch_260720_covrows/
```

输出内容：

```text
addcorr/train.log、last.pt、best_pm.pt
noaddcorr/train.log、last.pt、best_pm.pt
loss_by_epoch.csv
loss_vs_epoch.png
```

`loss_by_epoch.csv` 逐 epoch 记录两种模式的 loss、PM、PF、p_mean、skip 和学习率。
折线图横坐标为 epoch，正式默认范围为 1 至 100；纵坐标为当个 epoch 的平均训练
loss，两条曲线分别对应 addcorr 和 noaddcorr。若运行环境没有 matplotlib，脚本会
自动输出 `loss_vs_epoch.svg`。

仅用于快速检查时可以缩小训练规模：

```bash
python network/run_loss_vs_epoch.py --epochs 2 --steps-per-epoch 1 --batch-size 2 --eval-batches 1 --device cpu
```
