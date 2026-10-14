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

## 16. 传统协方差最大似然对比实验（261013）

分支：`main3_261013`。

新增独立目录 `covariance_based/`，实现原始 HT 论文中的传统 Covariance
baseline：协方差最大似然估计与盒约束坐标下降，不训练神经网络。

### 16.1 文件与调用链

```text
covariance_based/run_monte_carlo.py
  -> network/train.py: build_args(), build_system_config()
  -> network/data.py: ActivityDataGenerator.sample_batch(1, return_raw=True)
  -> 原始复数 B、Y、noise_var；标签只用于误差统计
  -> C = Y Y^H / M
  -> covariance_based/detector.py: covariance_mle_activity()
  -> 汇总所有 Monte Carlo 实例的漏检/虚警次数
  -> summary.csv、trial_results.csv、pm_pf_curve.csv、run_config.json
  -> 可选 PNG/SVG 图（安装 matplotlib 时）
```

`covariance_based/test_detector.py` 验证闭式解、[0,1] 边界、目标函数下降、
物理缩放一致性、全不活跃情况和随机种子可复现性。
`covariance_based/README.md` 说明算法、已知噪声假设、运行参数和输出。

### 16.2 检测器目标

```text
Sigma = B diag(a) B^H + noise_var I
min_a logdet(Sigma) + trace(Sigma^-1 C)
0 <= a_n <= 1

v = Sigma^-1 b_n
q = b_n^H v
r = v^H C v
delta = clip((r-q)/q^2, -a_n, 1-a_n)
a_n += delta
Sigma^-1 -= delta/(1+delta*q) * v v^H
```

每轮访问所有设备一次，默认随机排列；每轮重新构造逆矩阵以控制数值漂移。
使用 complex128 和共同物理功率缩放，避免路径损耗导致的极小数值。
不能将独立归一化的 `x_b` 和 `x_y` 直接代入传统物理协方差模型。

检测器仅使用导频矩阵、接收协方差和噪声方差，不使用真实标签、真实活跃数、
用户相关矩阵、事件位置、瞬时信道或先验活跃概率。
默认已知生成器返回的噪声方差；这一设置记录为 `known_generated`，正式对比时
应说明该接收端信息假设。`--receiver-noise-variance` 可提供固定噪声估计值。

### 16.3 Monte Carlo 统计

每次只生成一个接入实例；默认 `MC_TIMES=1000`，没有 epoch、训练集或训练 batch。
沿用当前 `train.py` 的系统参数：N=200、M=32、Lp=10、SNR=20、事件驱动活跃、
measured-SNR 噪声。`--config` 可显式改为从实验 JSON 读取系统配置。

```text
PM = 所有实例的漏检总数 / 所有实例的活跃设备总数
PF = 所有实例的虚警总数 / 所有实例的非活跃设备总数
```

不丢弃全不活跃实例，不用真实活跃数选 top-k。若分母为零，记录为 nan。
协方差输出是连续活跃估计，不是校准概率；似然目标不等于 Transformer 的 BCE
loss。主要使用 PM-PF 曲线或同一 PF 下的 PM 进行算法性能比较。

### 16.4 运行与输出路径

在项目根目录激活环境后执行；计算在 CPU 上运行，无需分配 GPU：

```bash
python covariance_based/run_monte_carlo.py --mc-times 1000
python covariance_based/run_monte_carlo.py --sweep L --mc-times 1000
python covariance_based/run_monte_carlo.py --sweep N --mc-times 1000
python covariance_based/run_monte_carlo.py --sweep SNR --mc-times 1000
```

默认扫描值：L=[4,6,...,30]、N=[50,100,150,200,250]、SNR=[10,12,14,16,18,20]。
支持 `--values` 自定义范围，其余系统参数保持继承值。
输出根目录由 `run_monte_carlo.py` 的 `OUTPUT_ROOT` 或 `--out-dir` 控制：

```text
checkpoint/covariance_261013/single/
checkpoint/covariance_261013/varying_L/
checkpoint/covariance_261013/varying_N/
checkpoint/covariance_261013/varying_SNR/
```

小规模检查与算法测试：

```bash
python covariance_based/run_monte_carlo.py --mc-times 3 --max-sweeps 20 --no-plot --out-dir /tmp/covariance_smoke
python -m unittest covariance_based.test_detector -v
```

改变 `MC_TIMES` 调整统计样本数；改变 `MAX_SWEEPS` 和 `TOLERANCE` 调整求解精度。
结果记录收敛比例；达到迭代上限的实例仍保留。`summary.csv` 中运行时间包含
协方差构造和求解，不包含数据生成和写文件。

### 16.5 macOS 运行库兼容（261013）

`covariance_based/run_monte_carlo.py` 和包入口 `covariance_based/__init__.py`
在 macOS 上、导入 NumPy/PyTorch 之前，将未显式设置的
`MKL_THREADING_LAYER` 默认为 `SEQUENTIAL`，避免 MKL 与 PyTorch 初始化
重复的 OpenMP 运行库而触发 `OMP: Error #15`。Linux 服务器不应用此设置；
已有的显式环境设置不被覆盖。此改动仅调整 MKL 的线程后端，不改变检测器、
数据分布、场景参数、Monte Carlo 次数或统计方式。

本机应使用已安装依赖的 `cat` 环境；`base` 缺少 NumPy，不能直接运行本实验。
在新的 Python 进程中执行：

```bash
conda activate cat
python covariance_based/run_monte_carlo.py --mc-times 3
```

旧版本脚本可在命令前加 `MKL_THREADING_LAYER=SEQUENTIAL`；
不要用 `KMP_DUPLICATE_LIB_OK=TRUE` 强行允许重复运行库。

### 16.6 对数坐标绘图（261013）

参考本地 HT 论文图4，将 `pm_pf_curve.png/svg` 改为横轴 PM、纵轴 PF，
双轴以10为底的对数坐标，主刻度显示为10的整数指数次方。默认显示
10^-4 至10^0；如果实际非零概率更小，会自动扩展下界，不裁掉这些数据。
`PLOT_MIN_PROBABILITY=1e-4` 控制默认下界。

`detection_vs_L/N/SNR.png/svg` 的参数横轴保持线性，PM/PF 纵轴改为同样的
对数格式，增加主/次网格。仅绘图时将0和未定义的概率屏蔽为曲线断点；
不将0替换成非零值，不改动CSV、概率统计、检测算法或实验参数。
`test_detector.py` 新增坐标方向、对数格式、极小概率及0/nan处理的绘图测试。
参数横轴显式使用完整扫描范围及扫描值刻度，避免屏蔽0后自动缩小横轴范围。

### 16.7 协方差方法临时使用 L=8（261013）

`covariance_based/run_monte_carlo.py` 新增 `PILOT_LEN_OVERRIDE=8`，仅覆盖
协方差实验的导频长度，其余参数仍继承当前 `network/train.py`。
Transformer 的 `LP=10` 不变。测试后将 `PILOT_LEN_OVERRIDE` 改为 `None`
即可恢复参数继承，不必修改 Transformer 配置。
命令行 `--pilot-len` 的优先级更高；L扫描仍采用各扫描值，N/SNR扫描保持L=8。
默认输出目录不变；需要保留旧结果时，使用独立的 `--out-dir`。

### 16.8 恢复协方差导频长度继承（261013）

L=8测试完成后，将 `PILOT_LEN_OVERRIDE` 恢复为 `None`，协方差方法重新
继承 `network/train.py` 的 `LP=10`。其余设置、绘图格式和已有测试结果不变。

### 16.9 论文参数下的事件驱动临时测试（261014）

仅修改协方差入口，启用 `USE_PAPER_EVENT_TEST_SETTINGS=True`；
`PAPER_EVENT_TEST_SETTINGS` 将N设为100、L设为8、小区半径设为250米、
活跃模式保持event、噪声模式改为thermal。固定噪声密度为-169 dBm/Hz，
带宽10 MHz，积分噪声功率为-99 dBm，即约1.2589e-13 W。
此时 `SNR_DB` 不参与噪声计算，thermal模式下不允许直接扫描SNR；
如需SNR扫描，应显式指定 `--noise-mode snr` 或 `large_scale_snr`。

事件到达参数、120米影响尺度、触发/背景概率、导频归一化和功率控制保持不变；
导频元素仍为CN(0,1/L)，因此本次并非完整复现论文的数据分布。
`network/train.py` 不变，命令行物理参数优先于测试预设。
默认结果保存到 `checkpoint/covariance_261014_paper_event/single/`，
避免覆盖以前的结果；`--out-dir` 仍可显式指定路径。

```bash
python covariance_based/run_monte_carlo.py
```

测试完成后，将 `USE_PAPER_EVENT_TEST_SETTINGS=False`，并保持
`PILOT_LEN_OVERRIDE=None`，即可恢复继承Transformer设置及原来的默认输出目录。

### 16.10 独立活跃概率0.1的参考测试（261014）

新增 `covariance_based/run_independent_reference.py`，复用上述论文参数测试
预设和 `run_monte_carlo.main()`。N=100、L=8、250米半径、thermal噪声保持
一致，仅显式覆盖 `activity_mode=independent` 和 `activity_prob=0.1`。
M、功率控制、导频归一化、检测器及统计方式不变，不修改事件测试开关。
即使关闭事件测试预设，参考入口仍显式提供相同的物理参数。

```text
run_independent_reference.py: build_reference_args()
  -> run_monte_carlo.py: main(), experiment_system()
  -> ActivityDataGenerator: independent Bernoulli(0.1)
  -> 原协方差最大似然检测器
  -> 独立目录下的CSV、配置和PM-PF双对数图
```

```bash
python covariance_based/run_independent_reference.py --mc-times 1000
```

默认输出为 `checkpoint/covariance_261014_paper_independent/single/`，
不覆盖事件测试。支持原Monte Carlo入口的命令行参数，显式参数优先。
此测试用于低负载参考校验，不能单独隔离空间相关性的影响：事件测试的
活跃比例及活跃数分布不同；也没有改变导频方差来完整复现论文。

本机已完成1000次参考测试：PM@0.5=0.178872，PF@0.5=0.006429，
平均活跃比例0.10091，1000个实例均达到收敛标准。
核对逐次计数、汇总指标、固定噪声方差和配置一致；已有事件结果哈希不变。

### 16.11 独立参考入口复现HT论文协方差实验（261014）

将 `covariance_based/run_independent_reference.py` 从参考参数包装器升级为
独立的论文参数仿真入口。默认对应本地HT论文图4的三条Covariance曲线：
`(L,M)=(7,32),(8,32),(8,64)`，每组5000次Monte Carlo。
N=100、半径250米、独立Bernoulli(0.1)活跃、最大功率23 dBm；
导频元素和小尺度信道均为CN(0,1)，不再调用项目的CN(0,1/L)导频生成器。
路径损耗为128.1+37.6log10(distance_km)，功率控制为
p_n=pmax*g_min/g_n；固定热噪声为-169 dBm/Hz乘10 MHz，即-99 dBm。
三组共享位置和标签，以及同一大尺寸导频、信道、噪声的前缀，保持各自边缘分布。

```text
run_independent_reference.py: parse_args(), experiment_system()
  -> PaperSystem: 独立物理配置，不继承train.py或事件测试开关
  -> paper_realization(): 单个complex128物理实现，生成原始B、Y和评价标签
  -> C=Y Y^H/M
  -> detector.py: covariance_mle_activity(B,C,noise_variance)
     已知大尺度增益、0<=a<=1的最大似然坐标下降，不使用活跃先验或标签
  -> direct_coordinate_check(): 每组前3次逐坐标重新求逆，交叉验证快速更新
  -> run_case(): 累计真实活跃/不活跃实例、门限曲线和边界分数统计
  -> CSV、activity_scores.npz、配置JSON及三条曲线的PM-PF双对数图
```

仅复用原协方差求解器、统计和坐标格式函数；Transformer、事件入口、已有结果不变。
默认零初始化、随机坐标顺序、最多500个完整扫描、停止容差1e-6；
这些是本实现选项，论文没有给出图4的全部实现细节，不能表述为作者原始设置。
g_min取本次设备中的最小增益，距离计算有1米数值保护；配置文件记录这些选择。
直接求逆检查验证实现一致性，不证明非凸问题的全局最优性。

默认门限为1001个线性点加靠近0/1的对数点，判断仍为a_hat>threshold。
保留真实0/1，新增active_at_zero、inactive_at_one、PM_at_zero和PF_below_one；
CSV中的0不替换成非零概率，绘图不人为延伸曲线以拟合论文。

```bash
python covariance_based/run_independent_reference.py
python covariance_based/run_independent_reference.py --cases 8:32 --mc-times 1000
python covariance_based/run_independent_reference.py --pilot-len 8 --num-antennas 64
python -m unittest covariance_based.test_detector covariance_based.test_paper_reference -v
```

默认保存至 `checkpoint/covariance_261014_paper_reproduction/`：
根目录为合并summary.csv、pm_pf_curve.csv、solver_audit.csv、run_config.json及PNG/SVG；
L7_M32、L8_M32、L8_M64子目录保存各组CSV、逐次统计和activity_scores.npz。
该NPZ仅存连续估计与标签，用于复核门限，不存训练batch。
`MC_TIMES=5000`和`REFERENCE_OUTPUT_DIR`定义在独立参考入口，命令行可覆盖。
`--cases`与单组`--pilot-len/--num-antennas`不能同时使用。
原 `covariance_261014_paper_independent/single/` 的1000次测试结果保留。

验证：三组各5000次独立活跃实验完成，15000次均收敛；PM/PF@0.5分别为
(7,32):0.261649/0.012074，(8,32):0.163625/0.005530，
(8,64):0.072669/0.002429。标签配对、各组CSV与NPZ计数、门限单调性和固定噪声一致；
9次独立求逆检查的最大分数差5.22e-15、最大目标函数差2.84e-14。
图已视觉检查；完整独立活跃曲线仍与论文数值不同，不宣称精确复现原图。
既有事件/参考结果以及train.py、data.py、原协方差入口/检测器哈希不变。

### 16.12 核查论文复现的活跃数分布（261014）

论文同时给出活跃比例0.1和Bernoulli标签，但没有明确图4中联合采样时是否固定K。
独立Bernoulli(0.1)会使K变化；N=100时均匀选取固定10个设备的每个边缘概率也为0.1，
但设备状态并非独立。因此不能将两种数据设置混同。

`run_independent_reference.py` 新增 `save_count_diagnostic()`：在检测完成后，
仅从已保存分数中选取真实K=round(N*p)的实例计算条件PM/PF，不改变检测或主结果。
独立活跃的5000次中有669次K=10，单独输出到 `K10_subset_diagnostic/`。
CSV、配置与图标题明确源样本数和选中次数，不将该子集称作完整5000次复现实验。
条件曲线更接近论文是实验线索，不能据此确认作者的原始采样代码。

新增显式 `--active-count 10` 进行完整固定K实验：在N个独立均匀随机数中选择
最小的K个秩，得到均匀K子集；保持位置、导频、信道和噪声的RNG配对。
配置与CSV记录 `activity_mode=fixed_count` 及有效边缘概率K/N；求解器仍不接收K。
默认不加此参数时继续使用独立Bernoulli，不修改事件实验入口或Transformer。
固定K默认输出为 `checkpoint/covariance_261014_paper_fixedK10/`，不覆盖独立实验。

```bash
python covariance_based/run_independent_reference.py --active-count 10
```

验证：24项测试通过，包括固定K均匀性、0/N边界、物理RNG配对和条件统计；
固定K三组各5000次实验完成，15000次均收敛；PM/PF@0.5分别为
(7,32):0.243220/0.007791，(8,32):0.122220/0.002109，
(8,64):0.031120/0.000616。各组活跃/不活跃总实例数为50000/450000。
在门限网格的PF<=0.01点中，最小PM分别为0.197140、0.028160、0.001820；
对应独立活跃完整实验为0.304831、0.098762、0.020363，不使用插值修改概率。

固定K的9次直接求逆检查最大分数差2.89e-15，最大目标函数差2.84e-14；
两模式共有的669个K=10实例，标签与估计分数逐元素完全一致，证实物理随机量配对。
两模式的完整30000次CSV/NPZ、门限单调性、收敛条件及固定噪声统计均核对通过。
固定K图已视觉检查：主要区间与论文形状和量级更接近，但端点仍未完全重合，
不能由此断言作者固定K或已精确复现全部原图数值；原始作者采样与求解细节仍待确认。
原协方差检测器、事件入口、Transformer及既有实验结果保持不变。

### 16.13 清理已替代的独立活跃参考结果（261014）

按用户要求删除 `checkpoint/covariance_261014_paper_independent/`。
该目录为早期N=100、L=8、M=32、半径250米、独立p=0.1和固定热噪声的
1000次参考测试，导频仍为CN(0,1/L)，并非完整论文参数复现。
当前参考入口不再读取或默认输出至该目录，不影响代码运行。
删除仅涉及旧结果的CSV、配置JSON和PNG/SVG，不修改任何算法或实验参数。

保留 `covariance_261014_paper_reproduction/` 的三组独立活跃结果、
`covariance_261014_paper_fixedK10/` 的三组固定K结果，
以及 `covariance_261014_paper_event/` 的事件测试结果。
上述目录的summary.csv哈希删除前后保持一致。
16.10、16.11及旧修改日志中的结果与保留说明为历史记录，保留原文字；
旧目录的当前状态以本清理记录为准，README同步更新为已删除。

### 16.14 协方差Monte Carlo恢复Transformer事件场景（261014）

仅修改 `covariance_based/run_monte_carlo.py` 的默认实验配置：
`USE_PAPER_EVENT_TEST_SETTINGS=False`、`PILOT_LEN_OVERRIDE=None`，
通过 `build_args()` / `build_system_config()` 完整继承 `network/train.py`。
当前基准为N=200、M=32、L=10、半径500米、最大功率23 dBm、event活跃和
measured-SNR 20 dB噪声；事件Poisson参数1、最多3个事件、响应尺度120米、
峰值触发0.9、背景活跃0.005，整体活跃比例不固定。

继续使用同一 `ActivityDataGenerator.sample_batch(1, return_raw=True)`，
导频元素CN(0,1/L)，噪声方差为mean(abs(BAH)^2)*10^(-SNR/10)，
保留原始B、Y和对应噪声方差，不使用分别归一化的神经网络特征求解似然。
场景及数据分布与Transformer一致，不表示与历史GPU/batch测试观测逐样本相同。
检测器仍不使用标签、真实活跃数、G、事件中心或先验活跃概率；
保留known_generated噪声方差假设，正式比较需说明该接收端额外信息。

`MC_TIMES=5000`，默认单点运行；不扫描L/N/SNR。
求解器MAX_SWEEPS=100、TOLERANCE=1e-4不变；达到扫描上限的样本仍保留统计。
输出改为 `checkpoint/covariance_261014_transformer_event/single/`，
保存summary.csv、trial_results.csv、pm_pf_curve.csv、run_config.json及PM-PF的PNG/SVG。
图的横轴为PM、纵轴为PF，采用10的指数双对数坐标，0值不替换为非零概率。
`run_independent_reference.py`、原协方差检测器、Transformer和数据生成器保持不变，
既有论文复现及旧事件结果不覆盖。

```bash
python covariance_based/run_monte_carlo.py
python -m unittest covariance_based.test_detector covariance_based.test_paper_reference -v
```

验证：26项测试和语法编译通过，30次小规模检查配置一致并生成CSV/PNG/SVG；
正式5000次完成并全部达到收敛标准，平均24.061轮。
PM@0.5=0.470537、PF@0.5=0.026875，平均活跃比例0.084461；
活跃/不活跃总实例数84461/915539，漏检/虚警39742/24605。
活跃人数范围0至71，其中671次全不活跃，这些实例仍保留在统计中。
门限0的PM=0.236334、PF=0.119794；门限0.24的PM=0.332935、PF=0.049495。
上述仅为实测门限扫描点，不与神经网络训练loss或不同数据上的门限0.5结果混比。

逐次计数与汇总一致，配置与train.py全字段一致，曲线单调且门限0.5处与汇总一致；
重放前30个CPU物理样本，活跃数及噪声方差与CSV逐项一致，实测SNR噪声公式核对通过。
PNG视觉检查通过，CSV中的真实0保持不变；独立论文入口、检测器、Transformer/数据代码，
以及已有独立活跃和固定K论文复现结果的哈希保持不变。本次只恢复场景并增加Monte Carlo次数。

### 16.15 GI-HT、HT与协方差方法共享样本PM-PF对比（261014）

新增 `network/run_pm_pf_comparison.py`，仅推理，不训练或修改loss。
加载loss-vs-epoch实验的addcorr/noaddcorr `last.pt`，校验系统配置、训练epoch和
除三个几何开关外的网络设置一致；GI-HT启用attention bias、feature mixer和
logit refinement，HT全部关闭且推理时不传入G。两者均使用covariance_rows。
正式对比使用第100个epoch；既有best_pm.pt不参与模型选择。

项目checkpoint中的两份last.pt不能加载，PyTorch报告缺少ZIP central directory；
在 `All experiments results/results_260720/checkpoint/loss_vs_epoch_260720_covrows/`
找到完整备份，严格加载state_dict成功，物理配置与Transformer基准一致。
使用命令行指定完整备份，只读这些权重，不覆盖项目中原有文件。

场景由checkpoint固化配置确定：N=200、M=32、L=10、R=500米、event和实测SNR20 dB；
共享生成器在CPU逐次sample_batch(1, return_raw=True)，seed=42，默认5000次。
每次同一B、Y、标签和噪声用于三种方法；G仅GI-HT使用。
协方差在同一观测上重新求解，保留known_generated噪声、100轮/1e-4及独立seed43的
坐标顺序；模型只做eval模式float32推理，--batch-size仅影响神经网络批处理。
TokenBatchNorm在eval时仍使用当前batch统计量，因此预测与batch分组有关；默认推理
batch继承权重中的训练值128，两模型分组一致，最后8个样本作为不足128的尾批保留并记录。
不将“物理样本不随推理batch改变”误解为“网络预测不随batch改变”。

所有方法按score > threshold统一扫描[0,1]的1001个门限，按真实活跃/不活跃总数
汇总PM/PF，不丢弃全不活跃样本。新增输出 `checkpoint/pm_pf_comparison_261014/`：
pm_pf_comparison.png/svg/pdf（三曲线、横PM纵PF、10的指数双对数），
pm_pf_curves.csv、summary.csv、activity_scores.npz、covariance_trials.csv及run_config.json。
保存权重哈希、共享物理样本哈希、全部分数和标签，支持复核与再绘图；
概率0不伪造非零值，原协方差结果和run_independent_reference.py不改动。

新增 `network/test_pm_pf_comparison.py`，检查严格门限、计数与端点、配置一致性、
无效权重报错、HT不输入G、生成样本不随推理batch改变、BatchNorm推理依赖batch以及三曲线输出。
验证：34项单元测试、语法编译及diff检查通过；16次真实权重小规模测试通过。
正式5000次共享观测评估完成，两个模型均使用第100个epoch、float32和batch=128，
最后尾批8个；协方差5000次全部收敛，平均24.061轮。
PM/PF@0.5：GI-HT=0.053445/0.095319，HT=0.072116/0.170757，
Covariance=0.470537/0.026875；三方法共享真实活跃84461、不活跃915539个用户实例。
在PF<=0.05的实测网格点中，最低PM分别为0.132523、0.333692、0.330922；
对应门限0.744、0.806、0.236，实际PF分别为0.049932、0.049680、0.049983。
不通过插值修改概率，不将相同数值门限视为相同虚警率。

NPZ标签/分数与CSV计数一致，三条曲线单调，门限1处PM=1、PF=0；
重新求解的协方差逐次活跃数、噪声、轮数及停止步长与原5000次结果完全一致，
在共有101个门限处，全部PM/PF及漏检/虚警计数与旧曲线一致。
PNG视觉检查通过；模型、生成器、检测器、论文复现入口及旧结果哈希保持不变。
本次推理样本不同于原训练日志的固定评估集，精度为CPU float32而非服务器AMP；
不得将新PM/PF数值要求为原日志逐字相同。原网络、loss和训练权重不修改。

### 16.16 降低事件触发负载并单独测试协方差方法（261014）

按用户要求同时减少单事件响应人数和事件到达次数：
`network/train.py` 的EVENT_TRIGGER_PROB由0.9改为0.6，EVENT_LAMBDA由1.0改为0.5；
同步 `network/data.py` SystemConfig和 `network/config.py` 默认事件参数。
仍采用距离高斯衰减、任一事件或背景触发及Bernoulli随机采样，生成公式不变。
保留N=200、L=10、M=32、R=500米、实测SNR20 dB、响应尺度120米、
最多3事件及背景概率0.005；不固定整体活跃比例、不强制固定K或截断活跃人数。
空间依赖矩阵的构造及事件响应范围不变，活动分布随新参数变化。

`covariance_based/run_monte_carlo.py` 继续完整继承train.py场景，默认5000次；
输出改为 `checkpoint/covariance_261014_low_load_event/single/`，保留旧场景结果。
新增从trial_results.csv真实标签人数汇总的activity_count_stats.json，记录每个
条件的K最小/最大/平均/中位数、90/95/99分位、全不活跃次数、非空场景统计和直方图。
不以检测器估计人数代替真实K，不删除空场景，也不向检测器提供K。
求解器100轮/1e-4、known_generated噪声、score > threshold及PM/PF统计方式不变；
本次命令使用1001个[0,1]门限，仅画协方差方法的双对数PM-PF图。

```bash
python covariance_based/run_monte_carlo.py --mc-times 5000 --num-thresholds 1001
python -m unittest covariance_based.test_detector covariance_based.test_paper_reference network.test_pm_pf_comparison -v
```

本机不训练Transformer。train.py及四个训练run的默认输出目录新增
261014_low_load标记，避免后续服务器实验覆盖旧结果；四个run仍继承train.py的新事件配置。
旧权重保存的事件参数不修改，run_pm_pf_comparison.py仍使用权重自身的场景。
后续同场景三方法对比应先在低负载分布重新训练GI-HT/HT，再指定新权重评估。
run_independent_reference.py、网络结构、loss、协方差检测器及已有实验结果不改动。
新增低负载参数一致性、空间响应/背景公式、G不变及真实K统计测试。
验证：38项单元测试、语法编译及diff检查通过；12次小规模完整运行通过。
四个Transformer run的参数/目录继承检查通过，本机未执行任何训练。
正式5000次协方差评估完成且全部达到收敛标准，平均12.3002轮。
真实活跃总实例32533、非活跃967467；阈值0.5漏检8707、虚警3668，
PM=0.267636、PF=0.003791，整体活跃比例0.032533，不是预设固定概率。

本次样本中K范围0至47，平均6.5066、中位数2；90/95/99分位分别18/22/33。
1069次全不活跃，3931次非空；非空K范围1至47，平均8.276011。
该范围是5000次样本的观察值，不是生成器的硬约束；模型仍允许0至N个活跃用户。
与原λ=1、峰值0.9的5000次实验相比，平均K由16.8922降至6.5066，最大K由71降至47。
新曲线门限0处PM=0.083208、PF=0.067051；当前门限规则下仍无PM=PF交点，
不因降低负载而强制平衡错误率，也不通过插值伪造未达到的性能。

逐次CSV、汇总计数、K直方图/分位数、曲线单调性和门限0.5/1端点核对通过；
新旧完整系统配置仅event_lambda和event_trigger_prob两字段不同。
前30个CPU物理样本重放的K及噪声方差与CSV完全一致，实测SNR噪声公式检查通过。
PNG视觉检查通过，保留概率0；网络结构、loss、检测器、论文复现入口及旧结果哈希不变。

### 16.17 恢复事件到达参数λ=1并重跑协方差（261014）

按用户要求，将train.py的EVENT_LAMBDA及SystemConfig/DEFAULT_CONFIG默认值由0.5恢复为1.0。
EVENT_TRIGGER_PROB保持0.6，其他系统、生成公式、网络、loss和协方差求解器不变。
协方差继续继承N=200、L=10、M=32、R=500米、实测SNR20 dB、120米响应尺度、
最多3事件和背景0.005；不固定K，不强制整体活跃比例。

协方差默认输出根改为 `checkpoint/covariance_261014_low_load_lambda1_event/`，
保留λ=0.5的 `covariance_261014_low_load_event/` 及全部历史结果。
train.py和四个训练run的默认输出路径加入low_load_lambda1标记，避免混淆两组场景。
同步默认参数测试和README；旧CALL_CHAIN和修改记录保留原文字，仅追加本记录。

```bash
python covariance_based/run_monte_carlo.py --mc-times 5000 --num-thresholds 1001
```

仅在本机执行5000次协方差评估，统一1001门限扫描并单独画双对数PM-PF图；
activity_count_stats.json继续记录真实K的范围、均值、分位数和空场景。
不训练Transformer，不修改论文复现入口、训练权重或旧实验文件。
38项测试、语法编译及四个run参数/路径继承检查通过。
用户中断时，本任务已实际完成5000次并输出结果，全部收敛，平均19.878轮；
K范围0至53、平均11.753，PM/PF@0.5为0.366426/0.011948。
用户随后要求测试阶段改为1000次，5000次输出保留；当前测试以16.18为准。

### 16.18 单独协方差测试默认改为1000次（261014）

按用户最新要求，将 `covariance_based/run_monte_carlo.py` 的MC_TIMES由5000改为1000，
EVENT_LAMBDA=1.0和EVENT_TRIGGER_PROB=0.6保持不变；不改变任何系统参数或算法。
新默认输出根为 `checkpoint/covariance_261014_low_load_lambda1_mc1000_event/`，
保留已完成的λ=1、5000次实验及所有之前的结果，不以裁剪CSV代替重新运行。
默认次数测试同步更新；network/run_pm_pf_comparison.py不再导入测试入口的MC_TIMES，
联合正式对比保持自身5000次默认值，新增测试验证两入口次数独立。
论文复现入口本身5000次默认值不修改。

```bash
python covariance_based/run_monte_carlo.py --num-thresholds 1001
```

本机仅重新运行1000次协方差测试、单独绘图并记录真实K；不训练Transformer。
原16.17及修改日志24为已完成的5000次历史记录，最新测试次数以本节为准。
验证：39项单元测试、语法编译及diff检查通过；本次1000次重新运行完成且全部收敛，
平均20.769轮。真实活跃/非活跃12351/187649，阈值0.5漏检/虚警4733/2501，
PM=0.383208、PF=0.013328，整体实测活跃比例0.061755。
K范围0至52、平均12.351、中位数11，90/95/99分位为29/33/约42.01；
132次全不活跃，868次非空，非空平均K为14.229263。
人数范围仅针对本次样本，不是硬上限；99分位采用NumPy默认线性分位插值。

CSV计数、统计JSON和直方图、曲线单调性及0.5/1端点检查通过；
前30个物理样本的K和噪声重放一致，1000次全部非耗时字段与之前5000次的前1000次一致。
新PNG视觉检查通过；旧5000次结果及其他受保护文件哈希未变化。
没有训练Transformer，没有更改网络、loss、生成器或协方差求解器。

最终文件系统核对时，原λ=1的5000次目录covariance_261014_low_load_lambda1_event已不存在。
上面的旧结果哈希/逐次配对检查是在目录仍存在时完成的；本次未执行任何删除操作，
不恢复该目录。当前1000次目录及其完整结果存在，旧5000次的数值仅作为已验证历史记录。

### 16.19 将事件响应尺度σ缩小为80米并测试协方差（261014）

按用户要求，将train.py的EVENT_SIGMA_M由120改为80，并同步SystemConfig及DEFAULT_CONFIG。
EVENT_LAMBDA=1、EVENT_TRIGGER_PROB=0.6、背景概率0.005、最多3事件和其他场景参数不变。
仍用高斯距离衰减、事件/背景联合触发及Bernoulli采样，不固定K，不增加硬人数上限。
σ是响应尺度而非硬覆盖半径。CORRELATION_LENGTH_M=0时，G的默认距离尺度自动由
sqrt(2)*120改为sqrt(2)*80；几何矩阵和信号/活跃生成公式不变。

协方差入口默认MC_TIMES仍为1000，本次使用1001个[0,1]门限，单独画双对数PM-PF图。
新输出 `checkpoint/covariance_261014_sigma80_lambda1_mc1000_event/single/`，
保留σ=120的1000次结果和已有实验。真实K范围、均值、分位数和直方图仍保存到
activity_count_stats.json，不将真实K提供给检测器；100轮/1e-4及known_generated噪声不变。
train.py和四个训练run的新默认目录使用sigma80_lambda1标记，避免服务器后续覆盖旧结果。
本机不训练Transformer，不修改网络、loss、协方差求解器或run_independent_reference.py。

```bash
python covariance_based/run_monte_carlo.py --num-thresholds 1001
```

更新默认参数/目录测试，空间响应测试改为按配置σ构造距离，新增G的尺度自适应检查。
验证：40项单元测试、语法编译及diff检查通过；四个训练run参数/目录继承检查通过。
实际1000次完成且全部收敛，平均12.879轮；真实活跃/非活跃6496/193504，
门限0.5漏检991、虚警294，PM=0.152555、PF=0.001519，实测活跃比例0.03248。
K实际范围0至28、平均6.496、中位数6，90/95/99分位15/18/约22.01；
136次空场景，864次非空，非空平均K为7.518519。范围不是生成器硬上限。
相比σ=120同样1000次的0至52/平均12.351，活跃人数及两项门限0.5错误率均降低。

完整系统配置仅event_sigma_m由120改为80；全部1000对CPU物理样本重放通过：
导频、用户位置、信道、事件中心和随机数推进逐项完全一致，σ80标签是σ120标签的子集，
对应G不大于原G；两组真实K和噪声与各自CSV一致，实测SNR噪声公式检查通过。
实际事件数分布仍为0/1/2/3事件对应341/372/198/89次，总1035事件、均值1.035。
因此本次只缩小响应尺度，没有改变事件到达样本或人为截断活跃人数。

CSV/汇总、K直方图和分位数、曲线单调性及门限0.5/1端点核对通过，PNG视觉检查通过。
门限0处PM=0.031404、PF=0.058144；门限扫描曲线的PM=PF线性插值约0.031627，
插值门限约0.002449，该值为离散经验曲线交点近似，不冒充精确可达的相等概率。
网络、loss、检测器、论文复现入口及σ120/其他受保护结果哈希不变；未训练Transformer。

### 16.20 将事件响应尺度σ调整为100米并测试协方差（261014）

按用户要求，将EVENT_SIGMA_M由80改为100，同步train.py、SystemConfig及DEFAULT_CONFIG。
λ=1、峰值触发0.6、背景0.005、最多3事件、N=200/L=10/M=32/R=500/SNR=20保持不变。
σ仍是高斯响应尺度而非硬覆盖半径；默认G距离尺度自动使用sqrt(2)*100米。
不修改事件/信号生成公式、网络、loss、协方差求解器或独立论文复现入口。

协方差仍测试1000次，扫描1001个[0,1]门限并单独绘制双对数PM-PF图；
新输出checkpoint/covariance_261014_sigma100_lambda1_mc1000_event/single/。
train.py及四个训练run默认目录改用sigma100_lambda1标记，不覆盖已有实验。
同步默认参数/目录测试，G尺度测试按当前σ构造距离；本机不训练Transformer。

```bash
python covariance_based/run_monte_carlo.py --num-thresholds 1001
```

验证：40项单元测试、语法编译、diff检查及四个训练run默认参数/目录继承检查通过。
实际1000次全部收敛，平均16.942轮；活跃9211、非活跃190789、漏检2496、虚警1094，
PM/PF@0.5=0.270980/0.005734。K实测范围0至39、均值9.211、中位数8，
90/95/99分位为21/25/31；132次空场景全部保留，非空868次平均K为10.611751。
相比σ=80的0至28/平均6.496、PM/PF@0.5=0.152555/0.001519，负载与错误率增大。
范围是本次样本统计，并非生成器硬上限。

1000对σ80/100原始样本配对重放通过：导频、位置、信道、事件中心和随机数推进一致，
σ80标签是σ100标签的子集，真实K和噪声与各自CSV一致，几何距离尺度随σ调整。
事件数仍为0/1/2/3事件对应341/372/198/89次，均值1.035。
计数/汇总、K统计与直方图、曲线单调性及门限0.5/1端点核对通过，PNG视觉检查通过。
门限0处PM=0.086744、PF=0.075088，当前[0,1]门限曲线无PM=PF交点，不给出伪交点。
网络、loss、检测器和独立复现入口哈希未变；σ80旧结果已在用户调整后的目录
checkpoint/covariance_261014_eventtriggerprob0.6_sigma80_lambda1/single/核验哈希不变。
本次未执行任何删除操作或Transformer训练。

### 16.21 将峰值事件触发概率降低为0.5并测试协方差（261014）

将EVENT_TRIGGER_PROB由0.6改为0.5，同步train.py、SystemConfig及DEFAULT_CONFIG。
σ=100、λ=1、背景0.005、最多3事件及N=200/L=10/M=32/R=500/SNR=20保持不变。
事件仍按高斯距离响应联合触发，不固定K；几何矩阵仅依赖位置和σ，本次G不变。
不修改数据生成公式、网络、loss、协方差求解器或独立论文复现入口。

仍用1000次Monte Carlo、1001个[0,1]门限，单独绘制双对数PM-PF图；
新输出checkpoint/covariance_261014_sigma100_trigger0p5_lambda1_mc1000_event/single/。
train.py及四个训练run默认目录增加trigger0p5标记，保留0.6场景结果；
旧权重不会自动更新场景，需要服务器重新训练后进行同场景比较。本机不训练Transformer。
同步默认概率/目录及事件响应测试，更新README，并在Modify log记录。

```bash
python covariance_based/run_monte_carlo.py --num-thresholds 1001
```

验证：40项单元测试、语法编译、diff及四个训练run默认参数/目录继承检查通过。
实际1000次全部收敛，平均15.053轮；活跃7886、非活跃192114、漏检1749、虚警656，
PM/PF@0.5=0.221785/0.003415。K实测0至33、均值7.886、中位数7，
90/95/99分位18/21/约26.01；保留134次空场景，866次非空平均K为9.106236。
相比触发0.6的K均值9.211、最大39及PM/PF@0.5=0.270980/0.005734，两项错误率降低。

1000对0.5/0.6原始样本重放通过：导频、位置、信道、事件中心、G和随机数推进一致，
0.5标签为0.6标签的子集；新K/噪声与CSV一致，0.6重放复核总活跃9211、最大39。
实际事件数仍为0/1/2/3事件对应341/372/198/89次，均值1.035。
计数/汇总、K统计与直方图、曲线单调性及门限0.5/1端点核对通过，PNG视觉检查通过。
门限曲线PM=PF线性插值约0.054999，插值门限约0.000575，不作为精确可达的经验概率。
网络、loss、检测器和独立复现入口哈希未变，本机未训练Transformer。

运行后目录发生外部变更，当前完整结果位于
checkpoint/covariance_261014_eventtriggerprob0.5_sigma100_lambda1/single/；
当前代码的未来默认输出仍为上述sigma100_trigger0p5目录。
此前0.6结果目录在最终核对时已不存在；本次未执行删除或恢复操作，
其已验证历史数值保留，仅重放物理样本验证负载，没有恢复旧检测器结果文件。

### 16.22 当前事件场景的服务器训练前检查（261014）

保持σ=100、EVENT_TRIGGER_PROB=0.5、λ=1及全部网络/loss/训练设置不变。
检查train.py及四个run：均继承当前事件场景和covariance_rows；
三个扫参run只修改目标L/N/SNR，loss_vs_epoch保持N=200/L=10/M=32/SNR=20。
默认100 epoch、2000 step/epoch、batch=128、eval_batches=20。
addcorr开启注意力偏置、图特征传播和logit修正；noaddcorr关闭两主开关后，
build_model_config同时关闭图传播，测试确认传入G、None或随机G时noaddcorr输出完全一致。

四个真实run均在CPU完成两个模式的小规模端到端测试：每模式1 epoch、2 step、
batch=4、eval_batches=2，不降低网络尺寸；检查有限loss、skip=0、前向/反向、
last.pt/best_pm.pt/train.log、严格权重重载、CSV和PNG生成及保存的场景/开关一致性。
测试输出只在临时目录，不覆盖正式实验，不把测试loss作为实验结论。
本机无CUDA，未验证服务器GPU的BF16支持、显存或完整100 epoch训练；
建议服务器先用默认batch=128完成短测，然后正式提交任务。

修复requirements.txt两行无效依赖Python 3.12.2和2.11.0+cu128，补入matplotlib，
保留torch/numpy/tqdm既有约束，不安装或更换当前环境。
README更正train.py读常量而非config.json、train.py不解析CLI覆盖的旧说明，
补充当前场景、四入口、输出及服务器短测指令。

项目根目录运行，GPU编号按服务器空闲情况选取：

```bash
CUDA_VISIBLE_DEVICES=5 python network/run_loss_vs_epoch.py --device cuda --epochs 1 --steps-per-epoch 2 --out-dir checkpoint/server_smoke_261014
CUDA_VISIBLE_DEVICES=5 nohup python -u network/run_loss_vs_epoch.py --device cuda > nohup_loss_vs_epoch_261014.log 2>&1 &
```

默认正式输出checkpoint/loss_vs_epoch_261014_sigma100_trigger0p5_lambda1_covrows/，
含addcorr/noaddcorr两个目录、loss_by_epoch.csv和loss_vs_epoch.png。
回传完整目录及两种epoch100的last.pt，之后在同一组物理观测上进行三方法PM-PF比较；
run_pm_pf_comparison.py默认仍指向旧260720权重，需显式指定新权重与新输出目录。
