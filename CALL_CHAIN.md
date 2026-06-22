# 调用链与文件说明

本文档按当前代码版本说明从数据生成、相关性建模、Transformer 前向、训练、评估到信道估计对照的主要调用链。

路径均为仓库相对路径。

## 1. 总体流程

当前工程的核心目标是基于 Heterogeneous Transformer 做用户活动检测，并在原始输入 `x_b, x_y` 的基础上加入空间相关性信息。

训练主链路如下：

1. `network/train.py`
2. 构造 `SystemConfig` 与 `model_cfg`
3. `ActivityDataGenerator.sample_batch()` 生成 `x_b, x_y, label`
4. `build_model_from_config()` 构造 Transformer
5. `model(x_b, x_y)` 输出 `logits, probs`
6. `weighted_activity_loss(logits, label)` 计算损失
7. 反向传播、评估 PM/PF、保存 checkpoint

评估与对照链路如下：

1. `network/evaluate.py` 加载 checkpoint，计算 PM/PF 曲线
2. `CE_methods/compare.py` 加载 checkpoint，使用 Transformer 概率辅助 AMP/LMMSE 等信道估计方法
3. `plot/plot_amp_nmse_vs_iter.py` 可视化 AMP 迭代 NMSE

## 2. 配置入口

### 训练脚本常量

当前 `network/train.py` 主要使用文件顶部的常量作为训练配置：

- `ACTIVITY_MODE`
- `USE_CORRELATION_FEATURE`
- `CORRELATION_ACTIVITY_STRENGTH`
- `CELL_RADIUS_M`
- `ACTIVITY_PROB`
- `N`, `M`, `LP`

位置：

- `network/train.py`

典型消融设置：

```python
ACTIVITY_MODE = "independent"
USE_CORRELATION_FEATURE = False
```

相关性增强设置：

```python
ACTIVITY_MODE = "correlated"
USE_CORRELATION_FEATURE = True
```

### 共享默认配置

`network/config.py` 中的 `DEFAULT_CONFIG` 供 `network/evaluate.py`、`network/compare_active_indices.py`、`plot/plot_amp_nmse_vs_iter.py` 等脚本读取，也作为没有 `config.json` 时的默认配置。

相关字段在：

- `network/config.py`

包括：

- `network_train.system.activity_mode`
- `network_train.system.use_correlation_feature`
- `network_train.system.correlation_activity_strength`
- `network_train.system.cell_radius_m`

注意：仓库根目录目前没有 `config.json`，所以默认会使用 `network/config.py` 里的 `DEFAULT_CONFIG`。

## 3. 数据生成调用链

入口：

- `network/data.py`
- `ActivityDataGenerator.sample_batch(batch_size, return_raw=False)`

主要步骤：

1. 读取 `SystemConfig`
2. 在半径 `cell_radius_m` 的圆形区域内采样用户二维位置 `positions`
3. 由用户位置计算到基站距离 `distances`
4. 由用户间距离计算相关性矩阵 `corr_matrix`
5. 根据 `activity_mode` 生成活动标签 `a`
6. 生成导频 `s`、大尺度衰落 `g`、发射功率控制 `p`、缩放导频 `b`
7. 生成信道 `h` 和噪声 `w`
8. 生成接收信号 `y = B A H + W`
9. 构造模型输入 `x_b, x_y`
10. 返回 batch 字典

### 用户位置

函数：

- `_sample_positions_m(batch_size)`

输出：

- `positions`: `[B, N, 2]`

采样方式为圆盘均匀分布：

- `r = R * sqrt(u)`
- `theta ~ Uniform(0, 2pi)`

其中 `R = cell_radius_m`，当前默认值为 `500.0` 米。

### 空间相关性

函数：

- `_correlation_from_positions(positions)`

对任意两个用户 `i, j`，先计算欧氏距离 `d(i,j)`，再归一化为：

```text
corr(i,j) = 1 - d(i,j) / (2R)
```

并裁剪到 `[0,1]`。由于圆形区域内最大用户间距离为 `2R`，该归一化与 500 米区域半径一致。

输出：

- `corr_matrix`: `[B, N, N]`
- `corr_feature`: `[B, N]`

其中 `corr_feature` 是每个用户与其他用户相关性的平均值，用作额外输入特征。

### 活跃标签

函数：

- `_sample_activity(sim)`

两种模式：

- `activity_mode="independent"`：每个用户独立按 `activity_prob` 采样。
- `activity_mode="correlated"`：先生成 seed 活跃用户，再用相关性矩阵计算邻居活跃强度，使空间相关用户更容易同时活跃。

相关强度由：

- `correlation_activity_strength`

控制。当前默认值为 `0.8`。

### Transformer 输入特征

`x_b` 来自缩放导频矩阵 `b` 的实部/虚部拼接：

```text
x_b baseline: [B, N, 2Lp]
```

如果：

```python
use_correlation_feature = True
```

则在最后一维追加 `corr_feature`：

```text
x_b correlation-enhanced: [B, N, 2Lp + 1]
```

`x_y` 来自接收信号协方差：

```text
C = Y Y^H / M
x_y = concat(real(vec(C)), imag(vec(C)))
x_y: [B, 2Lp^2]
```

返回的基础 batch：

- `x_b`
- `x_y`
- `label`

如果 `return_raw=True`，额外返回：

- `y`
- `b`
- `s`
- `pg`
- `beta`
- `h`
- `noise_var`
- `positions`
- `corr_matrix`
- `corr_feature`

## 4. 模型构建调用链

入口：

- `network/model.py`
- `build_model_from_config(cfg)`

该函数根据 `cfg["model_name"]` 构造：

- `HeterogeneousTransformer`
- `HeterogeneousTransformerLargeDim`

共同关键参数：

- `num_devices`
- `pilot_len`
- `use_correlation_feature`
- `dim`
- `num_layers`
- `num_heads`
- `head_dim`
- `ff_dim`
- `norm_type`

### 输入维度匹配

模型中 `embed_b` 的输入维度由 `use_correlation_feature` 决定：

```python
b_input_dim = 2 * pilot_len + (1 if use_correlation_feature else 0)
```

因此：

- `use_correlation_feature=False` 时，模型期望 `x_b[..., 2Lp]`
- `use_correlation_feature=True` 时，模型期望 `x_b[..., 2Lp+1]`

训练、评估和 checkpoint 必须保持这个字段一致。旧 checkpoint 如果是无相关性特征训练的，不能直接加载到启用相关性特征的新模型中。

## 5. Transformer 前向调用链

入口：

- `HeterogeneousTransformer.forward(x_b, x_y)`

步骤：

1. `embed_b(x_b)` 将 N 个用户导频 token 投影到 `D` 维
2. `embed_y(x_y).unsqueeze(1)` 将接收协方差投影成 1 个接收 token
3. 多层 `HeterogeneousEncoderLayer` 提取用户 token 与接收 token 间的上下文关系
4. `ContextDecoder` 输出每个用户的活动 logits 和概率

输出：

- `logits`: `[B, N]`
- `probs`: `[B, N]`

## 6. 异构编码器内部链路

文件：

- `network/model.py`

主要类：

- `HeterogeneousEncoderLayer`
- `HeterogeneousMHA`
- `HeterogeneousFFN`

每层执行：

1. 对用户 token 使用一套 Q/K/V 投影
2. 对接收 token 使用另一套 Q/K/V 投影
3. 拼接用户 token 和接收 token 做 scaled dot-product attention
4. 根据 token 类型使用不同输出投影
5. 残差连接和归一化
6. 用户 token 与接收 token 分别进入各自 FFN
7. 再次残差连接和归一化

这就是 Heterogeneous Transformer 的核心：不同物理意义的 token 使用不同参数，但仍在同一个 attention 空间里交互。

## 7. 解码器调用链

文件：

- `network/model.py`
- `ContextDecoder`

步骤：

1. 最终接收 token 作为 query
2. 用户 token 和接收 token 共同作为 key/value
3. 计算 context vector
4. 将 context 与每个用户 token 做匹配
5. 输出每用户 `logits`
6. `sigmoid(logits)` 得到活动概率 `probs`

## 8. 训练调用链

入口：

- `network/train.py`

主要流程：

1. `build_args()` 读取文件顶部常量
2. `build_system_config(args)` 构造 `SystemConfig`
3. `build_model_config(args)` 构造模型配置
4. `ActivityDataGenerator(system_cfg, device)` 创建数据生成器
5. `build_model_from_config(model_cfg)` 创建模型
6. 每个 step 调用 `data_gen.sample_batch(args.batch_size)`
7. 前向得到 `logits, probs`
8. `weighted_activity_loss(logits, targets, activity_prob)` 计算损失
9. AdamW 优化
10. 每个 epoch 评估 PM/PF
11. 保存：
    - `last.pt`
    - `best_pm.pt`

checkpoint 中保存：

- `model_state`
- `config`
- `model_config`
- `system_config`
- `epoch`

其中 `model_config.use_correlation_feature` 和 `system_config.use_correlation_feature` 对后续加载很重要。

## 9. 损失与指标

### 损失

文件：

- `network/losses.py`

函数：

- `weighted_activity_loss`

作用：

- 对稀疏活动检测使用加权 BCE
- `activity_prob` 控制正负样本权重

### PM/PF 指标

文件：

- `network/metrics.py`

函数：

- `pm_pf_at_threshold`
- `pm_pf_curve`

PM 表示漏检概率，PF 表示虚警概率。

## 10. 评估调用链

入口：

- `network/evaluate.py`

流程：

1. 读取配置
2. 加载 checkpoint
3. 从 checkpoint 读取 `system_config`
4. 从 checkpoint 读取 `model_config`
5. 重建模型并加载权重
6. 用同样系统配置生成测试 batch
7. 前向得到 `probs`
8. 计算 PM/PF 曲线
9. 导出 CSV

因为评估使用 checkpoint 中的配置重建模型，所以只要 checkpoint 是用当前相关性配置训练出来的，评估会自动保持维度一致。

## 11. 活跃索引比较调用链

入口：

- `network/compare_active_indices.py`

流程：

1. 加载 checkpoint
2. 重建模型
3. 生成测试 batch
4. 得到 `probs`
5. 用阈值或 top-k 转为预测活跃索引
6. 与真实 `label` 活跃索引比较
7. 输出漏检和虚警用户编号

## 12. 信道估计对照调用链

入口：

- `CE_methods/compare.py`

主要依赖：

- `CE_methods/estimators.py`
- `network/data.py`
- `network/model.py`
- `network/metrics.py`

流程：

1. 从配置读取 CE 对照参数
2. 加载 Transformer checkpoint
3. 用 checkpoint 中的 `system_config` 创建 `ActivityDataGenerator`
4. `sample_batch(return_raw=True)` 生成神经网络输入和原始物理量
5. Transformer 输出每用户概率 `probs`
6. 概率通过阈值或 top-k 转为活跃集合
7. 对每个样本执行：
   - `transformer+AMP`
   - `CAMP`
   - `detected+LMMSE`
   - `oracle_AMP`
   - `oracle_LMMSE`
8. 汇总 PM/PF 和 NMSE
9. 写入报告文件

相关性增强不会直接改变 CAMP/LMMSE 的公式，但会改变 Transformer 给出的活动概率和活跃集合，从而影响下游信道估计效果。

## 13. AMP 曲线绘图调用链

入口：

- `plot/plot_amp_nmse_vs_iter.py`

流程：

1. 加载 checkpoint
2. 生成 `return_raw=True` 的测试 batch
3. Transformer 输出概率
4. 将概率作为 CAMP 的 soft lambda
5. 与固定 lambda 的 CAMP 对比
6. 画出 NMSE 随迭代次数变化的曲线

## 14. 关键张量形状

设：

- `B`: batch size
- `N`: 用户数
- `Lp`: 导频长度
- `M`: 天线数
- `D`: Transformer 嵌入维度

数据生成：

```text
positions:    [B, N, 2]
corr_matrix:  [B, N, N]
corr_feature: [B, N]
s:            [B, Lp, N]
b:            [B, Lp, N]
y:            [B, Lp, M]
h:            [B, N, M]
label:        [B, N]
```

模型输入：

```text
x_b without correlation: [B, N, 2Lp]
x_b with correlation:    [B, N, 2Lp+1]
x_y:                     [B, 2Lp^2]
```

模型内部：

```text
h_b: [B, N, D]
h_y: [B, 1, D]
```

模型输出：

```text
logits: [B, N]
probs:  [B, N]
```

## 15. 常用文件索引

- 训练入口：`network/train.py`
- 数据生成：`network/data.py`
- 模型实现：`network/model.py`
- 损失函数：`network/losses.py`
- PM/PF 指标：`network/metrics.py`
- 共享配置：`network/config.py`
- PM/PF 评估：`network/evaluate.py`
- 活跃索引比较：`network/compare_active_indices.py`
- 信道估计对照：`CE_methods/compare.py`
- AMP/LMMSE 实现：`CE_methods/estimators.py`
- AMP NMSE 绘图：`plot/plot_amp_nmse_vs_iter.py`
- Matlab 参考：`AMP_Genie/`, `AMP_liuliang/`

## 16. 使用建议

建议至少跑两组实验做消融：

### Baseline

```python
ACTIVITY_MODE = "independent"
USE_CORRELATION_FEATURE = False
```

这对应用户活跃独立，Transformer 输入不包含相关性标量。

### Correlation-Enhanced

```python
ACTIVITY_MODE = "correlated"
USE_CORRELATION_FEATURE = True
```

这对应用户活跃具有空间相关性，且 Transformer 的 `x_b` 每个用户 token 额外包含一个相关性摘要特征。

两组实验需要分别训练 checkpoint，再分别运行评估脚本比较 PM/PF 曲线。不要把旧 checkpoint 直接加载到输入维度不同的新模型中。

260620:
1、消融对比：
train.py line 35-36 # 数据里是否加入活跃相关性，模型输入里是否加入相关性特征
activity_mode="correlated", use_correlation_feature=False 
activity_mode="correlated", use_correlation_feature=True
attention: train.py line 59 change save path