# 调用链与文件说明（项目概览）

本文档概述仓库“Correlated-Activation”的主要执行流程（call chain），并简要说明关键文件/模块的职责，方便快速定位训练、推断、评估与传统方法比较的实现位置。

**注**：下列路径均为工作区相对路径。

## 1. 入口脚本
- **训练**：[network/train.py](network/train.py#L1)
  - 构建训练参数（`build_args()`）并生成 `SystemConfig`、模型配置。
  - 使用 `ActivityDataGenerator`（来自 [network/data.py](network/data.py#L1)）生成训练/评估数据批次。
  - 使用 `build_model_from_config`（来自 [network/model.py](network/model.py#L1)）创建模型并移动到设备。
  - 前向：模型接收 `x_b`（设备导频特征）和 `x_y`（接收信号协方差特征），返回 `logits, probs`。
  - 损失：调用 [network/losses.py](network/losses.py#L1) 中的 `weighted_activity_loss` 计算加权二元交叉熵（适应稀疏活动概率）。
  - 优化步骤、混合精度、学习率调度、检查点保存。

- **评估**：[network/evaluate.py](network/evaluate.py#L1)
  - 从 checkpoint 加载模型参数、构建模型与 `ActivityDataGenerator`。
  - 运行若干批次得到 `probs` 与 `labels`，使用 [network/metrics.py](network/metrics.py#L1) 计算 PM/PF 曲线并导出 CSV。

## 2. 数据生成与预处理
- [network/data.py](network/data.py#L1)
  - `SystemConfig`：系统参数容器（用户数 N、天线 M、导频长度、噪声模式、SNR 等）。
  - `ActivityDataGenerator.sample_batch()`：按物理信道模型生成复杂导频 `s`、大尺度衰落 `g`、缩放导频 `b`、活动标签、信道 `h`、接收信号 `y`。
  - 生成用于模型输入的实部/虚部拼接特征 `x_b`（每用户导频）和 `x_y`（向量化协方差），并做 RMS 归一化。

## 3. 模型（神经网络）
- [network/model.py](network/model.py#L1)
  - 实现论文的 Heterogeneous Transformer：
    - `embed_b` / `embed_y`：分别把导频 token 与接收协方差 token 映射到嵌入空间。
    - 多层 `HeterogeneousEncoderLayer`（含 `HeterogeneousMHA` 与 `HeterogeneousFFN`），分别对设备 token 与接收 token 使用不同的投影/FFN/归一化参数。
    - `ContextDecoder`：将最终的接收 token 用作 query，计算 context 并对每个设备输出匹配 logits，再用 sigmoid 得到活动概率 `probs`。
  - 提供 `build_model_from_config(cfg)` 工厂函数，支持不同规模（baseline 与 large_dim）。

## 4. 损失与指标
- [network/losses.py](network/losses.py#L1)
  - `weighted_activity_loss`：对稀疏活动分布加权的 BCE 损失（论文式权重）。
- [network/metrics.py](network/metrics.py#L1)
  - `pm_pf_at_threshold`、`pm_pf_curve`：计算 Miss Prob (PM) 与 False alarm (PF) 指标与曲线。

## 5. 传统方法与对照（非神经网络）
- `CE_methods/` 目录：实现并比较经典的活动检测与信道估计方法。
  - [CE_methods/estimators.py](CE_methods/estimators.py#L1)：
    - `active_indices_from_probs`：把概率转为活跃索引（阈值或 top-k）。
    - `lmmse_channel_estimate`, `lmmse_formula_estimate`：基于检测集合做 LMMSE 信道估计。
    - 多个 AMP / CAMP 的实现（`thresh_prime_thresh_complex_gaussian`, `noisy_camp_mmse` 等），用于对照实验。
  - [CE_methods/compare.py](CE_methods/compare.py#L1)（及其它同目录脚本）：负责把神经网络输出与传统方法的结果进行比较、报告与可视化。

## 6. Matlab 仿真参考实现
- `AMP_Genie/` 与 `AMP_liuliang/`：包含若干 Matlab 脚本（`.m`），用于对照传统 AMP 实验和生成参考数据/报告（如 `CAMP_Genie.m` 等）。

## 7. 配置管理与实用脚本
- [network/config.py](network/config.py#L1)
  - 默认实验配置 `DEFAULT_CONFIG` 与 `load_experiment_config()`。
  - `section_namespace`、`apply_cli_overrides`：将 JSON 配置与 CLI 参数统一为 Namespace，供 `train.py` / `evaluate.py` 使用。

- `plot/` 目录：包含绘图脚本，例如 `plot_amp_nmse_vs_iter.py`、`plot_loss_curve.py`，用于把实验结果可视化。

## 8. 检查点与输出
- `checkpoint/`：训练过程中保存的 checkpoint（`last.pt`, `best_pm.pt` 等）和实验报告文本。

## 9. 典型调用链（从训练到评估）
1. `python network/train.py`（或在 IDE 中运行 `main()`）
2. `train.py` 调用 `build_args()`，构建 `SystemConfig`（`network/data.py`）与模型配置（`network/config.py` / 内联常量）。
3. 创建 `ActivityDataGenerator` 并在每个训练 step 调用 `sample_batch()` 生成 `x_b, x_y, label`。
4. 调用 `build_model_from_config()` 构造模型（`network/model.py`），把输入送入模型得到 `logits, probs`。
5. 计算 `weighted_activity_loss(logits, label)` 并反向传播、优化参数；按 epoch 保存 checkpoint。
6. 使用 `network/evaluate.py` 加载 checkpoint，重建模型与 `ActivityDataGenerator`，批量推理，得到概率与标签，计算 PM/PF 曲线并写入 CSV。
7. 如需对照实验，使用 `CE_methods/` 中的函数把网络 `probs` 转为索引，进行 LMMSE 或 AMP 信道估计，并把结果写入报告文件。

## 10. 快速文件索引（常查）
- 训练入口：[network/train.py](network/train.py#L1)
- 模型实现：[network/model.py](network/model.py#L1)
- 数据生成：[network/data.py](network/data.py#L1)
- 损失：[network/losses.py](network/losses.py#L1)
- 指标：[network/metrics.py](network/metrics.py#L1)
- 配置管理：[network/config.py](network/config.py#L1)
- 评估脚本：[network/evaluate.py](network/evaluate.py#L1)
- 传统方法：[CE_methods/estimators.py](CE_methods/estimators.py#L1)
- Matlab 参考：`AMP_Genie/`, `AMP_liuliang/`

---

如果你希望我把这份文档扩展为更详尽的调用序列图（例如按函数调用链逐行追踪），或把每个主要函数/类的参数与输入输出示例加入文档，我可以继续把 `CALL_CHAIN.md` 拓展为更详细的技术手册。
