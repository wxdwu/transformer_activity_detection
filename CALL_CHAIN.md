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

## 基于 Transformer 的随机接入 — 详细 Call Chain
下面给出从生成接入样本到输出用户活动概率的逐步调用链（call chain），对应本工程中具体的文件/函数，便于快速跟踪执行流。

1. 启动入口
  - 脚本：`network/train.py`（或 `network/evaluate.py` 用于评估）
  - 作用：构建参数（`build_args()`）、创建 `SystemConfig`、初始化模型与数据生成器，进入训练/评估循环。

2. 数据生成（物理层建模）
  - 文件：`network/data.py` → 类 `ActivityDataGenerator.sample_batch()`。
  - 步骤：
    - 生成复杂随机导频矩阵 `s`，模拟大规模衰落 `g` 与缩放因子 `scale`，得到缩放导频 `b`。
    - 按伯努利采样生成活动标签 `a`（稀疏随机接入）。
    - 生成真实信道 `h` 与接收信号 `y = B A H + W`。
    - 构造模型输入：`x_b`（每用户的实/虚部导频特征）与 `x_y`（向量化接收协方差的实/虚部）。

3. 特征归一化与批次准备
  - 文件：`network/data.py` 内实现。
  - 作用：对 `x_b`、`x_y` 做 RMS 归一化并转换为浮点张量，返回 `label` 供训练计算损失。

4. 嵌入层（token 化）
  - 文件：`network/model.py` → `HeterogeneousTransformer.embed_b` 与 `embed_y`。
  - 作用：将 `x_b`（N 个设备 token）映射到维度 `D`，将 `x_y`（接收 token）映射并扩维为 1 个 token。

5. 异构编码器（多层）
  - 文件：`network/model.py` → `HeterogeneousEncoderLayer`（包含 `HeterogeneousMHA` 与 `HeterogeneousFFN`）。
  - 作用：
    - 对设备 token 与接收 token 使用不同的 Q/K/V 投影（heterogeneous attention），计算跨 token 的注意力权重。
    - 每层执行残差连接与各自的归一化（BatchNorm/LayerNorm），以及 token 特定的前馈网络。
    - 堆叠 L 层以逐步提取跨设备与接收统计间的上下文信息。

6. 上下文解码与概率输出
  - 文件：`network/model.py` → `ContextDecoder`。
  - 作用：使用最终接收 token 作为 query，结合设备 token 计算 context 向量；将 context 与每个设备 token 匹配，输出 logits；通过 sigmoid 得到每个用户的活动概率 `probs`。

7. 损失计算与权重调整
  - 文件：`network/losses.py` → `weighted_activity_loss()`。
  - 作用：按稀疏活动先验对正负样本加权计算二元交叉熵，稳定训练并聚焦正确率与召回的权衡。

8. 反向传播与优化
  - 文件：`network/train.py` 中的训练循环。
  - 细节：支持 AdamW 优化器、可选混合精度（AMP）、梯度裁剪、学习率预热与衰减，按步保存 `last.pt` 与最优指标对应的 `best_pm.pt`。

9. 评估与指标
  - 文件：`network/evaluate.py` 与 `network/metrics.py`。
  - 作用：加载 checkpoint，重建数据生成器与模型，批量推理得到 `probs`，计算 PM/PF 曲线并导出 CSV，便于画 ROC/性能曲线。

10. 下游处理与对照实验
   - 文件：`CE_methods/estimators.py` 等。
   - 作用：把网络输出的 `probs` 转换为活跃索引（阈值或 top-k），并在已检测集合上执行 LMMSE 或 AMP 信道估计；与传统方法对比以评估检测/信道估计性能。

11. 日志、检查点与可重复性
   - 训练脚本会写入 `args`、`system_cfg` 与 `model_cfg` 到 `train.log`，并把检查点保存在 `checkpoint/` 目录，便于复现实验与后续评估。

## 参考对应文件（快速定位）
- 训练入口：[network/train.py](network/train.py#L1)
- 数据生成：[network/data.py](network/data.py#L1)
- 模型实现：[network/model.py](network/model.py#L1)
- 损失：[network/losses.py](network/losses.py#L1)
- 指标：[network/metrics.py](network/metrics.py#L1)
- 评估：[network/evaluate.py](network/evaluate.py#L1)
- 传统方法对照：[CE_methods/estimators.py](CE_methods/estimators.py#L1)

