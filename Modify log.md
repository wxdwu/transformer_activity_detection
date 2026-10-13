# Modify Log

260622:
1、事件驱动随机接入建模：
network/data.py # 新增 ACTIVITY_MODE="event"，由外部事件中心触发用户活跃，不再校准整体活跃率为 0.1
事件数量 K_event ~ Poisson(EVENT_LAMBDA)，事件中心在小区圆形区域内均匀采样
事件影响 g_ik = exp(-||u_i-e_k||^2/(2*sigma_event^2))
用户活跃概率 p_i = 1 - (1-background_prob) * Π_k(1-event_trigger_prob*g_ik)

260622:
2、相关矩阵进入 Transformer：
network/model.py # 新增 corr_matrix attention bias 和 logits refinement
attention score 使用 score_ij += alpha * corr_ij
decoder 后使用 logits_refined = logits + gamma * CorrNorm @ logits
alpha 和 gamma 都是可学习参数

260622:
3、训练和评估调用链：
network/train.py # 默认使用 event 数据和 corr_matrix 模型，保存到 checkpoint/event_corrmatrix_260622
network/train.py # loss 权重改为使用 batch 实际活跃比例
network/evaluate.py、network/compare_active_indices.py、CE_methods/compare.py、CE_methods/test_camp_genie_data.py、plot/plot_amp_nmse_vs_iter.py # 前向调用改为 model(x_b, x_y, corr_matrix)

260630:
4、在不改变 loss 口径的基础上增强相关性路径：
network/model.py # 新增 sparsify_correlation_matrix，对 corr_matrix 做 CORR_THRESHOLD 过滤和每用户 CORR_TOPK 邻居保留，降低弱相关边噪声传播
network/model.py # logits refinement 新增 CORR_REFINE_MODE="centered"，使用 logits + gamma*(CorrNorm@logits - mean(logits))，减少 additive refinement 对整体 p_mean 的抬升
network/model.py # 新增 CorrelationFeatureMixer，在 encoder 后、decoder 前做 h_b + tanh(eta)*MLP(CorrNorm@h_b - h_b) 的轻量图消息传递
network/train.py、network/config.py # 新增 CORR_TOPK=24、CORR_THRESHOLD=0.10、CORR_REFINE_MODE="centered"、USE_CORRELATION_FEATURE_MIXER=True、CORR_FEATURE_MIX_INIT=0.1
network/train.py、network/model.py # 当 USE_CORRELATION_ATTENTION_BIAS=False 且 USE_CORRELATION_LOGIT_REFINEMENT=False 时自动关闭 feature mixer，保证不加相关性对照不使用相关性矩阵
CALL_CHAIN.md # 追加 260630 相关性路径改进说明；本次未新增辅助 loss，训练 loss 仍保持 batch 活跃比例 weighted BCE

260707:
5、修正 SNR 噪声生成口径：
network/data.py # 将 NOISE_MODE="snr" 改为 measured SNR：先生成无噪声接收矩阵 BH=(B*A)@H，再按每个样本 mean(|BH|^2) 计算 noise_var=signal_power*10^(-SNR/10)
network/data.py # 保留旧版大尺度近似公式为 NOISE_MODE="large_scale_snr"，用于复现实验或对比旧结果
network/data.py # return_raw=True 时新增 bh，便于检查 measured SNR、信号功率和噪声方差
network/train.py # 更新 NOISE_MODE 注释，说明 snr 为 measured on BAH，large_scale_snr 为旧公式
CALL_CHAIN.md # 追加 260707 噪声生成口径修正说明

260708:
6、新增导频长度 L 变化实验入口：
network/train.py # 将原 main 训练流程封装为 run_training(args)，直接运行 network/train.py 的行为保持不变，新增返回每个 epoch 的 loss/PM/PF/p_mean 历史
network/train.py # GradScaler 新增 PyTorch 版本兼容：优先使用 torch.amp.GradScaler，旧版本回退到 torch.cuda.amp.GradScaler
network/run_varying_l.py # 新增脚本，默认循环 L=[4,6,8,...,30]，分别跑 addcorr/noaddcorr，每个 L 训练到 epoch=100，并记录最终 loss
network/run_varying_l.py # 输出合并 CSV 到 checkpoint/varying_L_260707/final_loss_by_L.csv，使用 mode 列区分 addcorr/noaddcorr；同一张图绘制两条 loss 曲线
network/run_varying_l.py # 支持 --modes addcorr/noaddcorr 调试单条曲线；保留 --mode 作为旧单模式参数；支持 --epochs、--steps-per-epoch、--batch-size、--eval-batches、--device、--pilot-lens
CALL_CHAIN.md # 追加 260707 varying-L 实验脚本说明和运行命令

260708:
7、新增信噪比 SNR 变化实验入口：
network/run_varing_SNR.py # 新增脚本，默认循环 SNR=[10,12,14,16,18,20] dB，分别跑 addcorr/noaddcorr，每个 SNR 训练到 epoch=100，并记录最终 loss
network/run_varing_SNR.py # 输出合并 CSV 到 checkpoint/varying_SNR_260707/final_loss_by_SNR.csv，使用 mode 列区分 addcorr/noaddcorr；同一张图绘制两条 loss 曲线
network/run_varing_SNR.py # 支持 --modes addcorr/noaddcorr 调试单条曲线；保留 --mode 作为旧单模式参数；支持 --epochs、--steps-per-epoch、--batch-size、--eval-batches、--device、--snrs
CALL_CHAIN.md # 追加 260707 varying-SNR 实验脚本说明和运行命令

260708:
8、新增用户数量 N 变化实验入口：
network/run_varing_number_of_users.py # 新增脚本，默认循环 N=[50,100,150,200,250]，分别跑 addcorr/noaddcorr，每个 N 训练到 epoch=100，并记录最终 loss
network/run_varing_number_of_users.py # 输出合并 CSV 到 checkpoint/varying_N_260707/final_loss_by_N.csv，使用 mode 列区分 addcorr/noaddcorr；同一张图绘制两条 loss 曲线
network/run_varing_number_of_users.py # 支持 --modes addcorr/noaddcorr 调试单条曲线；保留 --mode 作为旧单模式参数；支持 --epochs、--steps-per-epoch、--batch-size、--eval-batches、--device、--num-users
CALL_CHAIN.md # 追加 260707 varying-N 实验脚本说明和运行命令

260720:
9、从结构上消除大导频长度下的单协方差 token 压缩瓶颈：
network/model.py # 新增 signal_token_mode="covariance_rows"，将 x_y 恢复为复协方差矩阵并按行构造 [B,Lp,2Lp] 特征，共享投影为 [B,Lp,D] 多个 signal tokens
network/model.py # 新增可学习 covariance_row_position，保留协方差行顺序；原 flat 模式继续使用 2Lp^2->D 的论文单 token 投影
network/model.py # 扩展 HeterogeneousMHA、FFN、归一化和 ContextDecoder 支持 Ty 个 signal tokens；相关性 attention bias 仍只作用于 device-device 区域
network/model.py # decoder 使用全部 signal tokens 作为 queries，分别生成 context 后在 signal-token 维求均值；Feature Mixer 和 Centered Logits Refinement 保持不变
network/model.py # build_model_from_config 对旧 checkpoint 缺失 signal_token_mode 时默认使用 flat，保持旧模型参数和严格加载兼容
network/train.py、network/config.py # 新训练默认 SIGNAL_TOKEN_MODE="covariance_rows"；训练 loss、事件数据和 measured-SNR 噪声逻辑不变
network/run_varying_l.py # 新增 --signal-token-mode flat/covariance_rows，CSV 新增 signal_token_mode 字段；默认输出改为 checkpoint/varying_L_260720_covrows
network/train.py、network/run_varing_SNR.py、network/run_varing_number_of_users.py # 新结构实验使用独立的 260720_covrows 保存目录，避免覆盖既有结果
CALL_CHAIN.md # 追加第14节，记录问题原因、结构公式、调用链、兼容策略、运行命令和计算量变化

260720:
10、新增 loss 随 epoch 变化实验入口：
network/run_loss_vs_epoch.py # 新增脚本，直接继承 train.py 当前参数（包括 LP=10、EPOCHS=100 和 covariance_rows），依次运行 addcorr/noaddcorr
network/run_loss_vs_epoch.py # 逐 epoch 汇总 loss、PM、PF、p_mean、skip、学习率到 checkpoint/loss_vs_epoch_260720_covrows/loss_by_epoch.csv
network/run_loss_vs_epoch.py # 在同一张图绘制 addcorr/noaddcorr 的 epoch 1-100 loss 曲线；有 matplotlib 时输出 PNG，否则回退为 SVG
network/run_loss_vs_epoch.py # 支持 --epochs、--steps-per-epoch、--batch-size、--eval-batches、--device、--seed 和 --modes，便于本地小规模检查
CALL_CHAIN.md # 追加第15节，记录实验参数继承关系、相关性对照设置、输出路径和服务器运行命令

261013:
11、新增传统协方差方法的 Monte Carlo 对比实验：
covariance_based/detector.py # 实现已知大尺度增益下的协方差最大似然检测，活跃变量限制在 [0,1]，使用闭式坐标下降和 Sherman-Morrison 逆矩阵更新
covariance_based/detector.py # 使用 complex128、共同物理功率缩放和每轮逆矩阵重建，返回连续活跃估计、似然历史、轮数和收敛状态
covariance_based/run_monte_carlo.py # 默认继承 train.py 系统设置，每次只调用 sample_batch(1, return_raw=True)，Monte Carlo 循环统计 PM/PF，不训练网络
covariance_based/run_monte_carlo.py # 新增 L、N、SNR 参数扫描、门限扫描、逐次和汇总 CSV、完整配置 JSON，以及可选 PNG/SVG 图，输出到 checkpoint/covariance_261013
covariance_based/run_monte_carlo.py # 默认已知生成器噪声方差并记录 known_generated；不向检测器提供真实标签、活跃数、相关矩阵、事件位置、瞬时信道或活跃概率
covariance_based/test_detector.py # 新增闭式解、约束边界、非正交导频似然下降、物理缩放一致性、全不活跃和随机种子测试
covariance_based/README.md # 记录文献公式、数据一致性、噪声假设、MC 归一化口径、运行命令和输出路径；传统似然不与神经网络 BCE 混作同一 loss
CALL_CHAIN.md # 追加第16节，保留原有文字；记录传统方法调用链、实验设置和验证入口

261013:
12、修复本机协方差实验的 OpenMP 运行库冲突：
covariance_based/run_monte_carlo.py # 在 macOS 上导入 NumPy/PyTorch 前，默认设置 MKL_THREADING_LAYER=SEQUENTIAL，避免重复 OpenMP 初始化导致 OMP: Error #15
covariance_based/__init__.py # 包导入路径同样在加载 NumPy 前设置兼容后端，支持模块运行与包导入
covariance_based/README.md # 更新 cat 环境运行指令、macOS 兼容策略和已有环境设置优先规则；不使用 KMP_DUPLICATE_LIB_OK 强行绕过错误
CALL_CHAIN.md # 追加16.5节；Linux 服务器行为、实验参数、检测算法和用户当前的 MC_TIMES 设置不变
验证 # 在 cat 环境清除外部 MKL_THREADING_LAYER 和 KMP_DUPLICATE_LIB_OK 后，直接脚本完成3次 Monte Carlo 并输出 CSV/PNG/SVG；模块入口完成1次；7项算法单元测试全部通过

261013:
13、协方差实验图采用10的指数次方坐标：
covariance_based/run_monte_carlo.py # 参考本地 HT 论文图4，将 PM-PF 图统一为横轴 PM、纵轴 PF 的双对数图，采用 LogFormatterMathtext 显示10的整数指数次方
covariance_based/run_monte_carlo.py # L/N/SNR 参数扫描图保持横轴线性，PM/PF 纵轴改为对数；默认下界为1e-4，更小的非零概率自动扩展显示范围
covariance_based/run_monte_carlo.py # 0或nan仅在绘图时屏蔽，保留CSV真实数值，不用非零下界替代0；增加主次网格
covariance_based/test_detector.py # 新增对数坐标、PM/PF方向、极小概率和0/nan处理测试
covariance_based/README.md、CALL_CHAIN.md # 更新坐标格式及显示下界说明；检测算法、数据分布和Monte Carlo次数保持不变
covariance_based/run_monte_carlo.py # 参数横轴固定为完整扫描范围及取值，避免0/nan导致横轴自动缩小或出现错误范围
验证 # 10项算法/绘图单元测试通过；L=4/16各2次小规模检查成功；已有single图从CSV重绘，CSV哈希保持不变，PNG视觉检查通过

261013:
14、仅将协方差方法导频长度临时设为8：
covariance_based/run_monte_carlo.py # 新增 PILOT_LEN_OVERRIDE=8，仅在协方差实验配置中覆盖导频长度；设回None即可恢复继承train.py，命令行及L扫描可覆盖此值
covariance_based/README.md、CALL_CHAIN.md # 记录临时覆盖及恢复方法；network/train.py仍为LP=10，其余场景参数和Monte Carlo次数不变

261013:
15、L=8测试后恢复协方差方法的默认导频长度：
covariance_based/run_monte_carlo.py # 将 PILOT_LEN_OVERRIDE 恢复为None，重新继承train.py的LP=10；其余配置及已有实验结果保持不变
covariance_based/README.md、CALL_CHAIN.md # 同步当前默认配置并追加恢复记录
