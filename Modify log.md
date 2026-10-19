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

261014:
16、新增协方差方法的论文参数事件驱动临时测试：
covariance_based/run_monte_carlo.py # 启用 USE_PAPER_EVENT_TEST_SETTINGS，临时使用N=100、L=8、半径250米、thermal噪声和event活跃；固定噪声为-169 dBm/Hz乘10 MHz，即-99 dBm
covariance_based/run_monte_carlo.py # 保留事件到达参数、120米影响尺度、导频归一化及其他未指定参数；命令行配置优先；Transformer配置不变
covariance_based/run_monte_carlo.py # 测试输出独立保存到checkpoint/covariance_261014_paper_event；关闭测试开关即可恢复原系统参数继承和输出路径
covariance_based/run_monte_carlo.py # thermal模式拒绝无效的SNR扫描，避免只改变snr_db却不改变噪声功率
covariance_based/test_detector.py # 新增临时预设、恢复继承、命令行优先级及SNR扫描有效性测试
covariance_based/README.md、CALL_CHAIN.md # 记录实验参数、固定噪声功率、输出路径和恢复方法；本次不改变导频方差，不属于完整论文复现
验证 # 13项单元测试通过；2次小规模Monte Carlo正常输出CSV/PNG/SVG；逐次噪声方差均为约1.2589e-13 W；Transformer及已有实验结果未改动

261014:
17、新增独立活跃概率0.1的协方差参考测试：
covariance_based/run_independent_reference.py # 新增独立入口，复用论文参数预设和原Monte Carlo流程，仅将活跃模式设为independent、概率设为0.1
covariance_based/run_independent_reference.py # 默认保留N=100、L=8、半径250米、固定热噪声，结果保存到checkpoint/covariance_261014_paper_independent/single；不修改事件测试开关或Transformer
covariance_based/test_detector.py # 新增参考物理参数一致性、预设开关无关性、独立输出路径和命令行优先级测试
covariance_based/README.md、CALL_CHAIN.md # 记录调用链、运行命令、输出路径及负载差异；不将本次测试表述为隔离相关性的对照或完整论文复现
验证 # 15项单元测试通过；本机完成1000次参考实验，PM@0.5=0.178872、PF@0.5=0.006429、平均活跃比例0.10091，全部收敛；CSV统计及噪声方差验证通过，事件结果哈希不变，PNG视觉检查通过

261014:
18、将独立参考入口升级为HT论文图4的协方差复现：
covariance_based/run_independent_reference.py # 独立NumPy生成器使用论文的单位方差CN(0,1)导频、小尺度信道、固定热噪声、独立Bernoulli(0.1)活跃及功率控制，不修改Transformer和事件数据
covariance_based/run_independent_reference.py # 默认N=100、半径250米、pmax=23 dBm、(L,M)=(7,32)/(8,32)/(8,64)，每组5000次；结果独立保存到checkpoint/covariance_261014_paper_reproduction
covariance_based/run_independent_reference.py # 复用已知增益且活跃变量限于[0,1]的协方差MLE；默认500轮/1e-6容差，前3次通过逐坐标直接求逆独立交叉检查，未向求解器提供标签或活跃先验
covariance_based/run_independent_reference.py # 新增密集门限、真实0/1边界统计、合并三曲线PNG/SVG、每组逐次CSV及估计/标签NPZ；记录论文未给出的初始化、停止准则等复现选择，不伪造概率来贴合原图
covariance_based/test_detector.py、covariance_based/test_paper_reference.py # 更新独立入口配置测试，新增导频/噪声分布、物理恒等式、配对实例、直接求逆核验、统计回读和三曲线绘图测试
covariance_based/README.md、CALL_CHAIN.md # 同步入口、参数、调用链、运行命令、输出路径及复现限制，CALL_CHAIN保留原记录并追加16.11节
验证 # 三组各5000次独立活跃实验完成，15000次均收敛；PM/PF@0.5为(7,32)0.261649/0.012074、(8,32)0.163625/0.005530、(8,64)0.072669/0.002429；计数、门限、噪声及配对数据核对通过；9次直接求逆检查最大分数差5.22e-15；已有结果及Transformer/事件代码哈希不变，曲线尚未与原图重合

261014:
19、区分论文复现中的独立活跃与固定活跃数：
covariance_based/run_independent_reference.py # 新增条件统计函数，从5000次独立样本中选取真实K=10的669次，输出独立诊断CSV/PNG/SVG/配置；仅用于评价，不向检测器提供真实K，不替换主结果
covariance_based/run_independent_reference.py # 新增--active-count K，均匀采样固定K子集，明确记录fixed_count和边缘概率K/N；默认仍为独立Bernoulli，不将固定K表述为独立采样
covariance_based/run_independent_reference.py # 固定K=10实验默认保存至checkpoint/covariance_261014_paper_fixedK10，物理随机量与独立实验配对；不修改原协方差检测器或Transformer/事件参数
covariance_based/test_paper_reference.py # 新增条件统计、固定K边界/均匀性和物理配对检查；保证统计选样不影响检测器输入
covariance_based/README.md、CALL_CHAIN.md # 记录联合活跃分布的复现歧义、子集样本数、显式固定K选项和独立输出路径；子集图更接近原图属于线索，不作为作者设置已确认的结论
验证 # 24项测试、语法编译及diff检查通过；固定K三组各5000次完成并全部收敛，PM/PF@0.5为(7,32)0.243220/0.007791、(8,32)0.122220/0.002109、(8,64)0.031120/0.000616；PF<=0.01时最小PM为0.197140/0.028160/0.001820；CSV/NPZ核对通过，共有669个K=10实例与独立实验估计完全相同；固定K主要曲线更接近论文，但端点尚有差异，未宣称作者设置已确认或精确复现原图

261014:
20、清理已替代的旧独立活跃参考结果：
checkpoint/covariance_261014_paper_independent/ # 按用户要求删除旧1000次测试的CSV、JSON及PNG/SVG；旧测试使用N=100、L=8、M=32、独立p=0.1及CN(0,1/L)导频，不属于完整论文复现
covariance_based/README.md、CALL_CHAIN.md # 同步旧目录已删除状态，CALL_CHAIN追加16.13节并保留历史文字；当前参考入口不依赖旧结果，算法、参数和源代码不变
checkpoint/covariance_261014_paper_reproduction/、checkpoint/covariance_261014_paper_fixedK10/、checkpoint/covariance_261014_paper_event/ # 保留当前独立、固定K及事件实验结果
验证 # 旧目录已不存在，三个保留目录的summary.csv哈希删除前后一致；24项单元测试及git diff --check通过，删除不影响当前运行入口

261014:
21、协方差Monte Carlo恢复与Transformer一致的事件场景：
covariance_based/run_monte_carlo.py # 关闭USE_PAPER_EVENT_TEST_SETTINGS，保留PILOT_LEN_OVERRIDE=None，恢复完整继承train.py的N=200、M=32、L=10、500米半径、event及measured-SNR 20 dB配置
covariance_based/run_monte_carlo.py # 默认MC_TIMES改为5000；输出改为checkpoint/covariance_261014_transformer_event，避免覆盖旧结果；求解器100轮/1e-4容差及known_generated噪声方差假设保持不变
covariance_based/test_detector.py # 新增默认场景与train.py全字段一致、MC次数及共享生成器实测接收功率噪声检查
covariance_based/README.md、CALL_CHAIN.md # 更新默认设置、运行命令、输出路径和接收端信息假设；CALL_CHAIN保留历史内容并追加16.14节
保留 # run_independent_reference.py、原检测器、Transformer/共享数据生成器及既有论文复现结果不修改
验证 # 26项测试、语法编译和diff检查通过；正式5000次完成并全部收敛，PM@0.5=0.470537、PF@0.5=0.026875、平均活跃比例0.084461、平均24.061轮；逐次计数、门限曲线、完整配置及前30个物理样本重放一致，PNG视觉检查通过，受保护代码及已有独立活跃和固定K论文复现结果哈希不变

261014:
22、新增GI-HT、HT和协方差方法的共享样本PM-PF对比：
network/run_pm_pf_comparison.py # 加载loss-vs-epoch的第100个epoch完整last.pt，校验场景与非几何网络设置一致；GI-HT启用三几何分支，HT全部关闭且不传入G，不训练或改变loss
network/run_pm_pf_comparison.py # CPU逐次生成5000个共享事件观测；两网络批量float32推理，协方差同观测重新求解，保留已知噪声与原求解器设置
network/run_pm_pf_comparison.py # 推理batch默认继承训练权重的128；现有TokenBatchNorm在eval仍用当前batch统计，因此两模型采用相同分组，尾批8个保留并记录；不改网络归一化
network/run_pm_pf_comparison.py # 1001门限统一扫描，输出三曲线双对数PNG/SVG/PDF、CSV、共享标签/分数NPZ、权重和观测哈希；保存到checkpoint/pm_pf_comparison_261014，原结果不覆盖
network/test_pm_pf_comparison.py # 新增门限计数、0/1端点、场景和epoch一致性、损坏权重提示、HT不接收G、推理batch不改变样本和三曲线绘图测试
covariance_based/README.md、CALL_CHAIN.md # 记录入口、运行命令、统一测试样本和已知噪声假设；CALL_CHAIN追加16.15节，保留历史文字
权重 # 项目内loss_vs_epoch两份last.pt缺少ZIP central directory；All experiments results/results_260720下完整备份可严格加载，使用命令行路径只读备份，不修改原文件
验证 # 34项单元测试、语法编译和diff检查通过；16次真实权重小规模测试及正式5000次完成，协方差全部收敛；GI-HT/HT/Covariance的PM/PF@0.5为0.053445/0.095319、0.072116/0.170757、0.470537/0.026875
结果 # PF<=0.05网格点的最低PM分别为0.132523、0.333692、0.330922，对应实际PF为0.049932、0.049680、0.049983，不通过插值修改概率；不同算法不以共同0.5阈值代替同虚警率比较
复核 # 三方法共享84461个活跃及915539个不活跃实例，NPZ与CSV统计一致，门限曲线单调及端点正确；协方差与旧5000次逐次结果和共有101个门限完全一致，PNG视觉检查通过；模型、生成器、检测器、论文复现入口及旧结果哈希不变

261014:
23、降低单事件触发人数和事件到达参数并单独测试协方差：
network/train.py、network/data.py、network/config.py # 同步默认EVENT_TRIGGER_PROB=0.6、EVENT_LAMBDA=0.5；保留响应尺度120米、最多3事件、背景0.005及其他场景参数，不更改生成公式或固定K
network/train.py及四个训练run入口 # 默认训练/绘图目录改为261014_low_load_covrows标记，保留原路径下历史实验；本机不训练Transformer
covariance_based/run_monte_carlo.py # 输出改为checkpoint/covariance_261014_low_load_event；新增activity_count_stats.json，从逐次真实K记录范围、均值、分位数、空场景和直方图，不改变检测器输入、统计口径或求解器
covariance_based/test_detector.py # 新增三处事件默认值一致性、事件峰值/空间尺度与背景公式、几何矩阵不变、K统计与多实验点汇总测试
covariance_based/README.md、CALL_CHAIN.md # 同步低负载配置、运行命令和输出；CALL_CHAIN追加16.16节并保留历史文字；旧模型权重仍属于旧场景，后续需重新训练才能作低负载同场景对比
保留 # run_independent_reference.py、网络结构、loss、检测器、原权重和所有旧实验结果不修改；本次只跑协方差5000次并单独绘图
验证 # 38项单元测试、语法编译及diff检查通过，12次小规模运行和四个run继承检查通过；正式协方差5000次全部收敛，平均12.3002轮；PM/PF@0.5为0.267636/0.003791，活跃32533、非活跃967467、漏检8707、虚警3668
活跃人数 # K实际范围0至47、平均6.5066、中位数2、90/95/99分位18/22/33；空场景1069次，非空3931次的K为1至47、平均8.276011；该范围不是硬上限，模型仍允许0至N
复核 # 新旧系统配置仅两个事件参数不同；CSV与汇总、直方图/分位数、门限曲线及前30个物理样本重放一致；PNG视觉检查通过；网络、loss、检测器、论文复现入口和旧结果哈希不变；未训练Transformer
曲线 # 阈值0时PM=0.083208、PF=0.067051，当前规则下仍无PM=PF交点；不强制调整错误率以匹配图形

261014:
24、恢复事件到达参数λ=1并再次测试协方差：
network/train.py、network/data.py、network/config.py # EVENT_LAMBDA由0.5恢复1.0，EVENT_TRIGGER_PROB仍为0.6；其他场景参数及生成公式不变
covariance_based/run_monte_carlo.py # 默认输出改为checkpoint/covariance_261014_low_load_lambda1_event；本次5000次、1001门限，单独画协方差图并统计真实活跃人数
network/train.py及四个训练run # 默认输出目录加入low_load_lambda1标记，保留λ=0.5及旧场景结果；不执行Transformer训练
covariance_based/test_detector.py、README.md、CALL_CHAIN.md # 同步新默认值、目录及调用说明；CALL_CHAIN仅追加16.17，已有修改日志内容保留
保留 # 网络、loss、检测器、run_independent_reference.py、全部已有权重和结果不修改
验证 # 38项测试、语法编译及四个run继承检查通过；用户中断时任务已完成5000次且全部收敛，平均19.878轮；K范围0至53、平均11.753、PM/PF@0.5为0.366426/0.011948；用户随后要求改为1000次，原结果保留，当前测试见记录25

261014:
25、单独协方差测试默认改为1000次并重跑：
covariance_based/run_monte_carlo.py # MC_TIMES=1000；场景仍为λ=1、峰值触发0.6，其他场景与算法不变；新输出checkpoint/covariance_261014_low_load_lambda1_mc1000_event，不覆盖已完成5000次结果
covariance_based/test_detector.py # 更新默认MC次数和输出根检查
network/run_pm_pf_comparison.py、network/test_pm_pf_comparison.py # 联合正式对比独立保留MC_TIMES=5000，避免跟随协方差小规模测试次数改变；新增两入口默认次数独立测试
covariance_based/README.md、CALL_CHAIN.md # 同步当前1000次测试命令和输出，CALL_CHAIN追加16.18；原5000次已完成记录保留，不把其结果冒充1000次
保留 # EVENT_LAMBDA=1、EVENT_TRIGGER_PROB=0.6、网络、loss、检测器、论文复现入口及已有结果不修改；不训练Transformer
验证 # 39项单元测试、语法编译及diff检查通过；实际重跑1000次，全部收敛，平均20.769轮；PM/PF@0.5为0.383208/0.013328，活跃12351、非活跃187649、漏检4733、虚警2501
活跃人数 # 本次K范围0至52、平均12.351、中位数11，90/95/99分位29/33/约42.01；132次空场景，868次非空场景平均K为14.229263；范围不是硬限制
复核 # 计数、K统计/直方图、曲线单调性和端点、前30个物理样本重放通过；所有1000次非耗时字段与旧5000次的前1000次一致，PNG视觉检查通过；已有5000次及其他受保护结果哈希未变；未训练Transformer
最终核对 # 原λ=1的5000次目录在完成上述配对及哈希核验后已不在文件系统中；本次没有执行删除操作，不自动恢复该旧目录；新1000次完整结果仍存在，旧数值作为历史验证记录保留

261014:
26、缩小事件响应尺度σ为80米并测试协方差：
network/train.py、network/data.py、network/config.py # EVENT_SIGMA_M由120改为80；λ=1、峰值触发0.6、背景0.005、最多3事件及N/L/M/SNR等其余设置不变，不新增固定K或硬覆盖约束
covariance_based/run_monte_carlo.py # 1000次测试默认输出改为checkpoint/covariance_261014_sigma80_lambda1_mc1000_event；扫描1001门限并单独绘图，记录真实K统计，保留σ=120结果
network/train.py及四个训练run # 默认目录使用sigma80_lambda1标记，便于服务器后续训练且不覆盖旧实验；本机未训练Transformer
covariance_based/test_detector.py # 更新参数/目录断言，按当前σ验证空间响应与背景；新增默认G距离尺度随σ调整的测试
covariance_based/README.md、CALL_CHAIN.md # 同步新参数、目录及运行方式，CALL_CHAIN追加16.19；解释σ不是硬覆盖半径，旧权重的保存场景不自动改变
保留 # 网络、loss、事件/信号生成公式、检测器、论文复现入口和旧结果不修改
验证 # 40项单元测试、语法编译及diff检查通过，四个训练run继承检查通过；本次1000次全部收敛，平均12.879轮；PM/PF@0.5为0.152555/0.001519，活跃6496、非活跃193504、漏检991、虚警294
活跃人数 # K实际范围0至28、平均6.496、中位数6，90/95/99分位15/18/约22.01；136次空场景，864次非空场景平均K为7.518519；范围不是硬限制
配对复核 # 新旧配置仅σ不同；全部1000对原始观测重放中，导频、位置、信道、事件中心及随机数推进一致，σ80活跃标签为σ120子集，G尺度同步缩小；K、噪声、计数、直方图和曲线检查通过
事件数量 # 两组均为0/1/2/3事件对应341/372/198/89次，总1035、平均1.035；响应范围缩小没有改变事件到达次数
曲线 # PM=PF交点线性插值约0.031627、门限约0.002449，仅为经验曲线插值；PNG视觉检查通过；网络、loss、检测器、论文复现入口及旧结果哈希未变，未训练Transformer

261014:
27、调整事件响应尺度σ为100米并测试协方差：
network/train.py、network/data.py、network/config.py # σ由80改为100；λ=1、触发0.6及其他场景参数不变，高斯响应和G尺度保持一致，不增加硬人数限制
covariance_based/run_monte_carlo.py # 新输出checkpoint/covariance_261014_sigma100_lambda1_mc1000_event；仍1000次，扫描1001门限，单独绘制协方差PM-PF
network/train.py及四个训练run # 默认目录使用sigma100_lambda1标记，避免覆盖已有实验；本机不训练Transformer
covariance_based/test_detector.py # 更新默认参数/目录断言，G尺度测试按当前σ构造距离
covariance_based/README.md、CALL_CHAIN.md # 同步当前参数/路径/命令，CALL_CHAIN仅追加16.20，保留已有历史文字
保留 # 网络、loss、事件/信号生成公式、检测器和run_independent_reference.py不修改
验证 # 40项单元测试、语法编译、diff及四个run继承检查通过；实际1000次全部收敛，平均16.942轮；PM/PF@0.5=0.270980/0.005734，活跃9211、非活跃190789、漏检2496、虚警1094
活跃人数 # K实测0至39、均值9.211、中位数8，90/95/99分位21/25/31；132次空场景保留，868次非空平均K为10.611751，范围不是硬限制
配对复核 # 1000对σ80/100物理样本导频、位置、信道、事件中心及随机数推进一致，σ80标签为σ100子集；K、噪声、计数、直方图、曲线及旧结果哈希检查通过
事件数量 # 0/1/2/3事件对应341/372/198/89次，均值1.035，未更改事件到达样本
曲线 # 当前[0,1]门限无PM=PF交点；PNG已检查；σ100相较σ80负载和门限0.5两项错误率增大；未训练Transformer，未删除旧结果

261014:
28、降低峰值事件触发概率为0.5并测试协方差：
network/train.py、network/data.py、network/config.py # EVENT_TRIGGER_PROB由0.6改为0.5；σ=100、λ=1及其他场景参数不变；G仅依赖位置和σ，核函数不变
covariance_based/run_monte_carlo.py # 新输出checkpoint/covariance_261014_sigma100_trigger0p5_lambda1_mc1000_event；仍1000次，扫描1001门限，单独绘制协方差PM-PF
network/train.py及四个训练run # 默认目录增加trigger0p5标记，不覆盖0.6结果；本机不训练Transformer
covariance_based/test_detector.py # 同步默认概率/目录断言及事件距离响应的预期概率
covariance_based/README.md、CALL_CHAIN.md # 同步当前参数/路径/命令，CALL_CHAIN仅追加16.21，历史文字保留
保留 # 数据生成公式、网络、loss、检测器、独立论文复现入口和旧实验不修改
验证 # 40项单元测试、语法编译、diff及四个run继承检查通过；实际1000次全部收敛，平均15.053轮；PM/PF@0.5=0.221785/0.003415，活跃7886、非活跃192114、漏检1749、虚警656
活跃人数 # K实测0至33、均值7.886、中位数7，90/95/99分位18/21/约26.01；保留134次空场景，866次非空平均K为9.106236；相较触发0.6负载和两项错误率降低
配对复核 # 1000对0.5/0.6物理样本导频、位置、信道、事件中心、G及随机数推进一致，0.5活跃标签为0.6子集；0.6重放复核总活跃9211/最大39；新K、噪声、计数、直方图、曲线及受保护代码哈希检查通过
事件数量 # 0/1/2/3事件对应341/372/198/89次，均值1.035，未改变事件到达样本
曲线 # PM=PF经验曲线线性插值约0.054999、门限约0.000575；PNG已检查；未训练Transformer
实际路径 # 运行后结果目录被外部调整为checkpoint/covariance_261014_eventtriggerprob0.5_sigma100_lambda1/single/，代码未来默认输出不变；最终核对时旧0.6结果目录已不存在，本次未删除或恢复该目录，保留已验证历史数值

261014:
29、当前事件场景的Transformer服务器训练前检查：
requirements.txt # 删除两行无效安装项Python 3.12.2及2.11.0+cu128，补充matplotlib；保留torch/numpy/tqdm约束，不安装或更换环境
README.md # 更正train.py配置来源与CLI说明，补充当前σ100/触发0.5/λ1场景、四个run、服务器短测与输出回传方式
CALL_CHAIN.md # 仅追加16.22，记录四入口真实小规模测试及服务器启动命令
验证 # 四个run各跑addcorr/noaddcorr，均为完整网络、CPU、1 epoch/2 step/batch4/eval2；前向、反向、有限loss、skip=0、权重保存/严格重载、CSV/PNG和场景/开关检查通过；测试结果只在临时目录
对照检查 # noaddcorr三个几何模块全部关闭，真实G/None/随机G输出完全相同，两个方法均使用covariance_rows和当前事件分布
保留 # 场景、数据/噪声生成公式、网络、loss、训练超参数及正式实验结果不修改；未正式训练，未验证服务器CUDA/BF16/显存

261015:
30、峰值事件触发概率调整为0.8并测试协方差：
network/train.py、network/data.py、network/config.py # EVENT_TRIGGER_PROB由0.5改为0.8，σ=100、λ=1及其他物理参数不变；G核不变，0.8不是全局活跃概率
covariance_based/run_monte_carlo.py # 新输出checkpoint/covariance_261015_sigma100_trigger0p8_lambda1_mc1000_event；仍1000次、同种子/求解器/噪声设置，扫描1001门限并单独绘图
network/train.py及四个训练run # 默认目录改用261015_sigma100_trigger0p8_lambda1标记，不覆盖此前0.5结果
covariance_based/test_detector.py # 更新默认概率/路径与事件响应预期值
README.md、covariance_based/README.md、CALL_CHAIN.md # 同步当前设置/输出，CALL_CHAIN仅追加16.23；注明三方法对比需重新训练相同0.8场景，历史文字保留
保留 # 数据生成公式、网络、loss、检测器、独立复现入口及旧结果不改；本机不正式训练Transformer
验证 # 40项测试、全部network/covariance_based源码编译、diff及四个run继承检查通过；1000次全部收敛、平均20.187轮；PM/PF@0.5=0.366314/0.012340，活跃11916、非活跃188084、漏检4365、虚警2321
活跃人数 # K实测0至49、均值11.916、中位数11，90/95/99分位27/33/约41.01；保留132次空场景，868次非空平均K为13.728111，不是硬人数上限
配对复核 # 1000对0.5/0.8样本导频、位置、信道、事件中心、G和随机数推进一致，0.5标签为0.8子集；两组K/噪声与CSV一致；旧0.5汇总/逐次CSV及统计JSON、受保护代码哈希不变
事件数量 # 0/1/2/3事件对应341/372/198/89次，均值1.035，事件到达样本未变
曲线 # 新图已视觉检查，[0,1]门限无PM=PF交点；相比0.5场景负载与门限0.5两项错误率增大；本次未正式训练、未自动提交或推送

261015:
31、峰值触发0.5、σ150、λ1的协方差测试：
network/train.py、network/data.py、network/config.py # 将上次0.8恢复为0.5，σ100改为150，λ1及其他物理参数不变；G距离尺度自动使用sqrt(2)*150，生成公式不改
covariance_based/run_monte_carlo.py # 新输出checkpoint/covariance_261015_sigma150_trigger0p5_lambda1_mc1000_event；仍1000次、相同种子/求解器/噪声/1001门限，单独绘图并保存K统计
network/train.py及四个训练run # 默认目录使用261015_sigma150_trigger0p5_lambda1标记，不覆盖旧场景
covariance_based/test_detector.py # 同步参数/路径/响应预期值，G尺度测试与2倍当前σ比较，避免固定120米参照在σ变大后不成立
README.md、covariance_based/README.md、CALL_CHAIN.md # 同步新场景及命令，CALL_CHAIN仅追加16.24，保留历史；比较对象选此前σ100/触发0.5以隔离尺度影响
保留 # 数据/噪声生成公式、网络、loss、检测器、独立复现入口及旧结果不改；不正式训练Transformer
验证 # 40项测试、全部network/covariance_based源码编译、四个run继承及diff检查通过；1000次全部收敛、平均23.389轮；PM/PF@0.5=0.436228/0.020538，活跃14834、非活跃185166、漏检6471、虚警3803
活跃人数 # K实测0至57、均值14.834、中位数13.5，90/95/99分位约34.1/40/47.01；保留132次空场景，非空868次平均K为17.089862，范围不是硬限制
配对复核 # 与σ100/触发0.5的1000对样本仅σ不同，导频/位置/信道/事件中心/RNG推进一致，原标签为新标签子集，G尺度变宽；K/噪声与CSV一致，代码及旧σ100汇总/逐次CSV/统计JSON哈希不变
事件数量 # 0/1/2/3事件对应341/372/198/89次，均值1.035；曲线无PM=PF交点，PNG已检查
实际路径 # 运行后文件被归档至checkpoint/test/covariance_261015_sigma150_trigger0p5_lambda1_mc1000_event/single/，代码未来默认输出仍为顶层目录；本次未移动或删除旧结果，未正式训练/提交/推送

261015:
32、使用服务器新epoch100权重绘制GI-HT/HT/协方差PM-PF曲线：
输入 # loss_vs_epoch_261014_sigma100_trigger0p5_lambda1_covrows内addcorr/noaddcorr的last.pt，ZIP及严格加载通过；与covariance_261014_eventtriggerprob0.5_sigma100_lambda1保存场景一致，均σ100/峰值0.5/λ1、N200/L10/M32/SNR20
运行 # 复用network/run_pm_pf_comparison.py，显式指定新权重与输出；1000次、种子42、1001门限，三方法使用同一组原始观测；CPU float32、推理batch128且最后104，HT不输入G
输出 # checkpoint/pm_pf_comparison_261015_sigma100_trigger0p5_lambda1，含三曲线PNG/SVG/PDF、CSV、原始分数/标签NPZ、协方差逐次诊断及配置/哈希
CALL_CHAIN.md、covariance_based/README.md # 记录新输入、复现命令、共享测试与输出，不覆盖历史
保留 # 当前σ150训练默认值、网络、loss、两种服务器权重及旧结果不改；不重新训练、不自动提交或推送
完成 # 1000次共享评估，活跃7886/非活跃192114，K范围0至33/平均7.886；门限0.5时GI-HT PM/PF=0.017880/0.034865，HT=0.024854/0.046894，协方差=0.221785/0.003415
协方差复核 # 1001点的门限/PM/PF/计数与原曲线逐项一致，1000次噪声/活跃数/轮数/最大步/收敛诊断与原CSV一致；全部收敛，平均15.053轮
验证 # 原始NPZ分数/标签、3003点曲线、计数、端点及单调性通过，独立重放1000次核对共享数据哈希；两种权重及网络/loss/检测器哈希不变，PNG视觉和diff检查通过
等错误率 # 基于原始分数全部断点插值：GI-HT约0.027771、HT约0.037281、协方差约0.054907；区别于粗网格插值且不是精确可达的确定门限经验相等率

261015:
33、新增Box-NNLS协方差传统方法并运行四方法比较：
covariance_based/nnls.py、__init__.py # 新增有界NNLS API，0<=a<=1与MLE一致；完整复协方差拟合、共同物理功率缩放、SciPy BVLS，不输入标签/K/活跃先验/G；严格区分有界求解与LS后裁剪
network/run_pm_pf_comparison.py # 新增--include-nnls可选第四方法及独立迭代/容差；保留三方法默认与5000次默认次数，默认NNLS输出加后缀防覆盖；新增诊断CSV、原始分数和算法元信息
covariance_based/test_nnls.py、network/test_pm_pf_comparison.py # 新增解析解/独立活跃集枚举/缩放/秩亏/边界与输入测试，检查原三方法数据/分数/诊断不变及四曲线输出；53项测试通过
requirements.txt # 增加SciPy>=1.11,<1.16；本地NumPy实际1.26.4但元数据/残留目录含2.4.6，最终安装SciPy1.12.0，不更换或修复原NumPy/PyTorch；较新SciPy导入失败已处理
README.md、covariance_based/README.md、CALL_CHAIN.md # 补充方法、数学约束、求解器设置、依赖及命令，CALL_CHAIN仅追加16.26，历史记录不覆盖
运行 # 与上一张三方法图相同σ100/触发0.5/λ1场景、两种epoch100权重、1000次/种子42/1001门限；四方法共享观测，输出checkpoint/pm_pf_comparison_261015_sigma100_trigger0p5_lambda1_nnls
保留 # 网络、loss、MLE求解器、当前σ150训练默认值、已有权重/图/结果不改；不重新训练、不自动提交或推送
精度复核 # 初始tol=1e-8有1次由相对代价提前停止；收紧默认到1e-10并新增投影梯度收敛证书及测试，完整重跑；仅依据目标/KKT调整，不使用标签调求解器
完成 # 1000次四方法评估，Box-NNLS全部收敛、平均34.376迭代/最多99、最大保存投影梯度2.31e-14；门限0.5漏检2456/虚警1692，PM/PF=0.311438/0.008807
一致性 # 共享K范围0至33、均值7.886，原三方法分数/标签/全部曲线点及MLE逐次CSV完全不变；独立重放1000次核对目标/KKT及数据哈希，最大KKT违约约2.30e-14，两种权重和受保护代码哈希不变
最终验证 # 新增最优性测试后共54项测试通过，源码编译、diff及四曲线PNG视觉检查通过；输出含4004曲线点、四方法PNG/SVG/PDF与NNLS逐次诊断，旧三方法目录保留
等错误率 # 全部分数断点线性插值GI-HT/HT/MLE/Box-NNLS约0.027771/0.037281/0.054907/0.093076，对应-15.5641/-14.2851/-12.6037/-10.3116 dB；非精确可达的单一确定门限经验相等率，本场景NNLS弱于MLE

261015:
34、统一最终事件场景为σ100、λ1、峰值触发0.9并检查服务器训练入口：
network/train.py、network/data.py、network/config.py # σ150改为100、触发0.5改为0.9，λ保持1；背景0.005/最多3事件及N200/L10/M32/R500/SNR20训练设置保持，G默认尺度随σ同步
train.py及四个训练run # 默认输出使用261015_sigma100_trigger0p9_lambda1_covrows标记；两模式均继承相同场景、使用covariance_rows
covariance_based/run_monte_carlo.py # 继承最终Transformer场景，默认输出改为covariance_261015_sigma100_trigger0p9_lambda1_mc1000_event
network/run_pm_pf_comparison.py # 默认读取最终场景loss-vs-epoch内两模式last.pt，合并图目录同步最终场景；显式传入历史权重时仍读取其保存场景
covariance_based/test_detector.py、network/test_pm_pf_comparison.py # 同步参数/响应预期与输出路径检查，检查PM-PF默认权重与训练输出对应
README.md、covariance_based/README.md、CALL_CHAIN.md # 同步最终场景、输出及服务器命令；CALL_CHAIN仅追加16.27，历史文字保留
保留 # 活动/噪声生成公式、网络、loss、求解器及训练超参数不变；正式100-epoch训练由服务器完成
验证 # 54项单元测试、全源码编译及diff检查通过；四训练入口各跑两模式，完整base/CPU/2 epoch×2步/batch4/eval2，物理点L10/SNR20/N200；loss/PM/PF有限、skip=0、两种权重保存/严格重载、日志/CSV/PNG及配置检查通过
无相关性对照 # 保存的HT模型配置关闭三个几何模块，真实G/None/随机G前向结果逐项相同，两种模式均使用最终场景和covariance_rows
共享对比短测 # 临时epoch2权重下4次观测，GI-HT/HT/MLE/Box-NNLS输出与绘图通过，两种传统方法报告收敛；仅验证流程，不作为正式性能结论
范围 # 测试结果仅在/private/tmp/giht_final_scene_261015.Awlnf2/，loss图已检查；网络/loss/MLE/NNLS/独立复现入口与HEAD哈希一致；本机CPU无法验证服务器CUDA/BF16或batch128显存，未正式训练/提交/推送

261015:
35、新增最小Pe=PM+PF的L/N/SNR实验与result_261015结果树：
experiments/pe_metrics.py # 精确搜索全部分数断点及0/1端点，每方法/点一个全体MC最优门限；同分数成组、同最优值取最低阈值，保存完整曲线/原始分数/PM-PF/计数，Pe不除2，训练loss不变
experiments/run_pe.py # 复用现有L/N/SNR列表，统一传统Monte Carlo、逐点网络训练/评估及合并绘图；默认1000次/种子42，网络各点epoch100；支持单扫描/自定义列表、只评估/只绘图、匹配完成权重复用和独立权重根目录
covariance_based/run_pe_experiments.py、network/run_pe_experiments.py # 分别为MLE+Box-NNLS和GI-HT+HT入口，全部继承最终σ100/λ1/峰值0.9场景，其他未扫描参数不变
network/run_pm_pf_comparison.py # 共享评估增加方法子集，传统和神经分开运行仍生成同一组CPU观测；保留旧默认接口，保存训练配置，推理保留原batch及尾batch行为
experiments/paths.py、现有四个训练run/train.py/单点MLE和PM-PF入口 # 本次默认输出统一result_261015，按pe_vs_*、loss_vs_epoch、loss_sweeps、training、single_condition分类；未移动历史结果或独立论文复现目录
result_261015/README.md、logs/.gitkeep、README.md、covariance_based/README.md # 提供目录树、正式/服务器短测/合并命令，区分经验曲线最小Pe与独立校准误差；原loss-only入口功能保持
.gitignore # 仅放行result_261015树内CSV，权重及checkpoints目录仍忽略
检查机制 # 汇总核对场景、MC次数、种子、观测哈希及两网络推理batch/epoch；整组缺少类别时不伪报0，所有非收敛试验仍纳入统计；实际0值不伪造为正数
验证 # 66项单元测试、源码编译与diff通过；包含精确阈值/同分数/独立计数、方法子集一致性、结果合并拒绝混用、完成权重复用及中断重写指标发布检查
端到端短测 # L4/L16、N50/N250、SNR10/SNR20，传统和网络分阶段、各方法每点8次MC；完整base GI-HT/HT共12次CPU训练（2epoch×1步/batch4/eval1）均skip=0，逐点指标/原始分数/完整曲线和四方法观测哈希一致
输出核验 # 逐方法及逐点合并PM-PF、Pe汇总PNG/SVG/PDF、CSV/NPZ及只绘图模式通过；另从独立checkpoint-root在新目录联合评估四方法，分数/标签逐项相同；Pe和PM-PF图已视觉检查，真实0值保留
保留 # 短测结果仅在/private/tmp/pe_experiments_261015.vt1wnO/，正式结果根目录仅建说明/分类目录；网络/loss/两传统求解器/独立复现入口与HEAD哈希一致；未正式训练或完整扫描、未改历史结果、未提交或推送；本机未验证CUDA/BF16/正式显存

261015:
36、新增L/N/SNR三组神经Pe实验的独立运行入口：
network/run_pe_vs_L.py、run_pe_vs_N.py、run_pe_vs_SNR.py # 分别固定单个扫描轴，默认各点依次训练/评估GI-HT与HT，各100epoch；三个进程可在不同GPU并行，输出分属pe_vs_L/N/SNR和独立日志
experiments/run_pe.py # parse_args/main增加可选fixed_sweep，限制独立入口的扫描选择，通用入口的默认串行三组行为保持；其他CLI参数复用
experiments/test_pe_experiments.py # 新增扫描隔离/默认方法和epoch检查；67项测试、全源码编译和diff通过
README.md、result_261015/README.md、CALL_CHAIN.md # 同步三个独立入口及GPU5/6/7并行启动示例，CALL_CHAIN仅追加16.29
验证 # 三个独立CLI在CPU上同时执行临时结果的只绘图模式，分别覆盖L4/16、N50/250、SNR10/20，均成功；未正式训练/扫描，未验证服务器GPU并行，未提交或推送

261015:
37、按用户要求移除刚新增的临时服务器短测改动：
删除 # 独立短测脚本及配套测试、对应编译缓存和本次临时输出均清理，不保留可执行短测代码
文档 # 撤销两个README新增的短测说明，CALL_CHAIN和本日志仅记录删除
保留 # 正式Pe/并行入口/loss-vs-epoch代码、场景和已有实验结果不变；短测代码未提交或推送

261019:
38、合并下载的服务器结果并核对四方法数据兼容性：
result_261015_2/ -> result_261015/ # 仅复制缺失文件，跳过.DS_Store；源目录完整保留，目标已有文件不覆盖，未删除结果
合并内容 # GI-HT/HT的逐点结果、三类扫描权重及loss_vs_epoch；服务器已完成L4至22共10点、N共5点、SNR共6点，最终L绘图范围仍为4至20
文件验证 # 原目标492个文件SHA-256均未变，480个无冲突源文件与目标一致；78个冲突仅为汇总/逐点合并图及配置，保留目标版本，服务器版本仍在源目录
兼容性 # 场景/MC1000/种子42/指标相同；传统方法内部与两网络内部通过，所有重合点的跨环境dataset_sha256不同，保留校验，不强行生成四方法图
标签核验 # L/N/SNR的200万/75万/120万个设备标签完全一致，但标签不代表原始导频/接收信号/噪声/G一致；记录双方PyTorch/NumPy版本不同，未据此直接认定哈希差异原因
result_261015/MERGE_REPORT_261019.md、CALL_CHAIN.md # 记录文件合并状态、既有汇总仍为传统方法图及后续同机四方法统一评估命令，CALL_CHAIN仅追加16.31
后续 # 使用已下载epoch100权重，L限定4至20、N/SNR完整，在新shared_evaluation目录统一评估即可，无需重新训练；本次未执行正式评估、未改算法/loss/场景/校验、未提交或推送

261019:
39、按用户要求直接合并同场景下不同测试观测的已有曲线：
experiments/run_pe.py # 新增--allow-different-observations，仅限--stage plot，通过CLI显式传递，训练/评估阶段拒绝该开关
experiments/pe_metrics.py # check_compatible/refresh_sweep增加可选策略，仅放宽观测哈希一致性；场景/次数/种子/指标/门限规则及两网络epoch/batch检查保留，默认行为不变
汇总记录 # summary_config.json保存开关、实际哈希是否匹配及逐点/逐方法哈希；不改原metrics.json、原始分数、最优阈值或Pe，不将非配对观测宣称为相同样本
experiments/test_pe_experiments.py # 新增开关仅限绘图、默认严格、只放宽哈希及CLI不触发训练/评估测试；扩展临时绘图验证，确认原指标未变并记录策略
文档 # result_261015/README.md、MERGE_REPORT_261019.md和CALL_CHAIN.md同步直接合并命令，CALL_CHAIN仅追加16.32；统一重评估改为可选，不是本次合图前提
运行范围 # 用户自行运行L4至20以及完整N/SNR的绘图指令；本次不运行正式绘图、不训练、不重新Monte Carlo，不改网络/loss/数据分布，未提交或推送
验证 # 70项单元测试、修改源码编译通过；真实L9点/N5点/SNR6点共80份方法结果通过同场景新策略检查，四方法齐全，未触发正式绘图/训练/评估

261019:
40、取消Pe扫描中单算法单参数点的独立图片：
experiments/pe_metrics.py # save_method_result移除单方法plot_comparison，L/N/SNR中四种方法统一仅保存分数/完整曲线NPZ、采样曲线/诊断CSV和metrics.json，不再生成方法目录下的pm_pf_curve.png/svg/pdf
汇总绘图 # 保留Pe vs L/N/SNR图及各点多方法PM-PF图；仅一种可用方法时跳过该点比较图，仍保留Pe汇总与数据
experiments/test_pe_experiments.py # 更新图片不存在断言，新增四方法数据文件完整且不调用绘图器、单方法点只做汇总测试；中断保存模拟改为指标JSON发布失败，保持旧完成标记失效验证
文档 # 根README、covariance_based/README.md、result_261015/README.md同步新输出规则，CALL_CHAIN仅追加16.33
保留 # 不改算法、训练loss、阈值/Pe计算、数据/权重、合并开关或独立单条件实验；不删除历史已有图片，不运行正式训练/测试/绘图，未提交或推送
验证 # 72项单元测试及修改源码编译通过；临时目录确认单方法无图但数据完整、多方法PM-PF及Pe汇总继续生成

261019:
41、L扫描默认仅训练/测试/绘图4至14：
network/run_varying_l.py # 默认导频列表改为[4,6,8,10,12,14]，loss图Matplotlib/SVG横轴和CLI帮助同步
experiments/run_pe.py # 无需改动，SWEEPS自动继承新默认；独立run_pe_vs_L与通用Pe/传统Pe入口均默认仅处理6点
covariance_based/run_monte_carlo.py # DEFAULT_SWEEPS的L列表同步为4至14，传统扫描与网络保持一致
experiments/test_pe_experiments.py # 新增三阶段默认范围、传统/神经/旧loss列表一致、N/SNR和标称L10不变，以及备用SVG刻度/端点测试
文档 # 根README、covariance_based/README.md、result_261015/README.md同步范围，CALL_CHAIN仅追加16.34；旧命令的显式--values/--pilot-lens仍会覆盖默认，需要移除或更新
保留 # 已有L>14数据/权重不动，N/SNR扫描及标称L10、网络/loss/场景/优化不变；服务器需同步代码，本次不运行正式训练/测试/绘图，未提交或推送
验证 # 74项单元测试及相关源码编译通过，默认6点和SVG横轴核对成功；未开展正式实验或重绘历史图

261019:
42、按用户要求删除result_261015已有单算法单参数点图片：
清理范围 # 三类pe_vs_L/N/SNR的points/<点>/<GI-HT|HT|Covariance|Box-NNLS>/pm_pf_curve.png/svg/pdf，仅删除这类图片
数量 # 共276个文件，L144/N60/SNR72；PNG/SVG/PDF各92个，目标范围剩余单方法图片为0
保留验证 # 其余700个文件的路径及内容SHA-256删除前后完全一致，数据、权重、日志、多方法PM-PF及Pe汇总图均保留；未操作其他结果目录
范围 # 只清理历史图片并更新记录，CALL_CHAIN仅追加16.35；不改代码/配置，不训练/测试/重绘，未提交或推送

261019:
43、按用户要求删除L从16开始的全部逐点结果及对应训练文件：
清理范围 # result_261015/pe_vs_L/points内L16至L30的8个目录，以及checkpoints/addcorr、noaddcorr内L16至L24的10个目录；共18目录174文件
删除内容 # 这些点的四方法测试数据、逐点图、模型权重及各点train.log，含未完整训练的L24文件
保留验证 # 其他526个文件路径及内容SHA-256完全不变；L4/6/8/10/12/14、N/SNR、loss_vs_epoch和跨点控制台总日志保留，L>=16目录剩余0个
汇总验证 # 当前L汇总CSV已仅含6点四方法共24行，summary_config也仅4至14，无需改写汇总图/数据；CALL_CHAIN仅追加16.36
范围 # 不改算法或配置，不训练/测试/重绘，不清理其他实验目录，未提交或推送

261019:
44、忽略整个result_261015并取消已有元信息文件跟踪：
.gitignore # 移除该目录CSV放行规则，增加/result_261015/整体忽略，实验图表、数据、日志和权重均仅保留本地
Git索引 # 对此前已跟踪的result_261015/.DS_Store、README.md、logs/.gitkeep执行仅取消缓存跟踪，不删除本地文件；三条暂存删除记录需后续提交后才消失
文档 # CALL_CHAIN仅追加16.37；本次不提交/推送，不改其他目录忽略规则或算法，不运行实验
验证 # 本地543个文件全部忽略，目录跟踪文件为0，三个原跟踪文件保留本地；diff检查通过

261019:
45、从GitHub当前分支移除result_261015并保留本地结果：
Git提交 # 提交目录整体忽略规则和三个旧元信息文件的取消跟踪，推送wxdwu/main3_261013；只删除远端当前分支目录，不改写历史、不上传实际实验文件
README.md、covariance_based/README.md # 实验说明链接改到CALL_CHAIN，注明结果目录仅保留本地，避免远端失效链接
本地保护 # 543个实验文件保留且继续忽略；其他.DS_Store等无关修改不提交，不运行训练/测试/绘图；CALL_CHAIN仅追加16.38
