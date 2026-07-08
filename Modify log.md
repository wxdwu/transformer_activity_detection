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
