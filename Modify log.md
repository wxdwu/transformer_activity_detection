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
