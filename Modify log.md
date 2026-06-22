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
