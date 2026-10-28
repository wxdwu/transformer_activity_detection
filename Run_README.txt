squeue
nvidia-smi

run:
conda activate cat
CUDA_VISIBLE_DEVICES=5 nohup python train.py > train.log 2>&1 &


261015:
传统方法，运行 MLE 和 Box-NNLS 的全部三组实验：
python covariance_based/run_pe_experiments.py

python covariance_based/run_pe_experiments.py --sweeps M event_sigma_m

服务器运行 GI-HT 和 HT，三组实验分配到不同 GPU 并行运行：
mkdir -p result_261015/logs

CUDA_VISIBLE_DEVICES=4 nohup python -u network/run_loss_vs_epoch.py --device cuda > result_261015/logs/neural_loss_vs_epoch.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python network/run_pe_vs_L.py --device cuda > result_261015/logs/neural_L.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python network/run_pe_vs_N.py --device cuda > result_261015/logs/neural_N.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python network/run_pe_vs_SNR.py --device cuda > result_261015/logs/neural_SNR.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python -u network/run_pe_vs_M.py --device cuda > result_261015/logs/neural_M.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python -u network/run_pe_vs_event_sigma.py --device cuda > result_261015/logs/neural_event_sigma_m.log 2>&1 &

收集服务器结果后，将各点的 GI-HT/、HT/ 结果子目录合入本地对应的 result_261015/pe_vs_*/points/，保留已有传统方法结果，然后合并绘图：
python covariance_based/run_pe_experiments.py \
--stage plot \
--allow-different-observations
