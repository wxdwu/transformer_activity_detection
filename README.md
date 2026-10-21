
# Heterogeneous Transformer for Activity Detection

This repo reproduces the core part of:

- `Li et al., 2023, Heterogeneous transformer: a scale adaptable neural network architecture for device activity detection`

Scope of this version:

- Implement heterogeneous Transformer device activity detection for event-triggered access.
- Include covariance-row signal tokens and geometry-informed attention, feature propagation and logit refinement.

## Files

- `network/data.py`: system model based synthetic data generation (`Y, B, a`) and feature construction.
- `network/model.py`: heterogeneous transformer:
  - initial embedding layer
  - heterogeneous encoder layers
  - context decoder
- `network/losses.py`: weighted cross-entropy style loss (paper Eq. (4)).
- `network/metrics.py`: `PM/PF` metrics and threshold curve.
- `network/train.py`: training script.
- `network/run_pe_experiments.py`: train GI-HT/HT and plot minimum PM+PF vs L/N/SNR/M/event scale.
- `network/run_pe_vs_M.py`: dedicated antenna-count Pe sweep.
- `network/run_pe_vs_event_sigma.py`: dedicated event-response-scale Pe sweep.
- `covariance_based/run_pe_experiments.py`: matched MLE/Box-NNLS Monte Carlo Pe sweeps.
- `experiments/`: shared threshold optimization, evaluation and result aggregation.
- `covariance_based/nnls.py`: traditional box-constrained covariance NNLS, solved with SciPy BVLS.
- `network/evaluate.py`: evaluate `PM-PF` curve and export CSV.
- `CE_methods/estimators.py`: activity-index conversion plus CAMP and LMMSE channel-estimation methods.
- `CE_methods/compare.py`: run trained detector, call estimators, and write PM/PF/NMSE report.
- `plot/`: plotting scripts.

## Paper-to-Code Mapping

- System model Eq. (1): implemented in `ActivityDataGenerator.sample_batch`.
- Input features Eq. (5)-(6): real/imag concatenation for pilots and covariance vectorization for received signal.
- Initial embedding Eq. (7): separate projection for pilot tokens and signal token.
- Encoding Eq. (8)-(19): heterogeneous MHA + FFN with residual and normalization.
- Decoding Eq. (24)-(26): context attention + matching score + sigmoid probability.
- Training loss Eq. (4): weighted BCE for sparse activity.

## Quick Start

Install:

```bash
pip install -r requirements.txt
```

Run commands from the project root, not from inside `network/`. Activate a
Python environment with a CUDA-enabled PyTorch build for GPU training.
The current training entry points read constants from `network/train.py`,
not `config.json`. Current scene: N=200, L=10, M=32, radius=500 m, measured
SNR=20 dB, event lambda=1, sigma=100 m, peak trigger=0.9, background=0.005
and at most 3 events. The signal token mode is `covariance_rows`.

To train both addcorr and noaddcorr for 100 epochs and plot loss vs epoch:

```bash
python network/run_loss_vs_epoch.py --device cuda
```

The runner saves separate per-mode train.log, last.pt and best_pm.pt files,
plus loss_by_epoch.csv and loss_vs_epoch.png, under
`result_261015/loss_vs_epoch/`.
Noaddcorr disables all geometry modules while keeping the same scene and
covariance-row encoding. Plain `python network/train.py` trains only the mode
selected by its constants; it does not parse training command-line overrides.

For a short server check using the real training batch size:

```bash
python network/run_loss_vs_epoch.py --device cuda --epochs 1 --steps-per-epoch 2 --out-dir result_261015/server_smoke
```

Sweep runners also train both modes and inherit the same event settings:

```bash
python network/run_varying_l.py --device cuda --pilot-lens 4,6,8,10,12,14
python network/run_varing_SNR.py --device cuda
python network/run_varing_number_of_users.py --device cuda
```

The L runner defaults to 4,6,...,14 unless the list is supplied. SNR sweeps
10,12,...,20 and N sweeps 50,100,...,250; only the swept physical field changes.
Use separate GPUs and nohup log files for concurrent experiments. Do not run
two jobs with the same output root. The default AMP dtype is BF16; check GPU
support before formal training and use AMP_DTYPE="fp16" in train.py if needed.
Local CPU smoke tests do not validate server CUDA compatibility or GPU memory.

Copy the complete experiment directories back after training, including CSVs,
plots, logs and both modes' epoch-100 last.pt files with their saved configs.
After the new training finishes, `network/run_pm_pf_comparison.py` defaults
to both last.pt files in the new loss-vs-epoch directory above. Its output is
`result_261015/single_condition/pm_pf_comparison/`. Use
--giht-ckpt, --ht-ckpt and --out-dir to evaluate a different saved experiment;
its scene is always read from the selected checkpoints.

Add `--include-nnls` to the shared PM-PF comparison to evaluate GI-HT, HT,
covariance MLE and Box-NNLS on identical observations. Both traditional methods
use known scaled pilots and noise with activity coefficients bounded to [0,1].
NNLS uses no neural model, activity probability, true activity count or geometry.
The flag is opt-in; without it, the previous three-method behavior is retained.
An unspecified output directory gains a `_with_nnls` suffix when the flag is
enabled, keeping the old three-method output separate. Install the added SciPy
dependency from requirements.txt before using this option.

## Minimum Detection Error Experiments

The new metric is `Pe = min_threshold(PM + PF)` at each parameter point. Both
traditional methods and both neural methods use the same sweep lists and final
event scene. Every point saves its PM-PF curve data, exact optimum and raw scores;
per-method point directories do not generate figures. Combined per-point PM-PF
figures are generated when at least two methods are available, and summary plots
combine all compatible available methods.

```bash
python covariance_based/run_pe_experiments.py
python network/run_pe_experiments.py --device cuda
python covariance_based/run_pe_experiments.py --stage plot
```

The first command evaluates MLE and Box-NNLS; the second trains/evaluates
GI-HT and HT at epoch 100; the third redraws the joint figures after results
are collected. The general Pe runner now includes all five sweep axes; specify
`--sweeps` to run only a subset. Default MC count is 1000 per point for every method. All new
outputs are organized in `result_261015/`. The earlier loss-only runners now
save under `result_261015/loss_sweeps/`; their loss metric is unchanged.
The Pe N sweep defaults to `N=[100,140,180,220,260,300]` for all four methods
in training, evaluation and plotting. The earlier loss-only N runner retains
its original list. The previous Pe-vs-N results and checkpoints were cleared
locally; the six new points replace them under the same `pe_vs_N/` path.
On the server, stop any N jobs and clear the old `result_261015/pe_vs_N/`
before restarting if it still exists, so N=100 cannot reuse an old checkpoint.
The runners do not delete results automatically. Explicit `--values` overrides the defaults.
The result tree is local-only and ignored by Git. See
[the experiment call chain](CALL_CHAIN.md) for server commands, sweep overrides,
output organization and the offline threshold-selection convention.

For parallel neural sweeps on separate GPUs, use the dedicated entry points
`network/run_pe_vs_L.py`, `network/run_pe_vs_N.py` and
`network/run_pe_vs_SNR.py`, or the two new entry points below. Each trains/evaluates
both GI-HT and HT for its selected sweep and writes to a separate pe_vs_* directory.

## Antenna and Event-Scale Pe Sweeps

- M: 32, 48, 64, 80, 96, 112, 128 antennas.
- event_sigma_m: 60, 70, 80, 90, 100, 110, 120 meters.

Only the swept parameter changes. Both use N=200, L=10, measured SNR=20 dB,
event lambda=1, peak trigger=0.9, background=0.005, at most 3 events and
covariance_rows. The M sweep keeps sigma=100 m; the sigma sweep keeps M=32.
Sigma is the Gaussian event-response scale, not a hard coverage radius.
With the default correlation_length_m=0, the geometry affinity length also
updates to sqrt(2)*sigma. Increasing sigma can increase activation load;
the generator does not hold the activity ratio fixed.

Traditional MLE and Box-NNLS, without training:

```bash
python covariance_based/run_pe_experiments.py --sweeps M event_sigma_m
```

Server GI-HT/HT: each point trains both models for 100 epochs, then evaluates
their epoch-100 last.pt weights on 1000 MC trials. The two processes can run
concurrently on separate GPUs; points and models within each process are serial.

```bash
mkdir -p result_261015/logs
CUDA_VISIBLE_DEVICES=5 nohup python -u network/run_pe_vs_M.py --device cuda > result_261015/logs/neural_M.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -u network/run_pe_vs_event_sigma.py --device cuda > result_261015/logs/neural_event_sigma_m.log 2>&1 &
```

After collecting each method's results, redraw the two combined figures without
training or MC evaluation:

```bash
python covariance_based/run_pe_experiments.py --stage plot --sweeps M event_sigma_m --allow-different-observations
```

Outputs are `result_261015/pe_vs_M/pe_vs_M.png/svg/pdf` and
`result_261015/pe_vs_event_sigma_m/pe_vs_event_sigma_m.png/svg/pdf`, plus summary
CSVs. Each point (e.g. points/M032 or points/SIGMA60) retains its data and
multi-method PM-PF figure; per-method point directories do not generate figures.
Do not run overlapping writers on the same sweep/output tree. Keep local
traditional and server neural method subdirectories when merging, then redraw.
The entire result tree remains local-only and ignored by Git.

Evaluate PM/PF curve:

```bash
python network/evaluate.py
```

Output file:

- `pm_pf_curve.csv`

## Notes on Reproducibility

- This implementation follows the paper structure and equations for the detector.
- Some simulator details are engineering defaults (still physically consistent):
  - finite-sample synthetic generation each step (instead of pre-generated large fixed dataset)
  - batch normalization implemented with PyTorch `BatchNorm1d` over token features
- These choices keep the code stable and easy to extend for your next step (activity correlation module).
