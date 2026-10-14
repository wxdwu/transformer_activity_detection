
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
SNR=20 dB, event lambda=1, sigma=100 m, peak trigger=0.5, background=0.005
and at most 3 events. The signal token mode is `covariance_rows`.

To train both addcorr and noaddcorr for 100 epochs and plot loss vs epoch:

```bash
python network/run_loss_vs_epoch.py --device cuda
```

The runner saves separate per-mode train.log, last.pt and best_pm.pt files,
plus loss_by_epoch.csv and loss_vs_epoch.png, under
`checkpoint/loss_vs_epoch_261014_sigma100_trigger0p5_lambda1_covrows/`.
Noaddcorr disables all geometry modules while keeping the same scene and
covariance-row encoding. Plain `python network/train.py` trains only the mode
selected by its constants; it does not parse training command-line overrides.

For a short server check using the real training batch size:

```bash
python network/run_loss_vs_epoch.py --device cuda --epochs 1 --steps-per-epoch 2 --out-dir checkpoint/server_smoke_261014
```

Sweep runners also train both modes and inherit the same event settings:

```bash
python network/run_varying_l.py --device cuda --pilot-lens 4,6,8,10,12,14,16
python network/run_varing_SNR.py --device cuda
python network/run_varing_number_of_users.py --device cuda
```

The L runner defaults to 4,6,...,30 unless the list is supplied. SNR sweeps
10,12,...,20 and N sweeps 50,100,...,250; only the swept physical field changes.
Use separate GPUs and nohup log files for concurrent experiments. Do not run
two jobs with the same output root. The default AMP dtype is BF16; check GPU
support before formal training and use AMP_DTYPE="fp16" in train.py if needed.
Local CPU smoke tests do not validate server CUDA compatibility or GPU memory.

Copy the complete experiment directories back after training, including CSVs,
plots, logs and both modes' epoch-100 last.pt files with their saved configs.
For a new PM-PF comparison, pass these new weights explicitly to
`network/run_pm_pf_comparison.py` with --giht-ckpt, --ht-ckpt and a new --out-dir;
its default paths still refer to the older 260720 experiment. Old-scene weights
must not be mixed with the new event scene.

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
