
# Heterogeneous Transformer for Activity Detection

This repo reproduces the core part of:

- `Li et al., 2023, Heterogeneous transformer: a scale adaptable neural network architecture for device activity detection`

Scope of this version:

- Implement transformer-based active user/device detection only.
- Include a lightweight spatial-correlation extension for event-driven co-activity.

## Files

- `network/data.py`: system model based synthetic data generation (`Y, B, a`) and feature construction.
  - users are sampled in a circular area
  - pairwise distance is mapped to a normalized spatial correlation in `[0,1]`
  - correlated activity mode lets active seed users trigger nearby/correlated users
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
- Correlation extension: when `use_correlation_feature=True`, each user pilot token adds one scalar spatial-correlation feature, so `x_b` changes from `[B,N,2Lp]` to `[B,N,2Lp+1]`.
- Initial embedding Eq. (7): separate projection for pilot tokens and signal token.
- Encoding Eq. (8)-(19): heterogeneous MHA + FFN with residual and normalization.
- Decoding Eq. (24)-(26): context attention + matching score + sigmoid probability.
- Training loss Eq. (4): weighted BCE for sparse activity.

## Quick Start

Install:

```bash
pip install -r requirements.txt
```

Edit experiment settings in `config.json`, then run training with:

```bash
python network/train.py
```

The shared config controls the system parameters, model shape, training loop, and evaluation/detection defaults:

- `network_train`: settings used by `network/train.py`, including nested `system` and `model`.
- `network_evaluate`: settings used by `network/evaluate.py`.
- `CE_methods_compare`: settings used by `CE_methods/compare.py`.
- `network_compare_active_indices`: settings used by `network/compare_active_indices.py`.
- `plot_plot_amp_nmse_vs_iter`: settings used by `plot/plot_amp_nmse_vs_iter.py`.

Command-line arguments are still available for temporary overrides, for example:

```bash
python network/train.py --epochs 2 --steps_per_epoch 10 --batch_size 16
```

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
- The default system now uses a 500 m circular area and correlated activity labels. Set `activity_mode="independent"` and `use_correlation_feature=false` for a baseline ablation.
