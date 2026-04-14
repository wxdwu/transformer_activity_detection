# Heterogeneous Transformer for Activity Detection

This repo reproduces the core part of:

- `Li et al., 2023, Heterogeneous transformer: a scale adaptable neural network architecture for device activity detection`

Scope of this version:

- Implement transformer-based active user/device detection only.
- Do **not** include user correlation / event-driven co-activity module yet.

## Files

- `htad/data.py`: system model based synthetic data generation (`Y, B, a`) and feature construction.
- `htad/model.py`: heterogeneous transformer:
  - initial embedding layer
  - heterogeneous encoder layers
  - context decoder
- `htad/losses.py`: weighted cross-entropy style loss (paper Eq. (4)).
- `htad/metrics.py`: `PM/PF` metrics and threshold curve.
- `train.py`: training script.
- `evaluate.py`: evaluate `PM-PF` curve and export CSV.

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

Train (lightweight smoke run):

```bash
python train.py --epochs 2 --steps_per_epoch 10 --batch_size 16
```

Train (closer to paper architecture defaults):

```bash
python train.py \
  --num_devices 100 \
  --num_antennas 32 \
  --pilot_len 8 \
  --activity_prob 0.1 \
  --dim 128 \
  --num_layers 5 \
  --num_heads 8 \
  --head_dim 32 \
  --ff_dim 512 \
  --score_scale 10 \
  --epochs 100 \
  --steps_per_epoch 5000 \
  --batch_size 256 \
  --lr 1e-4 \
  --lr_decay_epochs 90,97 \
  --lr_decay_factor 0.1
```

Evaluate PM/PF curve:

```bash
python evaluate.py --ckpt checkpoints/last.pt --num_test_batches 100 --batch_size 128
```

Output file:

- `pm_pf_curve.csv`

## Notes on Reproducibility

- This implementation follows the paper structure and equations for the detector.
- Some simulator details are engineering defaults (still physically consistent):
  - finite-sample synthetic generation each step (instead of pre-generated large fixed dataset)
  - batch normalization implemented with PyTorch `BatchNorm1d` over token features
- These choices keep the code stable and easy to extend for your next step (activity correlation module).
