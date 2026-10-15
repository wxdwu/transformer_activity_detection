# Experiments: 261015

Run all commands from the project root in the environment containing PyTorch,
NumPy, SciPy and Matplotlib. The final scene is event-driven: sigma=100 m,
lambda=1, peak trigger=0.9, background=0.005, at most 3 events, N=200, L=10,
M=32, radius=500 m and measured SNR=20 dB. Only the swept field changes.

## Metric

Each point reports `min_threshold(PM + PF)`. PM and PF are conditional error
rates pooled over all Monte Carlo device instances. This metric is the sum,
without a factor 1/2, and differs from `(missed + false_alarms)/(N*trials)`.
Empty-activity realizations are retained in PF. If the entire evaluation
contains no active or no inactive instances, PM+PF is undefined and the run
reports an error requesting more trials.

For each method and parameter point, one threshold is selected over the entire
evaluation set, using the strict decision `score > threshold`. The search covers
every distinct score breakpoint and endpoints 0 and 1. Equal scores change
decisions together; ties in minimum error select the lowest threshold.
The result is an empirical minimum from an offline PM-PF sweep on this set,
not a deployable threshold calibrated on independent data and not a separate
label-dependent threshold for each realization. Training losses are unchanged.

`exact_curve.npz` contains the complete curve. The plotted curve and its CSV
contain up to 2001 sampled exact breakpoints plus the exact optimum. Decimation
affects plotting only, never threshold optimization. PM-PF axes are logarithmic.
Pe plots default to a logarithmic vertical axis; use `--pe-scale linear` for a
linear plot. Observed zero errors remain zero in data and are omitted on log
axes, without substituting arbitrary positive values.

## Sweep Values

- L: 4, 6, 8, ..., 30; N=200 and SNR=20 dB.
- N: 50, 100, 150, 200, 250; L=10 and SNR=20 dB.
- SNR: 10, 12, 14, 16, 18, 20 dB; L=10 and N=200.

The lists are inherited from the existing Transformer sweep parsers. For a
paper figure restricted to L=4..16, give the same override to both entry points:
`--sweeps L --values 4 6 8 10 12 14 16`.

## Traditional Methods

```bash
python covariance_based/run_pe_experiments.py
```

This runs covariance MLE and Box-NNLS on all three sweeps, without training.
Each point uses 1000 realizations and seed 42 by default; change the count with
`--mc-times`. Both methods receive the same raw B, Y and known generated noise
variance. They use no event locations, activity probabilities, true active
count, geometry or neural weights. Box-NNLS uses coefficients in [0,1].
All trials are included even if a solver reaches its iteration limit;
convergence and per-trial diagnostics are saved. Solver settings remain MLE
100 sweeps / coordinate tolerance 1e-4, NNLS 1000 iterations / tolerance 1e-10.

## GI-HT and HT on the Server

```bash
CUDA_VISIBLE_DEVICES=5 nohup python network/run_pe_experiments.py --device cuda > result_261015/logs/neural_all.log 2>&1 &
```

This trains GI-HT/addcorr and HT/noaddcorr separately at each point, for 100
epochs with the unchanged train.py optimization settings, then evaluates each
epoch-100 `last.pt`. All use covariance_rows; GI-HT enables all three geometry
modules and HT disables them. Each point has separate weights and train.log.
This evaluates final-epoch models, not the best-PM checkpoints. The MC count
is separate from training-time `--eval-batches`.

For concurrent sweeps on different GPUs:

```bash
mkdir -p result_261015/logs
CUDA_VISIBLE_DEVICES=5 nohup python network/run_pe_vs_L.py --device cuda > result_261015/logs/neural_L.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python network/run_pe_vs_N.py --device cuda > result_261015/logs/neural_N.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python network/run_pe_vs_SNR.py --device cuda > result_261015/logs/neural_SNR.log 2>&1 &
```

Choose the single all-sweep command or the three per-sweep commands. Different
sweeps write different directories. Do not run overlapping jobs on the same
sweep/method/output root. On a shared filesystem, run the traditional and neural
stages sequentially for a given sweep directory; different sweep directories
can run concurrently. Merge/redraw after all jobs finish.

The three dedicated entry points are restricted to L, N and SNR, respectively.
Each process runs its parameter values and both models sequentially on its
assigned GPU; the three processes run concurrently. Outputs are isolated in
pe_vs_L/, pe_vs_N/ and pe_vs_SNR/. All common flags remain available; for example
`python network/run_pe_vs_L.py --values 4 6 8 10 12 14 16 --device cuda` restricts
the L range. The general runner remains available with `--sweeps L`, N or SNR.

Completed matching checkpoints are reused on a repeated train-evaluate run.
Scene, architecture, epoch and training settings are checked before reuse;
partial or mismatched checkpoints raise an error. `--retrain` explicitly starts
fresh training and replaces the selected weights/logs. This does not resume an
interrupted optimizer. Use a separate output root for smoke tests:

```bash
CUDA_VISIBLE_DEVICES=5 python network/run_pe_experiments.py --sweeps L --values 10 --epochs 1 --steps-per-epoch 2 --eval-batches 2 --mc-times 16 --device cuda --out-root /tmp/pe_server_smoke_261015
```

This keeps the training batch at 128 unless `--batch-size` is supplied. Check
server BF16 support and memory before formal training, as for the existing runs.

## Evaluation and Combined Figures

To reevaluate existing checkpoints without training:

```bash
python network/run_pe_experiments.py --stage evaluate --device cuda
```

Data generation always uses CPU, one realization per call, with the same seed
and scene for all methods. Neural inference defaults to the checkpoint training
batch size and retains the last partial batch because TokenBatchNorm depends on
current-batch statistics. Inference uses float32 without autocast.

After copying server results back into this tree, preserving each method's
point subdirectory, redraw all available methods together:

```bash
python covariance_based/run_pe_experiments.py --stage plot
```

This needs no trained weights and performs no Monte Carlo simulation. It draws
both Pe summaries and per-point combined PM-PF figures. It rejects results with
different scene settings, counts, evaluation seeds or observation hashes. This
prevents combining unmatched Monte Carlo sets. CPU random-number behavior can
vary across framework versions. If hashes differ, evaluate all four methods on
one machine using the copied checkpoints and a fresh output root:

```bash
python network/run_pe_experiments.py --stage evaluate --methods GI-HT HT Covariance Box-NNLS --checkpoint-root result_261015 --out-root result_261015/shared_evaluation --device cpu
```

The same command can be used for the final four-method evaluation after server
training, followed by plotting under that new root. The source root contains
the per-sweep checkpoint trees, not an individual last.pt file.

## Directory Layout

```text
result_261015/
  logs/                              console logs from nohup
  pe_vs_L/                           analogous pe_vs_N/ and pe_vs_SNR/
    checkpoints/
      addcorr/L04/                   train.log, last.pt, best_pm.pt
      noaddcorr/L04/
    points/L04/                      one directory per sweep value
      Covariance/                    traditional MLE results
      Box-NNLS/                      traditional bounded NNLS results
      GI-HT/                         neural results
      HT/
        scores.npz                   raw scores and shared binary labels
        exact_curve.npz              all thresholds, PM, PF, Pe and counts
        pm_pf_curve.csv              plotting samples plus exact optimum
        pm_pf_curve.png/svg/pdf
        metrics.json                 exact optimum, scene, weights and hashes
        trials.csv                   all per-realization diagnostics
      pm_pf_comparison.png/svg/pdf    all available methods at this point
    pe_vs_L.csv                      minimum Pe, threshold, PM/PF and counts
    pe_vs_L.png/svg/pdf               all available methods
    summary_config.json              requested and completed points
  loss_vs_epoch/                     existing loss-vs-epoch runner
  loss_sweeps/varying_L/              existing loss-only sweep runners
  loss_sweeps/varying_N/
  loss_sweeps/varying_SNR/
  training/addcorr/                   standalone train.py
  single_condition/covariance/        existing standalone MLE runner
  single_condition/pm_pf_comparison/  existing single-point comparison runner
```

The existing `run_varying_l.py`, `run_varing_SNR.py` and
`run_varing_number_of_users.py` still draw training loss. Use the new
`run_pe_experiments.py` entry points for Pe experiments. Newly generated outputs
from these current training/evaluation entry points default to this root;
historical files are not moved. The independent paper reproduction retains
its separate historical output directory.

CSV files in this tree are allowed by .gitignore. Model weights and checkpoint
directories remain ignored; transfer complete checkpoint directories directly
between the server and local machine when needed.
