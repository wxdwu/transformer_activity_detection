# Covariance-Based Activity Detection

This folder implements a traditional covariance maximum-likelihood detector
with box-constrained coordinate descent. It does not train a network, load
weights, use activity labels during detection, or use the geometry matrix.

## Detector

For the original scaled pilot matrix `B` and sample covariance `C = Y Y^H / M`,
the detector solves

```text
minimize  logdet(Sigma) + trace(Sigma^-1 C)
subject to 0 <= a_n <= 1
Sigma = B diag(a) B^H + noise_var I
```

Each sweep visits every device once, in a random permutation by default.
Writing `v = Sigma^-1 b_n`, `q = b_n^H v`, and `r = v^H C v`, the exact
one-coordinate update is

```text
delta = clip((r - q) / q^2, -a_n, 1 - a_n)
a_n = a_n + delta
Sigma^-1 = Sigma^-1 - delta/(1 + delta*q) * v v^H
```

The implementation uses NumPy complex128 arithmetic. A common physical power
scale is applied to `B`, `C`, and the noise variance for numerical stability.
It refreshes the inverse after every sweep and stops when the largest feasible
coordinate update is at most `--tolerance`. `--max-sweeps` is a cap on complete
device sweeps, not epochs or individual coordinate updates. A limit-hit result
is retained and reported through `converged_fraction` and `max_coordinate_step`.

The scores are relaxed activity estimates, not calibrated activity probabilities.
The covariance likelihood is not the Transformer's weighted BCE training loss.
Use the PM/PF tradeoff or PM at the same PF for comparisons with learned methods;
a common threshold of 0.5 is only an additional operating point.

References:

- [Li et al., Heterogeneous Transformer, IEEE TWC, 2023](https://doi.org/10.1109/TWC.2022.3218579), covariance baseline with activities relaxed to `[0, 1]`.
- [Wei Yu, Covariance Based Detection tutorial, 2021](https://conferences.ece.ubc.ca/nasit2021/Slides/Tutorial1.pdf), coordinate descent and the known-large-scale-fading variant.

## Data and Monte Carlo Evaluation

`run_monte_carlo.py` inherits system parameters from `network/train.py` via
`build_args()` and `build_system_config()`. It calls the same
`ActivityDataGenerator.sample_batch(1, return_raw=True)` once per Monte Carlo
realization. Thus device locations, event activation, pilot distributions,
power control, fading, and noise use the shared generator with the selected
system settings (including the temporary covariance-only overrides below).
Only one realization is kept in memory; no training dataset is created.

The detector receives the unnormalized complex `B` and `Y Y^H / M`, not the
independently normalized Transformer embeddings. Labels are used only to count
detection errors. Generated geometry, event centers, channels, and activity
probabilities are not detector inputs.

By default, the receiver is given the generated noise variance, as in the usual
known-noise covariance baseline. This assumption is recorded as `known_generated`.
It must be stated when comparing against GI-HT. To evaluate a fixed receiver
noise estimate, pass `--receiver-noise-variance VALUE` in physical power units.
The standard Gaussian covariance likelihood is the detector model; the selected
noise mode is applied by the shared data generator.

All Monte Carlo realizations, including those with no active devices, contribute
to the false-alarm counts. PM is total missed detections / total active instances;
PF is total false alarms / total inactive instances. A class with no instances
is reported as `nan`, not as a zero error rate. No true active-user count is used
to choose the support or the detection threshold.

Run from the project root, after activating the environment with the project
dependencies. The traditional detector and data generation use CPU, so no GPU
assignment is needed.

```bash
python covariance_based/run_monte_carlo.py
python covariance_based/run_monte_carlo.py --sweep L
python covariance_based/run_monte_carlo.py --sweep N
python covariance_based/run_monte_carlo.py --sweep SNR
```

Defaults: `L=[4,6,...,14]`, `N=[50,100,150,200,250]`, and
`SNR=[10,12,14,16,18,20]`. Only the swept field changes; other parameters keep
their selected values. SNR sweeps require SNR-dependent noise, not thermal noise.

The default is now `USE_PAPER_EVENT_TEST_SETTINGS=False` and
`PILOT_LEN_OVERRIDE=None`, restoring all system settings from `network/train.py`:
currently `N=200`, `M=32`, `L=10`, radius `500 m`, event-driven activity, and
measured-SNR noise at `20 dB`. Event count is Poisson(1) capped at 3, response
scale is `100 m`, peak trigger probability is `0.9`, and background probability
is `0.005`. Pilot entries remain CN(0,1/L). Global activity is not forced to 0.1.
The event-response and physical-generation formulas are unchanged; the current
response scale is 100 m, peak trigger is 0.9 and event arrival is 1.0. This aligns the scene
and distribution for newly trained Transformers, not the exact historical
GPU-generated test observations.

`MC_TIMES=1000` now applies to each standalone covariance test condition,
overridable with `--mc-times`. The independent paper reproduction and the
three-method comparison retain their separate 5000-trial defaults.
The solver budget remains `MAX_SWEEPS=100`, `TOLERANCE=1e-4`; samples reaching
the cap are retained in the metrics and reported in convergence diagnostics.
The known-generated-noise assumption is unchanged: the receiver is given the
per-realization variance derived from the noiseless received signal. This
receiver side-information assumption must still be stated in comparisons.

The legacy `PAPER_EVENT_TEST_SETTINGS` preset is retained but disabled. Setting
its switch to True explicitly enables the old `N=100`, `L=8`, radius `250 m`,
thermal-noise event check and its separate output root. It is not the default
Transformer-matched experiment. Command-line physical settings take priority
over the temporary preset and pilot-length override.
The values printed at launch and `run_config.json`
are the authoritative configuration for each run.

The earlier higher-load single-point experiment (Poisson parameter 1.0 and
peak trigger 0.9) completed 5000 realizations with all
solves converged (mean 24.061 sweeps): PM at threshold 0.5 was 0.470537 and
PF was 0.026875. The observed activity ratio was 0.084461, not a prescribed
0.1. All 671 empty realizations are included in the counts. Configuration,
per-trial counts, threshold-curve monotonicity and the first 30 physical-sample
replays were checked; PNG/SVG are in
`checkpoint/covariance_261014_transformer_event/single/`.
These are new samples from the Transformer-matched distribution, not the same
saved test observations as the earlier GPU experiments.

### Independent-Activity Reference

`run_independent_reference.py` now reproduces the covariance experiment settings
in Fig. 4 of the local HT PDF (Fig. 5 in some later versions). Its independent
NumPy generator uses the paper's physical specifications, not the normalized-pilot
Transformer generator. Activity-count sampling is made explicit below.
It does not change `run_monte_carlo.py`, its temporary
event preset, or Transformer settings.

- Three default cases: `(L,M)=(7,32),(8,32),(8,64)`, each with 5000 realizations.
- `N=100`, uniform positions in a disk of radius `250 m`, independent
  Bernoulli activities with probability `0.1`; the active count is not fixed.
- Pilot and Rayleigh channel entries are CN(0,1). In particular, pilots are
  NOT divided by `sqrt(L)` or normalized separately from the physical model.
- Maximum transmit power `23 dBm`; path loss `128.1+37.6 log10(distance_km)`;
  power control `p_n=pmax*min(gain)/gain_n`.
- Fixed complex thermal-noise variance: `-169 dBm/Hz * 10 MHz = -99 dBm`,
  approximately `1.258925411794e-13 W`; no measured-SNR noise scaling.
- The existing known-gain covariance MLE retains the paper's `[0,1]` activity
  constraints. Neither the labels nor the activity probability enter its solve.

The physical specifications and target figure are checked against
[the authors' early manuscript, Sections II-B and V](https://arxiv.org/html/2112.10086v1#S5).
The known-gain coordinate update is also given in
[Chen, Sohrabi and Yu, Section II-C](https://arxiv.org/pdf/2103.00782).
These sources do not provide every implementation choice for the HT figure:
zero initialization, random coordinate order, `500` maximum sweeps and
`1e-6` stopping tolerance are reproducible choices here, not claimed author
settings. The minimum gain is taken over the sampled devices, with a `1 m`
distance guard; both are recorded in the configuration.

Across cases, positions and activity labels are shared; pilots, channels and
noise use prefixes of common maximum-size arrays. This reduces comparison
variance without changing each case's marginal distribution. Changing the set
of cases changes those maximum sizes and thus the seeded sample sequence.
The first three realizations of each case are independently checked by
rebuilding and directly inverting the covariance at every coordinate. This
verifies numerical updates, not global optimality of the nonconvex likelihood.

The paper states an active ratio of `0.1` and Bernoulli device labels, but does
not explicitly resolve independent draws versus a fixed support size in the
figure's joint sampling procedure. These are different joint distributions:
independent `p=0.1` gives variable K; a uniform fixed-K=10 subset of N=100 has
the same per-device marginal probability but always exactly 10 active devices.
The default remains independent Bernoulli. `--active-count 10` explicitly
selects the second interpretation; this choice is recorded as `fixed_count`,
not silently substituted for independent activity. The detector is not given K.

```bash
python covariance_based/run_independent_reference.py
python covariance_based/run_independent_reference.py --cases 8:32 --mc-times 1000
python covariance_based/run_independent_reference.py --pilot-len 8 --num-antennas 64
python covariance_based/run_independent_reference.py --active-count 10
```

Outputs now default to `checkpoint/covariance_261014_paper_reproduction/`.
The root has the combined `summary.csv`, `pm_pf_curve.csv`, `solver_audit.csv`,
`run_config.json`, and three-curve PNG/SVG. Each `L7_M32/`, `L8_M32/`, `L8_M64/`
subdirectory additionally contains per-trial results and `activity_scores.npz`
(scores and labels, not a large training dataset). `--mc-times` overrides this
entry's `MC_TIMES=5000`; it is separate from the event runner's count.
`--cases` selects L:M pairs; alternatively `--pilot-len`/`--num-antennas` run
a single pair. These two selection styles cannot be combined. Use `--help`
for supported reference parameters; event and SNR-sweep flags do not apply here.

A dense threshold grid, including exact `0`/`1` and logarithmically spaced
near-boundary thresholds, generates the PM/PF tradeoff. Exact-zero active
scores and exact-one inactive scores are counted in the summary. Zero error
probabilities remain zero in CSV and are only masked in logarithmic plots.
No points are fabricated to extend the curves toward the axes. A faithful
implementation of the published model does not guarantee a pixel-identical
figure without the original experiment code and its undocumented settings.
The obsolete `covariance_261014_paper_independent/single/` results were removed
on 2026-10-14 at the user's request. That earlier 1000-trial check used
CN(0,1/L) pilots and is not used by the current reproduction entry point.
Its historical setup and summary remain recorded in `CALL_CHAIN.md` and
`Modify log.md`; the current independent and fixed-K results are retained.

For independent activity, `K10_subset_diagnostic/` additionally evaluates saved
scores on realizations having exactly K=10 (more generally K=round(N*p)). Its
title, summary and configuration state how many trials are selected; it is a
conditional diagnostic, not an additional 5000-trial experiment or a replacement
for the all-trial result. No true K is passed to detection. The completed
independent experiment selected 669 of 5000 realizations per case. Those curves
are visually closer to the paper, a clue about activity sampling rather than
proof of the authors' original setup.

`--active-count 10` instead generates complete fixed-K experiments. Its default
output root is `checkpoint/covariance_261014_paper_fixedK10/`, separate from the
independent result. Each case still runs 5000 times, and the same physical RNG
draws are paired with the independent experiment; only label construction
changes. Optional `--out-dir` overrides the root. `--activity-prob` controls
independent draws; fixed support has effective marginal probability K/N.

Completed local evaluation: both sampling modes ran 5000 trials for each of
the three cases; all 30000 solves met the stopping criterion. On the threshold
grid, the minimum PM for PF <= 0.01 was:

| L | M | Independent p=0.1 | Fixed K=10 |
|---|---|-------------------|------------|
| 7 | 32 | 0.304831 | 0.197140 |
| 8 | 32 | 0.098762 | 0.028160 |
| 8 | 64 | 0.020363 | 0.001820 |

These are descriptive tradeoff points, not deployment thresholds selected on a
separate validation set. The fixed-K curves are visually closer to the paper
over the main region, but boundary behavior still differs. This supports an
activity-sampling hypothesis, not a claim to have recovered the authors' exact
code. Direct-inversion checks agree to approximately 1e-15, and the 669 shared
K=10 realizations have bit-identical scores in both runs. Existing event outputs
are preserved; only the obsolete normalized-pilot independent-reference
directory was removed as documented above.

Examples:

```bash
python covariance_based/run_monte_carlo.py --sweep L --values 4 6 8 10 12 14 16 --mc-times 1000
python covariance_based/run_monte_carlo.py --num-users 200 --pilot-len 10 --snr-db 20 --mc-times 5000 --max-sweeps 200
python covariance_based/run_monte_carlo.py --config config.json --mc-times 1000
python covariance_based/run_monte_carlo.py --mc-times 3 --max-sweeps 20 --no-plot --out-dir /tmp/covariance_smoke
```

Use `--help` for all physical parameters, including `--activity-mode`,
`--event-lambda`, `--event-sigma-m`, and `--background-activity-prob`.
The data seed and coordinate-order RNG are separate; each sweep point resets
the same seeds. This gives reproducible trials, not an identical test dataset
to separately run AI experiments. Compare all methods on the same observations
when producing a final shared-test-set comparison.

## Outputs

The output root is `result_261015/single_condition/covariance/`, configured by `OUTPUT_ROOT`
in `run_monte_carlo.py` or `--out-dir`. While the temporary preset is enabled,
the default root is instead `checkpoint/covariance_261014_paper_event/`,
configured by `PAPER_EVENT_TEST_OUTPUT_ROOT`, preserving existing results.
Single-point experiments use `single/`;
sweeps use `varying_L/`, `varying_N/`, or `varying_SNR/`.
Older `covariance_261013/` and paper-reference results are not overwritten.

- `summary.csv`: pooled PM/PF, error probability, accuracy, exact-support rate,
  average activity ratio, sweeps, convergence fraction, likelihood, and runtime.
- `trial_results.csv`: counts and solver diagnostics for every realization.
- `pm_pf_curve.csv`: threshold scan from 0 to 1 with pooled counts.
- `run_config.json`: full physical settings, seeds, solver settings, and noise assumption.
- `activity_count_stats.json`: true active-user count range, mean, quantiles,
  empty/nonempty trial statistics and a count histogram for each sweep point.
- `pm_pf_curve.png/svg`: PM/PF curves, if Matplotlib is installed.
- `detection_vs_L/N/SNR.png/svg`: parameter-sweep plots, if Matplotlib is installed.

The PM/PF tradeoff plot follows Fig. 4 of the local HT reference: PM on the
horizontal axis, PF on the vertical axis, both logarithmic with base-10 power
labels. Probability axes show `10^-4` through `10^0` by default and extend down
to smaller decades when needed. Parameter-sweep plots keep L/N/SNR linear and
use the same logarithmic probability scale vertically. Major and minor grids
are enabled; parameter axes retain every scanned value even if its probability
is zero or undefined. Zero or undefined probabilities are omitted (with gaps in curves),
not replaced by an artificial positive error rate; CSV values remain unchanged.
`PLOT_MIN_PROBABILITY` controls the default lower display limit.

The runtime includes constructing the sample covariance and solving the
likelihood, but excludes data generation and file output. Outputs are overwritten
when the same output directory is reused. Use a distinct `--out-dir` per seed or
configuration. `--no-plot` skips plotting, and missing Matplotlib does not prevent
CSV output. Existing `.gitignore` rules ignore newly generated CSV files.

## Verification

```bash
python -m unittest covariance_based.test_detector -v
python -m unittest covariance_based.test_paper_reference -v
```

On macOS, the script and package entry points default `MKL_THREADING_LAYER` to
`SEQUENTIAL` before importing NumPy or PyTorch. This prevents MKL from loading
a second OpenMP runtime in environments with conflicting native libraries.
An explicit environment setting is preserved; other operating systems are
unchanged. This runtime setting changes threading, not the solver or data model.
Start a fresh Python process if NumPy or PyTorch was already imported.

Use the environment containing the project dependencies (locally, `cat`), not
an environment without NumPy or PyTorch:

```bash
conda activate cat
python covariance_based/run_monte_carlo.py --mc-times 3
```

For a copy of the script without this compatibility change, prefix the command
with `MKL_THREADING_LAYER=SEQUENTIAL`. Do not suppress duplicate-runtime errors
with `KMP_DUPLICATE_LIB_OK=TRUE`.

## Shared-Test-Set Comparison With GI-HT and HT

`network/run_pm_pf_comparison.py` loads the epoch-100 `last.pt` checkpoints
from the loss-vs-epoch experiment and evaluates both networks and covariance
MLE on the same 5000 event-driven observations. No training is performed.
Its default count is independent of the standalone covariance test count;
use `--mc-times` to override it explicitly.
The physical configuration comes from the checkpoints, and both scenes,
training epochs and non-geometry architecture settings must match. GI-HT
must enable all three geometry branches; HT must disable them and receives
no geometry matrix during inference. Both networks use covariance-row tokens.

```bash
python network/run_pm_pf_comparison.py
# Specify complete trained checkpoints if the default files are missing/damaged:
python network/run_pm_pf_comparison.py --giht-ckpt /path/to/addcorr/last.pt --ht-ckpt /path/to/noaddcorr/last.pt
```

Outputs default to `checkpoint/pm_pf_comparison_261014/` (`--out-dir` overrides
this). The combined PNG/SVG/PDF uses PM horizontally and PF vertically, both
base-10 logarithmic. All methods scan 1001 thresholds in [0,1] using the strict
`score > threshold` rule; zero values remain zero in CSV and are omitted on
log axes. `pm_pf_curves.csv` stores pooled counts; `summary.csv` reports the
threshold-0.5 results; `activity_scores.npz` retains the three score arrays and
shared labels. `covariance_trials.csv` and `run_config.json` record convergence,
noise, checkpoint hashes, physical settings and a shared-observation hash.

Data is always generated one realization at a time on CPU with seed 42;
`--batch-size` controls only neural inference, not the dataset; by default it
inherits the training checkpoint's batch size (128 for this experiment).
TokenBatchNorm uses current-batch statistics even in eval mode, so predictions
can change with inference batch size. Both networks use identical groups;
the final partial batch is retained and its size is recorded. For 5000 trials,
there are 39 groups of 128 and one group of 8. No normalization changes are made.
Inference uses float32 without autocast and can use `--device cuda:0`.
Covariance MLE is recomputed on these observations with the original 100-sweep,
1e-4 stopping criterion and known generated noise variance. Existing covariance
plots and the independent paper reproduction are not overwritten. These curves
compare detection errors, not training loss; disclose the known-noise assumption.

The old epoch-100 checkpoints retain their original higher-load scene. This
comparison script reads settings from checkpoints, not the changed training
defaults. For a low-load comparison, train both models with the new event
defaults first, then specify those checkpoints and a new comparison output
directory. Do not overlay the new low-load covariance curve with old-scene
neural curves as a matched-scene comparison.

## Previous Lower-Load Event Experiment (261014, Lambda=0.5)

The previous event defaults were `EVENT_TRIGGER_PROB=0.6` and `EVENT_LAMBDA=0.5`,
reduced from 0.9 and 1.0. The shared dataclass and JSON defaults used the same
event values; explicit configurations and saved checkpoint scenes still win.
All other train.py scene settings, geometry response scale, background activity,
model, loss and solver formulas are unchanged. No hard bound on K or global
activity-rate calibration is added.

```bash
python covariance_based/run_monte_carlo.py --mc-times 5000 --num-thresholds 1001 --event-lambda 0.5 --out-dir checkpoint/covariance_261014_low_load_event/single
```

The covariance-only plot and count statistics are saved under
`checkpoint/covariance_261014_low_load_event/single/`. The observed min/max K
applies only to these sampled trials; the model's possible range remains 0..N.
That version's Transformer training output was `checkpoint/addcorr_261014_low_load_covrows/`.
Its epoch, L, N and SNR experiment scripts used
`*_261014_low_load_covrows/` output directories to preserve historical runs.
The independent paper reproduction script and existing results are unchanged.

The completed 5000-trial low-load run converged in every realization (mean
12.3002 sweeps). At threshold 0.5, PM=0.267636 and PF=0.003791. Observed K was
0..47, with mean 6.5066 and median 2; the 90th/95th/99th percentiles were
18/22/33. There were 1069 empty trials, all retained. For the 3931 nonempty
trials, K was 1..47 with mean 8.276011. CSV counts, saved statistics, curve
monotonicity/endpoints and the first 30 CPU sample replays were verified.
At threshold zero, PM=0.083208 and PF=0.067051; this curve still has no
equal-PM/PF crossing under the stated threshold rule. Reduced load does not
guarantee equal errors or improved relative neural gains. The PNG was inspected.

```bash
python -m unittest network.test_pm_pf_comparison -v
```

## Previous Event Experiment (261014, Sigma=120, Lambda=1)

At the user's request, `EVENT_LAMBDA` is restored to 1.0 in train.py, the shared
dataclass and the JSON defaults. `EVENT_TRIGGER_PROB=0.6` remains unchanged.
Compared with the preceding run, only the event arrival parameter changes;
N=200, L=10, M=32, radius 500 m, measured SNR 20 dB, response scale 120 m,
event cap 3 and background probability 0.005 are unchanged.

```bash
python covariance_based/run_monte_carlo.py --num-thresholds 1001 --event-sigma-m 120 --out-dir checkpoint/covariance_261014_low_load_lambda1_mc1000_event/single
```

That experiment's covariance output root was
`checkpoint/covariance_261014_low_load_lambda1_mc1000_event/` (single-point outputs in
`single/`). Training and the four neural experiments used
`*_261014_low_load_lambda1_covrows/` paths, preserving lambda=0.5 outputs.
All detector, data-generation, loss and network formulas remain unchanged.
The independent paper reference and all saved results/weights are preserved.

The preceding lambda=1 run had already completed 5000 trials when the user
interrupted to request 1000-trial testing. Its files were saved in
`checkpoint/covariance_261014_low_load_lambda1_event/single/`: K=0..53,
mean K=11.753, PM/PF@0.5=0.366426/0.011948, all solves converged.
The new standalone default is 1000, with the same lambda=1/peak=0.6 scene,
seed and solver; it is rerun rather than merely relabeling the 5000-trial data.

The new 1000-trial test completed with all solves converged (mean 20.769
sweeps). PM/PF@0.5=0.383208/0.013328. Observed K=0..52, mean 12.351,
median 11, p90=29, p95=33 and p99 about 42.01 (linear quantile interpolation).
All 132 empty trials are retained. Counts, histograms, curve endpoints and
the first 30 raw sample replays were checked; every non-runtime field matches
the first 1000 trials of the preceding run. The PNG was visually verified.
These statistics describe this sample, not a hard activity-count limit.

At the final filesystem check, the preceding lambda=1 5000-trial directory
was no longer present. Its hashes and paired-trial checks above were verified
before that disappearance. This workflow issued no deletion commands and does
not restore that old directory; the new 1000-trial outputs remain complete.

## Previous Event Experiment (261014, Sigma=80)

`EVENT_SIGMA_M=80` now replaces 120 in train.py, SystemConfig and the JSON
defaults. Lambda remains 1.0, peak trigger 0.6, background 0.005 and event cap 3.
N=200, L=10, M=32, cell radius 500 m and measured SNR 20 dB are unchanged.
This is a Gaussian response scale, not a hard coverage radius or active-user
cap. With `CORRELATION_LENGTH_M=0`, the geometry kernel automatically uses
sqrt(2)*80 m, keeping the side information consistent with the activity model.
No network, loss, detector or generation formulas are changed.

```bash
python covariance_based/run_monte_carlo.py --num-thresholds 1001
# Later on the server; no training is performed in this local test:
python network/run_loss_vs_epoch.py
```

The standalone covariance test remains 1000 trials and now saves its plot,
configuration and actual-K statistics under
`checkpoint/covariance_261014_sigma80_lambda1_mc1000_event/single/`.
Training and the four neural experiments use
`*_261014_sigma80_lambda1_covrows/` directories. Previous sigma=120 results
and the independent paper reproduction are not overwritten. Saved old model
checkpoints retain their original sigma; use newly trained checkpoints for
a matched-scene sigma=80 comparison.

The completed 1000-trial sigma=80 run converged in every realization (mean
12.879 sweeps). PM/PF@0.5=0.152555/0.001519. Observed K=0..28, mean 6.496,
median 6, p90=15, p95=18 and p99 about 22.01. All 136 empty trials are retained.
The sigma=120 run had K=0..52 and mean 12.351. All 1000 raw sample pairs were
replayed: pilots, positions, channels, event centers and RNG progression match;
the smaller-scale labels are a subset of the original labels. Counts, noise,
histograms, quantiles, curves and endpoints were verified, and the PNG inspected.
The actual event histogram remains [341,372,198,89] for counts 0..3, with mean
1.035 events per trial. The response scale change does not alter event arrivals.
Linear interpolation of the saved threshold curve gives PM=PF about 0.031627
(threshold about 0.002449); this is an interpolated equal-error estimate, not
a claim that a deterministic threshold exactly realizes equal empirical rates.
Previous sigma=120 results and protected code hashes are unchanged.

## Previous Event Experiment (261014, Sigma=100, Trigger=0.6)

The default response scale is now 100 m in train.py, SystemConfig and the JSON
defaults. Lambda=1, peak trigger=0.6, background=0.005, event cap=3, N=200,
L=10, M=32, radius=500 m and measured SNR=20 dB are unchanged. The geometry
kernel follows sqrt(2)*100 m when CORRELATION_LENGTH_M=0. Neither the activity
count nor the event coverage is hard capped by this response scale.
The detector, network, loss and data-generation formulas are unchanged.

```bash
python covariance_based/run_monte_carlo.py --num-thresholds 1001
```

The 1000-trial covariance test writes to
`checkpoint/covariance_261014_sigma100_lambda1_mc1000_event/single/`.
Training and the four neural experiment defaults use
`*_261014_sigma100_lambda1_covrows/` directories. No Transformer training is
performed here; use newly trained weights for a matched-scene comparison.

All 1000 solves converged (mean 16.942 sweeps). PM/PF@0.5=0.270980/0.005734;
K=0..39, mean 9.211, median 8, p90=21, p95=25, p99=31. All 132 empty trials
are retained; the 868 nonempty trials have mean K=10.611751. Sigma=80 had
K=0..28, mean 6.496 and PM/PF@0.5=0.152555/0.001519. The increased response
scale raises the observed load and both error rates in this test.
The actual event histogram remains [341,372,198,89], mean 1.035 events/trial.
All 1000 sigma=80/100 raw pairs were checked under identical random draws:
pilots, positions, channels and event centers match, with sigma=80 labels a
subset of sigma=100 labels. Noise, counts, curves, four run defaults and
protected code/result hashes were verified; 40 tests and compilation passed.
The PNG was inspected. The saved [0,1] threshold curve has no PM=PF crossing
(at threshold 0, PM=0.086744 and PF=0.075088); no equal-error value is inferred.

## Previous Event Experiment (261014, Sigma=100, Trigger=0.5)

The peak event-trigger probability is now 0.5 in train.py, SystemConfig and
the JSON defaults. Sigma=100 m, lambda=1, background=0.005, event cap=3,
N=200, L=10, M=32, radius=500 m and measured SNR=20 dB remain unchanged.
The geometry affinity is unchanged because its kernel depends on distances
and sigma, not the peak trigger probability. No generator, detector, network
or loss formulas are modified; no Transformer training is performed here.

```bash
python covariance_based/run_monte_carlo.py --num-thresholds 1001
```

The 1000-trial covariance test writes to
`checkpoint/covariance_261014_sigma100_trigger0p5_lambda1_mc1000_event/single/`.
Training and the four neural experiment defaults use
`*_261014_sigma100_trigger0p5_lambda1_covrows/`, preserving trigger=0.6 results.
Saved old weights retain their original scenario; retrain for matched-scene
neural comparisons.

The completed 1000-trial test converged in every realization (mean 15.053
sweeps). PM/PF@0.5=0.221785/0.003415; active/inactive totals=7886/192114 and
missed/false counts=1749/656. K=0..33, mean 7.886, median 7, p90=18, p95=21
and p99 about 26.01. All 134 empty trials are retained; the 866 nonempty
trials have mean K=9.106236. With trigger=0.6, mean K was 9.211, maximum 39
and PM/PF@0.5=0.270980/0.005734. These are sample statistics, not hard bounds.
The actual event histogram is unchanged at [341,372,198,89], mean 1.035.
Counts, curves, noise and all 1000 paired raw replays were verified; pilots,
positions, channels, events, affinity matrices and RNG progression match.
The trigger=0.5 active labels are a subset of trigger=0.6 labels. The 0.6
replay reproduces the historical total 9211 and maximum 39; it does not
recreate old detector output files. All 40 tests, compilation, four run
default checks and protected code hashes passed. The PNG was inspected.
Linear interpolation of the saved threshold curve gives PM=PF about 0.054999
at threshold 0.000575; this is an interpolated estimate, not an exactly
attainable deterministic empirical operating point.

The result directory was externally renamed after execution; the verified
files currently reside in
`checkpoint/covariance_261014_eventtriggerprob0.5_sigma100_lambda1/single/`.
The previous trigger=0.6 directory was no longer present at the final audit;
this workflow did not delete or restore it. Historical numbers remain above.

## Previous Event Experiment (261015, Sigma=100, Trigger=0.8)

The peak event-trigger probability is now 0.8 in train.py, SystemConfig and
the JSON defaults. Sigma=100 m, lambda=1, background=0.005, event cap=3,
N=200, L=10, M=32, radius=500 m and measured SNR=20 dB are unchanged.
This is the peak response probability to one event, not the global activity
probability. The affinity matrix is unchanged for the same device geometry
because its kernel depends on position and sigma. All generation, detection,
network and loss formulas are unchanged; no Transformer training is run here.

```bash
python covariance_based/run_monte_carlo.py --num-thresholds 1001
```

The standalone test remains 1000 trials with seed 42, solver budget 100
sweeps/tolerance 1e-4, known generated noise variance and 1001 thresholds.
New results are saved under
`checkpoint/covariance_261015_sigma100_trigger0p8_lambda1_mc1000_event/single/`.
Training and the four neural experiment defaults now use
`*_261015_sigma100_trigger0p8_lambda1_covrows/`; old trigger=0.5 results are
not overwritten. Use newly trained trigger=0.8 weights for a matched-scene
comparison, not the trigger=0.5 weights from the previous experiment.

All 1000 solves converged (mean 20.187 sweeps). PM/PF@0.5=0.366314/0.012340;
active/inactive totals=11916/188084 and missed/false counts=4365/2321.
Observed K=0..49, mean 11.916, median 11, p90=27, p95=33, p99 about 41.01.
All 132 empty trials are retained; the 868 nonempty trials have mean
K=13.728111. These ranges are sample statistics, not hard bounds. Compared
with trigger=0.5 (K=0..33, mean 7.886, PM/PF@0.5=0.221785/0.003415), the
increased triggering raises observed load and both error rates in this test.

All 1000 paired physical samples were replayed: pilots, user positions,
channels, event centers, affinity matrices and RNG progression are identical;
trigger=0.5 labels are a subset of trigger=0.8 labels. Actual K and noise
match the corresponding saved CSVs. The event histogram remains
[341,372,198,89] for 0..3 events, mean 1.035 events/trial. Counts, histograms,
quantiles, curves, endpoints and four neural runner defaults were verified.
All 40 tests and source compilation passed; protected model/loss/detector/
reference code and old trigger=0.5 summary/trial/statistics hashes are unchanged.
The PNG was inspected. At threshold 0, PM=0.146274 and PF=0.092390;
the saved [0,1] threshold curve has no PM=PF crossing, so no EER is reported.

## Previous Event Experiment (261015, Sigma=150, Trigger=0.5)

The default peak event-trigger probability is restored to 0.5 and sigma is
increased to 150 m in train.py, SystemConfig and the JSON defaults. Lambda=1,
background=0.005, event cap=3, N=200, L=10, M=32, radius=500 m and measured
SNR=20 dB are unchanged. Sigma is a Gaussian response scale, not a hard
coverage radius or activity-count cap. With CORRELATION_LENGTH_M=0, the
geometry kernel automatically uses sqrt(2)*150 m. The covariance detector
does not consume that affinity matrix. No generation, model, loss or solver
formulas are changed, and no Transformer training is performed here.

```bash
python covariance_based/run_monte_carlo.py --num-thresholds 1001
```

The standalone run still uses 1000 trials, seed 42, solver budget 100 sweeps,
tolerance 1e-4, known generated noise and 1001 thresholds. New output:
`checkpoint/covariance_261015_sigma150_trigger0p5_lambda1_mc1000_event/single/`.
Training and the four neural runner defaults use
`*_261015_sigma150_trigger0p5_lambda1_covrows/`. Old experiments are not
overwritten. The appropriate reference for isolating the sigma change is
the sigma=100/trigger=0.5 run, not the previous sigma=100/trigger=0.8 run.
Matched-scene neural comparisons require new sigma=150 checkpoints.

All 1000 solves converged (mean 23.389 sweeps). PM/PF@0.5=0.436228/0.020538;
active/inactive totals=14834/185166 and missed/false counts=6471/3803.
Observed K=0..57, mean 14.834, median 13.5, p90 about 34.1, p95=40 and p99
about 47.01 (linear quantiles). All 132 empty trials are retained; the 868
nonempty trials have mean K=17.089862. These ranges are empirical, not hard
bounds. At the same trigger=0.5, sigma=100 had K=0..33, mean 7.886 and
PM/PF@0.5=0.221785/0.003415. Increasing sigma raises the observed load and
both error rates in this test.

All 1000 sigma=100/150 physical sample pairs were replayed. Configurations
differ only in sigma; pilots, positions, channels, events and RNG progression
match. Sigma=100 activity labels are a subset of sigma=150 labels, and the
geometry affinity widens with sigma. K and noise match the corresponding
CSVs. Event counts 0..3 still occur [341,372,198,89] times, mean 1.035 events
per trial. Counts, histograms, quantiles, curves, endpoints, four runner
defaults and source compilation were verified, and all 40 tests passed.
Protected model/loss/detector/reference code and sigma=100/trigger=0.5
summary/trial/statistics hashes are unchanged. The PNG was inspected.
At threshold 0, PM=0.196238 and PF=0.109248; the saved [0,1] curve has no
PM=PF crossing, so no EER is reported.

The completed result directory was moved after execution; verified files
currently reside in
`checkpoint/test/covariance_261015_sigma150_trigger0p5_lambda1_mc1000_event/single/`.
The code's future default output remains the top-level path above. This
workflow does not move or delete existing experiments, train Transformers,
commit changes or push them to the remote.

## New Server Checkpoints: Three-Method Comparison (261015)

For the user-specified sigma=100/trigger=0.5/lambda=1 scene, use both epoch-100
last.pt files from loss_vs_epoch_261014_sigma100_trigger0p5_lambda1_covrows.
Their complete physical configurations match the saved covariance experiment
in covariance_261014_eventtriggerprob0.5_sigma100_lambda1. This evaluation
does not use the current sigma=150 training defaults or the old 260720 weights.

```bash
python network/run_pm_pf_comparison.py --giht-ckpt checkpoint/loss_vs_epoch_261014_sigma100_trigger0p5_lambda1_covrows/addcorr/last.pt --ht-ckpt checkpoint/loss_vs_epoch_261014_sigma100_trigger0p5_lambda1_covrows/noaddcorr/last.pt --out-dir checkpoint/pm_pf_comparison_261015_sigma100_trigger0p5_lambda1 --mc-times 1000 --device cpu
```

The count and seed reproduce the standalone covariance experiment's 1000
realizations. All three methods receive the same observations; the covariance
curve is recalculated on this shared set, not spliced from unrelated data.
Both models use their training batch size 128, retaining the final group of
104 because TokenBatchNorm uses current-batch statistics. HT receives no G;
all its geometry modules are disabled. Inference is float32 without autocast.
Thresholds, covariance settings and known-noise assumptions are unchanged.

The new output directory contains pm_pf_comparison.png/svg/pdf, pm_pf_curves.csv,
summary.csv, activity_scores.npz, covariance_trials.csv and run_config.json.
No training is run and no previous plots, checkpoints or scene defaults are
overwritten. For reproducible publication comparisons, retain these files and
their checkpoint/data hashes together.

The completed run used all 1000 shared realizations (7886 active and 192114
inactive device instances; K=0..33, mean 7.886). At threshold 0.5:

| Method | PM | PF | Missed | False Alarms |
| --- | --- | --- | --- | --- |
| GI-HT | 0.017880 | 0.034865 | 141 | 6698 |
| HT | 0.024854 | 0.046894 | 196 | 9009 |
| Covariance | 0.221785 | 0.003415 | 1749 | 656 |

All 1001 covariance curve points exactly match the specified standalone CSV,
as do all 1000 per-trial noise/count/convergence diagnostics (mean 15.053
sweeps, all converged). Raw scores, labels, curve counts and endpoints were
checked. An independent 1000-sample replay matched the saved observation
digest; checkpoint hashes were unchanged, and the combined PNG was inspected.
No model, loss or current training scene was changed.

Interpolating at all empirical score breakpoints gives equal-error estimates
GI-HT=0.027771, HT=0.037281, Covariance=0.054907. These are interpolated
estimates, not exactly attainable deterministic empirical operating points.
The covariance estimate differs slightly from the old 1001-grid estimate
0.054999 because of threshold resolution; the actual saved curve is unchanged.

## Box-Constrained NNLS Baseline (261015)

`covariance_based/nnls.py` adds a non-neural baseline for known scaled pilots:

```text
minimize ||C_hat - B diag(a) B^H - noise_variance I||_F^2
subject to 0 <= a_i <= 1
```

It is labeled `Box-NNLS` to distinguish it from classical NNLS with no upper
bound. The box matches the existing MLE baseline's available large-scale-gain
information and continuous activity relaxation. The known-gain NNLS variant
is described in Section III-C1 of
[Fengler et al., IEEE TIT 2021](https://arxiv.org/html/1910.11266v2).
Only B, sample covariance and noise variance enter the solver: no true labels,
active count, activity probability, geometry or event centers are used.

The complete complex covariance is realified by stacking real and imaginary
parts. All physical powers share one numerical scale; neither B nor C is
normalized independently, and C_hat-noise*I is not projected onto the PSD cone.
SciPy's `lsq_linear(method="bvls")` solves the bounded convex least-squares
problem, rather than fitting unconstrained LS and clipping afterward.
Tiny numerical boundary drift is snapped only after the bounded solve;
actual bound violations raise an error. Defaults are max_iter=1000 and
tolerance=1e-10. Iteration and tolerance meanings differ from MLE CD sweeps and
coordinate-step tolerance, so NNLS settings and diagnostics are separate.

The joint comparison adds an opt-in flag, preserving its original three-method
default and independent 5000-trial default count. For the current comparison
figure, use the epoch-100 sigma=100/trigger=0.5/lambda=1 checkpoints, not the
current sigma=150 training defaults:

```bash
python network/run_pm_pf_comparison.py --giht-ckpt checkpoint/loss_vs_epoch_261014_sigma100_trigger0p5_lambda1_covrows/addcorr/last.pt --ht-ckpt checkpoint/loss_vs_epoch_261014_sigma100_trigger0p5_lambda1_covrows/noaddcorr/last.pt --out-dir checkpoint/pm_pf_comparison_261015_sigma100_trigger0p5_lambda1_nnls --mc-times 1000 --include-nnls --device cpu
```

The output adds a fourth PM-PF curve, nnls_scores in activity_scores.npz,
nnls_trials.csv and NNLS solver/version/bounds metadata in run_config.json.
The trial CSV reports iterations, convergence, optimality, projected-gradient
norm, status, scaled
residual squared and its physical power scale. summary.csv uses mean_iterations
for NNLS and retains mean_sweeps for MLE. Unconverged trials are retained and
reported, not discarded. All methods use the same strict score>threshold rule.
The old three-method directory and its scores are not overwritten.

SciPy is added as a dependency (>=1.11,<1.16). This Mac's actual NumPy runtime
is 1.26.4 although its distribution metadata says 2.4.6 and both core directory
layouts remain present. After compatibility checks, SciPy 1.12.0 is used here;
the existing NumPy/PyTorch installations are not replaced or repaired. Clean
server environments should install the project's compatible dependencies.

Convergence additionally requires the box projected-gradient infinity norm
to meet the requested tolerance. A relative-cost stopping status alone is not
accepted. During validation, tolerance=1e-8 stopped one sample with optimality
about 1.6e-5; the default was tightened to 1e-10 and the complete evaluation
rerun. This precision check uses only the objective and KKT conditions, not
activity labels or detection performance.

The final 1000-trial run completed in about 83 seconds. All NNLS solves met
the convergence certificate (mean 34.376 iterations, maximum 99); the largest
saved projected-gradient norm was 2.31e-14. Independent reconstruction of all
1000 covariance objectives and KKT checks passed. The original three methods'
scores, labels, curves and MLE trial diagnostics are exactly unchanged.
All four methods share K=0..33, mean 7.886, and the same observation digest.
At threshold 0.5, Box-NNLS has PM=0.311438 and PF=0.008807.

Equal-error estimates from linear interpolation at all raw-score breakpoints:

| Method | Interpolated PM=PF | 10 log10(PM) (dB) |
| --- | --- | --- |
| GI-HT | 0.027771 | -15.5641 |
| HT | 0.037281 | -14.2851 |
| Covariance (MLE) | 0.054907 | -12.6037 |
| Box-NNLS | 0.093076 | -10.3116 |

These estimates need not be exactly attainable by a single deterministic
threshold on finite data. Box-NNLS does not outperform MLE in this scene.
The four-method PNG/SVG/PDF files are in
`checkpoint/pm_pf_comparison_261015_sigma100_trigger0p5_lambda1_nnls/`;
the PNG was visually checked. All 54 unit tests and source compilation passed.

## Final Event Scene (261015, Sigma=100, Trigger=0.9, Lambda=1)

The final experiment defaults are sigma=100 m, event lambda=1 and peak trigger
probability=0.9. They are synchronized in network/train.py, SystemConfig and
network/config.py. Other training settings remain N=200, L=10, M=32, radius
500 m, measured SNR=20 dB, background probability=0.005 and at most 3 events.
The peak trigger is a distance-dependent per-event response parameter, not a
global activity probability. The geometric kernel automatically uses
correlation length sqrt(2)*100 m when correlation_length_m=0.

All four Transformer experiment runners inherit this event scene and use
covariance_rows. Their output names now contain
261015_sigma100_trigger0p9_lambda1_covrows. The loss-vs-epoch runner trains
GI-HT/addcorr and HT/noaddcorr sequentially and stores their checkpoints under
separate mode subdirectories:

```bash
python network/run_loss_vs_epoch.py --device cuda
```

The standalone covariance runner inherits the same scene and writes to
checkpoint/covariance_261015_sigma100_trigger0p9_lambda1_mc1000_event.
After training the new epoch-100 weights, the joint comparison's default
checkpoint paths target the new loss-vs-epoch directory. Both MLE and NNLS
read the scene from the selected weights:

```bash
python covariance_based/run_monte_carlo.py
python network/run_pm_pf_comparison.py --include-nnls --device cpu
```

Historical sections above describe earlier experiments; their reported
metrics do not apply to this final scene. New neural training and a new
evaluation are needed for final-scene performance results.

Validation passed: 54 unit tests, source compilation, and small actual runs of
both modes in all four training entry points. The complete base network used
CPU, 2 epochs, 2 steps per epoch, batch 4 and 2 evaluation batches. Checkpoint
reload, saved scene/geometry switches, finite losses, zero skipped updates,
CSV/PNG output and a 4-realization four-method comparison were checked.
These are workflow checks, not final performance results. This machine has no
CUDA; server GPU/BF16 support and batch-128 memory still need a local check:

```bash
CUDA_VISIBLE_DEVICES=5 python network/run_loss_vs_epoch.py --device cuda --epochs 1 --steps-per-epoch 2 --eval-batches 2 --out-dir checkpoint/server_smoke_261015_sigma100_trigger0p9_lambda1
```

## Minimum PM+PF Sweep Experiments (261015)

`python covariance_based/run_pe_experiments.py` runs covariance MLE and
Box-NNLS on L=4,6,...,14, N=50,100,...,250 and SNR=10,12,...,20 dB, matching the
Transformer experiment lists. The other parameters use the final sigma100,
lambda1, peak-trigger0.9 event scene. Default MC count is 1000 per point.

One threshold per method/point minimizes PM+PF over the entire MC set using
all distinct score breakpoints, including endpoints. This is an offline
empirical minimum, without averaging per-realization minima or dividing by 2.
`metrics.json` saves the threshold, both conditional rates and counts.
The complete exact curve and raw scores are saved in NPZ files. Per-method point
directories save data only, without individual figures. Combined per-point
PM-PF figures (at least two available methods) and summary Pe figures are saved
as PNG/SVG/PDF. Historical individual figures are not deleted automatically.

The shared implementation is in experiments/. Neural training/evaluation is
provided by network/run_pe_experiments.py. New outputs are classified under
result_261015/pe_vs_L, pe_vs_N and pe_vs_SNR. Existing single-point and loss
runners also use result_261015 by default; earlier sections retain historical
paths where they describe completed old experiments. See
[the experiment guide](../result_261015/README.md) for complete commands and
the exact directory structure.
