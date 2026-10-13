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
power control, fading, and measured-SNR noise follow the existing experiments.
Only one realization is kept in memory; no training dataset is created.

The detector receives the unnormalized complex `B` and `Y Y^H / M`, not the
independently normalized Transformer embeddings. Labels are used only to count
detection errors. Generated geometry, event centers, channels, and activity
probabilities are not detector inputs.

By default, the receiver is given the generated noise variance, as in the usual
known-noise covariance baseline. This assumption is recorded as `known_generated`.
It must be stated when comparing against GI-HT. To evaluate a fixed receiver
noise estimate, pass `--receiver-noise-variance VALUE` in physical power units.
The standard Gaussian covariance likelihood is the detector model; the shared
measured-SNR data generator is retained without changing its distribution.

All Monte Carlo realizations, including those with no active devices, contribute
to the false-alarm counts. PM is total missed detections / total active instances;
PF is total false alarms / total inactive instances. A class with no instances
is reported as `nan`, not as a zero error rate. No true active-user count is used
to choose the support or the detection threshold.

Run from the project root, after activating the environment with the project
dependencies. The traditional detector and data generation use CPU, so no GPU
assignment is needed.

```bash
python covariance_based/run_monte_carlo.py --mc-times 1000
python covariance_based/run_monte_carlo.py --sweep L --mc-times 1000
python covariance_based/run_monte_carlo.py --sweep N --mc-times 1000
python covariance_based/run_monte_carlo.py --sweep SNR --mc-times 1000
```

Defaults: `L=[4,6,...,30]`, `N=[50,100,150,200,250]`, and
`SNR=[10,12,14,16,18,20]`. All other parameters stay at their inherited values.
The covariance-only pilot-length override is disabled (`PILOT_LEN_OVERRIDE=None`),
so the current single-point setup inherits `N=200, M=32, L=10, SNR=20 dB` with
event-triggered activity. Set `PILOT_LEN_OVERRIDE` to an integer for a temporary
covariance-only experiment, without changing `network/train.py`. `--pilot-len`
overrides this setting, and an L sweep uses its requested values; N/SNR sweeps
keep the inherited pilot length unless explicitly overridden.
The values printed at launch and `run_config.json`
are the authoritative configuration for each run.

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

The output root is `checkpoint/covariance_261013/`, configured by `OUTPUT_ROOT`
in `run_monte_carlo.py` or `--out-dir`. Single-point experiments use `single/`;
sweeps use `varying_L/`, `varying_N/`, or `varying_SNR/`.

- `summary.csv`: pooled PM/PF, error probability, accuracy, exact-support rate,
  average activity ratio, sweeps, convergence fraction, likelihood, and runtime.
- `trial_results.csv`: counts and solver diagnostics for every realization.
- `pm_pf_curve.csv`: threshold scan from 0 to 1 with pooled counts.
- `run_config.json`: full physical settings, seeds, solver settings, and noise assumption.
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
