from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from time import perf_counter

# Select MKL's non-OpenMP backend before NumPy or PyTorch loads native libraries.
if sys.platform == "darwin":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from covariance_based.detector import covariance_mle_activity
from network.config import load_experiment_config, system_config_from_experiment
from network.data import ActivityDataGenerator, SystemConfig
from network.train import build_args, build_system_config


MC_TIMES = 1000
PILOT_LEN_OVERRIDE = None  # None inherits network/train.py; set an integer for a local override.
USE_PAPER_EVENT_TEST_SETTINGS = False  # Default inherits the Transformer scene; True enables the legacy temporary preset.
PAPER_EVENT_TEST_SETTINGS = {
    "num_devices": 100,
    "pilot_len": 8,
    "cell_radius_m": 250.0,
    "noise_mode": "thermal",
    "noise_power_dbm_hz": -169.0,
    "bandwidth_hz": 10e6,
    "activity_mode": "event",
}
MAX_SWEEPS = 100
TOLERANCE = 1e-4
THRESHOLD = 0.5
NUM_THRESHOLDS = 101
SEED = 42
OUTPUT_ROOT = ROOT / "checkpoint" / "covariance_261015_sigma150_trigger0p5_lambda1_mc1000_event"
PAPER_EVENT_TEST_OUTPUT_ROOT = ROOT / "checkpoint" / "covariance_261014_paper_event"
PLOT_MIN_PROBABILITY = 1e-4
DEFAULT_SWEEPS = {
    "L": ("pilot_len", list(range(4, 31, 2))),
    "N": ("num_devices", list(range(50, 251, 50))),
    "SNR": ("snr_db", list(range(10, 21, 2))),
}
POINT_FIELDS = ["method", "sweep", "sweep_value", "N", "M", "L", "snr_db", "activity_mode", "noise_mode"]
TRIAL_FIELDS = POINT_FIELDS + [
    "trial", "active", "inactive", "missed", "false_alarms", "sweeps", "converged",
    "max_coordinate_step", "runtime_sec", "negative_log_likelihood", "receiver_noise_variance",
]
SUMMARY_FIELDS = POINT_FIELDS + [
    "mc_times", "threshold", "PM", "PF", "error_probability", "accuracy", "exact_support_rate",
    "active", "inactive", "missed", "false_alarms", "mean_activity_ratio", "mean_sweeps",
    "converged_fraction", "mean_runtime_sec", "mean_negative_log_likelihood", "noise_variance_source",
]
CURVE_FIELDS = POINT_FIELDS + ["threshold", "PM", "PF", "active", "inactive", "missed", "false_alarms"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo evaluation of traditional covariance MLE activity detection.")
    parser.add_argument("--config", type=Path, help="Optional experiment JSON; otherwise inherit network/train.py system parameters.")
    parser.add_argument("--mc-times", type=int, default=MC_TIMES)
    parser.add_argument("--max-sweeps", type=int, default=MAX_SWEEPS)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--num-thresholds", type=int, default=NUM_THRESHOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sweep", choices=["none", *DEFAULT_SWEEPS], default="none")
    parser.add_argument("--values", type=float, nargs="+", help="Override values for the selected L, N, or SNR sweep.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--receiver-noise-variance", type=float, help="Fixed receiver noise estimate in physical power units; default uses known generated variance.")
    parser.add_argument("--cyclic-order", action="store_true", help="Visit devices in index order instead of random permutations.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    template = SystemConfig()
    choices = {"activity_mode": ["event", "independent"], "noise_mode": ["snr", "large_scale_snr", "thermal"]}
    for field in fields(SystemConfig):
        flags = ["--" + field.name.replace("_", "-")]
        if field.name == "num_devices":
            flags.append("--num-users")
        parser.add_argument(*flags, dest=field.name, type=type(getattr(template, field.name)),
                            default=None, choices=choices.get(field.name))
    args = parser.parse_args(argv)
    if args.mc_times < 1 or args.max_sweeps < 1 or args.num_thresholds < 2 or args.log_every < 0:
        parser.error("mc-times/max-sweeps must be positive, num-thresholds >= 2, and log-every >= 0.")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("tolerance must be finite and positive.")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0, 1].")
    if args.receiver_noise_variance is not None and (
        not math.isfinite(args.receiver_noise_variance) or args.receiver_noise_variance <= 0
    ):
        parser.error("receiver-noise-variance must be finite and positive.")
    if args.values is not None and args.sweep == "none":
        parser.error("values requires --sweep L, N, or SNR.")
    return args


def experiment_system(args: argparse.Namespace) -> SystemConfig:
    if args.config is None:
        cfg = build_system_config(build_args())
    else:
        if not args.config.is_file():
            raise FileNotFoundError(f"Experiment config not found: {args.config}")
        cfg = system_config_from_experiment(load_experiment_config(args.config))
    if USE_PAPER_EVENT_TEST_SETTINGS:
        cfg = replace(cfg, **PAPER_EVENT_TEST_SETTINGS)
    if PILOT_LEN_OVERRIDE is not None:
        cfg = replace(cfg, pilot_len=PILOT_LEN_OVERRIDE)
    overrides = {f.name: getattr(args, f.name) for f in fields(SystemConfig) if getattr(args, f.name) is not None}
    return replace(cfg, **overrides)


def experiment_points(cfg: SystemConfig, args: argparse.Namespace) -> list[tuple[str, float | str, SystemConfig]]:
    if args.sweep == "none":
        return [("none", "", cfg)]
    if args.sweep == "SNR" and cfg.noise_mode == "thermal":
        raise ValueError("SNR sweep requires --noise-mode snr or large_scale_snr; fixed thermal noise ignores snr_db.")
    name, defaults = DEFAULT_SWEEPS[args.sweep]
    values = defaults if args.values is None else args.values
    points = []
    for value in values:
        if not math.isfinite(value):
            raise ValueError("Sweep values must be finite.")
        if name != "snr_db":
            if value < 1 or int(value) != value:
                raise ValueError("L and N sweep values must be positive integers.")
            value = int(value)
        points.append((args.sweep, value, replace(cfg, **{name: value})))
    if len({p[1] for p in points}) != len(points):
        raise ValueError("Sweep values must not contain duplicates.")
    return points


def validate_system(cfg: SystemConfig) -> None:
    if min(cfg.num_devices, cfg.num_antennas, cfg.pilot_len, cfg.event_max_count) < 1:
        raise ValueError("N, M, L, and event_max_count must be positive.")
    for name, value in asdict(cfg).items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if cfg.cell_radius_m <= 0 or cfg.event_sigma_m <= 0 or cfg.bandwidth_hz <= 0 or cfg.event_lambda < 0:
        raise ValueError("Spatial scales/bandwidth must be positive and event_lambda nonnegative.")
    for name in ("activity_prob", "event_trigger_prob", "background_activity_prob"):
        if not 0 <= getattr(cfg, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1].")


def rate(count: int, total: int) -> float:
    return float(count / total) if total else float("nan")


def detection_counts(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = scores[None, :] > thresholds[:, None]
    missed = np.count_nonzero(~predicted & labels[None, :], axis=1)
    false_alarms = np.count_nonzero(predicted & ~labels[None, :], axis=1)
    return missed, false_alarms


@torch.no_grad()
def run_condition(
    cfg: SystemConfig,
    args: argparse.Namespace,
    sweep: str,
    value: float | str,
    trial_writer: csv.DictWriter,
) -> tuple[dict, list[dict]]:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    generator = ActivityDataGenerator(cfg, torch.device("cpu"))
    thresholds = np.linspace(0.0, 1.0, args.num_thresholds)
    missed = np.zeros(len(thresholds), dtype=np.int64)
    false_alarms = np.zeros(len(thresholds), dtype=np.int64)
    total_active = total_inactive = point_missed = point_false = exact = converged = 0
    total_runtime = total_sweeps = total_objective = 0.0
    metadata = {
        "method": "covariance_mle_cd", "sweep": sweep, "sweep_value": value,
        "N": cfg.num_devices, "M": cfg.num_antennas, "L": cfg.pilot_len,
        "snr_db": cfg.snr_db, "activity_mode": cfg.activity_mode, "noise_mode": cfg.noise_mode,
    }
    for trial in range(1, args.mc_times + 1):
        batch = generator.sample_batch(1, return_raw=True)
        b = batch["b"][0].numpy().astype(np.complex128)
        y = batch["y"][0].numpy().astype(np.complex128)
        labels = batch["label"][0].numpy().astype(bool)
        noise = float(batch["noise_var"][0]) if args.receiver_noise_variance is None else args.receiver_noise_variance
        start = perf_counter()
        covariance = y @ y.conj().T / cfg.num_antennas
        result = covariance_mle_activity(b, covariance, noise, max_sweeps=args.max_sweeps,
                                         tolerance=args.tolerance, random_order=not args.cyclic_order, rng=rng)
        runtime = perf_counter() - start
        trial_missed, trial_false = detection_counts(result.activities, labels, np.array([args.threshold]))
        curve_missed, curve_false = detection_counts(result.activities, labels, thresholds)
        active = int(labels.sum())
        inactive = cfg.num_devices - active
        fn, fp = int(trial_missed[0]), int(trial_false[0])
        total_active += active
        total_inactive += inactive
        point_missed += fn
        point_false += fp
        missed += curve_missed
        false_alarms += curve_false
        exact += int(fn == 0 and fp == 0)
        converged += int(result.converged)
        total_sweeps += result.sweeps
        total_runtime += runtime
        total_objective += result.objective_history[-1]
        trial_writer.writerow({**metadata, "trial": trial, "active": active, "inactive": inactive,
                               "missed": fn, "false_alarms": fp, "sweeps": result.sweeps,
                               "converged": result.converged, "max_coordinate_step": result.max_coordinate_step,
                               "runtime_sec": runtime, "negative_log_likelihood": result.objective_history[-1],
                               "receiver_noise_variance": noise})
        if args.log_every and (trial % args.log_every == 0 or trial == args.mc_times):
            print(f"{sweep}={value} MC {trial}/{args.mc_times} | PM={rate(point_missed, total_active):.6f} "
                  f"PF={rate(point_false, total_inactive):.6f} | converged={converged}/{trial}", flush=True)

    instances = args.mc_times * cfg.num_devices
    error = (point_missed + point_false) / instances
    summary = {**metadata, "mc_times": args.mc_times, "threshold": args.threshold,
               "PM": rate(point_missed, total_active), "PF": rate(point_false, total_inactive),
               "error_probability": error, "accuracy": 1.0 - error, "exact_support_rate": exact / args.mc_times,
               "active": total_active, "inactive": total_inactive, "missed": point_missed,
               "false_alarms": point_false, "mean_activity_ratio": total_active / instances,
               "mean_sweeps": total_sweeps / args.mc_times, "converged_fraction": converged / args.mc_times,
               "mean_runtime_sec": total_runtime / args.mc_times,
               "mean_negative_log_likelihood": total_objective / args.mc_times,
               "noise_variance_source": "known_generated" if args.receiver_noise_variance is None else "fixed_receiver_estimate"}
    curves = [{**metadata, "threshold": float(th), "PM": rate(int(fn), total_active),
               "PF": rate(int(fp), total_inactive), "active": total_active, "inactive": total_inactive,
               "missed": int(fn), "false_alarms": int(fp)} for th, fn, fp in zip(thresholds, missed, false_alarms)]
    return summary, curves


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_activity_counts(counts: list[int], num_devices: int) -> dict:
    values = np.asarray(counts)
    if num_devices < 1 or values.ndim != 1 or not values.size:
        raise ValueError("Activity counts must be a nonempty one-dimensional sequence.")
    if not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any((values < 0) | (values > num_devices)):
        raise ValueError("Activity counts must be integers in [0, num_devices].")
    values = values.astype(np.int64)
    nonempty = values[values > 0]
    return {
        "mc_times": int(values.size), "min_active": int(values.min()), "max_active": int(values.max()),
        "mean_active": float(values.mean()), "median_active": float(np.median(values)),
        "p90_active": float(np.percentile(values, 90)), "p95_active": float(np.percentile(values, 95)),
        "p99_active": float(np.percentile(values, 99)), "zero_active_trials": int(np.sum(values == 0)),
        "nonempty_trials": int(nonempty.size), "min_active_nonempty": int(nonempty.min()) if nonempty.size else None,
        "mean_active_nonempty": float(nonempty.mean()) if nonempty.size else None,
        "mean_activity_ratio": float(values.mean() / num_devices),
        "histogram_by_active_count": np.bincount(values, minlength=num_devices + 1).tolist(),
    }


def save_activity_count_stats(out_dir: Path) -> list[dict]:
    # Read generated truth counts, not the detector's estimated support sizes.
    groups: dict[tuple[str, ...], list[int]] = {}
    with (out_dir / "trial_results.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[name] for name in POINT_FIELDS)
            groups.setdefault(key, []).append(int(row["active"]))
    records = []
    for key, counts in groups.items():
        metadata = dict(zip(POINT_FIELDS, key))
        for name in ("N", "M", "L"):
            metadata[name] = int(metadata[name])
        metadata["snr_db"] = float(metadata["snr_db"])
        stats = summarize_activity_counts(counts, metadata["N"])
        records.append({**metadata, **stats})
        print(f"Activity counts {metadata['sweep']}={metadata['sweep_value']}: "
              f"K={stats['min_active']}..{stats['max_active']}, mean={stats['mean_active']:.4f}, "
              f"p95={stats['p95_active']:g}, p99={stats['p99_active']:g}, "
              f"empty={stats['zero_active_trials']}/{stats['mc_times']}", flush=True)
    (out_dir / "activity_count_stats.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return records


def positive_plot_values(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values > 0.0), values, np.nan)


def probability_log_axis(ax, axis: str, values: list[float]) -> None:
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    positive = positive_plot_values(values)
    positive = positive[np.isfinite(positive)]
    lower = PLOT_MIN_PROBABILITY
    if positive.size:
        lower = min(lower, 10.0 ** math.floor(math.log10(float(positive.min()))))
    if axis == "x":
        ax.set_xscale("log", base=10, nonpositive="mask")
        ax.set_xlim(lower, 1.0)
        target = ax.xaxis
    elif axis == "y":
        ax.set_yscale("log", base=10, nonpositive="mask")
        ax.set_ylim(lower, 1.0)
        target = ax.yaxis
    else:
        raise ValueError("Probability axis must be x or y.")
    target.set_major_locator(LogLocator(base=10, numticks=max(7, int(-math.log10(lower)) + 3)))
    target.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
    target.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    target.set_minor_formatter(NullFormatter())


def plot_results(out_dir: Path, summaries: list[dict], curves: list[list[dict]], sweep: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; CSV results are complete. Install matplotlib to enable plots.")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for summary, curve in zip(summaries, curves):
        label = "Covariance (MLE-CD)" if sweep == "none" else f"{sweep}={summary['sweep_value']:g}"
        ax.plot(positive_plot_values([r["PM"] for r in curve]),
                positive_plot_values([r["PF"] for r in curve]), label=label)
    ax.set(xlabel=r"Missed-detection probability $P_M$", ylabel=r"False-alarm probability $P_F$")
    probability_log_axis(ax, "x", [r["PM"] for curve in curves for r in curve])
    probability_log_axis(ax, "y", [r["PF"] for curve in curves for r in curve])
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out_dir / f"pm_pf_curve.{suffix}", dpi=300)
    plt.close(fig)
    if sweep != "none":
        labels = {"L": "Pilot length L", "N": "Number of devices N", "SNR": "SNR (dB)"}
        xs = [s["sweep_value"] for s in summaries]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ("PM", "PF")):
            values = [s[metric] for s in summaries]
            ax.plot(xs, positive_plot_values(values), "o-")
            ax.set(xlabel=labels[sweep], ylabel=metric)
            if xs:
                padding = 0.5 if min(xs) == max(xs) else 0.0
                ax.set_xlim(min(xs) - padding, max(xs) + padding)
                ax.set_xticks(xs)
            probability_log_axis(ax, "y", values)
            ax.grid(True, which="major", alpha=0.3)
            ax.grid(True, which="minor", linestyle=":", alpha=0.15)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(out_dir / f"detection_vs_{sweep}.{suffix}", dpi=300)
        plt.close(fig)
    print("Logarithmic plots omit zero/undefined probabilities; CSV values are unchanged.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = experiment_system(args)
    points = experiment_points(cfg, args)
    for _, _, system in points:
        validate_system(system)
    output_root = PAPER_EVENT_TEST_OUTPUT_ROOT if USE_PAPER_EVENT_TEST_SETTINGS else OUTPUT_ROOT
    out_dir = args.out_dir or output_root / ("single" if args.sweep == "none" else f"varying_{args.sweep}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    settings = {**vars(args), "out_dir": str(out_dir.resolve()), "system_config": asdict(cfg),
                "use_paper_event_test_settings": USE_PAPER_EVENT_TEST_SETTINGS,
                "points": [{"sweep": sweep, "value": value, "system_config": asdict(system)} for sweep, value, system in points],
                "solver": "box-constrained covariance MLE with coordinate descent", "solver_device": "cpu",
                "noise_variance_source": "known_generated" if args.receiver_noise_variance is None else "fixed_receiver_estimate"}
    (out_dir / "run_config.json").write_text(json.dumps(settings, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    print(f"System: {json.dumps(asdict(cfg), sort_keys=True)}", flush=True)
    print(f"MC times per point: {args.mc_times}; CPU solver; output: {out_dir.resolve()}", flush=True)
    print(f"Receiver noise variance: {settings['noise_variance_source']}", flush=True)
    summaries, all_curves = [], []
    with (out_dir / "trial_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        for sweep, value, system in points:
            summary, curve = run_condition(system, args, sweep, value, writer)
            summaries.append(summary)
            all_curves.append(curve)
            write_csv(out_dir / "summary.csv", SUMMARY_FIELDS, summaries)
            write_csv(out_dir / "pm_pf_curve.csv", CURVE_FIELDS, [r for point in all_curves for r in point])
            handle.flush()
            print(f"Completed {sweep}={value} | PM@{args.threshold:.2f}={summary['PM']:.6f} "
                  f"PF@{args.threshold:.2f}={summary['PF']:.6f} | mean sweeps={summary['mean_sweeps']:.1f} "
                  f"converged={summary['converged_fraction']:.1%}", flush=True)
    save_activity_count_stats(out_dir)
    if not args.no_plot:
        plot_results(out_dir, summaries, all_curves, args.sweep)
    print(f"Saved results to: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
