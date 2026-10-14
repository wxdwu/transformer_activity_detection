from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from covariance_based.detector import covariance_mle_activity
from covariance_based.run_monte_carlo import (
    CURVE_FIELDS, POINT_FIELDS, SUMMARY_FIELDS, TRIAL_FIELDS, detection_counts,
    positive_plot_values, probability_log_axis, rate, write_csv,
)
import numpy as np


REFERENCE_OUTPUT_DIR = ROOT / "checkpoint" / "covariance_261014_paper_reproduction"
PAPER_CASES = ((7, 32), (8, 32), (8, 64))
MC_TIMES = 5000
MAX_SWEEPS = 500
TOLERANCE = 1e-6
PAPER_URL = "https://arxiv.org/html/2112.10086v1#S5"
SUMMARY_EXTRA_FIELDS = ["active_at_zero", "inactive_at_one", "PM_at_zero", "PF_below_one"]
AUDIT_FIELDS = ["L", "M", "trial", "sweeps", "max_score_difference",
                "objective_difference", "direct_max_coordinate_step"]


@dataclass(frozen=True)
class PaperSystem:
    num_devices: int = 100
    num_antennas: int = 32
    pilot_len: int = 8
    cell_radius_m: float = 250.0
    pmax_dbm: float = 23.0
    noise_power_dbm_hz: float = -169.0
    bandwidth_hz: float = 10e6
    activity_mode: str = "independent"
    activity_prob: float = 0.1
    noise_mode: str = "thermal"
    active_count: int | None = None

    @property
    def noise_variance(self) -> float:
        return 10.0 ** ((self.noise_power_dbm_hz - 30.0) / 10.0) * self.bandwidth_hz


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the three covariance curves in the local HT paper's Fig. 4.")
    parser.add_argument("--mc-times", type=int, default=MC_TIMES)
    parser.add_argument("--max-sweeps", type=int, default=MAX_SWEEPS)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-thresholds", type=int, default=1001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases", nargs="+", help="L:M pairs, e.g. --cases 7:32 8:32 8:64.")
    parser.add_argument("--pilot-len", type=int, help="Run a single L (default single-case M is 32).")
    parser.add_argument("--num-antennas", type=int, help="Run a single M (default single-case L is 8).")
    parser.add_argument("--num-devices", "--num-users", type=int, default=100)
    parser.add_argument("--cell-radius-m", type=float, default=250.0)
    parser.add_argument("--pmax-dbm", type=float, default=23.0)
    parser.add_argument("--noise-power-dbm-hz", type=float, default=-169.0)
    parser.add_argument("--bandwidth-hz", type=float, default=10e6)
    parser.add_argument("--activity-prob", type=float, default=0.1, help="Independent Bernoulli probability; fixed count uses K/N.")
    parser.add_argument("--active-count", type=int, help="Optional uniformly sampled fixed-size support; NOT independent Bernoulli.")
    parser.add_argument("--activity-mode", choices=["independent", "fixed_count"], default="independent")
    parser.add_argument("--noise-mode", choices=["thermal"], default="thermal")
    parser.add_argument("--cyclic-order", action="store_true")
    parser.add_argument("--audit-trials", type=int, default=3,
                        help="Check the first trials of each case with direct inversion at every coordinate.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if min(args.mc_times, args.max_sweeps) < 1 or args.num_thresholds < 2:
        parser.error("mc-times/max-sweeps must be positive and num-thresholds >= 2.")
    if args.audit_trials < 0 or args.log_every < 0 or args.seed < 0:
        parser.error("audit-trials/log-every/seed must be nonnegative.")
    if not np.isfinite(args.tolerance) or args.tolerance <= 0 or not 0 <= args.threshold <= 1:
        parser.error("tolerance must be finite and positive; threshold must be in [0, 1].")
    if args.cases and (args.pilot_len is not None or args.num_antennas is not None):
        parser.error("Use --cases or a single --pilot-len/--num-antennas, not both.")
    if args.num_devices < 1:
        parser.error("num-devices must be positive.")
    if not np.isfinite(args.activity_prob) or not 0 <= args.activity_prob <= 1:
        parser.error("activity-prob must be finite and in [0, 1].")
    if args.active_count is not None:
        if not 0 <= args.active_count <= args.num_devices:
            parser.error("active-count must be between zero and num-devices.")
        args.activity_mode = "fixed_count"
    elif args.activity_mode == "fixed_count":
        parser.error("fixed_count requires --active-count K.")
    if args.out_dir is None:
        args.out_dir = (REFERENCE_OUTPUT_DIR if args.active_count is None else
                        ROOT / "checkpoint" / f"covariance_261014_paper_fixedK{args.active_count}")
    try:
        if args.cases:
            cases = [tuple(map(int, pair.split(":"))) for pair in args.cases]
        elif args.pilot_len is not None or args.num_antennas is not None:
            cases = [(args.pilot_len if args.pilot_len is not None else 8,
                      args.num_antennas if args.num_antennas is not None else 32)]
        else:
            cases = list(PAPER_CASES)
        if any(len(pair) != 2 or min(pair) < 1 for pair in cases) or len(set(cases)) != len(cases):
            raise ValueError
    except ValueError:
        parser.error("cases must be distinct positive L:M pairs.")
    args.cases = cases
    cfg = experiment_system(args)
    if (cfg.num_devices < 1 or cfg.cell_radius_m <= 0 or cfg.bandwidth_hz <= 0
            or not 0 <= cfg.activity_prob <= 1
            or not all(np.isfinite(value) for value in (cfg.cell_radius_m, cfg.pmax_dbm,
                       cfg.noise_power_dbm_hz, cfg.bandwidth_hz, cfg.activity_prob))
            or not np.isfinite(cfg.noise_variance) or cfg.noise_variance <= 0):
        parser.error("Invalid physical parameters.")
    return args


def experiment_system(args: argparse.Namespace) -> PaperSystem:
    names = ("num_devices", "cell_radius_m", "pmax_dbm", "noise_power_dbm_hz", "bandwidth_hz",
             "activity_mode", "noise_mode", "active_count")
    probability = args.activity_prob if args.active_count is None else args.active_count / args.num_devices
    return PaperSystem(**{name: getattr(args, name) for name in names}, activity_prob=probability,
                       pilot_len=args.pilot_len if args.pilot_len is not None else 8,
                       num_antennas=args.num_antennas if args.num_antennas is not None else 32)


def complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def paper_realization(cfg: PaperSystem, rng: np.random.Generator,
                      max_l: int, max_m: int) -> dict[str, np.ndarray]:
    """Draw one physical realization; prefix slicing pairs the three cases."""
    radius = cfg.cell_radius_m * np.sqrt(rng.random(cfg.num_devices))
    angle = rng.uniform(0.0, 2.0 * np.pi, cfg.num_devices)
    positions = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    distance_km = np.maximum(radius, 1.0) / 1000.0
    gain = 10.0 ** (-(128.1 + 37.6 * np.log10(distance_km)) / 10.0)
    power = 10.0 ** ((cfg.pmax_dbm - 30.0) / 10.0) * gain.min() / gain
    # Paper pilots have unit entry variance, not unit expected vector energy.
    pilots = complex_normal(rng, (max_l, cfg.num_devices))
    activity_uniform = rng.random(cfg.num_devices)
    labels = activity_uniform < cfg.activity_prob
    if cfg.active_count is not None:
        # Uniform ranks give a uniform K-subset while preserving RNG pairing.
        labels = np.zeros(cfg.num_devices, dtype=bool)
        if cfg.active_count:
            labels[np.argpartition(activity_uniform, cfg.active_count - 1)[:cfg.active_count]] = True
    channel = complex_normal(rng, (cfg.num_devices, max_m))
    noise = complex_normal(rng, (max_l, max_m)) * np.sqrt(cfg.noise_variance)
    b = pilots[:cfg.pilot_len] * np.sqrt(power * gain)
    h = channel[:, :cfg.num_antennas]
    w = noise[:cfg.pilot_len, :cfg.num_antennas]
    y = (b * labels) @ h + w
    return {"b": b, "y": y, "labels": labels, "pilots": pilots[:cfg.pilot_len],
            "channel": h, "noise": w, "positions": positions, "gain": gain, "power": power}


def direct_coordinate_check(b: np.ndarray, c: np.ndarray, noise: float,
                            sweeps: int, rng: np.random.Generator,
                            random_order: bool) -> tuple[np.ndarray, float, float]:
    """Independent check: reconstruct and invert Sigma at every coordinate."""
    b, c = b / np.sqrt(noise), c / noise
    a = np.zeros(b.shape[1])
    eye = np.eye(b.shape[0])

    def inverse() -> np.ndarray:
        return np.linalg.inv((b * a) @ b.conj().T + eye)

    for _ in range(sweeps):
        order = rng.permutation(b.shape[1]) if random_order else range(b.shape[1])
        for index in order:
            v = inverse() @ b[:, index]
            q = np.vdot(b[:, index], v).real
            r = np.vdot(v, c @ v).real
            a[index] = np.clip(a[index] + (r - q) / q ** 2, 0.0, 1.0)
    sigma = (b * a) @ b.conj().T + eye
    inv = inverse()
    v = inv @ b
    q = np.sum(b.conj() * v, axis=0).real
    r = np.sum(v.conj() * (c @ v), axis=0).real
    step = np.max(np.abs(np.clip(a + (r - q) / q ** 2, 0.0, 1.0) - a))
    objective = np.linalg.slogdet(sigma)[1] + np.trace(inv @ c).real + b.shape[0] * np.log(noise)
    return a, float(objective), float(step)


def threshold_grid(number: int) -> np.ndarray:
    edges = np.geomspace(1e-8, 0.1, 120)
    return np.unique(np.concatenate((np.linspace(0.0, 1.0, number), edges, 1.0 - edges)))


def run_case(cfg: PaperSystem, args: argparse.Namespace, max_l: int, max_m: int,
             out_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    thresholds = threshold_grid(args.num_thresholds)
    missed = np.zeros(len(thresholds), dtype=np.int64)
    false = np.zeros_like(missed)
    active = inactive = fn_total = fp_total = exact = converged = zeros = ones = 0
    runtime_total = sweeps_total = objective_total = 0.0
    scores = np.empty((args.mc_times, cfg.num_devices))
    labels_all = np.empty_like(scores, dtype=bool)
    audits = []
    metadata = {"method": "covariance_mle_cd", "sweep": "paper_L_M",
                "sweep_value": f"L{cfg.pilot_len}_M{cfg.num_antennas}",
                "N": cfg.num_devices, "M": cfg.num_antennas, "L": cfg.pilot_len,
                "snr_db": "", "activity_mode": cfg.activity_mode, "noise_mode": "thermal"}
    with (out_dir / "trial_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        for trial in range(1, args.mc_times + 1):
            raw = paper_realization(cfg, rng, max_l, max_m)
            b, y, labels = raw["b"], raw["y"], raw["labels"]
            solver_seed = np.random.SeedSequence([args.seed, trial, 1])
            start = perf_counter()
            c = y @ y.conj().T / cfg.num_antennas
            result = covariance_mle_activity(b, c, cfg.noise_variance,
                max_sweeps=args.max_sweeps, tolerance=args.tolerance,
                random_order=not args.cyclic_order, rng=np.random.default_rng(solver_seed))
            runtime = perf_counter() - start
            if trial <= args.audit_trials:
                direct, objective, step = direct_coordinate_check(b, c, cfg.noise_variance,
                    result.sweeps, np.random.default_rng(solver_seed), not args.cyclic_order)
                difference = float(np.max(np.abs(result.activities - direct)))
                objective_difference = abs(result.objective_history[-1] - objective)
                if (difference > 2e-6 or objective_difference > 1e-7
                        or (result.converged and step > args.tolerance * 1.1)
                        or np.any(np.diff(result.objective_history) > 1e-7)):
                    raise AssertionError(f"Direct-inversion solver audit failed: {difference=}, {objective_difference=}, {step=}.")
                audits.append({"L": cfg.pilot_len, "M": cfg.num_antennas, "trial": trial,
                               "sweeps": result.sweeps, "max_score_difference": difference,
                               "objective_difference": objective_difference, "direct_max_coordinate_step": step})
            scores[trial - 1], labels_all[trial - 1] = result.activities, labels
            point_fn, point_fp = detection_counts(result.activities, labels, np.array([args.threshold]))
            curve_fn, curve_fp = detection_counts(result.activities, labels, thresholds)
            n_active = int(labels.sum())
            n_inactive = cfg.num_devices - n_active
            fn, fp = int(point_fn[0]), int(point_fp[0])
            active += n_active
            inactive += n_inactive
            fn_total += fn
            fp_total += fp
            missed += curve_fn
            false += curve_fp
            exact += int(fn == 0 and fp == 0)
            converged += int(result.converged)
            zeros += int(np.count_nonzero(labels & (result.activities == 0.0)))
            ones += int(np.count_nonzero(~labels & (result.activities == 1.0)))
            runtime_total += runtime
            sweeps_total += result.sweeps
            objective_total += result.objective_history[-1]
            writer.writerow({**metadata, "trial": trial, "active": n_active, "inactive": n_inactive,
                "missed": fn, "false_alarms": fp, "sweeps": result.sweeps, "converged": result.converged,
                "max_coordinate_step": result.max_coordinate_step, "runtime_sec": runtime,
                "negative_log_likelihood": result.objective_history[-1],
                "receiver_noise_variance": cfg.noise_variance})
            if args.log_every and (trial % args.log_every == 0 or trial == args.mc_times):
                print(f"L={cfg.pilot_len}, M={cfg.num_antennas} MC {trial}/{args.mc_times} | "
                      f"PM={rate(fn_total, active):.6f} PF={rate(fp_total, inactive):.6f} | "
                      f"converged={converged}/{trial}", flush=True)
    count = args.mc_times * cfg.num_devices
    summary = {**metadata, "mc_times": args.mc_times, "threshold": args.threshold,
        "PM": rate(fn_total, active), "PF": rate(fp_total, inactive),
        "error_probability": (fn_total + fp_total) / count, "accuracy": 1 - (fn_total + fp_total) / count,
        "exact_support_rate": exact / args.mc_times, "active": active, "inactive": inactive,
        "missed": fn_total, "false_alarms": fp_total, "mean_activity_ratio": active / count,
        "mean_sweeps": sweeps_total / args.mc_times, "converged_fraction": converged / args.mc_times,
        "mean_runtime_sec": runtime_total / args.mc_times,
        "mean_negative_log_likelihood": objective_total / args.mc_times,
        "noise_variance_source": "known_generated", "active_at_zero": zeros,
        "inactive_at_one": ones, "PM_at_zero": rate(zeros, active), "PF_below_one": rate(ones, inactive)}
    curve = [{**metadata, "threshold": float(th), "PM": rate(int(fn), active), "PF": rate(int(fp), inactive),
              "active": active, "inactive": inactive, "missed": int(fn), "false_alarms": int(fp)}
             for th, fn, fp in zip(thresholds, missed, false)]
    write_csv(out_dir / "summary.csv", SUMMARY_FIELDS + SUMMARY_EXTRA_FIELDS, [summary])
    write_csv(out_dir / "pm_pf_curve.csv", CURVE_FIELDS, curve)
    write_csv(out_dir / "solver_audit.csv", AUDIT_FIELDS, audits)
    np.savez_compressed(out_dir / "activity_scores.npz", scores=scores, labels=labels_all)
    return summary, curve, audits


def plot_reference(out_dir: Path, summaries: list[dict], curves: list[list[dict]],
                   title: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for index, (summary, curve) in enumerate(zip(summaries, curves)):
        ax.plot(positive_plot_values([r["PM"] for r in curve]),
                positive_plot_values([r["PF"] for r in curve]), color="#c66a16",
                linestyle=("-", "--", ":")[index % 3], linewidth=1.8,
                label=rf"Covariance, $L={summary['L']}, M={summary['M']}$")
    ax.set(xlabel=r"Missed-detection probability $P_M$", ylabel=r"False-alarm probability $P_F$")
    if title is not None:
        ax.set_title(title, fontsize=10)
    probability_log_axis(ax, "x", [r["PM"] for curve in curves for r in curve])
    probability_log_axis(ax, "y", [r["PF"] for curve in curves for r in curve])
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out_dir / f"pm_pf_curve.{suffix}", dpi=300)
    plt.close(fig)


def save_count_diagnostic(out_dir: Path, summaries: list[dict], active_count: int,
                          number_thresholds: int, make_plot: bool) -> tuple[list[dict], list[list[dict]]]:
    """Condition evaluation on true K; do not pass K to the detector."""
    selected_summaries, selected_curves = [], []
    thresholds = threshold_grid(number_thresholds)
    for summary in summaries:
        with np.load(out_dir / f"L{summary['L']}_M{summary['M']}" / "activity_scores.npz") as data:
            mask = data["labels"].sum(axis=1) == active_count
            scores, labels = data["scores"][mask], data["labels"][mask]
        if not len(labels):
            continue
        active, inactive = int(labels.sum()), int((~labels).sum())
        metadata = {key: summary[key] for key in POINT_FIELDS}
        curve = [{**metadata, "threshold": float(th),
                  "PM": rate(int(np.count_nonzero(labels & (scores <= th))), active),
                  "PF": rate(int(np.count_nonzero(~labels & (scores > th))), inactive),
                  "active": active, "inactive": inactive,
                  "missed": int(np.count_nonzero(labels & (scores <= th))),
                  "false_alarms": int(np.count_nonzero(~labels & (scores > th)))} for th in thresholds]
        th = summary["threshold"]
        selected_summaries.append({**metadata, "source_mc_times": summary["mc_times"],
            "selected_trials": len(labels), "true_active_count": active_count, "threshold": th,
            "PM": rate(int(np.count_nonzero(labels & (scores <= th))), active),
            "PF": rate(int(np.count_nonzero(~labels & (scores > th))), inactive),
            "PM_at_zero": rate(int(np.count_nonzero(labels & (scores == 0))), active),
            "PF_below_one": rate(int(np.count_nonzero(~labels & (scores == 1))), inactive)})
        selected_curves.append(curve)
    if selected_summaries:
        diagnostic_dir = out_dir / f"K{active_count}_subset_diagnostic"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        write_csv(diagnostic_dir / "summary.csv", list(selected_summaries[0]), selected_summaries)
        write_csv(diagnostic_dir / "pm_pf_curve.csv", CURVE_FIELDS,
                  [r for curve in selected_curves for r in curve])
        diagnostic_settings = {"purpose": "conditional diagnostic, not the all-trial experiment",
            "true_active_count": active_count, "source_root": str(out_dir.resolve()),
            "source_mc_times": [s["source_mc_times"] for s in selected_summaries],
            "selected_trials": [s["selected_trials"] for s in selected_summaries],
            "detector_uses_true_K": False, "new_physical_samples_generated": False}
        (diagnostic_dir / "run_config.json").write_text(
            json.dumps(diagnostic_settings, indent=2) + "\n", encoding="utf-8")
        if make_plot:
            counts = ", ".join(str(s["selected_trials"]) for s in selected_summaries)
            plot_reference(diagnostic_dir, selected_summaries, selected_curves,
                           f"Conditional diagnostic: true K={active_count}; selected trials per case: {counts}")
        print(f"Conditional K={active_count} diagnostic saved to: {diagnostic_dir}", flush=True)
    return selected_summaries, selected_curves


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = experiment_system(args)
    points = [replace(cfg, pilot_len=l, num_antennas=m) for l, m in args.cases]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {**vars(args), "out_dir": str(out_dir.resolve()), "paper_url": PAPER_URL,
        "system_config": asdict(cfg), "points": [asdict(point) for point in points],
        "pilot_entry_distribution": "CN(0, 1)", "channel_entry_distribution": "CN(0, 1)",
        "noise_variance_w": cfg.noise_variance, "dtype": "complex128",
        "activity_generation": ("independent Bernoulli; active count not fixed" if cfg.active_count is None
                                else "uniform fixed-K support; marginal probability K/N; not independent Bernoulli"),
        "active_count_diagnostic": ("evaluate saved scores conditional on K=round(N*p); separate from all-trial results"
                                    if cfg.active_count is None else "not needed: all samples have fixed K"),
        "path_loss_db": "128.1 + 37.6 log10(distance_km)",
        "power_control": "p_n = pmax * min_device_gain / device_gain_n",
        "paired_cases": "same positions/labels; prefixes of common pilots, channels and noise",
        "solver": "known-gain box-constrained covariance MLE; zero initialization",
        "reproduction_choices_not_specified_by_paper": [
            "RNG seed, coordinate order, zero initialization, stopping tolerance and sweep cap",
            "dense threshold grid and pooled error-count aggregation",
            "minimum gain over sampled devices; 1 m distance guard",
            "paired realizations across cases; no original HT figure source code used"],
        "activity_sampling_caveat": "paper states activity ratio 0.1 and Bernoulli labels, but does not explicitly resolve fixed K versus independent draws",
        "audit": "same coordinate order with independent direct inversion; not a global-optimality proof"}
    (out_dir / "run_config.json").write_text(json.dumps(settings, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Paper covariance cases: {args.cases}; MC per case: {args.mc_times}", flush=True)
    print(f"Activity sampling: {settings['activity_generation']}; effective marginal p={cfg.activity_prob:g}", flush=True)
    print(f"Pilots CN(0,1); thermal noise={cfg.noise_variance:.12g} W; output: {out_dir.resolve()}", flush=True)
    summaries, curves, audits = [], [], []
    max_l, max_m = max(p.pilot_len for p in points), max(p.num_antennas for p in points)
    for point in points:
        summary, curve, checks = run_case(point, args, max_l, max_m,
                                         out_dir / f"L{point.pilot_len}_M{point.num_antennas}")
        summaries.append(summary)
        curves.append(curve)
        audits.extend(checks)
        write_csv(out_dir / "summary.csv", SUMMARY_FIELDS + SUMMARY_EXTRA_FIELDS, summaries)
        write_csv(out_dir / "pm_pf_curve.csv", CURVE_FIELDS, [r for curve in curves for r in curve])
        write_csv(out_dir / "solver_audit.csv", AUDIT_FIELDS, audits)
    if not args.no_plot:
        plot_reference(out_dir, summaries, curves)
    if cfg.active_count is None:
        save_count_diagnostic(out_dir, summaries, round(cfg.num_devices * cfg.activity_prob),
                              args.num_thresholds, not args.no_plot)
    print(f"Saved paper reproduction to: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
