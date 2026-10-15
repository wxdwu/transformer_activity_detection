from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

if sys.platform == "darwin":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from covariance_based.detector import covariance_mle_activity
from covariance_based.nnls import NNLS_MAX_ITER, NNLS_TOLERANCE, covariance_nnls_activity
from covariance_based.run_monte_carlo import (
    MAX_SWEEPS,
    SEED,
    TOLERANCE,
    positive_plot_values,
    probability_log_axis,
    rate,
    validate_system,
    write_csv,
)
from network.data import ActivityDataGenerator, SystemConfig
from network.model import build_model_from_config
from experiments.paths import RESULT_ROOT


CHECKPOINT_ROOT = RESULT_ROOT / "loss_vs_epoch"
OUTPUT_DIR = RESULT_ROOT / "single_condition" / "pm_pf_comparison"
MC_TIMES = 5000
METHODS = ("GI-HT", "HT", "Covariance")
NNLS_METHOD = "Box-NNLS"
GEOMETRY_FLAGS = (
    "use_correlation_attention_bias",
    "use_correlation_feature_mixer",
    "use_correlation_logit_refinement",
)
CURVE_FIELDS = ["method", "threshold", "PM", "PF", "active", "inactive", "missed", "false_alarms"]
SUMMARY_FIELDS = [
    "method", "checkpoint_epoch", "mc_times", "threshold", "PM", "PF",
    "error_probability", "active", "inactive", "missed", "false_alarms",
    "mean_activity_ratio", "converged_fraction", "mean_sweeps", "mean_iterations",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GI-HT, HT and covariance baselines on identical trials.")
    parser.add_argument("--giht-ckpt", type=Path, default=CHECKPOINT_ROOT / "addcorr" / "last.pt")
    parser.add_argument("--ht-ckpt", type=Path, default=CHECKPOINT_ROOT / "noaddcorr" / "last.pt")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--mc-times", type=int, default=MC_TIMES)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Neural inference batch size; default inherits the checkpoint training batch size.")
    parser.add_argument("--num-thresholds", type=int, default=1001)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-sweeps", type=int, default=MAX_SWEEPS)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--include-nnls", action="store_true", help="Add box-constrained covariance NNLS to the comparison.")
    parser.add_argument("--nnls-max-iter", type=int, default=NNLS_MAX_ITER)
    parser.add_argument("--nnls-tolerance", type=float, default=NNLS_TOLERANCE)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device such as cuda:0.")
    parser.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args(argv)
    if min(args.mc_times, args.max_sweeps) < 1 or args.num_thresholds < 2 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("Counts must be positive and num-thresholds must be at least 2.")
    if args.log_every < 0 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("log-every must be nonnegative and tolerance must be finite and positive.")
    if args.nnls_max_iter < 1 or not math.isfinite(args.nnls_tolerance) or args.nnls_tolerance <= 0:
        parser.error("nnls-max-iter must be positive and nnls-tolerance must be finite and positive.")
    if args.out_dir is None:
        args.out_dir = OUTPUT_DIR.with_name(OUTPUT_DIR.name + "_with_nnls") if args.include_nnls else OUTPUT_DIR
    return args


def load_detector(path: Path, method: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Specify a complete trained last.pt.")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (RuntimeError, EOFError) as error:
        raise RuntimeError(f"Cannot load {path}; copy the complete checkpoint from the training server. {error}") from error
    cfg = checkpoint["model_config"]
    expected = method == "GI-HT"
    if any(bool(cfg.get(flag, False)) != expected for flag in GEOMETRY_FLAGS):
        raise ValueError(f"{method} requires all geometry branches {'enabled' if expected else 'disabled'}.")
    if cfg.get("signal_token_mode") != "covariance_rows":
        raise ValueError(f"{method} must use covariance_rows for this comparison.")
    system = SystemConfig(**checkpoint["system_config"])
    validate_system(system)
    if (cfg["num_devices"], cfg["pilot_len"]) != (system.num_devices, system.pilot_len):
        raise ValueError(f"{method} model and system dimensions disagree.")
    model = build_model_from_config(cfg)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    metadata = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "epoch": int(checkpoint["epoch"]),
        "training_batch_size": int(checkpoint["config"]["batch_size"]),
        "training_config": checkpoint["config"],
        "system_config": asdict(system),
        "model_config": cfg,
    }
    return model, metadata


def validate_pair(giht: dict, ht: dict) -> SystemConfig:
    if giht["system_config"] != ht["system_config"]:
        raise ValueError("GI-HT and HT were trained under different system configurations.")
    if giht["epoch"] != ht["epoch"]:
        raise ValueError("Use checkpoints from the same training epoch.")
    if giht["training_batch_size"] != ht["training_batch_size"]:
        raise ValueError("Use checkpoints with the same training batch size.")
    first = {k: v for k, v in giht["model_config"].items() if k not in GEOMETRY_FLAGS}
    second = {k: v for k, v in ht["model_config"].items() if k not in GEOMETRY_FLAGS}
    if first != second:
        raise ValueError("Model architectures/settings differ beyond the geometry switches.")
    return SystemConfig(**giht["system_config"])


def curve_from_scores(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, method: str) -> list[dict]:
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Scores and labels must have the same shape and finite scores.")
    if np.any((scores < 0) | (scores > 1)) or np.any((labels != 0) & (labels != 1)):
        raise ValueError("Scores must be in [0,1] and labels must be binary.")
    active_mask = labels.astype(bool)
    positives = np.sort(scores[active_mask].astype(np.float64))
    negatives = np.sort(scores[~active_mask].astype(np.float64))
    # Strict score > threshold: scores equal to the threshold are inactive.
    missed = np.searchsorted(positives, thresholds, side="right")
    false_alarms = len(negatives) - np.searchsorted(negatives, thresholds, side="right")
    return [
        {"method": method, "threshold": float(th), "PM": rate(int(fn), len(positives)),
         "PF": rate(int(fp), len(negatives)), "active": len(positives), "inactive": len(negatives),
         "missed": int(fn), "false_alarms": int(fp)}
        for th, fn, fp in zip(thresholds, missed, false_alarms)
    ]


def plot_comparison(out_dir: Path, curves: dict[str, list[dict]], *, stem: str = "pm_pf_comparison") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"GI-HT": ("#0072B2", "-", "o"), "HT": ("#D55E00", "--", "s"),
              "Covariance": ("#009E73", "-.", "^"), NNLS_METHOD: ("#CC79A7", ":", "D")}
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for method, rows in curves.items():
        color, linestyle, marker = styles[method]
        positive = [r for r in rows if np.isfinite(r["PM"]) and np.isfinite(r["PF"]) and r["PM"] > 0 and r["PF"] > 0]
        spacing = max(1, len(positive) // 10)
        ax.plot(positive_plot_values([r["PM"] for r in rows]),
                positive_plot_values([r["PF"] for r in rows]), color=color, linestyle=linestyle,
                linewidth=1.7, label=f"{method} (MLE)" if method == "Covariance" else method)
        selected = positive[::spacing]
        ax.plot([r["PM"] for r in selected], [r["PF"] for r in selected], linestyle="none",
                marker=marker, markersize=4, markerfacecolor="none", color=color)
    all_rows = [r for rows in curves.values() for r in rows]
    ax.set(xlabel=r"Missed-detection probability $P_M$", ylabel=r"False-alarm probability $P_F$")
    probability_log_axis(ax, "x", [r["PM"] for r in all_rows])
    probability_log_axis(ax, "y", [r["PF"] for r in all_rows])
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    ax.legend(loc="best")
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)


@torch.inference_mode()
def evaluate_shared_trials(models: dict[str, torch.nn.Module], cfg: SystemConfig,
                           args: argparse.Namespace, device: torch.device, *,
                           methods: tuple[str, ...] | None = None) -> tuple[dict, np.ndarray, list[dict], str]:
    methods = methods if methods is not None else ((*METHODS, NNLS_METHOD) if args.include_nnls else METHODS)
    if not methods or len(set(methods)) != len(methods) or set(methods) - {*METHODS, NNLS_METHOD}:
        raise ValueError("Choose distinct supported comparison methods.")
    if set(models) != (set(methods) & {"GI-HT", "HT"}):
        raise ValueError("Loaded neural models must match the selected methods.")
    torch.manual_seed(args.seed)
    coordinate_rng = np.random.default_rng(args.seed + 1)
    generator = ActivityDataGenerator(cfg, torch.device("cpu"))
    scores = {method: np.empty((args.mc_times, cfg.num_devices), dtype=np.float64) for method in methods}
    labels = np.empty((args.mc_times, cfg.num_devices), dtype=np.uint8)
    diagnostics: list[dict] = []
    digest = hashlib.sha256()
    pending: list[dict] = []
    start, next_log = perf_counter(), args.log_every
    for index in range(args.mc_times):
        # Single-trial generation is independent of neural batching and matches run_monte_carlo.py.
        batch = generator.sample_batch(1, return_raw=True)
        labels[index] = batch["label"][0].numpy().astype(np.uint8)
        for key in ("b", "y", "label", "corr_matrix", "noise_var"):
            digest.update(batch[key].numpy().tobytes())
        b = batch["b"][0].numpy().astype(np.complex128)
        y = batch["y"][0].numpy().astype(np.complex128)
        noise = float(batch["noise_var"][0])
        covariance = y @ y.conj().T / cfg.num_antennas
        diagnostics.append({"trial": index + 1, "active": int(labels[index].sum()),
                            "receiver_noise_variance": noise})
        if "Covariance" in methods:
            result = covariance_mle_activity(b, covariance, noise,
                                             max_sweeps=args.max_sweeps, tolerance=args.tolerance, rng=coordinate_rng)
            scores["Covariance"][index] = result.activities
            diagnostics[-1].update(sweeps=result.sweeps, converged=int(result.converged),
                                   max_coordinate_step=result.max_coordinate_step)
        if NNLS_METHOD in methods:
            nnls = covariance_nnls_activity(b, covariance, noise, max_iter=args.nnls_max_iter,
                                            tolerance=args.nnls_tolerance)
            scores[NNLS_METHOD][index] = nnls.activities
            diagnostics[-1].update(nnls_iterations=nnls.iterations, nnls_converged=int(nnls.converged),
                                   nnls_optimality=nnls.optimality, nnls_status=nnls.status,
                                   nnls_projected_gradient_norm=nnls.projected_gradient_norm,
                                   nnls_scaled_residual_squared=nnls.scaled_residual_squared,
                                   nnls_power_scale=nnls.power_scale)
        if models:
            pending.append({key: batch[key] for key in ("x_b", "x_y", "corr_matrix")})
        if not models or len(pending) == args.batch_size or index + 1 == args.mc_times:
            if models:
                features = {key: torch.cat([row[key] for row in pending]).to(device) for key in pending[0]}
                lower = index + 1 - len(pending)
                for method, model in models.items():
                    geometry = features["corr_matrix"] if method == "GI-HT" else None
                    _, probabilities = model(features["x_b"], features["x_y"], geometry)
                    scores[method][lower:index + 1] = probabilities.cpu().numpy()
                pending.clear()
            if args.log_every and (index + 1 >= next_log or index + 1 == args.mc_times):
                print(f"Shared trials {index + 1}/{args.mc_times} | elapsed={perf_counter() - start:.1f}s", flush=True)
                next_log = index + 1 + args.log_every
    return scores, labels, diagnostics, digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    giht, giht_meta = load_detector(args.giht_ckpt, "GI-HT", device)
    ht, ht_meta = load_detector(args.ht_ckpt, "HT", device)
    cfg = validate_pair(giht_meta, ht_meta)
    if args.batch_size is None:
        args.batch_size = giht_meta["training_batch_size"]
    print(f"Epoch {giht_meta['epoch']} checkpoints | device={device} | shared trials={args.mc_times} | inference batch={args.batch_size}", flush=True)
    print("TokenBatchNorm uses current-batch statistics even in eval mode; both networks use identical groups.", flush=True)
    print(f"System: {json.dumps(asdict(cfg), sort_keys=True)}", flush=True)
    scores, labels, diagnostics, digest = evaluate_shared_trials({"GI-HT": giht, "HT": ht}, cfg, args, device)
    thresholds = np.linspace(0.0, 1.0, args.num_thresholds)
    curves = {method: curve_from_scores(scores[method], labels, thresholds, method) for method in scores}
    summaries = []
    for method in scores:
        row = curve_from_scores(scores[method], labels, np.array([0.5]), method)[0]
        converged_fraction = mean_sweeps = mean_iterations = ""
        if method == "Covariance":
            converged_fraction = sum(r["converged"] for r in diagnostics) / args.mc_times
            mean_sweeps = sum(r["sweeps"] for r in diagnostics) / args.mc_times
        elif method == NNLS_METHOD:
            converged_fraction = sum(r["nnls_converged"] for r in diagnostics) / args.mc_times
            mean_iterations = sum(r["nnls_iterations"] for r in diagnostics) / args.mc_times
        row.update({"checkpoint_epoch": giht_meta["epoch"] if method in ("GI-HT", "HT") else "",
                    "mc_times": args.mc_times,
                    "error_probability": (row["missed"] + row["false_alarms"]) / labels.size,
                    "mean_activity_ratio": float(labels.mean()),
                    "converged_fraction": converged_fraction, "mean_sweeps": mean_sweeps,
                    "mean_iterations": mean_iterations})
        summaries.append(row)
        print(f"{method}: PM@0.5={row['PM']:.6f}, PF@0.5={row['PF']:.6f}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "pm_pf_curves.csv", CURVE_FIELDS, [r for rows in curves.values() for r in rows])
    write_csv(args.out_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    mle_diagnostics = [{key: value for key, value in row.items() if not key.startswith("nnls_")} for row in diagnostics]
    write_csv(args.out_dir / "covariance_trials.csv", list(mle_diagnostics[0]), mle_diagnostics)
    if args.include_nnls:
        nnls_diagnostics = [{"trial": row["trial"], "active": row["active"],
                             "receiver_noise_variance": row["receiver_noise_variance"],
                             **{key.removeprefix("nnls_"): value for key, value in row.items() if key.startswith("nnls_")}}
                            for row in diagnostics]
        write_csv(args.out_dir / "nnls_trials.csv", list(nnls_diagnostics[0]), nnls_diagnostics)
    np.savez_compressed(args.out_dir / "activity_scores.npz", labels=labels,
                        giht_scores=scores["GI-HT"], ht_scores=scores["HT"], covariance_scores=scores["Covariance"],
                        **({"nnls_scores": scores[NNLS_METHOD]} if args.include_nnls else {}))
    settings = {**vars(args), "system_config": asdict(cfg), "checkpoints": {"GI-HT": giht_meta, "HT": ht_meta},
                "device_used": str(device), "inference_precision": "float32, no autocast",
                "generator_device": "cpu", "generator_batch_size": 1, "dataset_sha256": digest,
                "last_inference_batch_size": (args.mc_times - 1) % args.batch_size + 1,
                "normalization_note": "TokenBatchNorm uses current-batch statistics, including the final partial batch.",
                "noise_variance_source": "known_generated", "decision_rule": "score > threshold",
                "threshold_scan": [0.0, 1.0], "coordinate_rng_seed": args.seed + 1}
    if args.include_nnls:
        from importlib.metadata import version
        settings["nnls_solver"] = {"method": "scipy.optimize.lsq_linear", "algorithm": "bvls",
                                   "scipy_version": version("scipy"), "bounds": [0.0, 1.0],
                                   "max_iter": args.nnls_max_iter, "tolerance": args.nnls_tolerance,
                                   "criterion": "squared Frobenius covariance residual, common power scaling",
                                   "convergence_check": "SciPy success and box projected-gradient infinity norm <= tolerance",
                                   "label": NNLS_METHOD}
    (args.out_dir / "run_config.json").write_text(json.dumps(settings, indent=2, default=str) + "\n", encoding="utf-8")
    plot_comparison(args.out_dir, curves)
    print(f"Saved comparison: {args.out_dir.resolve()}", flush=True)
    print("Logarithmic plots omit zero/undefined probabilities; CSV values are unchanged.")


if __name__ == "__main__":
    main()
