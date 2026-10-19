from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import copy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

if sys.platform == "darwin":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import torch

from covariance_based.nnls import NNLS_MAX_ITER, NNLS_TOLERANCE
from covariance_based.run_monte_carlo import MAX_SWEEPS, MC_TIMES, TOLERANCE, validate_system
from experiments.paths import RESULT_ROOT
from experiments.pe_metrics import METHODS, refresh_sweep, save_method_result
from network.run_pm_pf_comparison import (
    GEOMETRY_FLAGS, evaluate_shared_trials, load_detector, validate_pair,
)
from network.run_varing_SNR import format_snr_value, parse_snr_values
from network.run_varing_number_of_users import parse_num_users
from network.run_varying_l import parse_pilot_lens
from network.train import build_args, build_model_config, build_system_config, resolve_device, run_training


SWEEPS = {"L": ("pilot_len", parse_pilot_lens("")),
          "N": ("num_devices", parse_num_users("")),
          "SNR": ("snr_db", parse_snr_values(""))}
MODE_NAMES = {"GI-HT": "addcorr", "HT": "noaddcorr"}


def parse_args(argv: list[str] | None, default_methods: tuple[str, ...], default_stage: str, *,
               fixed_sweep: str | None = None) -> argparse.Namespace:
    if fixed_sweep is not None and fixed_sweep not in SWEEPS:
        raise ValueError(f"Unknown fixed sweep: {fixed_sweep}")
    allowed_sweeps = [fixed_sweep] if fixed_sweep is not None else list(SWEEPS)
    parser = argparse.ArgumentParser(description="Minimum PM+PF experiments with per-point threshold selection.")
    parser.add_argument("--sweeps", nargs="+", choices=allowed_sweeps, default=allowed_sweeps)
    parser.add_argument("--values", nargs="+", type=float, help="Custom values, allowed with a single sweep.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(default_methods))
    parser.add_argument("--stage", choices=["train-evaluate", "evaluate", "plot"], default=default_stage)
    parser.add_argument("--allow-different-observations", action="store_true",
                        help="Plot same-scene results without requiring identical observation hashes; plot stage only.")
    parser.add_argument("--out-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, help="Alternate result root containing trained checkpoints; evaluate stage only.")
    parser.add_argument("--mc-times", type=int, default=MC_TIMES)
    parser.add_argument("--seed", type=int, default=build_args().seed)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=build_args().epochs)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int, help="Training batch size; default inherits train.py.")
    parser.add_argument("--eval-batches", type=int, help="Training-time evaluation batches; does not set mc-times.")
    parser.add_argument("--inference-batch-size", type=int, help="Default inherits the loaded checkpoint training batch size.")
    parser.add_argument("--retrain", action="store_true", help="Replace selected training checkpoints even if already complete.")
    parser.add_argument("--max-sweeps", type=int, default=MAX_SWEEPS)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--nnls-max-iter", type=int, default=NNLS_MAX_ITER)
    parser.add_argument("--nnls-tolerance", type=float, default=NNLS_TOLERANCE)
    parser.add_argument("--pe-scale", choices=["linear", "log"], default="log")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(argv)
    if not args.methods or len(set(args.methods)) != len(args.methods) or len(set(args.sweeps)) != len(args.sweeps):
        parser.error("Choose distinct methods and sweeps.")
    if args.values is not None and (len(args.sweeps) != 1 or not args.values):
        parser.error("Custom values require exactly one sweep.")
    if args.checkpoint_root is not None and args.stage != "evaluate":
        parser.error("checkpoint-root requires --stage evaluate.")
    if args.allow_different_observations and args.stage != "plot":
        parser.error("--allow-different-observations requires --stage plot.")
    for name in ("mc_times", "epochs", "steps_per_epoch", "batch_size", "eval_batches",
                 "inference_batch_size", "max_sweeps", "nnls_max_iter"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"{name} must be positive.")
    if args.log_every < 0:
        parser.error("log-every must be nonnegative.")
    for name in ("tolerance", "nnls_tolerance"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive.")
    for sweep in args.sweeps:
        sweep_points(sweep, args.values)
    return args


def sweep_points(sweep: str, values: list[float] | None = None) -> list[tuple[float, str]]:
    chosen = SWEEPS[sweep][1] if values is None else values
    if len(set(chosen)) != len(chosen) or any(not math.isfinite(v) for v in chosen):
        raise ValueError("Sweep values must be distinct and finite.")
    if sweep != "SNR" and any(v < 1 or int(v) != v for v in chosen):
        raise ValueError("L and N must be positive integers.")
    points = []
    for raw in sorted(chosen):
        value = float(raw) if sweep == "SNR" else int(raw)
        label = (f"L{value:02d}" if sweep == "L" else f"N{value:03d}" if sweep == "N"
                 else "SNR" + format_snr_value(value))
        points.append((value, label))
    return points


def training_args(cli: argparse.Namespace, sweep: str, value: float, out: Path, method: str) -> SimpleNamespace:
    args = copy(build_args())
    setattr(args, SWEEPS[sweep][0], value)
    args.seed, args.epochs, args.device = cli.seed, cli.epochs, resolve_device(cli.device)
    for name in ("steps_per_epoch", "batch_size", "eval_batches"):
        if getattr(cli, name) is not None:
            setattr(args, name, getattr(cli, name))
    for name in GEOMETRY_FLAGS:
        setattr(args, name, method == "GI-HT")
    args.save_dir = out
    args.log_file = out / "train.log"
    return args


def prepare_models(cli: argparse.Namespace, sweep: str, value: float, label: str,
                   directory: Path, cfg, device: torch.device) -> tuple[dict, dict, int]:
    models, metadata = {}, {}
    for method in cli.methods:
        if method not in MODE_NAMES:
            continue
        source = cli.checkpoint_root / f"pe_vs_{sweep}" if cli.checkpoint_root is not None else directory
        out = source / "checkpoints" / MODE_NAMES[method] / label
        path = out / "last.pt"
        args = training_args(cli, sweep, value, out, method)
        if cli.stage == "train-evaluate" and (cli.retrain or not path.exists()):
            if cli.retrain and args.log_file.exists():
                args.log_file.unlink()
            run_training(args)
        model, meta = load_detector(path, method, device)
        if meta["system_config"] != asdict(cfg) or meta["model_config"] != build_model_config(args):
            raise ValueError(f"Checkpoint does not match the requested scene/model: {path}")
        if meta["epoch"] != cli.epochs:
            raise ValueError(f"Checkpoint epoch is {meta['epoch']}, expected {cli.epochs}: {path}. "
                             "Use a new output root or --retrain for a fresh training run.")
        if cli.stage == "train-evaluate":
            for name in ("steps_per_epoch", "batch_size", "seed", "lr", "warmup_epochs", "grad_clip",
                         "lr_decay_epochs", "lr_decay_factor", "amp", "amp_dtype", "eval_batches", "fixed_eval_set"):
                if meta["training_config"].get(name) != getattr(args, name):
                    raise ValueError(f"Saved training setting {name} differs: {path}. Use --retrain or --stage evaluate.")
        models[method], metadata[method] = model, meta
    if len(models) == 2:
        validate_pair(metadata["GI-HT"], metadata["HT"])
    batch_size = cli.inference_batch_size
    if batch_size is None:
        batch_size = next(iter(metadata.values()))["training_batch_size"] if metadata else build_args().batch_size
    return models, metadata, batch_size


def check_existing_point(point: Path, cfg, cli: argparse.Namespace, digest: str | None = None) -> None:
    for method in METHODS:
        path = point / method / "metrics.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record["system_config"] != asdict(cfg) or record["mc_times"] != cli.mc_times
                or record["evaluation_seed"] != cli.seed):
            raise ValueError(f"Existing point uses a different scene/count/seed: {path}. Use a different --out-root.")
        if digest is not None and record["dataset_sha256"] != digest:
            raise ValueError(f"Generated observations differ from existing results: {path}")


def method_trials(diagnostics: list[dict], method: str) -> list[dict]:
    common = ("trial", "active", "receiver_noise_variance")
    if method == "Covariance":
        return [{k: row[k] for k in (*common, "sweeps", "converged", "max_coordinate_step")} for row in diagnostics]
    if method == "Box-NNLS":
        return [{**{k: row[k] for k in common},
                 **{k.removeprefix("nnls_"): v for k, v in row.items() if k.startswith("nnls_")}}
                for row in diagnostics]
    return [{k: row[k] for k in common} for row in diagnostics]


def main(argv: list[str] | None = None, *, default_methods: tuple[str, ...] = ("Covariance", "Box-NNLS"),
         default_stage: str = "evaluate", fixed_sweep: str | None = None) -> None:
    cli = parse_args(argv, default_methods, default_stage, fixed_sweep=fixed_sweep)
    torch.set_num_threads(1)
    base = build_system_config(build_args())
    validate_system(base)
    for sweep in cli.sweeps:
        field, _ = SWEEPS[sweep]
        points = sweep_points(sweep, cli.values)
        directory = cli.out_root / f"pe_vs_{sweep}"
        if cli.stage == "plot":
            refresh_sweep(directory, sweep, field, points, cli.pe_scale,
                          allow_different_observations=cli.allow_different_observations)
            continue
        if sweep == "SNR" and base.noise_mode == "thermal":
            raise ValueError("SNR experiments require SNR-dependent noise.")
        for value, label in points:
            cfg = replace(base, **{field: value})
            validate_system(cfg)
            point = directory / "points" / label
            check_existing_point(point, cfg, cli)
            device = torch.device(resolve_device(cli.device) if any(m in MODE_NAMES for m in cli.methods) else "cpu")
            models, metadata, batch_size = prepare_models(cli, sweep, value, label, directory, cfg, device)
            settings = SimpleNamespace(seed=cli.seed, mc_times=cli.mc_times, batch_size=batch_size,
                                       max_sweeps=cli.max_sweeps, tolerance=cli.tolerance,
                                       include_nnls="Box-NNLS" in cli.methods, nnls_max_iter=cli.nnls_max_iter,
                                       nnls_tolerance=cli.nnls_tolerance, log_every=cli.log_every)
            print(f"\nEvaluate {sweep}={value}: methods={cli.methods}, trials={cli.mc_times}, "
                  f"inference_batch={batch_size}, device={device}", flush=True)
            scores, labels, diagnostics, digest = evaluate_shared_trials(models, cfg, settings, device,
                                                                        methods=tuple(cli.methods))
            check_existing_point(point, cfg, cli, digest)
            for method in cli.methods:
                solver = ({"max_sweeps": cli.max_sweeps, "coordinate_tolerance": cli.tolerance}
                          if method == "Covariance" else
                          {"max_iter": cli.nnls_max_iter, "tolerance": cli.nnls_tolerance, "bounds": [0, 1]}
                          if method == "Box-NNLS" else {})
                if method == "Box-NNLS":
                    import scipy
                    solver.update(algorithm="scipy.optimize.lsq_linear/bvls", scipy_version=scipy.__version__)
                provenance = {"sweep": sweep, "value": value, "system_config": asdict(cfg),
                              "mc_times": cli.mc_times, "evaluation_seed": cli.seed, "dataset_sha256": digest,
                              "generator": "CPU float32, one realization per call",
                              "device_used": str(device) if method in models else "cpu",
                              "inference_precision": "float32, no autocast" if method in models else "complex128 solver",
                              "inference_batch_size": batch_size if method in models else None,
                              "last_inference_batch_size": ((cli.mc_times - 1) % batch_size + 1) if method in models else None,
                              "checkpoint_epoch": metadata[method]["epoch"] if method in models else None,
                              "checkpoint": metadata.get(method), "solver": solver,
                              "noise_variance_source": "known_generated" if method not in models else "not a network input",
                              "numpy_version": np.__version__, "torch_version": str(torch.__version__),
                              "activity_counts": {"min": int(labels.sum(axis=1).min()), "max": int(labels.sum(axis=1).max()),
                                                  "mean": float(labels.sum(axis=1).mean()),
                                                  "zero_active_trials": int(np.sum(labels.sum(axis=1) == 0))},
                              "batch_normalization": "current-batch statistics, including the last partial batch"}
                record = save_method_result(point / method, method, scores[method], labels, provenance,
                                            method_trials(diagnostics, method))
                print(f"{method}: min Pe={record['min_pe']:.8g}, threshold={record['optimal_threshold']:.8g}, "
                      f"PM={record['PM']:.8g}, PF={record['PF']:.8g}", flush=True)
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()
        refresh_sweep(directory, sweep, field, points, cli.pe_scale)
        print(f"Saved Pe and per-point PM-PF results: {directory.resolve()}", flush=True)


if __name__ == "__main__":
    main()
