from __future__ import annotations

import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from .data import SystemConfig


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


DEFAULT_CONFIG: dict[str, Any] = {
    "network_train": {
        "system": {
            "num_devices": 100,
            "num_antennas": 32,
            "pilot_len": 8,
            "activity_prob": 0.1,
            "cell_radius_m": 500.0,
            "noise_power_dbm_hz": -169.0,
            "bandwidth_hz": 10e6,
            "pmax_dbm": 23.0,
            "noise_mode": "snr",
            "snr_db": 20.0,
            "activity_mode": "correlated",
            "use_correlation_feature": True,
            "correlation_activity_strength": 0.8,
        },
        "model": {
            "model_name": "base",
            "dim": 128,
            "num_layers": 5,
            "num_heads": 8,
            "head_dim": 32,
            "ff_dim": 512,
            "score_scale": 10.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "ctx_attn_dropout": 0.0,
            "norm_type": "batch",
        },
        "device": "auto",
        "save_dir": "checkpoint/checkpoints",
        "log_file": "",
        "seed": 42,
        "epochs": 100,
        "steps_per_epoch": 5000,
        "batch_size": 256,
        "lr": 1e-4,
        "amp": True,
        "amp_dtype": "bf16",
        "warmup_epochs": 0,
        "grad_clip": 0.0,
        "lr_decay_epochs": "90,97",
        "lr_decay_factor": 0.1,
        "eval_batches": 20,
        "eval_threshold": 0.5,
        "fixed_eval_set": True,
    },
    "network_evaluate": {
        "ckpt": "checkpoint/checkpoints/last.pt",
        "device": "auto",
        "num_test_batches": 80,
        "batch_size": 128,
        "num_thresholds": 41,
        "out_csv": "pm_pf_curve.csv",
    },
    "CE_methods_compare": {
        "ckpt": "checkpoint/checkpoints/last.pt",
        "device": "auto",
        "num_test_batches": 80,
        "batch_size": 128,
        "threshold": 0.5,
        "topk": 0,
        "methods": ["CAMP", "transformer+AMP", "oracle_AMP", "detected+LMMSE", "oracle_LMMSE"],
        "channel_var": 1.0,
        "reg_eps": 1e-18,
        "prior_mode": "unit",
        "camp_iters": 12,
        "camp_iters_fixed": 6,
        "camp_fading_mode": "unscaled_gain",
        "camp_lambda_floor": 1e-6,
        "camp_prob_calib": "sigmoid_center",
        "camp_prob_center": 0.5,
        "camp_prob_alpha": 12.0,
        "camp_damping": 0.7,
        "camp_fixed_lambda": 0.1,
        "out_txt": "CE_methods/detection_lmmse_report.txt",
    },
    "CE_methods_camp_genie_data": {
        "ckpt": "checkpoint/checkpoints/last.pt",
        "device": "auto",
        "mc_times": 100,
        "max_iter": 40,
        "seed": 1,
        "threshold": 0.5,
        "matrix": "s",
        "out_txt": "",
    },
    "network_compare_active_indices": {
        "ckpt": "checkpoint/checkpoints/last.pt",
        "device": "auto",
        "num_samples": 10,
        "threshold": 0.5,
        "topk": 0,
        "out_txt": "index_diff_report_10samples.txt",
    },
    "plot_plot_amp_nmse_vs_iter": {
        "ckpt": "checkpoint/checkpoints/last.pt",
        "device": "auto",
        "num_test_batches": 3,
        "batch_size": 32,
        "max_iters": 10,
        "camp_damping": 0.7,
        "camp_lambda_floor": 1e-6,
        "camp_fixed_lambda": 0.1,
        "camp_prob_calib": "none",
        "camp_prob_center": 0.5,
        "camp_prob_alpha": 12.0,
        "out_png": "amp_nmse_vs_iter.png",
        "out_txt": "amp_nmse_vs_iter.txt",
    },
}


def _normalize_legacy_sections(cfg: dict[str, Any]) -> dict[str, Any]:
    """Accept older config section names while the public config uses script names."""
    out = deepcopy(cfg)

    if any(k in out for k in ("system", "model", "train")):
        train_cfg = dict(out.get("network_train", {}))
        if "system" in out:
            train_cfg.setdefault("system", out["system"])
        if "model" in out:
            train_cfg.setdefault("model", out["model"])
        if "train" in out:
            _deep_update(train_cfg, out["train"])
        out["network_train"] = train_cfg

    aliases = {
        "evaluate": "network_evaluate",
        "detection": "CE_methods_compare",
        "compare": "network_compare_active_indices",
        "compare_active_indices": "network_compare_active_indices",
        "amp_plot": "plot_plot_amp_nmse_vs_iter",
    }
    for old, new in aliases.items():
        if old in out and new not in out:
            out[new] = out[old]
    return out


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_experiment_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            _deep_update(cfg, _normalize_legacy_sections(json.load(f)))
    return cfg


def resolve_device_name(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def section_namespace(cfg: dict[str, Any], section: str) -> Namespace:
    values = dict(cfg[section])
    if "device" in values:
        values["device"] = resolve_device_name(str(values["device"]))
    return Namespace(**values)


def system_config_from_experiment(cfg: dict[str, Any]) -> SystemConfig:
    return SystemConfig(**cfg["network_train"]["system"])


def model_config_from_experiment(cfg: dict[str, Any]) -> dict[str, Any]:
    train_cfg = cfg["network_train"]
    model_cfg = dict(train_cfg["model"])
    model_cfg["num_devices"] = train_cfg["system"]["num_devices"]
    model_cfg["pilot_len"] = train_cfg["system"]["pilot_len"]
    model_cfg.setdefault("use_correlation_feature", train_cfg["system"].get("use_correlation_feature", True))
    return {k: v for k, v in model_cfg.items() if v is not None}


def apply_cli_overrides(args: Namespace, cli: Namespace, keys: list[str]) -> Namespace:
    for key in keys:
        value = getattr(cli, key, None)
        if value is not None:
            if key == "device":
                value = resolve_device_name(str(value))
            setattr(args, key, value)
    return args
