from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.config import apply_cli_overrides, load_experiment_config, section_namespace
from network.data import ActivityDataGenerator as IndependentActivityDataGenerator
from network.data import SystemConfig as IndependentSystemConfig
from network.data_correlated import ActivityDataGenerator as CorrelatedActivityDataGenerator
from network.data_correlated import SystemConfig as CorrelatedSystemConfig
from network.metrics import pm_pf_curve
from network.model import build_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PM/PF curve from a trained checkpoint.")
    parser.add_argument("--config", type=str, default="config.json", help="Experiment config JSON path.")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_test_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_thresholds", type=int, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    exp_cfg = load_experiment_config(cli.config)
    args = section_namespace(exp_cfg, "network_evaluate")
    apply_cli_overrides(args, cli, ["ckpt", "device", "num_test_batches", "batch_size", "num_thresholds", "out_csv"])
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("model_config", ckpt["config"])
    data_mode = str(ckpt.get("config", {}).get("data_mode", cfg.get("data_mode", "independent"))).lower()
    if "group_beta_params" in ckpt["system_config"] or data_mode == "correlated":
        sys_cfg = CorrelatedSystemConfig(**ckpt["system_config"])
        data_gen = CorrelatedActivityDataGenerator(sys_cfg, device=device)
    else:
        sys_cfg = IndependentSystemConfig(**ckpt["system_config"])
        data_gen = IndependentActivityDataGenerator(sys_cfg, device=device)

    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for _ in range(args.num_test_batches):
            batch = data_gen.sample_batch(args.batch_size)
            _, probs = model(batch["x_b"], batch["x_y"])
            all_probs.append(probs)
            all_labels.append(batch["label"])

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    curve = pm_pf_curve(probs, labels, num_thresholds=args.num_thresholds)

    out = Path(args.out_csv)
    with out.open("w", encoding="utf-8") as f:
        f.write("threshold,PM,PF\n")
        for th, pm, pf in curve:
            f.write(f"{th:.6f},{pm:.6f},{pf:.6f}\n")

    print(f"Saved PM/PF curve to: {out.resolve()}")


if __name__ == "__main__":
    main()
