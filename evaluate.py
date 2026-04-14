from __future__ import annotations

import argparse
from pathlib import Path

import torch

from htad.data import ActivityDataGenerator, SystemConfig
from htad.metrics import pm_pf_curve
from htad.model import HeterogeneousTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PM/PF curve from a trained checkpoint.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_test_batches", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_thresholds", type=int, default=41)
    parser.add_argument("--out_csv", type=str, default="pm_pf_curve.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    sys_cfg = SystemConfig(**ckpt["system_config"])
    cfg = ckpt["config"]

    model = HeterogeneousTransformer(
        num_devices=cfg["num_devices"],
        pilot_len=cfg["pilot_len"],
        dim=cfg["dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        head_dim=cfg["head_dim"],
        ff_dim=cfg["ff_dim"],
        score_scale=cfg["score_scale"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data_gen = ActivityDataGenerator(sys_cfg, device=device)
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
