from __future__ import annotations

import argparse
from pathlib import Path

import torch

from htad.data import ActivityDataGenerator, SystemConfig
from htad.estimators import active_indices_from_probs
from htad.model import HeterogeneousTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare oracle active indices vs model-predicted active indices.")
    p.add_argument("--ckpt", type=str, default="checkpoints_lp16_gpu/last.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--topk", type=int, default=0, help="If >0, use top-k predicted users instead of threshold.")
    p.add_argument(
        "--out_txt",
        type=str,
        default="checkpoints_lp16_gpu/index_diff_report_10samples.txt",
        help="Output text report path.",
    )
    return p.parse_args()


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> HeterogeneousTransformer:
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
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model_from_ckpt(ckpt, device)
    data_gen = ActivityDataGenerator(SystemConfig(**ckpt["system_config"]), device=device)

    batch = data_gen.sample_batch(args.num_samples, return_raw=False)
    _, probs = model(batch["x_b"], batch["x_y"])  # [B, N]
    labels = batch["label"]  # [B, N]

    topk = args.topk if args.topk > 0 else None
    pred_idx_list = active_indices_from_probs(probs, threshold=args.threshold, topk=topk)
    true_idx_list = active_indices_from_probs(labels, threshold=0.5, topk=None)

    total_fn = 0
    total_fp = 0
    total_true = 0
    total_pred = 0
    lines: list[str] = []
    lines.append(f"Checkpoint: {ckpt_path.resolve()}")
    lines.append(f"Device: {device}")
    lines.append(f"Num samples: {args.num_samples}")
    lines.append(f"Threshold: {args.threshold}")
    lines.append(f"Top-k mode: {topk if topk is not None else 'off'}")
    lines.append("")

    for i in range(args.num_samples):
        true_set = set(true_idx_list[i].tolist())
        pred_set = set(pred_idx_list[i].tolist())
        miss = sorted(list(true_set - pred_set))  # FN
        false_alarm = sorted(list(pred_set - true_set))  # FP
        true_sorted = sorted(list(true_set))
        pred_sorted = sorted(list(pred_set))

        total_fn += len(miss)
        total_fp += len(false_alarm)
        total_true += len(true_sorted)
        total_pred += len(pred_sorted)

        lines.append(f"Sample {i}:")
        lines.append(f"  True idx      : {true_sorted}")
        lines.append(f"  Pred idx      : {pred_sorted}")
        lines.append(f"  Missed (FN)   : {miss}")
        lines.append(f"  False alarmFP : {false_alarm}")
        lines.append(
            f"  Counts (T/P/FN/FP): {len(true_sorted)} / {len(pred_sorted)} / {len(miss)} / {len(false_alarm)}"
        )
        lines.append("")

    fn_rate = total_fn / max(total_true, 1)
    fp_rate = total_fp / max(total_pred, 1)
    lines.append("Summary:")
    lines.append(f"  Total true actives: {total_true}")
    lines.append(f"  Total pred actives: {total_pred}")
    lines.append(f"  Total FN: {total_fn}")
    lines.append(f"  Total FP: {total_fp}")
    lines.append(f"  FN rate (FN/True): {fn_rate:.4f}")
    lines.append(f"  FP ratio (FP/Pred): {fp_rate:.4f}")

    report = "\n".join(lines)
    print(report)

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

