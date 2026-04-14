from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from htad.data import ActivityDataGenerator, SystemConfig
from htad.losses import weighted_activity_loss
from htad.metrics import pm_pf_at_threshold
from htad.model import HeterogeneousTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heterogeneous transformer for device activity detection.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_devices", type=int, default=100)
    parser.add_argument("--num_antennas", type=int, default=32)
    parser.add_argument("--pilot_len", type=int, default=8)
    parser.add_argument("--activity_prob", type=float, default=0.1)
    parser.add_argument("--pmax_dbm", type=float, default=23.0)

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=32)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--score_scale", type=float, default=10.0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay_epochs", type=str, default="18")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1)

    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--eval_threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = SystemConfig(
        num_devices=args.num_devices,
        num_antennas=args.num_antennas,
        pilot_len=args.pilot_len,
        activity_prob=args.activity_prob,
        pmax_dbm=args.pmax_dbm,
    )
    data_gen = ActivityDataGenerator(cfg=cfg, device=device)

    model = HeterogeneousTransformer(
        num_devices=args.num_devices,
        pilot_len=args.pilot_len,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        ff_dim=args.ff_dim,
        score_scale=args.score_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    decay_set = {int(x.strip()) for x in args.lr_decay_epochs.split(",") if x.strip()}
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_pm = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for _ in pbar:
            batch = data_gen.sample_batch(args.batch_size)
            logits, probs = model(batch["x_b"], batch["x_y"])
            loss = weighted_activity_loss(
                logits=logits,
                targets=batch["label"],
                activity_prob=args.activity_prob,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            pbar.set_postfix(loss=f"{running_loss / (pbar.n + 1):.4f}")

        if epoch in decay_set:
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * args.lr_decay_factor

        model.eval()
        eval_probs = []
        eval_labels = []
        with torch.no_grad():
            for _ in range(args.eval_batches):
                batch = data_gen.sample_batch(args.batch_size)
                _, p = model(batch["x_b"], batch["x_y"])
                eval_probs.append(p)
                eval_labels.append(batch["label"])

        eval_probs_t = torch.cat(eval_probs, dim=0)
        eval_labels_t = torch.cat(eval_labels, dim=0)
        pm, pf = pm_pf_at_threshold(eval_probs_t, eval_labels_t, args.eval_threshold)
        avg_loss = running_loss / float(args.steps_per_epoch)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.6f} | PM@{args.eval_threshold:.2f}={pm:.6f} | "
            f"PF@{args.eval_threshold:.2f}={pf:.6f} | lr={lr_now:.2e}"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "config": vars(args),
            "system_config": cfg.__dict__,
            "epoch": epoch,
        }
        torch.save(ckpt, save_dir / "last.pt")
        if pm < best_pm:
            best_pm = pm
            torch.save(ckpt, save_dir / "best_pm.pt")

    print(f"Training done. Checkpoints saved in: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
