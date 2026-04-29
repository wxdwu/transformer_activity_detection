from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_log(log_path: Path) -> tuple[list[int], list[float]]:
    pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+loss=([0-9]*\.?[0-9]+)")
    epoch_to_loss: dict[int, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        epoch = int(m.group(1))
        loss = float(m.group(2))
        # If duplicated epochs exist in log, keep the latest one.
        epoch_to_loss[epoch] = loss

    if not epoch_to_loss:
        raise ValueError(f"No 'Epoch ... | loss=...' records found in: {log_path}")

    epochs = sorted(epoch_to_loss.keys())
    losses = [epoch_to_loss[e] for e in epochs]
    return epochs, losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training loss vs epoch from train log text.")
    parser.add_argument("--log", type=str, required=True, help="Path to text log containing 'Epoch ... | loss=...' lines.")
    parser.add_argument("--out", type=str, default="loss_curve.png", help="Output image file path.")
    parser.add_argument("--title", type=str, default="Training Loss vs Epoch")
    parser.add_argument("--label", type=str, default="run")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    epochs, losses = parse_log(log_path)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(epochs, losses, linewidth=2.0, label=args.label)
    plt.xlabel("Training epoch")
    plt.ylabel("Training loss")
    plt.title(args.title)
    plt.grid(True, which="both", linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi)
    print(f"Saved plot to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
