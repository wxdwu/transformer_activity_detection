from __future__ import annotations

import argparse
import csv
import sys
from copy import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.train import build_args, resolve_device, run_training


def parse_modes(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return ["addcorr", "noaddcorr"]
    modes = [value.strip() for value in raw.split(",") if value.strip()]
    allowed = {"addcorr", "noaddcorr"}
    unknown = [mode for mode in modes if mode not in allowed]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}. Allowed modes: {sorted(allowed)}")
    return modes


def write_history_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "signal_token_mode",
        "pilot_len",
        "epoch",
        "loss",
        "pm",
        "pf",
        "p_mean",
        "skip",
        "lr",
        "save_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def plot_loss_vs_epoch(
    path: Path,
    history_by_mode: dict[str, list[dict[str, float | int | str]]],
    pilot_len: int,
    dpi: int,
) -> Path | None:
    if not any(history_by_mode.values()):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        svg_path = path.with_suffix(".svg")
        write_loss_vs_epoch_svg(svg_path, history_by_mode, pilot_len)
        print(f"matplotlib is unavailable; saved SVG plot to: {svg_path.resolve()}")
        return svg_path

    max_epoch = max(
        int(row["epoch"])
        for history in history_by_mode.values()
        for row in history
    )
    plt.figure(figsize=(7.5, 4.8))
    for mode, history in history_by_mode.items():
        if not history:
            continue
        epochs = [int(row["epoch"]) for row in history]
        losses = [float(row["loss"]) for row in history]
        plt.plot(epochs, losses, linewidth=2.0, label=mode)
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title(f"Training Loss vs Epoch (L={pilot_len})")
    plt.xlim(1, max_epoch)
    if max_epoch == 1:
        plt.xticks([1])
    elif max_epoch <= 10:
        plt.xticks(list(range(1, max_epoch + 1)))
    else:
        ticks = [1, *range(10, max_epoch + 1, 10)]
        if ticks[-1] != max_epoch:
            ticks.append(max_epoch)
        plt.xticks(ticks)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    return path


def write_loss_vs_epoch_svg(
    path: Path,
    history_by_mode: dict[str, list[dict[str, float | int | str]]],
    pilot_len: int,
) -> None:
    width, height = 960, 600
    left, right, top, bottom = 95, 35, 50, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_rows = [row for history in history_by_mode.values() for row in history]
    if not all_rows:
        return

    x_min = 1
    x_max = max(int(row["epoch"]) for row in all_rows)
    losses = [float(row["loss"]) for row in all_rows]
    y_min, y_max = min(losses), max(losses)
    if y_min == y_max:
        padding = max(abs(y_min) * 0.1, 1e-3)
    else:
        padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding

    def scale_x(epoch: int) -> float:
        if x_max == x_min:
            return left + plot_width / 2.0
        return left + (epoch - x_min) / (x_max - x_min) * plot_width

    def scale_y(loss: float) -> float:
        return top + (y_max - loss) / (y_max - y_min) * plot_height

    if x_max <= 10:
        x_ticks = list(range(1, x_max + 1))
    else:
        x_ticks = [1, *range(10, x_max + 1, 10)]
        if x_ticks[-1] != x_max:
            x_ticks.append(x_max)
    y_ticks = [y_min + index * (y_max - y_min) / 5.0 for index in range(6)]
    colors = {"addcorr": "#1f77b4", "noaddcorr": "#d62728"}

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="600" viewBox="0 0 960 600">',
        '<rect width="960" height="600" fill="white"/>',
        f'<text x="480" y="30" text-anchor="middle" font-family="Arial" font-size="20">Training Loss vs Epoch (L={pilot_len})</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222" stroke-width="1.5"/>',
    ]
    for tick in x_ticks:
        x_position = scale_x(tick)
        lines.append(
            f'<line x1="{x_position:.2f}" y1="{top}" x2="{x_position:.2f}" '
            f'y2="{height-bottom}" stroke="#e6e6e6" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x_position:.2f}" y="{height-bottom+25}" text-anchor="middle" '
            f'font-family="Arial" font-size="13">{tick}</text>'
        )
    for tick in y_ticks:
        y_position = scale_y(tick)
        lines.append(
            f'<line x1="{left}" y1="{y_position:.2f}" x2="{width-right}" '
            f'y2="{y_position:.2f}" stroke="#e6e6e6" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left-10}" y="{y_position+4:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="13">{tick:.4f}</text>'
        )
    lines.extend(
        [
            '<text x="480" y="570" text-anchor="middle" font-family="Arial" font-size="16">Epoch</text>',
            '<text x="24" y="300" text-anchor="middle" font-family="Arial" font-size="16" transform="rotate(-90 24 300)">loss</text>',
        ]
    )

    legend_x, legend_y = width - right - 170, top + 20
    for offset, (mode, history) in enumerate(history_by_mode.items()):
        if not history:
            continue
        color = colors.get(mode, "#444")
        points = " ".join(
            f'{scale_x(int(row["epoch"])):.2f},{scale_y(float(row["loss"])):.2f}'
            for row in history
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        legend_line_y = legend_y + offset * 24
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_line_y}" x2="{legend_x+28}" '
            f'y2="{legend_line_y}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{legend_x+36}" y="{legend_line_y+4}" font-family="Arial" '
            f'font-size="14">{mode}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train addcorr/noaddcorr models and plot training loss for every epoch."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "checkpoint" / "loss_vs_epoch_260720_covrows",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.py EPOCHS. By default, inherit the current train.py value.",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--modes",
        type=str,
        default="",
        help="Comma-separated modes. Default: addcorr,noaddcorr.",
    )
    parser.add_argument(
        "--mode",
        choices=["addcorr", "noaddcorr"],
        default=None,
        help="Single-mode shortcut for debugging.",
    )
    parser.add_argument(
        "--append-log",
        action="store_true",
        help="Append to existing per-mode train.log files.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    modes = parse_modes(cli.mode if cli.mode is not None else cli.modes)
    all_rows: list[dict[str, float | int | str]] = []
    history_by_mode: dict[str, list[dict[str, float | int | str]]] = {
        mode: [] for mode in modes
    }
    csv_path = cli.out_dir / "loss_by_epoch.csv"
    plot_path = cli.out_dir / "loss_vs_epoch.png"
    last_plot_path: Path | None = None
    pilot_len: int | None = None

    for mode in modes:
        args = copy(build_args())
        if cli.epochs is not None:
            args.epochs = int(cli.epochs)
        if cli.steps_per_epoch is not None:
            args.steps_per_epoch = int(cli.steps_per_epoch)
        if cli.batch_size is not None:
            args.batch_size = int(cli.batch_size)
        if cli.eval_batches is not None:
            args.eval_batches = int(cli.eval_batches)
        if cli.device is not None:
            args.device = resolve_device(cli.device)
        if cli.seed is not None:
            args.seed = int(cli.seed)

        if mode == "noaddcorr":
            args.use_correlation_attention_bias = False
            args.use_correlation_logit_refinement = False

        run_dir = cli.out_dir / mode
        args.save_dir = run_dir
        args.log_file = run_dir / "train.log"
        if args.log_file.exists() and not cli.append_log:
            args.log_file.unlink()

        pilot_len = int(args.pilot_len)
        print(
            f"\n=== Running {mode}: L={args.pilot_len}, epochs={args.epochs}, "
            f"signal_token_mode={args.signal_token_mode}, save_dir={run_dir} ==="
        )
        training_result = run_training(args)
        mode_history: list[dict[str, float | int | str]] = []
        for epoch_result in training_result["history"]:
            row = {
                "mode": mode,
                "signal_token_mode": str(args.signal_token_mode),
                "pilot_len": int(args.pilot_len),
                "epoch": int(epoch_result["epoch"]),
                "loss": float(epoch_result["loss"]),
                "pm": float(epoch_result["pm"]),
                "pf": float(epoch_result["pf"]),
                "p_mean": float(epoch_result["p_mean"]),
                "skip": int(epoch_result["skip"]),
                "lr": float(epoch_result["lr"]),
                "save_dir": str(run_dir.resolve()),
            }
            all_rows.append(row)
            mode_history.append(row)
        history_by_mode[mode] = mode_history
        write_history_csv(csv_path, all_rows)
        last_plot_path = plot_loss_vs_epoch(
            plot_path,
            history_by_mode,
            pilot_len=pilot_len,
            dpi=cli.dpi,
        )

    print(f"\nSaved CSV: {csv_path.resolve()}")
    if last_plot_path is not None:
        print(f"Saved plot: {last_plot_path.resolve()}")


if __name__ == "__main__":
    main()
