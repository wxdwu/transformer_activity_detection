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


def parse_num_users(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(range(50, 251, 50))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_modes(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return ["addcorr", "noaddcorr"]
    modes = [x.strip() for x in raw.split(",") if x.strip()]
    allowed = {"addcorr", "noaddcorr"}
    unknown = [mode for mode in modes if mode not in allowed]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}. Allowed modes: {sorted(allowed)}")
    return modes


def write_results_csv(path: Path, results: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mode", "num_users", "loss", "pm", "pf", "p_mean", "epoch", "save_dir"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def plot_loss_vs_num_users(
    path: Path,
    results_by_mode: dict[str, list[dict[str, float | int | str]]],
    dpi: int,
) -> Path | None:
    if not any(results_by_mode.values()):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        svg_path = path.with_suffix(".svg")
        write_loss_vs_num_users_svg(svg_path, results_by_mode)
        print(f"matplotlib is unavailable; saved SVG plot to: {svg_path.resolve()}")
        return svg_path

    plt.figure(figsize=(7, 4.5))
    for mode, rows in results_by_mode.items():
        if not rows:
            continue
        xs = [int(row["num_users"]) for row in rows]
        ys = [float(row["loss"]) for row in rows]
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=mode)
    plt.xlabel("Number of users N")
    plt.ylabel("loss")
    plt.title("Final Training Loss vs Number of Users")
    plt.xlim(50, 250)
    plt.xticks(list(range(50, 251, 50)))
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    return path


def write_loss_vs_num_users_svg(
    path: Path,
    results_by_mode: dict[str, list[dict[str, float | int | str]]],
) -> None:
    width, height = 900, 560
    left, right, top, bottom = 90, 35, 45, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = 50, 250
    all_ys = [
        float(row["loss"])
        for rows in results_by_mode.values()
        for row in rows
    ]
    if not all_ys:
        return
    y_min, y_max = min(all_ys), max(all_ys)
    if y_min == y_max:
        pad = max(abs(y_min) * 0.1, 1e-3)
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    x_ticks = list(range(50, 251, 50))
    y_ticks = [y_min + i * (y_max - y_min) / 5.0 for i in range(6)]
    colors = {"addcorr": "#1f77b4", "noaddcorr": "#d62728"}

    lines: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="560" viewBox="0 0 900 560">',
        '<rect width="900" height="560" fill="white"/>',
        '<text x="450" y="28" text-anchor="middle" font-family="Arial" font-size="20">Final Training Loss vs Number of Users</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222" stroke-width="1.5"/>',
    ]
    for tick in x_ticks:
        x_pos = sx(tick)
        lines.append(f'<line x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{height-bottom}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="{x_pos:.2f}" y="{height-bottom+24}" text-anchor="middle" font-family="Arial" font-size="13">{tick}</text>')
    for tick in y_ticks:
        y_pos = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y_pos:.2f}" x2="{width-right}" y2="{y_pos:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        lines.append(f'<text x="{left-10}" y="{y_pos+4:.2f}" text-anchor="end" font-family="Arial" font-size="13">{tick:.4f}</text>')
    lines.extend(
        [
            '<text x="450" y="535" text-anchor="middle" font-family="Arial" font-size="16">Number of users N</text>',
            '<text x="22" y="280" text-anchor="middle" font-family="Arial" font-size="16" transform="rotate(-90 22 280)">loss</text>',
        ]
    )
    legend_x, legend_y = width - right - 165, top + 18
    for offset, (mode, rows) in enumerate(results_by_mode.items()):
        if not rows:
            continue
        color = colors.get(mode, "#444")
        xs = [int(row["num_users"]) for row in rows]
        ys = [float(row["loss"]) for row in rows]
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in zip(xs, ys):
            lines.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4.5" fill="{color}"/>')
        y0 = legend_y + offset * 24
        lines.append(f'<line x1="{legend_x}" y1="{y0}" x2="{legend_x+28}" y2="{y0}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+36}" y="{y0+4}" font-family="Arial" font-size="14">{mode}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one model per user count N and plot final epoch loss vs N."
    )
    parser.add_argument("--num-users", type=str, default="", help="Comma-separated N list. Default: 50,100,...,250.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "checkpoint" / "varying_N_260720_covrows",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--modes", type=str, default="", help="Comma-separated modes. Default: addcorr,noaddcorr.")
    parser.add_argument("--mode", choices=["addcorr", "noaddcorr"], default=None, help="Legacy single-mode shortcut.")
    parser.add_argument("--append-log", action="store_true", help="Append to existing per-N train.log files.")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    num_users_values = parse_num_users(cli.num_users)
    modes = parse_modes(cli.mode if cli.mode is not None else cli.modes)
    results: list[dict[str, float | int | str]] = []
    results_by_mode: dict[str, list[dict[str, float | int | str]]] = {mode: [] for mode in modes}
    csv_path = cli.out_dir / "final_loss_by_N.csv"
    plot_path = cli.out_dir / "final_loss_by_N.png"
    last_plot_path: Path | None = None

    for mode in modes:
        for num_users in num_users_values:
            args = copy(build_args())
            args.num_devices = int(num_users)
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

            run_dir = cli.out_dir / mode / f"N{num_users:03d}"
            args.save_dir = run_dir
            args.log_file = run_dir / "train.log"
            if args.log_file.exists() and not cli.append_log:
                args.log_file.unlink()

            print(f"\n=== Running {mode}: N={num_users}, epochs={args.epochs}, save_dir={run_dir} ===")
            train_result = run_training(args)
            final = dict(train_result["final"])
            row = {
                "mode": mode,
                "num_users": int(num_users),
                "loss": float(final["loss"]),
                "pm": float(final["pm"]),
                "pf": float(final["pf"]),
                "p_mean": float(final["p_mean"]),
                "epoch": int(final["epoch"]),
                "save_dir": str(run_dir.resolve()),
            }
            results.append(row)
            results_by_mode[mode].append(row)
            write_results_csv(csv_path, results)
            last_plot_path = plot_loss_vs_num_users(plot_path, results_by_mode, cli.dpi)
            print(f"Recorded {mode} N={num_users}: loss={row['loss']:.6f}")

    print(f"\nSaved CSV: {csv_path.resolve()}")
    if last_plot_path is not None:
        print(f"Saved plot: {last_plot_path.resolve()}")


if __name__ == "__main__":
    main()
