from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CE_methods.estimators import _thresh_prime_thresh_complex_gaussian_matlab
from network.config import apply_cli_overrides, load_experiment_config, section_namespace
from network.data import ActivityDataGenerator, SystemConfig
from network.model import build_model_from_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot AMP NMSE vs iteration for amp_soft and amp_fixed.")
    p.add_argument("--config", type=str, default="config.json", help="Experiment config JSON path.")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_test_batches", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_iters", type=int, default=None)
    p.add_argument("--camp_damping", type=float, default=None)
    p.add_argument("--camp_lambda_floor", type=float, default=None)
    p.add_argument("--camp_fixed_lambda", type=float, default=None)
    p.add_argument("--camp_prob_calib", type=str, default=None, choices=["none", "sigmoid_center"])
    p.add_argument("--camp_prob_center", type=float, default=None)
    p.add_argument("--camp_prob_alpha", type=float, default=None)
    p.add_argument(
        "--out_png",
        type=str,
        default=None,
    )
    p.add_argument(
        "--out_txt",
        type=str,
        default=None,
    )
    return p.parse_args()


def complex_nmse(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.sum(torch.abs(est - target) ** 2)
    den = torch.sum(torch.abs(target) ** 2) + eps
    return float((num / den).item())


def apply_prob_calib(
    probs: torch.Tensor,
    mode: str,
    center: float,
    alpha: float,
) -> torch.Tensor:
    if mode == "none":
        return probs
    if mode == "sigmoid_center":
        c = torch.tensor(center, device=probs.device, dtype=probs.dtype)
        a = torch.tensor(alpha, device=probs.device, dtype=probs.dtype)
        return torch.sigmoid(a * (probs - c))
    raise ValueError(f"Unknown prob calib mode: {mode}")


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> torch.nn.Module:
    cfg = ckpt.get("model_config", ckpt["config"])
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def camp_nmse_curve_one_sample(
    y: torch.Tensor,
    s: torch.Tensor,
    pg: torch.Tensor,
    x_true: torch.Tensor,
    lambda_vec: torch.Tensor,
    max_iters: int,
    damping: float,
    lambda_floor: float,
) -> list[float]:
    # Match current run_detection_lmmse AMP path: unscaled_gain + column normalization.
    l, n = s.shape
    m = y.shape[1]
    rtype = y.real.dtype
    dtype = y.dtype

    d = torch.linalg.norm(s, dim=0).to(dtype=rtype).clamp_min(1e-12)
    a_use = s / d.to(dtype=dtype).unsqueeze(0)
    fading_use = pg.to(dtype=rtype).clamp_min(1e-20) * (d * d)

    lam = lambda_vec.to(dtype=rtype).clamp(0.0, 1.0)
    z = y.clone()
    x = torch.zeros((n, m), dtype=dtype, device=y.device)
    tau = torch.sqrt(torch.sum(torch.abs(y) ** 2) / float(m * l)).to(dtype=rtype)
    damp = float(max(0.0, min(1.0, damping)))

    nmse_curve: list[float] = []
    for _ in range(max_iters):
        inp = a_use.conj().transpose(-1, -2) @ z + x
        x_new, avg_xprime = _thresh_prime_thresh_complex_gaussian_matlab(
            y=inp,
            sigma=tau,
            lambda_vec=lam,
            p_ls=fading_use,
            lambda_floor=lambda_floor,
        )
        x = (1.0 - damp) * x + damp * x_new
        z_new = y - a_use @ x + (float(n) / float(l)) * (z @ avg_xprime)
        z = (1.0 - damp) * z + damp * z_new
        tau = torch.sqrt(torch.sum(torch.abs(z) ** 2) / float(m * l)).to(dtype=rtype)

        # Undo normalization and unscaled_gain mapping back to effective channel X_true = A*H.
        x_hat = x / d.to(dtype=dtype).unsqueeze(-1)
        x_hat = x_hat / torch.sqrt(pg.to(dtype=rtype).clamp_min(1e-20)).to(dtype=dtype).unsqueeze(-1)
        nmse_curve.append(complex_nmse(x_hat, x_true))
    return nmse_curve


@torch.no_grad()
def main() -> None:
    cli = parse_args()
    exp_cfg = load_experiment_config(cli.config)
    args = section_namespace(exp_cfg, "plot_plot_amp_nmse_vs_iter")
    apply_cli_overrides(
        args,
        cli,
        [
            "ckpt",
            "device",
            "num_test_batches",
            "batch_size",
            "max_iters",
            "camp_damping",
            "camp_lambda_floor",
            "camp_fixed_lambda",
            "camp_prob_calib",
            "camp_prob_center",
            "camp_prob_alpha",
            "out_png",
            "out_txt",
        ],
    )
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model_from_ckpt(ckpt, device)
    data_gen = ActivityDataGenerator(SystemConfig(**ckpt["system_config"]), device=device)

    soft_nmse_sum = torch.zeros((args.max_iters,), dtype=torch.float64)
    fixed_nmse_sum = torch.zeros((args.max_iters,), dtype=torch.float64)
    count = 0

    for _ in range(args.num_test_batches):
        batch = data_gen.sample_batch(args.batch_size, return_raw=True)
        _, probs = model(batch["x_b"], batch["x_y"])

        y = batch["y"]
        s = batch["s"]
        pg = batch["pg"]
        h = batch["h"]
        labels = batch["label"]

        for i in range(args.batch_size):
            x_true = labels[i].unsqueeze(-1).to(h.dtype) * h[i]

            lam_soft = apply_prob_calib(
                probs[i].to(dtype=y[i].real.dtype),
                mode=args.camp_prob_calib,
                center=args.camp_prob_center,
                alpha=args.camp_prob_alpha,
            )
            lam_fixed = torch.full_like(lam_soft, float(args.camp_fixed_lambda))

            soft_curve = camp_nmse_curve_one_sample(
                y=y[i],
                s=s[i],
                pg=pg[i],
                x_true=x_true,
                lambda_vec=lam_soft,
                max_iters=args.max_iters,
                damping=args.camp_damping,
                lambda_floor=args.camp_lambda_floor,
            )
            fixed_curve = camp_nmse_curve_one_sample(
                y=y[i],
                s=s[i],
                pg=pg[i],
                x_true=x_true,
                lambda_vec=lam_fixed,
                max_iters=args.max_iters,
                damping=args.camp_damping,
                lambda_floor=args.camp_lambda_floor,
            )

            soft_nmse_sum += torch.tensor(soft_curve, dtype=torch.float64)
            fixed_nmse_sum += torch.tensor(fixed_curve, dtype=torch.float64)
            count += 1

    soft_mean = (soft_nmse_sum / max(count, 1)).tolist()
    fixed_mean = (fixed_nmse_sum / max(count, 1)).tolist()
    soft_db = [10.0 * math.log10(max(v, 1e-30)) for v in soft_mean]
    fixed_db = [10.0 * math.log10(max(v, 1e-30)) for v in fixed_mean]
    xs = list(range(1, args.max_iters + 1))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(xs, soft_db, marker="o", linewidth=2.0, label="AMP-CAMP + soft probs")
    plt.plot(xs, fixed_db, marker="s", linewidth=2.0, label=f"AMP-CAMP + fixed lambda={args.camp_fixed_lambda:g}")
    plt.xlabel("Iteration")
    plt.ylabel("Average NMSE (dB)")
    plt.title(f"AMP NMSE vs Iteration (Batches={args.num_test_batches}, Batch={args.batch_size})")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)

    lines = [
        f"Checkpoint: {ckpt_path.resolve()}",
        f"Device: {device}",
        f"Num test batches: {args.num_test_batches}",
        f"Batch size: {args.batch_size}",
        f"Total samples: {count}",
        f"Max iters: {args.max_iters}",
        f"CAMP damping: {args.camp_damping}",
        f"CAMP prob calib (soft): {args.camp_prob_calib}",
        "",
        "Iter,AMP-soft(dB),AMP-fixed(dB)",
    ]
    for i in range(args.max_iters):
        lines.append(f"{i+1},{soft_db[i]:.4f},{fixed_db[i]:.4f}")

    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved figure to: {out_png.resolve()}")
    print(f"Saved values to: {out_txt.resolve()}")


if __name__ == "__main__":
    main()

