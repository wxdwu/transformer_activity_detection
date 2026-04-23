from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from htad.data import ActivityDataGenerator, SystemConfig
from htad.amp_estimator import camp_mmse_channel_estimate
from htad.estimators import (
    active_indices_from_probs,
    lmmse_channel_estimate,
    lmmse_formula_estimate,
    ls_channel_estimate,
    ls_min_norm_estimate,
)
from htad.metrics import pm_pf_at_threshold
from htad.model import HeterogeneousTransformer

METHOD_DESCRIPTIONS = {
    "amp_soft": "AMP-CAMP + soft probs",
    "amp_fixed": "AMP-CAMP + fixed lambda",
    "lmmse_detected": "LMMSE + detected active-set",
    "oracle_lmmse": "Oracle-LMMSE + true active-set",
    "lmmse_formula_detected": "LMMSE-formula + detected active-set",
    "ls_detected": "LS + detected active-set",
    "ls_minnorm_detected": "LS-minnorm + detected active-set",
    "oracle_amp": "Oracle-AMP + true label probs",
}
DEFAULT_METHODS = ["amp_soft", "amp_fixed", "lmmse_detected", "oracle_lmmse"]


def complex_nmse(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.sum(torch.abs(est - target) ** 2)
    den = torch.sum(torch.abs(target) ** 2) + eps
    return float((num / den).item())


def nmse_db(nmse: float, floor_db: float = -300.0) -> float:
    v = max(float(nmse), 1e-30)
    db = 10.0 * math.log10(v)
    if db < floor_db:
        db = floor_db
    return float(db)


def parse_methods(raw_methods: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in raw_methods:
        parts = [x.strip() for x in item.split(",") if x.strip()]
        expanded.extend(parts)
    if len(expanded) == 1 and expanded[0].lower() == "all":
        return list(METHOD_DESCRIPTIONS.keys())

    unknown = [m for m in expanded if m not in METHOD_DESCRIPTIONS]
    if unknown:
        valid = ", ".join(METHOD_DESCRIPTIONS.keys())
        raise ValueError(f"Unknown methods: {unknown}. Valid methods: {valid}")

    # Keep input order, de-duplicate.
    out: list[str] = []
    seen: set[str] = set()
    for m in expanded:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run activity detection + channel estimation from a checkpoint.")
    p.add_argument("--ckpt", type=str, default="checkpoints/last.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_test_batches", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.5, help="Activity detection threshold.")
    p.add_argument("--topk", type=int, default=0, help="If >0, use top-k users as active instead of threshold.")
    p.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help=(
            "Methods to run (space/comma separated). "
            "Use 'all' to run all methods. "
            f"Default: {','.join(DEFAULT_METHODS)}"
        ),
    )

    # Linear-estimator config
    p.add_argument("--channel_var", type=float, default=1.0, help="Prior channel variance.")
    p.add_argument("--reg_eps", type=float, default=1e-18, help="Diagonal regularization epsilon.")
    p.add_argument(
        "--prior_mode",
        type=str,
        default="unit",
        choices=["unit", "large_scale", "unscaled_gain"],
        help="Prior mode for lmmse_formula.",
    )

    # AMP config
    p.add_argument("--camp_iters", type=int, default=12, help="Iteration count for CAMP-MMSE.")
    p.add_argument(
        "--camp_iters_fixed",
        type=int,
        default=6,
        help="Iteration count for traditional AMP baseline (amp_fixed).",
    )
    p.add_argument(
        "--camp_fading_mode",
        type=str,
        default="unscaled_gain",
        choices=["unit", "large_scale", "unscaled_gain"],
        help="Per-user prior variance mode for CAMP-MMSE.",
    )
    p.add_argument("--camp_lambda_floor", type=float, default=1e-6, help="Clamp floor for per-user lambda in CAMP.")
    p.add_argument("--camp_prob_calib", type=str, default="sigmoid_center", choices=["sigmoid_center", "none"])
    p.add_argument("--camp_prob_center", type=float, default=0.5)
    p.add_argument("--camp_prob_alpha", type=float, default=12.0)
    p.add_argument("--camp_damping", type=float, default=0.7, help="AMP damping in [0,1].")
    p.add_argument("--camp_fixed_lambda", type=float, default=0.1, help="Fixed lambda for amp_fixed.")

    p.add_argument("--out_txt", type=str, default="detection_lmmse_report.txt", help="Output summary text file.")
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


def estimate_from_active_set(
    estimator: str,
    y_i: torch.Tensor,
    b_i: torch.Tensor,
    idx_i: torch.Tensor,
    n_users: int,
    noise_var: float,
    channel_var: float,
    beta_i: torch.Tensor | None = None,
    s_i: torch.Tensor | None = None,
    pg_i: torch.Tensor | None = None,
    probs_i: torch.Tensor | None = None,
    prior_mode: str = "unit",
    reg_eps: float = 1e-18,
    camp_iters: int = 12,
    camp_fading_mode: str = "unit",
    camp_lambda_floor: float = 1e-6,
    camp_prob_calib: str = "sigmoid_center",
    camp_prob_center: float = 0.5,
    camp_prob_alpha: float = 12.0,
    camp_damping: float = 0.7,
) -> torch.Tensor:
    if estimator == "camp_mmse":
        if probs_i is None:
            raise ValueError("`probs_i` is required for estimator='camp_mmse'.")
        amp_a = b_i
        need_unscale = False
        sqrt_pg = None
        if camp_fading_mode == "unscaled_gain" and pg_i is not None:
            fading = (channel_var * pg_i).to(dtype=y_i.real.dtype).clamp_min(1e-20)
            if s_i is not None:
                amp_a = s_i
                sqrt_pg = torch.sqrt(pg_i.to(dtype=y_i.real.dtype).clamp_min(1e-20))
                need_unscale = True
        elif camp_fading_mode == "large_scale" and beta_i is not None:
            fading = beta_i.to(dtype=y_i.real.dtype).clamp_min(1e-20)
        else:
            fading = torch.full_like(probs_i, float(channel_var), dtype=y_i.real.dtype)

        x_hat = camp_mmse_channel_estimate(
            y=y_i,
            b=amp_a,
            probs=probs_i.to(dtype=y_i.real.dtype),
            fading=fading,
            noise_var=float(noise_var),
            max_iters=int(camp_iters),
            lambda_floor=camp_lambda_floor,
            normalize_columns=True,
            prob_calib=camp_prob_calib,
            prob_center=camp_prob_center,
            prob_alpha=camp_prob_alpha,
            damping=camp_damping,
        )
        if need_unscale and sqrt_pg is not None:
            x_hat = x_hat / sqrt_pg.to(dtype=x_hat.dtype).unsqueeze(-1)
        return x_hat

    if estimator == "lmmse":
        return lmmse_channel_estimate(
            y=y_i,
            b=b_i,
            active_indices=idx_i,
            noise_var=noise_var,
            channel_var=channel_var,
            reg_eps=reg_eps,
            return_active_only=False,
        )

    if estimator == "ls":
        return ls_channel_estimate(
            y=y_i,
            b=b_i,
            active_indices=idx_i,
            reg_eps=reg_eps,
            return_active_only=False,
        )

    idx = torch.as_tensor(idx_i, device=y_i.device, dtype=torch.long).flatten()
    m = y_i.shape[1]
    out = torch.zeros((n_users, m), dtype=y_i.dtype, device=y_i.device)
    if idx.numel() == 0:
        return out

    a_s = b_i[:, idx]
    if estimator == "lmmse_formula":
        if prior_mode == "unscaled_gain" and s_i is not None and pg_i is not None:
            s_s = s_i[:, idx]
            pg_s = pg_i[idx].to(dtype=y_i.real.dtype).clamp_min(1e-20)
            rx = torch.diag((channel_var * pg_s).to(dtype=y_i.dtype))
            u_s = lmmse_formula_estimate(y=y_i, a=s_s, rx=rx, rn=noise_var, reg_eps=reg_eps)
            sqrt_pg = torch.sqrt(pg_s).to(dtype=y_i.dtype).unsqueeze(-1)
            h_s = u_s / sqrt_pg
        elif prior_mode == "large_scale" and beta_i is not None:
            beta_s = beta_i[idx].to(dtype=y_i.real.dtype).clamp_min(1e-20)
            rx = torch.diag(beta_s.to(dtype=y_i.dtype))
            h_s = lmmse_formula_estimate(y=y_i, a=a_s, rx=rx, rn=noise_var, reg_eps=reg_eps)
        else:
            h_s = lmmse_formula_estimate(y=y_i, a=a_s, rx=channel_var, rn=noise_var, reg_eps=reg_eps)
    elif estimator == "ls_minnorm":
        h_s = ls_min_norm_estimate(y=y_i, a=a_s, reg_eps=reg_eps)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    out[idx] = h_s
    return out


def estimate_for_method(
    method: str,
    *,
    y_i: torch.Tensor,
    b_i: torch.Tensor,
    s_i: torch.Tensor,
    pg_i: torch.Tensor,
    beta_i: torch.Tensor,
    probs_i: torch.Tensor,
    label_i: torch.Tensor,
    pred_idx_i: torch.Tensor,
    true_idx_i: torch.Tensor,
    n_users: int,
    noise_var: float,
    args: argparse.Namespace,
) -> torch.Tensor:
    common = dict(
        y_i=y_i,
        b_i=b_i,
        n_users=n_users,
        noise_var=noise_var,
        channel_var=args.channel_var,
        beta_i=beta_i,
        s_i=s_i,
        pg_i=pg_i,
        prior_mode=args.prior_mode,
        reg_eps=args.reg_eps,
        camp_fading_mode=args.camp_fading_mode,
        camp_lambda_floor=args.camp_lambda_floor,
        camp_prob_center=args.camp_prob_center,
        camp_prob_alpha=args.camp_prob_alpha,
        camp_damping=args.camp_damping,
    )

    if method == "amp_soft":
        return estimate_from_active_set(
            estimator="camp_mmse",
            idx_i=pred_idx_i,
            probs_i=probs_i,
            camp_prob_calib=args.camp_prob_calib,
            camp_iters=args.camp_iters,
            **common,
        )
    if method == "amp_fixed":
        return estimate_from_active_set(
            estimator="camp_mmse",
            idx_i=pred_idx_i,
            probs_i=torch.full_like(probs_i, float(args.camp_fixed_lambda)),
            camp_prob_calib="none",
            camp_iters=args.camp_iters_fixed,
            **common,
        )
    if method == "oracle_amp":
        return estimate_from_active_set(
            estimator="camp_mmse",
            idx_i=true_idx_i,
            probs_i=label_i,
            camp_prob_calib="none",
            camp_iters=args.camp_iters,
            **common,
        )
    if method == "lmmse_detected":
        return estimate_from_active_set(estimator="lmmse", idx_i=pred_idx_i, probs_i=None, camp_prob_calib="none", **common)
    if method == "oracle_lmmse":
        return estimate_from_active_set(estimator="lmmse", idx_i=true_idx_i, probs_i=None, camp_prob_calib="none", **common)
    if method == "lmmse_formula_detected":
        return estimate_from_active_set(
            estimator="lmmse_formula",
            idx_i=pred_idx_i,
            probs_i=None,
            camp_prob_calib="none",
            **common,
        )
    if method == "ls_detected":
        return estimate_from_active_set(estimator="ls", idx_i=pred_idx_i, probs_i=None, camp_prob_calib="none", **common)
    if method == "ls_minnorm_detected":
        return estimate_from_active_set(
            estimator="ls_minnorm",
            idx_i=pred_idx_i,
            probs_i=None,
            camp_prob_calib="none",
            **common,
        )

    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)

    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model_from_ckpt(ckpt, device)
    sys_cfg = SystemConfig(**ckpt["system_config"])
    data_gen = ActivityDataGenerator(sys_cfg, device=device)

    all_probs = []
    all_labels = []
    nmse_by_method: dict[str, list[float]] = {m: [] for m in methods}

    topk = args.topk if args.topk > 0 else None

    for _ in range(args.num_test_batches):
        batch = data_gen.sample_batch(args.batch_size, return_raw=True)
        _, probs = model(batch["x_b"], batch["x_y"])  # [B, N]

        labels = batch["label"]
        y = batch["y"]
        b = batch["b"]
        s = batch["s"]
        pg = batch["pg"]
        beta = batch["beta"]
        h = batch["h"]
        noise_var = float(batch["noise_var"].item())

        pred_idx_list = active_indices_from_probs(probs, threshold=args.threshold, topk=topk)
        true_idx_list = active_indices_from_probs(labels, threshold=0.5, topk=None)

        for i in range(args.batch_size):
            x_true = labels[i].unsqueeze(-1).to(h.dtype) * h[i]  # [N, M]
            for method in methods:
                x_hat = estimate_for_method(
                    method,
                    y_i=y[i],
                    b_i=b[i],
                    s_i=s[i],
                    pg_i=pg[i],
                    beta_i=beta[i],
                    probs_i=probs[i],
                    label_i=labels[i],
                    pred_idx_i=pred_idx_list[i],
                    true_idx_i=true_idx_list[i],
                    n_users=labels.shape[1],
                    noise_var=noise_var,
                    args=args,
                )
                nmse_by_method[method].append(complex_nmse(x_hat, x_true))

        all_probs.append(probs)
        all_labels.append(labels)

    probs_all = torch.cat(all_probs, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    pm, pf = pm_pf_at_threshold(probs_all, labels_all, threshold=args.threshold)

    lines = [
        f"Checkpoint: {ckpt_path.resolve()}",
        f"Device: {device}",
        f"Methods: {', '.join(methods)}",
        f"Test batches: {args.num_test_batches}",
        f"Batch size: {args.batch_size}",
        f"Detection threshold: {args.threshold}",
        f"Top-k mode: {topk if topk is not None else 'off'}",
        f"Prior mode: {args.prior_mode}",
        f"CAMP iters: {args.camp_iters}",
        f"CAMP iters (fixed AMP): {args.camp_iters_fixed}",
        f"CAMP fading mode: {args.camp_fading_mode}",
        f"CAMP fixed lambda: {args.camp_fixed_lambda}",
        f"CAMP lambda floor: {args.camp_lambda_floor:.1e}",
        f"CAMP prob calib: {args.camp_prob_calib}",
        f"CAMP prob center: {args.camp_prob_center}",
        f"CAMP prob alpha: {args.camp_prob_alpha}",
        f"CAMP damping: {args.camp_damping}",
        f"Reg eps: {args.reg_eps:.3e}",
        f"PM: {pm:.6f}",
        f"PF: {pf:.6f}",
    ]

    for method in methods:
        method_nmse = float(torch.tensor(nmse_by_method[method]).mean().item())
        lines.append(f"NMSE ({METHOD_DESCRIPTIONS[method]}): {nmse_db(method_nmse):.2f} dB")

    report = "\n".join(lines)
    print(report)

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
