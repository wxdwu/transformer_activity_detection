from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CE_methods.estimators import camp_genie, camp_genie_from_active
from network.data import ActivityDataGenerator, SystemConfig
from network.metrics import pm_pf_at_threshold
from network.model import build_model_from_config


# =========================
# Experiment Parameters
# =========================
CKPT = Path("checkpoint/checkpoints_N200_Lp30_M32_snr20_normpilot_bs128_steps2000/best_pm.pt")
DEVICE = "auto"
OUT_TXT = None  # None -> save to CKPT.parent / "camp_genie_data_report.txt"

N = 200  # number of users
M = 32  # number of antennas
LP = 30  # pilot length
ACTIVITY_PROB = 0.1
CELL_RADIUS_M = 250.0
PMAX_DBM = 23.0
NOISE_MODE = "snr"  # "snr" or "thermal"
SNR_DB = 20.0
NOISE_POWER_DBM_HZ = -169.0
BANDWIDTH_HZ = 10e6

MC_TIMES = 100
MAX_ITER = 40
SEED = 1
MATRIX = "s"  # "s": A=S and X=sqrt(pg)*H; "b": A=B and X=H
THRESHOLD = 0.5  # only for probability diagnostics


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def build_args() -> SimpleNamespace:
    out_txt = OUT_TXT
    if out_txt is None or str(out_txt) == "":
        out_txt = CKPT.parent / "camp_genie_data_report.txt"
    return SimpleNamespace(
        ckpt=CKPT,
        device=DEVICE,
        out_txt=Path(out_txt),
        num_devices=N,
        num_antennas=M,
        pilot_len=LP,
        activity_prob=ACTIVITY_PROB,
        cell_radius_m=CELL_RADIUS_M,
        pmax_dbm=PMAX_DBM,
        noise_mode=NOISE_MODE,
        snr_db=SNR_DB,
        noise_power_dbm_hz=NOISE_POWER_DBM_HZ,
        bandwidth_hz=BANDWIDTH_HZ,
        mc_times=MC_TIMES,
        max_iter=MAX_ITER,
        seed=SEED,
        matrix=MATRIX,
        threshold=THRESHOLD,
    )


def nmse_db_from_mse(mse: torch.Tensor, target: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    target_power_per_entry = (torch.linalg.norm(target) ** 2).real / float(target.numel())
    return 10.0 * torch.log10(mse / target_power_per_entry.clamp_min(eps))


def load_checkpoint_safely(path: Path, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> torch.nn.Module:
    cfg = ckpt.get("model_config", ckpt["config"])
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def build_system_config(args: SimpleNamespace) -> SystemConfig:
    return SystemConfig(
        num_devices=args.num_devices,
        num_antennas=args.num_antennas,
        pilot_len=args.pilot_len,
        activity_prob=args.activity_prob,
        cell_radius_m=args.cell_radius_m,
        noise_power_dbm_hz=args.noise_power_dbm_hz,
        bandwidth_hz=args.bandwidth_hz,
        pmax_dbm=args.pmax_dbm,
        noise_mode=args.noise_mode,
        snr_db=args.snr_db,
    )


def summarize_network_probs(probs_all: torch.Tensor, labels_all: torch.Tensor, threshold: float) -> dict[str, float]:
    probs = probs_all.flatten().float().cpu()
    labels = labels_all.flatten().float().cpu()
    active = labels > 0.5
    inactive = ~active
    pm, pf = pm_pf_at_threshold(probs_all, labels_all, threshold=threshold)

    summary: dict[str, float] = {
        "pm": float(pm),
        "pf": float(pf),
        "prob_mean": float(probs.mean().item()),
        "prob_std": float(probs.std(unbiased=False).item()),
        "prob_active_mean": float(probs[active].mean().item()) if active.any() else float("nan"),
        "prob_inactive_mean": float(probs[inactive].mean().item()) if inactive.any() else float("nan"),
        "prob_active_p10": float(torch.quantile(probs[active], 0.10).item()) if active.any() else float("nan"),
        "prob_active_p50": float(torch.quantile(probs[active], 0.50).item()) if active.any() else float("nan"),
        "prob_active_p90": float(torch.quantile(probs[active], 0.90).item()) if active.any() else float("nan"),
        "prob_inactive_p10": float(torch.quantile(probs[inactive], 0.10).item()) if inactive.any() else float("nan"),
        "prob_inactive_p50": float(torch.quantile(probs[inactive], 0.50).item()) if inactive.any() else float("nan"),
        "prob_inactive_p90": float(torch.quantile(probs[inactive], 0.90).item()) if inactive.any() else float("nan"),
    }
    if probs.numel() > 1 and labels.std(unbiased=False) > 0 and probs.std(unbiased=False) > 0:
        cov = torch.mean((probs - probs.mean()) * (labels - labels.mean()))
        corr = cov / (probs.std(unbiased=False) * labels.std(unbiased=False))
        summary["prob_label_corr"] = float(corr.item())
    else:
        summary["prob_label_corr"] = float("nan")
    return summary


@torch.no_grad()
def run_data_experiment(args: SimpleNamespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    system = build_system_config(args)
    data_gen = ActivityDataGenerator(system, device)

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    model = build_model_from_ckpt(load_checkpoint_safely(args.ckpt, device), device)

    nmse_blind_all: list[torch.Tensor] = []
    nmse_net_all: list[torch.Tensor] = []
    nmse_genie_all: list[torch.Tensor] = []
    probs_eval: list[torch.Tensor] = []
    labels_eval: list[torch.Tensor] = []

    for mc_idx in range(args.mc_times):
        print(f"Monte Carlo {mc_idx + 1} / {args.mc_times}")
        # Base model uses BatchNorm without running stats; it needs at least
        # two samples even in eval mode. Only sample 0 is used for CAMP.
        batch = data_gen.sample_batch(2, return_raw=True)
        label = batch["label"][0]
        sigma_w = math.sqrt(float(batch["noise_var"][0].item()))
        _, probs = model(batch["x_b"], batch["x_y"])
        lambda_net = probs[0].to(device=device, dtype=label.dtype)
        probs_eval.append(lambda_net.detach().cpu())
        labels_eval.append(label.detach().cpu())

        if args.matrix == "b":
            sensing = batch["b"][0]
            target = label.unsqueeze(-1).to(batch["h"].dtype) * batch["h"][0]
            fading = torch.ones_like(label)
        else:
            sensing = batch["s"][0]
            sqrt_pg = torch.sqrt(batch["pg"][0].clamp_min(1e-20)).to(batch["h"].dtype)
            target = label.unsqueeze(-1).to(batch["h"].dtype) * sqrt_pg.unsqueeze(-1) * batch["h"][0]
            fading = batch["pg"][0]

        lambda_blind = torch.full_like(label, float(system.activity_prob))
        _, _, mse_blind, _, _ = camp_genie(
            sensing, batch["y"][0], target, args.max_iter, lambda_blind, fading, sigma_w
        )
        _, _, mse_net, _, _ = camp_genie(
            sensing, batch["y"][0], target, args.max_iter, lambda_net, fading, sigma_w
        )
        _, _, mse_genie, _, _ = camp_genie_from_active(
            sensing, batch["y"][0], target, label, args.max_iter, fading, sigma_w
        )

        nmse_blind_all.append(nmse_db_from_mse(mse_blind, target).detach().cpu())
        nmse_net_all.append(nmse_db_from_mse(mse_net, target).detach().cpu())
        nmse_genie_all.append(nmse_db_from_mse(mse_genie, target).detach().cpu())

    blind_avg = torch.stack(nmse_blind_all, dim=1).mean(dim=1)
    net_avg = torch.stack(nmse_net_all, dim=1).mean(dim=1)
    genie_avg = torch.stack(nmse_genie_all, dim=1).mean(dim=1)
    prob_summary = summarize_network_probs(
        torch.stack(probs_eval, dim=0),
        torch.stack(labels_eval, dim=0),
        threshold=args.threshold,
    )
    return blind_avg, net_avg, genie_avg, prob_summary


def write_report(
    path: Path,
    blind_avg: torch.Tensor,
    net_avg: torch.Tensor,
    genie_avg: torch.Tensor,
    prob_summary: dict[str, float],
    args: SimpleNamespace,
) -> None:
    lines = [
        "CAMP_Genie on network/data.py samples",
        f"Checkpoint: {args.ckpt}",
        f"N/L/M: {args.num_devices}/{args.pilot_len}/{args.num_antennas}",
        f"Activity probability: {args.activity_prob}",
        f"Noise mode: {args.noise_mode}",
        f"SNR dB: {args.snr_db}",
        f"Matrix mapping: {args.matrix}",
        f"Monte Carlo times: {args.mc_times}",
        f"Max iter: {args.max_iter}",
        f"Detection threshold for probability diagnostics: {args.threshold}",
        f"Seed: {args.seed}",
        "",
        "Mapping:",
    ]
    if args.matrix == "b":
        lines.extend(
            [
                "A = batch['b'][i]",
                "X = batch['label'][i][:,None] * batch['h'][i]",
                "fading = ones(N), because active H rows are CN(0,1)",
            ]
        )
    else:
        lines.extend(
            [
                "A = batch['s'][i]",
                "X = sqrt(batch['pg'][i])[:,None] * batch['label'][i][:,None] * batch['h'][i]",
                "fading = batch['pg'][i], because X active-row variance is pg",
            ]
        )
    lines.extend(
        [
            "Y = A @ X + W",
            "Network probabilities are passed directly as per-user lambda, without thresholding.",
            "",
            "iter,standard_camp_nmse_db,network_prob_camp_nmse_db,genie_camp_nmse_db",
        ]
    )
    for idx, (blind, net, genie) in enumerate(
        zip(blind_avg.tolist(), net_avg.tolist(), genie_avg.tolist()), start=1
    ):
        lines.append(f"{idx},{blind:.8f},{net:.8f},{genie:.8f}")
    lines.extend(
        [
            "",
            "Network probability diagnostics:",
            f"PM@{args.threshold:.2f}: {prob_summary['pm']:.6f}",
            f"PF@{args.threshold:.2f}: {prob_summary['pf']:.6f}",
            f"Mean probability: {prob_summary['prob_mean']:.6f}",
            f"Probability std: {prob_summary['prob_std']:.6f}",
            f"Active probability mean: {prob_summary['prob_active_mean']:.6f}",
            f"Inactive probability mean: {prob_summary['prob_inactive_mean']:.6f}",
            f"Active probability p10/p50/p90: {prob_summary['prob_active_p10']:.6f}, "
            f"{prob_summary['prob_active_p50']:.6f}, {prob_summary['prob_active_p90']:.6f}",
            f"Inactive probability p10/p50/p90: {prob_summary['prob_inactive_p10']:.6f}, "
            f"{prob_summary['prob_inactive_p50']:.6f}, {prob_summary['prob_inactive_p90']:.6f}",
            f"Probability-label correlation: {prob_summary['prob_label_corr']:.6f}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_args()
    blind_avg, net_avg, genie_avg, prob_summary = run_data_experiment(args)
    write_report(args.out_txt, blind_avg, net_avg, genie_avg, prob_summary, args)
    print("Standard CAMP on data.py average NMSE (dB):")
    print(blind_avg)
    print("Network-probability CAMP on data.py average NMSE (dB):")
    print(net_avg)
    print("Genie-aided CAMP on data.py average NMSE (dB):")
    print(genie_avg)
    print("Network probability diagnostics:")
    for key, value in prob_summary.items():
        print(f"{key}: {value:.6f}")
    print(f"Report written to {args.out_txt}")


if __name__ == "__main__":
    main()
