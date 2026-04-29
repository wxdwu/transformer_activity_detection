from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CE_methods.estimators import camp_genie, camp_genie_from_active
from network.config import load_experiment_config
from network.data import ActivityDataGenerator, SystemConfig


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def nmse_db_from_mse(mse: torch.Tensor, target: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    target_power_per_entry = (torch.linalg.norm(target) ** 2).real / float(target.numel())
    return 10.0 * torch.log10(mse / target_power_per_entry.clamp_min(eps))


def run_data_experiment(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    cfg_dict = load_experiment_config(args.config)
    system = SystemConfig(**cfg_dict["network_train"]["system"])
    data_gen = ActivityDataGenerator(system, device)

    nmse_blind_all: list[torch.Tensor] = []
    nmse_genie_all: list[torch.Tensor] = []

    for batch_idx in range(args.num_batches):
        print(f"Data batch {batch_idx + 1} / {args.num_batches}")
        batch = data_gen.sample_batch(args.batch_size, return_raw=True)
        sigma_w = math.sqrt(float(batch["noise_var"].item()))

        for sample_idx in range(args.batch_size):
            label = batch["label"][sample_idx]

            if args.matrix == "b":
                # data.py: Y = (B * a) @ H + W = B @ (a * H) + W
                sensing = batch["b"][sample_idx]
                target = label.unsqueeze(-1).to(batch["h"].dtype) * batch["h"][sample_idx]
                fading = torch.ones_like(label)
            else:
                # Equivalent form: B = S diag(sqrt(pg)),
                # Y = S @ (sqrt(pg) * a * H) + W.
                sensing = batch["s"][sample_idx]
                sqrt_pg = torch.sqrt(batch["pg"][sample_idx].clamp_min(1e-20)).to(batch["h"].dtype)
                target = label.unsqueeze(-1).to(batch["h"].dtype) * sqrt_pg.unsqueeze(-1) * batch["h"][sample_idx]
                fading = batch["pg"][sample_idx]

            lambda_blind = torch.full_like(label, float(system.activity_prob))
            _, _, mse_blind, _, _ = camp_genie(
                sensing,
                batch["y"][sample_idx],
                target,
                args.max_iter,
                lambda_blind,
                fading,
                sigma_w,
            )
            _, _, mse_genie, _, _ = camp_genie_from_active(
                sensing,
                batch["y"][sample_idx],
                target,
                label,
                args.max_iter,
                fading,
                sigma_w,
            )

            nmse_blind_all.append(nmse_db_from_mse(mse_blind, target).detach().cpu())
            nmse_genie_all.append(nmse_db_from_mse(mse_genie, target).detach().cpu())

    blind_avg = torch.stack(nmse_blind_all, dim=1).mean(dim=1)
    genie_avg = torch.stack(nmse_genie_all, dim=1).mean(dim=1)
    return blind_avg, genie_avg


def write_report(path: Path, blind_avg: torch.Tensor, genie_avg: torch.Tensor, args: argparse.Namespace) -> None:
    lines = [
        "CAMP_Genie on network/data.py samples",
        f"Config: {args.config}",
        f"Matrix mapping: {args.matrix}",
        f"Num batches: {args.num_batches}",
        f"Batch size: {args.batch_size}",
        f"Max iter: {args.max_iter}",
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
            "",
            "iter,standard_camp_nmse_db,genie_camp_nmse_db",
        ]
    )
    for idx, (blind, genie) in enumerate(zip(blind_avg.tolist(), genie_avg.tolist()), start=1):
        lines.append(f"{idx},{blind:.8f},{genie:.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CAMP_Genie on network/data.py generated data.")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--matrix", choices=["b", "s"], default="s")
    parser.add_argument("--out-txt", type=Path, default=Path("CE_methods/camp_genie_data_report.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blind_avg, genie_avg = run_data_experiment(args)
    write_report(args.out_txt, blind_avg, genie_avg, args)
    print("Standard CAMP on data.py average NMSE (dB):")
    print(blind_avg)
    print("Genie-aided CAMP on data.py average NMSE (dB):")
    print(genie_avg)
    print(f"Report written to {args.out_txt}")


if __name__ == "__main__":
    main()
