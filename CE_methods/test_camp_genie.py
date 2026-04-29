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


def complex_gaussian(*shape: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    real = torch.randn(*shape, device=device, dtype=real_dtype)
    imag = torch.randn(*shape, device=device, dtype=real_dtype)
    return (real + 1j * imag).to(dtype=dtype) / math.sqrt(2.0)


def run_mc(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(args.device)
    dtype = torch.complex128 if args.dtype == "complex128" else torch.complex64
    torch.manual_seed(args.seed)

    n = args.num_devices
    l = args.pilot_len
    m = args.num_antennas
    k = int(round(n * args.sparsity))

    nmse_blind_all = torch.zeros((args.max_iter, args.mc_times), dtype=torch.float64, device=device)
    nmse_genie_all = torch.zeros((args.max_iter, args.mc_times), dtype=torch.float64, device=device)

    for mc in range(args.mc_times):
        print(f"Monte Carlo experiment {mc + 1} / {args.mc_times}")

        active_idx = torch.zeros((n,), dtype=torch.float64, device=device)
        perm = torch.randperm(n, device=device)
        active_idx[perm[:k]] = 1.0

        beta = torch.ones((n,), dtype=torch.float64, device=device)

        x = torch.zeros((n, m), dtype=dtype, device=device)
        active_users = torch.nonzero(active_idx, as_tuple=False).flatten()
        x[active_users, :] = complex_gaussian(k, m, device=device, dtype=dtype)

        a = complex_gaussian(l, n, device=device, dtype=dtype) / math.sqrt(float(l))

        noise_power = 10.0 ** (-float(args.snr_db) / 10.0) * (float(k) / float(l))
        sigma_w = math.sqrt(noise_power)
        noise = complex_gaussian(l, m, device=device, dtype=dtype) * sigma_w
        y = a @ x + noise

        lambda_blind = float(args.sparsity) * torch.ones((n,), dtype=torch.float64, device=device)
        _, _, mse_blind, _, _ = camp_genie(a, y, x, args.max_iter, lambda_blind, beta, sigma_w)

        _, _, mse_genie, _, _ = camp_genie_from_active(a, y, x, active_idx, args.max_iter, beta, sigma_w)

        x_power_per_entry = (torch.linalg.norm(x) ** 2).real.to(dtype=torch.float64) / float(n * m)
        nmse_blind_all[:, mc] = 10.0 * torch.log10(mse_blind.to(dtype=torch.float64) / x_power_per_entry)
        nmse_genie_all[:, mc] = 10.0 * torch.log10(mse_genie.to(dtype=torch.float64) / x_power_per_entry)

    return nmse_blind_all.mean(dim=1), nmse_genie_all.mean(dim=1)


def write_report(path: Path, nmse_blind_avg: torch.Tensor, nmse_genie_avg: torch.Tensor, args: argparse.Namespace) -> None:
    lines = [
        "CAMP_Genie Python test matching AMP_Genie/test_mc.m",
        f"N: {args.num_devices}",
        f"L: {args.pilot_len}",
        f"M: {args.num_antennas}",
        f"Sparsity: {args.sparsity}",
        f"SNR dB: {args.snr_db}",
        f"Max iter: {args.max_iter}",
        f"MC times: {args.mc_times}",
        "",
        "iter,standard_camp_nmse_db,genie_camp_nmse_db",
    ]
    blind_cpu = nmse_blind_avg.detach().cpu()
    genie_cpu = nmse_genie_avg.detach().cpu()
    for idx, (blind, genie) in enumerate(zip(blind_cpu.tolist(), genie_cpu.tolist()), start=1):
        lines.append(f"{idx},{blind:.8f},{genie:.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo test for CAMP and Genie-aided CAMP.")
    parser.add_argument("--num-devices", type=int, default=200)
    parser.add_argument("--pilot-len", type=int, default=30)
    parser.add_argument("--num-antennas", type=int, default=32)
    parser.add_argument("--sparsity", type=float, default=0.1)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--mc-times", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex128")
    parser.add_argument("--out-txt", type=Path, default=Path("CE_methods/camp_genie_test_report.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nmse_blind_avg, nmse_genie_avg = run_mc(args)
    write_report(args.out_txt, nmse_blind_avg, nmse_genie_avg, args)
    print("Standard CAMP average NMSE (dB):")
    print(nmse_blind_avg.detach().cpu())
    print("Genie-aided CAMP average NMSE (dB):")
    print(nmse_genie_avg.detach().cpu())
    print(f"Report written to {args.out_txt}")


if __name__ == "__main__":
    main()
