from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class SystemConfig:
    num_devices: int = 100
    num_antennas: int = 32
    pilot_len: int = 30
    activity_prob: float = 0.1
    cell_radius_m: float = 250.0
    noise_power_dbm_hz: float = -169.0
    bandwidth_hz: float = 10e6
    pmax_dbm: float = 23.0
    noise_mode: str = "snr"
    snr_db: float = 20.0


class ActivityDataGenerator:
    def __init__(self, cfg: SystemConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    def _complex_gaussian(self, *shape: int) -> torch.Tensor:
        # Generate complex gaussian on CUDA when available for this generator
        # (better performance), otherwise on CPU. This avoids generating
        # complex tensors on MPS which lacks full complex support.
        dev = self.device if getattr(self.device, "type", "cpu") == "cuda" else torch.device("cpu")
        real = torch.randn(*shape, device=dev)
        imag = torch.randn(*shape, device=dev)
        return (real + 1j * imag) / math.sqrt(2.0)

    def _sample_distances_m(self, batch_size: int) -> torch.Tensor:
        # Deprecated for inter-user distances; kept for backward compatibility.
        # Uniform users in a disk: r = R * sqrt(u)
        u = torch.rand(batch_size, self.cfg.num_devices, device=self.device)
        return self.cfg.cell_radius_m * torch.sqrt(u.clamp_min(1e-8))

    def _sample_positions_m(self, batch_size: int) -> torch.Tensor:
        """Sample user positions (x,y) uniformly in a disk of radius cell_radius_m.

        Returns tensor shape [B, N, 2].
        """
        bsz, n = batch_size, self.cfg.num_devices
        # radius r = R * sqrt(u)
        u = torch.rand(bsz, n, device=self.device)
        r = self.cfg.cell_radius_m * torch.sqrt(u.clamp_min(1e-8))
        theta = 2.0 * math.pi * torch.rand(bsz, n, device=self.device)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y], dim=-1)

    def _large_scale_gain(self, distances_m: torch.Tensor) -> torch.Tensor:
        d_km = (distances_m / 1000.0).clamp_min(1e-3)
        pl_db = 128.1 + 37.6 * torch.log10(d_km)
        return torch.pow(10.0, -pl_db / 10.0)

    def _noise_variance(self) -> float:
        noise_power_dbm = self.cfg.noise_power_dbm_hz + 10.0 * math.log10(self.cfg.bandwidth_hz)
        noise_power_w = 10.0 ** ((noise_power_dbm - 30.0) / 10.0)
        return float(noise_power_w)

    def _batch_noise_variance(self, pg: torch.Tensor, activity: torch.Tensor, pilot_len: int) -> torch.Tensor:
        if self.cfg.noise_mode == "thermal":
            thermal = self._noise_variance()
            return torch.full((pg.shape[0],), thermal, device=self.device, dtype=pg.dtype)
        if self.cfg.noise_mode == "snr":
            signal_power_per_pilot = (pg * activity).sum(dim=1) / float(pilot_len)
            snr_scale = 10.0 ** (-float(self.cfg.snr_db) / 10.0)
            return (snr_scale * signal_power_per_pilot).clamp_min(1e-30)
        raise ValueError(f"Unknown noise_mode: {self.cfg.noise_mode}")

    @staticmethod
    def _complex_to_real_feature(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.real, x.imag], dim=-1)

    @staticmethod
    def _rms_normalize(x: torch.Tensor, eps: float = 1e-24) -> torch.Tensor:
        # Per-sample RMS normalization for numerical stability.
        dims = tuple(range(1, x.dim()))
        scale = torch.sqrt(torch.mean(x * x, dim=dims, keepdim=True) + eps)
        return x / scale

    def sample_batch(self, batch_size: int, return_raw: bool = False) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        bsz, n, lp, m = batch_size, cfg.num_devices, cfg.pilot_len, cfg.num_antennas

        # Choose backend for complex ops: use CUDA when this generator's device is CUDA,
        # otherwise use CPU (keeps MPS free of complex ops).
        backend = torch.device("cuda") if getattr(self.device, "type", "cpu") == "cuda" else torch.device("cpu")

        # Pilot sequences: S in C^{Lp x N}, entries CN(0, 1/Lp)
        s = (torch.randn(bsz, lp, n, device=backend) + 1j * torch.randn(bsz, lp, n, device=backend)) / math.sqrt(
            2.0 * float(lp)
        )

        # Distances and large-scale gains on CPU
        u = torch.rand(bsz, n, device=backend)
        distances = self.cfg.cell_radius_m * torch.sqrt(u.clamp_min(1e-8))
        g = self._large_scale_gain(distances)
        g_min = g.min(dim=1, keepdim=True).values
        pmax_w = 10.0 ** ((cfg.pmax_dbm - 30.0) / 10.0)
        p = pmax_w * (g_min / g.clamp_min(1e-16))
        scale = torch.sqrt((p * g).clamp_min(1e-20))
        pg = p * g

        # Scaled pilot matrix B = S G^{1/2}
        b = s * scale.unsqueeze(1)

        # Activity labels a in {0,1} on CPU
        a = torch.bernoulli(torch.full((bsz, n), cfg.activity_prob, device=backend)).to(torch.float32)

        # Positions and correlation features on CPU
        u2 = torch.rand(bsz, n, device=backend)
        r = self.cfg.cell_radius_m * torch.sqrt(u2.clamp_min(1e-8))
        theta = 2.0 * math.pi * torch.rand(bsz, n, device=backend)
        x = r * torch.cos(theta)
        y_pos = r * torch.sin(theta)
        positions = torch.stack([x, y_pos], dim=-1)
        pos_exp1 = positions.unsqueeze(2)
        pos_exp2 = positions.unsqueeze(1)
        diffs = pos_exp1 - pos_exp2
        dists = torch.sqrt((diffs * diffs).sum(dim=-1).clamp_min(0.0))
        sim = 1.0 - (dists / (2.0 * float(self.cfg.cell_radius_m)))
        sim = sim.clamp(min=0.0, max=1.0)
        sim = sim * (1.0 - torch.eye(n, device=backend).unsqueeze(0))
        corr = sim.sum(dim=2) / float(max(1, n - 1))
        corr = corr.to(torch.float32)

        # Channels H and noise W on CPU
        h = (torch.randn(bsz, n, m, device=backend) + 1j * torch.randn(bsz, n, m, device=backend)) / math.sqrt(2.0)
        if self.cfg.noise_mode == "thermal":
            thermal = self._noise_variance()
            noise_var = torch.full((bsz,), thermal, device=cpu, dtype=torch.float32)
        else:
            signal_power_per_pilot = (pg * a).sum(dim=1) / float(lp)
            snr_scale = 10.0 ** (-float(self.cfg.snr_db) / 10.0)
            noise_var = (snr_scale * signal_power_per_pilot).clamp_min(1e-30)

        w = torch.sqrt(noise_var).view(bsz, 1, 1) * (
            torch.randn(bsz, lp, m, device=backend) + 1j * torch.randn(bsz, lp, m, device=backend)
        ) / math.sqrt(2.0)

        # Y = B A H + W
        bh = (b * a.unsqueeze(1)) @ h
        y = bh + w

        # Convert complex data to real-feature representations on CPU
        x_b = self._complex_to_real_feature(b.transpose(1, 2).contiguous())
        c = (y @ y.conj().transpose(-1, -2)) / float(m)
        x_y = self._complex_to_real_feature(c.reshape(bsz, -1))

        # Append correlation scalar as an extra feature dimension per user: [B, N, 2Lp+1]
        # Ensure corr is on same backend before concatenation
        corr_feat = corr.unsqueeze(-1).to(x_b.real.device)
        x_b = torch.cat([x_b.to(torch.float32), corr_feat], dim=-1)
        x_b = self._rms_normalize(x_b)
        x_y = self._rms_normalize(x_y.to(torch.float32))

        # Move final real tensors to target device
        # Move final real tensors to target device
        x_b = x_b.to(self.device)
        x_y = x_y.to(self.device)
        a = a.to(self.device)

        out = {
            "x_b": x_b,  # [B, N, 2Lp]
            "x_y": x_y,  # [B, 2Lp^2]
            "label": a,  # [B, N]
        }
        if return_raw:
            out.update(
                {
                    "y": y,  # [B, Lp, M], complex
                    "b": b,  # [B, Lp, N], complex
                    "s": s,  # [B, Lp, N], complex (unscaled pilots)
                    "pg": pg,  # [B, N], equivalent large-scale power factor
                    "beta": g,  # [B, N], large-scale fading gain
                    "h": h,  # [B, N, M], complex (ground-truth)
                    "noise_var": noise_var.to(dtype=torch.float32),  # [B], per-sample noise variance
                }
            )
        return out
