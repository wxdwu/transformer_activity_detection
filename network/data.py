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
    cell_radius_m: float = 500.0
    noise_power_dbm_hz: float = -169.0
    bandwidth_hz: float = 10e6
    pmax_dbm: float = 23.0
    noise_mode: str = "snr"
    snr_db: float = 20.0
    activity_mode: str = "correlated"
    use_correlation_feature: bool = True
    correlation_activity_strength: float = 0.8


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

    def _correlation_from_positions(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute pairwise spatial correlation and per-user summary features.

        The maximum possible distance inside a disk of radius R is 2R, so
        corr(i,j)=1-d(i,j)/(2R) maps distances into [0,1].
        """
        n = positions.shape[1]
        diffs = positions.unsqueeze(2) - positions.unsqueeze(1)
        dists = torch.sqrt((diffs * diffs).sum(dim=-1).clamp_min(0.0))
        max_dist = max(2.0 * float(self.cfg.cell_radius_m), 1e-12)
        sim = (1.0 - dists / max_dist).clamp(min=0.0, max=1.0)
        eye = torch.eye(n, device=positions.device, dtype=sim.dtype).unsqueeze(0)
        sim = sim * (1.0 - eye)
        corr_summary = sim.sum(dim=2) / float(max(1, n - 1))
        return sim.to(torch.float32), corr_summary.to(torch.float32)

    def _sample_activity(self, sim: torch.Tensor) -> torch.Tensor:
        """Sample activity labels, optionally driven by spatial correlation."""
        cfg = self.cfg
        bsz, n, _ = sim.shape
        base_prob = float(cfg.activity_prob)
        seeds = torch.bernoulli(torch.full((bsz, n), base_prob, device=sim.device)).to(torch.float32)
        if cfg.activity_mode == "independent":
            return seeds
        if cfg.activity_mode != "correlated":
            raise ValueError(f"Unknown activity_mode: {cfg.activity_mode}")

        denom = sim.sum(dim=2).clamp_min(1e-12)
        neighbor_activity = (sim @ seeds.unsqueeze(-1)).squeeze(-1) / denom
        strength = float(max(0.0, min(1.0, cfg.correlation_activity_strength)))
        corr_prob = ((1.0 - strength) * base_prob + strength * neighbor_activity).clamp(0.0, 1.0)
        return torch.bernoulli(corr_prob).to(torch.float32)

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

        positions = self._sample_positions_m(bsz).to(backend)
        distances = torch.linalg.norm(positions, dim=-1)
        corr_matrix, corr = self._correlation_from_positions(positions)

        # Pilot sequences: S in C^{Lp x N}, entries CN(0, 1/Lp)
        s = (torch.randn(bsz, lp, n, device=backend) + 1j * torch.randn(bsz, lp, n, device=backend)) / math.sqrt(
            2.0 * float(lp)
        )

        # Distances from the base station and large-scale gains.
        g = self._large_scale_gain(distances)
        g_min = g.min(dim=1, keepdim=True).values
        pmax_w = 10.0 ** ((cfg.pmax_dbm - 30.0) / 10.0)
        p = pmax_w * (g_min / g.clamp_min(1e-16))
        scale = torch.sqrt((p * g).clamp_min(1e-20))
        pg = p * g

        # Scaled pilot matrix B = S G^{1/2}
        b = s * scale.unsqueeze(1)

        # Activity labels a in {0,1}. In correlated mode, active seed users
        # increase the activation probability of spatially correlated users.
        a = self._sample_activity(corr_matrix)

        # Channels H and noise W on CPU
        h = (torch.randn(bsz, n, m, device=backend) + 1j * torch.randn(bsz, n, m, device=backend)) / math.sqrt(2.0)
        if self.cfg.noise_mode == "thermal":
            thermal = self._noise_variance()
            noise_var = torch.full((bsz,), thermal, device=backend, dtype=torch.float32)
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

        x_b = self._rms_normalize(x_b.to(torch.float32))
        if cfg.use_correlation_feature:
            # Append one spatial-correlation summary scalar per user:
            # [B, N, 2Lp] -> [B, N, 2Lp+1].
            corr_feat = corr.unsqueeze(-1).to(device=x_b.device, dtype=x_b.dtype)
            x_b = torch.cat([x_b, corr_feat], dim=-1)
        x_y = self._rms_normalize(x_y.to(torch.float32))

        # Move final real tensors to target device
        x_b = x_b.to(self.device)
        x_y = x_y.to(self.device)
        a = a.to(self.device)

        out = {
            "x_b": x_b,  # [B, N, 2Lp] or [B, N, 2Lp+1] with correlation feature
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
                    "positions": positions.to(dtype=torch.float32),  # [B, N, 2], user coordinates in meters
                    "corr_matrix": corr_matrix,  # [B, N, N], pairwise spatial correlation in [0,1]
                    "corr_feature": corr,  # [B, N], per-user average spatial correlation
                }
            )
        return out
