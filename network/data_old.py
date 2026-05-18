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


class ActivityDataGenerator:
    def __init__(self, cfg: SystemConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    def _complex_gaussian(self, *shape: int) -> torch.Tensor:
        real = torch.randn(*shape, device=self.device)
        imag = torch.randn(*shape, device=self.device)
        return (real + 1j * imag) / math.sqrt(2.0)

    def _sample_distances_m(self, batch_size: int) -> torch.Tensor:
        # Uniform users in a disk: r = R * sqrt(u)
        u = torch.rand(batch_size, self.cfg.num_devices, device=self.device)
        return self.cfg.cell_radius_m * torch.sqrt(u.clamp_min(1e-8))

    def _large_scale_gain(self, distances_m: torch.Tensor) -> torch.Tensor:
        d_km = (distances_m / 1000.0).clamp_min(1e-3)
        pl_db = 128.1 + 37.6 * torch.log10(d_km)
        return torch.pow(10.0, -pl_db / 10.0)

    def _noise_variance(self) -> float:
        noise_power_dbm = self.cfg.noise_power_dbm_hz + 10.0 * math.log10(self.cfg.bandwidth_hz)
        noise_power_w = 10.0 ** ((noise_power_dbm - 30.0) / 10.0)
        return float(noise_power_w)

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

        # Pilot sequences: S in C^{Lp x N}
        s = self._complex_gaussian(bsz, lp, n)

        distances = self._sample_distances_m(bsz)
        g = self._large_scale_gain(distances)
        g_min = g.min(dim=1, keepdim=True).values
        pmax_w = 10.0 ** ((cfg.pmax_dbm - 30.0) / 10.0)
        p = pmax_w * (g_min / g.clamp_min(1e-16))
        scale = torch.sqrt((p * g).clamp_min(1e-20))
        pg = p * g

        # Scaled pilot matrix B = S G^{1/2}
        b = s * scale.unsqueeze(1)

        # Activity labels a in {0,1}
        a = torch.bernoulli(
            torch.full((bsz, n), cfg.activity_prob, device=self.device)
        ).to(torch.float32)

        # Channels H and noise W
        h = self._complex_gaussian(bsz, n, m)
        noise_var = self._noise_variance()
        w = math.sqrt(noise_var) * self._complex_gaussian(bsz, lp, m)

        # Y = B A H + W
        bh = (b * a.unsqueeze(1)) @ h
        y = bh + w

        # Eq. (5): per-device real/imag features for pilots
        x_b = self._complex_to_real_feature(b.transpose(1, 2).contiguous())

        # Eq. (6): vectorized sample covariance C = Y Y^H / M
        c = (y @ y.conj().transpose(-1, -2)) / float(m)
        x_y = self._complex_to_real_feature(c.reshape(bsz, -1))

        x_b = self._rms_normalize(x_b.to(torch.float32))
        x_y = self._rms_normalize(x_y.to(torch.float32))

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
                    "noise_var": torch.tensor(noise_var, device=self.device, dtype=torch.float32),
                }
            )
        return out
