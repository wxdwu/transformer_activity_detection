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
    activity_mode: str = "event"
    event_lambda: float = 1.0
    event_max_count: int = 3
    event_sigma_m: float = 120.0
    event_trigger_prob: float = 0.9
    background_activity_prob: float = 0.005
    correlation_length_m: float = 0.0


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

    def _sample_positions_m(self, batch_size: int, num_points: int | None = None) -> torch.Tensor:
        count = self.cfg.num_devices if num_points is None else num_points
        u = torch.rand(batch_size, count, device=self.device)
        r = self.cfg.cell_radius_m * torch.sqrt(u.clamp_min(1e-8))
        theta = 2.0 * math.pi * torch.rand(batch_size, count, device=self.device)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y], dim=-1)

    def _correlation_from_positions(self, positions: torch.Tensor) -> torch.Tensor:
        n = positions.shape[1]
        diffs = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist2 = (diffs * diffs).sum(dim=-1)
        ell = float(self.cfg.correlation_length_m)
        if ell <= 0.0:
            ell = math.sqrt(2.0) * float(self.cfg.event_sigma_m)
        corr = torch.exp(-dist2 / (2.0 * max(ell * ell, 1e-12)))
        eye = torch.eye(n, device=positions.device, dtype=corr.dtype).unsqueeze(0)
        return (corr * (1.0 - eye)).to(torch.float32)

    def _sample_event_activity(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        bsz, n, _ = positions.shape
        max_events = max(1, int(cfg.event_max_count))

        event_counts = torch.poisson(
            torch.full((bsz,), float(cfg.event_lambda), device=self.device)
        ).to(torch.long).clamp(max=max_events)
        event_centers = self._sample_positions_m(bsz, num_points=max_events)
        event_mask = (
            torch.arange(max_events, device=self.device).unsqueeze(0) < event_counts.unsqueeze(1)
        ).to(torch.float32)

        diffs = positions.unsqueeze(2) - event_centers.unsqueeze(1)
        dist2 = (diffs * diffs).sum(dim=-1)
        sigma2 = max(float(cfg.event_sigma_m) ** 2, 1e-12)
        influence = torch.exp(-dist2 / (2.0 * sigma2)) * event_mask.unsqueeze(1)
        trigger = (float(cfg.event_trigger_prob) * influence).clamp(0.0, 1.0)
        event_prob = 1.0 - torch.prod(1.0 - trigger, dim=2)

        bg = float(cfg.background_activity_prob)
        activity_prob = 1.0 - (1.0 - bg) * (1.0 - event_prob)
        activity_prob = activity_prob.clamp(0.0, 1.0)
        activity = torch.bernoulli(activity_prob).to(torch.float32)
        return activity, activity_prob.to(torch.float32), event_centers.to(torch.float32)

    def _sample_activity(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        bsz, n, _ = positions.shape
        if cfg.activity_mode == "independent":
            prob = torch.full((bsz, n), float(cfg.activity_prob), device=self.device)
            activity = torch.bernoulli(prob).to(torch.float32)
            empty_events = torch.empty((bsz, 0, 2), device=self.device, dtype=torch.float32)
            return activity, prob.to(torch.float32), empty_events
        if cfg.activity_mode == "event":
            return self._sample_event_activity(positions)
        raise ValueError(f"Unknown activity_mode: {cfg.activity_mode}")

    def _large_scale_gain(self, distances_m: torch.Tensor) -> torch.Tensor:
        d_km = (distances_m / 1000.0).clamp_min(1e-3)
        pl_db = 128.1 + 37.6 * torch.log10(d_km)
        return torch.pow(10.0, -pl_db / 10.0)

    def _noise_variance(self) -> float:
        noise_power_dbm = self.cfg.noise_power_dbm_hz + 10.0 * math.log10(self.cfg.bandwidth_hz)
        noise_power_w = 10.0 ** ((noise_power_dbm - 30.0) / 10.0)
        return float(noise_power_w)

    def _measured_snr_noise_variance(self, signal: torch.Tensor) -> torch.Tensor:
        signal_power = (signal.abs() ** 2).mean(dim=(1, 2))
        snr_scale = 10.0 ** (-float(self.cfg.snr_db) / 10.0)
        return (snr_scale * signal_power).clamp_min(1e-30)

    def _large_scale_snr_noise_variance(
        self,
        pg: torch.Tensor,
        activity: torch.Tensor,
        pilot_len: int,
    ) -> torch.Tensor:
        signal_power_per_pilot = (pg * activity).sum(dim=1) / float(pilot_len)
        snr_scale = 10.0 ** (-float(self.cfg.snr_db) / 10.0)
        return (snr_scale * signal_power_per_pilot).clamp_min(1e-30)

    def _batch_noise_variance(
        self,
        pg: torch.Tensor,
        activity: torch.Tensor,
        pilot_len: int,
        signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cfg.noise_mode == "thermal":
            thermal = self._noise_variance()
            return torch.full((pg.shape[0],), thermal, device=self.device, dtype=pg.dtype)
        if self.cfg.noise_mode == "snr":
            if signal is None:
                raise ValueError('noise_mode="snr" requires the noiseless receive signal.')
            return self._measured_snr_noise_variance(signal).to(dtype=pg.dtype)
        if self.cfg.noise_mode == "large_scale_snr":
            return self._large_scale_snr_noise_variance(pg, activity, pilot_len)
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

        # Pilot sequences: S in C^{Lp x N}, with entries CN(0, 1/Lp).
        # This matches AMP_Genie/test_mc.m: (randn + 1i*randn) / sqrt(2*L).
        s = self._complex_gaussian(bsz, lp, n) / math.sqrt(float(lp))

        positions = self._sample_positions_m(bsz)
        distances = torch.linalg.norm(positions, dim=-1)
        corr_matrix = self._correlation_from_positions(positions)
        g = self._large_scale_gain(distances)
        g_min = g.min(dim=1, keepdim=True).values
        pmax_w = 10.0 ** ((cfg.pmax_dbm - 30.0) / 10.0)
        p = pmax_w * (g_min / g.clamp_min(1e-16))
        scale = torch.sqrt((p * g).clamp_min(1e-20))
        pg = p * g

        # Scaled pilot matrix B = S G^{1/2}
        b = s * scale.unsqueeze(1)

        # Activity labels a in {0,1}. In event mode, external event centers
        # trigger nearby users; no global activity-rate calibration is applied.
        a, activity_prob_map, event_centers = self._sample_activity(positions)

        # Channels H and noiseless received signal BAH
        h = self._complex_gaussian(bsz, n, m)
        bh = (b * a.unsqueeze(1)) @ h

        # In SNR mode, match MATLAB awgn(x, SNR, "measured"):
        # measure average power on the noiseless receive matrix [Lp, M].
        noise_var = self._batch_noise_variance(pg, a, lp, signal=bh)
        w = torch.sqrt(noise_var).view(bsz, 1, 1) * self._complex_gaussian(bsz, lp, m)

        # Y = B A H + W
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
            "corr_matrix": corr_matrix,  # [B, N, N], event-induced spatial correlation
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
                    "bh": bh,  # [B, Lp, M], complex noiseless received signal
                    "noise_var": noise_var.to(dtype=torch.float32),  # [B], per-sample noise variance
                    "positions": positions.to(torch.float32),  # [B, N, 2], user coordinates in meters
                    "event_centers": event_centers,  # [B, Kmax, 2], sampled event centers
                    "activity_prob_map": activity_prob_map,  # [B, N], event-driven activation probabilities
                }
            )
        return out
