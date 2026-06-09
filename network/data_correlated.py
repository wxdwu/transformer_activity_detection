from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import ActivityDataGenerator as IndependentActivityDataGenerator
from .data import SystemConfig as BaseSystemConfig


@dataclass
class CorrelatedSystemConfig(BaseSystemConfig):
    # Four groups of 25 users when num_devices=100.  Each pair has
    # mean alpha / (alpha + beta) < 0.3 and intra-group correlation
    # 1 / (1 + alpha + beta) > 0.5 with visibly different strengths.
    group_beta_params: tuple[tuple[float, float], ...] = (
        (0.085, 0.765),  # mean=0.10, corr=0.5405
        (0.0975, 0.5525),  # mean=0.15, corr=0.6061
        (0.099, 0.351),  # mean=0.22, corr=0.6897
        (0.07, 0.18),  # mean=0.28, corr=0.8000
    )

    def __post_init__(self) -> None:
        means = [alpha / (alpha + beta) for alpha, beta in self.group_beta_params]
        self.activity_prob = sum(means) / len(means)


class CorrelatedActivityDataGenerator(IndependentActivityDataGenerator):
    """Generate data with group-homogeneous correlated activity.

    For each batch sample and each group g:
        q_g ~ Beta(alpha_g, beta_g)
        a_n | q_g ~ Bernoulli(q_g), for users n in group g

    Users are conditionally independent given q_g, while marginal users
    in the same group have correlation 1 / (1 + alpha_g + beta_g).
    Pilots, channel, path loss, power control, and noise follow data.py.
    """

    cfg: CorrelatedSystemConfig

    def __init__(self, cfg: CorrelatedSystemConfig, device: torch.device) -> None:
        super().__init__(cfg, device)
        self._validate_group_beta_params()

    def _validate_group_beta_params(self) -> None:
        params = self.cfg.group_beta_params
        if len(params) != 4:
            raise ValueError("group_beta_params must contain exactly four (alpha, beta) pairs.")
        if self.cfg.num_devices % len(params) != 0:
            raise ValueError("num_devices must be divisible by the number of groups.")
        for alpha, beta in params:
            if alpha <= 0.0 or beta <= 0.0:
                raise ValueError("Beta distribution alpha and beta must be positive.")
            mean = alpha / (alpha + beta)
            if mean >= 0.3:
                raise ValueError(
                    f"Each group activity mean must be < 0.3, got {mean:.4f} "
                    f"for alpha={alpha}, beta={beta}."
                )

    def activity_stats(self) -> list[dict[str, float]]:
        stats = []
        for alpha, beta in self.cfg.group_beta_params:
            stats.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "mean": alpha / (alpha + beta),
                    "correlation": 1.0 / (1.0 + alpha + beta),
                }
            )
        return stats

    def group_activity_prior(self) -> torch.Tensor:
        group_size = self.cfg.num_devices // len(self.cfg.group_beta_params)
        priors = []
        for alpha, beta in self.cfg.group_beta_params:
            mean = alpha / (alpha + beta)
            priors.extend([mean] * group_size)
        return torch.tensor(priors, device=self.device, dtype=torch.float32)

    def _activity_prior_feature(self, batch_size: int) -> torch.Tensor:
        prior = self.group_activity_prior()
        centered = prior - prior.mean()
        scaled = centered / prior.std(unbiased=False).clamp_min(1e-6)
        return scaled.view(1, -1, 1).expand(batch_size, -1, -1)

    def _sample_activity(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        group_size = self.cfg.num_devices // len(self.cfg.group_beta_params)
        labels = []
        probs = []

        for alpha, beta in self.cfg.group_beta_params:
            concentration = torch.tensor([alpha, beta], device=self.device)
            q_g = torch.distributions.Beta(concentration[0], concentration[1]).sample((batch_size, 1))
            a_g = torch.bernoulli(q_g.expand(batch_size, group_size)).to(torch.float32)
            labels.append(a_g)
            probs.append(q_g.expand(batch_size, group_size))

        return torch.cat(labels, dim=1), torch.cat(probs, dim=1).to(torch.float32)

    def sample_batch(self, batch_size: int, return_raw: bool = False) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        bsz, n, lp, m = batch_size, cfg.num_devices, cfg.pilot_len, cfg.num_antennas

        # Pilot sequences: S in C^{Lp x N}, with entries CN(0, 1/Lp).
        # This matches AMP_Genie/test_mc.m: (randn + 1i*randn) / sqrt(2*L).
        s = self._complex_gaussian(bsz, lp, n) / (float(lp) ** 0.5)

        distances = self._sample_distances_m(bsz)
        g = self._large_scale_gain(distances)
        g_min = g.min(dim=1, keepdim=True).values
        pmax_w = 10.0 ** ((cfg.pmax_dbm - 30.0) / 10.0)
        p = pmax_w * (g_min / g.clamp_min(1e-16))
        scale = torch.sqrt((p * g).clamp_min(1e-20))
        pg = p * g

        # Scaled pilot matrix B = S G^{1/2}
        b = s * scale.unsqueeze(1)

        # Group-homogeneous correlated activity labels a in {0,1}.
        a, q = self._sample_activity(bsz)

        # Channels H and noise W
        h = self._complex_gaussian(bsz, n, m)
        noise_var = self._batch_noise_variance(pg, a, lp)
        w = torch.sqrt(noise_var).view(bsz, 1, 1) * self._complex_gaussian(bsz, lp, m)

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
        x_b = torch.cat(
            [x_b, self._matched_energy_feature(b, c), self._activity_prior_feature(bsz)],
            dim=-1,
        )

        out = {
            "x_b": x_b,  # [B, N, 2Lp]
            "x_y": x_y,  # [B, 2Lp^2]
            "label": a,  # [B, N]
            "activity_prior": self.group_activity_prior().unsqueeze(0).expand(bsz, n),
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
                    "activity_prob": q,  # [B, N], shared q_g expanded within each group
                }
            )
        return out


# Drop-in names for code that mirrors `from network.data import ...`.
SystemConfig = CorrelatedSystemConfig
ActivityDataGenerator = CorrelatedActivityDataGenerator
