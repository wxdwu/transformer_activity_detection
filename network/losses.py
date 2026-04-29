from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_activity_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    activity_prob: float,
) -> torch.Tensor:
    # Paper Eq. (4): weighted BCE (active class gets larger weight under sparse activity)
    n = targets.shape[1]
    k_over_n = float(activity_prob)
    pos_w = 1.0 - k_over_n
    neg_w = k_over_n

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    sample_weight = targets * pos_w + (1.0 - targets) * neg_w
    return (2.0 / float(n)) * (sample_weight * bce).sum(dim=1).mean()
