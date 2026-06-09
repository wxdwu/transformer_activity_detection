from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_activity_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    activity_prob: float | torch.Tensor,
) -> torch.Tensor:
    # Paper Eq. (4): weighted BCE (active class gets larger weight under sparse activity)
    n = targets.shape[1]
    if torch.is_tensor(activity_prob):
        k_over_n = activity_prob.to(device=targets.device, dtype=targets.dtype)
    else:
        k_over_n = torch.as_tensor(float(activity_prob), device=targets.device, dtype=targets.dtype)
    k_over_n = k_over_n.clamp(1e-4, 1.0 - 1e-4)
    pos_w = 1.0 - k_over_n
    neg_w = k_over_n

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    sample_weight = targets * pos_w + (1.0 - targets) * neg_w
    return (2.0 / float(n)) * (sample_weight * bce).sum(dim=1).mean()
