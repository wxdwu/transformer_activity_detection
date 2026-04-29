from __future__ import annotations

import torch


@torch.no_grad()
def pm_pf_at_threshold(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> tuple[float, float]:
    pred = (probs > threshold).to(labels.dtype)
    positives = labels.sum().item()
    negatives = (1.0 - labels).sum().item()

    if positives <= 0:
        pm = 0.0
    else:
        hit = (pred * labels).sum().item()
        pm = 1.0 - (hit / positives)

    if negatives <= 0:
        pf = 0.0
    else:
        fa = (pred * (1.0 - labels)).sum().item()
        pf = fa / negatives

    return pm, pf


@torch.no_grad()
def pm_pf_curve(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_thresholds: int = 41,
) -> list[tuple[float, float, float]]:
    thresholds = torch.linspace(0.0, 1.0, num_thresholds, device=probs.device)
    curve = []
    for th in thresholds:
        pm, pf = pm_pf_at_threshold(probs, labels, float(th.item()))
        curve.append((float(th.item()), pm, pf))
    return curve
