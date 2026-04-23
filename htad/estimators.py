from __future__ import annotations

from typing import Sequence

import torch


@torch.no_grad()
def active_indices_from_probs(
    probs: torch.Tensor,
    threshold: float = 0.5,
    topk: int | None = None,
) -> list[torch.Tensor]:
    """
    Convert activity probabilities to active user indices.

    Args:
        probs:
            [N] or [B, N], probability of each user being active.
        threshold:
            Active decision threshold.
        topk:
            Optional: keep only top-k largest probabilities per sample.

    Returns:
        A list of index tensors. For [N], returns length-1 list.
    """
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    if probs.dim() != 2:
        raise ValueError(f"`probs` must be [N] or [B, N], got shape={tuple(probs.shape)}")

    bsz, n = probs.shape
    out: list[torch.Tensor] = []
    for b in range(bsz):
        p = probs[b]
        if topk is not None:
            k = max(1, min(int(topk), n))
            idx = torch.topk(p, k=k, largest=True).indices
        else:
            idx = torch.nonzero(p > threshold, as_tuple=False).flatten()
        out.append(idx)
    return out


def lmmse_channel_estimate(
    y: torch.Tensor,
    b: torch.Tensor,
    active_indices: torch.Tensor | Sequence[int],
    noise_var: float,
    channel_var: float = 1.0,
    reg_eps: float = 1e-18,
    return_active_only: bool = False,
) -> torch.Tensor:
    """
    LMMSE channel estimation under:
        Y = B_S H_S + W,
        H_S ~ CN(0, channel_var * I), W ~ CN(0, noise_var * I)

    Args:
        y:
            Received signal matrix, shape [Lp, M], complex tensor.
        b:
            Pilot matrix for all users, shape [Lp, N], complex tensor.
        active_indices:
            Active user indices (predicted by detector).
        noise_var:
            Noise variance sigma^2.
        channel_var:
            Per-user channel variance in prior covariance.
        reg_eps:
            Small diagonal regularizer for numerical stability.
        return_active_only:
            If True, return [K, M] estimated channels of active users only.
            Else return [N, M], inactive users filled with zeros.

    Returns:
        Estimated channel matrix.
    """
    if y.dim() != 2 or b.dim() != 2:
        raise ValueError(f"`y` and `b` must be rank-2, got y={tuple(y.shape)}, b={tuple(b.shape)}")
    if y.shape[0] != b.shape[0]:
        raise ValueError(f"Pilot length mismatch: y.shape[0]={y.shape[0]} vs b.shape[0]={b.shape[0]}")
    if not torch.is_complex(y) or not torch.is_complex(b):
        raise TypeError("`y` and `b` must be complex tensors.")

    lp, n = b.shape
    m = y.shape[1]
    device = y.device
    dtype = y.dtype

    idx = torch.as_tensor(active_indices, device=device, dtype=torch.long).flatten()
    if idx.numel() == 0:
        if return_active_only:
            return torch.zeros((0, m), dtype=dtype, device=device)
        return torch.zeros((n, m), dtype=dtype, device=device)

    b_s = b[:, idx]  # [Lp, K]
    i_lp = torch.eye(lp, dtype=dtype, device=device)

    # A = channel_var * B_S B_S^H + noise_var * I
    a = channel_var * (b_s @ b_s.conj().transpose(-1, -2)) + noise_var * i_lp
    if reg_eps > 0.0:
        a = a + reg_eps * i_lp
    # Solve A X = Y
    x = torch.linalg.solve(a, y)  # [Lp, M]
    # H_hat = channel_var * B_S^H A^{-1} Y
    h_s_hat = channel_var * (b_s.conj().transpose(-1, -2) @ x)  # [K, M]

    if return_active_only:
        return h_s_hat

    h_hat = torch.zeros((n, m), dtype=dtype, device=device)
    h_hat[idx] = h_s_hat
    return h_hat


def lmmse_formula_estimate(
    y: torch.Tensor,
    a: torch.Tensor,
    rx: torch.Tensor | float | None = None,
    rn: torch.Tensor | float | None = None,
    reg_eps: float = 1e-18,
) -> torch.Tensor:
    """
    Generic LMMSE in the form:
        x_hat = R_x A^H (A R_x A^H + R_n)^(-1) y

    Args:
        y:
            Observation, shape [m, t], complex.
        a:
            System matrix A, shape [m, k], complex.
        rx:
            Signal covariance R_x, shape [k, k] or scalar. If None -> I.
        rn:
            Noise covariance R_n, shape [m, m] or scalar. If None -> I.
        reg_eps:
            Small diagonal regularizer for numerical stability.
    """
    if y.dim() != 2 or a.dim() != 2:
        raise ValueError(f"`y` and `a` must be rank-2, got y={tuple(y.shape)}, a={tuple(a.shape)}")
    if y.shape[0] != a.shape[0]:
        raise ValueError(f"Dimension mismatch: y.shape[0]={y.shape[0]} vs a.shape[0]={a.shape[0]}")
    if not torch.is_complex(y) or not torch.is_complex(a):
        raise TypeError("`y` and `a` must be complex tensors.")

    m, k = a.shape
    device = a.device
    dtype = a.dtype

    i_k = torch.eye(k, dtype=dtype, device=device)
    i_m = torch.eye(m, dtype=dtype, device=device)

    if rx is None:
        rx_m = i_k
    elif isinstance(rx, (float, int)):
        rx_m = float(rx) * i_k
    else:
        rx_m = rx.to(device=device, dtype=dtype)

    if rn is None:
        rn_m = i_m
    elif isinstance(rn, (float, int)):
        rn_m = float(rn) * i_m
    else:
        rn_m = rn.to(device=device, dtype=dtype)

    s = a @ rx_m @ a.conj().transpose(-1, -2) + rn_m
    if reg_eps > 0.0:
        s = s + reg_eps * i_m
    z = torch.linalg.solve(s, y)  # [m, t]
    x_hat = rx_m @ a.conj().transpose(-1, -2) @ z  # [k, t]
    return x_hat


def ls_channel_estimate(
    y: torch.Tensor,
    b: torch.Tensor,
    active_indices: torch.Tensor | Sequence[int],
    reg_eps: float = 1e-18,
    return_active_only: bool = False,
) -> torch.Tensor:
    """
    Least-squares (ridge-regularized) channel estimation under:
        Y = B_S H_S + W

    Uses normal equations:
        H_hat = (B_S^H B_S + reg_eps I)^{-1} B_S^H Y

    Args:
        y:
            Received signal matrix, shape [Lp, M], complex tensor.
        b:
            Pilot matrix for all users, shape [Lp, N], complex tensor.
        active_indices:
            Active user indices (predicted by detector).
        reg_eps:
            Small diagonal regularizer for numerical stability.
        return_active_only:
            If True, return [K, M] estimated channels of active users only.
            Else return [N, M], inactive users filled with zeros.

    Returns:
        Estimated channel matrix.
    """
    if y.dim() != 2 or b.dim() != 2:
        raise ValueError(f"`y` and `b` must be rank-2, got y={tuple(y.shape)}, b={tuple(b.shape)}")
    if y.shape[0] != b.shape[0]:
        raise ValueError(f"Pilot length mismatch: y.shape[0]={y.shape[0]} vs b.shape[0]={b.shape[0]}")
    if not torch.is_complex(y) or not torch.is_complex(b):
        raise TypeError("`y` and `b` must be complex tensors.")

    _, n = b.shape
    m = y.shape[1]
    device = y.device
    dtype = y.dtype

    idx = torch.as_tensor(active_indices, device=device, dtype=torch.long).flatten()
    if idx.numel() == 0:
        if return_active_only:
            return torch.zeros((0, m), dtype=dtype, device=device)
        return torch.zeros((n, m), dtype=dtype, device=device)

    b_s = b[:, idx]  # [Lp, K]
    k = b_s.shape[1]
    i_k = torch.eye(k, dtype=dtype, device=device)

    gram = b_s.conj().transpose(-1, -2) @ b_s  # [K, K]
    rhs = b_s.conj().transpose(-1, -2) @ y  # [K, M]
    if reg_eps > 0.0:
        gram = gram + reg_eps * i_k
    h_s_hat = torch.linalg.solve(gram, rhs)  # [K, M]

    if return_active_only:
        return h_s_hat

    h_hat = torch.zeros((n, m), dtype=dtype, device=device)
    h_hat[idx] = h_s_hat
    return h_hat


def ls_min_norm_estimate(
    y: torch.Tensor,
    a: torch.Tensor,
    reg_eps: float = 1e-18,
) -> torch.Tensor:
    """
    Minimum-norm LS for underdetermined / ill-conditioned systems:
        x* = A^H (A A^H + eps I)^(-1) y
    """
    if y.dim() != 2 or a.dim() != 2:
        raise ValueError(f"`y` and `a` must be rank-2, got y={tuple(y.shape)}, a={tuple(a.shape)}")
    if y.shape[0] != a.shape[0]:
        raise ValueError(f"Dimension mismatch: y.shape[0]={y.shape[0]} vs a.shape[0]={a.shape[0]}")
    if not torch.is_complex(y) or not torch.is_complex(a):
        raise TypeError("`y` and `a` must be complex tensors.")

    m = a.shape[0]
    i_m = torch.eye(m, dtype=a.dtype, device=a.device)
    gram = a @ a.conj().transpose(-1, -2)
    if reg_eps > 0.0:
        gram = gram + reg_eps * i_m
    z = torch.linalg.solve(gram, y)
    return a.conj().transpose(-1, -2) @ z


def thresh_prime_thresh_complex_gaussian(
    y: torch.Tensor,
    sigma: float | torch.Tensor,
    lambda_vec: torch.Tensor,
    p_ls: torch.Tensor,
    lambda_floor: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    AMP denoiser with Bernoulli-Gaussian prior (complex), where each user has
    its own activity probability lambda_n.

    Args:
        y:
            Effective noisy input, shape [N, M], complex.
        sigma:
            Effective noise std (tau).
        lambda_vec:
            Per-user activity probability, shape [N], real in [0, 1].
        p_ls:
            Per-user Gaussian variance parameter, shape [N], real > 0.
        lambda_floor:
            Clamp for numerical stability.

    Returns:
        eta:
            Denoised estimate, shape [N, M], complex.
        eta_prime_avg:
            Average Jacobian term, shape [M, M], complex.
    """
    if y.dim() != 2 or not torch.is_complex(y):
        raise ValueError(f"`y` must be complex [N, M], got shape={tuple(y.shape)}")
    n, m = y.shape
    device = y.device
    real_dtype = y.real.dtype

    lam = lambda_vec.to(device=device, dtype=real_dtype).flatten()
    pls = p_ls.to(device=device, dtype=real_dtype).flatten()
    if lam.numel() != n or pls.numel() != n:
        raise ValueError(
            f"Size mismatch: y has N={n}, lambda_vec has {lam.numel()}, p_ls has {pls.numel()}"
        )
    lam = lam.clamp(lambda_floor, 1.0 - lambda_floor)
    pls = pls.clamp_min(1e-20)

    sigma_t = torch.as_tensor(sigma, device=device, dtype=real_dtype).clamp_min(1e-12)
    sigma2 = sigma_t * sigma_t

    a = pls / (pls + sigma2)
    c = pls / (sigma2 * (pls + sigma2))
    coeff0 = (a * a) / sigma2

    row_norm2 = torch.sum(torch.abs(y) ** 2, dim=1)  # [N]
    log_b = torch.log1p(-lam) - torch.log(lam) + float(m) * torch.log((pls + sigma2) / sigma2)
    log_t0 = log_b - c * row_norm2
    s = torch.sigmoid(log_t0)  # s = t0/(1+t0), numerically stable
    one_minus_s = 1.0 - s
    coeff1 = a * one_minus_s  # a/(1+t0)
    t0_over_t2 = s * one_minus_s  # t0/(1+t0)^2

    eta = coeff1.to(dtype=y.dtype).unsqueeze(-1) * y

    eye_m = torch.eye(m, device=device, dtype=y.dtype)
    eta_prime_avg = torch.zeros((m, m), device=device, dtype=y.dtype)
    for i in range(n):
        yyh = y[i].conj().unsqueeze(-1) @ y[i].unsqueeze(0)  # [M, M]
        scale = (coeff0[i] * t0_over_t2[i]).to(dtype=y.dtype)
        eta_prime_mtx = coeff1[i].to(dtype=y.dtype) * eye_m + scale * yyh
        eta_prime_avg = eta_prime_avg + eta_prime_mtx
    eta_prime_avg = eta_prime_avg / float(n)
    return eta, eta_prime_avg


def noisy_camp_mmse(
    a: torch.Tensor,
    y: torch.Tensor,
    lambda_vec: torch.Tensor,
    fading: torch.Tensor,
    sigma_w: float,
    max_iters: int = 10,
    x_true: torch.Tensor | None = None,
    lambda_floor: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Complex AMP-MMSE with per-user activity probabilities.

    Model:
        y = A x + w
        x_n ~ (1-lambda_n) delta_0 + lambda_n CN(0, fading_n)

    Args:
        a:
            Sensing matrix A, shape [L, N], complex.
        y:
            Observation, shape [L, M], complex.
        lambda_vec:
            Per-user activity probabilities, shape [N].
        fading:
            Per-user variance parameter (pLS), shape [N].
        sigma_w:
            Noise std.
        max_iters:
            AMP iterations.
        x_true:
            Optional oracle x for MSE tracking, shape [N, M].
        lambda_floor:
            Clamp for lambda stability.

    Returns:
        xnoise:
            A^H z + x from last iteration, shape [N, M].
        x:
            Estimated x, shape [N, M].
        mse:
            Iteration MSE, shape [max_iters].
        tau_real:
            State evolution using true mse, shape [max_iters + 1].
        tau_est:
            Residual-estimated tau, shape [max_iters + 1].
    """
    if a.dim() != 2 or y.dim() != 2 or not torch.is_complex(a) or not torch.is_complex(y):
        raise ValueError("`a` and `y` must be complex rank-2 tensors.")
    l, n = a.shape
    if y.shape[0] != l:
        raise ValueError(f"Dimension mismatch: a is [{l}, {n}] but y is {tuple(y.shape)}")
    m = y.shape[1]
    device = y.device
    dtype = y.dtype
    real_dtype = y.real.dtype

    z = y.clone()
    x = torch.zeros((n, m), dtype=dtype, device=device)

    tau = torch.sqrt(torch.sum(torch.abs(y) ** 2) / float(m * l)).to(dtype=real_dtype)
    mse = torch.zeros((max_iters,), dtype=real_dtype, device=device)
    tau_real = torch.zeros((max_iters + 1,), dtype=real_dtype, device=device)
    tau_est = torch.zeros((max_iters + 1,), dtype=real_dtype, device=device)
    tau_real[0] = tau
    tau_est[0] = tau

    for it in range(max_iters):
        inp = a.conj().transpose(-1, -2) @ z + x
        x, avg_xprime = thresh_prime_thresh_complex_gaussian(
            y=inp,
            sigma=tau,
            lambda_vec=lambda_vec,
            p_ls=fading,
            lambda_floor=lambda_floor,
        )

        if x_true is not None:
            mse[it] = torch.mean(torch.abs(x - x_true) ** 2).to(dtype=real_dtype)
            tau_real[it + 1] = torch.sqrt(
                torch.tensor(sigma_w * sigma_w, device=device, dtype=real_dtype)
                + (float(n) / float(l)) * mse[it]
            )
        else:
            tau_real[it + 1] = tau_real[it]

        z = y - a @ x + (float(n) / float(l)) * (z @ avg_xprime)
        tau = torch.sqrt(torch.sum(torch.abs(z) ** 2) / float(m * l)).to(dtype=real_dtype)
        tau_est[it + 1] = tau

    xnoise = a.conj().transpose(-1, -2) @ z + x
    return xnoise, x, mse, tau_real, tau_est
