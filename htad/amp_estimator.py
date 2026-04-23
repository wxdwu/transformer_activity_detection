from __future__ import annotations

import torch


def _thresh_prime_thresh_complex_gaussian_matlab(
    y: torch.Tensor,
    sigma: torch.Tensor,
    lambda_vec: torch.Tensor,
    p_ls: torch.Tensor,
    lambda_floor: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MATLAB `threshPrimeThreshComplexGaussian` equivalent, extended from scalar
    lambda to per-user lambda_vec.
    """
    if y.dim() != 2 or not torch.is_complex(y):
        raise ValueError(f"`y` must be complex [N, M], got shape={tuple(y.shape)}")
    n, m = y.shape
    device = y.device
    dtype = y.dtype
    rtype = y.real.dtype

    lam = lambda_vec.to(device=device, dtype=rtype).flatten()
    pls = p_ls.to(device=device, dtype=rtype).flatten().clamp_min(1e-20)
    if lam.numel() != n or pls.numel() != n:
        raise ValueError("`lambda_vec` and `p_ls` must have length N.")
    lam = lam.clamp(0.0, 1.0)

    sigma2 = sigma.to(dtype=rtype).clamp_min(1e-12) ** 2
    a = pls / (pls + sigma2)
    c = pls / (sigma2 * (pls + sigma2))
    coeff0 = (a * a) / sigma2

    tiny = torch.tensor(lambda_floor, dtype=rtype, device=device)
    is_zero = lam <= tiny
    is_one = lam >= (1.0 - tiny)
    is_mid = ~(is_zero | is_one)

    coeff1 = torch.zeros((n,), dtype=rtype, device=device)
    t0_over_t2 = torch.zeros((n,), dtype=rtype, device=device)

    coeff1[is_one] = a[is_one]
    if torch.any(is_mid):
        lam_mid = lam[is_mid].clamp(lambda_floor, 1.0 - lambda_floor)
        a_mid = a[is_mid]
        c_mid = c[is_mid]
        row_norm2_mid = torch.sum(torch.abs(y[is_mid]) ** 2, dim=1)
        # t0 = b*exp(-c*||y_n||^2), computed in log-domain for stability.
        log_b_mid = torch.log1p(-lam_mid) - torch.log(lam_mid) + float(m) * torch.log((pls[is_mid] + sigma2) / sigma2)
        log_t0_mid = log_b_mid - c_mid * row_norm2_mid
        s_mid = torch.sigmoid(log_t0_mid)  # t0/(1+t0)
        coeff1[is_mid] = a_mid * (1.0 - s_mid)  # a/(1+t0)
        t0_over_t2[is_mid] = s_mid * (1.0 - s_mid)  # t0/(1+t0)^2

    eta = coeff1.to(dtype=dtype).unsqueeze(-1) * y
    eta_prime_avg = torch.zeros((m, m), dtype=dtype, device=device)
    eye_m = torch.eye(m, dtype=dtype, device=device)
    for i in range(n):
        if is_zero[i]:
            continue
        if is_one[i]:
            eta_prime_avg = eta_prime_avg + coeff1[i].to(dtype=dtype) * eye_m
            continue
        yyh = y[i].conj().unsqueeze(-1) @ y[i].unsqueeze(0)
        eta_prime_mtx = coeff1[i].to(dtype=dtype) * eye_m + (coeff0[i] * t0_over_t2[i]).to(dtype=dtype) * yyh
        eta_prime_avg = eta_prime_avg + eta_prime_mtx
    eta_prime_avg = eta_prime_avg / float(n)
    return eta, eta_prime_avg


def noisy_camp_mmse_matlab(
    a: torch.Tensor,
    y: torch.Tensor,
    lambda_vec: torch.Tensor,
    fading: torch.Tensor,
    sigma_w: float,
    max_iters: int = 10,
    x_true: torch.Tensor | None = None,
    lambda_floor: float = 1e-6,
    damping: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MATLAB `noisyCAMPmmseforKLS` equivalent with per-user lambda.
    """
    if a.dim() != 2 or y.dim() != 2 or not torch.is_complex(a) or not torch.is_complex(y):
        raise ValueError("`a` and `y` must be complex rank-2 tensors.")
    l, n = a.shape
    if y.shape[0] != l:
        raise ValueError(f"Dimension mismatch: a is [{l}, {n}] but y is {tuple(y.shape)}")
    m = y.shape[1]
    device = y.device
    dtype = y.dtype
    rtype = y.real.dtype

    z = y.clone()
    x = torch.zeros((n, m), dtype=dtype, device=device)
    tau = torch.sqrt(torch.sum(torch.abs(y) ** 2) / float(m * l)).to(dtype=rtype)

    mse = torch.zeros((max_iters,), dtype=rtype, device=device)
    tau_real = torch.zeros((max_iters + 1,), dtype=rtype, device=device)
    tau_est = torch.zeros((max_iters + 1,), dtype=rtype, device=device)
    tau_real[0] = tau
    tau_est[0] = tau
    damp = float(max(0.0, min(1.0, damping)))

    for it in range(max_iters):
        inp = a.conj().transpose(-1, -2) @ z + x
        x_new, avg_xprime = _thresh_prime_thresh_complex_gaussian_matlab(
            y=inp,
            sigma=tau,
            lambda_vec=lambda_vec,
            p_ls=fading,
            lambda_floor=lambda_floor,
        )
        x = (1.0 - damp) * x + damp * x_new
        if x_true is not None:
            mse[it] = torch.mean(torch.abs(x - x_true) ** 2).to(dtype=rtype)
            tau_real[it + 1] = torch.sqrt(
                torch.tensor(float(sigma_w) ** 2, dtype=rtype, device=device) + (float(n) / float(l)) * mse[it]
            )
        else:
            tau_real[it + 1] = tau_real[it]

        z_new = y - a @ x + (float(n) / float(l)) * (z @ avg_xprime)
        z = (1.0 - damp) * z + damp * z_new
        tau = torch.sqrt(torch.sum(torch.abs(z) ** 2) / float(m * l)).to(dtype=rtype)
        tau_est[it + 1] = tau

    xnoise = a.conj().transpose(-1, -2) @ z + x
    return xnoise, x, mse, tau_real, tau_est


def camp_mmse_channel_estimate(
    y: torch.Tensor,
    b: torch.Tensor,
    probs: torch.Tensor,
    fading: torch.Tensor,
    noise_var: float,
    max_iters: int = 12,
    lambda_floor: float = 1e-6,
    normalize_columns: bool = True,
    col_norm_eps: float = 1e-12,
    prob_calib: str = "sigmoid_center",
    prob_center: float = 0.5,
    prob_alpha: float = 12.0,
    damping: float = 0.7,
) -> torch.Tensor:
    """
    AMP channel estimate wrapper for Y = B X + W.

    To match AMP assumptions on sensing-matrix scale, optional column
    normalization is applied with exact variable transform:
        B = Bn * D,  Xn = D * X,  Y = Bn * Xn + W.
    """
    if y.dim() != 2 or b.dim() != 2 or not torch.is_complex(y) or not torch.is_complex(b):
        raise ValueError("`y` and `b` must be complex rank-2 tensors.")
    if y.shape[0] != b.shape[0]:
        raise ValueError(f"Dimension mismatch: y.shape[0]={y.shape[0]} vs b.shape[0]={b.shape[0]}")

    n = b.shape[1]
    lam_raw = probs.to(device=y.device, dtype=y.real.dtype).flatten()
    if prob_calib == "sigmoid_center":
        center = torch.tensor(prob_center, device=y.device, dtype=y.real.dtype)
        alpha = torch.tensor(prob_alpha, device=y.device, dtype=y.real.dtype)
        lam = torch.sigmoid(alpha * (lam_raw - center))
    elif prob_calib == "none":
        lam = lam_raw
    else:
        raise ValueError(f"Unknown prob_calib: {prob_calib}")
    pls = fading.to(device=y.device, dtype=y.real.dtype).flatten()
    if lam.numel() != n or pls.numel() != n:
        raise ValueError("`probs` and `fading` must have length N.")

    if normalize_columns:
        d = torch.linalg.norm(b, dim=0).to(dtype=y.real.dtype).clamp_min(col_norm_eps)
        a_use = b / d.to(dtype=b.dtype).unsqueeze(0)
        fading_use = pls * (d * d)
        _, x_norm, _, _, _ = noisy_camp_mmse_matlab(
            a=a_use,
            y=y,
            lambda_vec=lam,
            fading=fading_use,
            sigma_w=float(noise_var) ** 0.5,
            max_iters=max_iters,
            x_true=None,
            lambda_floor=lambda_floor,
            damping=damping,
        )
        x_hat = x_norm / d.to(dtype=x_norm.dtype).unsqueeze(-1)
        return x_hat

    _, x_hat, _, _, _ = noisy_camp_mmse_matlab(
        a=b,
        y=y,
        lambda_vec=lam,
        fading=pls,
        sigma_w=float(noise_var) ** 0.5,
        max_iters=max_iters,
        x_true=None,
        lambda_floor=lambda_floor,
        damping=damping,
    )
    return x_hat
