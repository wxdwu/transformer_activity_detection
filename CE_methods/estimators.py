from __future__ import annotations

"""
传统活动检测后信道估计方法集合。

这个文件集中放所有非神经网络的后处理/信道估计方法：
- active_indices_from_probs: 把网络输出概率转成活跃用户索引。
- LMMSE: 根据检测出的活跃集合做线性最小均方误差信道估计。
- CAMP-MMSE: 基于伯努利-高斯先验的复数 AMP 信道估计。

统一约定：
N  = 用户数
L  = 导频长度
M  = 基站天线数
K  = 当前检测出的活跃用户数
Y  = 接收信号，形状 [L, M]
B  = 缩放后的导频矩阵，形状 [L, N]
S  = 未缩放导频矩阵，形状 [L, N]
H  = 信道矩阵，形状 [N, M]
"""

from typing import Sequence

import torch


@torch.no_grad()
def active_indices_from_probs(
    probs: torch.Tensor,
    threshold: float = 0.5,
    topk: int | None = None,
) -> list[torch.Tensor]:
    """
    将活跃概率转换成活跃用户索引。

    参数：
        probs:
            [N] 或 [B, N]，每个用户的活跃概率。
        threshold:
            活跃判决阈值。
        topk:
            可选。如果设置，则每个样本只保留概率最大的 top-k 个用户。

    返回：
        一个索引张量列表。若输入是 [N]，则返回长度为 1 的列表。
    """
    # 网络输出的 probs 可能是一条样本 [N]，也可能是一批样本 [B, N]。
    # 为了统一处理，单条样本先补一个批次维。
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    if probs.dim() != 2:
        raise ValueError(f"`probs` must be [N] or [B, N], got shape={tuple(probs.shape)}")

    bsz, n = probs.shape
    out: list[torch.Tensor] = []
    for b in range(bsz):
        p = probs[b]
        if topk is not None:
            # 前 k 大模式：不看阈值，直接取概率最大的 k 个用户作为活跃用户。
            k = max(1, min(int(topk), n))
            idx = torch.topk(p, k=k, largest=True).indices
        else:
            # 阈值模式：概率大于阈值的用户判为活跃。
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
    在如下模型下做 LMMSE 信道估计：
        Y = B_S H_S + W,
        H_S ~ CN(0, channel_var * I), W ~ CN(0, noise_var * I)

    参数：
        y:
            接收信号矩阵，形状 [L, M]，复数张量。
        b:
            所有用户的导频矩阵，形状 [L, N]，复数张量。
        active_indices:
            活跃用户索引，通常由检测器预测得到。
        noise_var:
            噪声方差 sigma^2。
        channel_var:
            每个用户的信道先验方差。
        reg_eps:
            很小的对角正则项，用于提升数值稳定性。
        return_active_only:
            若为 True，只返回活跃用户估计信道 [K, M]。
            否则返回完整 [N, M]，非活跃用户填 0。

    返回：
        估计得到的信道矩阵。
    """
    # 输入维度：
    # y: [L, M]，接收信号。
    # b: [L, N]，所有用户的导频矩阵。
    # active_indices: [K]，当前认为活跃的 K 个用户。
    # 目标：估计 H_hat: [N, M]，非活跃用户行填 0。
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
        # 如果没有活跃用户，直接返回全 0。
        if return_active_only:
            return torch.zeros((0, m), dtype=dtype, device=device)
        return torch.zeros((n, m), dtype=dtype, device=device)

    b_s = b[:, idx]  # [Lp, K]
    i_lp = torch.eye(lp, dtype=dtype, device=device)

    # A = channel_var * B_S B_S^H + noise_var * I
    # 这是 LMMSE 公式中的观测协方差：
    # R_y = B_S R_h B_S^H + R_w
    # 其中 R_h = channel_var * I，R_w = noise_var * I。
    a = channel_var * (b_s @ b_s.conj().transpose(-1, -2)) + noise_var * i_lp
    if reg_eps > 0.0:
        a = a + reg_eps * i_lp
    # 求解 A X = Y。
    # 等价于 X = A^{-1} Y，但用 torch.linalg.solve 更稳定，不显式求逆。
    x = torch.linalg.solve(a, y)  # [Lp, M]
    # H_hat = channel_var * B_S^H A^{-1} Y
    # LMMSE 闭式解：H_hat = R_h B_S^H R_y^{-1} Y。
    h_s_hat = channel_var * (b_s.conj().transpose(-1, -2) @ x)  # [K, M]

    if return_active_only:
        return h_s_hat

    h_hat = torch.zeros((n, m), dtype=dtype, device=device)
    # 把 K 个活跃用户的估计值写回完整 [N, M] 矩阵。
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
    通用形式的 LMMSE：
        x_hat = R_x A^H (A R_x A^H + R_n)^(-1) y

    参数：
        y:
            观测矩阵，形状 [m, t]，复数。
        a:
            系统矩阵 A，形状 [m, k]，复数。
        rx:
            信号协方差 R_x，形状 [k, k] 或标量。若为 None，则使用 I。
        rn:
            噪声协方差 R_n，形状 [m, m] 或标量。若为 None，则使用 I。
        reg_eps:
            很小的对角正则项，用于提升数值稳定性。
    """
    # 这是通用形式的 LMMSE：
    # y = A x + n
    # x ~ CN(0, R_x), n ~ CN(0, R_n)
    # x_hat = R_x A^H (A R_x A^H + R_n)^(-1) y
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
        # 默认 R_x = I。
        rx_m = i_k
    elif isinstance(rx, (float, int)):
        # 标量先验方差：R_x = rx * I。
        rx_m = float(rx) * i_k
    else:
        rx_m = rx.to(device=device, dtype=dtype)

    if rn is None:
        # 默认 R_n = I。
        rn_m = i_m
    elif isinstance(rn, (float, int)):
        # 标量噪声方差：R_n = rn * I。
        rn_m = float(rn) * i_m
    else:
        rn_m = rn.to(device=device, dtype=dtype)

    s = a @ rx_m @ a.conj().transpose(-1, -2) + rn_m
    if reg_eps > 0.0:
        s = s + reg_eps * i_m
    z = torch.linalg.solve(s, y)  # [m, t]
    # 不显式计算 s^{-1}，而是先解 z = s^{-1} y。
    x_hat = rx_m @ a.conj().transpose(-1, -2) @ z  # [k, t]
    return x_hat


def thresh_prime_thresh_complex_gaussian(
    y: torch.Tensor,
    sigma: float | torch.Tensor,
    lambda_vec: torch.Tensor,
    p_ls: torch.Tensor,
    lambda_floor: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    AMP 中使用的复数 Bernoulli-Gaussian 先验 MMSE 去噪器。
    每个用户都有自己的活跃概率 lambda_n。

    参数：
        y:
            等效带噪输入，形状 [N, M]，复数。
        sigma:
            等效噪声标准差 tau。
        lambda_vec:
            每个用户的活跃概率，形状 [N]，实数，范围 [0, 1]。
        p_ls:
            每个用户的高斯先验方差参数，形状 [N]，正实数。
        lambda_floor:
            用于数值稳定的概率截断下限。

    返回：
        eta:
            去噪后的估计，形状 [N, M]，复数。
        eta_prime_avg:
            平均雅可比矩阵/导数项，形状 [M, M]，复数，用于 Onsager 修正。
    """
    # 这是 AMP 的 MMSE 去噪器。
    # 输入 y 可理解为“当前迭代下每个用户信道的带噪观测”：
    # y_n = x_n + 等效噪声。
    # 先验是伯努利-高斯分布：
    # x_n = 0，概率 1-lambda_n
    # x_n ~ CN(0, p_ls_n)，概率 lambda_n
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

    # sigma 是 AMP 当前估计的等效噪声标准差 tau。
    sigma_t = torch.as_tensor(sigma, device=device, dtype=real_dtype).clamp_min(1e-12)
    sigma2 = sigma_t * sigma_t

    a = pls / (pls + sigma2)
    c = pls / (sigma2 * (pls + sigma2))
    coeff0 = (a * a) / sigma2

    row_norm2 = torch.sum(torch.abs(y) ** 2, dim=1)  # [N]
    # log_b/log_t0 是后验活跃概率推导中的中间量。
    # 用 log 域计算是为了防止 exp 溢出。
    log_b = torch.log1p(-lam) - torch.log(lam) + float(m) * torch.log((pls + sigma2) / sigma2)
    log_t0 = log_b - c * row_norm2
    s = torch.sigmoid(log_t0)  # s = t0/(1+t0)，数值稳定写法。
    one_minus_s = 1.0 - s
    coeff1 = a * one_minus_s  # a/(1+t0)
    t0_over_t2 = s * one_minus_s  # t0/(1+t0)^2

    eta = coeff1.to(dtype=y.dtype).unsqueeze(-1) * y

    # eta_prime_avg 是 Onsager 修正项需要的平均导数/雅可比矩阵。
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
    带逐用户活跃概率的复数 AMP-MMSE。

    模型：
        y = A x + w
        x_n ~ (1-lambda_n) delta_0 + lambda_n CN(0, fading_n)

    参数：
        a:
            感知矩阵 A，形状 [L, N]，复数。
        y:
            观测矩阵，形状 [L, M]，复数。
        lambda_vec:
            每个用户的活跃概率，形状 [N]。
        fading:
            每个用户的先验方差参数，形状 [N]。
        sigma_w:
            噪声标准差。
        max_iters:
            AMP 迭代次数。
        x_true:
            可选的真实 x，用于跟踪 MSE，形状 [N, M]。
        lambda_floor:
            用于数值稳定的 lambda 截断下限。

    返回：
        xnoise:
            最后一轮的 A^H z + x，形状 [N, M]。
        x:
            估计得到的 x，形状 [N, M]。
        mse:
            每轮迭代的 MSE，形状 [max_iters]。
        tau_real:
            使用真实 MSE 得到的状态演化 tau，形状 [max_iters + 1]。
        tau_est:
            由残差估计得到的 tau，形状 [max_iters + 1]。
    """
    # 这是较早的一版 CAMP-MMSE 实现，保留用于对照。
    # 当前 compare.py 默认调用的是后面的 noisy_camp_mmse_matlab/camp_mmse_channel_estimate。
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

    # tau 是 AMP 中的等效噪声标准差，初始由接收信号能量估计。
    tau = torch.sqrt(torch.sum(torch.abs(y) ** 2) / float(m * l)).to(dtype=real_dtype)
    mse = torch.zeros((max_iters,), dtype=real_dtype, device=device)
    tau_real = torch.zeros((max_iters + 1,), dtype=real_dtype, device=device)
    tau_est = torch.zeros((max_iters + 1,), dtype=real_dtype, device=device)
    tau_real[0] = tau
    tau_est[0] = tau

    for it in range(max_iters):
        # AMP 伪观测：A^H z + x。
        inp = a.conj().transpose(-1, -2) @ z + x
        # MMSE 去噪器：根据伯努利-高斯先验更新 x。
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

        # AMP 残差更新，最后一项是 Onsager 修正。
        z = y - a @ x + (float(n) / float(l)) * (z @ avg_xprime)
        tau = torch.sqrt(torch.sum(torch.abs(z) ** 2) / float(m * l)).to(dtype=real_dtype)
        tau_est[it + 1] = tau

    xnoise = a.conj().transpose(-1, -2) @ z + x
    return xnoise, x, mse, tau_real, tau_est


def _thresh_prime_thresh_complex_gaussian_matlab(
    y: torch.Tensor,
    sigma: torch.Tensor,
    lambda_vec: torch.Tensor,
    p_ls: torch.Tensor,
    lambda_floor: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MATLAB 函数 `threshPrimeThreshComplexGaussian` 的 PyTorch 对齐版本。
    这里从标量 lambda 扩展为逐用户 lambda_vec。
    """
    # 这是与 MATLAB 版本对齐的 MMSE 去噪器。
    # 相比上面的 thresh_prime_thresh_complex_gaussian，这里显式处理
    # lambda 接近 0 或 1 的边界情况，数值上更稳。
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
    # 三种情况分开处理：
    # lambda≈0：几乎必不活跃，eta 近似 0。
    # lambda≈1：几乎必活跃，退化为 Gaussian MMSE。
    # 中间值：正常伯努利-高斯后验。
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
        log_b_mid = torch.log1p(-lam_mid) - torch.log(lam_mid) + float(m) * torch.log((pls[is_mid] + sigma2) / sigma2)
        log_t0_mid = log_b_mid - c_mid * row_norm2_mid
        s_mid = torch.sigmoid(log_t0_mid)
        coeff1[is_mid] = a_mid * (1.0 - s_mid)
        t0_over_t2[is_mid] = s_mid * (1.0 - s_mid)

    eta = coeff1.to(dtype=dtype).unsqueeze(-1) * y
    # 计算平均导数，用于 CAMP 的 Onsager 修正项。
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
    MATLAB 函数 `noisyCAMPmmseforKLS` 的 PyTorch 对齐版本，
    支持逐用户活跃概率 lambda。
    """
    # CAMP 迭代主体，模型为：
    # y = A x + w
    # x 的每一行对应一个用户在 M 根天线上的信道向量。
    # lambda_vec 是每个用户活跃概率，fading 是每个用户的高斯先验方差。
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
    # tau 初值由接收信号均方能量估计。
    tau = torch.sqrt(torch.sum(torch.abs(y) ** 2) / float(m * l)).to(dtype=rtype)

    mse = torch.zeros((max_iters,), dtype=rtype, device=device)
    tau_real = torch.zeros((max_iters + 1,), dtype=rtype, device=device)
    tau_est = torch.zeros((max_iters + 1,), dtype=rtype, device=device)
    tau_real[0] = tau
    tau_est[0] = tau
    damp = float(max(0.0, min(1.0, damping)))

    for it in range(max_iters):
        # 1) 构造 AMP 伪观测。
        inp = a.conj().transpose(-1, -2) @ z + x
        # 2) MMSE 去噪，得到新的 x 估计和 Onsager 导数项。
        x_new, avg_xprime = _thresh_prime_thresh_complex_gaussian_matlab(
            y=inp,
            sigma=tau,
            lambda_vec=lambda_vec,
            p_ls=fading,
            lambda_floor=lambda_floor,
        )
        # 3) 使用阻尼，降低迭代震荡。
        x = (1.0 - damp) * x + damp * x_new
        if x_true is not None:
            mse[it] = torch.mean(torch.abs(x - x_true) ** 2).to(dtype=rtype)
            tau_real[it + 1] = torch.sqrt(
                torch.tensor(float(sigma_w) ** 2, dtype=rtype, device=device) + (float(n) / float(l)) * mse[it]
            )
        else:
            tau_real[it + 1] = tau_real[it]

        # 4) 残差更新，包含 Onsager 修正。
        z_new = y - a @ x + (float(n) / float(l)) * (z @ avg_xprime)
        # 5) 残差也做阻尼。
        z = (1.0 - damp) * z + damp * z_new
        # 6) 用新残差估计下一轮 tau。
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
    面向信道估计任务的 CAMP-MMSE 包装函数，模型为 Y = B X + W。
    """
    # 对外使用的 CAMP-MMSE 包装函数。
    # y: [L, M] 接收信号。
    # b: [L, N] 导频/感知矩阵。
    # probs: [N] 每个用户的活跃概率 lambda。
    # fading: [N] 每个用户的先验方差。
    # 返回 x_hat: [N, M]，每行是一个用户的信道估计。
    if y.dim() != 2 or b.dim() != 2 or not torch.is_complex(y) or not torch.is_complex(b):
        raise ValueError("`y` and `b` must be complex rank-2 tensors.")
    if y.shape[0] != b.shape[0]:
        raise ValueError(f"Dimension mismatch: y.shape[0]={y.shape[0]} vs b.shape[0]={b.shape[0]}")

    n = b.shape[1]
    lam_raw = probs.to(device=y.device, dtype=y.real.dtype).flatten()
    if prob_calib == "sigmoid_center":
        # 将网络输出概率再做一次 sigmoid 校准：
        # center 控制中心点，alpha 控制陡峭程度。
        center = torch.tensor(prob_center, device=y.device, dtype=y.real.dtype)
        alpha = torch.tensor(prob_alpha, device=y.device, dtype=y.real.dtype)
        lam = torch.sigmoid(alpha * (lam_raw - center))
    elif prob_calib == "none":
        # 不校准，直接把 probs 当作 lambda。
        lam = lam_raw
    else:
        raise ValueError(f"Unknown prob_calib: {prob_calib}")
    pls = fading.to(device=y.device, dtype=y.real.dtype).flatten()
    if lam.numel() != n or pls.numel() != n:
        raise ValueError("`probs` and `fading` must have length N.")

    if normalize_columns:
        # AMP 对感知矩阵列尺度较敏感，因此先做列归一化：
        # B = A_use * diag(d)
        # Y = B X = A_use * (diag(d) X)
        # 先估计 X_norm = diag(d) X，最后再除以 d 还原 X。
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
        # 反归一化，恢复原变量尺度。
        return x_norm / d.to(dtype=x_norm.dtype).unsqueeze(-1)

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


def thresh_prime_thresh_complex_gaussian_genie(
    y: torch.Tensor,
    sigma: float | torch.Tensor,
    lambda_vec: torch.Tensor,
    p_ls: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Python translation of AMP_Genie/threshPrimeThreshComplexGaussian_Genie.m.

    This intentionally follows the MATLAB loop-level logic: per-user lambda,
    Bernoulli-Gaussian MMSE shrinkage, and averaged Onsager derivative matrix.
    """
    # Same formulas as the MATLAB-loop version, but evaluated in log-domain.
    # Directly computing b * exp(-c||y||^2) can produce inf * 0 = nan.
    return _thresh_prime_thresh_complex_gaussian_matlab(
        y=y,
        sigma=torch.as_tensor(sigma, dtype=y.real.dtype, device=y.device),
        lambda_vec=lambda_vec,
        p_ls=p_ls,
        lambda_floor=1e-9,
    )


def camp_genie(
    a: torch.Tensor,
    y: torch.Tensor,
    xsig: torch.Tensor,
    max_iters: int,
    lambda_vec: torch.Tensor,
    fading: torch.Tensor,
    sigma_w: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Python translation of AMP_Genie/CAMP_Genie.m.

    The damping, lambda clipping, tau update, and MSE bookkeeping are kept
    aligned with the MATLAB script.
    """
    if a.dim() != 2 or y.dim() != 2 or xsig.dim() != 2:
        raise ValueError("`a`, `y`, and `xsig` must be rank-2 tensors.")
    if not torch.is_complex(a) or not torch.is_complex(y) or not torch.is_complex(xsig):
        raise TypeError("`a`, `y`, and `xsig` must be complex tensors.")

    n = a.shape[1]
    m = y.shape[1]
    l = y.shape[0]
    if a.shape[0] != l or xsig.shape != (n, m):
        raise ValueError("Dimension mismatch among `a`, `y`, and `xsig`.")

    device = y.device
    dtype = y.dtype
    rtype = y.real.dtype
    max_iters = int(max_iters)

    epsilon = torch.tensor(1e-9, dtype=rtype, device=device)
    lam = lambda_vec.to(device=device, dtype=rtype).flatten().clone()
    lam = torch.where(lam < epsilon, epsilon, lam)
    lam = torch.where(lam > 1.0 - epsilon, 1.0 - epsilon, lam)

    z = y.clone()
    x = torch.zeros((n, m), dtype=dtype, device=device)
    sum_gain = torch.sum(torch.abs(y) ** 2)
    tau = torch.sqrt(sum_gain) * torch.sqrt(torch.tensor(1.0 / float(m * l), dtype=rtype, device=device))

    mse = torch.zeros((max_iters,), dtype=rtype, device=device)
    tau_real = torch.zeros((max_iters + 1,), dtype=rtype, device=device)
    tau_est = torch.zeros((max_iters,), dtype=rtype, device=device)
    tau_real[0] = tau
    tau_est[0] = tau

    xold = torch.zeros((n, m), dtype=dtype, device=device)
    sigma_w_t = torch.tensor(float(sigma_w), dtype=rtype, device=device)

    for it in range(max_iters):
        input_to_denoiser = a.conj().transpose(-1, -2) @ z + x
        x_new, avgxprime = thresh_prime_thresh_complex_gaussian_genie(
            input_to_denoiser,
            tau,
            lam,
            fading,
        )

        if it == 0:
            x = x_new
        else:
            x = 0.95 * x_new + 0.05 * xold
        xold = x

        mse[it] = torch.sum(torch.abs(x - xsig) ** 2).to(dtype=rtype) / float(n * m)
        tau_real[it + 1] = torch.sqrt(sigma_w_t * sigma_w_t + float(n) / float(l) * mse[it])

        z = y - a @ x + float(n) / float(l) * (z @ avgxprime)
        sum_gain = torch.sum(torch.abs(z) ** 2)
        tau = torch.sqrt(sum_gain) * torch.sqrt(torch.tensor(1.0 / float(m * l), dtype=rtype, device=device))

        if it != max_iters - 1:
            tau_est[it + 1] = tau

    xnoise = a.conj().transpose(-1, -2) @ z + x
    return xnoise, x, mse, tau_real, tau_est


def camp_genie_from_active(
    a: torch.Tensor,
    y: torch.Tensor,
    xsig: torch.Tensor,
    active: torch.Tensor,
    max_iters: int,
    fading: torch.Tensor,
    sigma_w: float,
    epsilon: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run CAMP_Genie with true 0/1 activity converted to MATLAB-style probabilities."""
    lam = active.to(device=y.device, dtype=y.real.dtype).flatten()
    eps = torch.tensor(float(epsilon), dtype=y.real.dtype, device=y.device)
    lam = torch.where(lam == 0, eps, lam)
    lam = torch.where(lam == 1, 1.0 - eps, lam)
    return camp_genie(a, y, xsig, max_iters, lam, fading, sigma_w)
