from __future__ import annotations

"""
检测 + 信道估计对比脚本。

这个文件负责把“训练好的网络”和“传统信道估计方法”串起来：
1. 从 config.json 读取 CE_methods_compare 配置。
2. 加载 checkpoint 中训练好的 Transformer。
3. 生成测试数据，并用 Transformer 输出每个用户的活跃概率。
4. 根据概率得到预测活跃集合 pred_idx。
5. 按 CE_methods_compare.methods 选择 CAMP/LMMSE 等方法做信道估计。
6. 统计活动检测指标 PM/PF，以及信道估计指标 NMSE。

真正的传统方法实现放在 CE_methods/estimators.py；
本文件主要负责流程组织、方法分发和报告输出。
"""

import math
import sys
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CE_methods.estimators import (
    active_indices_from_probs,
    camp_mmse_channel_estimate,
    lmmse_channel_estimate,
    lmmse_formula_estimate,
)
from network.config import load_experiment_config, section_namespace
from network.data import ActivityDataGenerator, SystemConfig
from network.metrics import pm_pf_at_threshold
from network.model import build_model_from_config


CONFIG_PATH = "config.json"

METHOD_DESCRIPTIONS = {
    # key 必须和 config.json -> CE_methods_compare.methods 中的字符串一致。
    # value 只用于最终报告中的可读说明。
    "transformer+AMP": "transformer输出概率-CAMP",
    "CAMP": "CAMP",
    "detected+LMMSE": "transformer输出概率 + 阈值判断 + LMMSE",
    "oracle_LMMSE": "真实活跃索引-LMMSE",
    "oracle_AMP": "真实活跃用户对应概率 + Oracle-AMP",
    "lmmse_formula_detected": "LMMSE-formula + detected active-set",
}


def load_checkpoint_safely(path: Path, map_location: torch.device) -> dict:
    """Load checkpoint with safer PyTorch behavior when available."""
    # 新版 PyTorch 支持 weights_only=True，可以减少反序列化任意对象的风险。
    # 旧版 PyTorch 没有这个参数，所以用 TypeError 做兼容。
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def complex_nmse(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    """NMSE = ||est - target||^2 / ||target||^2 for complex tensors."""
    # est/target 一般是 [N, M]：
    # N 是用户数，M 是天线数。复数误差能量用 abs(.)^2 计算。
    num = torch.sum(torch.abs(est - target) ** 2)
    den = torch.sum(torch.abs(target) ** 2) + eps
    return float((num / den).item())


def nmse_db(nmse: float, floor_db: float = -300.0) -> float:
    """Convert linear NMSE to dB."""
    db = 10.0 * math.log10(max(float(nmse), 1e-30))
    return max(db, floor_db)


def parse_methods(raw_methods: list[str]) -> list[str]:
    """
    Validate config.json CE_methods_compare.methods.

    Supports either ["all"] or a list such as
    ["transformer+AMP", "CAMP", "detected+LMMSE"].
    """
    # 支持两种配置写法：
    #   "methods": ["CAMP", "detected+LMMSE"]
    #   "methods": ["CAMP,detected+LMMSE"]
    expanded: list[str] = []
    for item in raw_methods:
        expanded.extend(x.strip() for x in str(item).split(",") if x.strip())
    if len(expanded) == 1 and expanded[0].lower() == "all":
        return list(METHOD_DESCRIPTIONS.keys())

    unknown = [m for m in expanded if m not in METHOD_DESCRIPTIONS]
    if unknown:
        valid = ", ".join(METHOD_DESCRIPTIONS.keys())
        raise ValueError(f"Unknown methods in config.json: {unknown}. Valid methods: {valid}")

    methods: list[str] = []
    seen: set[str] = set()
    for method in expanded:
        if method not in seen:
            methods.append(method)
            seen.add(method)
    return methods


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> torch.nn.Module:
    """Rebuild the model from checkpoint config and load trained weights."""
    # checkpoint 里保存了训练时的网络结构配置。
    # 新 checkpoint 用 model_config；旧 checkpoint 可能只有 config。
    cfg = ckpt.get("model_config", ckpt["config"])
    model = build_model_from_config(cfg).to(device)
    state = ckpt["model_state"]
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        # Compatible with older checkpoints saved before Dropout changed FFN key names.
        # 旧模型中 FFN 第二个 Linear 是 .2.；加入 Dropout 后变成 .3.。
        # Dropout 没有参数，因此只需要重命名 state_dict key。
        if "ffn.ff_b.2." not in str(exc) and "ffn.ff_y.2." not in str(exc):
            raise
        remapped = {}
        for k, v in state.items():
            k2 = k.replace(".ffn.ff_b.2.", ".ffn.ff_b.3.")
            k2 = k2.replace(".ffn.ff_y.2.", ".ffn.ff_y.3.")
            remapped[k2] = v
        model.load_state_dict(remapped)
    model.eval()
    return model


def estimate_from_active_set(
    estimator: str,
    y_i: torch.Tensor,
    b_i: torch.Tensor,
    idx_i: torch.Tensor,
    n_users: int,
    noise_var: float,
    channel_var: float,
    beta_i: torch.Tensor | None = None,
    s_i: torch.Tensor | None = None,
    pg_i: torch.Tensor | None = None,
    probs_i: torch.Tensor | None = None,
    prior_mode: str = "unit",
    reg_eps: float = 1e-18,
    camp_iters: int = 12,
    camp_fading_mode: str = "unit",
    camp_lambda_floor: float = 1e-6,
    camp_prob_calib: str = "sigmoid_center",
    camp_prob_center: float = 0.5,
    camp_prob_alpha: float = 12.0,
    camp_damping: float = 0.7,
) -> torch.Tensor:
    """
    Estimate full [N, M] channel matrix from one active set.

    estimator controls the algorithm; inactive rows are returned as zeros.
    """
    # 这个函数处理“单条样本”的信道估计。
    # y_i: [L, M]，接收信号矩阵 Y。
    # b_i: [L, N]，缩放后的导频矩阵 B。
    # s_i: [L, N]，未缩放导频矩阵 S。
    # idx_i: 被认为活跃的用户索引。
    # 输出统一为 [N, M]，方便和真实信道矩阵 x_true 计算 NMSE。
    if estimator == "camp_mmse":
        if probs_i is None:
            raise ValueError("`probs_i` is required for estimator='camp_mmse'.")

        # CAMP 可以使用每个用户的 soft 活跃概率 probs_i。
        # 默认使用缩放后的导频 B；在 unscaled_gain 模式下改用 S，并用 pg 做先验方差。
        amp_a = b_i
        need_unscale = False
        sqrt_pg = None
        if camp_fading_mode == "unscaled_gain" and pg_i is not None:
            # pg_i 是发射功率和大尺度衰落合并后的等效功率因子。
            fading = (channel_var * pg_i).to(dtype=y_i.real.dtype).clamp_min(1e-20)
            if s_i is not None:
                amp_a = s_i
                sqrt_pg = torch.sqrt(pg_i.to(dtype=y_i.real.dtype).clamp_min(1e-20))
                need_unscale = True
        elif camp_fading_mode == "large_scale" and beta_i is not None:
            # 使用大尺度衰落 beta 作为每个用户的先验方差。
            fading = beta_i.to(dtype=y_i.real.dtype).clamp_min(1e-20)
        else:
            # unit 模式：所有用户用同一个 channel_var 作为先验方差。
            fading = torch.full_like(probs_i, float(channel_var), dtype=y_i.real.dtype)

        # 调用真正的 CAMP-MMSE 实现，位于 CE_methods/estimators.py。
        x_hat = camp_mmse_channel_estimate(
            y=y_i,
            b=amp_a,
            probs=probs_i.to(dtype=y_i.real.dtype),
            fading=fading,
            noise_var=float(noise_var),
            max_iters=int(camp_iters),
            lambda_floor=camp_lambda_floor,
            normalize_columns=True,
            prob_calib=camp_prob_calib,
            prob_center=camp_prob_center,
            prob_alpha=camp_prob_alpha,
            damping=camp_damping,
        )
        if need_unscale and sqrt_pg is not None:
            # 如果使用 S 作为观测矩阵，CAMP 估计出的变量包含 sqrt(pg) 缩放，
            # 这里除回去，使输出和真实 h 的尺度一致。
            x_hat = x_hat / sqrt_pg.to(dtype=x_hat.dtype).unsqueeze(-1)
        return x_hat

    if estimator == "lmmse":
        # LMMSE 只估计 idx_i 中的活跃用户，return_active_only=False 会放回 [N, M] 全矩阵。
        return lmmse_channel_estimate(
            y=y_i,
            b=b_i,
            active_indices=idx_i,
            noise_var=noise_var,
            channel_var=channel_var,
            reg_eps=reg_eps,
            return_active_only=False,
        )

    idx = torch.as_tensor(idx_i, device=y_i.device, dtype=torch.long).flatten()
    m = y_i.shape[1]
    out = torch.zeros((n_users, m), dtype=y_i.dtype, device=y_i.device)
    if idx.numel() == 0:
        # 没有检测到活跃用户时，返回全 0 信道估计。
        return out

    a_s = b_i[:, idx]
    if estimator == "lmmse_formula":
        # 显式公式版 LMMSE：
        # x_hat = R_x A^H (A R_x A^H + R_n)^(-1) y。
        if prior_mode == "unscaled_gain" and s_i is not None and pg_i is not None:
            s_s = s_i[:, idx]
            pg_s = pg_i[idx].to(dtype=y_i.real.dtype).clamp_min(1e-20)
            rx = torch.diag((channel_var * pg_s).to(dtype=y_i.dtype))
            u_s = lmmse_formula_estimate(y=y_i, a=s_s, rx=rx, rn=noise_var, reg_eps=reg_eps)
            h_s = u_s / torch.sqrt(pg_s).to(dtype=y_i.dtype).unsqueeze(-1)
        elif prior_mode == "large_scale" and beta_i is not None:
            beta_s = beta_i[idx].to(dtype=y_i.real.dtype).clamp_min(1e-20)
            rx = torch.diag(beta_s.to(dtype=y_i.dtype))
            h_s = lmmse_formula_estimate(y=y_i, a=a_s, rx=rx, rn=noise_var, reg_eps=reg_eps)
        else:
            h_s = lmmse_formula_estimate(y=y_i, a=a_s, rx=channel_var, rn=noise_var, reg_eps=reg_eps)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    # h_s 只包含活跃集合 K 个用户；这里写回完整 [N, M] 的对应行。
    out[idx] = h_s
    return out


def estimate_for_method(
    method: str,
    *,
    y_i: torch.Tensor,
    b_i: torch.Tensor,
    s_i: torch.Tensor,
    pg_i: torch.Tensor,
    beta_i: torch.Tensor,
    probs_i: torch.Tensor,
    label_i: torch.Tensor,
    pred_idx_i: torch.Tensor,
    true_idx_i: torch.Tensor,
    n_users: int,
    noise_var: float,
    args: Namespace,
) -> torch.Tensor:
    """Dispatch one configured method to CAMP/LMMSE."""
    # 这个函数把 config.json 中的方法名映射到具体估计器。
    # pred_idx_i 来自 Transformer 概率阈值判断。
    # true_idx_i 来自真实标签，用于 oracle 上界方法。
    common = dict(
        y_i=y_i,
        b_i=b_i,
        n_users=n_users,
        noise_var=noise_var,
        channel_var=args.channel_var,
        beta_i=beta_i,
        s_i=s_i,
        pg_i=pg_i,
        prior_mode=args.prior_mode,
        reg_eps=args.reg_eps,
        camp_fading_mode=args.camp_fading_mode,
        camp_lambda_floor=args.camp_lambda_floor,
        camp_prob_center=args.camp_prob_center,
        camp_prob_alpha=args.camp_prob_alpha,
        camp_damping=args.camp_damping,
    )

    if method == "transformer+AMP":
        # Transformer 输出的每用户概率 probs_i 直接作为 CAMP 的 soft lambda。
        return estimate_from_active_set(
            estimator="camp_mmse",
            idx_i=pred_idx_i,
            probs_i=probs_i,
            camp_prob_calib=args.camp_prob_calib,
            camp_iters=args.camp_iters,
            **common,
        )
    if method == "CAMP":
        # 不使用 Transformer 概率，所有用户使用固定活跃概率 camp_fixed_lambda。
        return estimate_from_active_set(
            estimator="camp_mmse",
            idx_i=pred_idx_i,
            probs_i=torch.full_like(probs_i, float(args.camp_fixed_lambda)),
            camp_prob_calib="none",
            camp_iters=args.camp_iters_fixed,
            **common,
        )
    if method == "oracle_AMP":
        # 使用真实 label 作为活跃概率，属于 oracle 参考上界。
        return estimate_from_active_set(
            estimator="camp_mmse",
            idx_i=true_idx_i,
            probs_i=label_i,
            camp_prob_calib="none",
            camp_iters=args.camp_iters,
            **common,
        )
    if method == "detected+LMMSE":
        # Transformer 概率 -> 阈值/TopK 得到 pred_idx_i -> LMMSE。
        return estimate_from_active_set(estimator="lmmse", idx_i=pred_idx_i, probs_i=None, camp_prob_calib="none", **common)
    if method == "oracle_LMMSE":
        # 真实活跃索引 true_idx_i -> LMMSE，用于区分“检测误差”和“信道估计误差”。
        return estimate_from_active_set(estimator="lmmse", idx_i=true_idx_i, probs_i=None, camp_prob_calib="none", **common)
    if method == "lmmse_formula_detected":
        return estimate_from_active_set(estimator="lmmse_formula", idx_i=pred_idx_i, probs_i=None, camp_prob_calib="none", **common)
    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def main() -> None:
    # 1) 读取 config.json 中 CE_methods_compare 部分。
    # 本脚本不再靠命令行传实验参数，后续主要改 config.json。
    exp_cfg = load_experiment_config(CONFIG_PATH)
    args = section_namespace(exp_cfg, "CE_methods_compare")
    methods = parse_methods(args.methods)

    # 2) 加载训练好的 checkpoint 和模型。
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint_safely(ckpt_path, map_location=device)
    model = build_model_from_ckpt(ckpt, device)
    # 测试数据的系统参数使用 checkpoint 里的 system_config，确保和训练设置一致。
    data_gen = ActivityDataGenerator(SystemConfig(**ckpt["system_config"]), device=device)

    all_probs = []
    all_labels = []
    nmse_by_method: dict[str, list[float]] = {m: [] for m in methods}
    topk = args.topk if args.topk > 0 else None

    for _ in range(args.num_test_batches):
        # return_raw=True 会返回信道估计需要的原始物理量：
        # y, b, s, pg, beta, h, noise_var。
        batch = data_gen.sample_batch(args.batch_size, return_raw=True)
        # Transformer 前向输出：
        # probs: [B, N]，每个样本中每个用户的活跃概率。
        _, probs = model(batch["x_b"], batch["x_y"])

        labels = batch["label"]
        # 预测活跃集合：由 Transformer 概率通过 threshold 或 topk 转换而来。
        pred_idx_list = active_indices_from_probs(probs, threshold=args.threshold, topk=topk)
        # 真实活跃集合：由标签 label 得到，供 oracle 方法使用。
        true_idx_list = active_indices_from_probs(labels, threshold=0.5, topk=None)
        noise_var = float(batch["noise_var"].item())

        for i in range(args.batch_size):
            # x_true 是完整真实信道矩阵 [N, M]：
            # 非活跃用户 label=0，对应行被置零；活跃用户保留真实 h。
            x_true = labels[i].unsqueeze(-1).to(batch["h"].dtype) * batch["h"][i]
            for method in methods:
                # 对同一条样本运行 config.json 里指定的每一种方法。
                x_hat = estimate_for_method(
                    method,
                    y_i=batch["y"][i],
                    b_i=batch["b"][i],
                    s_i=batch["s"][i],
                    pg_i=batch["pg"][i],
                    beta_i=batch["beta"][i],
                    probs_i=probs[i],
                    label_i=labels[i],
                    pred_idx_i=pred_idx_list[i],
                    true_idx_i=true_idx_list[i],
                    n_users=labels.shape[1],
                    noise_var=noise_var,
                    args=args,
                )
                nmse_by_method[method].append(complex_nmse(x_hat, x_true))

        all_probs.append(probs)
        all_labels.append(labels)

    # 3) 汇总所有 batch 的检测输出，计算 PM/PF。
    probs_all = torch.cat(all_probs, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    pm, pf = pm_pf_at_threshold(probs_all, labels_all, threshold=args.threshold)

    # 4) 组织报告内容，包括检测指标和每个方法的信道估计 NMSE。
    lines = [
        f"Checkpoint: {ckpt_path.resolve()}",
        f"Device: {device}",
        f"Methods: {', '.join(methods)}",
        f"Test batches: {args.num_test_batches}",
        f"Batch size: {args.batch_size}",
        f"Detection threshold: {args.threshold}",
        f"Top-k mode: {topk if topk is not None else 'off'}",
        f"Prior mode: {args.prior_mode}",
        f"CAMP iters: {args.camp_iters}",
        f"CAMP iters (fixed AMP): {args.camp_iters_fixed}",
        f"CAMP fading mode: {args.camp_fading_mode}",
        f"CAMP fixed lambda: {args.camp_fixed_lambda}",
        f"CAMP lambda floor: {args.camp_lambda_floor:.1e}",
        f"CAMP prob calib: {args.camp_prob_calib}",
        f"CAMP prob center: {args.camp_prob_center}",
        f"CAMP prob alpha: {args.camp_prob_alpha}",
        f"CAMP damping: {args.camp_damping}",
        f"Reg eps: {args.reg_eps:.3e}",
        f"PM: {pm:.6f}",
        f"PF: {pf:.6f}",
    ]
    for method in methods:
        method_nmse = float(torch.tensor(nmse_by_method[method]).mean().item())
        lines.append(f"NMSE ({METHOD_DESCRIPTIONS[method]}): {nmse_db(method_nmse):.2f} dB")

    report = "\n".join(lines)
    print(report)

    # 5) 打印并保存报告。
    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
