from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.config import (
    apply_cli_overrides,
    load_experiment_config,
    model_config_from_experiment,
    section_namespace,
    system_config_from_experiment,
)
from network.data import ActivityDataGenerator
from network.losses import weighted_activity_loss
from network.metrics import pm_pf_at_threshold
from network.model import build_model_from_config


def parse_args() -> argparse.Namespace:
    # 命令行参数只作为临时覆盖项使用；真正的默认实验设置来自 config.json。
    # 这里把 default 设为 None，是为了区分“用户没有传这个参数”和“用户想覆盖配置”。
    parser = argparse.ArgumentParser(description="Train heterogeneous transformer for device activity detection.")
    parser.add_argument("--config", type=str, default="config.json", help="Experiment config JSON path.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None, help="Optional path to append per-epoch summary logs.")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use CUDA mixed precision training when available.",
    )
    parser.add_argument("--amp_dtype", type=str, default=None, choices=["bf16", "fp16"])
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--lr_decay_epochs", type=str, default=None)
    parser.add_argument("--lr_decay_factor", type=float, default=None)

    parser.add_argument("--eval_batches", type=int, default=None)
    parser.add_argument("--eval_threshold", type=float, default=None)
    parser.add_argument(
        "--fixed_eval_set",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use a fixed validation set across epochs for stable PM/PF curves.",
    )
    return parser.parse_args()


def main() -> None:
    # 1) 读取配置文件，并用命令行参数覆盖其中的 network_train 部分。
    # 优先级：命令行参数 > config.json > network/config.py 中的 DEFAULT_CONFIG。
    cli = parse_args()
    exp_cfg = load_experiment_config(cli.config)
    args = section_namespace(exp_cfg, "network_train")
    apply_cli_overrides(
        args,
        cli,
        [
            "device",
            "save_dir",
            "log_file",
            "seed",
            "epochs",
            "steps_per_epoch",
            "batch_size",
            "lr",
            "amp",
            "amp_dtype",
            "warmup_epochs",
            "grad_clip",
            "lr_decay_epochs",
            "lr_decay_factor",
            "eval_batches",
            "eval_threshold",
            "fixed_eval_set",
        ],
    )

    # 2) 固定随机种子，选择训练设备，并设置混合精度训练选项。
    # device="auto" 会在 network.config.section_namespace 中自动解析成 cuda 或 cpu。
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    #AMP = Automatic Mixed Precision，自动混合精度训练
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    # fp16 需要 GradScaler 防止梯度下溢；bf16 通常不需要 scaler。
    use_scaler = bool(use_amp and amp_dtype == torch.float16)
    if device.type == "cuda":
        # 允许 TF32，可以在 NVIDIA GPU 上加速矩阵乘法，通常对训练精度影响很小。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # 3) 根据 config.json 的 network_train.system 部分构造仿真系统和在线数据生成器。
    # 每次 sample_batch 都会随机生成一批新的导频、信道、活跃标签和接收信号。
    cfg = system_config_from_experiment(exp_cfg)
    data_gen = ActivityDataGenerator(cfg=cfg, device=device)

    # 4) 根据 config.json 的 network_train.model 部分构造网络。
    # model_cfg 会自动补入 num_devices 和 pilot_len，因为这两个参数决定输入层尺寸。
    model_cfg = model_config_from_experiment(exp_cfg)

    model = build_model_from_config(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    decay_set = {int(x.strip()) for x in args.lr_decay_epochs.split(",") if x.strip()}
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else None
    if log_path is not None:
        # 如果配置了 log_file，就把命令、系统参数和模型参数写入日志，方便复现实验。
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command_line = subprocess.list2cmdline([sys.executable, *sys.argv])
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f"Command: {command_line}\n")
            f.write(f"Config: {cli.config}\n")
            f.write(f"Args: {json.dumps(vars(args), sort_keys=True)}\n")
            f.write(f"System: {json.dumps(cfg.__dict__, sort_keys=True)}\n")
            f.write(f"Model: {json.dumps(model_cfg, sort_keys=True)}\n")
            f.write("\n")

    best_pm = float("inf")
    eval_cache = None
    if args.fixed_eval_set:
        # 固定验证集可以让不同 epoch 的 PM/PF 曲线更稳定，减少随机验证数据带来的抖动。
        eval_cache = [data_gen.sample_batch(args.batch_size) for _ in range(args.eval_batches)]
    decay_epochs_sorted = sorted(decay_set)
    for epoch in range(1, args.epochs + 1):
        # 5) 在每个 epoch 开始时更新学习率。
        # warmup_epochs > 0 时先线性升高学习率；之后按 lr_decay_epochs 进行阶梯衰减。
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            lr_now = args.lr * (float(epoch) / float(args.warmup_epochs))
        else:
            passed = sum(1 for d in decay_epochs_sorted if d < epoch)
            lr_now = args.lr * (args.lr_decay_factor ** passed)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        model.train()
        running_loss = 0.0
        n_steps = 0
        skipped_nonfinite = 0
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for _ in pbar:
            # 6) 在线生成一个训练 batch。
            # x_b: 每个设备的导频特征；x_y: 接收信号协方差特征；label: 真实活跃状态。
            batch = data_gen.sample_batch(args.batch_size)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, probs = model(batch["x_b"], batch["x_y"])
                loss = weighted_activity_loss(
                    logits=logits,
                    targets=batch["label"],
                    activity_prob=cfg.activity_prob,
                )
            if not torch.isfinite(loss):
                # 极端数值异常时跳过该 step，避免把 NaN/Inf 写入模型参数。
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                # fp16 混合精度路径：先 scale loss，再反传和更新参数。
                scaler.scale(loss).backward()
                if args.grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 普通 fp32 或 bf16 路径。
                loss.backward()
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            running_loss += float(loss.item())
            n_steps += 1
            pbar.set_postfix(loss=f"{running_loss / max(1, n_steps):.4f}")

        # 7) 每个 epoch 结束后做一次验证，只计算 PM/PF，不更新模型参数。
        model.eval()
        eval_probs = []
        eval_labels = []
        with torch.no_grad():
            eval_iter = eval_cache if eval_cache is not None else [data_gen.sample_batch(args.batch_size) for _ in range(args.eval_batches)]
            for batch in eval_iter:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    _, p = model(batch["x_b"], batch["x_y"])
                eval_probs.append(p)
                eval_labels.append(batch["label"])

        eval_probs_t = torch.cat(eval_probs, dim=0)
        eval_labels_t = torch.cat(eval_labels, dim=0)
        pm, pf = pm_pf_at_threshold(eval_probs_t, eval_labels_t, args.eval_threshold)
        avg_loss = running_loss / float(max(1, n_steps))
        lr_now = optimizer.param_groups[0]["lr"]
        mean_prob = float(eval_probs_t.mean().item())
        # PM: 漏检概率；PF: 虚警概率；p_mean: 模型输出的平均活跃概率，用来观察输出是否塌缩。
        summary = (
            f"Epoch {epoch:03d} | loss={avg_loss:.6f} | PM@{args.eval_threshold:.2f}={pm:.6f} | "
            f"PF@{args.eval_threshold:.2f}={pf:.6f} | p_mean={mean_prob:.4f} | "
            f"skip={skipped_nonfinite} | lr={lr_now:.2e}"
        )
        print(summary)
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(summary + "\n")

        # 8) 保存 checkpoint。
        # last.pt 始终保存最新 epoch；best_pm.pt 保存验证集 PM 最低的模型。
        ckpt = {
            "model_state": model.state_dict(),
            "config": vars(args),
            "model_config": model_cfg,
            "experiment_config": exp_cfg,
            "system_config": cfg.__dict__,
            "epoch": epoch,
        }
        torch.save(ckpt, save_dir / "last.pt")
        if pm < best_pm:
            best_pm = pm
            torch.save(ckpt, save_dir / "best_pm.pt")

    print(f"Training done. Checkpoints saved in: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
