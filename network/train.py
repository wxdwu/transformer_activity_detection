from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.data import ActivityDataGenerator, SystemConfig
from network.losses import weighted_activity_loss
from network.metrics import pm_pf_at_threshold
from network.model import build_model_from_config


# =========================
# System/Data Parameters
# =========================
N = 200  # number of users
M = 32  # number of antennas
LP = 30  # pilot length
ACTIVITY_PROB = 0.1
CELL_RADIUS_M = 500.0
PMAX_DBM = 23.0
NOISE_MODE = "snr"  # "snr" or "thermal"
SNR_DB = 20.0
NOISE_POWER_DBM_HZ = -169.0
BANDWIDTH_HZ = 10e6
ACTIVITY_MODE = "event"  # "independent" or "event"
EVENT_LAMBDA = 1.0
EVENT_MAX_COUNT = 3
EVENT_SIGMA_M = 120.0
EVENT_TRIGGER_PROB = 0.9
BACKGROUND_ACTIVITY_PROB = 0.005
CORRELATION_LENGTH_M = 0.0

# =========================
# Model Parameters
# =========================
MODEL_NAME = "base"
USE_CORRELATION_ATTENTION_BIAS = True
USE_CORRELATION_LOGIT_REFINEMENT = True
CORR_ATTN_INIT = 1.0
CORR_REFINE_INIT = 0.5
DIM = 128
NUM_LAYERS = 5
NUM_HEADS = 8
HEAD_DIM = 32
FF_DIM = 512
SCORE_SCALE = 10.0
ATTN_DROPOUT = 0.0
FFN_DROPOUT = 0.0
CTX_ATTN_DROPOUT = 0.0
NORM_TYPE = "batch"

# =========================
# Training Parameters
# =========================
DEVICE = "auto"
SAVE_DIR = ROOT / "checkpoint/event_corrmatrix_260622"
LOG_FILE = SAVE_DIR / "train.log"
SEED = 42
EPOCHS = 100
STEPS_PER_EPOCH = 2000
BATCH_SIZE = 128
LR = 1e-4
USE_AMP = True
AMP_DTYPE = "bf16"  # "bf16" or "fp16"
WARMUP_EPOCHS = 0
GRAD_CLIP = 0.0
LR_DECAY_EPOCHS = "90,97"
LR_DECAY_FACTOR = 0.1
EVAL_BATCHES = 20
EVAL_THRESHOLD = 0.5
FIXED_EVAL_SET = True


def resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def build_args() -> SimpleNamespace:
    return SimpleNamespace(
        num_devices=N,
        num_antennas=M,
        pilot_len=LP,
        activity_prob=ACTIVITY_PROB,
        cell_radius_m=CELL_RADIUS_M,
        pmax_dbm=PMAX_DBM,
        noise_mode=NOISE_MODE,
        snr_db=SNR_DB,
        noise_power_dbm_hz=NOISE_POWER_DBM_HZ,
        bandwidth_hz=BANDWIDTH_HZ,
        activity_mode=ACTIVITY_MODE,
        event_lambda=EVENT_LAMBDA,
        event_max_count=EVENT_MAX_COUNT,
        event_sigma_m=EVENT_SIGMA_M,
        event_trigger_prob=EVENT_TRIGGER_PROB,
        background_activity_prob=BACKGROUND_ACTIVITY_PROB,
        correlation_length_m=CORRELATION_LENGTH_M,
        model_name=MODEL_NAME,
        use_correlation_attention_bias=USE_CORRELATION_ATTENTION_BIAS,
        use_correlation_logit_refinement=USE_CORRELATION_LOGIT_REFINEMENT,
        corr_attn_init=CORR_ATTN_INIT,
        corr_refine_init=CORR_REFINE_INIT,
        dim=DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        ff_dim=FF_DIM,
        score_scale=SCORE_SCALE,
        attn_dropout=ATTN_DROPOUT,
        ffn_dropout=FFN_DROPOUT,
        ctx_attn_dropout=CTX_ATTN_DROPOUT,
        norm_type=NORM_TYPE,
        device=resolve_device(DEVICE),
        save_dir=SAVE_DIR,
        log_file=LOG_FILE,
        seed=SEED,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        batch_size=BATCH_SIZE,
        lr=LR,
        amp=USE_AMP,
        amp_dtype=AMP_DTYPE,
        warmup_epochs=WARMUP_EPOCHS,
        grad_clip=GRAD_CLIP,
        lr_decay_epochs=LR_DECAY_EPOCHS,
        lr_decay_factor=LR_DECAY_FACTOR,
        eval_batches=EVAL_BATCHES,
        eval_threshold=EVAL_THRESHOLD,
        fixed_eval_set=FIXED_EVAL_SET,
    )


def build_system_config(args: SimpleNamespace) -> SystemConfig:
    return SystemConfig(
        num_devices=args.num_devices,
        num_antennas=args.num_antennas,
        pilot_len=args.pilot_len,
        activity_prob=args.activity_prob,
        cell_radius_m=args.cell_radius_m,
        noise_power_dbm_hz=args.noise_power_dbm_hz,
        bandwidth_hz=args.bandwidth_hz,
        pmax_dbm=args.pmax_dbm,
        noise_mode=args.noise_mode,
        snr_db=args.snr_db,
        activity_mode=args.activity_mode,
        event_lambda=args.event_lambda,
        event_max_count=args.event_max_count,
        event_sigma_m=args.event_sigma_m,
        event_trigger_prob=args.event_trigger_prob,
        background_activity_prob=args.background_activity_prob,
        correlation_length_m=args.correlation_length_m,
    )


def build_model_config(args: SimpleNamespace) -> dict:
    return {
        "model_name": args.model_name,
        "num_devices": args.num_devices,
        "pilot_len": args.pilot_len,
        "use_correlation_attention_bias": args.use_correlation_attention_bias,
        "use_correlation_logit_refinement": args.use_correlation_logit_refinement,
        "corr_attn_init": args.corr_attn_init,
        "corr_refine_init": args.corr_refine_init,
        "dim": args.dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "ff_dim": args.ff_dim,
        "score_scale": args.score_scale,
        "attn_dropout": args.attn_dropout,
        "ffn_dropout": args.ffn_dropout,
        "ctx_attn_dropout": args.ctx_attn_dropout,
        "norm_type": args.norm_type,
    }


def main() -> None:
    args = build_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = bool(use_amp and amp_dtype == torch.float16)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    system_cfg = build_system_config(args)
    model_cfg = build_model_config(args)
    data_gen = ActivityDataGenerator(cfg=system_cfg, device=device)
    model = build_model_from_config(model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    decay_epochs = sorted(int(x.strip()) for x in args.lr_decay_epochs.split(",") if x.strip())

    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        command_line = subprocess.list2cmdline([sys.executable, *sys.argv])
        with args.log_file.open("a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f"Command: {command_line}\n")
            f.write(f"Args: {json.dumps(vars(args), sort_keys=True, default=str)}\n")
            f.write(f"System: {json.dumps(system_cfg.__dict__, sort_keys=True)}\n")
            f.write(f"Model: {json.dumps(model_cfg, sort_keys=True)}\n\n")

    best_pm = float("inf")
    eval_cache = None
    if args.fixed_eval_set:
        eval_cache = [data_gen.sample_batch(args.batch_size) for _ in range(args.eval_batches)]

    for epoch in range(1, args.epochs + 1):
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            lr_now = args.lr * float(epoch) / float(args.warmup_epochs)
        else:
            passed_decays = sum(1 for d in decay_epochs if d < epoch)
            lr_now = args.lr * (args.lr_decay_factor ** passed_decays)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        model.train()
        running_loss = 0.0
        steps_done = 0
        skipped_nonfinite = 0
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for _ in pbar:
            batch = data_gen.sample_batch(args.batch_size)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(batch["x_b"], batch["x_y"], batch.get("corr_matrix"))
                loss = weighted_activity_loss(
                    logits=logits,
                    targets=batch["label"],
                    activity_prob=float(batch["label"].mean().item()),
                )

            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                if args.grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            steps_done += 1
            running_loss += float(loss.item())
            pbar.set_postfix(loss=f"{running_loss / max(1, steps_done):.4f}")

        model.eval()
        eval_probs = []
        eval_labels = []
        with torch.no_grad():
            eval_iter = eval_cache if eval_cache is not None else [
                data_gen.sample_batch(args.batch_size) for _ in range(args.eval_batches)
            ]
            for batch in eval_iter:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    _, probs = model(batch["x_b"], batch["x_y"], batch.get("corr_matrix"))
                eval_probs.append(probs)
                eval_labels.append(batch["label"])

        probs_all = torch.cat(eval_probs, dim=0)
        labels_all = torch.cat(eval_labels, dim=0)
        pm, pf = pm_pf_at_threshold(probs_all, labels_all, args.eval_threshold)
        avg_loss = running_loss / float(max(1, steps_done))
        mean_prob = float(probs_all.mean().item())
        summary = (
            f"Epoch {epoch:03d} | loss={avg_loss:.6f} | PM@{args.eval_threshold:.2f}={pm:.6f} | "
            f"PF@{args.eval_threshold:.2f}={pf:.6f} | p_mean={mean_prob:.4f} | "
            f"skip={skipped_nonfinite} | lr={lr_now:.2e}"
        )
        print(summary)
        if args.log_file:
            with args.log_file.open("a", encoding="utf-8") as f:
                f.write(summary + "\n")

        ckpt = {
            "model_state": model.state_dict(),
            "config": vars(args),
            "model_config": model_cfg,
            "system_config": system_cfg.__dict__,
            "epoch": epoch,
        }
        torch.save(ckpt, args.save_dir / "last.pt")
        if pm < best_pm:
            best_pm = pm
            torch.save(ckpt, args.save_dir / "best_pm.pt")

    print(f"Training done. Checkpoints saved in: {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()
