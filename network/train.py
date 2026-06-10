from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.data import ActivityDataGenerator as IndependentActivityDataGenerator
from network.data import SystemConfig as IndependentSystemConfig
from network.data_correlated import ActivityDataGenerator as CorrelatedActivityDataGenerator
from network.data_correlated import SystemConfig as CorrelatedSystemConfig
from network.losses import weighted_activity_loss
from network.model import build_model_from_config


# =========================
# System/Data Parameters
# =========================
DATA_MODE = "correlated"  # "independent" or "correlated"
N = 100  # number of users
M = 32  # number of antennas
LP = 30  # pilot length
ACTIVITY_PROB = 0.1
CELL_RADIUS_M = 250.0
PMAX_DBM = 23.0
NOISE_MODE = "snr"  # "snr" or "thermal"
SNR_DB = 20.0
NOISE_POWER_DBM_HZ = -169.0
BANDWIDTH_HZ = 10e6

# =========================
# Model Parameters
# =========================
# Manual model/save choices for the three correlated-data loss experiments:
#   MODEL_NAME = "grouped";        SAVE_DIR = Path("models/checkpoint_grouped")
#   MODEL_NAME = "base-dimension"; SAVE_DIR = Path("models/checkpoint_base_dimension")
#   MODEL_NAME = "base";           SAVE_DIR = Path("models/checkpoint_base")
MODEL_NAME = "base"
DIM = 128
NUM_LAYERS = 5
NUM_HEADS = 8
HEAD_DIM = 32
FF_DIM = 512
SCORE_SCALE = 10.0
NUM_GROUPS = 4
ATTN_DROPOUT = 0.0
FFN_DROPOUT = 0.0
CTX_ATTN_DROPOUT = 0.0
NORM_TYPE = "batch"

# =========================
# Training Parameters
# =========================
DEVICE = "auto"
# SAVE_DIR = Path("models/checkpoint_grouped")
# SAVE_DIR = Path("models/checkpoint_base_dimension")
# SAVE_DIR = Path("models/checkpoint_base")
SAVE_DIR = Path("models/checkpoint_grouped")
LOG_FILE = SAVE_DIR / "train.log"
MODEL_SEED = 42
DATA_SEED = 20260518
TEST_DATA_SEED = 20260519
EPOCHS = 100
STEPS_PER_EPOCH = 10000
BATCH_SIZE = 128
TEST_SAMPLES = 5000
FIXED_TEST_SET = Path("models/fixed_correlated_test_5000.pt")
LR = 3e-4
WEIGHT_DECAY = 0.0
USE_AMP = True
AMP_DTYPE = "bf16"  # "bf16" or "fp16"
WARMUP_EPOCHS = 1
GRAD_CLIP = 1.0
LR_DECAY_EPOCHS = "90,97"
LR_DECAY_FACTOR = 0.1


def resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def build_args() -> SimpleNamespace:
    return SimpleNamespace(
        num_devices=N,
        data_mode=DATA_MODE,
        num_antennas=M,
        pilot_len=LP,
        activity_prob=ACTIVITY_PROB,
        cell_radius_m=CELL_RADIUS_M,
        pmax_dbm=PMAX_DBM,
        noise_mode=NOISE_MODE,
        snr_db=SNR_DB,
        noise_power_dbm_hz=NOISE_POWER_DBM_HZ,
        bandwidth_hz=BANDWIDTH_HZ,
        model_name=MODEL_NAME,
        dim=DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        ff_dim=FF_DIM,
        score_scale=SCORE_SCALE,
        num_groups=NUM_GROUPS,
        attn_dropout=ATTN_DROPOUT,
        ffn_dropout=FFN_DROPOUT,
        ctx_attn_dropout=CTX_ATTN_DROPOUT,
        norm_type=NORM_TYPE,
        device=resolve_device(DEVICE),
        save_dir=SAVE_DIR,
        log_file=LOG_FILE,
        model_seed=MODEL_SEED,
        data_seed=DATA_SEED,
        test_data_seed=TEST_DATA_SEED,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        batch_size=BATCH_SIZE,
        test_samples=TEST_SAMPLES,
        fixed_test_set=FIXED_TEST_SET,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        amp=USE_AMP,
        amp_dtype=AMP_DTYPE,
        warmup_epochs=WARMUP_EPOCHS,
        grad_clip=GRAD_CLIP,
        lr_decay_epochs=LR_DECAY_EPOCHS,
        lr_decay_factor=LR_DECAY_FACTOR,
    )


def build_system_config(args: SimpleNamespace) -> IndependentSystemConfig | CorrelatedSystemConfig:
    config_cls = CorrelatedSystemConfig if args.data_mode == "correlated" else IndependentSystemConfig
    return config_cls(
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
    )


def build_data_generator(
    args: SimpleNamespace,
    system_cfg: IndependentSystemConfig | CorrelatedSystemConfig,
    device: torch.device,
) -> IndependentActivityDataGenerator | CorrelatedActivityDataGenerator:
    if args.data_mode == "correlated":
        return CorrelatedActivityDataGenerator(cfg=system_cfg, device=device)
    if args.data_mode == "independent":
        return IndependentActivityDataGenerator(cfg=system_cfg, device=device)
    raise ValueError(f"Unknown data_mode: {args.data_mode}")


def uses_legacy_base_features(model_name: str) -> bool:
    return model_name.lower() == "base"


def pilot_feature_dim(args: SimpleNamespace) -> int:
    if uses_legacy_base_features(args.model_name):
        return 2 * args.pilot_len
    return 2 * args.pilot_len + 2


def build_model_config(args: SimpleNamespace) -> dict:
    return {
        "model_name": args.model_name,
        "num_devices": args.num_devices,
        "pilot_len": args.pilot_len,
        "pilot_feature_dim": pilot_feature_dim(args),
        "dim": args.dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "ff_dim": args.ff_dim,
        "score_scale": args.score_scale,
        "num_groups": args.num_groups,
        "attn_dropout": args.attn_dropout,
        "ffn_dropout": args.ffn_dropout,
        "ctx_attn_dropout": args.ctx_attn_dropout,
        "norm_type": args.norm_type,
    }


def prepare_model_inputs(
    batch: dict[str, torch.Tensor],
    args: SimpleNamespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_b = batch["x_b"]
    if uses_legacy_base_features(args.model_name):
        x_b = x_b[..., : 2 * args.pilot_len]
    return x_b, batch["x_y"]


def activity_prior_for_loss(
    batch: dict[str, torch.Tensor],
    system_cfg: IndependentSystemConfig | CorrelatedSystemConfig,
) -> float | torch.Tensor:
    return batch.get("activity_prior", system_cfg.activity_prob)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def move_batch_to_cpu(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in batch.items()}


def system_config_metadata(system_cfg: IndependentSystemConfig | CorrelatedSystemConfig) -> dict:
    meta = dict(system_cfg.__dict__)
    if "group_beta_params" in meta:
        meta["group_beta_params"] = [list(pair) for pair in meta["group_beta_params"]]
    return meta


def test_cache_metadata(args: SimpleNamespace, system_cfg: IndependentSystemConfig | CorrelatedSystemConfig) -> dict:
    return {
        "data_mode": args.data_mode,
        "system": system_config_metadata(system_cfg),
        "test_samples": args.test_samples,
        "batch_size": args.batch_size,
        "test_data_seed": args.test_data_seed,
    }


def load_or_create_test_cache(
    args: SimpleNamespace,
    system_cfg: IndependentSystemConfig | CorrelatedSystemConfig,
    data_gen: IndependentActivityDataGenerator | CorrelatedActivityDataGenerator,
) -> list[dict[str, torch.Tensor]]:
    expected_meta = test_cache_metadata(args, system_cfg)
    path = Path(args.fixed_test_set)
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if cache.get("metadata") == expected_meta:
            return cache["batches"]

    path.parent.mkdir(parents=True, exist_ok=True)
    batches = []
    num_batches = math.ceil(args.test_samples / args.batch_size)
    for batch_idx in range(num_batches):
        this_batch_size = min(args.batch_size, args.test_samples - batch_idx * args.batch_size)
        torch.manual_seed(args.test_data_seed + batch_idx)
        batches.append(move_batch_to_cpu(data_gen.sample_batch(this_batch_size)))
    torch.save({"metadata": expected_meta, "batches": batches}, path)
    return batches


def train_batch_seed(args: SimpleNamespace, epoch: int, step: int) -> int:
    batch_index = (epoch - 1) * args.steps_per_epoch + step
    return args.data_seed + batch_index


def evaluate_test_loss(
    model: torch.nn.Module,
    test_batches: list[dict[str, torch.Tensor]],
    args: SimpleNamespace,
    system_cfg: IndependentSystemConfig | CorrelatedSystemConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for cpu_batch in test_batches:
            batch = move_batch_to_device(cpu_batch, device)
            x_b, x_y = prepare_model_inputs(batch, args)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(x_b, x_y)
                loss = weighted_activity_loss(
                    logits=logits,
                    targets=batch["label"],
                    activity_prob=activity_prior_for_loss(batch, system_cfg),
                )
            bsz = int(batch["label"].shape[0])
            loss_sum += float(loss.item()) * bsz
            sample_count += bsz
    return loss_sum / float(max(1, sample_count))


def append_log(path: Path | None, line: str) -> None:
    if path:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def main() -> None:
    args = build_args()

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
    data_gen = build_data_generator(args, system_cfg, device=device)
    test_batches = load_or_create_test_cache(args, system_cfg, data_gen)

    torch.manual_seed(args.model_seed)
    model = build_model_from_config(model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    decay_epochs = sorted(int(x.strip()) for x in args.lr_decay_epochs.split(",") if x.strip())

    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        command_line = subprocess.list2cmdline([sys.executable, *sys.argv])
        append_log(args.log_file, "")
        append_log(args.log_file, f"Command: {command_line}")
        append_log(args.log_file, f"Args: {json.dumps(vars(args), sort_keys=True, default=str)}")
        append_log(args.log_file, f"System: {json.dumps(system_config_metadata(system_cfg), sort_keys=True)}")
        append_log(args.log_file, f"Model: {json.dumps(model_cfg, sort_keys=True)}")
        append_log(args.log_file, f"Fixed test set: {Path(args.fixed_test_set).resolve()}")
        append_log(args.log_file, "")

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
        for step in pbar:
            torch.manual_seed(train_batch_seed(args, epoch, step))
            batch = data_gen.sample_batch(args.batch_size)
            x_b, x_y = prepare_model_inputs(batch, args)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(x_b, x_y)
                loss = weighted_activity_loss(
                    logits=logits,
                    targets=batch["label"],
                    activity_prob=activity_prior_for_loss(batch, system_cfg),
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

        avg_loss = running_loss / float(max(1, steps_done))
        summary = (
            f"Epoch {epoch:03d} | loss={avg_loss:.6f} | "
            f"skip={skipped_nonfinite} | lr={lr_now:.2e}"
        )
        print(summary)
        append_log(args.log_file, summary)

        ckpt = {
            "model_state": model.state_dict(),
            "config": vars(args),
            "model_config": model_cfg,
            "system_config": system_config_metadata(system_cfg),
            "epoch": epoch,
        }
        torch.save(ckpt, args.save_dir / "last.pt")

    test_loss = evaluate_test_loss(
        model=model,
        test_batches=test_batches,
        args=args,
        system_cfg=system_cfg,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
    )
    final_summary = f"Final test_loss={test_loss:.6f} | test_samples={args.test_samples}"
    print(final_summary)
    append_log(args.log_file, final_summary)

    ckpt = {
        "model_state": model.state_dict(),
        "config": vars(args),
        "model_config": model_cfg,
        "system_config": system_config_metadata(system_cfg),
        "epoch": args.epochs,
        "test_loss": test_loss,
    }
    torch.save(ckpt, args.save_dir / "last.pt")

    print(f"Training done. Checkpoint saved in: {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()
