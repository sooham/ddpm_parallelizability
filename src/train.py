"""Training loop for DDPM/DDIM models with wandb logging."""

import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim import Adam

from .config import Config
from .dataset import create_dataloader
from .diffusion import GaussianDiffusion
from .model import create_model

# Enable TF32 tensor cores on Ampere+ GPUs (RTX 30xx, A100, etc.) — ~2× matmul speed
torch.set_float32_matmul_precision('high')


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str) -> torch.device:
    """Resolve the device to use (MPS, CUDA, or CPU fallback)."""
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        if preferred not in ("cpu",):
            print(f"Warning: '{preferred}' not available, falling back to CPU")
        return torch.device("cpu")


@torch.no_grad()
def generate_samples(
    diffusion: GaussianDiffusion,
    shape: tuple,
    device: torch.device,
) -> torch.Tensor:
    """Generate sample images (in [0, 1]) for logging."""
    diffusion.eval()
    samples = diffusion.sample(shape, device)
    diffusion.train()
    samples = (samples + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    samples = torch.clamp(samples, 0.0, 1.0)
    return samples


def _count_flops_per_batch(diffusion, x0: torch.Tensor) -> int:
    """Count exact FLOPs for one training step (forward + backward) using FlopCounterMode.

    Traces the full computation graph of training_loss + backward() to get a
    precise FLOP count for the given model and input shape.  Does not modify
    model parameters (the backward is discarded).
    """
    from torch.utils.flop_counter import FlopCounterMode

    x0 = x0.detach().clone()
    model = diffusion.model
    model.train()

    with FlopCounterMode(depth=2, display=False) as fcm:
        loss, _betas = diffusion.training_loss(x0)
        loss.backward()

    # Wipe the backward's .grad so the real training starts clean
    for p in model.parameters():
        p.grad = None

    return fcm.get_total_flops()


def _get_event() -> tuple[type | None, callable]:
    """Return (EventClass, synchronize_fn) for the current device, or (None, noop) for CPU."""
    if torch.cuda.is_available():
        return torch.cuda.Event, torch.cuda.synchronize
    if torch.backends.mps.is_available():
        return torch.mps.Event, torch.mps.synchronize
    # CPU — no event API; fall back to wall-clock only
    return None, (lambda: None)


def _build_lr_scheduler(optimizer, cfg, total_steps: int):
    """Build a learning-rate scheduler with warmup → decay → minimum phases.

    Phases (as fraction of total_steps):
      0 … warmup_pct          linear ramp  0 → lr_max
      warmup_pct … decay_pct  cosine/linear lr_max → lr_min
      decay_pct … 1.0         constant at lr_min
    """
    warmup_steps = int(total_steps * cfg.lr_warmup_pct)
    decay_steps = int(total_steps * cfg.lr_decay_pct)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup: 0 → lr_max
            return (step / max(1, warmup_steps)) * (cfg.lr_max / cfg.learning_rate)
        elif step < decay_steps:
            # Decay: lr_max → lr_min
            progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
            if cfg.lr_schedule == "cosine":
                coeff = 0.5 * (1.0 + np.cos(progress * np.pi))
            else:  # linear
                coeff = 1.0 - progress
            lr = cfg.lr_min + (cfg.lr_max - cfg.lr_min) * coeff
            return lr / cfg.learning_rate
        else:
            # Minimum phase
            return cfg.lr_min / cfg.learning_rate

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config: Config) -> None:
    """Run the full training loop."""
    cfg = config.training
    device = get_device(cfg.device)

    set_seed(cfg.seed)

    # --- Data ---
    # MPS/CPU can't use multiprocessing; CUDA can
    nw = 0 if device.type != "cuda" else cfg.num_workers
    pin = cfg.pin_memory and device.type == "cuda"
    train_loader = create_dataloader(
        name=cfg.dataset,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        split="train",
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        cache_dir="./datasets",
    )

    # Determine in_channels from data (1 for grayscale, 3 for RGB)
    sample_batch = next(iter(train_loader))
    in_channels = sample_batch.shape[1]
    train_size = len(train_loader.dataset)
    test_loader = create_dataloader(
        name=cfg.dataset,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        split="test",
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        cache_dir="./datasets",
    )
    test_size = len(test_loader.dataset)

    # Update model config to match data
    config.model.in_channels = in_channels
    config.model.out_channels = in_channels

    # --- Model (created after channels are known) ---
    model = create_model(config.model).to(device)

    # Lazy-build the model (MLPDenoiser needs to see input shape before layers exist).
    dummy_x = torch.randn(1, in_channels, cfg.image_size, cfg.image_size, device=device)
    dummy_t = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        model(dummy_x, dummy_t)

    # --- Autocast dtype ---
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        # T4/V100: no bfloat16 hardware — force float16 regardless of driver report
        if "T4" in gpu_name or "V100" in gpu_name:
            amp_dtype = torch.float16
        elif torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
        use_amp = True
    else:
        amp_dtype = torch.float32
        use_amp = False

    diffusion = GaussianDiffusion(model, config.diffusion).to(device)
    scaler = torch.amp.GradScaler(device.type) if (use_amp and amp_dtype == torch.float16) else None

    total_params = sum(p.numel() for p in model.parameters())
    batches_per_epoch = len(train_loader)

    # Count exact FLOPs for one training step (fwd + bwd) on a real batch
    flops_per_batch = _count_flops_per_batch(
        diffusion,
        sample_batch[:cfg.batch_size].to(device),
    )

    print(f"Dataset: {cfg.dataset} | Train: {train_size:,} | Test: {test_size:,}")
    print(f"Image size: {cfg.image_size} | Channels: {in_channels}")
    print(f"Model parameters: {total_params:,}")
    print(f"AMP: {amp_dtype}" if use_amp else "AMP: off")
    print(f"FLOPs/batch (fwd+bwd): {flops_per_batch / 1e9:.2f}G (traced)")
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Device: {device}" if device.type != "cuda" else f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Diffusion steps: {config.diffusion.timesteps} | Sampler: {config.diffusion.sampler}")

    # --- EMA (must be created BEFORE torch.compile — deepcopy of compiled model breaks) ---
    ema_model = None
    ema_p_list = None
    if cfg.ema_decay > 0:
        ema_model = deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False
        ema_p_list = list(ema_model.parameters())
        p_list = list(model.parameters())

    # --- torch.compile (fuses ops → fewer CPU→GPU dispatches) ---
    if device.type == "cuda":
        try:
            model = torch.compile(diffusion.model, mode="default")
            diffusion.model = model
            # Warmup with float32 data (autocast handles fp16 internally during training)
            _dummy_x = torch.randn(cfg.batch_size, in_channels, cfg.image_size, cfg.image_size, device=device)
            _dummy_t = torch.randint(0, config.diffusion.timesteps, (_dummy_x.shape[0],), device=device)
            _loss, _ = diffusion.training_loss(_dummy_x)
            _loss.backward()
            for p in model.parameters():
                p.grad = None
            print("torch.compile: enabled (fwd+bwd compiled)")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")
    else:
        print("torch.compile: off (MPS/CPU)")

    # --- Optimizer (fused Adam reduces kernel launches on GPU) ---
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas, fused=True)

    # --- LR Scheduler ---
    total_steps = cfg.epochs * batches_per_epoch
    if cfg.lr_schedule in ("cosine", "linear"):
        scheduler = _build_lr_scheduler(optimizer, cfg, total_steps)
        print(f"LR schedule: {cfg.lr_schedule} | warmup={cfg.lr_warmup_pct:.0%} | "
              f"decay={cfg.lr_decay_pct:.0%} | max={cfg.lr_max:.1e} | min={cfg.lr_min:.1e}")
    else:
        scheduler = None

    # --- wandb ---
    wandb_run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        config={
            "dataset": cfg.dataset,
            "image_size": cfg.image_size,
            "in_channels": in_channels,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "timesteps": config.diffusion.timesteps,
            "beta_schedule": config.diffusion.beta_schedule,
            "sampler": config.diffusion.sampler,
            "ddim_steps": config.diffusion.ddim_steps,
            "ddim_eta": config.diffusion.ddim_eta,
            "model_type": config.model.model_type,
            "model_channels": config.model.model_channels,
            "channel_mult": config.model.channel_mult,
            "num_res_blocks": config.model.num_res_blocks,
            "attention_resolutions": config.model.attention_resolutions,
            "mlp_hidden_dims": config.model.mlp_hidden_dims,
            "mlp_activation": config.model.mlp_activation,
            "dropout": config.model.dropout,
            "device": str(device),
            "total_params": total_params,
        },
        reinit=True,
    )

    # --- Checkpoint directory ---
    ckpt_dir = Path("./checkpoints") / cfg.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    global_step = 0

    Event, sync = _get_event()

    LOG_INTERVAL = 50

    for epoch in range(1, cfg.epochs + 1):
        t_epoch_start = time.time()
        epoch_losses = []
        epoch_batch_times = []
        epoch_device_times = []

        # Window accumulators (flushed every LOG_INTERVAL batches)
        win_losses = []
        win_betas = []
        win_start_ev = None
        win_end_ev = None

        t0_window = time.time()  # wall-clock window start

        for batch_idx, batch in enumerate(train_loader, 1):
            # Align device span with wall clock (first batch of each window)
            if Event is not None and win_start_ev is None:
                win_start_ev = Event(enable_timing=True)
                win_end_ev = Event(enable_timing=True)
                win_start_ev.record()

            x0 = batch.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.autocast(device.type, dtype=amp_dtype):
                    loss, betas_t = diffusion.training_loss(x0)
            else:
                loss, betas_t = diffusion.training_loss(x0)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # EMA update (fused foreach ops — single kernel per op)
            if ema_model is not None:
                with torch.no_grad():
                    torch._foreach_mul_(ema_p_list, cfg.ema_decay)
                    torch._foreach_add_(ema_p_list, p_list, alpha=(1.0 - cfg.ema_decay))

            lr = optimizer.param_groups[0]['lr']
            mean_beta = betas_t.mean().item()

            global_step += 1

            # Accumulate
            epoch_losses.append(loss.item())

            win_losses.append(loss.item())
            win_betas.append(mean_beta)

            # Flush window every LOG_INTERVAL batches (or on the last batch)
            if batch_idx % LOG_INTERVAL == 0 or batch_idx == batches_per_epoch:
                n = len(win_losses)
                avg_loss = np.mean(win_losses)
                avg_beta = np.mean(win_betas)

                # ---- grad norm (compute once per window, not every batch) ----
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                avg_grad = total_norm ** 0.5

                # ---- device time: one sync for the whole window ----
                if Event is not None:
                    win_end_ev.record()
                    sync()
                    dt_dev_total = win_start_ev.elapsed_time(win_end_ev) / 1000.0  # ms → s
                    avg_dt_dev = dt_dev_total / n
                    win_start_ev = None
                    win_end_ev = None
                else:
                    avg_dt_dev = 0.0

                # ---- wall clock: measure over full window including GPU drain ----
                dt_wall_total = time.time() - t0_window
                avg_dt_wall = dt_wall_total / n
                t0_window = time.time()  # reset for next window

                epoch_batch_times.extend([avg_dt_wall] * n)
                epoch_device_times.extend([avg_dt_dev] * n)

                bps_wall = 1.0 / avg_dt_wall if avg_dt_wall > 0 else 0.0
                bps_dev = 1.0 / avg_dt_dev if avg_dt_dev > 0 else 0.0
                mflops = (flops_per_batch / 1e6) * bps_dev

                # Log to wandb (once per window; x-axis = batch count)
                wandb_run.log({
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/grad_norm": avg_grad,
                    "train/beta": avg_beta,
                    "train/batch_per_sec": bps_wall,
                    "train/batch_per_sec_dev": bps_dev,
                    "train/mflops": mflops,
                }, step=global_step)

                # Console progress
                elapsed = time.time() - t_epoch_start
                print(f"  Batch {batch_idx:4d}/{batches_per_epoch} | "
                      f"loss: {avg_loss:.4f} | lr: {lr:.2e} | β: {avg_beta:.4f} | "
                      f"grad: {avg_grad:.3f} | "
                      f"{bps_wall:.1f} b/s | {mflops:,.0f} MFLOPS  "
                      f"[{n} steps | {elapsed:.0f}s]")

                # Reset window
                win_losses = []
                win_betas = []

        # ---- Epoch summary (full-epoch averages) ----
        avg_loss = np.mean(epoch_losses)
        avg_dt_wall = np.mean(epoch_batch_times) if epoch_batch_times else 0.001
        avg_dt_dev = np.mean(epoch_device_times) if epoch_device_times else 0.001
        bps_wall = 1.0 / avg_dt_wall
        bps_dev = 1.0 / avg_dt_dev
        mflops = (flops_per_batch / 1e6) * bps_dev
        dt_epoch = time.time() - t_epoch_start
        epm = 60.0 / dt_epoch if dt_epoch > 0 else 0.0
        print(f"Epoch {epoch:3d}/{cfg.epochs} | Loss: {avg_loss:.6f} | "
              f"{bps_wall:.1f} b/s | {mflops:,.0f} MFLOPS | "
              f"{epm:.2f} ep/min | {dt_epoch:.0f}s | Steps: {global_step}")

        wandb_run.log({
            "train/epoch_loss": avg_loss,
            "train/epoch_duration_s": dt_epoch,
            "train/ep_per_min": epm,
            "train/epoch": epoch,
        }, step=global_step)

        # --- Generate sample every epoch ---
        sampling_model = ema_model if ema_model is not None else model
        original_model = diffusion.model
        diffusion.model = sampling_model

        n = min(cfg.num_samples, cfg.batch_size)
        samples = generate_samples(diffusion, (n, in_channels, cfg.image_size, cfg.image_size), device)
        diffusion.model = original_model

        # Quick sanity: are we getting real images or noise?
        print(f"  Samples: min={samples.min().item():.2f} max={samples.max().item():.2f} unique={samples.unique().numel()}/{samples.numel()}")

        grid = _make_image_grid(samples)
        wandb_run.log({
            "samples": wandb.Image(grid, caption=f"Epoch {epoch}"),
        }, step=global_step + 1)

        # --- Save checkpoint ---
        if cfg.checkpoint and (epoch % cfg.save_every == 0 or epoch == cfg.epochs):
            _save_checkpoint(ckpt_dir, epoch, model, ema_model, optimizer, config)

    # --- Final save ---
    if cfg.checkpoint:
        final_path = ckpt_dir / "final_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "config": config,
        }, final_path)
        print(f"Final model saved: {final_path}")

    wandb_run.finish()


def _save_checkpoint(ckpt_dir, epoch, model, ema_model, optimizer, config):
    ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
    data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if ema_model is not None:
        data["ema_state_dict"] = ema_model.state_dict()
    torch.save(data, ckpt_path)
    print(f"  Saved: {ckpt_path}")


def _make_image_grid(images: torch.Tensor, nrow: int = 4) -> torch.Tensor:
    """Arrange (N, C, H, W) images in [0, 1] into a grid."""
    from torchvision.utils import make_grid
    return make_grid(images, nrow=nrow, pad_value=1.0)
