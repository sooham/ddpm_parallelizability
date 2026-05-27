"""Configuration for DDPM/DDIM training and inference."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""

    # Model type: "unet" (convolutional) or "mlp" (dense layers for simple data like MNIST)
    model_type: Literal["unet", "mlp"] = "unet"

    # Input/output channels (set automatically from dataset: 1 for MNIST, 3 for CIFAR-10)
    in_channels: int = 3
    out_channels: int = 3

    # ---- U-Net specific ----
    # Base feature dimension
    model_channels: int = 128

    # Channel multipliers per resolution level (increasing depth)
    channel_mult: tuple[int, ...] = (1, 2, 2, 2)

    # Number of residual blocks per resolution level
    num_res_blocks: int = 2

    # Attention resolutions (feature map sizes at which to apply self-attention)
    attention_resolutions: tuple[int, ...] = (16,)

    # Dropout rate
    dropout: float = 0.1

    # Group normalization groups
    num_groups: int = 32

    # Whether to use attention at all resolutions (ignored if attention_resolutions is set)
    attention_all: bool = False

    # ---- MLP specific ----
    # Hidden layer dimensions (e.g., [1024, 2048, 1024])
    mlp_hidden_dims: tuple[int, ...] = (1024, 2048, 1024)

    # Time embedding dimension for MLP
    mlp_time_dim: int = 256

    # Activation function for MLP hidden layers
    mlp_activation: Literal["gelu", "relu", "silu"] = "gelu"


@dataclass
class DiffusionConfig:
    """Diffusion process hyperparameters."""

    # Number of diffusion timesteps
    timesteps: int = 1000

    # Noise schedule: "linear" or "cosine"
    beta_schedule: Literal["linear", "cosine"] = "linear"

    # Linear schedule bounds
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Cosine schedule offset
    cosine_s: float = 0.008

    # Sampling mode
    sampler: Literal["ddpm", "ddim"] = "ddpm"

    # DDIM sampling steps (only used when sampler == "ddim")
    ddim_steps: int = 50

    # DDIM eta (0 = deterministic, 1 = equivalent to DDPM)
    ddim_eta: float = 0.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Dataset
    dataset: Literal["mnist", "cifar10"] = "cifar10"

    # Image size (MNIST: 28, CIFAR-10: 32)
    image_size: int = 32

    # Batch size
    batch_size: int = 128

    # Number of training epochs
    epochs: int = 100

    # Learning rate
    learning_rate: float = 2e-4

    # LR schedule: "cosine", "linear", or "constant" (uses `learning_rate` only)
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"
    lr_max: float = 8e-4   # peak after warmup
    lr_min: float = 1e-4   # floor at end of decay
    lr_warmup_pct: float = 0.10  # first 10% of steps = linear warmup
    lr_decay_pct: float = 0.90   # by this % of steps, lr reaches lr_min

    # Adam optimizer parameters
    adam_betas: tuple[float, float] = (0.9, 0.999)

    # Exponential moving average (EMA) decay
    ema_decay: float = 0.9999

    # How often to save checkpoints and generate samples
    save_every: int = 10

    # Whether to save model checkpoints at all
    checkpoint: bool = False

    # How many samples to generate for logging
    num_samples: int = 16

    # Device: "cuda", "mps", or "cpu"
    device: str = "cuda"

    # DataLoader subprocesses (increase for GPU; 0 = main-process only)
    num_workers: int = 4

    # Pin CPU memory for faster GPU transfer (only meaningful for CUDA)
    pin_memory: bool = True

    # Mixed precision training
    use_amp: bool = False

    # Gradient clipping (disabled by default; DDPM is naturally stable)
    grad_clip: float = 0.0

    # Wandb logging
    wandb_project: str = "ddpm-experiments"
    wandb_entity: str | None = None  # Set to your wandb username/team

    # Random seed
    seed: int = 42


@dataclass
class Config:
    """Top-level configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
