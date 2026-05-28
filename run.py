#!/usr/bin/env python3
"""CLI entry point for DDPM/DDIM training.

Usage:
    python run.py --dataset mnist --epochs 20 --batch-size 64
    python run.py --dataset cifar10 --sampler ddim --ddim-steps 50
    python run.py --dataset mnist --model-type mlp --epochs 20  # MLP on MNIST
"""

import argparse

from src.config import Config, DiffusionConfig, ModelConfig, TrainingConfig
from src.train import train


def main():
    parser = argparse.ArgumentParser(description="Train a DDPM/DDIM diffusion model")

    # Dataset
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["mnist", "cifar10", "circle"],
                        help="Dataset to train on")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Image size (default: 28 for MLP, 32 for UNet)")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lr-schedule", type=str, default="cosine",
                        choices=["cosine", "linear", "constant"],
                        help="LR schedule: cosine/linear (with warmup→decay→min) or constant")
    parser.add_argument("--lr-max", type=float, default=8e-4,
                        help="Peak LR after warmup")
    parser.add_argument("--lr-min", type=float, default=1e-4,
                        help="Minimum LR at end of decay")
    parser.add_argument("--lr-warmup-pct", type=float, default=0.10,
                        help="Fraction of steps for linear warmup")
    parser.add_argument("--lr-decay-pct", type=float, default=0.90,
                        help="Fraction of steps where LR reaches min")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["mps", "cuda", "cpu"],
                        help="Device to use")

    # Model
    parser.add_argument("--model-type", type=str, default="unet",
                        choices=["unet", "mlp"],
                        help="Model architecture: 'unet' (convolutional) or 'mlp' (dense, for MNIST)")
    parser.add_argument("--model-channels", type=int, default=128,
                        help="Base model channels (UNet only)")
    parser.add_argument("--channel-mult", type=int, nargs="+", default=[1, 2, 2, 2],
                        help="Channel multipliers per level (UNet only)")
    parser.add_argument("--num-res-blocks", type=int, default=2,
                        help="Residual blocks per level (UNet only)")
    parser.add_argument("--mlp-hidden-dims", type=int, nargs="+", default=[768, 1536, 768],
                        help="Hidden layer dimensions for MLP model")
    parser.add_argument("--mlp-activation", type=str, default="gelu",
                        choices=["gelu", "relu", "silu"],
                        help="Activation function for MLP hidden layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Diffusion
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--beta-schedule", type=str, default="linear",
                        choices=["linear", "cosine"],
                        help="Noise schedule type")
    parser.add_argument("--sampler", type=str, default="ddpm",
                        choices=["ddpm", "ddim"],
                        help="Sampling method")
    parser.add_argument("--ddim-steps", type=int, default=50,
                        help="DDIM sampling steps")
    parser.add_argument("--ddim-eta", type=float, default=0.0,
                        help="DDIM stochasticity (0=deterministic, 1=DDPM-like)")

    # Logging
    parser.add_argument("--wandb-project", type=str, default="ddpm-experiments",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Weights & Biases entity (username/team)")

    # Misc
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--checkpoint", action="store_true", default=False,
                        help="Enable model checkpointing (off by default)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Auto-detect image size: MLP uses native 28 for MNIST (no upscale needed),
    # UNet needs power-of-2 size for downsampling layers.
    if args.image_size is None:
        args.image_size = 28 if args.model_type == "mlp" else 32

    config = Config(
        model=ModelConfig(
            model_type=args.model_type,
            in_channels=3,  # auto-detected from data
            out_channels=3,
            model_channels=args.model_channels,
            channel_mult=tuple(args.channel_mult),
            num_res_blocks=args.num_res_blocks,
            attention_resolutions=(16,),
            dropout=args.dropout,
            mlp_hidden_dims=tuple(args.mlp_hidden_dims),
            mlp_activation=args.mlp_activation,
        ),
        diffusion=DiffusionConfig(
            timesteps=args.timesteps,
            beta_schedule=args.beta_schedule,
            sampler=args.sampler,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
        ),
        training=TrainingConfig(
            dataset=args.dataset,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            lr_schedule=args.lr_schedule,
            lr_max=args.lr_max,
            lr_min=args.lr_min,
            lr_warmup_pct=args.lr_warmup_pct,
            lr_decay_pct=args.lr_decay_pct,
            device=args.device,
            num_workers=4,
            pin_memory=True,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            save_every=args.save_every,
            checkpoint=args.checkpoint,
            seed=args.seed,
        ),
    )

    print("=" * 60)
    print("Configuration:")
    print(f"  Dataset:      {config.training.dataset}")
    print(f"  Image size:   {config.training.image_size}")
    print(f"  Model type:   {config.model.model_type}")
    if config.model.model_type == "mlp":
        print(f"  MLP dims:     {config.model.mlp_hidden_dims}")
        print(f"  MLP act:      {config.model.mlp_activation}")
    print(f"  Batch size:   {config.training.batch_size}")
    print(f"  Epochs:       {config.training.epochs}")
    print(f"  Learning rate:{config.training.learning_rate}")
    print(f"  LR schedule:  {config.training.lr_schedule}")
    print(f"  Timesteps:    {config.diffusion.timesteps}")
    print(f"  Beta schedule:{config.diffusion.beta_schedule}")
    print(f"  Sampler:      {config.diffusion.sampler}")
    print(f"  Device:       {config.training.device}")
    print("=" * 60)

    train(config)


if __name__ == "__main__":
    main()
