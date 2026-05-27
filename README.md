# DDPM/DDIM Experiments

PyTorch implementations of **Denoising Diffusion Probabilistic Models** (DDPM) and **Denoising Diffusion Implicit Models** (DDIM) for MNIST and CIFAR-10.

---

## Models

Two denoiser architectures, following:
- **DDPM** — Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- **DDIM** — Song et al., "Denoising Diffusion Implicit Models" (2021)

| Architecture | Type | Best for |
|-------------|------|----------|
| **UNet** | Convolutional U-Net with residual blocks, self-attention, time embeddings | CIFAR-10, complex images |
| **MLP** | Dense residual blocks with GELU activations and time conditioning | MNIST (simpler, faster) |

Source code under `./src/`:
| File | Purpose |
|------|---------|
| `src/config.py` | Configuration dataclasses (model type, diffusion, training) |
| `src/model.py` | UNet + MLPDenoiser + factory (`create_model`) |
| `src/diffusion.py` | Gaussian diffusion: forward process, DDPM/DDIM sampling, loss |
| `src/dataset.py` | Data loading via HuggingFace `datasets` |
| `src/train.py` | Training loop with EMA, checkpointing, and wandb logging |

---

## Datasets

| Dataset   | Resolution | Channels | Source |
|-----------|------------|----------|--------|
| MNIST     | 28×28 → 32×32 | 1 (grayscale) | HuggingFace |
| CIFAR-10  | 32×32      | 3 (RGB)  | HuggingFace |

Datasets are auto-downloaded and cached under `./datasets/`.

---

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. MNIST with MLP (fast, small model)

```bash
python run.py --dataset mnist --model-type mlp --timesteps 200 --epochs 20 --batch-size 128
```

### 3. MNIST with UNet

```bash
python run.py --dataset mnist --model-type unet --model-channels 64 --channel-mult 1 2 2 --epochs 20
```

### 4. CIFAR-10 (UNet)

```bash
python run.py --dataset cifar10 --epochs 100 --batch-size 128
```

### 5. DDIM fast sampling

```bash
python run.py --dataset cifar10 --sampler ddim --ddim-steps 50
```

### 6. All CLI options

```bash
python run.py --help
```

---

## Configuration

All hyperparameters are configurable via CLI or by editing `src/config.py`:

| Group | Key Parameters |
|-------|---------------|
| **Model** | `--model-type` (`unet`/`mlp`), `--model-channels`, `--channel-mult`, `--mlp-hidden-dims`, `--mlp-activation` |
| **Training** | `--epochs`, `--batch-size`, `--lr`, `--device` (`mps`/`cuda`/`cpu`) |
| **Diffusion** | `--timesteps` (T), `--beta-schedule` (`linear`/`cosine`), `--sampler` (`ddpm`/`ddim`) |
| **DDIM** | `--ddim-steps`, `--ddim-eta` (0 = deterministic, 1 = DDPM-like) |

---

## MLP Architecture

The MLP denoiser uses residual dense blocks:

```
flatten(image) → Linear → [ResBlock × N] → Linear → unflatten → noise prediction
```

Each `ResBlock`: `Linear → +time_embed → GELU → Dropout → Linear → +skip`

Default MLP config for MNIST: `--mlp-hidden-dims 1024 2048 1024` (~2.7M params)

---

## Hardware

- **Primary target:** Apple Silicon (MPS)
- **Also supported:** CUDA, CPU fallback

---

## Logging

Loss curves and generated samples are tracked via **Weights & Biases (wandb)**:

```bash
python run.py --wandb-entity your_username
```

---

## Outputs

- **Checkpoints:** `./checkpoints/{dataset}/checkpoint_epoch_N.pt` and `final_model.pt`
- **Samples:** Logged to wandb every `--save-every` epochs
- **Loss curves:** Real-time in wandb dashboard

---

## Dependencies

Managed by `uv`. Key packages:
- `torch` (MPS + CUDA)
- `torchvision`
- `datasets` (HuggingFace)
- `wandb`
- `matplotlib`
