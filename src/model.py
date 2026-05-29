"""U-Net model for DDPM/DDIM, following Ho et al. 2020.

Architecture:
- Encoder-decoder U-Net with skip connections
- Residual blocks with group normalization and time embedding conditioning
- Self-attention at specified resolutions
- Sinusoidal time step embeddings
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal position embedding for diffusion timesteps."""
    half_dim = embedding_dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb_scale
    )
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([emb.sin(), emb.cos()], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    """Projects sinusoidal time embedding to the model dimension."""

    def __init__(self, embed_dim: int, model_channels: int):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, model_channels * 4)
        self.linear2 = nn.Linear(model_channels * 4, model_channels * 4)
        self.act = nn.SiLU()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = get_timestep_embedding(t, self.linear1.in_features)
        emb = emb.to(dtype=self.linear1.weight.dtype)
        emb = self.act(self.linear1(emb))
        emb = self.act(self.linear2(emb))
        return emb


class ResBlock(nn.Module):
    """Residual block with group norm, time embedding conditioning, and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_channels: int,
        num_groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_channels, out_channels),
        )

        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention with group norm and residual connection."""

    def __init__(self, channels: int, num_groups: int = 32, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)

        scale = self.head_dim ** -0.5
        attn = torch.softmax(q @ k * scale, dim=-1)
        out = attn @ v

        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Strided convolution for downsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsampling followed by convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DownBlock(nn.Module):
    """One resolution level in the U-Net encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_channels: int,
        num_res_blocks: int,
        num_groups: int,
        dropout: float,
        has_attention: bool,
        downsample: bool,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        ch = in_channels
        for i in range(num_res_blocks):
            block_out = out_channels if i == num_res_blocks - 1 else out_channels
            self.res_blocks.append(
                ResBlock(ch, block_out, time_emb_channels, num_groups, dropout)
            )
            ch = block_out

        self.attentions = nn.ModuleList()
        if has_attention:
            for _ in range(num_res_blocks):
                self.attentions.append(AttentionBlock(out_channels, num_groups))
        else:
            self.attentions.append(nn.Identity())

        self.downsample = Downsample(out_channels) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x, t_emb)
            if i < len(self.attentions):
                x = self.attentions[i](x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """One resolution level in the U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_emb_channels: int,
        num_res_blocks: int,
        num_groups: int,
        dropout: float,
        has_attention: bool,
        upsample: bool,
    ):
        super().__init__()
        self.upsample = Upsample(in_channels) if upsample else nn.Identity()

        self.res_blocks = nn.ModuleList()
        ch = in_channels + skip_channels
        for i in range(num_res_blocks):
            block_out = out_channels if i == num_res_blocks - 1 else out_channels
            self.res_blocks.append(
                ResBlock(ch, block_out, time_emb_channels, num_groups, dropout)
            )
            ch = block_out

        self.attentions = nn.ModuleList()
        if has_attention:
            for _ in range(num_res_blocks):
                self.attentions.append(AttentionBlock(out_channels, num_groups))
        else:
            self.attentions.append(nn.Identity())

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x, t_emb)
            if i < len(self.attentions):
                x = self.attentions[i](x)
        return x


class UNet(nn.Module):
    """U-Net for diffusion models (DDPM / DDIM)."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        in_channels = config.in_channels
        model_channels = config.model_channels
        out_channels = config.out_channels
        num_res_blocks = config.num_res_blocks
        channel_mult = config.channel_mult
        attention_resolutions = set(config.attention_resolutions)
        num_groups = config.num_groups
        dropout = config.dropout

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = TimeEmbedding(model_channels, model_channels)

        # Class embedding (added to time embedding for conditioning)
        self.num_classes = config.num_classes
        if self.num_classes > 0:
            self.class_embed = nn.Embedding(self.num_classes, time_embed_dim)

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # --- Encoder ---
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        num_levels = len(channel_mult)

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            is_last = level == num_levels - 1
            self.down_blocks.append(
                DownBlock(
                    in_channels=ch,
                    out_channels=out_ch,
                    time_emb_channels=time_embed_dim,
                    num_res_blocks=num_res_blocks,
                    num_groups=num_groups,
                    dropout=dropout,
                    has_attention=(out_ch in [model_channels * m for m in channel_mult if m in attention_resolutions or config.attention_all]),
                    downsample=not is_last,
                )
            )
            ch = out_ch

        # --- Middle ---
        mid_ch = model_channels * channel_mult[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_embed_dim, num_groups, dropout)
        self.mid_attn = AttentionBlock(mid_ch, num_groups)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_embed_dim, num_groups, dropout)

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            skip_ch = model_channels * channel_mult[num_levels - 1 - level]
            is_first = level == 0
            self.up_blocks.append(
                UpBlock(
                    in_channels=ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    time_emb_channels=time_embed_dim,
                    num_res_blocks=num_res_blocks,
                    num_groups=num_groups,
                    dropout=dropout,
                    has_attention=(out_ch in [model_channels * m for m in channel_mult if m in attention_resolutions or config.attention_all]),
                    upsample=not is_first,
                )
            )
            ch = out_ch

        # --- Output ---
        self.out_norm = nn.GroupNorm(num_groups, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W) noisy input.
            t: (B,) timestep indices.
            labels: (B,) optional class labels for conditioning.

        Returns:
            (B, C, H, W) predicted noise (epsilon prediction).
        """
        t_emb = self.time_embed(t)
        if labels is not None and self.num_classes > 0:
            t_emb = t_emb + self.class_embed(labels)

        h = self.input_conv(x)

        # Encoder
        skips: List[torch.Tensor] = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, t_emb)
            skips.append(skip)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            h = up_block(h, skip, t_emb)

        # Output
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)

        return h


# ---------------------------------------------------------------------------
#  MLP Denoiser (for simple datasets like MNIST)
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """Simple MLP denoiser with FiLM time conditioning (no skip connections).

    FiLM modulates hidden features via scale+shift from the time embedding.
    No skip connections (unlike MLPDenoiser) to avoid dead-gradient issues.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        in_channels = config.in_channels
        pixel_dim = in_channels * 28 * 28
        hidden = 1024

        self.time_embed = TimeEmbedding(config.model_channels, config.model_channels)
        time_dim = config.model_channels * 4
        self.time_proj = nn.Linear(time_dim, hidden)

        self.in_proj = nn.Linear(pixel_dim, hidden)
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, pixel_dim)

        # FiLM projectors (time → scale + shift per layer)
        self.film1 = nn.Linear(time_dim, 2 * hidden)
        self.film2 = nn.Linear(time_dim, 2 * hidden)
        nn.init.zeros_(self.film1.weight); nn.init.zeros_(self.film1.bias)
        nn.init.zeros_(self.film2.weight); nn.init.zeros_(self.film2.bias)

        # Class embedding (added to time embedding for conditioning)
        self.num_classes = config.num_classes
        if self.num_classes > 0:
            self.class_embed = nn.Embedding(self.num_classes, time_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        B, C, H, W = x.shape
        flat = x.reshape(B, -1)
        t_emb = self.time_embed(t)
        if labels is not None and self.num_classes > 0:
            t_emb = t_emb + self.class_embed(labels)

        h = self.in_proj(flat) + self.time_proj(t_emb)

        # FiLM block 1
        s1, sh1 = self.film1(t_emb).chunk(2, dim=-1)
        h = self.l1(h)
        h = h * (1.0 + s1) + sh1
        h = torch.nn.functional.gelu(h)

        # FiLM block 2
        s2, sh2 = self.film2(t_emb).chunk(2, dim=-1)
        h = self.l2(h)
        h = h * (1.0 + s2) + sh2
        h = torch.nn.functional.gelu(h)

        out = self.out_proj(h)
        return out.reshape(B, C, H, W)


class MLPDenoiser(nn.Module):
    """Dense denoiser for DDPM/DDIM — predicts noise from flattened images + timestep.

    Pixel (y, x) coordinates are concatenated to the flattened input so the
    model knows spatial position.  FiLM-conditioned residual MLP blocks with
    zero-initialised time projections for stable training.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        in_channels = config.in_channels
        hidden_dims = config.mlp_hidden_dims
        time_dim = config.mlp_time_dim
        activation = config.mlp_activation

        # Time embedding
        self.time_embed = TimeEmbedding(config.model_channels, config.model_channels)
        time_out = config.model_channels * 4
        self.time_proj = nn.Linear(time_out, time_dim)

        # Class embedding (added to time embedding)
        self.num_classes = config.num_classes
        if self.num_classes > 0:
            self.class_embed = nn.Embedding(self.num_classes, time_out)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()

        self.input_proj: nn.Linear | None = None
        self.output_proj: nn.Linear | None = None

        self.blocks = nn.ModuleList()
        for _ in hidden_dims:
            self.blocks.append(_ResMLPBlock(1, 1, time_dim, self.act, config.dropout))

        self.pos_coords: torch.Tensor | None = None
        self._built = False

    def _build(self, x: torch.Tensor) -> None:
        B, C, H, W = x.shape
        flat_dim = C * H * W
        hidden_dims = self.config.mlp_hidden_dims
        time_dim = self.config.mlp_time_dim

        ys = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1).expand(H, W) / H
        xs = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(0).expand(H, W) / W
        self.pos_coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1)

        self.input_proj = nn.Linear(flat_dim + 2 * H * W, hidden_dims[0], device=x.device)
        self.output_proj = nn.Linear(hidden_dims[-1], flat_dim, device=x.device)

        for i, block in enumerate(self.blocks):
            in_dim = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            block.build(in_dim, hidden_dims[i], x.device)

        self._built = True

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        B, C, H, W = x.shape
        if not self._built:
            self._build(x)

        flat = x.reshape(B, -1)
        pos = self.pos_coords.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        h = self.input_proj(torch.cat([flat, pos], dim=-1))

        t_emb = self.time_embed(t)
        if labels is not None and self.num_classes > 0:
            t_emb = t_emb + self.class_embed(labels)
        t_emb = self.time_proj(t_emb)

        for block in self.blocks:
            h = block(h, t_emb)

        out = self.output_proj(h)
        return out.reshape(B, C, H, W)


class _ResMLPBlock(nn.Module):
    """Residual MLP block with FiLM time conditioning and pre-norm LayerNorm.

    Structure (DiT-style):
        LayerNorm → Linear(in_dim, hidden_dim)
        → FiLM (scale + shift from time embedding)
        → Activation → Linear(hidden_dim, hidden_dim)
        + residual connection (with projection if dims differ)

    LayerNorm before the block stabilises training and prevents dead gradients
    from the FiLM conditioning.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        time_dim: int,
        activation: nn.Module,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        self.linear1: nn.Linear | None = None
        self.time_proj = nn.Linear(time_dim, 2 * hidden_dim)  # scale + shift
        self.act = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2: nn.Linear | None = None
        self.norm: nn.LayerNorm | None = None

        self.skip: nn.Module | None = None
        self._built = False

    def build(self, in_dim: int, hidden_dim: int, device: torch.device) -> None:
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim, device=device)
        self.time_proj = nn.Linear(self.time_dim, 2 * hidden_dim, device=device)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.norm = nn.LayerNorm(in_dim, device=device)
        self.skip = (
            nn.Linear(in_dim, hidden_dim, device=device)
            if in_dim != hidden_dim
            else nn.Identity()
        )
        self._built = True

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Pre-norm (DiT-style): normalise before block, not after
        h = self.norm(x)
        h = self.linear1(h)
        scale_shift = self.time_proj(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def create_model(config: ModelConfig) -> nn.Module:
    """Create a denoiser model based on config.model_type.

    Args:
        config: Model configuration.

    Returns:
        A denoiser nn.Module (UNet or MLPDenoiser).
    """
    if config.model_type == "unet":
        return UNet(config)
    elif config.model_type == "mlp":
        return MLPDenoiser(config)
    elif config.model_type == "simple":
        return SimpleMLP(config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
