"""Diffusion process: forward diffusion, DDPM sampling, DDIM sampling, and loss.

Based on:
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Song et al., "Denoising Diffusion Implicit Models" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DiffusionConfig


def make_beta_schedule(config: DiffusionConfig) -> torch.Tensor:
    """Create the noise schedule (β values).

    Args:
        config: Diffusion configuration.

    Returns:
        betas: (T,) tensor of β values.
    """
    T = config.timesteps

    if config.beta_schedule == "linear":
        return torch.linspace(config.beta_start, config.beta_end, T, dtype=torch.float32)

    elif config.beta_schedule == "cosine":
        # Cosine schedule as proposed in "Improved DDPM" (Nichol & Dhariwal, 2021)
        s = config.cosine_s
        steps = T + 1
        t = torch.linspace(0, T, steps, dtype=torch.float32) / T
        alpha_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(betas, max=0.999)

    else:
        raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")


class GaussianDiffusion(nn.Module):
    """Gaussian diffusion process for DDPM training and DDPM/DDIM sampling.

    Trained with epsilon-prediction (predict the noise added to x_0).
    """

    def __init__(self, model: nn.Module, config: DiffusionConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.T = config.timesteps

        # Pre-compute diffusion constants
        betas = make_beta_schedule(config)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # ᾱ_{t-1}

        # Register as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Pre-compute coefficients for q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Pre-compute coefficients for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

        # Pre-compute DDPM sampling coefficients (Ho et al. Algorithm 2, Eq. 11)
        # x0_pred = x_t / √ᾱ_t  -  √(1-ᾱ_t)/√ᾱ_t · ε_θ
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "coeff_x0_eps",
            torch.sqrt(1.0 - alphas_cumprod) / torch.sqrt(alphas_cumprod),
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion: sample x_t given x_0.

        q(x_t | x_0) = N(√ᾱ_t · x_0, (1 - ᾱ_t)I)

        Args:
            x0: (B, C, H, W) clean images.
            t: (B,) timestep indices.
            noise: (B, C, H, W) optional pre-sampled noise.

        Returns:
            x_t: (B, C, H, W) noisy images.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def training_loss(self, x0: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the simplified DDPM training loss.

        L_simple = E_{t, x_0, ε} [ ||ε - ε_θ(x_t, t, y)||^2 ]

        Args:
            x0: (B, C, H, W) clean images.
            labels: (B,) optional class labels for conditioning.

        Returns:
            (loss, betas_t): Scalar MSE loss and the β values for the sampled timesteps.
        """
        B = x0.shape[0]
        device = x0.device

        t = torch.randint(0, self.T, (B,), device=device)
        betas_t = self.betas[t]

        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        predicted_noise = self.model(x_t, t, labels) if labels is not None else self.model(x_t, t)

        loss = F.mse_loss(predicted_noise, noise)

        return loss, betas_t

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """Single DDPM reverse diffusion step.

        Samples x_{t-1} from p_θ(x_{t-1} | x_t).

        Args:
            x_t: (B, C, H, W) current noisy image at time t.
            t: (B,) timestep indices.
            t_index: integer timestep (scalar) for indexing buffers.

        Returns:
            x_{t-1}: (B, C, H, W) denoised image for the previous timestep.
        """
        betas_t = self.betas[t_index]
        sqrt_recip_alpha_cumprod_t = self.sqrt_recip_alphas_cumprod[t_index]
        coeff_x0_eps_t = self.coeff_x0_eps[t_index]

        # Predict noise
        eps = self.model(x_t, t)

        # Estimate x_0 (Ho et al. Eq. 11)
        # x0 = (x_t − √(1−ᾱ_t)·ε_θ) / √ᾱ_t
        x0_pred = sqrt_recip_alpha_cumprod_t * x_t - coeff_x0_eps_t * eps

        # Clamp x0_pred for stability (optional, from DDPM paper)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Compute mean of posterior q(x_{t-1} | x_t, x_0)
        posterior_mean = (
            betas_t * torch.sqrt(self.alphas_cumprod_prev[t_index]) / (1.0 - self.alphas_cumprod[t_index])
        ) * x0_pred + (
            (1.0 - self.alphas_cumprod_prev[t_index]) * torch.sqrt(self.alphas[t_index]) / (1.0 - self.alphas_cumprod[t_index])
        ) * x_t

        # Add noise if t > 0
        if t_index > 0:
            noise = torch.randn_like(x_t)
            posterior_var = self.posterior_variance[t_index]
            return posterior_mean + torch.sqrt(posterior_var) * noise
        else:
            return posterior_mean

    @torch.no_grad()
    def ddim_sample_step(
        self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, eta: float
    ) -> torch.Tensor:
        """Single DDIM reverse diffusion step.

        Args:
            x_t: (B, C, H, W) current noisy image at time t.
            t: (B,) current timestep indices.
            t_prev: (B,) previous timestep indices in the DDIM trajectory.
            eta: stochasticity parameter (0 = deterministic, 1 = DDPM-like).

        Returns:
            x_{t-1}: (B, C, H, W) image for the previous DDIM step.
        """
        # Predict noise
        eps = self.model(x_t, t)

        # Get cumulative products
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_bar_t_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)

        # Predicted x_0
        x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev - eta ** 2 * self.posterior_variance[t].view(-1, 1, 1, 1)) * eps

        # Predicted x_{t-1}
        x_prev = torch.sqrt(alpha_bar_t_prev) * x0_pred + dir_xt

        if eta > 0:
            noise = torch.randn_like(x_t)
            sigma = eta * torch.sqrt(self.posterior_variance[t].view(-1, 1, 1, 1))
            x_prev = x_prev + sigma * noise

        return x_prev

    @torch.no_grad()
    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Generate images using DDPM or DDIM sampling.

        Args:
            shape: (B, C, H, W) shape of images to generate.
            device: torch device.

        Returns:
            Generated images in [-1, 1] range.
        """
        if self.config.sampler == "ddpm":
            return self._ddpm_sample(shape, device)
        elif self.config.sampler == "ddim":
            return self._ddim_sample(shape, device)
        else:
            raise ValueError(f"Unknown sampler: {self.config.sampler}")

    @torch.no_grad()
    def _ddpm_sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Full DDPM reverse process (T steps)."""
        x = torch.randn(shape, device=device)

        for t_index in reversed(range(self.T)):
            t = torch.full((shape[0],), t_index, device=device, dtype=torch.long)
            x = self.p_sample(x, t, t_index)

        return x

    @torch.no_grad()
    def _ddim_sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Full DDIM reverse process using skipped steps."""
        ddim_steps = min(self.config.ddim_steps, self.T)
        eta = self.config.ddim_eta

        # Create evenly spaced timestep trajectory
        step_ratio = max(1, self.T // ddim_steps)
        times = list(reversed(range(0, self.T, step_ratio)))
        times_next = times[1:] + [0]

        x = torch.randn(shape, device=device)

        for t_val, t_prev_val in zip(times, times_next):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            t_prev = torch.full((shape[0],), t_prev_val, device=device, dtype=torch.long)
            x = self.ddim_sample_step(x, t, t_prev, eta)

        return x
