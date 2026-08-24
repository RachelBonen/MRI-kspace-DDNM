# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# diffusion.py
# Gaussian (DDPM) diffusion process: the noise schedule, the forward
# corruption q(x_t | x_0), the training objective (predict epsilon), and
# ancestral DDPM / deterministic DDIM samplers used for sanity checks and,
# later, as the backbone for data-consistency reconstruction (sample.py).

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    """Return the betas for the forward diffusion process."""
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)
    if schedule == "cosine":
        # Nichol & Dhariwal (2021) cosine schedule.
        steps = timesteps + 1
        s = 0.008
        x = torch.linspace(0, timesteps, steps)
        acp = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return torch.clip(betas, 1e-4, 0.999)
    raise ValueError(f"unknown schedule: {schedule}")


def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    """Gather values from `a` at indices `t` and reshape to broadcast over `shape`."""
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Holds the diffusion constants and implements training/sampling.

    The model passed in must map (x_t, t) -> predicted noise epsilon.
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 256,
        channels: int = 1,
        timesteps: int = 1000,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps

        betas = make_beta_schedule(schedule, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # register as buffers so they move with .to(device) and are saved.
        reg = self.register_buffer
        reg("betas", betas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        reg("posterior_variance", posterior_var)
        reg("posterior_log_variance", torch.log(posterior_var.clamp(min=1e-20)))
        reg("posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        reg("posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    # ---- forward process ----
    def q_sample(self, x0, t, noise=None):
        """Sample x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def predict_x0_from_noise(self, x_t, t, noise):
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # ---- training objective ----
    def p_losses(self, x0, t, noise=None, loss_type: str = "l2"):
        """Standard DDPM loss: predict the noise added at step t."""
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t)
        if loss_type == "l1":
            return F.l1_loss(pred, noise)
        if loss_type == "l2":
            return F.mse_loss(pred, noise)
        if loss_type == "huber":
            return F.smooth_l1_loss(pred, noise)
        raise ValueError(loss_type)

    def forward(self, x0, loss_type: str = "l2"):
        """Convenience: sample a random t per image and return the loss."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        return self.p_losses(x0, t, loss_type=loss_type)

    # ---- reverse process (samplers, used for sanity checks) ----
    @torch.no_grad()
    def p_sample(self, x_t, t):
        """One ancestral DDPM reverse step."""
        noise_pred = self.model(x_t, t)
        x0 = self.predict_x0_from_noise(x_t, t, noise_pred).clamp(-1, 1)
        mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x0
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if t[0] == 0:
            return mean
        var = _extract(self.posterior_variance, t, x_t.shape)
        return mean + torch.sqrt(var) * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, batch_size: int = 4, device: Optional[torch.device] = None):
        """Full ancestral DDPM sampling loop from pure noise."""
        device = device or self.betas.device
        x = torch.randn(batch_size, self.channels, self.image_size,
                        self.image_size, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x

    @torch.no_grad()
    def ddim_sample(self, batch_size: int = 4, steps: int = 50, eta: float = 0.0,
                    device: Optional[torch.device] = None):
        """Deterministic (eta=0) DDIM sampling with a reduced step count."""
        device = device or self.betas.device
        times = torch.linspace(self.timesteps - 1, 0, steps + 1).long().to(device)
        x = torch.randn(batch_size, self.channels, self.image_size,
                        self.image_size, device=device)
        for i in range(steps):
            t = torch.full((batch_size,), times[i], device=device, dtype=torch.long)
            t_next = times[i + 1]
            ac_t = _extract(self.alphas_cumprod, t, x.shape)
            ac_next = self.alphas_cumprod[t_next]
            noise_pred = self.model(x, t)
            x0 = self.predict_x0_from_noise(x, t, noise_pred).clamp(-1, 1)
            sigma = eta * torch.sqrt((1 - ac_next) / (1 - ac_t) *
                                     (1 - ac_t / ac_next))
            dir_xt = torch.sqrt(1 - ac_next - sigma ** 2) * noise_pred
            x = torch.sqrt(ac_next) * x0 + dir_xt
            if eta > 0 and t_next > 0:
                x = x + sigma * torch.randn_like(x)
        return x
