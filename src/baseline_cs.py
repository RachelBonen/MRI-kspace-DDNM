# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# baseline_cs.py
# Classical, non-deep-learning baseline (rubric requirement), matching the
# methods the reference GAN paper compares against:
#
#   * zero_fill        - inverse FFT of the zero-filled k-space (lower bound).
#   * cs_tv_recon      - Compressed-Sensing MRI with Total-Variation
#                        regularization, solved by FISTA. This is the strong
#                        classical opponent: it enforces data consistency with
#                        the measured k-space while promoting a piecewise-smooth
#                        (sparse-gradient) image - the sparsity prior from the
#                        course's compressed-sensing material.
#
# Solves:  min_x  1/2 || M.F x - y ||^2  +  lambda * TV(x)
# TV is handled by Chambolle's projection algorithm as the prox operator.

import torch

from .kspace import fft2c, ifft2c


# ---------------------------------------------------------------------------
# discrete gradient / divergence (adjoint pair) for a single 2D image
# ---------------------------------------------------------------------------
def _grad(u: torch.Tensor) -> torch.Tensor:
    """Forward-difference gradient of a real (H, W) image -> (2, H, W)."""
    gx = torch.zeros_like(u)
    gy = torch.zeros_like(u)
    gx[:-1, :] = u[1:, :] - u[:-1, :]
    gy[:, :-1] = u[:, 1:] - u[:, :-1]
    return torch.stack([gx, gy], dim=0)


def _div(p: torch.Tensor) -> torch.Tensor:
    """Divergence = negative adjoint of _grad. p: (2, H, W) -> (H, W)."""
    px, py = p[0], p[1]
    dx = torch.zeros_like(px)
    dy = torch.zeros_like(py)
    dx[0, :] = px[0, :]
    dx[1:-1, :] = px[1:-1, :] - px[:-2, :]
    dx[-1, :] = -px[-2, :]
    dy[:, 0] = py[:, 0]
    dy[:, 1:-1] = py[:, 1:-1] - py[:, :-2]
    dy[:, -1] = -py[:, -2]
    return dx + dy


def _prox_tv_real(b: torch.Tensor, weight: float, n_iter: int = 20) -> torch.Tensor:
    """Chambolle prox: argmin_x 1/2||x-b||^2 + weight*TV(x), for real (H,W)."""
    if weight <= 0:
        return b
    tau = 0.25
    p = torch.zeros((2,) + b.shape, device=b.device, dtype=b.dtype)
    for _ in range(n_iter):
        gp = _grad(_div(p) - b / weight)
        denom = 1.0 + tau * torch.sqrt((gp ** 2).sum(dim=0, keepdim=True))
        p = (p + tau * gp) / denom
    return b - weight * _div(p)


def _prox_tv_complex(b: torch.Tensor, weight: float, n_iter: int = 20) -> torch.Tensor:
    """Apply the TV prox to the real and imaginary parts of a complex image."""
    re = _prox_tv_real(b.real.contiguous(), weight, n_iter)
    im = _prox_tv_real(b.imag.contiguous(), weight, n_iter)
    return torch.complex(re, im)


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def zero_fill(y: torch.Tensor) -> torch.Tensor:
    """Zero-filled reconstruction: complex image = IFFT of masked k-space."""
    return ifft2c(y)


def cs_tv_recon(
    y: torch.Tensor,          # measured k-space (H, W) complex
    mask2d: torch.Tensor,     # (H, W) real 0/1
    lam: float = 0.01,
    n_iter: int = 100,
    tv_iter: int = 10,
    step: float = 1.0,
) -> torch.Tensor:
    """FISTA solution of TV-regularized CS-MRI. Returns complex (H, W) image.

    A = M.F is orthonormal-composed-with-a-0/1 mask, so the data-term gradient
    has Lipschitz constant 1 and a step of 1.0 is stable.
    """
    x = ifft2c(y)          # zero-filled initialization
    z = x.clone()
    t = 1.0
    for _ in range(n_iter):
        # gradient of 1/2||M.F z - y||^2  =  F^H ( M.(F z) - y )
        grad = ifft2c(mask2d * fft2c(z) - y)
        x_new = _prox_tv_complex(z - step * grad, lam * step, tv_iter)
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x
