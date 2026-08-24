# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# kspace.py
# Shared k-space utilities used by BOTH the classical baseline and the
# diffusion (DDNM) reconstruction, so the two methods see *identical* masks,
# transforms and normalization (required for a fair comparison).
#
#   * centered 2D FFT / IFFT (fftshift convention, orthonormal norm)
#   * 1D variable-density row mask (Gaussian, dense center), seeded
#   * forward operator A = mask . FFT  and its adjoint / zero-filled input
#
# Convention: images are complex tensors of shape (..., H, W). The measurement
# operator keeps whole *rows* of k-space (phase-encode lines), matching the
# 1D Cartesian undersampling described in the project brief.

from typing import Tuple

import numpy as np
import torch
import torch.fft as fft


# ---------------------------------------------------------------------------
# Centered, orthonormal 2D Fourier transforms
# ---------------------------------------------------------------------------
def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT (image -> k-space)."""
    return fft.fftshift(fft.fft2(fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
                        dim=(-2, -1))


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    """Centered 2D inverse FFT (k-space -> image)."""
    return fft.fftshift(fft.ifft2(fft.ifftshift(k, dim=(-2, -1)), norm="ortho"),
                        dim=(-2, -1))


# ---------------------------------------------------------------------------
# Variable-density 1D row mask
# ---------------------------------------------------------------------------
def gaussian_1d_mask(
    height: int,
    keep_frac: float,
    seed: int = 0,
    std_scale: float = 6.0,
    center_lines: int = 0,
) -> np.ndarray:
    """Boolean row mask of length `height`.

    Rows (phase-encode lines) are drawn from a normal distribution centered on
    the middle of k-space, so low frequencies are densely sampled and high
    frequencies sparsely - exactly the variable-density scheme in the brief.

    Args:
        height: number of k-space rows (image height).
        keep_frac: fraction of rows to keep (e.g. 0.2, 0.3, 0.5).
        seed: RNG seed for reproducibility.
        std_scale: std of the sampling Gaussian = height / std_scale.
        center_lines: optionally force-keep this many central lines (ACS band).
                      Set 0 to follow the brief exactly (pure Gaussian).

    Returns:
        mask: np.bool_ array of shape (height,), True = sampled.
    """
    rng = np.random.default_rng(seed)
    n_keep = int(round(keep_frac * height))
    mask = np.zeros(height, dtype=bool)

    if center_lines > 0:
        c0 = height // 2 - center_lines // 2
        mask[c0:c0 + center_lines] = True

    std = height / std_scale
    # draw until we have n_keep unique in-range rows
    while mask.sum() < n_keep:
        draws = rng.normal(loc=height / 2.0, scale=std,
                           size=(n_keep - mask.sum()) * 2)
        idx = np.round(draws).astype(int)
        idx = idx[(idx >= 0) & (idx < height)]
        for j in idx:
            if mask.sum() >= n_keep:
                break
            mask[j] = True
    return mask


def mask_to_2d(mask_1d: np.ndarray, width: int, device=None,
               dtype=torch.float32) -> torch.Tensor:
    """Broadcast a 1D row mask (H,) to a 2D k-space mask (H, W)."""
    m = torch.from_numpy(mask_1d.astype(np.float32))[:, None].repeat(1, width)
    return m.to(device=device, dtype=dtype)


def seeded_row_mask(size: int, ratio: float, base_seed: int, idx: int,
                    std_scale: float = 6.0, center_lines: int = 0,
                    device=None) -> torch.Tensor:
    """Deterministic per-(slice, ratio) 2D mask so that the baseline and the
    diffusion model see IDENTICAL undersampling (required for a fair
    comparison). The seed is a stable function of the slice index and ratio."""
    seed = base_seed * 1000003 + idx * 1009 + int(round(ratio * 100))
    m1d = gaussian_1d_mask(size, ratio, seed=seed, std_scale=std_scale,
                           center_lines=center_lines)
    return mask_to_2d(m1d, size, device=device)


# ---------------------------------------------------------------------------
# channel <-> complex helpers (models work in real channels; operator in C)
# ---------------------------------------------------------------------------
def to_complex(x: torch.Tensor, channels: int) -> torch.Tensor:
    """(..., C, H, W) real -> (..., H, W) complex."""
    if channels == 2:
        return torch.complex(x[..., 0, :, :], x[..., 1, :, :])
    return x[..., 0, :, :].to(torch.complex64)


def from_complex(z: torch.Tensor, channels: int) -> torch.Tensor:
    """(..., H, W) complex -> (..., C, H, W) real."""
    if channels == 2:
        return torch.stack([z.real, z.imag], dim=-3)
    return z.real.unsqueeze(-3)


# ---------------------------------------------------------------------------
# Forward model  y = M . FFT(x)   and the zero-filled input
# ---------------------------------------------------------------------------
def subsample_kspace(
    image: torch.Tensor, mask2d: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the accelerated-acquisition forward model.

    Args:
        image: complex image tensor (..., H, W).
        mask2d: real 0/1 mask (H, W) broadcastable to `image`.

    Returns:
        y:    measured (masked) k-space, complex, zeros on unsampled rows.
        x_zf: zero-filled reconstruction = IFFT(y), complex (the model input).
    """
    k = fft2c(image)
    y = k * mask2d
    x_zf = ifft2c(y)
    return y, x_zf


def data_consistency(x0: torch.Tensor, y: torch.Tensor,
                     mask2d: torch.Tensor) -> torch.Tensor:
    """DDNM range/null-space correction for the Fourier+mask operator.

    Replaces the *sampled* k-space rows of the current estimate x0 with the
    measured data y, keeping x0's values on the unsampled (null-space) rows:

        x0_dc = IFFT( M . y  +  (1 - M) . FFT(x0) )

    For orthonormal FFT and a 0/1 row-selection mask this is exactly
    x0 - A^+(A x0 - y), i.e. DDNM's x_hat_0 correction (noiseless case).
    """
    k = fft2c(x0)
    k_dc = mask2d * y + (1.0 - mask2d) * k
    return ifft2c(k_dc)
