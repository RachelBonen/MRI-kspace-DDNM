# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# metrics.py
# PSNR and SSIM computed consistently for every method. Images live in the
# model space [-1, 1]; we rescale to [0, 1] with a fixed data_range before
# measuring. For complex data (2 channels) the brief asks for PSNR/SSIM on the
# real and imaginary parts *separately* - we compute both and also return their
# mean for the summary tables.

from typing import Dict

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def _to01(a: np.ndarray) -> np.ndarray:
    """Map a [-1, 1] image to [0, 1] and clip."""
    return np.clip((a + 1.0) / 2.0, 0.0, 1.0)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(sk_psnr(_to01(b), _to01(a), data_range=1.0))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(sk_ssim(_to01(b), _to01(a), data_range=1.0))


def compute_metrics(rec, gt, channels: int) -> Dict[str, float]:
    """Return PSNR/SSIM for a reconstruction vs ground truth.

    Args:
        rec, gt: arrays/tensors of shape (C, H, W) in [-1, 1] (model space).
        channels: 1 (magnitude) or 2 (complex real/imag).

    Returns dict with keys: psnr, ssim  (plus per-component keys for complex).
    """
    rec = np.asarray(rec.detach().cpu()) if hasattr(rec, "detach") else np.asarray(rec)
    gt = np.asarray(gt.detach().cpu()) if hasattr(gt, "detach") else np.asarray(gt)

    if channels == 2:
        pr_re, pr_im = _psnr(rec[0], gt[0]), _psnr(rec[1], gt[1])
        ss_re, ss_im = _ssim(rec[0], gt[0]), _ssim(rec[1], gt[1])
        return {
            "psnr": 0.5 * (pr_re + pr_im), "ssim": 0.5 * (ss_re + ss_im),
            "psnr_real": pr_re, "psnr_imag": pr_im,
            "ssim_real": ss_re, "ssim_imag": ss_im,
        }
    return {"psnr": _psnr(rec[0], gt[0]), "ssim": _ssim(rec[0], gt[0])}
