# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# sample.py
# Reconstruct subsampled MRI with the trained diffusion prior using DDNM
# (Denoising Diffusion Null-space Model, Wang et al. ICLR 2023).
#
# At every reverse-diffusion step we:
#   (A) let the prior denoise one step and form the clean estimate x_hat_0;
#   (B) apply k-space DATA CONSISTENCY: overwrite the sampled rows of
#       x_hat_0 with the measured data y, keeping the model's guess only on the
#       unsampled (null-space) rows  ->  x0_dc = IFFT( M.y + (1-M).FFT(x_hat0) ).
# One trained prior handles every acceleration ratio; only the mask changes.
#
# Example:
#   python -m src.diffusion.sample \
#       --ckpt runs/ixi_t1_mag/latest.pt --data-root data/ixi_t1_test \
#       --ratios 0.2,0.3,0.5 --steps 100 --num-slices 50 \
#       --out results/ddnm --save-images

import argparse
import os

import numpy as np
import torch

from ..kspace import (data_consistency, from_complex, seeded_row_mask,
                      subsample_kspace, to_complex)
from .dataset import MRISliceDataset
from .diffusion import GaussianDiffusion
from .model import UNet


# ---------------------------------------------------------------------------
# load a trained prior from a checkpoint
# ---------------------------------------------------------------------------
def load_diffusion(ckpt_path: str, device) -> GaussianDiffusion:
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    mults = tuple(int(m) for m in str(a["channel_mults"]).split(","))
    attn = tuple(int(x) for x in str(a["attn_res"]).split(",") if x != "")
    model = UNet(
        in_channels=a["channels"], base_channels=a["base_channels"],
        channel_mults=mults, num_res_blocks=a["num_res_blocks"],
        attn_resolutions=attn, image_size=a["image_size"],
    ).to(device)
    # prefer EMA weights for sampling
    state = ckpt.get("ema", ckpt["model"])
    model.load_state_dict(state)
    model.eval()
    diffusion = GaussianDiffusion(
        model, image_size=a["image_size"], channels=a["channels"],
        timesteps=a["timesteps"], schedule=a["schedule"],
    ).to(device)
    return diffusion, a["channels"], a["image_size"]


# ---------------------------------------------------------------------------
# DDNM reconstruction loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def reconstruct_ddnm(
    diffusion: GaussianDiffusion,
    y: torch.Tensor,          # measured k-space (B, H, W) complex
    mask2d: torch.Tensor,     # (H, W) real 0/1
    channels: int,
    steps: int = 100,
    sampler: str = "ddim",
    eta: float = 0.0,
    device=None,
) -> torch.Tensor:
    """Return the restored image in model space (B, C, H, W), values in [-1, 1]."""
    device = device or diffusion.betas.device
    B = y.shape[0]
    size = diffusion.image_size
    acp = diffusion.alphas_cumprod

    # respaced timestep sub-sequence T-1 ... 0
    times = torch.linspace(diffusion.timesteps - 1, 0, steps + 1).long().to(device)

    x = torch.randn(B, channels, size, size, device=device)
    for i in range(steps):
        t = torch.full((B,), int(times[i]), device=device, dtype=torch.long)
        t_next = int(times[i + 1])

        # (1) denoiser -> predicted noise and clean estimate
        eps = diffusion.model(x, t)
        x0 = diffusion.predict_x0_from_noise(x, t, eps).clamp(-1, 1)

        # (2) DDNM data-consistency in k-space
        z0 = to_complex(x0, channels)
        z0 = data_consistency(z0, y, mask2d)
        x0 = from_complex(z0, channels).clamp(-1, 1)

        ac_next = acp[t_next]
        if sampler == "ddim":
            sigma = eta * torch.sqrt(
                (1 - ac_next) / (1 - acp[t]) * (1 - acp[t] / ac_next).clamp(min=0)
            ).view(B, 1, 1, 1) if t_next > 0 else torch.zeros(1, device=device)
            dir_xt = torch.sqrt((1 - ac_next - sigma ** 2).clamp(min=0)) * eps
            x = torch.sqrt(ac_next) * x0 + dir_xt
            if eta > 0 and t_next > 0:
                x = x + sigma * torch.randn_like(x)
        else:  # ancestral DDPM step using the corrected x0
            coef1 = diffusion.posterior_mean_coef1[t].view(B, 1, 1, 1)
            coef2 = diffusion.posterior_mean_coef2[t].view(B, 1, 1, 1)
            mean = coef1 * x0 + coef2 * x
            if t_next < 0 or int(times[i]) == 0:
                x = mean
            else:
                var = diffusion.posterior_variance[t].view(B, 1, 1, 1)
                x = mean + torch.sqrt(var) * torch.randn_like(x)

    # final estimate + one last hard data-consistency projection
    eps = diffusion.model(x, torch.zeros(B, device=device, dtype=torch.long))
    x0 = diffusion.predict_x0_from_noise(
        x, torch.zeros(B, device=device, dtype=torch.long), eps).clamp(-1, 1)
    z0 = data_consistency(to_complex(x0, channels), y, mask2d)
    return from_complex(z0, channels).clamp(-1, 1)


# ---------------------------------------------------------------------------
# quick PSNR for progress logging (full evaluation lives in evaluate.py)
# ---------------------------------------------------------------------------
def quick_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = (a.clamp(-1, 1) + 1) / 2
    b = (b.clamp(-1, 1) + 1) / 2
    mse = torch.mean((a - b) ** 2).item()
    if mse < 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def save_png(tensor_chw: torch.Tensor, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    #img = (tensor_chw[0].clamp(-1, 1).cpu().numpy() + 1) / 2  # first channel
    #plt.imsave(path, img, cmap="gray")
    img = (tensor_chw[0].clamp(-1, 1).cpu().numpy() + 1) / 2  # (C,H,W) in [0,1]
    img = np.squeeze(img)          # drop singleton dims
    if img.ndim == 3:              # (C,H,W) -> use first channel (magnitude)
        img = img[0]
    plt.imsave(path, img, cmap="gray")


def parse_args():
    p = argparse.ArgumentParser(description="DDNM reconstruction with a diffusion prior")
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--data-root", required=True, type=str)
    p.add_argument("--mode", default="volumes", choices=["volumes", "slices"])
    p.add_argument("--pattern", default="*", type=str)
    p.add_argument("--num-central-slices", default=20, type=int)
    p.add_argument("--slice-axis", default=2, type=int)
    p.add_argument("--ratios", default="0.2,0.3,0.5", type=str)
    p.add_argument("--steps", default=100, type=int, help="reverse-diffusion steps")
    p.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--eta", default=0.0, type=float)
    p.add_argument("--std-scale", default=6.0, type=float, help="mask Gaussian width")
    p.add_argument("--center-lines", default=0, type=int)
    p.add_argument("--num-slices", default=50, type=int, help="cap test slices (0=all)")
    p.add_argument("--batch-size", default=4, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--out", default="results/ddnm", type=str)
    p.add_argument("--save-images", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    diffusion, channels, size = load_diffusion(args.ckpt, device)
    print(f"[model] loaded prior: channels={channels} size={size} device={device}")

    ds = MRISliceDataset(
        root=args.data_root, mode=args.mode, pattern=args.pattern,
        num_central_slices=args.num_central_slices, slice_axis=args.slice_axis,
        image_size=size, channels=channels)
    n = len(ds) if args.num_slices in (0, None) else min(args.num_slices, len(ds))
    ratios = [float(r) for r in args.ratios.split(",")]
    print(f"[data] evaluating {n} slices at ratios {ratios}")

    results = {r: [] for r in ratios}  # sample-wise quick PSNR per ratio
    for r in ratios:
        for start in range(0, n, args.batch_size):
            idxs = list(range(start, min(start + args.batch_size, n)))
            x_true = torch.stack([ds[i] for i in idxs]).to(device)  # (B,C,H,W)

            # per-slice reproducible masks (B,H,W) so every method sees the
            # same undersampling; broadcasts through the operator.
            mask2d = torch.stack([
                seeded_row_mask(size, r, args.seed, i,
                                std_scale=args.std_scale,
                                center_lines=args.center_lines, device=device)
                for i in idxs])

            z_true = to_complex(x_true, channels)
            y, x_zf = subsample_kspace(z_true, mask2d)

            x_rec = reconstruct_ddnm(
                diffusion, y, mask2d, channels,
                steps=args.steps, sampler=args.sampler, eta=args.eta,
                device=device)

            for b, i in enumerate(idxs):
                ps = quick_psnr(x_rec[b:b + 1], x_true[b:b + 1])
                results[r].append(ps)
                if args.save_images:
                    tag = f"r{int(r*100)}_slice{i}"
                    save_png(x_zf[b:b + 1].real if channels == 1
                             else from_complex(x_zf[b:b + 1], channels),
                             os.path.join(args.out, f"{tag}_input.png"))
                    save_png(x_rec[b:b + 1], os.path.join(args.out, f"{tag}_ddnm.png"))
                    save_png(x_true[b:b + 1], os.path.join(args.out, f"{tag}_gt.png"))
            # also save arrays for evaluate.py to compute PSNR/SSIM properly
            np.save(os.path.join(args.out, f"rec_r{int(r*100)}_b{start}.npy"),
                    x_rec.cpu().numpy())
            np.save(os.path.join(args.out, f"gt_r{int(r*100)}_b{start}.npy"),
                    x_true.cpu().numpy())

        arr = np.array(results[r])
        print(f"[ratio {r:.0%}] quick PSNR  mean {arr.mean():.2f}  std {arr.std():.2f}"
              f"  (n={len(arr)})")

    print(f"[done] arrays + images written to {args.out}")
    

if __name__ == "__main__":
    main()
