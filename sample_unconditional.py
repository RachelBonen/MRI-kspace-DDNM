# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# sample_unconditional.py
# ------------------------
# Unconditional sampling from the trained diffusion prior - NO k-space, NO mask,
# NO measurements. Starts from pure Gaussian noise and lets the reverse process
# hallucinate a brain. This is the figure for the "DDPM Prior Inference" slide:
# it demonstrates that the prior really learned anatomy, before any data
# consistency is attached.
#
# Produces (in --out):
#   trajectory.png   2 rows x N columns:
#                      row 1 = x_t         (the noisy state, noise -> brain)
#                      row 2 = x0_hat      (the model's running guess of the
#                                           clean image at that same step)
#                    This is literally the loop drawn on the slide.
#   grid.png         a row of --num final samples (different random seeds)
#   frames/*.png     every panel also saved individually, in case you want to
#                    place them one-by-one in PowerPoint.
#
# Run from the repo root, on the machine that holds the checkpoint:
#
#   python sample_unconditional.py --ckpt runs/mag_128/ckpt_100000.pt \
#       --num 4 --steps 100 --snapshots 6 --out results/prior_samples
#
# CPU works too (128x128 is small), just slower - add --device cpu.

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.diffusion.sample import load_diffusion


# ---------------------------------------------------------------------------
# reverse diffusion from pure noise, recording snapshots along the way
# ---------------------------------------------------------------------------
@torch.no_grad()
def ddim_trajectory(diffusion, n_images, channels, size, steps, n_snapshots,
                    eta, device):
    """Run unconditional DDIM and return snapshots of (t, x_t, x0_hat)."""
    acp = diffusion.alphas_cumprod
    times = torch.linspace(diffusion.timesteps - 1, 0, steps + 1).long().to(device)

    # which loop iterations to photograph (evenly spread, always incl. the first)
    snap_at = set(np.linspace(0, steps - 1, n_snapshots - 1).astype(int).tolist())

    x = torch.randn(n_images, channels, size, size, device=device)
    snaps = []

    for i in range(steps):
        t_val = int(times[i])
        t = torch.full((n_images,), t_val, device=device, dtype=torch.long)
        t_next = int(times[i + 1])

        eps = diffusion.model(x, t)
        x0 = diffusion.predict_x0_from_noise(x, t, eps).clamp(-1, 1)

        if i in snap_at:
            snaps.append((t_val, x.detach().cpu().clone(), x0.detach().cpu().clone()))

        ac_next = acp[t_next]
        sigma = 0.0
        if eta > 0 and t_next > 0:
            ac_t = acp[t]
            sigma = eta * torch.sqrt(
                (1 - ac_next) / (1 - ac_t) * (1 - ac_t / ac_next).clamp(min=0)
            ).view(n_images, 1, 1, 1)
        dir_xt = torch.sqrt((1 - ac_next - (sigma ** 2 if eta > 0 else 0)).clamp(min=0)) * eps
        x = torch.sqrt(ac_next) * x0 + dir_xt
        if eta > 0 and t_next > 0:
            x = x + sigma * torch.randn_like(x)

    # final estimate at t = 0
    t0 = torch.zeros(n_images, device=device, dtype=torch.long)
    eps = diffusion.model(x, t0)
    x0 = diffusion.predict_x0_from_noise(x, t0, eps).clamp(-1, 1)
    snaps.append((0, x.detach().cpu().clone(), x0.detach().cpu().clone()))

    return snaps, x0.detach().cpu()


# ---------------------------------------------------------------------------
def to01(t):
    """(C,H,W) in [-1,1] -> (H,W) in [0,1], first channel."""
    a = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    return np.clip((a[0] + 1) / 2, 0, 1)


def save_single(img_chw, path):
    plt.imsave(path, to01(img_chw), cmap="gray", vmin=0, vmax=1)


def make_trajectory_figure(snaps, which, out_path, frames_dir):
    """2-row figure: top = x_t, bottom = x0_hat, columns = timesteps."""
    n = len(snaps)
    fig, ax = plt.subplots(2, n, figsize=(2.1 * n, 4.7))
    if n == 1:
        ax = ax.reshape(2, 1)

    for c, (t_val, xt, x0) in enumerate(snaps):
        ax[0, c].imshow(to01(xt[which]), cmap="gray", vmin=0, vmax=1)
        ax[0, c].set_title(f"t = {t_val}", fontsize=11)
        ax[1, c].imshow(to01(x0[which]), cmap="gray", vmin=0, vmax=1)
        for r in (0, 1):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for s in ax[r, c].spines.values():
                s.set_edgecolor("#bbbbbb")

        save_single(xt[which], os.path.join(frames_dir, f"xt_t{t_val:04d}.png"))
        save_single(x0[which], os.path.join(frames_dir, f"x0hat_t{t_val:04d}.png"))

    ax[0, 0].set_ylabel("$x_t$\n(state)", fontsize=12, rotation=0,
                        ha="right", va="center", labelpad=28)
    ax[1, 0].set_ylabel(r"$\hat{x}_0$" "\n(guess)", fontsize=12, rotation=0,
                        ha="right", va="center", labelpad=28)

    fig.suptitle("Unconditional reverse diffusion — pure noise to a hallucinated brain\n"
                 "(no k-space, no mask, no measurements)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")


def make_grid_figure(finals, out_path, frames_dir):
    n = finals.shape[0]
    fig, ax = plt.subplots(1, n, figsize=(2.3 * n, 2.6))
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(to01(finals[i]), cmap="gray", vmin=0, vmax=1)
        ax[i].axis("off")
        save_single(finals[i], os.path.join(frames_dir, f"final_{i}.png"))
    fig.suptitle("Samples from the trained prior (each from a different noise seed)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Unconditional sampling from the trained diffusion prior")
    ap.add_argument("--ckpt", required=True, type=str,
                    help="path to the trained checkpoint (uses EMA weights)")
    ap.add_argument("--num", default=4, type=int, help="how many samples to draw")
    ap.add_argument("--steps", default=100, type=int, help="DDIM reverse steps")
    ap.add_argument("--snapshots", default=6, type=int,
                    help="how many columns in the trajectory figure")
    ap.add_argument("--which", default=0, type=int,
                    help="which of the --num samples to show in the trajectory")
    ap.add_argument("--eta", default=0.0, type=float,
                    help="0 = deterministic DDIM (what the report uses)")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="results/prior_samples", type=str)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    diffusion, channels, size = load_diffusion(args.ckpt, device)
    print(f"[model] channels={channels} size={size} device={device}")
    print(f"[run] {args.num} samples, {args.steps} DDIM steps, eta={args.eta}")

    snaps, finals = ddim_trajectory(
        diffusion, args.num, channels, size,
        steps=args.steps, n_snapshots=args.snapshots, eta=args.eta, device=device)

    make_trajectory_figure(snaps, args.which,
                           os.path.join(args.out, "trajectory.png"), frames_dir)
    make_grid_figure(finals, os.path.join(args.out, "grid.png"), frames_dir)
    print(f"[done] individual panels in {frames_dir}")


if __name__ == "__main__":
    main()
