# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# train.py
# Trains the unconditional diffusion prior on fully-sampled MRI slices.
# The trained model (EMA weights) is later loaded by sample.py, where
# k-space data-consistency turns it into a reconstruction method for the
# 20% / 30% / 50% subsampled experiments.
#
# Example:
#   python -m src.diffusion.train \
#       --data-root data/ixi_t1 --mode volumes --channels 1 \
#       --image-size 256 --batch-size 8 --steps 200000 --out runs/ixi_t1

import argparse
import copy
import os
import time

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import MRISliceDataset
from .diffusion import GaussianDiffusion
from .model import UNet


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential moving average of model parameters for stabler sampling."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def parse_args():
    p = argparse.ArgumentParser(description="Train MRI diffusion prior")
    # data
    p.add_argument("--data-root", required=True, type=str)
    p.add_argument("--mode", default="volumes", choices=["volumes", "slices"])
    p.add_argument("--pattern", default="*", type=str)
    p.add_argument("--num-central-slices", default=40, type=int)
    p.add_argument("--slice-axis", default=2, type=int)
    p.add_argument("--channels", default=1, type=int, choices=[1, 2])
    p.add_argument("--image-size", default=256, type=int)
    p.add_argument("--val-frac", default=0.05, type=float)
    # model
    p.add_argument("--base-channels", default=64, type=int)
    p.add_argument("--channel-mults", default="1,2,4,8", type=str)
    p.add_argument("--num-res-blocks", default=2, type=int)
    p.add_argument("--attn-res", default="32,16", type=str,
                   help="image sizes (px) where self-attention is applied; "
                        "for 256px/4-level configs the bottleneck is 32px")
    # diffusion
    p.add_argument("--timesteps", default=1000, type=int)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    p.add_argument("--loss", default="l2", choices=["l1", "l2", "huber"])
    # optim
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--lr", default=2e-4, type=float)
    p.add_argument("--steps", default=200000, type=int)
    p.add_argument("--ema-decay", default=0.9999, type=float)
    p.add_argument("--grad-clip", default=1.0, type=float)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--amp", action="store_true", help="mixed-precision training")
    # io / logging
    p.add_argument("--out", default="runs/diffusion", type=str)
    p.add_argument("--log-every", default=100, type=int)
    p.add_argument("--ckpt-every", default=5000, type=int)
    p.add_argument("--sample-every", default=5000, type=int)
    p.add_argument("--resume", default="", type=str)
    p.add_argument("--seed", default=0, type=int)
    return p.parse_args()


def save_samples(diffusion, ema_model, path, n=4):
    """Save a small grid of EMA samples to eyeball training progress."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    orig = diffusion.model
    diffusion.model = ema_model
    imgs = diffusion.ddim_sample(batch_size=n, steps=50).cpu()
    diffusion.model = orig
    imgs = (imgs.clamp(-1, 1) + 1) / 2  # -> [0, 1]
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for k in range(n):
        ax = axes[k] if n > 1 else axes
        ax.imshow(imgs[k, 0], cmap="gray")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ----
    full = MRISliceDataset(
        root=args.data_root, mode=args.mode, pattern=args.pattern,
        num_central_slices=args.num_central_slices, slice_axis=args.slice_axis,
        image_size=args.image_size, channels=args.channels,
    )
    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)
    print(f"[data] {len(full)} slices -> {n_train} train / {n_val} val")

    # ---- model + diffusion ----
    mults = tuple(int(m) for m in args.channel_mults.split(","))
    attn = tuple(int(a) for a in args.attn_res.split(",") if a != "")
    model = UNet(
        in_channels=args.channels, base_channels=args.base_channels,
        channel_mults=mults, num_res_blocks=args.num_res_blocks,
        attn_resolutions=attn, image_size=args.image_size,
    ).to(device)
    diffusion = GaussianDiffusion(
        model, image_size=args.image_size, channels=args.channels,
        timesteps=args.timesteps, schedule=args.schedule,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] UNet params: {n_params:.1f}M  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=args.ema_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt.get("step", 0)
        print(f"[resume] from {args.resume} at step {start_step}")

    # ---- training loop ----
    data = cycle(train_loader)
    model.train()
    t0 = time.time()
    running = 0.0
    for step in range(start_step, args.steps):
        x0 = next(data).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            loss = diffusion(x0, loss_type=args.loss)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        ema.update(model)

        running += loss.item()
        if (step + 1) % args.log_every == 0:
            avg = running / args.log_every
            running = 0.0
            ips = args.log_every * args.batch_size / (time.time() - t0)
            t0 = time.time()
            print(f"step {step+1:>7}/{args.steps}  loss {avg:.4f}  "
                  f"{ips:.1f} img/s")

        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps:
            path = os.path.join(args.out, f"ckpt_{step+1}.pt")
            torch.save({"model": model.state_dict(),
                        "ema": ema.shadow.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step + 1, "args": vars(args)}, path)
            torch.save({"model": model.state_dict(),
                        "ema": ema.shadow.state_dict(),
                        "step": step + 1, "args": vars(args)},
                       os.path.join(args.out, "latest.pt"))
            print(f"[ckpt] saved {path}")

        if (step + 1) % args.sample_every == 0:
            save_samples(diffusion, ema.shadow,
                         os.path.join(args.out, f"samples_{step+1}.png"))

    print("[done] training complete")


if __name__ == "__main__":
    main()
