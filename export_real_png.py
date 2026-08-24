# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# export_real_png.py
# ------------------
# Exports REAL training slices as a PNG grid, using the *same* preprocessing the
# diffusion model sees (central axial slices, center-crop/pad to image-size,
# per-slice normalization to [-1,1] -> displayed as [0,1]). Use it to compare
# "what goes in" (this grid) against "what is synthesized" (runs/.../samples_*.png).
#
# CPU-only and read-only: safe to run in parallel with training.
#
# Run from the repo root (Final_Project_kspace):
#   python export_real_png.py --data-root data/train --n 16 --out real_slices.png

import argparse
import glob
import os
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_volume(path):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        z = np.load(path)
        return z[z.files[0]]
    raise ValueError(f"unsupported: {path}")


def central_slice(vol, slice_axis):
    """Take the single central slice along slice_axis (matches model default)."""
    mid = vol.shape[slice_axis] // 2
    return np.take(vol, mid, axis=slice_axis)


def resize(img, size):
    """Center crop or zero-pad a 2D slice to (size, size) - same as dataset.py."""
    h, w = img.shape[-2:]
    if h > size:
        top = (h - size) // 2
        img = img[top:top + size, :]
    if w > size:
        left = (w - size) // 2
        img = img[:, left:left + size]
    h, w = img.shape[-2:]
    if h < size or w < size:
        img = np.pad(img, [((size - h) // 2, size - h - (size - h) // 2),
                           ((size - w) // 2, size - w - (size - w) // 2)],
                     mode="constant")
    return img


def to_unit(img):
    """Scale to [-1,1] per slice (as in training), then map to [0,1] for display."""
    img = np.abs(img) if np.iscomplexobj(img) else img
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    img = (img - lo) / (hi - lo)          # [0,1]
    img = img * 2.0 - 1.0                  # [-1,1] (what the model sees)
    return (img + 1.0) / 2.0               # back to [0,1] for saving


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/train")
    ap.add_argument("--image-size", default=128, type=int)
    ap.add_argument("--slice-axis", default=2, type=int)
    ap.add_argument("--n", default=16, type=int, help="number of slices in the grid")
    ap.add_argument("--out", default="real_slices.png")
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_root, "**", "*.npy"),
                             recursive=True))
    if not files:
        raise SystemExit(f"[!] no .npy files under {args.data_root}")
    random.seed(args.seed)
    random.shuffle(files)

    imgs = []
    for f in files:
        if len(imgs) >= args.n:
            break
        try:
            vol = load_volume(f)
            sl = central_slice(vol, args.slice_axis)
            sl = resize(sl, args.image_size)
            imgs.append(to_unit(sl))
        except Exception as e:
            print(f"[skip] {os.path.basename(f)}: {e}")

    if not imgs:
        raise SystemExit("[!] could not load any slices")

    cols = int(np.ceil(np.sqrt(len(imgs))))
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, im in zip(np.atleast_1d(axes).ravel(), imgs):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"[OK] saved {len(imgs)} real slices -> {args.out}")
    print(f"     compare against: runs/mag_128/samples_*.png (synthesized)")


if __name__ == "__main__":
    main()
