# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# make_eda_figure.py
# ------------------
# Data-description (EDA) figure for the report: takes one central slice and shows
# the 1-D variable-density mask and the resulting zero-filled aliased input at
# each acceleration ratio (20/30/50%), next to the fully-sampled ground truth.
# Uses the project's own kspace operators so it matches the pipeline exactly.
#
# CPU-only, read-only. Run from the repo root (Final_Project_kspace):
#   python make_eda_figure.py --data-root data/test --out figures/eda_undersampling.png

import argparse
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.baseline_cs import zero_fill
from src.kspace import seeded_row_mask, subsample_kspace, to_complex, from_complex


def load_central(path, size, slice_axis=2):
    vol = np.load(path, allow_pickle=True)
    mid = vol.shape[slice_axis] // 2
    sl = np.take(vol, mid, axis=slice_axis).astype(np.float32)
    # per-slice [-1,1] then crop/pad to size (matches dataset.py order)
    lo, hi = float(sl.min()), float(sl.max())
    sl = (sl - lo) / (hi - lo + 1e-8) * 2 - 1
    h, w = sl.shape
    if h > size: sl = sl[(h - size) // 2:(h - size) // 2 + size, :]
    if w > size: sl = sl[:, (w - size) // 2:(w - size) // 2 + size]
    h, w = sl.shape
    sl = np.pad(sl, [((size - h) // 2, size - h - (size - h) // 2),
                     ((size - w) // 2, size - w - (size - w) // 2)], mode="constant")
    return sl


def disp(x):
    return np.clip((np.asarray(x) + 1) / 2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/test")
    ap.add_argument("--image-size", default=128, type=int)
    ap.add_argument("--ratios", default="0.2,0.3,0.5")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--out", default="figures/eda_undersampling.png")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_root, "**", "*.npy"), recursive=True))
    if not files:
        raise SystemExit(f"[!] no .npy under {args.data_root}")
    ratios = [float(r) for r in args.ratios.split(",")]
    size = args.image_size

    gt = load_central(files[0], size)
    x_true = torch.from_numpy(gt)[None, None]        # (1,1,H,W)
    z = to_complex(x_true, 1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ncol = 1 + len(ratios)
    fig, axes = plt.subplots(2, ncol, figsize=(3 * ncol, 6))

    axes[0, 0].imshow(disp(gt), cmap="gray"); axes[0, 0].set_title("fully sampled")
    axes[1, 0].axis("off")
    for ax in axes[:, 0]:
        ax.set_xticks([]); ax.set_yticks([])

    for j, r in enumerate(ratios, start=1):
        m = seeded_row_mask(size, r, args.seed, 0).unsqueeze(0)     # (1,H,W)
        y, _ = subsample_kspace(z, m)
        zf = from_complex(zero_fill(y[0]), 1)[0].cpu().numpy()      # (H,W)
        axes[0, j].imshow(m[0].cpu().numpy(), cmap="gray", aspect="auto")
        axes[0, j].set_title(f"mask, keep {int(r*100)}%")
        axes[1, j].imshow(disp(zf), cmap="gray")
        axes[1, j].set_title(f"zero-filled ({int(r*100)}%)")
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("1-D variable-density undersampling: masks and zero-filled aliased inputs",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[OK] saved {args.out}")


if __name__ == "__main__":
    main()
