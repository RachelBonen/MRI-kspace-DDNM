# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# export_slices_from_npy.py
# -------------------------
# Extracts 2D PNG slices from the 3D .npy volumes, using the SAME preprocessing
# the diffusion model sees (central band along the slice axis, center-crop/pad
# to image-size, per-slice normalization to [-1,1] -> shown as [0,1]).
# Use it to inspect exactly what the model is trained on.
#
# How many PNGs per volume?
#   --mode central  (default): the central --num-central-slices (e.g. 40) slices
#                              => this is WHAT THE MODEL SEES.
#   --mode all               : every slice along the axis (~90 for a 99x120x90
#                              volume).
#
# CPU-only, read-only on the source: safe to run anytime.
#
# Run from the repo root (Final_Project_kspace):
#   # 40 central slices from 3 volumes (what the model trains on):
#   python export_slices_from_npy.py --data-root data/train --max-volumes 3 \
#       --num-central-slices 40 --out data/png_preview
#
#   # every slice of a single file:
#   python export_slices_from_npy.py --npy data/train/SUBJECT.npy --mode all \
#       --out data/png_preview

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_volume(path):
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return np.load(path, allow_pickle=True)


def central_indices(n, k):
    """Indices of the central k slices out of n (matches dataset.py)."""
    if k >= n:
        return list(range(n))
    start = (n - k) // 2
    return list(range(start, start + k))


def resize(img, size):
    """Center crop or zero-pad a 2D slice to (size,size) - matches dataset.py."""
    if size <= 0:
        return img
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
    """Per-slice [-1,1] normalization (as in training), mapped to [0,1] to save."""
    img = np.abs(img) if np.iscomplexobj(img) else img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    img = (img - lo) / (hi - lo)      # [0,1]
    img = img * 2.0 - 1.0             # [-1,1] = what the model sees
    return (img + 1.0) / 2.0          # back to [0,1] for saving


def process_file(path, args, out_root):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        vol = load_volume(path)
    except Exception as e:
        print(f"[skip] {stem}: {e}")
        return 0
    if vol.ndim != 3:
        print(f"[skip] {stem}: not 3D (shape {vol.shape})")
        return 0

    n = vol.shape[args.slice_axis]
    if args.mode == "all":
        idxs = list(range(n))
    else:
        idxs = central_indices(n, args.num_central_slices)

    sub = os.path.join(out_root, stem)
    os.makedirs(sub, exist_ok=True)
    for s in idxs:
        sl = np.take(vol, s, axis=args.slice_axis)
        sl = resize(sl, args.image_size)
        sl = to_unit(sl)
        plt.imsave(os.path.join(sub, f"{stem}_slice{s:03d}.png"),
                   sl, cmap="gray")
    return len(idxs)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-root", help="directory of .npy volumes")
    src.add_argument("--npy", help="a single .npy file")
    ap.add_argument("--mode", default="central", choices=["central", "all"])
    ap.add_argument("--num-central-slices", default=40, type=int,
                    help="how many central slices (mode=central); model uses 40")
    ap.add_argument("--slice-axis", default=2, type=int)
    ap.add_argument("--image-size", default=128, type=int,
                    help="crop/pad to this size (0 = keep native)")
    ap.add_argument("--max-volumes", default=5, type=int,
                    help="cap number of volumes (0 = all)")
    ap.add_argument("--out", default="data/png_preview")
    args = ap.parse_args()

    if args.npy:
        files = [args.npy]
    else:
        files = sorted(glob.glob(os.path.join(args.data_root, "**", "*.npy"),
                                 recursive=True))
        if args.max_volumes:
            files = files[:args.max_volumes]
    if not files:
        raise SystemExit("[!] no .npy files found")

    os.makedirs(args.out, exist_ok=True)
    total_pngs, total_vols = 0, 0
    for f in files:
        k = process_file(f, args, args.out)
        if k:
            total_vols += 1
            total_pngs += k
            print(f"[ok] {os.path.basename(f)} -> {k} PNGs")

    print(f"\n[done] {total_vols} volumes -> {total_pngs} PNGs in '{args.out}/'")
    if total_vols:
        print(f"       ~{total_pngs // total_vols} PNGs per volume "
              f"(mode={args.mode})")


if __name__ == "__main__":
    main()
