# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# check_axes.py
# -------------
# Verifies that every train/test volume shares the same voxel grid, so that
# slicing along a fixed axis (default 2) means the SAME anatomical plane for
# ALL subjects. If volumes are co-registered they all have identical shape;
# any volume with a different shape - or a different length along the slice
# axis - is flagged as "not synchronized".
#
# It also (optionally) saves a montage of the central slice along the chosen
# axis from several random subjects, so you can eyeball that the plane is
# consistent (all axial, in your case).
#
# CPU-only, read-only: safe to run alongside training.
#
# Run from the repo root (Final_Project_kspace):
#   python check_axes.py --roots data/train data/test --slice-axis 2
#   python check_axes.py --roots data/train data/test --montage axes_montage.png --n-montage 12

import argparse
import glob
import os
from collections import Counter

import numpy as np


def load_shape(path):
    """Return the array shape, or None with a reason if it can't be read."""
    try:
        arr = np.load(path, allow_pickle=False, mmap_mode="r")
        return tuple(arr.shape), None
    except Exception as e:
        # retry allowing pickle (object arrays), just to read the shape
        try:
            arr = np.load(path, allow_pickle=True)
            return tuple(np.asarray(arr).shape), f"pickled ({type(e).__name__})"
        except Exception as e2:
            return None, f"unreadable: {e2}"


def scan(roots):
    files = []
    for r in roots:
        files += glob.glob(os.path.join(r, "**", "*.npy"), recursive=True)
    files = sorted(set(files))
    records = []  # (path, shape_or_None, note)
    for f in files:
        shp, note = load_shape(f)
        records.append((f, shp, note))
    return records


def montage(records, slice_axis, out_path, n):
    import random
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [(f, s) for f, s, note in records if s is not None and len(s) == 3]
    random.seed(0)
    random.shuffle(ok)
    ok = ok[:n]
    if not ok:
        print("[montage] no 3D volumes to show")
        return
    cols = int(np.ceil(np.sqrt(len(ok))))
    rows = int(np.ceil(len(ok) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (f, s) in zip(np.atleast_1d(axes).ravel(), ok):
        vol = np.abs(np.load(f, allow_pickle=True))
        mid = vol.shape[slice_axis] // 2
        sl = np.take(vol, mid, axis=slice_axis)
        ax.imshow(np.squeeze(sl), cmap="gray")
        ax.set_title(os.path.basename(f)[:14], fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    print(f"[montage] saved central axis-{slice_axis} slices -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["data/train", "data/test"])
    ap.add_argument("--slice-axis", default=2, type=int)
    ap.add_argument("--montage", default="", help="path to save a slice montage (optional)")
    ap.add_argument("--n-montage", default=12, type=int)
    ap.add_argument("--list-outliers", action="store_true",
                    help="print every outlier filename (not just a sample)")
    args = ap.parse_args()

    records = scan(args.roots)
    total = len(records)
    unreadable = [(f, n) for f, s, n in records if s is None]
    readable = [(f, s, n) for f, s, n in records if s is not None]

    shape_counts = Counter(s for _, s, _ in readable)
    axis_counts = Counter(s[args.slice_axis] if len(s) > args.slice_axis else None
                          for _, s, _ in readable)

    print(f"scanned {total} files across {args.roots}")
    print(f"  readable: {len(readable)}   unreadable/odd: {len(unreadable)}")

    print("\nShape distribution (all volumes should share ONE shape):")
    for shp, c in shape_counts.most_common():
        print(f"  {str(shp):>20} : {c}")

    print(f"\nLength along slice axis {args.slice_axis} "
          f"(this is the plane you slice):")
    for L, c in axis_counts.most_common():
        print(f"  axis{args.slice_axis} = {L} : {c}")

    # The "synchronized" shape is the dominant one.
    if shape_counts:
        modal_shape = shape_counts.most_common(1)[0][0]
        modal_axis = modal_shape[args.slice_axis]
        outliers = [f for f, s, _ in readable if s != modal_shape]
        axis_outliers = [f for f, s, _ in readable
                         if len(s) <= args.slice_axis or s[args.slice_axis] != modal_axis]

        print(f"\nExpected (dominant) shape: {modal_shape}")
        print(f"  volumes matching it exactly      : {len(readable) - len(outliers)}")
        print(f"  volumes with a DIFFERENT shape   : {len(outliers)}")
        print(f"  volumes where axis {args.slice_axis} length differs: "
              f"{len(axis_outliers)}  <-- these break plane consistency")

        if axis_outliers:
            print("\n[!] NOT synchronized - axis-length outliers "
                  f"({'all' if args.list_outliers else 'first 15'}):")
            for f in (axis_outliers if args.list_outliers else axis_outliers[:15]):
                print(f"    {f}")
        else:
            print("\n[OK] All readable volumes share the same length along "
                  f"axis {args.slice_axis}: slicing that axis means the SAME "
                  "plane for every subject.")

    if unreadable:
        print(f"\nUnreadable/odd files ({len(unreadable)}) - skipped by training too:")
        for f, n in (unreadable if args.list_outliers else unreadable[:15]):
            print(f"    {os.path.basename(f)}: {n}")

    if args.montage:
        montage(records, args.slice_axis, args.montage, args.n_montage)


if __name__ == "__main__":
    main()
