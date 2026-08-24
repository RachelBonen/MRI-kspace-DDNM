# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# prepare_data.py
# ----------------
# Builds the data/train and data/test layout that the loader
# (src/diffusion/dataset.py) expects, from the selected_npy directory and the
# brain_age metadata CSVs.
#
# Default: creates symlinks (no copy - saves disk) to the .npy files according
# to the split in the CSVs:
#   train CSV  -> data/train/
#   val CSV    -> data/train/   (change with --val-to)
#   test CSV   -> data/test/
#
# Run from the repo root (Final_Project_kspace):
#   python prepare_data.py --dataset-root /path/to/MRI_2026_datasets/brain_age
#   python prepare_data.py --dataset-root /path/to/brain_age --copy   # copy instead of symlink
#
# The script auto-detects the filename column in the CSV. If detection fails,
# pass it manually:
#   python prepare_data.py --dataset-root ... --id-col filename

import argparse
import glob
import os
import shutil

import pandas as pd


def find_csv(root, split):
    m = glob.glob(os.path.join(root, f"*{split}*metadata*.csv"))
    return m[0] if m else None


def detect_id_column(df, npy_names):
    """Guess which column holds the npy file name/id by matching existing files."""
    npy_stems = {os.path.splitext(n)[0] for n in npy_names}
    best_col, best_hits = None, 0
    for col in df.columns:
        vals = df[col].astype(str)
        hits = sum(
            1 for v in vals
            if v in npy_names
            or os.path.splitext(v)[0] in npy_stems
            or v in npy_stems
        )
        if hits > best_hits:
            best_col, best_hits = col, hits
    return best_col, best_hits


def resolve_npy(value, npy_dir, npy_names):
    """Return the full path to an npy file from a CSV value (with/without extension)."""
    for cand in (value, value + ".npy",
                 os.path.splitext(value)[0] + ".npy"):
        if cand in npy_names:
            return os.path.join(npy_dir, cand)
    return None


def link_split(df, id_col, npy_dir, npy_names, dst, copy=False):
    os.makedirs(dst, exist_ok=True)
    linked, missing = 0, 0
    for value in df[id_col].astype(str):
        src = resolve_npy(value, npy_dir, npy_names)
        if src is None:
            missing += 1
            continue
        target = os.path.join(dst, os.path.basename(src))
        if os.path.exists(target) or os.path.islink(target):
            continue
        if copy:
            shutil.copy2(src, target)
        else:
            os.symlink(os.path.abspath(src), target)
        linked += 1
    return linked, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True,
                    help="brain_age directory (contains selected_npy and the CSVs)")
    ap.add_argument("--npy-dir", default=None,
                    help="default: <dataset-root>/selected_npy")
    ap.add_argument("--out", default="data",
                    help="output directory (default: data next to the repo)")
    ap.add_argument("--id-col", default=None,
                    help="name of the id column in the CSV (default: auto-detect)")
    ap.add_argument("--val-to", default="train", choices=["train", "test", "skip"],
                    help="where to place the val set (default: train)")
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of creating symlinks")
    args = ap.parse_args()

    npy_dir = args.npy_dir or os.path.join(args.dataset_root, "selected_npy")
    npy_names = {os.path.basename(p)
                 for p in glob.glob(os.path.join(npy_dir, "*.npy"))}
    if not npy_names:
        raise SystemExit(f"[!] no .npy files found in {npy_dir}")
    print(f"found {len(npy_names)} npy files in {npy_dir}")

    splits = {s: find_csv(args.dataset_root, s) for s in ["train", "val", "test"]}
    dest_map = {"train": "train", "test": "test", "val": args.val_to}

    # detect the id column from the first CSV that exists
    id_col = args.id_col
    if id_col is None:
        for s, path in splits.items():
            if path:
                df = pd.read_csv(path)
                id_col, hits = detect_id_column(df, npy_names)
                print(f"detected id column: '{id_col}' "
                      f"({hits}/{len(df)} matches in {s})")
                if hits == 0:
                    print(f"  available columns: {list(df.columns)}")
                    raise SystemExit("[!] auto-detection failed - pass --id-col manually")
                break

    total = 0
    for split, path in splits.items():
        if not path:
            print(f"[i] no CSV for {split}, skipping")
            continue
        dst_split = dest_map[split]
        if dst_split == "skip":
            print(f"[i] skipping {split} (--val-to skip)")
            continue
        df = pd.read_csv(path)
        dst = os.path.join(args.out, dst_split)
        linked, missing = link_split(df, id_col, npy_dir, npy_names, dst, args.copy)
        total += linked
        print(f"{split:5s} -> {dst}: {linked} files"
              + (f", {missing} not found" if missing else ""))

    print(f"\n[OK] total {total} files ready under '{args.out}/'.")
    print("Next step: smoke test -")
    print("  python -m src.diffusion.train --data-root data/train --image-size 64 \\")
    print("      --base-channels 32 --channel-mults 1,2,4 --batch-size 4 \\")
    print("      --steps 200 --log-every 20 --ckpt-every 200 --sample-every 200 --out runs/smoke")


if __name__ == "__main__":
    main()
