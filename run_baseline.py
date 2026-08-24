# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# run_baseline.py
# ---------------
# Runs the CLASSICAL baseline only (zero-fill + CS-TV), independently of the
# diffusion model, so you can get baseline results before training finishes.
# Uses the SAME masks / FFT / normalization as the diffusion pipeline (via
# src.kspace and src.metrics) for a fair comparison later.
#
# Outputs (to --out):
#   * baseline_metrics.csv          PSNR/SSIM mean+/-std per ratio, per method
#   * baseline_psnr_vs_ratio.png    zero-fill vs CS-TV
#   * baseline_ssim_vs_ratio.png
#   * example_r{RATIO}_slice{i}_{input,cs,gt}.png   (a few qualitative panels)
#
# CPU-only, read-only on data: safe to run while training continues.
#
# Run from the repo root (Final_Project_kspace):
#   python run_baseline.py --data-root data/test --ratios 0.2,0.3,0.5 \
#       --num-slices 50 --lam 0.01 --out results/baseline

import argparse
import csv
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.baseline_cs import cs_tv_recon, zero_fill
from src.kspace import from_complex, seeded_row_mask, subsample_kspace, to_complex
from src.metrics import compute_metrics
from src.diffusion.dataset import MRISliceDataset

METHODS = ["zerofill", "cs"]
LABELS = {"zerofill": "Zero-fill", "cs": "CS-MRI (baseline)"}


def _disp(img_chw):
    """First channel mapped [-1,1] -> [0,1] for saving."""
    a = np.asarray(img_chw.detach().cpu()) if hasattr(img_chw, "detach") else np.asarray(img_chw)
    a = np.squeeze(a)
    if a.ndim == 3:
        a = a[0]
    return np.clip((a + 1.0) / 2.0, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description="Classical baseline (zero-fill + CS-TV)")
    ap.add_argument("--data-root", default="data/test")
    ap.add_argument("--image-size", default=128, type=int)
    ap.add_argument("--channels", default=1, type=int, choices=[1, 2])
    ap.add_argument("--mode", default="volumes", choices=["volumes", "slices"])
    ap.add_argument("--num-central-slices", default=20, type=int)
    ap.add_argument("--slice-axis", default=2, type=int)
    ap.add_argument("--ratios", default="0.2,0.3,0.5", type=str)
    ap.add_argument("--num-slices", default=50, type=int, help="cap test slices (0=all)")
    ap.add_argument("--lam", default=0.01, type=float, help="CS-TV regularization")
    ap.add_argument("--cs-iter", default=100, type=int)
    ap.add_argument("--tv-iter", default=10, type=int)
    ap.add_argument("--std-scale", default=6.0, type=float)
    ap.add_argument("--center-lines", default=0, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--save-examples", default=4, type=int,
                    help="how many example image panels to save (at first ratio)")
    ap.add_argument("--out", default="results/baseline", type=str)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")   # baseline is light; keep off the training GPU
    os.makedirs(args.out, exist_ok=True)

    ds = MRISliceDataset(
        root=args.data_root, mode=args.mode,
        num_central_slices=args.num_central_slices, slice_axis=args.slice_axis,
        image_size=args.image_size, channels=args.channels)
    n = len(ds) if args.num_slices in (0, None) else min(args.num_slices, len(ds))
    ratios = [float(r) for r in args.ratios.split(",")]
    size, ch = args.image_size, args.channels
    print(f"[baseline] {n} slices x {len(ratios)} ratios, channels={ch}, "
          f"image-size={size}, lambda={args.lam}")

    # scores[(ratio, method)] = {"psnr": [...], "ssim": [...]}
    scores = {(r, m): {"psnr": [], "ssim": []} for r in ratios for m in METHODS}

    for r in ratios:
        for i in range(n):
            x_true = ds[i].unsqueeze(0).to(device)                 # (1,C,H,W) in [-1,1]
            mask2d = seeded_row_mask(size, r, args.seed, i,
                                     std_scale=args.std_scale,
                                     center_lines=args.center_lines,
                                     device=device).unsqueeze(0)    # (1,H,W)
            z_true = to_complex(x_true, ch)
            y, _ = subsample_kspace(z_true, mask2d)                 # y: (1,H,W) complex

            zf = from_complex(zero_fill(y[0]), ch)                  # (C,H,W)
            cs = from_complex(
                cs_tv_recon(y[0], mask2d[0], lam=args.lam,
                            n_iter=args.cs_iter, tv_iter=args.tv_iter), ch)

            m_zf = compute_metrics(zf, x_true[0], ch)
            m_cs = compute_metrics(cs, x_true[0], ch)
            for key in ("psnr", "ssim"):
                scores[(r, "zerofill")][key].append(m_zf[key])
                scores[(r, "cs")][key].append(m_cs[key])

            # save a few example panels at EVERY ratio (for the report figure)
            if i < args.save_examples:
                for name, img in [("input", zf), ("cs", cs), ("gt", x_true[0])]:
                    plt.imsave(os.path.join(
                        args.out, f"example_r{int(r*100)}_slice{i}_{name}.png"),
                        _disp(img), cmap="gray")

        # progress line
        cs_ps = np.array(scores[(r, "cs")]["psnr"])
        cs_ss = np.array(scores[(r, "cs")]["ssim"])
        print(f"[ratio {r:.0%}] CS-TV  PSNR {cs_ps.mean():5.2f}+/-{cs_ps.std():.2f}"
              f"   SSIM {cs_ss.mean():.4f}+/-{cs_ss.std():.4f}")

    # ---- write CSV table ----
    csv_path = os.path.join(args.out, "baseline_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ratio", "method", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "n"])
        for r in ratios:
            for m in METHODS:
                ps = np.array(scores[(r, m)]["psnr"])
                ss = np.array(scores[(r, m)]["ssim"])
                w.writerow([f"{r:.2f}", LABELS[m], f"{ps.mean():.4f}", f"{ps.std():.4f}",
                            f"{ss.mean():.4f}", f"{ss.std():.4f}", len(ps)])
    print(f"\n[OK] metrics table -> {csv_path}")

    # ---- line plots ----
    for key, ylab, fname in [("psnr", "PSNR (dB)", "baseline_psnr_vs_ratio.png"),
                             ("ssim", "SSIM", "baseline_ssim_vs_ratio.png")]:
        plt.figure(figsize=(6, 4))
        xs = [r * 100 for r in ratios]
        for m in METHODS:
            means = [np.mean(scores[(r, m)][key]) for r in ratios]
            stds = [np.std(scores[(r, m)][key]) for r in ratios]
            plt.errorbar(xs, means, yerr=stds, marker="o", capsize=4,
                         linewidth=2, label=LABELS[m])
        plt.xlabel("k-space kept (%)")
        plt.ylabel(ylab)
        plt.title(f"Baseline {ylab} vs. sampling ratio (mean +/- std)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, fname), dpi=130)
        plt.close()
        print(f"[OK] plot -> {os.path.join(args.out, fname)}")

    print("\nDone. These are your BASELINE results; the DDNM column gets added "
          "later by src.evaluate once the prior is trained.")


if __name__ == "__main__":
    main()
