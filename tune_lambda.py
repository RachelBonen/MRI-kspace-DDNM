# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# tune_lambda.py
# --------------
# Per-ratio tuning of the CS-TV regularization weight lambda, so the classical
# baseline is not handicapped by a single fixed lambda. Sweeps a grid of lambda
# values at each acceleration ratio on a VALIDATION split (default data/train,
# to avoid tuning on the test set), and reports the best lambda per ratio.
#
# CPU-only, read-only on data. Run from the repo root (Final_Project_kspace):
#   python tune_lambda.py --data-root data/train --ratios 0.2,0.3,0.5 \
#       --lams 0.002,0.005,0.01,0.02,0.05,0.1 --num-slices 40 --out results/lambda_tune
#
# Then use the printed best-lambda-per-ratio in the final evaluation.

import argparse
import csv
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.baseline_cs import cs_tv_recon
from src.kspace import from_complex, seeded_row_mask, subsample_kspace, to_complex
from src.metrics import compute_metrics
from src.diffusion.dataset import MRISliceDataset


def main():
    ap = argparse.ArgumentParser(description="Per-ratio CS-TV lambda tuning")
    ap.add_argument("--data-root", default="data/train",
                    help="validation split to tune on (NOT the test set)")
    ap.add_argument("--image-size", default=128, type=int)
    ap.add_argument("--channels", default=1, type=int, choices=[1, 2])
    ap.add_argument("--num-central-slices", default=20, type=int)
    ap.add_argument("--slice-axis", default=2, type=int)
    ap.add_argument("--ratios", default="0.2,0.3,0.5")
    ap.add_argument("--lams", default="0.002,0.005,0.01,0.02,0.05,0.1",
                    help="comma-separated lambda grid")
    ap.add_argument("--num-slices", default=40, type=int, help="slices to average over")
    ap.add_argument("--cs-iter", default=100, type=int)
    ap.add_argument("--tv-iter", default=10, type=int)
    ap.add_argument("--std-scale", default=6.0, type=float)
    ap.add_argument("--select-by", default="ssim", choices=["ssim", "psnr"],
                    help="metric to pick the best lambda")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--out", default="results/lambda_tune")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    os.makedirs(args.out, exist_ok=True)

    ds = MRISliceDataset(root=args.data_root, mode="volumes",
                         num_central_slices=args.num_central_slices,
                         slice_axis=args.slice_axis, image_size=args.image_size,
                         channels=args.channels)
    n = min(args.num_slices, len(ds))
    ratios = [float(r) for r in args.ratios.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    size, ch = args.image_size, args.channels
    print(f"[tune] {n} slices, ratios={ratios}, lambdas={lams}, select-by={args.select_by}")

    # results[(ratio, lam)] = {"psnr": mean, "ssim": mean}
    rows = []
    best = {}
    for r in ratios:
        curve = {"psnr": [], "ssim": []}
        for lam in lams:
            ps, ss = [], []
            for i in range(n):
                x_true = ds[i].unsqueeze(0)
                m = seeded_row_mask(size, r, args.seed, i, std_scale=args.std_scale,
                                    device=device).unsqueeze(0)
                y, _ = subsample_kspace(to_complex(x_true, ch), m)
                cs = from_complex(cs_tv_recon(y[0], m[0], lam=lam,
                                              n_iter=args.cs_iter, tv_iter=args.tv_iter), ch)
                mm = compute_metrics(cs, x_true[0], ch)
                ps.append(mm["psnr"]); ss.append(mm["ssim"])
            pm, sm = float(np.mean(ps)), float(np.mean(ss))
            curve["psnr"].append(pm); curve["ssim"].append(sm)
            rows.append([f"{r:.2f}", f"{lam:g}", f"{pm:.4f}", f"{sm:.4f}"])
            print(f"  ratio {r:.0%}  lambda {lam:<6g}  PSNR {pm:5.2f}  SSIM {sm:.4f}")
        # pick best lambda for this ratio
        key = curve[args.select_by]
        bi = int(np.argmax(key))
        best[r] = (lams[bi], curve["psnr"][bi], curve["ssim"][bi])
        print(f"  -> best lambda at {r:.0%}: {lams[bi]:g} "
              f"(PSNR {curve['psnr'][bi]:.2f}, SSIM {curve['ssim'][bi]:.4f})\n")

        # per-ratio curve plot
        fig, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.plot(lams, curve["psnr"], "o-", color="tab:blue", label="PSNR")
        ax1.set_xscale("log"); ax1.set_xlabel("lambda"); ax1.set_ylabel("PSNR (dB)", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(lams, curve["ssim"], "s--", color="tab:red", label="SSIM")
        ax2.set_ylabel("SSIM", color="tab:red")
        ax1.axvline(lams[bi], color="gray", ls=":")
        ax1.set_title(f"CS-TV lambda sweep at {r:.0%} sampling")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"lambda_sweep_r{int(r*100)}.png"), dpi=130)
        plt.close(fig)

    # write CSV
    with open(os.path.join(args.out, "lambda_tuning.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ratio", "lambda", "psnr_mean", "ssim_mean"])
        w.writerows(rows)

    print("=" * 50)
    print("Best lambda per ratio (use these in the final evaluation):")
    for r in ratios:
        lam, pm, sm = best[r]
        print(f"  {r:.0%}: lambda={lam:g}  (PSNR {pm:.2f}, SSIM {sm:.4f})")
    print(f"\n[OK] wrote {args.out}/lambda_tuning.csv and per-ratio sweep plots.")
    print("Note: tuned on the validation split, then applied to the held-out test set.")


if __name__ == "__main__":
    main()
