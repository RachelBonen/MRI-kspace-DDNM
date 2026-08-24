# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# evaluate.py
# End-to-end evaluation that produces every figure/table the brief asks for.
# For each test slice and each ratio (20/30/50%) it runs, on the SAME mask:
#   * zero-fill        (lower bound)
#   * CS-MRI (TV)      (classical baseline)
#   * DDNM diffusion   (our model)
# and computes PSNR/SSIM. Outputs:
#   * metrics_table.csv        mean +/- std per ratio per method
#   * psnr_vs_ratio.png, ssim_vs_ratio.png   (baseline vs ours, error bars)
#   * scatter_psnr.png, scatter_ssim.png     (sample-wise, with Pearson r)
#   * qualitative_*.png        the four required example panels
#
# Example:
#   python -m src.evaluate --ckpt runs/ixi_mag/latest.pt \
#       --data-root data/test --ratios 0.2,0.3,0.5 --steps 100 \
#       --num-slices 50 --out results/eval

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch

from .baseline_cs import cs_tv_recon, zero_fill
from .diffusion.dataset import MRISliceDataset
from .diffusion.sample import load_diffusion, reconstruct_ddnm
from .kspace import (from_complex, seeded_row_mask, subsample_kspace,
                     to_complex)
from .metrics import compute_metrics

METHODS = ["zerofill", "cs", "ddnm"]
LABELS = {"zerofill": "Zero-fill", "cs": "CS-MRI (baseline)", "ddnm": "DDNM (ours)"}


def _disp(img_chw) -> np.ndarray:
    """First channel of a model-space (C,H,W) image mapped to [0,1] for plotting."""
    a = np.asarray(img_chw.detach().cpu()) if hasattr(img_chw, "detach") \
        else np.asarray(img_chw)
    return np.clip((a[0] + 1) / 2, 0, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate baseline vs DDNM diffusion")
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--data-root", required=True, type=str)
    p.add_argument("--mode", default="volumes", choices=["volumes", "slices"])
    p.add_argument("--pattern", default="*", type=str)
    p.add_argument("--num-central-slices", default=20, type=int)
    p.add_argument("--slice-axis", default=2, type=int)
    p.add_argument("--ratios", default="0.2,0.3,0.5", type=str)
    p.add_argument("--steps", default=100, type=int)
    p.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--eta", default=0.0, type=float)
    p.add_argument("--std-scale", default=6.0, type=float)
    p.add_argument("--center-lines", default=0, type=int)
    p.add_argument("--num-slices", default=50, type=int)
    p.add_argument("--lam", default=0.01, type=float, help="CS-TV regularization")
    p.add_argument("--cs-iter", default=100, type=int)
    p.add_argument("--tv-iter", default=10, type=int)
    p.add_argument("--select-ratio", default=0.3, type=float,
                   help="ratio used to pick the 4 qualitative examples")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--out", default="results/eval", type=str)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    diffusion, channels, size = load_diffusion(args.ckpt, device)
    ds = MRISliceDataset(root=args.data_root, mode=args.mode, pattern=args.pattern,
                         num_central_slices=args.num_central_slices,
                         slice_axis=args.slice_axis, image_size=size,
                         channels=channels)
    n = len(ds) if args.num_slices in (0, None) else min(args.num_slices, len(ds))
    ratios = [float(r) for r in args.ratios.split(",")]
    print(f"[eval] {n} slices x {len(ratios)} ratios, channels={channels}, "
          f"device={device}")

    # records[(ratio, method)] -> list of dicts {psnr, ssim, idx}
    records = defaultdict(list)
    # qualitative image cache at the selection ratio
    qual = {}  # idx -> dict(input, cs, ddnm, gt, cs_psnr, ddnm_psnr)

    for idx in range(n):
        x_true = ds[idx].unsqueeze(0).to(device)          # (1,C,H,W)
        z_true = to_complex(x_true, channels)             # (1,H,W)
        for r in ratios:
            mask2d = seeded_row_mask(size, r, args.seed, idx,
                                     std_scale=args.std_scale,
                                     center_lines=args.center_lines,
                                     device=device).unsqueeze(0)   # (1,H,W)
            y, _ = subsample_kspace(z_true, mask2d)

            recs = {
                "zerofill": from_complex(zero_fill(y[0]), channels),
                "cs": from_complex(
                    cs_tv_recon(y[0], mask2d[0], lam=args.lam,
                                n_iter=args.cs_iter, tv_iter=args.tv_iter),
                    channels),
                "ddnm": reconstruct_ddnm(diffusion, y, mask2d, channels,
                                         steps=args.steps, sampler=args.sampler,
                                         eta=args.eta, device=device)[0],
            }
            m = {}
            for name in METHODS:
                mt = compute_metrics(recs[name], x_true[0], channels)
                records[(r, name)].append({"idx": idx, **mt})
                m[name] = mt

            if abs(r - args.select_ratio) < 1e-6:
                qual[idx] = {
                    "input": _disp(recs["zerofill"]),
                    "cs": _disp(recs["cs"]),
                    "ddnm": _disp(recs["ddnm"]),
                    "gt": _disp(x_true[0]),
                    "cs_psnr": m["cs"]["psnr"], "ddnm_psnr": m["ddnm"]["psnr"],
                }
        if (idx + 1) % 10 == 0:
            print(f"  processed {idx + 1}/{n} slices")

    _write_table(records, ratios, args.out)
    _line_plots(records, ratios, args.out)
    _scatter_plots(records, ratios, args.out)
    _qualitative(qual, args.select_ratio, args.out)
    print(f"[done] results written to {args.out}")


# ---------------------------------------------------------------------------
def _agg(records, ratio, method, key):
    vals = np.array([d[key] for d in records[(ratio, method)]])
    return vals.mean(), vals.std()


def _write_table(records, ratios, out):
    path = os.path.join(out, "metrics_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ratio", "method", "PSNR_mean", "PSNR_std",
                    "SSIM_mean", "SSIM_std", "n"])
        print("\n=== Metrics (mean +/- std) ===")
        for r in ratios:
            for method in METHODS:
                pm, ps = _agg(records, r, method, "psnr")
                sm, ss = _agg(records, r, method, "ssim")
                nrec = len(records[(r, method)])
                w.writerow([f"{r:.2f}", method, f"{pm:.3f}", f"{ps:.3f}",
                            f"{sm:.4f}", f"{ss:.4f}", nrec])
                print(f"  {r:4.0%}  {LABELS[method]:<18}  "
                      f"PSNR {pm:5.2f}+/-{ps:4.2f}   SSIM {sm:.3f}+/-{ss:.3f}")
    print(f"[table] {path}")


def _line_plots(records, ratios, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, fname, ylab in [("psnr", "psnr_vs_ratio.png", "PSNR (dB)"),
                             ("ssim", "ssim_vs_ratio.png", "SSIM")]:
        plt.figure(figsize=(6, 4.2))
        for method in ["cs", "ddnm"]:
            means = [_agg(records, r, method, key)[0] for r in ratios]
            stds = [_agg(records, r, method, key)[1] for r in ratios]
            xs = [r * 100 for r in ratios]
            plt.errorbar(xs, means, yerr=stds, marker="o", capsize=4,
                         label=LABELS[method])
        plt.xlabel("k-space sampled (%)")
        plt.ylabel(ylab)
        plt.title(f"{ylab} vs sampling ratio")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out, fname), dpi=130)
        plt.close()


def _scatter_plots(records, ratios, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, fname, lab in [("psnr", "scatter_psnr.png", "PSNR (dB)"),
                            ("ssim", "scatter_ssim.png", "SSIM")]:
        plt.figure(figsize=(5.2, 5.2))
        all_b, all_m = [], []
        for r in ratios:
            # align by slice idx
            base = {d["idx"]: d[key] for d in records[(r, "cs")]}
            ours = {d["idx"]: d[key] for d in records[(r, "ddnm")]}
            idxs = sorted(set(base) & set(ours))
            bx = [base[i] for i in idxs]
            my = [ours[i] for i in idxs]
            all_b += bx
            all_m += my
            plt.scatter(bx, my, s=18, alpha=0.6, label=f"{r:.0%}")
        lo = min(all_b + all_m)
        hi = max(all_b + all_m)
        plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        r_pearson = np.corrcoef(all_b, all_m)[0, 1] if len(all_b) > 1 else float("nan")
        plt.xlabel(f"Baseline CS-MRI {lab}")
        plt.ylabel(f"DDNM (ours) {lab}")
        plt.title(f"Sample-wise {lab}   (Pearson r = {r_pearson:.3f})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out, fname), dpi=130)
        plt.close()


def _qualitative(qual, ratio, out):
    if not qual:
        print("[qualitative] no cached slices at the selection ratio")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idxs = list(qual.keys())
    dmin = {i: min(qual[i]["cs_psnr"], qual[i]["ddnm_psnr"]) for i in idxs}
    delta = {i: qual[i]["ddnm_psnr"] - qual[i]["cs_psnr"] for i in idxs}
    picks = {
        "both_good": max(idxs, key=lambda i: dmin[i]),
        "both_poor": min(idxs, key=lambda i: dmin[i]),
        "baseline_wins": min(idxs, key=lambda i: delta[i]),
        "ours_wins": max(idxs, key=lambda i: delta[i]),
    }
    titles = ["Input (zero-fill)", "Baseline CS-MRI", "DDNM (ours)", "Ground truth"]
    for tag, i in picks.items():
        q = qual[i]
        fig, ax = plt.subplots(1, 4, figsize=(12, 3.4))
        for k, (col, t) in enumerate(zip(["input", "cs", "ddnm", "gt"], titles)):
            ax[k].imshow(q[col], cmap="gray", vmin=0, vmax=1)
            ax[k].set_title(t, fontsize=10)
            ax[k].axis("off")
        fig.suptitle(f"{tag}  (slice {i}, {ratio:.0%} sampled)  |  "
                     f"CS {q['cs_psnr']:.2f} dB vs DDNM {q['ddnm_psnr']:.2f} dB",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"qualitative_{tag}.png"), dpi=130)
        plt.close(fig)
    print(f"[qualitative] saved 4 example panels for ratio {ratio:.0%}")


if __name__ == "__main__":
    main()
