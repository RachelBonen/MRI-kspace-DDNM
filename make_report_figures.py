# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# make_report_figures.py
# ----------------------
# Turns the per-slice CSV written by eval_by_slice.py (results_by_slice.csv)
# into the report figures/table, with NO re-computation (reads existing numbers):
#   * metrics_table.csv           PSNR/SSIM mean+/-std per ratio, per method
#   * psnr_vs_ratio.png           CS-TV vs DDNM, error bars = std
#   * ssim_vs_ratio.png
#   * scatter_psnr.png            per-slice CS-TV vs DDNM, with Pearson r
#   * scatter_ssim.png
#
# Run from the repo root after the big evaluation:
#   python make_report_figures.py --csv results/full_diverse/results_by_slice.csv \
#       --out results/full_diverse

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_ORDER = ["zerofill", "cs", "ddnm"]
LABELS = {"zerofill": "Zero-fill", "cs": "CS-MRI (baseline)", "ddnm": "DDNM (ours)"}


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["psnr"] = float(r["psnr"])
        r["ssim"] = float(r["ssim"])
    # normalize method names (accept either raw keys or labels)
    inv = {v: k for k, v in LABELS.items()}
    for r in rows:
        r["method"] = inv.get(r["method"], r["method"])
    ratios = sorted(set(float(r["ratio"]) for r in rows))
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    return rows, ratios, methods


def write_table(rows, ratios, methods, out):
    path = os.path.join(out, "metrics_table.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ratio", "method", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "n"])
        for r in ratios:
            for m in methods:
                ps = np.array([x["psnr"] for x in rows if float(x["ratio"]) == r and x["method"] == m])
                ss = np.array([x["ssim"] for x in rows if float(x["ratio"]) == r and x["method"] == m])
                w.writerow([f"{r:.2f}", LABELS[m], f"{ps.mean():.4f}", f"{ps.std():.4f}",
                            f"{ss.mean():.4f}", f"{ss.std():.4f}", len(ps)])
    print(f"[OK] table -> {path}")


def line_plots(rows, ratios, methods, out):
    for key, ylab, fname in [("psnr", "PSNR (dB)", "psnr_vs_ratio.png"),
                             ("ssim", "SSIM", "ssim_vs_ratio.png")]:
        plt.figure(figsize=(6, 4))
        xs = [r * 100 for r in ratios]
        for m in ["cs", "ddnm"]:      # the two compared methods
            if m not in methods:
                continue
            means = [np.mean([x[key] for x in rows if float(x["ratio"]) == r and x["method"] == m]) for r in ratios]
            stds = [np.std([x[key] for x in rows if float(x["ratio"]) == r and x["method"] == m]) for r in ratios]
            plt.errorbar(xs, means, yerr=stds, marker="o", capsize=4, linewidth=2, label=LABELS[m])
        plt.xlabel("k-space kept (%)")
        plt.ylabel(ylab)
        plt.title(f"{ylab} vs. sampling ratio (mean +/- std)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out, fname), dpi=130)
        plt.close()
        print(f"[OK] plot -> {os.path.join(out, fname)}")


def scatter_plots(rows, ratios, out):
    # pair CS vs DDNM per (volume, slice, ratio)
    for key, lab, fname in [("psnr", "PSNR (dB)", "scatter_psnr.png"),
                            ("ssim", "SSIM", "scatter_ssim.png")]:
        plt.figure(figsize=(5.2, 5.2))
        allx, ally = [], []
        for r in ratios:
            pair = defaultdict(dict)
            for x in rows:
                if float(x["ratio"]) == r and x["method"] in ("cs", "ddnm"):
                    pair[(x["volume"], x["slice"])][x["method"]] = x[key]
            cs = [v["cs"] for v in pair.values() if "cs" in v and "ddnm" in v]
            dd = [v["ddnm"] for v in pair.values() if "cs" in v and "ddnm" in v]
            allx += cs
            ally += dd
            rr = np.corrcoef(cs, dd)[0, 1] if len(cs) > 1 else float("nan")
            plt.scatter(cs, dd, s=12, alpha=0.5, label=f"{r:.0%} kept (r={rr:.2f})")
        lo = min(allx + ally)
        hi = max(allx + ally)
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
        plt.xlabel(f"CS-MRI (baseline) {lab}")
        plt.ylabel(f"DDNM (ours) {lab}")
        plt.title(f"Per-slice {lab}: baseline vs. ours")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out, fname), dpi=130)
        plt.close()
        print(f"[OK] scatter -> {os.path.join(out, fname)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/full_diverse/results_by_slice.csv")
    ap.add_argument("--out", default="results/full_diverse")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, ratios, methods = load(args.csv)
    print(f"[figures] {len(rows)} records, ratios={ratios}, methods={methods}")
    write_table(rows, ratios, methods, args.out)
    line_plots(rows, ratios, methods, args.out)
    scatter_plots(rows, ratios, args.out)
    print("\nDone. Figures + table are consistent with the full evaluation.")


if __name__ == "__main__":
    main()
