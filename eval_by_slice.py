# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# eval_by_slice.py
# ----------------
# Volume-aware evaluation of zero-fill / CS-TV / DDNM that:
#   (1) draws slices from DIFFERENT brains (one or a few per volume), so you can
#       evaluate >=50 distinct patients, and
#   (2) records each slice's POSITION within the central-40 band, so you can see
#       whether reconstruction quality depends on slice location (start vs.
#       middle vs. end of the band).
#
# Reuses the same operators/metrics as the main pipeline for a fair comparison.
#
# Typical uses (run from the repo root):
#   # (A) 50 DIFFERENT brains, one central slice each:
#   python eval_by_slice.py --ckpt runs/mag_128/ckpt_100000.pt --data-root data/test \
#       --num-volumes 50 --slices-per-volume 1 --ratios 0.2,0.3,0.5 --out results/by_brain
#
#   # (B) slice-position analysis: fewer brains, many slices spanning the band:
#   python eval_by_slice.py --ckpt runs/mag_128/ckpt_100000.pt --data-root data/test \
#       --num-volumes 10 --slices-per-volume 40 --ratios 0.3 --out results/by_position

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.baseline_cs import cs_tv_recon, zero_fill
from src.kspace import from_complex, seeded_row_mask, subsample_kspace, to_complex
from src.metrics import compute_metrics
from src.diffusion.sample import load_diffusion, reconstruct_ddnm

METHODS = ["zerofill", "cs", "ddnm"]
LABELS = {"zerofill": "Zero-fill", "cs": "CS-MRI (baseline)", "ddnm": "DDNM (ours)"}


# ---- preprocessing identical to dataset.py -------------------------------
def central_band(n, k):
    if k >= n:
        return list(range(n))
    start = (n - k) // 2
    return list(range(start, start + k))


def resize(img, size):
    h, w = img.shape[-2:]
    if h > size:
        img = img[(h - size) // 2:(h - size) // 2 + size, :]
    if w > size:
        img = img[:, (w - size) // 2:(w - size) // 2 + size]
    h, w = img.shape[-2:]
    if h < size or w < size:
        img = np.pad(img, [((size - h) // 2, size - h - (size - h) // 2),
                           ((size - w) // 2, size - w - (size - w) // 2)],
                     mode="constant")
    return img


def to_unit(img):
    img = np.abs(img) if np.iscomplexobj(img) else img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="data/test")
    ap.add_argument("--ratios", default="0.2,0.3,0.5")
    ap.add_argument("--num-volumes", default=50, type=int,
                    help="how many DIFFERENT brains (0 = ALL volumes in data-root)")
    ap.add_argument("--slices-per-volume", default=1, type=int,
                    help="slices sampled from the central band per volume")
    ap.add_argument("--num-central-slices", default=40, type=int, help="band width")
    ap.add_argument("--slice-axis", default=2, type=int)
    ap.add_argument("--steps", default=150, type=int, help="DDNM reverse steps")
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    ap.add_argument("--lam", default=0.01, type=float)
    ap.add_argument("--cs-iter", default=100, type=int)
    ap.add_argument("--tv-iter", default=10, type=int)
    ap.add_argument("--std-scale", default=6.0, type=float)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--out", default="results/by_slice")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    diffusion, channels, size = load_diffusion(args.ckpt, device)
    ratios = [float(r) for r in args.ratios.split(",")]

    files = sorted(glob.glob(os.path.join(args.data_root, "**", "*.npy"), recursive=True))
    num_volumes = len(files) if args.num_volumes in (0, None) else args.num_volumes
    args.num_volumes = num_volumes
    est = num_volumes * args.slices_per_volume * len(ratios)
    print(f"[eval] up to {args.num_volumes} volumes x {args.slices_per_volume} "
          f"slices x {len(ratios)} ratios = ~{est} DDNM reconstructions "
          f"(channels={channels}, size={size})")

    rows = []            # detailed per-slice records (kept in memory for summary)
    gidx = 0             # global counter -> reproducible masks
    vols_used = 0

    # Write the per-slice CSV INCREMENTALLY so partial results survive a crash.
    det = os.path.join(args.out, "results_by_slice.csv")
    det_fh = open(det, "w", newline="")
    det_w = csv.DictWriter(det_fh, fieldnames=["volume", "slice", "band_pos",
                                               "rel_pos", "ratio", "method",
                                               "psnr", "ssim"])
    det_w.writeheader()
    for f in files:
        if vols_used >= args.num_volumes:
            break
        try:
            vol = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if vol.ndim != 3:
            continue
        n = vol.shape[args.slice_axis]
        band = central_band(n, args.num_central_slices)           # indices in volume
        if args.slices_per_volume >= len(band):
            picks = list(range(len(band)))
        elif args.slices_per_volume == 1:
            picks = [len(band) // 2]          # the CENTRAL slice (band middle)
        else:
            picks = np.linspace(0, len(band) - 1, args.slices_per_volume).round().astype(int).tolist()

        for bpos in picks:
            sidx = band[bpos]
            rel = bpos / (len(band) - 1) if len(band) > 1 else 0.5
            sl = np.take(vol, sidx, axis=args.slice_axis)
            # normalize to [-1,1] FIRST, then pad to size (border -> 0 = mid-gray),
            # matching dataset.py exactly so the model sees its training distribution.
            x = resize(to_unit(sl), size)[None, ...]              # (1,H,W) -> treat as C=1
            x_true = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,1,H,W)

            for r in ratios:
                mask2d = seeded_row_mask(size, r, args.seed, gidx,
                                         std_scale=args.std_scale,
                                         device=device).unsqueeze(0)
                z = to_complex(x_true, channels)
                y, _ = subsample_kspace(z, mask2d)
                recs = {
                    "zerofill": from_complex(zero_fill(y[0]), channels),
                    "cs": from_complex(cs_tv_recon(y[0], mask2d[0], lam=args.lam,
                                                   n_iter=args.cs_iter,
                                                   tv_iter=args.tv_iter), channels),
                    "ddnm": reconstruct_ddnm(diffusion, y, mask2d, channels,
                                             steps=args.steps, sampler=args.sampler,
                                             eta=0.0, device=device)[0],
                }
                for m in METHODS:
                    met = compute_metrics(recs[m], x_true[0], channels)
                    rec = {"volume": os.path.basename(f), "slice": sidx,
                           "band_pos": bpos, "rel_pos": round(rel, 3),
                           "ratio": r, "method": m,
                           "psnr": met["psnr"], "ssim": met["ssim"]}
                    rows.append(rec)
                    det_w.writerow(rec)
            gidx += 1
        vols_used += 1
        det_fh.flush()                       # persist this volume's rows now
        if vols_used % 10 == 0:
            print(f"  processed {vols_used} volumes...")

    det_fh.close()
    print(f"[eval] used {vols_used} distinct volumes, {len(rows)} records")
    print(f"[OK] per-slice records -> {det}")

    # ---- (1) summary over DIFFERENT brains: mean+/-std per ratio/method ----
    print(f"\n=== Summary over {vols_used} volumes "
          f"({len(rows)//(len(ratios)*len(METHODS))} slices/ratio) ===")
    print(f"{'ratio':>6} {'method':<18} {'PSNR mean+/-std':>18} {'SSIM mean+/-std':>18}")
    summ = os.path.join(args.out, "summary_by_brain.csv")
    with open(summ, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ratio", "method", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "n"])
        for r in ratios:
            for m in METHODS:
                ps = np.array([x["psnr"] for x in rows if x["ratio"] == r and x["method"] == m])
                ss = np.array([x["ssim"] for x in rows if x["ratio"] == r and x["method"] == m])
                print(f"{r:>5.0%} {LABELS[m]:<18} "
                      f"{ps.mean():>7.2f}+/-{ps.std():<7.2f} "
                      f"{ss.mean():>7.4f}+/-{ss.std():<7.4f}")
                w.writerow([f"{r:.2f}", LABELS[m], f"{ps.mean():.4f}", f"{ps.std():.4f}",
                            f"{ss.mean():.4f}", f"{ss.std():.4f}", len(ps)])
    print(f"[OK] summary -> {summ}")

    # ---- (2) slice-position analysis (only if >1 slice per volume) ----
    if args.slices_per_volume > 1:
        print("\n=== Quality vs. slice position in the central band ===")
        # bin by thirds: start / middle / end
        def bin_of(rel):
            return "start" if rel < 1/3 else ("middle" if rel < 2/3 else "end")
        for r in ratios:
            print(f"[ratio {r:.0%}]  (DDNM)")
            for b in ["start", "middle", "end"]:
                ps = np.array([x["psnr"] for x in rows
                               if x["ratio"] == r and x["method"] == "ddnm"
                               and bin_of(x["rel_pos"]) == b])
                ss = np.array([x["ssim"] for x in rows
                               if x["ratio"] == r and x["method"] == "ddnm"
                               and bin_of(x["rel_pos"]) == b])
                if len(ps):
                    print(f"   {b:>6}: PSNR {ps.mean():5.2f}  SSIM {ss.mean():.4f}  (n={len(ps)})")

        # line plot: metric vs band position, per ratio, for DDNM
        for key, ylab in [("psnr", "PSNR (dB)"), ("ssim", "SSIM")]:
            plt.figure(figsize=(6, 4))
            for r in ratios:
                by_pos = defaultdict(list)
                for x in rows:
                    if x["ratio"] == r and x["method"] == "ddnm":
                        by_pos[x["band_pos"]].append(x[key])
                xs = sorted(by_pos)
                ys = [np.mean(by_pos[p]) for p in xs]
                plt.plot(xs, ys, marker="o", label=f"{r:.0%} kept")
            plt.xlabel("slice position in central band (0 = start, 39 = end)")
            plt.ylabel(ylab)
            plt.title(f"DDNM {ylab} vs. slice position")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out = os.path.join(args.out, f"ddnm_{key}_vs_position.png")
            plt.savefig(out, dpi=130)
            plt.close()
            print(f"[OK] position plot -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
