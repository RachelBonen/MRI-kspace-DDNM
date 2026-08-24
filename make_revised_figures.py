# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# make_revised_figures.py
# -----------------------
# Regenerates ALL image figures for the report + presentation after the
# advisor's comments:
#   (1) brains displayed UPRIGHT (standard axial orientation, anterior at the
#       top) instead of lying on their side,
#   (2) larger panels + a ZOOM-IN row so fine detail is visible at print
#       resolution,
#   (3) a NEW multi-patient figure: several different brains at the SAME slice
#       position, with their reconstructions,
#   (4) the input's sampling ratio (20/30/50%) written on every example.
#
# IMPORTANT: the rotation is DISPLAY-ONLY. All computation (masks, FFT, CS,
# DDNM) runs in the original stored orientation the prior was trained on;
# panels are rotated with np.rot90(k=-1) just before drawing. Mask / k-space
# panels are rotated the same way, so in every figure the undersampled
# (phase-encode) direction is horizontal and the aliasing is consistent.
#
# Run from the repo root (Final_Project_kspace), same environment as
# eval_by_slice.py:
#
#   python make_revised_figures.py --ckpt runs/mag_128/ckpt_100000.pt \
#       --data-root data/test --out figures_v2
#
# Outputs (all under --out):
#   example_slices.png            upright version of the dataset figure
#   eda_undersampling.png         upright masks + zero-filled inputs, % labels
#   baseline_examples.png         upright CS-TV example, % labels, zoom row
#   prior_trajectory.png          upright unconditional-sampling trajectory
#   qualitative_{both_good,ours_wins,both_poor,baseline_wins}.png
#                                 4-column panels + zoom row, % label on input
#   qualitative_{...}_slide.png   single-row versions sized for the slides
#   multi_patient_r{20,30,50}.png N patients at the SAME band position
#   multi_patient_r30_slide.png   slide-sized version (3 patients)
#   picked_cases.csv              which volume/slice each panel shows
#   slide_assets/                 upright 512x512 PNGs to swap into the pptx
#                                 (clean / aliased / kspace / mask / CS / DDNM)
#
# Runtime: ~30-45 min on the 1080Ti with defaults (most of it DDNM sampling).
# Use --quick for a fast smoke test (small pool, 50 steps).

import argparse
import csv
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.baseline_cs import cs_tv_recon, zero_fill
from src.kspace import (fft2c, from_complex, seeded_row_mask,
                        subsample_kspace, to_complex)
from src.metrics import compute_metrics
from src.diffusion.sample import load_diffusion, reconstruct_ddnm

DPI = 220          # high-res output so zoomed print stays sharp

# ---------------------------------------------------------------------------
# display helpers (rotation happens ONLY here)
# ---------------------------------------------------------------------------
ROT_K = -1         # np.rot90 k: -1 = 90deg clockwise -> anterior at the top


def disp(x):
    """Model-space [-1,1] (H,W) array/tensor -> upright [0,1] display array."""
    a = np.asarray(x.detach().cpu()) if hasattr(x, "detach") else np.asarray(x)
    a = np.squeeze(a)
    return np.rot90(np.clip((a + 1.0) / 2.0, 0.0, 1.0), ROT_K)


def disp_mask(m):
    """0/1 mask (H,W) -> upright display (lines become vertical)."""
    a = np.asarray(m.detach().cpu()) if hasattr(m, "detach") else np.asarray(m)
    return np.rot90(np.squeeze(a), ROT_K)


def disp_kspace(z):
    """Complex k-space (H,W) -> upright log-magnitude display in [0,1]."""
    a = np.asarray(z.detach().cpu()) if hasattr(z, "detach") else np.asarray(z)
    a = np.log1p(np.abs(np.squeeze(a)))
    if a.max() > 0:
        a = a / a.max()
    return np.rot90(a, ROT_K)


def find_roi(gt01, box=36, margin=16):
    """Pick the most detail-rich box (highest local gradient energy) in the
    upright ground-truth display image. Returns (row0, col0, box)."""
    gy, gx = np.gradient(gt01)
    energy = gy ** 2 + gx ** 2
    H, W = gt01.shape
    best, best_rc = -1.0, (H // 2 - box // 2, W // 2 - box // 2)
    step = 4
    for r0 in range(margin, H - margin - box, step):
        for c0 in range(margin, W - margin - box, step):
            e = float(energy[r0:r0 + box, c0:c0 + box].sum())
            if e > best:
                best, best_rc = e, (r0, c0)
    return best_rc[0], best_rc[1], box


def imshow_clean(ax, img, title=None, fs=10):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=fs)
    ax.axis("off")


# ---------------------------------------------------------------------------
# preprocessing identical to dataset.py / eval_by_slice.py
# ---------------------------------------------------------------------------
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


def load_slice(vol, sidx, size, slice_axis=2):
    sl = np.take(vol, sidx, axis=slice_axis)
    return resize(to_unit(sl), size)


# ---------------------------------------------------------------------------
# reconstruction of one slice at one ratio (all three methods)
# ---------------------------------------------------------------------------
def recon_all(diffusion, channels, x_np, ratio, gidx, args, device):
    """x_np: (H,W) in [-1,1]. Returns dict of (C,H,W) tensors + metrics."""
    size = x_np.shape[-1]
    x_true = torch.from_numpy(x_np[None, None].astype(np.float32)).to(device)
    mask2d = seeded_row_mask(size, ratio, args.seed, gidx,
                             std_scale=args.std_scale,
                             device=device).unsqueeze(0)
    z = to_complex(x_true, channels)
    y, _ = subsample_kspace(z, mask2d)
    out = {
        "gt": x_true[0],
        "mask": mask2d[0],
        "y": y[0],
        "zerofill": from_complex(zero_fill(y[0]), channels),
        "cs": from_complex(cs_tv_recon(y[0], mask2d[0], lam=args.lam,
                                       n_iter=args.cs_iter,
                                       tv_iter=args.tv_iter), channels),
        "ddnm": reconstruct_ddnm(diffusion, y, mask2d, channels,
                                 steps=args.steps, sampler="ddim",
                                 eta=0.0, device=device)[0],
    }
    for m in ("zerofill", "cs", "ddnm"):
        out[m + "_met"] = compute_metrics(out[m], x_true[0], channels)
    return out


# ---------------------------------------------------------------------------
# figure builders
# ---------------------------------------------------------------------------
def fig_example_slices(files, size, out):
    """8 upright slices spanning the central band of one volume."""
    vol = np.load(files[0], allow_pickle=True)
    band = central_band(vol.shape[2], 40)
    picks = np.linspace(0, len(band) - 1, 8).round().astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.4))
    for ax, bpos in zip(axes.ravel(), picks):
        x = load_slice(vol, band[bpos], size)
        imshow_clean(ax, disp(x), f"band position {bpos}", fs=10)
    fig.suptitle("Fully-sampled axial slices across the central 40-slice band "
                 "(anterior at the top)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "example_slices.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("[OK] example_slices.png")


def fig_eda(files, size, args, out, device):
    """Upright masks + zero-filled inputs with % labels."""
    vol = np.load(files[0], allow_pickle=True)
    band = central_band(vol.shape[2], 40)
    x = load_slice(vol, band[len(band) // 2], size)
    x_true = torch.from_numpy(x[None, None].astype(np.float32)).to(device)
    z = to_complex(x_true, 1)

    ratios = [0.2, 0.3, 0.5]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.6))
    imshow_clean(axes[0, 0], disp(x), "fully sampled", fs=11)
    axes[1, 0].axis("off")
    for j, r in enumerate(ratios, start=1):
        m = seeded_row_mask(size, r, args.seed, 0,
                            std_scale=args.std_scale, device=device)
        y, _ = subsample_kspace(z, m.unsqueeze(0))
        zf = from_complex(zero_fill(y[0]), 1)[0]
        axes[0, j].imshow(disp_mask(m), cmap="gray", aspect="equal",
                          interpolation="nearest")
        axes[0, j].set_title(f"mask, keep {int(r*100)}%", fontsize=11)
        axes[0, j].axis("off")
        imshow_clean(axes[1, j], disp(zf),
                     f"zero-filled input ({int(r*100)}% sampled)", fs=11)
    fig.suptitle("1-D variable-density undersampling: masks and zero-filled "
                 "aliased inputs (upright display; undersampled direction "
                 "horizontal)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "eda_undersampling.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("[OK] eda_undersampling.png")


def fig_baseline_examples(diffusion, channels, files, size, args, out, device):
    """CS-TV example: rows = 20/30/50%, columns = input/CS/GT, + zoom row is
    not needed here (three ratios already); % written on every input."""
    vol = np.load(files[0], allow_pickle=True)
    band = central_band(vol.shape[2], 40)
    x = load_slice(vol, band[len(band) // 2], size)
    ratios = [0.2, 0.3, 0.5]
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 10.2))
    for i, r in enumerate(ratios):
        res = recon_all(diffusion, channels, x, r, 0, args, device)
        imshow_clean(axes[i, 0], disp(res["zerofill"]),
                     f"zero-filled input ({int(r*100)}% sampled)", fs=11)
        imshow_clean(axes[i, 1], disp(res["cs"]),
                     f"CS-TV  ({res['cs_met']['psnr']:.1f} dB)", fs=11)
        imshow_clean(axes[i, 2], disp(res["gt"]), "ground truth", fs=11)
    fig.suptitle("CS-TV baseline across sampling ratios (rows: 20/30/50%)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "baseline_examples.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("[OK] baseline_examples.png")


def fig_prior_trajectory(diffusion, channels, out, device, frames=6):
    """Unconditional DDIM trajectory (no mask / measurement), upright."""
    size = diffusion.image_size
    T = diffusion.timesteps
    acp = diffusion.alphas_cumprod
    steps = 100
    times = torch.linspace(T - 1, 0, steps + 1).long().to(device)
    keep = np.linspace(0, steps - 1, frames).round().astype(int).tolist()

    torch.manual_seed(0)
    x = torch.randn(1, channels, size, size, device=device)
    snaps = []
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((1,), int(times[i]), device=device, dtype=torch.long)
            eps = diffusion.model(x, t)
            x0 = diffusion.predict_x0_from_noise(x, t, eps).clamp(-1, 1)
            if i in keep:
                snaps.append((int(times[i]), x[0, 0].cpu().numpy().copy(),
                              x0[0, 0].cpu().numpy().copy()))
            t_next = int(times[i + 1])
            ac_next = acp[t_next]
            dir_xt = torch.sqrt((1 - ac_next).clamp(min=0)) * eps
            x = torch.sqrt(ac_next) * x0 + dir_xt
    snaps.append((0, x[0, 0].cpu().numpy(), x[0, 0].cpu().numpy()))

    n = len(snaps)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.6))
    for j, (t, xt, x0) in enumerate(snaps):
        imshow_clean(axes[0, j], disp(xt), f"$x_t$,  t={t}", fs=9)
        imshow_clean(axes[1, j], disp(x0), r"$\hat{x}_0$", fs=9)
    fig.suptitle("Unconditional sampling from the trained prior "
                 "(no measurement; anatomy is invented)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "prior_trajectory.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("[OK] prior_trajectory.png")


def qual_panel(res, ratio, path, title, zoom=True, slide=False):
    """One 4-column comparison panel; optionally with a zoom row."""
    cols = [("zerofill", f"Input (zero-filled, {int(ratio*100)}% sampled)"),
            ("cs", f"CS-TV  ({res['cs_met']['psnr']:.1f} dB)"),
            ("ddnm", f"DDNM (ours)  ({res['ddnm_met']['psnr']:.1f} dB)"),
            ("gt", "Ground truth")]
    imgs = {k: disp(res[k]) for k, _ in cols}
    nrow = 2 if zoom else 1
    fs = 13 if slide else 11
    h = 7.6 if zoom else (4.2 if slide else 3.9)
    fig, axes = plt.subplots(nrow, 4, figsize=(14, h))
    axes = np.atleast_2d(axes)
    if zoom:
        r0, c0, box = find_roi(imgs["gt"])
    for j, (key, t) in enumerate(cols):
        imshow_clean(axes[0, j], imgs[key], t, fs=fs)
        if zoom:
            axes[0, j].add_patch(Rectangle((c0, r0), box, box, fill=False,
                                           edgecolor="yellow", linewidth=1.4))
            crop = imgs[key][r0:r0 + box, c0:c0 + box]
            imshow_clean(axes[1, j], crop, None)
            for s in axes[1, j].spines.values():
                s.set_visible(True)
    if zoom:
        axes[1, 0].set_ylabel("zoom", fontsize=fs)
    if title:
        fig.suptitle(title, fontsize=fs + 1)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_qualitative(diffusion, channels, files, size, args, out, device,
                    case_writer):
    """Pool over several volumes x band positions at select-ratio, pick the
    4 canonical cases, render report (zoom) + slide versions."""
    positions = [int(p) for p in args.pool_positions.split(",")]
    pool = []
    gidx = 10_000        # offset so masks differ from the multi-patient figure
    vols = files[:args.pool_volumes]
    print(f"[qualitative] pool = {len(vols)} volumes x {len(positions)} "
          f"positions at {args.select_ratio:.0%} "
          f"({len(vols) * len(positions)} DDNM reconstructions)")
    for f in vols:
        try:
            vol = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if vol.ndim != 3:
            continue
        band = central_band(vol.shape[2], 40)
        for bpos in positions:
            x = load_slice(vol, band[bpos], size)
            res = recon_all(diffusion, channels, x, args.select_ratio,
                            gidx, args, device)
            pool.append({"file": os.path.basename(f), "bpos": bpos,
                         "res": res,
                         "cs": res["cs_met"]["psnr"],
                         "dd": res["ddnm_met"]["psnr"]})
            gidx += 1
        print(f"  done {os.path.basename(f)}")

    dmin = lambda p: min(p["cs"], p["dd"])
    delta = lambda p: p["dd"] - p["cs"]
    picks = {
        "both_good": max(pool, key=dmin),
        "both_poor": min(pool, key=dmin),
        "baseline_wins": min(pool, key=delta),
        "ours_wins": max(pool, key=delta),
    }
    titles = {
        "both_good": "Both methods succeed",
        "both_poor": "Both methods fail (hard edge slice)",
        "baseline_wins": "Largest CS-TV advantage",
        "ours_wins": "DDNM succeeds where CS-TV fails",
    }
    for tag, p in picks.items():
        title = (f"{titles[tag]}  |  volume {p['file']}, band position "
                 f"{p['bpos']}")
        qual_panel(p["res"], args.select_ratio,
                   os.path.join(out, f"qualitative_{tag}.png"),
                   title, zoom=True)
        qual_panel(p["res"], args.select_ratio,
                   os.path.join(out, f"qualitative_{tag}_slide.png"),
                   None, zoom=False, slide=True)
        case_writer.writerow([f"qualitative_{tag}", p["file"], p["bpos"],
                              f"{args.select_ratio:.2f}",
                              f"{p['cs']:.2f}", f"{p['dd']:.2f}"])
        print(f"[OK] qualitative_{tag}(.png/_slide.png)  "
              f"CS {p['cs']:.1f} dB vs DDNM {p['dd']:.1f} dB")
    return picks


def fig_multi_patient(diffusion, channels, files, size, args, out, device,
                      case_writer):
    """Comment 3: SAME band position, several different patients.
    One figure per ratio (rows = patients), plus a slide version at 30%."""
    bpos = args.multi_position
    vols = files[args.pool_volumes:args.pool_volumes + args.multi_patients]
    if len(vols) < args.multi_patients:      # not enough left -> reuse start
        vols = files[:args.multi_patients]
    slices = []
    for f in vols:
        vol = np.load(f, allow_pickle=True)
        band = central_band(vol.shape[2], 40)
        slices.append((os.path.basename(f), load_slice(vol, band[bpos], size)))

    for r in [float(v) for v in args.ratios.split(",")]:
        n = len(slices)
        fig, axes = plt.subplots(n, 4, figsize=(13, 3.35 * n))
        axes = np.atleast_2d(axes)
        for i, (name, x) in enumerate(slices):
            res = recon_all(diffusion, channels, x, r, 20_000 + i, args,
                            device)
            imshow_clean(axes[i, 0], disp(res["zerofill"]),
                         f"Input ({int(r*100)}% sampled)" if i == 0 else None,
                         fs=12)
            imshow_clean(axes[i, 1], disp(res["cs"]),
                         "CS-TV" if i == 0 else None, fs=12)
            axes[i, 1].set_xlabel(f"{res['cs_met']['psnr']:.1f} dB",
                                  fontsize=10)
            axes[i, 1].axis("on")
            axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
            imshow_clean(axes[i, 2], disp(res["ddnm"]),
                         "DDNM (ours)" if i == 0 else None, fs=12)
            axes[i, 2].set_xlabel(f"{res['ddnm_met']['psnr']:.1f} dB",
                                  fontsize=10)
            axes[i, 2].axis("on")
            axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])
            imshow_clean(axes[i, 3], disp(res["gt"]),
                         "Ground truth" if i == 0 else None, fs=12)
            axes[i, 0].axis("on")
            axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
            axes[i, 0].set_ylabel(f"patient {i+1}", fontsize=11)
            case_writer.writerow([f"multi_patient_r{int(r*100)}", name, bpos,
                                  f"{r:.2f}", f"{res['cs_met']['psnr']:.2f}",
                                  f"{res['ddnm_met']['psnr']:.2f}"])
        fig.suptitle(f"Same slice position (band position {bpos}), "
                     f"{n} different patients, {int(r*100)}% sampling",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"multi_patient_r{int(r*100)}.png"),
                    dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] multi_patient_r{int(r*100)}.png")

        if abs(r - 0.3) < 1e-6:
            # slide version: first 3 patients, single compact grid
            m = min(3, n)
            fig, axes = plt.subplots(m, 4, figsize=(13, 3.3 * m))
            axes = np.atleast_2d(axes)
            for i in range(m):
                name, x = slices[i]
                res = recon_all(diffusion, channels, x, r, 20_000 + i, args,
                                device)
                imshow_clean(axes[i, 0], disp(res["zerofill"]),
                             "Input (30% sampled)" if i == 0 else None, fs=13)
                imshow_clean(axes[i, 1], disp(res["cs"]),
                             "CS-TV" if i == 0 else None, fs=13)
                imshow_clean(axes[i, 2], disp(res["ddnm"]),
                             "DDNM (ours)" if i == 0 else None, fs=13)
                imshow_clean(axes[i, 3], disp(res["gt"]),
                             "Ground truth" if i == 0 else None, fs=13)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "multi_patient_r30_slide.png"),
                        dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print("[OK] multi_patient_r30_slide.png")


def slide_assets(diffusion, channels, files, size, args, out, device):
    """Upright 512x512 grayscale PNGs to swap into the presentation media."""
    sub = os.path.join(out, "slide_assets")
    os.makedirs(sub, exist_ok=True)
    vol = np.load(files[0], allow_pickle=True)
    band = central_band(vol.shape[2], 40)
    x = load_slice(vol, band[len(band) // 2], size)
    res = recon_all(diffusion, channels, x, 0.3, 0, args, device)

    def save(img01, name):
        plt.imsave(os.path.join(sub, name), np.kron(img01, np.ones((4, 4))),
                   cmap="gray", vmin=0, vmax=1)

    save(disp(res["gt"]), "clean_upright.png")
    save(disp(res["zerofill"]), "aliased30_upright.png")
    save(disp(res["cs"]), "cs30_upright.png")
    save(disp(res["ddnm"]), "ddnm30_upright.png")
    save(disp_kspace(fft2c(to_complex(
        torch.from_numpy(x[None, None].astype(np.float32)).to(device), 1))[0]),
        "kspace_full_upright.png")
    save(disp_kspace(res["y"]), "kspace_under30_upright.png")
    save(disp_mask(res["mask"]).astype(float), "mask30_upright.png")
    print(f"[OK] slide assets -> {sub}/")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="data/test")
    ap.add_argument("--out", default="figures_v2")
    ap.add_argument("--ratios", default="0.2,0.3,0.5")
    ap.add_argument("--select-ratio", default=0.3, type=float)
    ap.add_argument("--steps", default=100, type=int)
    ap.add_argument("--lam", default=0.01, type=float)
    ap.add_argument("--cs-iter", default=100, type=int)
    ap.add_argument("--tv-iter", default=10, type=int)
    ap.add_argument("--std-scale", default=6.0, type=float)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--pool-volumes", default=12, type=int,
                    help="volumes in the qualitative selection pool")
    ap.add_argument("--pool-positions", default="0,6,13,20,26,33,39",
                    help="band positions sampled per pool volume")
    ap.add_argument("--multi-patients", default=4, type=int,
                    help="patients in the multi-patient figure")
    ap.add_argument("--multi-position", default=20, type=int,
                    help="fixed band position for the multi-patient figure")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: tiny pool, 50 steps")
    args = ap.parse_args()

    if args.quick:
        args.pool_volumes, args.pool_positions = 2, "0,20,39"
        args.multi_patients, args.steps = 2, 50

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    diffusion, channels, size = load_diffusion(args.ckpt, device)
    files = sorted(glob.glob(os.path.join(args.data_root, "**", "*.npy"),
                             recursive=True))
    if not files:
        raise SystemExit(f"[!] no .npy under {args.data_root}")
    print(f"[figures] {len(files)} volumes, device={device}, size={size}")

    cases = open(os.path.join(args.out, "picked_cases.csv"), "w", newline="")
    cw = csv.writer(cases)
    cw.writerow(["figure", "volume", "band_pos", "ratio",
                 "cs_psnr", "ddnm_psnr"])

    with torch.no_grad():
        fig_example_slices(files, size, args.out)
        fig_eda(files, size, args, args.out, device)
        fig_baseline_examples(diffusion, channels, files, size, args,
                              args.out, device)
        fig_prior_trajectory(diffusion, channels, args.out, device)
        slide_assets(diffusion, channels, files, size, args, args.out, device)
        fig_multi_patient(diffusion, channels, files, size, args, args.out,
                          device, cw)
        fig_qualitative(diffusion, channels, files, size, args, args.out,
                        device, cw)
    cases.close()
    print("\nDone. Copy the new PNGs from", args.out,
          "next to the report and rebuild it; use slide_assets/ +"
          " *_slide.png for the presentation.")


if __name__ == "__main__":
    main()
