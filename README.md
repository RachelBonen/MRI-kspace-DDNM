# MRI Restoration from Subsampled *k*-space — Diffusion (DDNM) vs. Classical CS

**Course:** Magnetic Resonance Imaging (361.2.6501) — Final Project
**Team:** Yuval Meirom (312121098), Rachel Bonen (318742632)

Reconstruct brain-MRI slices from **1-D variable-density undersampled *k*-space**
(keeping 20% / 30% / 50% of the phase-encode rows). We compare two reconstructors
that share the *same* forward operator, masks and normalization, so the comparison
is fair:

* **Classical baseline** — zero-filling followed by **Compressed-Sensing MRI with
  Total-Variation** regularization, solved with FISTA.
* **Our model** — an **unconditional denoising diffusion prior (DDPM)** used as a
  reconstructor through **DDNM** (Denoising Diffusion Null-space Model), which
  enforces exact *k*-space data consistency at every reverse-diffusion step.

A **single** trained diffusion prior handles all three sampling ratios — only the
undersampling mask changes.

> See `figures/ddnm_data_consistency.svg` for a diagram of the DDNM
> data-consistency reconstruction loop.

---

## 1. Requirements

Python 3.10+. Install the dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` covers PyTorch, NumPy, matplotlib, scikit-image and tqdm.
Two optional packages are only needed for specific inputs/utilities:

* `nibabel` — only if your volumes are NIfTI (`.nii` / `.nii.gz`).
* `pandas` — only for `prepare_data.py` (reads the split metadata CSVs).

A CUDA GPU is strongly recommended for **training** the prior. The baseline,
evaluation and all figure scripts also run on CPU.

```bash
# quick check
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 2. Data (not included)

**No datasets or medical image files are stored in this repository**, as required
by the course guidelines. Place your own fully-sampled volumes (or pre-extracted
2-D slices) under a patient-disjoint train/test split:

```
data/
  train/   # volumes used to train the diffusion prior
  test/    # held-out volumes used for evaluation
```

Supported per-file formats: `.npy`, `.npz`, `.nii`, `.nii.gz`. By default the
loader extracts **central axial slices** from each 3-D volume. The `data/` folder
is git-ignored.

If you have the course Brain-Age dataset laid out with split-metadata CSVs, you can
build the `train/` and `test/` folders automatically (creates symlinks, no copy):

```bash
python prepare_data.py --dataset-root /path/to/brain_age
# use --copy to copy the files instead of symlinking
```

All scripts use **relative paths only** and are meant to be run from the repository
root.

## 3. How to run

### 3.1 Smoke test (≈1–2 min — always run first)

Confirms the full training pipeline works on your machine before spending GPU-hours:

```bash
python -m src.diffusion.train --data-root data/train --image-size 64 \
    --base-channels 32 --channel-mults 1,2,4 --batch-size 4 \
    --steps 200 --log-every 20 --ckpt-every 200 --sample-every 200 \
    --out runs/smoke
```

Success = decreasing loss, plus `runs/smoke/latest.pt` and `runs/smoke/samples_200.png`.

### 3.2 Train the diffusion prior

```bash
python -m src.diffusion.train --data-root data/train --channels 1 \
    --image-size 256 --batch-size 8 --lr 2e-4 --steps 200000 --amp \
    --out runs/mag_256
```

Checkpoints (`ckpt_<step>.pt`, `latest.pt`) hold **EMA** weights, which sampling
uses. Resume with `--resume runs/mag_256/latest.pt`; lower `--batch-size` on CUDA
OOM. `runs/` is git-ignored.

### 3.3 Classical baseline only (no GPU / no trained model needed)

```bash
python run_baseline.py --data-root data/test --ratios 0.2,0.3,0.5 \
    --num-slices 50 --lam 0.01 --out results/baseline
```

Optionally tune the CS-TV regularization strength λ on a validation split:

```bash
python tune_lambda.py --data-root data/train --ratios 0.2,0.3,0.5 \
    --lams 0.002,0.005,0.01,0.02,0.05,0.1 --out results/lambda_tune
```

### 3.4 Evaluate — baseline vs. DDNM (produces the report figures/tables)

```bash
python -m src.evaluate --ckpt runs/mag_256/latest.pt \
    --data-root data/test --ratios 0.2,0.3,0.5 --steps 100 \
    --num-slices 50 --lam 0.01 --out results/eval
```

Outputs in the chosen `--out` folder:

| File | Content |
|---|---|
| `metrics_table.csv` | PSNR / SSIM mean ± std per ratio, per method |
| `psnr_vs_ratio.png`, `ssim_vs_ratio.png` | baseline vs. ours, error bars = std |
| `scatter_psnr.png`, `scatter_ssim.png` | sample-wise, with Pearson *r* |
| `qualitative_*.png` | 4 comparison panels (both-good / both-poor / baseline-wins / ours-wins) |

`--steps` = reverse-diffusion steps (100–250 typical), `--sampler ddpm` for
ancestral sampling, `--lam` = CS-TV strength.

### 3.5 Volume-aware / slice-position evaluation (optional)

`eval_by_slice.py` draws slices from *different* brains and records each slice's
position in the central band, so you can evaluate many distinct patients and check
whether quality depends on slice location:

```bash
python eval_by_slice.py --ckpt runs/mag_256/latest.pt --data-root data/test \
    --num-volumes 100 --slices-per-volume 8 --ratios 0.2,0.3,0.5 \
    --out results/full_diverse
```

### 3.6 DDNM reconstruction only, and unconditional prior samples (optional)

```bash
# reconstruct + dump images without the full comparison
python -m src.diffusion.sample --ckpt runs/mag_256/latest.pt \
    --data-root data/test --ratios 0.2,0.3,0.5 --steps 100 \
    --num-slices 50 --out results/ddnm --save-images

# unconditional samples from the prior (the "prior trajectory" figure)
python sample_unconditional.py --ckpt runs/mag_256/latest.pt \
    --num 4 --steps 1000 --out results/prior_samples
```

## 4. Repository structure

```
Final_Project_kspace/
├── README.md                 # this file
├── requirements.txt
├── PLAN.md                   # design notes
├── prepare_data.py           # build data/train + data/test from a dataset root
├── run_baseline.py           # classical zero-fill + CS-TV baseline (standalone)
├── tune_lambda.py            # CS-TV lambda sweep on a validation split
├── eval_by_slice.py          # volume-aware / slice-position evaluation
├── sample_unconditional.py   # unconditional samples from the trained prior
├── check_axes.py             # sanity-check that volumes share one voxel grid
├── make_eda_figure.py        # data-description (undersampling) figure
├── make_report_figures.py    # turn a per-slice CSV into report tables/plots
├── make_revised_figures.py   # regenerate all report/presentation figures
├── export_real_png.py        # export real training slices as a PNG grid
├── export_slices_from_npy.py # export 2-D PNG slices from 3-D .npy volumes
├── src/
│   ├── kspace.py             # centered FFT/IFFT, seeded VD mask, forward op, DDNM DC
│   ├── metrics.py            # PSNR / SSIM (magnitude and complex real/imag)
│   ├── baseline_cs.py        # zero-fill + CS-MRI (TV, FISTA) classical baseline
│   ├── evaluate.py           # runs all methods -> tables, plots, qualitative panels
│   └── diffusion/
│       ├── model.py          # DDPM U-Net noise predictor
│       ├── diffusion.py      # noise schedule, training loss, DDPM/DDIM samplers
│       ├── dataset.py        # volume/slice loader, normalization, magnitude/complex
│       ├── train.py          # trains the unconditional prior (EMA, AMP, checkpoints)
│       └── sample.py         # DDNM reconstruction loop over the trained prior
├── figures/                  # pipeline + DDNM data-consistency diagrams, EDA figure
├── figures_v2/               # report/presentation figures (comparison panels, plots)
└── results/                  # metrics CSVs and result plots (no raw patient images)
```

Data (`data/`), training runs (`runs/`), and large binaries (`*.npy`, `*.pt`,
`*.zip`, …) are git-ignored — see `.gitignore`.

## 5. Results (800 test slices: 100 patients × 8 slice positions)

Mean PSNR (dB) / SSIM per method and sampling ratio — from
`results/full_diverse/metrics_table.csv`:

| Ratio | Zero-fill | CS-MRI (baseline) | **DDNM (ours)** |
|:---:|:---:|:---:|:---:|
| 20% | 14.7 / 0.423 | 18.6 / 0.677 | **27.6 / 0.832** |
| 30% | 17.5 / 0.520 | 24.3 / 0.817 | **31.7 / 0.920** |
| 50% | 24.5 / 0.722 | 33.4 / 0.948 | **36.8 / 0.974** |

The learned diffusion prior improves PSNR over the classical CS-TV baseline by
**+9.0 / +7.4 / +3.5 dB** at 20% / 30% / 50% sampling — the gain is largest under
the most aggressive undersampling.

## 6. Reproducibility

* Every script takes `--seed` (default 0). Undersampling masks are a deterministic
  function of `(slice index, ratio)`, so the **baseline and the diffusion model see
  identical undersampling** for every slice.
* Both methods share `src/kspace.py`'s operator and `src/metrics.py`'s metric
  definitions — a controlled comparison.
* Per the course guidelines, **datasets and medical image files are never committed**.
