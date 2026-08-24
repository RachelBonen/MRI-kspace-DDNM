# Project Plan — Diffusion MRI Restoration vs. the Shaul et al. (2020) GAN

**Task 3.2:** restore MRI slices from 1D variable-density undersampled k-space (20/30/50%).
**Your model:** Family 2 — an *unconditional score-based diffusion prior + k-space data consistency*.
**Comparison target:** the PSNR/SSIM numbers reported in the reference article (Shaul et al., 2020), reproduced on the same dataset for a fair, apples-to-apples comparison.

---

## 0. The core idea in one paragraph

The article trains a **GAN** to fill missing k-space. You instead train a **diffusion model** that learns the distribution of clean brain MRI slices, then at reconstruction time you run the reverse (denoising) process while **forcing the sampled k-space rows to stay equal to what the "scanner" measured**. The diffusion prior hallucinates realistic anatomy; the data-consistency step keeps it honest. This is the modern successor to their GAN and is exactly what makes it a "meaningful variation" for the rubric.

---

## 1. Fair-comparison strategy (read this first — it's the whole point)

You said the baseline should be *the article's reported PSNR/SSIM*. To make that a **valid** comparison and not an apples-to-oranges one, two things must be true:

**(a) Same experimental protocol as the article.** Their numbers only mean something for you if your test images, masks, normalization, and metric definitions match theirs. Their IXI setup:

- Dataset: **IXI**, **T1-weighted**, healthy adults, public at brain-development.org.
- Slices: **80 sagittal slices per volume, 256×256 pixels**.
- Splits: 490 volumes train / 98 val / **86 test volumes** (random).
- Masks: **1D random rows drawn from a normal distribution** (dense center), keeping **50/30/20%** — identical to the project brief.
- PSNR computed over the **entire image including skull** (not brain-only).
- Data is complex; real and imaginary components handled separately.

**(b) You reproduce their classical baselines yourself.** The project rubric *requires a classical, non-DL baseline* (15 pts). Conveniently, the article's own baselines are classical: **zero-filling** and **CS-MRI (TV-based compressed sensing)**. So the plan is:

1. Implement **zero-fill** + **CS-MRI (TV)** yourself. These are your required classical baseline.
2. **Validate your pipeline** by checking that your reproduced zero-fill / CS-MRI PSNR/SSIM land close to the article's Table 1 values (below). If they match, your k-space masking, normalization, and metric code are correct — this is what earns you the "right comparison."
3. Put the article's **published "Proposed" GAN numbers** straight into your results table as the state-of-the-art anchor you're trying to beat/approach with diffusion.

> Net result: your results table has **three** contenders per ratio — classical baseline (yours, validated), the article's GAN (published), and your diffusion model (yours). That's a stronger story than the assignment even asks for.

### Article IXI numbers to reproduce / beat (Table 1, PSNR / SSIM)

| Method | 20% | 30% | 50% |
|---|---|---|---|
| Zero-filled | 25.96 / 0.69 | 28.61 / 0.77 | 31.50 / 0.84 |
| CS-MRI (TV) | 29.95 / 0.86 | 33.80 / 0.93 | 38.19 / 0.97 |
| ADMM-Net | 29.63 / 0.81 | 32.76 / 0.90 | 34.27 / 0.94 |
| **Proposed (GAN)** | **31.02 / 0.93** | **35.46 / 0.97** | **38.14 / 0.98** |

*Your targets: reproduce the zero-fill and CS-MRI rows; aim your diffusion model at the "Proposed (GAN)" row.*

> ⚠️ If you cannot use IXI (e.g., only the course HPC dataset is available), you can still run the full experiment, but then you must **re-run zero-fill + CS-MRI on your dataset** and compare diffusion vs. *those* numbers — you cannot compare your diffusion directly to the article's table, because the datasets differ. Decide this early (see §8, step 1).

---

## 2. Dataset & preprocessing

- **Get IXI T1** (register/download from brain-development.org). If size/time is a concern, use a subset of volumes and/or the **central sagittal slices** (allowed by the brief); apply the *same* slice rule to every method and split, and document it.
- Convert each 3D volume → 2D slices (256×256). Keep the article's train/val/test split style (patient-disjoint).
- **Normalization:** fix one scheme and reuse everywhere (e.g., per-slice scale by max magnitude, or a global percentile scale). Record it; PSNR/SSIM are sensitive to it.
- **Complex handling:** store each slice as a 2-channel array (real, imag) so you can report PSNR/SSIM on real and imag separately, as the brief requires. Start on magnitude to debug, then move to complex.
- **Seed everything** (mask generation, splits, training) for reproducibility.

## 3. k-space subsampling (shared by all methods)

`x (clean) → FFT2 → apply 1D Gaussian variable-density row mask (keep 20/30/50%, no replacement, seeded) → zero-fill missing rows → IFFT2 → x_zf (aliased complex input)`.

Save, for each slice: the mask `M`, the measured k-space `y = M ⊙ FFT(x)`, and `x_zf`. `y` and `M` are what the data-consistency step needs.

## 4. Baseline (classical, required) — reproduce the article's

- **Zero-filled:** `IFFT2(zero-filled k-space)` = `x_zf`. Lower bound.
- **CS-MRI (TV):** solve `min_x ½‖M·FFT(x) − y‖² + λ·TV(x)` with ISTA/ADMM (a short iterative loop, or use `sigpy`/`bart`). Tune `λ` on validation. This is the strong classical opponent and the article's main non-DL benchmark.
- Produce a simple pipeline diagram (undersample → CS/TV solver → restored image).

## 5. Your model — Family 2 diffusion + data consistency

**5a. Train an unconditional diffusion prior** on clean IXI slices:
- Denoiser: standard **U-Net noise predictor** (DDPM-style), operating on 2-channel complex (or 1-channel magnitude) 256×256 images.
- Objective: standard denoising/score-matching loss (predict added noise ε at random timesteps).
- Trains *once* on clean data — no mask needed during training. The same prior serves all three ratios.

**5b. Reconstruct with measurement guidance** (pick one; all enforce data consistency):
- **DDNM (recommended, simplest for a known linear mask):** at each reverse step, decompose the estimate into range + null space of the measurement operator and overwrite the range part with the measured `y`. Clean, few hyperparameters, strong for Cartesian MRI masks.
- **Score-MRI / predictor–corrector (Chung & Ye, 2022):** unconditional score model + alternating data-consistency projection (replace sampled rows with `y` each step).
- **DPS (Diffusion Posterior Sampling):** add a gradient step on `‖M·FFT(x̂₀) − y‖²` at each timestep. Most flexible, a bit more tuning.
- Use **DDIM** sampling to cut inference steps (speed) — you have GPU but you'll run 3 ratios × 86 test volumes × 80 slices, so efficiency matters.

**5c. Hard data consistency at the end:** after sampling, do a final `K_out = M⊙y + (1−M)⊙FFT(x̂)` so the measured lines are exactly preserved. Report this as the final image.

Deliverables for the report: architecture diagram, noise schedule, loss, sampler + DC method, data splits.

## 6. Evaluation & required outputs

Per the brief, for **each ratio (20/30/50%)** and each test slice, compute **PSNR** and **SSIM** (real and imag separately; consistent normalization). Then produce:

1. **Table:** mean ± std PSNR/SSIM, one row per ratio, for baseline and diffusion (plus the article's published GAN/CS rows as reference).
2. **Two line plots** (PSNR, SSIM) vs. sampling ratio, one line per method, with error bars / shaded std.
3. **Scatter plots:** sample-wise baseline-vs-diffusion for PSNR and SSIM, with **Pearson r** annotated, ratios marked.
4. **Four qualitative examples** (input / baseline / yours / ground truth): both good, both bad, baseline-wins, yours-wins.

## 7. Repo structure

```
Final_Project_kspace/
  README.md
  data/            # (not committed) IXI slices
  src/
    masking.py     # seeded Gaussian variable-density mask, undersample, zero-fill
    metrics.py     # PSNR, SSIM (real/imag), normalization
    baseline_cs.py # zero-fill + TV compressed sensing
    diffusion/
      model.py     # U-Net noise predictor
      train.py     # train unconditional prior
      sample.py    # DDNM / Score-MRI / DPS reconstruction + data consistency
    evaluate.py    # tables, line plots, scatter+Pearson, qualitative panels
  notebooks/       # EDA + figures
```
Each file starts with team names + IDs; relative paths only; fixed seeds; runs end-to-end.

## 8. Milestones

1. **Decide dataset** — IXI (to match the article) vs. course HPC dataset. IXI is required if you want to compare directly to Table 1. → get data, build slice pipeline.
2. **Masking + metrics harness** (§3, §6); verify on one slice visually.
3. **Baseline:** zero-fill + CS-MRI; **check numbers against Table 1** to validate the pipeline.
4. **Diffusion prior:** train U-Net denoiser on clean slices.
5. **Reconstruction:** add DDNM/Score-MRI data consistency; run 20/30/50%.
6. **Evaluate:** all tables/plots/scatters + 4 qualitative examples.
7. **Report + README + GitHub.**

## 9. Risks / things to watch

- **Protocol drift:** if your reproduced CS-MRI/zero-fill don't match Table 1, your masks/normalization/PSNR definition differ — fix before trusting any comparison.
- **Hallucination at 20%:** diffusion can invent anatomy; data consistency limits it. Discuss explicitly as a medical failure mode (rubric: interpretation).
- **Complex vs magnitude:** commit early; metrics require real/imag separately.
- **Runtime:** iterative sampling × many slices — use DDIM, subset the test set if needed (document it).
- **Fairness:** identical slices, seeds, masks, normalization across all methods.

---

*Next: I can scaffold `src/` with runnable stubs for `masking.py`, `metrics.py`, and `baseline_cs.py`, or start the diffusion training script — say the word.*
