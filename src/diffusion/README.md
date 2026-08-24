# Diffusion prior — training

*Team: Yuval Meirom (312121098), Rachel Bonen (318742632)*

Trains an **unconditional** DDPM-style diffusion model on fully-sampled MRI
slices. The model learns the distribution of clean brain MRI images; then
`sample.py` reconstructs subsampled scans by running the reverse
diffusion while enforcing **k-space data consistency** (DDNM). One
trained prior serves all three acceleration ratios (20% / 30% / 50%).

## Files
- `model.py` — U-Net noise predictor (time embedding, residual blocks, attention).
- `diffusion.py` — noise schedule, forward `q_sample`, training loss, DDPM/DDIM samplers.
- `dataset.py` — loads volumes (`.npy/.npz/.nii`) → central slices, or pre-made 2D slices; normalizes to `[-1, 1]`; magnitude (1ch) or complex (2ch).
- `train.py` — training loop with EMA, checkpoints, AMP, and periodic sample grids.

## Install
```bash
pip install -r requirements.txt
```

## Train
Magnitude model (simplest, good for first runs):
```bash
python -m src.diffusion.train \
    --data-root data/train --mode volumes --slice-axis 2 \
    --channels 1 --image-size 256 \
    --batch-size 8 --lr 2e-4 --steps 200000 --amp \
    --out runs/mag_256
```

Complex model (real+imag, needed for the real/imag metrics in the brief):
```bash
python -m src.diffusion.train --data-root data/train --channels 2 \
    --image-size 256 --batch-size 8 --steps 200000 --amp --out runs/cplx_256
```

Quick smoke test on a tiny model/size:
```bash
python -m src.diffusion.train --data-root data/train --image-size 64 \
    --base-channels 32 --channel-mults 1,2,4 --batch-size 4 \
    --steps 200 --log-every 20 --ckpt-every 200 --out runs/smoke
```

## Outputs
`runs/<name>/`: `ckpt_<step>.pt`, `latest.pt` (contains EMA weights under `"ema"`),
and `samples_<step>.png` grids. Use the **EMA** weights for reconstruction.

## Notes
- Set `--seed` for reproducibility (masks/splits/training).
- `--channels 2` expects complex-valued volumes; magnitude-only data should use `--channels 1`.
