# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# dataset.py
# Loads fully-sampled MRI slices for training the (unconditional) diffusion
# prior. Supports:
#   * a directory of 3D volumes (.npy / .npz / .nii / .nii.gz), from which
#     central 2D slices are extracted, OR
#   * a directory of pre-extracted 2D slices (.npy).
# Images are normalized to [-1, 1] (the range the diffusion model expects).
# For complex data, real and imaginary parts are stacked as 2 channels.

import glob
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_volume(path: str) -> np.ndarray:
    """Load a 3D array from .npy/.npz or NIfTI (if nibabel is installed)."""
    ext = path.lower()
    if ext.endswith(".npy"):
        return np.load(path)
    if ext.endswith(".npz"):
        arr = np.load(path)
        return arr[arr.files[0]]
    if ext.endswith(".nii") or ext.endswith(".nii.gz"):
        import nibabel as nib  # optional dependency
        return np.asarray(nib.load(path).get_fdata())
    raise ValueError(f"unsupported file type: {path}")


def _central_slice_indices(n: int, num_slices: int) -> List[int]:
    """Return `num_slices` indices centered in [0, n)."""
    if num_slices >= n:
        return list(range(n))
    start = (n - num_slices) // 2
    return list(range(start, start + num_slices))


def _to_unit_range(img: np.ndarray) -> np.ndarray:
    """Scale a single (real) slice to [-1, 1] using its own min/max."""
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    img = (img - lo) / (hi - lo)          # -> [0, 1]
    return (img * 2.0 - 1.0).astype(np.float32)  # -> [-1, 1]


class MRISliceDataset(Dataset):
    """Dataset of fully-sampled MRI slices in [-1, 1].

    Args:
        root: directory containing volumes or slices.
        mode: "volumes" (extract central slices) or "slices" (already 2D).
        num_central_slices: how many central slices per volume (mode="volumes").
        slice_axis: axis along which to slice a 3D volume.
        image_size: output H=W (center-crop / pad).
        channels: 1 for magnitude, 2 for complex (real, imag).
        pattern: glob pattern for files under root.
    """

    def __init__(
        self,
        root: str,
        mode: str = "volumes",
        num_central_slices: int = 40,
        slice_axis: int = 2,
        image_size: int = 256,
        channels: int = 1,
        pattern: str = "*",
    ):
        super().__init__()
        assert mode in ("volumes", "slices")
        assert channels in (1, 2)
        self.mode = mode
        self.image_size = image_size
        self.channels = channels
        self.slice_axis = slice_axis

        files = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
        files = [f for f in files if f.lower().endswith(
            (".npy", ".npz", ".nii", ".nii.gz"))]
        if not files:
            raise FileNotFoundError(f"no data files found under {root}")

        # Build a flat index of (file, slice_idx) so __getitem__ is cheap.
        self.index = []
        if mode == "slices":
            self.index = [(f, None) for f in files]
        else:
            for f in files:
                try:
                    vol = _load_volume(f)
                except Exception as e:  # skip unreadable files but keep going
                    print(f"[dataset] skipping {f}: {e}")
                    continue
                n = vol.shape[slice_axis]
                for s in _central_slice_indices(n, num_central_slices):
                    self.index.append((f, s))
        if not self.index:
            raise RuntimeError("dataset index is empty after scanning files")

        self._cache_path = None
        self._cache_vol = None

    def __len__(self) -> int:
        return len(self.index)

    def _get_slice(self, path: str, s):
        if s is None:  # already a 2D slice file
            return _load_volume(path)
        if path != self._cache_path:  # tiny 1-volume cache for locality
            self._cache_vol = _load_volume(path)
            self._cache_path = path
        return np.take(self._cache_vol, s, axis=self.slice_axis)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Center crop or zero-pad to (image_size, image_size)."""
        h, w = img.shape[-2:]
        size = self.image_size
        # crop
        if h > size:
            top = max((h - size) // 2, 0)
            img = img[..., top:top + size, :]
        if w > size:
            left = max((w - size) // 2, 0)
            img = img[..., :, left:left + size]
        # pad
        h, w = img.shape[-2:]
        pad = [(0, 0)] * (img.ndim - 2) + [
            ((size - h) // 2, size - h - (size - h) // 2),
            ((size - w) // 2, size - w - (size - w) // 2),
        ]
        if h < size or w < size:
            img = np.pad(img, pad, mode="constant")
        return img

    def __getitem__(self, i: int) -> torch.Tensor:
        path, s = self.index[i]
        img = self._get_slice(path, s)
        img = np.asarray(img)

        if self.channels == 2:
            # complex image -> [real, imag] channels, each scaled to [-1, 1]
            if not np.iscomplexobj(img):
                img = img.astype(np.complex64)
            re = _to_unit_range(np.real(img))
            im = _to_unit_range(np.imag(img))
            out = np.stack([re, im], axis=0)
        else:
            # magnitude image, single channel
            if np.iscomplexobj(img):
                img = np.abs(img)
            out = _to_unit_range(img)[None, ...]

        out = self._resize(out)
        return torch.from_numpy(out.astype(np.float32))
