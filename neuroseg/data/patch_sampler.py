"""3D patch sampling strategies. Always operates on one subject at a time (never pre-split patches)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

MODES = ("brain", "tumor", "boundary", "background")


def extract_patch(arr: np.ndarray, center, size, pad_value=0) -> np.ndarray:
    """Crop `size` around `center` on the last 3 axes, zero-padding outside the array."""
    size = tuple(size)
    shp = arr.shape[-3:]
    start = [int(c) - s // 2 for c, s in zip(center, size)]
    out = np.full((*arr.shape[:-3], *size), pad_value, dtype=arr.dtype)
    src, dst = [], []
    for a, s, n in zip(start, size, shp):
        lo, hi = max(a, 0), min(a + s, n)
        if hi <= lo:
            return out
        src.append(slice(lo, hi))
        dst.append(slice(lo - a, hi - a))
    out[(Ellipsis, *dst)] = np.asarray(arr[(Ellipsis, *src)])
    return out


class PatchSampler:
    """Chooses patch centres.

    modes (probabilities in `probs`):
      brain      random voxel inside the brain (non-zero image) - 'random foreground'
      tumor      class-balanced: pick a present tumour class uniformly, then a random voxel of that class
      boundary   random voxel on the tumour surface (label>0 minus its erosion)
      background random brain voxel with label 0
    Falls back gracefully when a mode has no candidate voxels. `jitter` shifts centres so the target is not
    always centred.
    """

    def __init__(
        self, patch_size, probs: dict | None = None, jitter: float = 0.25, class_balanced: bool = True
    ):
        self.patch = tuple(patch_size)
        self.probs = probs or {"brain": 0.2, "tumor": 0.5, "boundary": 0.15, "background": 0.15}
        self.jitter = jitter
        self.class_balanced = class_balanced

    def _pick(self, mask: np.ndarray, rng) -> tuple | None:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return None
        return tuple(int(v) for v in np.unravel_index(idx[rng.integers(idx.size)], mask.shape))

    def sample_center(
        self, label: np.ndarray, image: np.ndarray | None, rng: np.random.Generator, mode: str | None = None
    ) -> tuple:
        lab = np.asarray(label)
        if mode is None:
            ks = list(self.probs)
            p = np.array([self.probs[k] for k in ks], float)
            mode = ks[int(rng.choice(len(ks), p=p / p.sum()))]
        brain = (np.asarray(image[0]) != 0) if image is not None else np.ones(lab.shape, bool)
        c = None
        if mode == "tumor":
            present = [k for k in np.unique(lab) if k > 0]
            if present:
                k = present[int(rng.integers(len(present)))] if self.class_balanced else None
                c = self._pick(lab == k if k is not None else lab > 0, rng)
        elif mode == "boundary":
            tm = lab > 0
            if tm.any():
                c = self._pick(tm & ~ndi.binary_erosion(tm), rng)
        elif mode == "background":
            c = self._pick(brain & (lab == 0), rng)
        if c is None:
            c = self._pick(brain, rng) or tuple(s // 2 for s in lab.shape)
        if self.jitter > 0:
            c = tuple(
                int(np.clip(ci + rng.integers(-int(p * self.jitter), int(p * self.jitter) + 1), 0, n - 1))
                for ci, p, n in zip(c, self.patch, lab.shape)
            )
        return c
