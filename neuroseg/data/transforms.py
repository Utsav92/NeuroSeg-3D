"""Training-only augmentations. Spatial ops use ONE shared transform for image and label.

All functions take/return numpy: image [C,D,H,W] float32, label [D,H,W] uint8.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def random_flips(img, lab, rng, p=0.5):
    for ax in range(3):
        if rng.random() < p:
            img = np.flip(img, axis=ax + 1)
            lab = np.flip(lab, axis=ax)
    return np.ascontiguousarray(img), np.ascontiguousarray(lab)


def random_affine(img, lab, rng, max_deg=15.0, scale=(0.9, 1.1), shear=0.0):
    """Rotation + isotropic scale (+ optional shear) via a single affine grid shared by image and label.

    Image: trilinear; label: nearest (so label values are never blended). Out-of-field = 0 / background.
    """
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg, 3))
    cx, cy, cz = np.cos(ang)
    sx, sy, sz = np.sin(ang)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    s = rng.uniform(*scale)
    A = (Rz @ Ry @ Rx) / s
    if shear:
        A = A + rng.uniform(-shear, shear, (3, 3)) * (1 - np.eye(3))
    theta = torch.tensor(np.concatenate([A, np.zeros((3, 1))], 1)[None], dtype=torch.float32)
    x = torch.from_numpy(np.ascontiguousarray(img))[None]
    y = torch.from_numpy(np.ascontiguousarray(lab)).float()[None, None]
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    xo = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0].numpy()
    yo = (
        F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0]
        .numpy()
        .astype(np.uint8)
    )
    return xo, yo


def _bg_mask(img):
    return img != 0


def gaussian_noise(img, rng, std=(0.0, 0.1)):
    m = _bg_mask(img)
    return img + rng.normal(0, rng.uniform(*std), img.shape).astype(np.float32) * m


def gaussian_smooth(img, rng, sigma=(0.5, 1.0)):
    s = rng.uniform(*sigma)
    return np.stack([gaussian_filter(c, s) for c in img]) * _bg_mask(img)


def intensity_scale_shift(img, rng, scale=(0.85, 1.15), shift=0.1):
    m = _bg_mask(img)
    C = img.shape[0]
    a = rng.uniform(*scale, size=(C, 1, 1, 1)).astype(np.float32)
    b = rng.uniform(-shift, shift, size=(C, 1, 1, 1)).astype(np.float32)
    return (img * a + b) * m


def gamma_adjust(img, rng, gamma=(0.7, 1.5)):
    out = img.copy()
    for i, c in enumerate(img):
        m = c != 0
        if not m.any():
            continue
        lo, hi = c[m].min(), c[m].max()
        if hi - lo < 1e-6:
            continue
        g = rng.uniform(*gamma)
        n = (c - lo) / (hi - lo)
        out[i] = np.where(m, np.clip(n, 0, 1) ** g * (hi - lo) + lo, 0)
    return out


def bias_field(img, rng, strength=0.3, grid=4):
    """Smooth multiplicative field exp(strength * lowfreq), different per channel."""
    out = np.empty_like(img)
    for i, c in enumerate(img):
        low = torch.from_numpy(rng.normal(0, 1, (1, 1, grid, grid, grid)).astype(np.float32))
        f = F.interpolate(low, size=c.shape, mode="trilinear", align_corners=True)[0, 0].numpy()
        out[i] = c * np.exp(strength * f)
    return out


def modality_dropout(img, rng, max_drop=1):
    C = img.shape[0]
    if C < 2:
        return img
    k = int(rng.integers(1, min(max_drop, C - 1) + 1))
    drop = rng.choice(C, size=k, replace=False)
    out = img.copy()
    out[drop] = 0
    return out


class Augmenter:
    """Config-driven augmentation pipeline. `level`: 'none' | 'basic' (flips + intensity scale) | 'full'."""

    def __init__(self, level: str = "full", cfg: dict | None = None):
        self.level = level
        self.cfg = {
            "p_flip": 0.5, "p_affine": 0.3, "max_deg": 15.0, "scale": (0.9, 1.1), "p_noise": 0.15, "p_smooth": 0.1,
            "p_scale": 0.25, "p_gamma": 0.15, "p_bias": 0.15, "p_moddrop": 0.1,
        }  # fmt: skip
        self.cfg.update(cfg or {})

    def __call__(self, img: np.ndarray, lab: np.ndarray, rng: np.random.Generator):
        if self.level == "none":
            return img, lab
        c = self.cfg
        img, lab = random_flips(img, lab, rng, c["p_flip"])
        if self.level == "basic":
            if rng.random() < c["p_scale"]:
                img = intensity_scale_shift(img, rng)
            return img.astype(np.float32), lab
        if rng.random() < c["p_affine"]:
            img, lab = random_affine(img, lab, rng, c["max_deg"], tuple(c["scale"]))
        if rng.random() < c["p_noise"]:
            img = gaussian_noise(img, rng)
        if rng.random() < c["p_smooth"]:
            img = gaussian_smooth(img, rng)
        if rng.random() < c["p_scale"]:
            img = intensity_scale_shift(img, rng)
        if rng.random() < c["p_gamma"]:
            img = gamma_adjust(img, rng)
        if rng.random() < c["p_bias"]:
            img = bias_field(img, rng)
        if rng.random() < c["p_moddrop"]:
            img = modality_dropout(img, rng)
        return np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(lab)
