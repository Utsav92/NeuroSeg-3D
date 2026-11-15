"""Synthetic MSD-style NIfTI cases for tests and smoke runs.

These are blobs with modality-dependent intensities - NOT medical images. Performance on them says
nothing about real-world accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np


def make_case(rng: np.random.Generator, shape=(40, 44, 36)) -> tuple[np.ndarray, np.ndarray]:
    """Returns (image [X,Y,Z,4] float32 in MSD order FLAIR,T1w,t1gd,T2w; label [X,Y,Z] in MSD encoding 0/1 edema/2 NET/3 ET)."""
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).astype(np.float32)
    ctr = np.array(shape) / 2
    brain = (((grid - ctr) / (np.array(shape) * 0.45)) ** 2).sum(-1) < 1
    tc = ctr + rng.uniform(-6, 6, 3)
    r = rng.uniform(5, 9)

    def ell(scale):
        return (((grid - tc) / (r * scale * np.array([1.0, 1.1, 0.9]))) ** 2).sum(-1) < 1

    edema, core, enh = ell(1.6) & brain, ell(1.0) & brain, ell(0.5) & brain
    lab = np.zeros(shape, np.int16)
    lab[edema] = 1
    lab[core] = 2
    lab[enh] = 3
    base = rng.uniform(80, 120, 4)
    img = np.zeros((*shape, 4), np.float32)
    # FLAIR bright on edema+core, T1w dark on core, t1gd bright on enhancing, T2w bright on edema+core
    gain = {1: [0.8, -0.1, 0.0, 0.7], 2: [1.0, -0.4, 0.1, 0.9], 3: [1.1, -0.2, 1.2, 0.8]}
    for c in range(4):
        v = base[c] * (1 + 0.15 * rng.standard_normal(shape).astype(np.float32) * 0.2)
        for k, g in gain.items():
            v = np.where(lab == k, base[c] * (1 + g[c]), v)
        img[..., c] = np.where(brain, v + rng.normal(0, 3, shape), 0)
    return np.clip(img, 0, None).astype(np.float32), lab


def write_msd_like(
    root: str | Path, n_cases: int = 8, seed: int = 0, shape=(40, 44, 36), affine: np.ndarray | None = None
) -> Path:
    root = Path(root)
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    aff = np.eye(4) if affine is None else affine
    training = []
    for i in range(n_cases):
        img, lab = make_case(rng, shape)
        name = f"SYNTH_{i:03d}.nii.gz"
        nib.save(nib.Nifti1Image(img, aff), str(root / "imagesTr" / name))
        nib.save(nib.Nifti1Image(lab.astype(np.uint8), aff), str(root / "labelsTr" / name))
        training.append({"image": f"./imagesTr/{name}", "label": f"./labelsTr/{name}"})
    dj = {
        "name": "SYNTHETIC_BRAIN",
        "description": "Synthetic blobs for software tests (NOT medical data)",
        "licence": "CC0",
        "tensorImageSize": "4D",
        "modality": {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"},
        "labels": {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"},
        "numTraining": n_cases,
        "numTest": 0,
        "training": training,
        "test": [],
    }
    (root / "dataset.json").write_text(json.dumps(dj, indent=2))
    return root
