"""Post-processing of probability maps into label maps and volume summaries."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from neuroseg.evaluation.metrics import REGIONS, region_masks
from neuroseg.utils.geometry import volume_ml


def probs_to_labels(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(0).astype(np.uint8)


def remove_small_components(label: np.ndarray, min_voxels: int = 0) -> np.ndarray:
    """Remove tumour connected components (any tumour class, 26-connectivity) smaller than min_voxels."""
    if min_voxels <= 0:
        return label
    cc, n = ndi.label(label > 0, structure=np.ones((3, 3, 3)))
    if n == 0:
        return label
    sizes = ndi.sum(np.ones_like(cc), cc, index=np.arange(1, n + 1))
    small = np.isin(cc, np.flatnonzero(sizes < min_voxels) + 1)
    out = label.copy()
    out[small] = 0
    return out


def region_probability_maps(probs: np.ndarray) -> dict[str, np.ndarray]:
    """P(WT)=p1+p2+p3, P(TC)=p1+p3, P(ET)=p3 from softmax probabilities [4,...]."""
    return {r: probs[list(ids)].sum(0) for r, ids in REGIONS.items()}


def volume_summary(label: np.ndarray, spacing_mm) -> dict:
    """Voxel counts and physical volumes (mL) per region and per tissue class."""
    masks = region_masks(label)
    out = {"voxel_volume_mm3": float(np.prod(spacing_mm)), "regions": {}, "classes": {}}
    for r, m in masks.items():
        out["regions"][r] = {"voxels": int(m.sum()), "volume_ml": volume_ml(m, spacing_mm)}
    for k, name in enumerate(["background", "necrotic_nonenhancing", "edema", "enhancing"]):
        m = label == k
        out["classes"][name] = {"voxels": int(m.sum()), "volume_ml": volume_ml(m, spacing_mm)}
    return out
