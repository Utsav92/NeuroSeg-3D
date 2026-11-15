"""Physical-space segmentation metrics for WT / TC / ET.

Label convention (contiguous training labels): 0 background, 1 necrotic/non-enhancing core (NCR/NET),
2 peritumoral edema (ED), 3 enhancing tumor (ET).

Empty-mask conventions (documented in docs/EVALUATION.md):
  * GT empty and prediction empty  -> Dice = IoU = 1, sensitivity = NaN (undefined), precision = NaN,
    specificity computed normally, HD95 = ASSD = 0.
  * exactly one of GT / prediction empty -> Dice = IoU = 0; HD95 / ASSD are undefined and returned as NaN
    (aggregation reports how many cases were excluded). Pass `empty_penalty` to substitute a fixed value.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
METRIC_NAMES = ["dice", "iou", "sensitivity", "specificity", "precision", "hd95", "assd"]


def region_masks(label: np.ndarray, regions: dict | None = None) -> dict[str, np.ndarray]:
    regions = regions or REGIONS
    return {k: np.isin(label, v) for k, v in regions.items()}


def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    if pred.sum() == 0 and gt.sum() == 0:
        dice = iou = 1.0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels: mask minus its 6-connected erosion."""
    er = ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1), border_value=0)
    return mask & ~er


def surface_distances(
    pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Directed surface distances (mm): pred->gt and gt->pred. Requires both masks non-empty."""
    sp, sg = _surface(pred.astype(bool)), _surface(gt.astype(bool))
    dt_g = ndi.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    return dt_g[sp], dt_p[sg]


def surface_metrics(pred, gt, spacing=(1.0, 1.0, 1.0), empty_penalty: float | None = None) -> dict:
    pe, ge = not np.any(pred), not np.any(gt)
    if pe and ge:
        return {"hd95": 0.0, "assd": 0.0}
    if pe or ge:
        v = float("nan") if empty_penalty is None else float(empty_penalty)
        return {"hd95": v, "assd": v}
    d_pg, d_gp = surface_distances(pred, gt, spacing)
    return {
        "hd95": float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))),
        "assd": float((d_pg.sum() + d_gp.sum()) / (d_pg.size + d_gp.size)),
    }


def evaluate_case(pred_label: np.ndarray, gt_label: np.ndarray, spacing, empty_penalty=None) -> dict:
    """Per-region metrics + volumes (mL) for one patient. Returns {region: {metric: value}}."""
    pm, gm = region_masks(pred_label), region_masks(gt_label)
    vox_ml = float(np.prod(spacing)) / 1000.0
    out = {}
    for r in REGIONS:
        m = overlap_metrics(pm[r], gm[r])
        m.update(surface_metrics(pm[r], gm[r], spacing, empty_penalty))
        pv, gv = pm[r].sum() * vox_ml, gm[r].sum() * vox_ml
        m.update(
            pred_volume_ml=float(pv),
            gt_volume_ml=float(gv),
            abs_volume_error_ml=float(abs(pv - gv)),
            rel_volume_error=float((pv - gv) / gv) if gv > 0 else float("nan"),
        )
        out[r] = m
    return out


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Voxel confusion matrix [gt, pred] (diagnostic only; voxels are NOT independent samples)."""
    idx = gt.astype(np.int64).ravel() * num_classes + pred.astype(np.int64).ravel()
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def batch_region_dice(pred_label: np.ndarray, gt_label: np.ndarray) -> dict:
    """Fast Dice per region (used during training validation)."""
    pm, gm = region_masks(pred_label), region_masks(gt_label)
    return {r: overlap_metrics(pm[r], gm[r])["dice"] for r in REGIONS}
