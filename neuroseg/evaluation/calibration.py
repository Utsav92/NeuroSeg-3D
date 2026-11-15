"""Calibration and uncertainty-vs-error analysis.

Predictive uncertainty is a statement about the model's output distribution, NOT validated clinical risk.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def reliability(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10, max_voxels: int = 2_000_000, seed: int = 0
) -> dict:
    """Top-label confidence reliability. probs [C,...] softmax, labels [...] ints. Returns ECE, MCE, bin stats.

    Voxels are subsampled (uniformly) for tractability; the per-bin counts are *voxel* counts and must not
    be used for confidence intervals across patients.
    """
    C = probs.shape[0]
    p = probs.reshape(C, -1)
    y = labels.reshape(-1)
    if p.shape[1] > max_voxels:
        idx = np.random.default_rng(seed).choice(p.shape[1], max_voxels, replace=False)
        p, y = p[:, idx], y[idx]
    conf = p.max(0)
    correct = (p.argmax(0) == y).astype(float)
    edges = np.linspace(1.0 / C, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece, mce, rows = 0.0, 0.0, []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            rows.append({"bin": b, "count": 0, "confidence": np.nan, "accuracy": np.nan})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        gap = abs(c - a)
        ece += m.mean() * gap
        mce = max(mce, gap)
        rows.append({"bin": b, "count": int(m.sum()), "confidence": c, "accuracy": a})
    onehot = np.eye(C)[y].T
    brier = float(((p - onehot) ** 2).sum(0).mean())
    return {"ece": float(ece), "mce": float(mce), "brier": brier, "bins": rows}


def entropy_error_agreement(
    entropy: np.ndarray, pred: np.ndarray, gt: np.ndarray, max_voxels: int = 2_000_000, seed: int = 0
) -> dict:
    """Does uncertainty point at errors? AUROC of entropy for detecting misclassified voxels (within brain mask
    = voxels where either pred or gt is non-background, plus a margin is *not* added to keep this definition simple).
    """
    err = (pred != gt).ravel()
    ent = entropy.ravel()
    roi = ((pred > 0) | (gt > 0)).ravel()
    ent, err = ent[roi], err[roi]
    if err.size == 0 or err.all() or not err.any():
        return {
            "auroc_entropy_vs_error": float("nan"),
            "n_voxels": int(err.size),
            "error_rate": float(err.mean()) if err.size else float("nan"),
        }
    if err.size > max_voxels:
        idx = np.random.default_rng(seed).choice(err.size, max_voxels, replace=False)
        ent, err = ent[idx], err[idx]
    return {
        "auroc_entropy_vs_error": float(roc_auc_score(err, ent)),
        "mean_entropy_error": float(ent[err].mean()),
        "mean_entropy_correct": float(ent[~err].mean()),
        "n_voxels": int(err.size),
        "error_rate": float(err.mean()),
    }
