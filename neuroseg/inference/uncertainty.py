"""Predictive uncertainty: Monte-Carlo dropout, deep ensembles, entropy maps.

These quantify the *model's* predictive spread. They are not calibrated clinical risk estimates.
"""

from __future__ import annotations

import numpy as np
import torch

from .sliding_window import sliding_window_predict


def entropy(probs: torch.Tensor | np.ndarray, eps: float = 1e-8):
    """Predictive entropy over the class axis 0 (nats). Max = ln(K)."""
    if isinstance(probs, np.ndarray):
        return -(probs * np.log(probs + eps)).sum(0)
    return -(probs * torch.log(probs + eps)).sum(0)


def has_dropout(model) -> bool:
    return any(isinstance(m, (torch.nn.Dropout, torch.nn.Dropout3d)) and m.p > 0 for m in model.modules())


@torch.no_grad()
def predict_with_uncertainty(
    models,
    image,
    roi,
    overlap=0.5,
    device="cpu",
    mc_samples: int = 0,
    tta: bool = False,
    sw_batch_size: int = 1,
) -> dict:
    """Streaming mean over (ensemble members x MC-dropout samples).

    Returns CPU tensors: mean_probs [K,...], entropy (predictive, total), expected_entropy (aleatoric part),
    mutual_information (epistemic part = total - expected), n_samples.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    if mc_samples > 0:
        for m in models:
            if not has_dropout(m):
                raise ValueError(
                    "MC dropout requested but model has no dropout layers (train with model.dropout > 0)"
                )
    mean, mean_ent, n = None, None, 0
    for m in models:
        runs = max(mc_samples, 1)
        if mc_samples > 0:
            m.eval()
            m.set_dropout_mc(True)
        try:
            for _ in range(runs):
                p = sliding_window_predict(m, image, roi, overlap, sw_batch_size, device, tta=tta)
                e = entropy(p)
                n += 1
                if mean is None:
                    mean, mean_ent = p.clone(), e.clone()
                else:
                    mean += (p - mean) / n
                    mean_ent += (e - mean_ent) / n
        finally:
            if mc_samples > 0:
                m.set_dropout_mc(False)
    total = entropy(mean)
    return {
        "mean_probs": mean,
        "entropy": total,
        "expected_entropy": mean_ent,
        "mutual_information": (total - mean_ent).clamp_min(0),
        "n_samples": n,
    }
