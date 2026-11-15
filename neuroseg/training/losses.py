"""Segmentation losses (all differentiable, all computed on the logits' device)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """target [B,D,H,W] int -> [B,C,D,H,W] float."""
    return F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()


def _select(t: torch.Tensor, include_background: bool) -> torch.Tensor:
    return t if include_background else t[:, 1:]


class SoftDiceLoss(nn.Module):
    """Differentiable multi-class soft Dice, computed per sample and class then averaged.

    Empty-class handling (per sample): classes present in the target use `smooth` (1e-5). Classes ABSENT from the
    target use `empty_smooth` (1.0): Dice = s / (sum(p) + s), i.e. no penalty when the predicted mass is ~0 and a
    saturating penalty proportional to false-positive mass otherwise. (With a tiny smoothing constant an absent
    class would score ~0 Dice even for near-perfect predictions and dominate the loss.)
    `classwise()` returns the per-class Dice (mean over batch) for reporting.
    """

    def __init__(
        self,
        include_background: bool = False,
        smooth: float = 1e-5,
        squared: bool = False,
        empty_smooth: float = 1.0,
    ):
        super().__init__()
        self.include_background, self.smooth, self.squared, self.empty_smooth = (
            include_background,
            smooth,
            squared,
            empty_smooth,
        )

    def classwise(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, 1)
        y = one_hot(target, logits.shape[1])
        p, y = _select(p, self.include_background), _select(y, self.include_background)
        dims = (2, 3, 4)
        inter = (p * y).sum(dims)
        denom = (p * p).sum(dims) + (y * y).sum(dims) if self.squared else p.sum(dims) + y.sum(dims)
        present = y.sum(dims) > 0
        sm = torch.where(
            present, torch.full_like(inter, self.smooth), torch.full_like(inter, self.empty_smooth)
        )
        dice = (2 * inter + sm) / (denom + sm)  # [B, C]
        return dice.mean(0)

    def forward(self, logits, target):
        return 1.0 - self.classwise(logits, target).mean()


class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice (Sudre 2017): class weights = 1/(volume^2), computed per sample."""

    def __init__(self, include_background: bool = False, smooth: float = 1e-5):
        super().__init__()
        self.include_background, self.smooth = include_background, smooth

    def forward(self, logits, target):
        p = _select(torch.softmax(logits, 1), self.include_background)
        y = _select(one_hot(target, logits.shape[1]), self.include_background)
        dims = (2, 3, 4)
        w = 1.0 / (y.sum(dims) ** 2 + 1.0)
        inter = (w * (p * y).sum(dims)).sum(1)
        denom = (w * (p.sum(dims) + y.sum(dims))).sum(1)
        return (1.0 - (2 * inter + self.smooth) / (denom + self.smooth)).mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        w = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("w", w)

    def forward(self, logits, target):
        return F.cross_entropy(logits, target.long(), weight=self.w)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        w = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("w", w)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, 1)
        lp = logp.gather(1, target.long().unsqueeze(1)).squeeze(1)
        loss = -((1 - lp.exp()) ** self.gamma) * lp
        if self.w is not None:
            loss = loss * self.w[target.long()]
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky: alpha penalises FP, beta penalises FN (beta>alpha favours recall for small lesions)."""

    def __init__(
        self, alpha: float = 0.3, beta: float = 0.7, include_background: bool = False, smooth: float = 1e-5
    ):
        super().__init__()
        self.alpha, self.beta, self.include_background, self.smooth = alpha, beta, include_background, smooth

    def tversky(self, logits, target):
        p = _select(torch.softmax(logits, 1), self.include_background)
        y = _select(one_hot(target, logits.shape[1]), self.include_background)
        dims = (2, 3, 4)
        tp = (p * y).sum(dims)
        fp = (p * (1 - y)).sum(dims)
        fn = ((1 - p) * y).sum(dims)
        return ((tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)).mean(0)

    def forward(self, logits, target):
        return 1.0 - self.tversky(logits, target).mean()


class FocalTverskyLoss(TverskyLoss):
    def __init__(self, gamma: float = 0.75, **kw):
        super().__init__(**kw)
        self.gamma = gamma

    def forward(self, logits, target):
        return ((1.0 - self.tversky(logits, target)) ** self.gamma).mean()


def signed_distance_map(mask: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Signed distance in millimetres: negative inside the object, positive outside (Kervadec convention).

    Uses scipy's EDT with `sampling=spacing`, so distances are physical. Empty masks -> zeros.
    """
    from scipy.ndimage import distance_transform_edt

    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    out = distance_transform_edt(~mask, sampling=spacing)
    inn = distance_transform_edt(mask, sampling=spacing)
    return (out - inn).astype(np.float32)


class BoundaryLoss(nn.Module):
    """Boundary loss (Kervadec 2019): mean(softmax_fg * signed_distance_gt).

    The distance maps are non-differentiable *targets* computed from labels (CPU/scipy, then moved to the
    logits' device); gradients flow through the softmax probabilities, so no gradient is detached.
    Intended to be combined with a region loss (it is not stable alone).
    """

    def __init__(self, spacing=(1.0, 1.0, 1.0), include_background: bool = False):
        super().__init__()
        self.spacing, self.include_background = spacing, include_background

    def forward(self, logits, target):
        p = torch.softmax(logits, 1)
        C = logits.shape[1]
        t = target.detach().cpu().numpy()
        sdm = np.zeros((t.shape[0], C, *t.shape[1:]), np.float32)
        for b in range(t.shape[0]):
            for c in range(C):
                sdm[b, c] = signed_distance_map(t[b] == c, self.spacing)
        sdm_t = torch.from_numpy(sdm).to(logits.device)
        # normalise by a fixed scale so the magnitude is comparable with region losses
        sdm_t = sdm_t / (max(logits.shape[2:]) * float(np.mean(self.spacing)))
        return (_select(p, self.include_background) * _select(sdm_t, self.include_background)).mean()


class CombinedLoss(nn.Module):
    """Weighted sum of named components, e.g. {dice: 0.5, ce: 0.5}. Loss = sum_i w_i * L_i."""

    def __init__(self, components: dict[str, nn.Module], weights: dict[str, float]):
        super().__init__()
        self.components = nn.ModuleDict(components)
        self.weights = weights

    def forward(self, logits, target):
        return sum(self.weights[k] * m(logits, target) for k, m in self.components.items())

    def breakdown(self, logits, target) -> dict:
        return {k: float(m(logits, target).detach()) for k, m in self.components.items()}


def build_loss(cfg: dict, class_weights=None, spacing=(1.0, 1.0, 1.0)) -> CombinedLoss:
    """cfg example: {components: {dice: 0.5, ce: 0.5}, include_background: false, ce_class_weights: true,
    tversky: {alpha: 0.3, beta: 0.7}, focal_gamma: 2.0}"""
    comps = cfg.get("components", {"dice": 0.5, "ce": 0.5})
    inc = cfg.get("include_background", False)
    cw = class_weights if cfg.get("ce_class_weights", False) else None
    tv = cfg.get("tversky", {})
    mods: dict[str, nn.Module] = {}
    for name in comps:
        if name == "dice":
            mods[name] = SoftDiceLoss(inc)
        elif name == "generalized_dice":
            mods[name] = GeneralizedDiceLoss(inc)
        elif name == "ce":
            mods[name] = CrossEntropyLoss(cw)
        elif name == "focal":
            mods[name] = FocalLoss(cfg.get("focal_gamma", 2.0), cw)
        elif name == "tversky":
            mods[name] = TverskyLoss(tv.get("alpha", 0.3), tv.get("beta", 0.7), inc)
        elif name == "focal_tversky":
            mods[name] = FocalTverskyLoss(
                cfg.get("focal_tversky_gamma", 0.75),
                alpha=tv.get("alpha", 0.3),
                beta=tv.get("beta", 0.7),
                include_background=inc,
            )
        elif name == "boundary":
            mods[name] = BoundaryLoss(spacing, inc)
        else:
            raise ValueError(f"unknown loss component '{name}'")
    return CombinedLoss(mods, {k: float(v) for k, v in comps.items()})


def class_weights_from_counts(counts, mode: str = "inverse_sqrt") -> list[float]:
    """Class weights from *training-split* voxel counts only."""
    c = np.asarray(counts, dtype=np.float64) + 1.0
    f = c / c.sum()
    w = 1.0 / np.sqrt(f) if mode == "inverse_sqrt" else 1.0 / f
    return (w / w.mean()).tolist()
