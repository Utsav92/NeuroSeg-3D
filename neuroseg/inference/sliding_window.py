"""Gaussian-weighted sliding-window inference for whole volumes."""

from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_importance(roi, sigma_scale: float = 0.125) -> torch.Tensor:
    """3D Gaussian weights (peak at centre) so overlapping patches blend smoothly; min value floored > 0."""
    ks = []
    for r in roi:
        x = torch.arange(r, dtype=torch.float32) - (r - 1) / 2
        ks.append(torch.exp(-0.5 * (x / (sigma_scale * r)) ** 2))
    w = ks[0][:, None, None] * ks[1][None, :, None] * ks[2][None, None, :]
    w = w / w.max()
    return w.clamp_min(1e-3)


def tile_starts(size: int, roi: int, overlap: float) -> list[int]:
    if size <= roi:
        return [0]
    stride = max(int(roi * (1 - overlap)), 1)
    n = int(np.ceil((size - roi) / stride)) + 1
    starts = [min(i * stride, size - roi) for i in range(n)]
    return sorted(set(starts))


def _flips(tta: bool):
    if not tta:
        return [()]
    return [(), (2,), (3,), (4,)]  # identity + single-axis flips (spatial dims of [B,C,D,H,W])


@torch.no_grad()
def sliding_window_predict(
    model,
    image: torch.Tensor,
    roi=(64, 64, 64),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    device: torch.device | str = "cpu",
    gaussian: bool = True,
    tta: bool = False,
    output: str = "prob",
) -> torch.Tensor:
    """image [C,D,H,W] (CPU tensor) -> [K,D,H,W] float32 CPU tensor of softmax probabilities (or logits).

    The volume is zero-padded up to at least `roi`; tiles tile the whole volume (last tile flush with the end);
    accumulation happens on CPU so only a few patches ever live on the accelerator.
    """
    roi = tuple(roi)
    was_training = model.training
    if was_training:
        model.eval()
    C, *shape = image.shape
    pad = [max(r - s, 0) for r, s in zip(roi, shape)]
    x = F.pad(image, (0, pad[2], 0, pad[1], 0, pad[0])) if any(pad) else image
    D, H, W = x.shape[1:]
    starts = list(
        itertools.product(
            tile_starts(D, roi[0], overlap), tile_starts(H, roi[1], overlap), tile_starts(W, roi[2], overlap)
        )
    )
    weight = gaussian_importance(roi) if gaussian else torch.ones(roi)
    out_acc, w_acc = None, torch.zeros(D, H, W)
    flips = _flips(tta)
    for i in range(0, len(starts), sw_batch_size):
        chunk = starts[i : i + sw_batch_size]
        batch = torch.stack([x[:, a : a + roi[0], b : b + roi[1], c : c + roi[2]] for a, b, c in chunk]).to(
            device
        )
        acc = None
        for fl in flips:
            inp = torch.flip(batch, fl) if fl else batch
            o = model(inp)
            o = torch.softmax(o, 1) if output == "prob" else o
            o = torch.flip(o, fl) if fl else o
            acc = o if acc is None else acc + o
        acc = (acc / len(flips)).float().cpu()
        if out_acc is None:
            out_acc = torch.zeros(acc.shape[1], D, H, W)
        for (a, b, c), o in zip(chunk, acc):
            out_acc[:, a : a + roi[0], b : b + roi[1], c : c + roi[2]] += o * weight
            w_acc[a : a + roi[0], b : b + roi[1], c : c + roi[2]] += weight
    out = out_acc / w_acc.clamp_min(1e-8)
    out = out[:, : shape[0], : shape[1], : shape[2]]
    if was_training:
        model.train()
    return out
