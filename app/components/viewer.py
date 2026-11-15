"""2D slice rendering with physical aspect ratio, overlays, error maps and uncertainty heatmaps."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuroseg.evaluation.visualization import LABEL_CMAP

AXES = {"Axial": 2, "Coronal": 1, "Sagittal": 0}


def _sl(v, axis, i):
    return np.rot90(np.take(v, i, axis=axis))


def aspect(spacing, axis):
    i, j = [a for a in range(3) if a != axis]
    return spacing[j] / spacing[i]  # rows follow axis j, columns axis i after rot90


def render(
    image,
    spacing,
    plane="Axial",
    idx=0,
    pred=None,
    gt=None,
    entropy=None,
    opacity=0.45,
    show_error=False,
    title=None,
):
    """All arrays canonical (RAS) 3D. Returns a matplotlib figure."""
    axis = AXES[plane]
    asp = aspect(spacing, axis)
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.imshow(_sl(image, axis, idx), cmap="gray", aspect=asp)
    if entropy is not None:
        im = ax.imshow(
            _sl(entropy, axis, idx), cmap="magma", alpha=opacity + 0.2, vmin=0, vmax=np.log(4), aspect=asp
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.set_label("entropy (nats)", color="w")
        cb.ax.yaxis.set_tick_params(color="w", labelcolor="w")
    if pred is not None and not show_error:
        ax.imshow(
            _sl(pred, axis, idx),
            cmap=LABEL_CMAP,
            vmin=0,
            vmax=3,
            alpha=opacity,
            interpolation="nearest",
            aspect=asp,
        )
    if gt is not None and not show_error:
        ax.contour(np.rot90(np.take(gt, idx, axis=axis)) > 0, levels=[0.5], colors="cyan", linewidths=1.0)
    if show_error and pred is not None and gt is not None:
        p, g = _sl(pred, axis, idx) > 0, _sl(gt, axis, idx) > 0
        rgba = np.zeros((*p.shape, 4))
        rgba[p & g] = (0.1, 0.8, 0.2, 0.7)
        rgba[p & ~g] = (0.95, 0.1, 0.1, 0.85)
        rgba[~p & g] = (0.2, 0.4, 1.0, 0.85)
        ax.imshow(rgba, interpolation="nearest", aspect=asp)
    ax.set_axis_off()
    ax.set_title(title or f"{plane} slice {idx}", color="w", fontsize=10)
    fig.tight_layout()
    return fig
