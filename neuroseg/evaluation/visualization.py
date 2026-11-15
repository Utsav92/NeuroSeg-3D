"""Static (matplotlib) and interactive (plotly) visualisations. Segmentation colours are consistent everywhere."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

CLASS_COLORS = {1: "#d62728", 2: "#2ca02c", 3: "#ffd400"}  # necrotic/NET red, edema green, enhancing yellow
CLASS_LABELS = {1: "Necrotic / non-enhancing", 2: "Edema", 3: "Enhancing tumor"}
REGION_COLORS = {"WT": "#2ca02c", "TC": "#d62728", "ET": "#ffd400"}
LABEL_CMAP = ListedColormap(["#00000000", CLASS_COLORS[1], CLASS_COLORS[2], CLASS_COLORS[3]])


def _slice(vol, axis, idx):
    return np.take(vol, idx, axis=axis)


def _orient(s):
    return np.rot90(s)


def overlay_slice(image, label, axis=2, idx=None, alpha=0.45, title=None, ax=None, cmap_img="gray"):
    """image [D,H,W], label [D,H,W] (0..3). Returns matplotlib Axes."""
    idx = image.shape[axis] // 2 if idx is None else idx
    ax = ax or plt.subplots(figsize=(4, 4))[1]
    ax.imshow(_orient(_slice(image, axis, idx)), cmap=cmap_img)
    if label is not None:
        ax.imshow(
            _orient(_slice(label, axis, idx)),
            cmap=LABEL_CMAP,
            vmin=0,
            vmax=3,
            alpha=alpha,
            interpolation="nearest",
        )
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def error_map_slice(pred, gt, axis=2, idx=None, image=None, ax=None):
    """Green = correct tumour, red = false positive, blue = false negative (tumour = any label > 0)."""
    idx = pred.shape[axis] // 2 if idx is None else idx
    ax = ax or plt.subplots(figsize=(4, 4))[1]
    p, g = _orient(_slice(pred, axis, idx)) > 0, _orient(_slice(gt, axis, idx)) > 0
    rgb = np.zeros((*p.shape, 4))
    rgb[p & g] = (0.1, 0.7, 0.2, 0.7)
    rgb[p & ~g] = (0.9, 0.1, 0.1, 0.8)
    rgb[~p & g] = (0.1, 0.3, 0.95, 0.8)
    if image is not None:
        ax.imshow(_orient(_slice(image, axis, idx)), cmap="gray")
    else:
        ax.imshow(np.zeros(p.shape), cmap="gray")
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title("error map (green TP, red FP, blue FN)", fontsize=8)
    return ax


def uncertainty_slice(entropy, axis=2, idx=None, image=None, ax=None, vmax=None):
    idx = entropy.shape[axis] // 2 if idx is None else idx
    ax = ax or plt.subplots(figsize=(4, 4))[1]
    if image is not None:
        ax.imshow(_orient(_slice(image, axis, idx)), cmap="gray")
    im = ax.imshow(
        _orient(_slice(entropy, axis, idx)),
        cmap="magma",
        alpha=0.7 if image is not None else 1.0,
        vmin=0,
        vmax=vmax or np.log(4),
    )
    ax.set_axis_off()
    ax.set_title("predictive entropy (nats)", fontsize=8)
    return im


def case_figure(image, pred, gt=None, entropy=None, axis=2, idx=None, path=None):
    """Panel: image | prediction | (GT) | (error) | (uncertainty) at the slice with the largest predicted tumour."""
    if idx is None:
        other = tuple(a for a in range(3) if a != axis)
        idx = int(np.argmax((pred > 0).sum(other))) if (pred > 0).any() else image.shape[axis] // 2
    n = 2 + (gt is not None) * 2 + (entropy is not None)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    axes = np.atleast_1d(axes)
    k = 0
    overlay_slice(image, None, axis, idx, ax=axes[k], title="MRI")
    k += 1
    overlay_slice(image, pred, axis, idx, ax=axes[k], title="prediction")
    k += 1
    if gt is not None:
        overlay_slice(image, gt, axis, idx, ax=axes[k], title="ground truth")
        k += 1
        error_map_slice(pred, gt, axis, idx, image, ax=axes[k])
        k += 1
    if entropy is not None:
        uncertainty_slice(entropy, axis, idx, image, ax=axes[k])
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def dice_boxplot(df, path=None, metric="dice"):
    fig, ax = plt.subplots(figsize=(5, 4))
    regs = ["WT", "TC", "ET"]
    data = [df[df.region == r][metric].dropna().values for r in regs]
    try:
        ax.boxplot(data, tick_labels=regs, showmeans=True)
    except TypeError:  # matplotlib < 3.9
        ax.boxplot(data, labels=regs, showmeans=True)
    ax.set_ylabel(metric.upper() if metric != "dice" else "Dice")
    ax.set_title(f"Per-patient {metric} (n={df.patient.nunique()})")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def loss_curves(metrics_csv, path=None):
    import pandas as pd

    df = pd.read_csv(metrics_csv)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(df.epoch, df.train_loss, label="train")
    if "val_loss" in df:
        v = df.dropna(subset=["val_loss"])
        axes[0].plot(v.epoch, v.val_loss, label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    for r in ("WT", "TC", "ET"):
        c = f"val_dice_{r}"
        if c in df:
            v = df.dropna(subset=[c])
            axes[1].plot(v.epoch, v[c], label=r, color=REGION_COLORS[r])
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation Dice")
    axes[1].legend()
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def confusion_figure(cm, path=None, names=("bg", "NCR/NET", "edema", "ET")):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cmn[i, j]:.2f}",
                ha="center",
                va="center",
                color="w" if cmn[i, j] > 0.5 else "k",
                fontsize=8,
            )
    fig.colorbar(im)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def reliability_figure(bins, ece, path=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    conf = [b["confidence"] for b in bins if b["count"]]
    acc = [b["accuracy"] for b in bins if b["count"]]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(conf, acc, "o-")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"reliability (ECE={ece:.3f})")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


# ----------------------------------------------------------------------------- 3D (plotly)


def surface_mesh(mask: np.ndarray, spacing_mm, level: float = 0.5, step: int = 2, smooth_sigma: float = 0.8):
    """Marching-cubes surface in PHYSICAL millimetres (spacing applied). Visualisation only - not a
    diagnostic-grade reconstruction. Returns (verts[N,3], faces[M,3]) or None if empty."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    if not np.any(mask):
        return None
    m = np.pad(mask.astype(np.float32), 1)
    if smooth_sigma:
        m = gaussian_filter(m, smooth_sigma)
    try:
        verts, faces, _, _ = marching_cubes(
            m, level=level, spacing=tuple(spacing_mm), step_size=step, allow_degenerate=False
        )
    except (ValueError, RuntimeError):
        return None
    verts = verts - np.array(spacing_mm)  # undo the 1-voxel pad offset
    return verts, faces


def figure_3d(label: np.ndarray, spacing_mm, show=("WT", "TC", "ET"), opacity=None, step=2):
    """Nested translucent surfaces for WT/TC/ET with correct physical aspect."""
    import plotly.graph_objects as go

    from neuroseg.evaluation.metrics import region_masks

    masks = region_masks(label)
    opacity = opacity or {"WT": 0.18, "TC": 0.4, "ET": 0.9}
    fig = go.Figure()
    for r in show:
        mesh = surface_mesh(masks[r], spacing_mm, step=step)
        if mesh is None:
            continue
        v, f = mesh
        fig.add_trace(
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color=REGION_COLORS[r],
                opacity=opacity[r],
                name=r,
                showlegend=True,
                flatshading=True,
            )
        )
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)"),
        margin=dict(l=0, r=0, t=30, b=0), legend=dict(title="Region"), title="Segmentation surfaces (research visualisation)",
    )  # fmt: skip
    return fig


def mesh_to_obj(verts, faces) -> str:
    lines = [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in verts] + [
        f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces
    ]
    return "\n".join(lines)
