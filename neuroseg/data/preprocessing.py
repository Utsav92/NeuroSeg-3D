"""Geometry-preserving preprocessing: canonical orientation, per-volume robust normalization, cropping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from neuroseg.utils import geometry as G

CANONICAL_MODALITIES = ["flair", "t1", "t1ce", "t2"]
NUM_CLASSES = 4
CLASS_NAMES = ["background", "necrotic_nonenhancing", "edema", "enhancing"]


def normalize_channel(x: np.ndarray, clip_pct=(0.5, 99.5), eps: float = 1e-6) -> tuple[np.ndarray, dict]:
    """Per-volume robust z-score on the foreground (nonzero voxels); background stays exactly 0.

    No dataset-level statistics are used, so there is no train/validation leakage by construction.
    Guards: empty foreground -> zeros; zero variance -> subtract mean only.
    """
    x = x.astype(np.float32)
    fg = x != 0
    out = np.zeros_like(x)
    if not fg.any():
        return out, {"empty_foreground": True}
    vals = x[fg]
    lo, hi = np.percentile(vals, clip_pct)
    vals = np.clip(vals, lo, hi)
    mu, sd = float(vals.mean()), float(vals.std())
    if not np.isfinite(sd) or sd < eps:
        out[fg] = vals - mu
        return out, {"mean": mu, "std": sd, "clip": [float(lo), float(hi)], "zero_variance": True}
    out[fg] = (vals - mu) / sd
    return out, {"mean": mu, "std": sd, "clip": [float(lo), float(hi)]}


def normalize_volume(img: np.ndarray, clip_pct=(0.5, 99.5)) -> tuple[np.ndarray, list[dict]]:
    chans, stats = [], []
    for c in img:
        n, s = normalize_channel(c, clip_pct)
        chans.append(n)
        stats.append(s)
    return np.stack(chans), stats


def remap_labels(label: np.ndarray, label_map: dict[int, int]) -> np.ndarray:
    """Map dataset label values to contiguous training labels; unknown values raise (never silently dropped)."""
    present = set(np.unique(label).tolist())
    unknown = present - set(label_map)
    if unknown:
        raise ValueError(f"label values {sorted(unknown)} not in label_map {label_map}")
    lut = np.zeros(max(max(label_map), int(label.max())) + 1, dtype=np.uint8)
    for k, v in label_map.items():
        lut[k] = v
    return lut[label.astype(np.int64)]


def content_hash(img: np.ndarray) -> str:
    """Hash of coarse image content, used to detect exact duplicate studies."""
    sub = np.ascontiguousarray(img[:, ::4, ::4, ::4]).astype(np.float32)
    return hashlib.sha1(sub.tobytes()).hexdigest()


def preprocess_case(
    images: np.ndarray,
    affine: np.ndarray,
    label: np.ndarray | None,
    label_map: dict[int, int] | None,
    modalities: list[str],
    clip_pct=(0.5, 99.5),
    crop_margin: int = 2,
    target_spacing: tuple | None = None,
    spacing_tol: float = 0.05,
) -> dict:
    """images [C,D,H,W] in the *original* orientation. Returns arrays + SpatialMeta dict + stats."""
    if images.ndim != 4:
        raise ValueError("images must be [C,D,H,W]")
    if label is not None and label.shape != images.shape[1:]:
        raise ValueError(f"label shape {label.shape} != image shape {images.shape[1:]}")
    affine = np.asarray(affine, dtype=np.float64)
    orig_shape = tuple(images.shape[1:])
    orig_axcodes = G.axcodes(affine)

    canon = []
    for c in images:
        arr, new_aff = G.to_canonical(c, affine)
        canon.append(arr)
    img = np.stack(canon)
    lab = G.to_canonical(label, affine)[0] if label is not None else None
    canon_shape = tuple(img.shape[1:])
    spacing = G.voxel_spacing(new_aff)

    resampled = False
    if (
        target_spacing is not None
        and np.max(np.abs(np.asarray(spacing) - np.asarray(target_spacing))) > spacing_tol
    ):
        img = np.stack([G.resample(c, spacing, target_spacing, order=1) for c in img])
        lab = G.resample(lab, spacing, target_spacing, order=0) if lab is not None else None
        resampled = True
        spacing_out = tuple(target_spacing)
    else:
        spacing_out = spacing

    img, norm_stats = normalize_volume(img, clip_pct)
    start, shape = G.foreground_bbox(img, crop_margin)
    img_c = G.crop(img, start, shape)
    out = {
        "image": img_c.astype(np.float16),
        "spacing_mm": spacing_out,
        "norm": norm_stats,
        "modalities": modalities,
        "resampled": resampled,
        "canonical_pre_resample_shape": canon_shape,
        "canonical_pre_resample_spacing": spacing,
        "hash": content_hash(img_c),
    }
    meta = G.SpatialMeta(
        original_shape=orig_shape,
        original_affine=affine.tolist(),
        original_axcodes=orig_axcodes,
        canonical_shape=tuple(img.shape[1:]),
        canonical_affine=new_aff.tolist(),
        crop_start=start,
        crop_shape=shape,
        spacing_mm=spacing_out,
    )
    out["spatial"] = meta.to_dict()
    if lab is not None:
        lab = remap_labels(lab, label_map) if label_map is not None else lab.astype(np.uint8)
        lab_c = G.crop(lab, start, shape).astype(np.uint8)
        out["label"] = lab_c
        out["label_counts"] = np.bincount(lab_c.ravel(), minlength=NUM_CLASSES).tolist()
    return out


def save_processed(case_dir: str | Path, out: dict, case_id: str, extra: dict | None = None) -> None:
    d = Path(case_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "image.npy", out["image"])
    if "label" in out:
        np.save(d / "label.npy", out["label"])
    meta = {k: v for k, v in out.items() if k not in ("image", "label")}
    meta["case_id"] = case_id
    meta.update(extra or {})
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def preprocess_for_inference(
    images: np.ndarray,
    affine: np.ndarray,
    modalities: list[str],
    clip_pct=(0.5, 99.5),
    crop_margin: int = 2,
    target_spacing=None,
) -> dict:
    return preprocess_case(images, affine, None, None, modalities, clip_pct, crop_margin, target_spacing)
