"""Image geometry: canonical orientation, cropping, alignment checks, reversible transforms, volumes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import nibabel as nib
import numpy as np
from nibabel import orientations as nio


@dataclass
class SpatialMeta:
    """Everything needed to map a canonical-space prediction back to the original scan grid."""

    original_shape: tuple
    original_affine: list  # 4x4
    original_axcodes: tuple
    canonical_shape: tuple
    canonical_affine: list
    crop_start: tuple  # (z0,y0,x0) in canonical array coords
    crop_shape: tuple
    spacing_mm: tuple  # canonical spacing

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> SpatialMeta:
        return SpatialMeta(
            tuple(d["original_shape"]),
            d["original_affine"],
            tuple(d["original_axcodes"]),
            tuple(d["canonical_shape"]),
            d["canonical_affine"],
            tuple(d["crop_start"]),
            tuple(d["crop_shape"]),
            tuple(d["spacing_mm"]),
        )


RAS = nio.axcodes2ornt(("R", "A", "S"))


def voxel_spacing(affine: np.ndarray) -> tuple:
    return tuple(float(x) for x in np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(0)))


def to_canonical(arr: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorient a 3D (or 3D+trailing dims) array to RAS+. Returns (array, new_affine). Lossless (flips/transposes)."""
    ornt = nio.io_orientation(affine)
    xform = nio.ornt_transform(ornt, RAS)
    out = nio.apply_orientation(arr, xform)
    new_aff = np.asarray(affine) @ nio.inv_ornt_aff(xform, arr.shape[:3])
    return np.ascontiguousarray(out), new_aff


def from_canonical(arr: np.ndarray, original_affine: np.ndarray) -> np.ndarray:
    """Inverse of `to_canonical` (exact)."""
    ornt = nio.io_orientation(original_affine)
    inv = nio.ornt_transform(RAS, ornt)
    return np.ascontiguousarray(nio.apply_orientation(arr, inv))


def axcodes(affine: np.ndarray) -> tuple:
    return tuple(nio.aff2axcodes(affine))


def foreground_bbox(vol: np.ndarray, margin: int = 2) -> tuple[tuple, tuple]:
    """Bounding box (start, shape) of nonzero voxels over all channels of (C,D,H,W) or (D,H,W)."""
    m = (vol != 0).any(0) if vol.ndim == 4 else (vol != 0)
    if not m.any():
        shp = m.shape
        return (0, 0, 0), tuple(shp)
    idx = np.where(m)
    start = tuple(max(int(i.min()) - margin, 0) for i in idx)
    stop = tuple(min(int(i.max()) + 1 + margin, s) for i, s in zip(idx, m.shape))
    return start, tuple(b - a for a, b in zip(start, stop))


def crop(vol: np.ndarray, start: tuple, shape: tuple) -> np.ndarray:
    sl = tuple(slice(a, a + s) for a, s in zip(start, shape))
    return vol[(Ellipsis, *sl)]


def uncrop(vol: np.ndarray, start: tuple, canon_shape: tuple, fill=0) -> np.ndarray:
    """Place `vol` (…,d,h,w) into zeros of canon_shape at `start`."""
    lead = vol.shape[:-3]
    out = np.full((*lead, *canon_shape), fill, dtype=vol.dtype)
    sl = tuple(slice(a, a + s) for a, s in zip(start, vol.shape[-3:]))
    out[(Ellipsis, *sl)] = vol
    return out


def to_original_space(pred_cropped: np.ndarray, meta: SpatialMeta, fill=0) -> np.ndarray:
    """Cropped canonical-space array -> original scan grid (same shape/orientation as input NIfTI)."""
    canon = uncrop(pred_cropped, meta.crop_start, meta.canonical_shape, fill)
    return from_canonical(canon, np.asarray(meta.original_affine))


def check_alignment(
    shapes: list, affines: list, atol_mm: float = 1e-3, names: list | None = None
) -> list[str]:
    """Return a list of human-readable problems (empty = aligned in the same physical grid)."""
    problems = []
    names = names or [f"vol{i}" for i in range(len(shapes))]
    for n, s, a in zip(names[1:], shapes[1:], affines[1:]):
        if tuple(s[:3]) != tuple(shapes[0][:3]):
            problems.append(f"{n}: shape {tuple(s[:3])} != {names[0]} shape {tuple(shapes[0][:3])}")
        elif not np.allclose(a, affines[0], atol=atol_mm):
            problems.append(
                f"{n}: affine differs from {names[0]} by max {np.abs(np.asarray(a) - affines[0]).max():.4g}"
            )
    return problems


def voxel_volume_ml(spacing_mm: tuple) -> float:
    return float(np.prod(spacing_mm)) / 1000.0


def volume_ml(mask: np.ndarray, spacing_mm: tuple) -> float:
    return float(np.count_nonzero(mask)) * voxel_volume_ml(spacing_mm)


def resample(vol: np.ndarray, spacing: tuple, target: tuple, order: int) -> np.ndarray:
    """Resample (D,H,W) to target spacing. order=1 linear for intensities, order=0 nearest for labels."""
    from scipy.ndimage import zoom

    factors = [s / t for s, t in zip(spacing, target)]
    return zoom(vol, factors, order=order, mode="nearest", prefilter=False)


def save_nifti_like(arr: np.ndarray, affine: np.ndarray, path, header=None, dtype=None) -> None:
    arr = arr.astype(dtype) if dtype is not None else arr
    img = nib.Nifti1Image(arr, affine, header)
    if header is not None:
        img.set_data_dtype(arr.dtype)
    nib.save(img, str(path))


def resample_to_shape(vol: np.ndarray, out_shape: tuple, order: int) -> np.ndarray:
    """Resample the last 3 axes of `vol` to exactly `out_shape` (used to undo `resample`)."""
    from scipy.ndimage import zoom

    if tuple(vol.shape[-3:]) == tuple(out_shape):
        return vol
    f = [o / s for o, s in zip(out_shape, vol.shape[-3:])]
    if vol.ndim == 4:
        return np.stack([zoom(v, f, order=order, mode="nearest", prefilter=False) for v in vol])
    return zoom(vol, f, order=order, mode="nearest", prefilter=False)
