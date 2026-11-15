"""Dataset verification, geometry checks and leakage assertions (these terminate execution on failure)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from neuroseg.utils import geometry as G


class LeakageError(RuntimeError):
    pass


class DatasetError(RuntimeError):
    pass


def assert_no_leakage(splits: dict[str, list[str]]) -> None:
    """splits: {split_name: [patient_ids]}. Raises LeakageError if any patient is in >1 split or duplicated."""
    seen: dict[str, str] = {}
    for name, ids in splits.items():
        dup = [k for k, v in Counter(ids).items() if v > 1]
        if dup:
            raise LeakageError(f"split '{name}' contains duplicate patient ids: {dup[:5]}")
        for pid in ids:
            if pid in seen:
                raise LeakageError(f"patient '{pid}' appears in both '{seen[pid]}' and '{name}'")
            seen[pid] = name


def assert_case_patient_consistency(
    case_to_patient: dict[str, str], splits_cases: dict[str, list[str]]
) -> None:
    """Every case (time point/study) of a patient must live in the same split."""
    where: dict[str, set] = {}
    for name, cases in splits_cases.items():
        for c in cases:
            where.setdefault(case_to_patient[c], set()).add(name)
    bad = {p: s for p, s in where.items() if len(s) > 1}
    if bad:
        raise LeakageError(f"patients spanning multiple splits via different cases: {bad}")


def verify_dataset_json(root: str | Path, expect_modalities: int | None = 4) -> dict:
    root = Path(root)
    dj = root / "dataset.json"
    if not dj.exists():
        raise DatasetError(f"{dj} not found")
    d = json.loads(dj.read_text())
    mods = [d["modality"][str(i)] for i in range(len(d["modality"]))]
    if expect_modalities and len(mods) != expect_modalities:
        raise DatasetError(f"expected {expect_modalities} modalities, dataset.json lists {mods}")
    labels = {int(k): v for k, v in d["labels"].items()}
    training = d.get("training", [])
    missing = [t for t in training if not (root / t["image"]).exists() or not (root / t["label"]).exists()]
    if missing:
        raise DatasetError(f"{len(missing)} listed training files missing, e.g. {missing[0]}")
    ids = [Path(t["image"]).name.replace(".nii.gz", "") for t in training]
    if len(set(ids)) != len(ids):
        raise DatasetError("duplicate case identifiers in dataset.json")
    return {
        "name": d.get("name"),
        "description": d.get("description"),
        "licence": d.get("licence") or d.get("license"),
        "reference": d.get("reference"),
        "release": d.get("release"),
        "tensorImageSize": d.get("tensorImageSize"),
        "modalities": mods,
        "labels": labels,
        "num_training_listed": len(training),
        "num_test_listed": len(d.get("test", [])),
        "num_training_declared": d.get("numTraining"),
        "case_ids": ids,
    }


def check_nifti_geometry(
    image_paths: list[Path], label_path: Path | None = None, expect_channels: int | None = None
) -> list[str]:
    """Shape/affine consistency of modalities and mask (header-only reads)."""
    shapes, affines, names = [], [], []
    for p in image_paths:
        im = nib.load(str(p))
        shp = im.shape[:3]
        if len(im.shape) == 4 and expect_channels and im.shape[3] != expect_channels:
            return [f"{p.name}: expected {expect_channels} channels, got {im.shape[3]}"]
        shapes.append(shp)
        affines.append(im.affine)
        names.append(p.name)
    if label_path is not None:
        lb = nib.load(str(label_path))
        shapes.append(lb.shape[:3])
        affines.append(lb.affine)
        names.append(label_path.name)
    return G.check_alignment(shapes, affines, names=names)


def validate_inference_inputs(
    volumes: dict[str, tuple[np.ndarray, np.ndarray]], required: list[str], spacing_ref=None, spacing_tol=0.25
) -> list[str]:
    """volumes: {modality: (array3d, affine)}. Returns problems; empty list = OK."""
    problems = [f"missing modality '{m}'" for m in required if m not in volumes]
    if problems:
        return problems
    names = list(required)
    shapes = [volumes[m][0].shape for m in names]
    affines = [volumes[m][1] for m in names]
    for m in names:
        a = volumes[m][0]
        if a.ndim != 3:
            problems.append(f"{m}: expected 3D volume, got shape {a.shape}")
        elif not np.isfinite(a).all():
            problems.append(f"{m}: contains NaN/Inf")
        elif not a.any():
            problems.append(f"{m}: volume is entirely zero")
    problems += G.check_alignment(shapes, affines, names=names)
    if spacing_ref is not None and not problems:
        sp = G.voxel_spacing(affines[0])
        if max(abs(a - b) for a, b in zip(sp, spacing_ref)) > spacing_tol:
            problems.append(
                f"voxel spacing {tuple(round(s, 3) for s in sp)} differs from training spacing {tuple(spacing_ref)} (resample explicitly)"
            )
    return problems


def split_report(splits_cases: dict[str, list[str]], label_available: dict[str, bool] | None = None) -> dict:
    return {
        name: {
            "n_cases": len(c),
            "n_with_labels": sum(1 for x in c if (label_available or {}).get(x, True)),
        }
        for name, c in splits_cases.items()
    }
