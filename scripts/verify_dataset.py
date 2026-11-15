#!/usr/bin/env python
"""Verify dataset.json, file presence, modality order, label semantics, geometry and record provenance."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import nibabel as nib
import numpy as np
from _bootstrap import ROOT

from neuroseg.data import validation as V
from neuroseg.data.dataset import get_adapter
from neuroseg.data.pipeline import resolve_dataset
from neuroseg.utils import geometry as G
from neuroseg.utils.logging import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--config", default=str(ROOT / "configs/datasets.yaml"))
    ap.add_argument("--full", action="store_true", help="scan labels of every case (default: first 40)")
    a = ap.parse_args()
    cfgs = load_yaml(a.config)
    name, cfg = resolve_dataset(cfgs, a.dataset)
    root = Path(cfg["root"])
    info = V.verify_dataset_json(root)
    adapter = get_adapter(cfg["adapter"], cfg)
    exp = cfg.get("expected", {})
    problems = []
    if exp.get("modalities") and adapter.modalities != exp["modalities"]:
        problems.append(f"modality order {adapter.modalities} != expected {exp['modalities']}")
    if exp.get("n_training") and info["num_training_listed"] != exp["n_training"]:
        problems.append(f"n_training {info['num_training_listed']} != expected {exp['n_training']}")
    recs = adapter.list_cases()
    spacings, orients, shapes, geom_problems, label_vals = Counter(), Counter(), Counter(), [], Counter()
    for i, r in enumerate(recs):
        im = nib.load(r.image_paths[0])
        spacings[tuple(round(s, 3) for s in G.voxel_spacing(im.affine))] += 1
        orients["".join(G.axcodes(im.affine))] += 1
        shapes[tuple(im.shape)] += 1
        geom_problems += V.check_nifti_geometry(
            [Path(p) for p in r.image_paths],
            Path(r.label_path),
            expect_channels=len(r.modalities) if len(r.image_paths) == 1 else None,
        )
        if a.full or i < 40:
            label_vals.update(np.unique(np.asarray(nib.load(r.label_path).dataobj)).astype(int).tolist())
    unknown = set(label_vals) - set(info["labels"])
    if unknown:
        problems.append(f"label values {sorted(unknown)} not declared in dataset.json")
    if geom_problems:
        problems.append(f"{len(geom_problems)} geometry problems, e.g. {geom_problems[0]}")
    if exp.get("spacing_mm") and set(spacings) != {tuple(exp["spacing_mm"])}:
        problems.append(f"spacings observed {dict(spacings)} differ from expected {exp['spacing_mm']}")
    report = {
        "dataset": name,
        "dataset_json_sha256": hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest(),
        "dataset_json": {k: v for k, v in info.items() if k != "case_ids"},
        "modality_order_from_dataset_json": info["modalities"],
        "canonical_modalities": adapter.modalities,
        "label_names": info["labels"],
        "label_map_to_contiguous": adapter.label_map,
        "n_cases": len(recs),
        "voxel_spacings_mm": {str(k): v for k, v in spacings.items()},
        "orientations": dict(orients),
        "shapes": {str(k): v for k, v in shapes.most_common(5)},
        "label_values_observed": sorted(label_vals),
        "labels_scanned": "all" if a.full else min(40, len(recs)),
        "unlabelled_test_images_listed": info["num_test_listed"],
        "problems": problems,
        "download_manifest": (
            json.loads(Path(f"data/manifests/{name}_download.json").read_text())
            if Path(f"data/manifests/{name}_download.json").exists()
            else None
        ),
    }
    Path("data/manifests").mkdir(parents=True, exist_ok=True)
    Path(f"data/manifests/{name}_verification.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "download_manifest"}, indent=2, default=str))
    if problems:
        raise SystemExit("VERIFICATION FAILED")
    print("VERIFICATION PASSED")


if __name__ == "__main__":
    main()
