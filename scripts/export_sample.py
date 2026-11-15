#!/usr/bin/env python
"""Split one MSD 4D case into per-modality NIfTI files (sample_mri/<case>/) to try scripts/predict.py and the Studio.
The files are copies of CC-BY-SA data for local use only (sample_mri/ is git-ignored); do not redistribute."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import nibabel as nib
import numpy as np
from _bootstrap import ROOT

from neuroseg.data.dataset import get_adapter
from neuroseg.data.pipeline import load_splits, resolve_dataset
from neuroseg.utils.logging import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="sample_mri")
    ap.add_argument(
        "--with-label",
        action="store_true",
        help="also write the ground-truth mask (for the Studio's performance tab)",
    )
    a = ap.parse_args()
    cfgs = load_yaml(str(ROOT / "configs/datasets.yaml"))
    name, cfg = resolve_dataset(cfgs, a.dataset)
    ad = get_adapter(cfg["adapter"], cfg)
    cid = load_splits(cfg["splits_dir"])[a.split]["cases"][a.index]
    rec = next(r for r in ad.list_cases() if r.case_id == cid)
    img, aff, lab, hdr = ad.load(rec)
    out = Path(a.out) / cid
    out.mkdir(parents=True, exist_ok=True)
    for m, c in zip(ad.modalities, img):
        nib.save(nib.Nifti1Image(c.astype(np.float32), aff, hdr), str(out / f"{cid}_{m}.nii.gz"))
    if a.with_label:
        nib.save(nib.Nifti1Image(lab.astype(np.uint8), aff), str(out / f"{cid}_GT_msd_labels.nii.gz"))
    print("wrote", out, "(split:", a.split + ")")


if __name__ == "__main__":
    main()
