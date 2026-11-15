#!/usr/bin/env python
"""Quantitative inspection: geometry, intensities, label semantics and tumour volumes (writes JSON + figure)."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
import numpy as np
from _bootstrap import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroseg.data.dataset import get_adapter
from neuroseg.data.pipeline import resolve_dataset
from neuroseg.data.preprocessing import remap_labels
from neuroseg.utils import geometry as G
from neuroseg.utils.logging import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--config", default=str(ROOT / "configs/datasets.yaml"))
    ap.add_argument("--n", type=int, default=60, help="cases to inspect")
    a = ap.parse_args()
    name, cfg = resolve_dataset(load_yaml(a.config), a.dataset)
    ad = get_adapter(cfg["adapter"], cfg)
    recs = ad.list_cases()
    rng = np.random.default_rng(0)
    pick = [recs[i] for i in sorted(rng.choice(len(recs), min(a.n, len(recs)), replace=False))]
    vols = {"WT": [], "TC": [], "ET": []}
    stats = {m: [] for m in ad.modalities}
    for r in pick:
        img, aff, lab, _ = ad.load(r)
        sp = G.voxel_spacing(aff)
        lab = remap_labels(lab, ad.label_map)
        vox = np.prod(sp) / 1000
        vols["WT"].append(float((lab > 0).sum() * vox))
        vols["TC"].append(float(np.isin(lab, (1, 3)).sum() * vox))
        vols["ET"].append(float((lab == 3).sum() * vox))
        for m, c in zip(ad.modalities, img):
            nz = c[c != 0]
            stats[m].append(
                [
                    float(nz.min()),
                    float(np.percentile(nz, 50)),
                    float(np.percentile(nz, 99.5)),
                    float(nz.max()),
                    float((c != 0).mean()),
                ]
            )
    out = {
        "dataset": name,
        "n_inspected": len(pick),
        "modalities": ad.modalities,
        "label_names_raw": ad.labels if hasattr(ad, "labels") else None,
        "label_map_raw_to_contiguous": ad.label_map,
        "contiguous_classes": {0: "background", 1: "necrotic/non-enhancing", 2: "edema", 3: "enhancing"},
        "volume_ml": {
            k: {
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "n_zero": int(np.sum(np.array(v) == 0)),
            }
            for k, v in vols.items()
        },
        "intensity_min_median_p99.5_max_brainfrac": {
            m: np.mean(v, 0).round(2).tolist() for m, v in stats.items()
        },
    }
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/metrics/{name}_inspection.json").write_text(json.dumps(out, indent=2))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.boxplot([vols[k] for k in vols])
    ax.set_xticks(range(1, len(vols) + 1), list(vols))
    ax.set_ylabel("volume (mL)")
    ax.set_title(f"{name}: ground-truth region volumes (n={len(pick)})")
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"outputs/figures/{name}_volumes.png", dpi=130)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
