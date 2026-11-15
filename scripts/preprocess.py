#!/usr/bin/env python
"""Canonical orientation + per-volume robust normalisation + foreground crop -> data/processed/<dataset>/<case>/."""

import argparse
import json

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from neuroseg.data.dataset import get_adapter
from neuroseg.data.pipeline import preprocess_dataset, resolve_dataset
from neuroseg.utils.logging import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/datasets.yaml"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfgs = load_yaml(a.config)
    name, cfg = resolve_dataset(cfgs, a.dataset)
    ad = get_adapter(cfg["adapter"], cfg)
    recs = [r for r in ad.list_cases() if r.label_path][: a.limit]
    info = preprocess_dataset(
        ad, cfg["processed_dir"], recs, cfgs["preprocess"], a.overwrite, workers=a.workers
    )
    print(json.dumps({k: v for k, v in info.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
