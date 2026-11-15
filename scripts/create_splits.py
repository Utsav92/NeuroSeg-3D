#!/usr/bin/env python
"""Deterministic patient-level train/validation/test split with hard leakage assertions."""

import argparse
import json

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from neuroseg.data.dataset import get_adapter
from neuroseg.data.pipeline import make_splits, resolve_dataset, save_splits
from neuroseg.utils.logging import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/datasets.yaml"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfgs = load_yaml(a.config)
    name, cfg = resolve_dataset(cfgs, a.dataset)
    seed = a.seed if a.seed is not None else cfgs["split"]["seed"]
    recs = get_adapter(cfg["adapter"], cfg).list_cases()
    splits = make_splits(recs, seed, tuple(cfgs["split"]["fractions"]))
    save_splits(splits, cfg["splits_dir"])
    print(json.dumps(splits["_info"], indent=2))
    print(f"leakage checks passed; wrote {cfg['splits_dir']}/{{train,validation,test}}.json")


if __name__ == "__main__":
    main()
