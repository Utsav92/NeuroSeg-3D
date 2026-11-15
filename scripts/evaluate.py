#!/usr/bin/env python
"""Full-volume evaluation of a checkpoint on a split (test split is only ever evaluated here, after model selection)."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from neuroseg.data.pipeline import load_splits
from neuroseg.evaluation.evaluator import evaluate_split
from neuroseg.inference.predictor import Predictor
from neuroseg.training.checkpointing import resolve_checkpoint
from neuroseg.utils.logging import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="checkpoint file or experiment directory")
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--datasets", default=str(ROOT / "configs/datasets.yaml"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--mc-samples", type=int, default=0)
    ap.add_argument("--ensemble", nargs="*", default=[], help="extra checkpoints for a deep ensemble")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="required for --split test (keeps the test set sealed by default)",
    )
    a = ap.parse_args()
    if a.split == "test" and not a.confirm_final_test:
        raise SystemExit(
            "The test split is sealed until final evaluation. Re-run with --confirm-final-test once model selection is finished."
        )
    pred = Predictor(a.checkpoint, a.device, a.ensemble)
    ds_name = a.dataset or pred.info["dataset"]
    cfgs = load_yaml(a.datasets)
    ds = cfgs["datasets"][ds_name]
    sp = load_splits(ds["splits_dir"])
    from neuroseg.data.pipeline import splits_hash

    if pred.info["splits_hash"] and pred.info["splits_hash"] != splits_hash(sp):
        raise SystemExit(
            "splits on disk differ from the splits this checkpoint was trained with; refusing to evaluate (possible leakage)"
        )
    dirs = [Path(ds["processed_dir"]) / c for c in sp[a.split]["cases"]][: a.limit]
    exp = Path(resolve_checkpoint(a.checkpoint)).parent.name
    out = Path(a.out or f"outputs/metrics/{exp}")
    res = evaluate_split(pred, dirs, out, a.overlap, a.tta, a.mc_samples, split_name=a.split)
    print(json.dumps([r for r in res["summary"] if r["metric"] in ("dice", "hd95")], indent=2))
    print("results written to", out)


if __name__ == "__main__":
    main()
