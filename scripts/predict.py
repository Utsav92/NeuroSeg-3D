#!/usr/bin/env python
"""Run inference on four co-registered MRI modalities. Research use only."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from neuroseg import RESEARCH_ONLY_NOTICE
from neuroseg.evaluation.visualization import case_figure
from neuroseg.inference.predictor import (
    InputError,
    NoCheckpointError,
    Predictor,
    find_modality_files,
    save_result,
)
from neuroseg.utils import geometry as G
from neuroseg.utils.logging import compose_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--input-dir",
        required=True,
        help="directory with one NIfTI per modality (names contain flair/t1/t1ce/t2)",
    )
    ap.add_argument("--output-dir", default="outputs/predictions")
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--config", default=str(ROOT / "configs/inference.yaml"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--mc-samples", type=int, default=None)
    ap.add_argument("--save-probs", action="store_true")
    ap.add_argument("--allow-resample", action="store_true")
    a = ap.parse_args()
    cfg = compose_config(a.config)
    print(RESEARCH_ONLY_NOTICE)
    try:
        p = Predictor(a.checkpoint, a.device or cfg.get("device", "auto"))
        res = p.predict_dir(
            a.input_dir, overlap=cfg.get("overlap", 0.5), tta=a.tta or cfg.get("tta", False),
            mc_samples=a.mc_samples if a.mc_samples is not None else cfg.get("mc_dropout_samples", 0),
            return_probs=a.save_probs or cfg.get("save_probabilities", False),
            allow_resample=a.allow_resample or cfg.get("allow_resample", False),
            min_component_voxels=cfg.get("min_component_voxels", 0),
        )  # fmt: skip
    except (InputError, NoCheckpointError) as e:
        raise SystemExit(f"ERROR: {e}") from e
    cid = a.case_id or Path(a.input_dir).resolve().name
    paths = save_result(res, a.output_dir, cid)
    files = find_modality_files(a.input_dir, p.modalities)
    ref = G.to_canonical(
        p.load_volumes(
            {p.modalities[min(2, len(p.modalities) - 1)]: files[p.modalities[min(2, len(p.modalities) - 1)]]}
        )[p.modalities[min(2, len(p.modalities) - 1)]][0],
        res.affine,
    )[0]
    case_figure(
        ref,
        G.to_canonical(res.labels, res.affine)[0],
        None,
        G.to_canonical(res.entropy, res.affine)[0],
        path=Path(a.output_dir) / f"{cid}_overview.png",
    )
    print(json.dumps(res.volumes["regions"], indent=2))
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
