#!/usr/bin/env python
"""Train a model from random initialisation.

Config composition (later wins):  configs/datasets.yaml  <  --config  <  --hardware  <  --set key=value
Example: python scripts/train.py --config configs/model_resunet.yaml --hardware configs/train_mac.yaml
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml
from _bootstrap import ROOT

from neuroseg.data.dataset import MSDBrainAdapter
from neuroseg.data.pipeline import make_splits, preprocess_dataset, resolve_dataset, save_splits
from neuroseg.data.synthetic import write_msd_like
from neuroseg.training.trainer import Trainer
from neuroseg.utils.logging import compose_config, load_yaml


def prepare_synthetic(cfg):
    s = cfg["synthetic"]
    root = write_msd_like(s["raw_dir"], s["n_cases"], cfg.get("seed", 0), tuple(s["shape"]))
    ad = MSDBrainAdapter(root)
    recs = ad.list_cases()
    save_splits(make_splits(recs, cfg.get("seed", 0)), s["splits_dir"])
    preprocess_dataset(
        ad, s["processed_dir"], recs, cfg.get("preprocess", {}), overwrite=True, log=lambda *_: None
    )
    cfg["data"]["processed_dir"], cfg["data"]["splits_dir"] = s["processed_dir"], s["splits_dir"]
    cfg["label_info"] = {"label_map": ad.label_map, "raw_labels": ad.labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--datasets", default=str(ROOT / "configs/datasets.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=5")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = compose_config(a.config, a.hardware, overrides=a.set)
    if "synthetic" in cfg:
        prepare_synthetic(cfg)
    else:
        dcfg = load_yaml(a.datasets)
        name, ds = resolve_dataset(dcfg, cfg["data"]["dataset"])
        cfg["data"]["processed_dir"], cfg["data"]["splits_dir"] = ds["processed_dir"], ds["splits_dir"]
        cfg["preprocess"] = dcfg["preprocess"]
        if (
            not (Path(ds["splits_dir"]) / "train.json").exists()
            or not (Path(ds["processed_dir"]) / "manifest.json").exists()
        ):
            raise SystemExit(
                "splits/preprocessed data missing: run scripts/create_splits.py and scripts/preprocess.py first"
            )
        cfg["label_info"] = {
            "label_map": (
                {
                    int(k): v
                    for k, v in json.loads(Path(f"data/manifests/{name}_verification.json").read_text())[
                        "label_map_to_contiguous"
                    ].items()
                }
                if Path(f"data/manifests/{name}_verification.json").exists()
                else None
            )
        }
    out = Path(cfg.get("output_dir", "outputs")) / "configs"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg['experiment']}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    rec = Trainer(cfg, resume=a.resume).fit()
    print(json.dumps({k: v for k, v in rec.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
