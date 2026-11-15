#!/usr/bin/env python
"""Controlled experiments 1-6. Generates one fully-resolved config per variant (same seed, splits, budget),
optionally runs them, and maintains outputs/experiments/registry.json + the table in docs/EXPERIMENTS.md.

  python scripts/run_experiments.py --experiment arch_comparison --hardware configs/train_mac.yaml          # plan only
  python scripts/run_experiments.py --experiment arch_comparison --hardware configs/train_mac.yaml --run    # train all variants
  python scripts/run_experiments.py --report                                                                # refresh the docs table
Nothing is reported as a result unless a run actually completed and recorded it.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml
from _bootstrap import ROOT

from neuroseg.utils.logging import compose_config, deep_merge

BASE = "configs/model_resunet.yaml"
EXPERIMENTS = {
    "arch_comparison": {  # Exp 1: equal conditions
        "unet3d": {"_base": "configs/model_unet.yaml"},
        "resunet3d": {"_base": "configs/model_resunet.yaml", "model": {"dropout": 0.0}},
        "attention_unet3d": {"_base": "configs/model_attention.yaml"},
    },
    "modality_ablation": {  # Exp 2: separate models trained per input set
        "flair_only": {"data": {"modalities": ["flair"]}},
        "t1ce_only": {"data": {"modalities": ["t1ce"]}},
        "t1_t1ce": {"data": {"modalities": ["t1", "t1ce"]}},
        "all_four": {"data": {"modalities": ["flair", "t1", "t1ce", "t2"]}},
    },
    "loss_comparison": {  # Exp 3
        "dice": {"loss": {"components": {"dice": 1.0}}},
        "dice_ce": {"loss": {"components": {"dice": 0.5, "ce": 0.5}}},
        "tversky": {
            "loss": {"components": {"tversky": 0.5, "ce": 0.5}, "tversky": {"alpha": 0.3, "beta": 0.7}}
        },
    },
    "patch_size": {  # Exp 4 (96^3 needs ample memory; check scripts/benchmark_device.py first)
        "p48": {"data": {"patch_size": [48, 48, 48]}},
        "p64": {"data": {"patch_size": [64, 64, 64]}},
        "p96": {"data": {"patch_size": [96, 96, 96]}},
    },
    "augmentation": {  # Exp 5
        "basic": {"data": {"augmentation": "basic"}},
        "full": {"data": {"augmentation": "full"}},
    },
}
CALIBRATION_NOTE = "Experiment 6 (calibration/uncertainty) is evaluation-only: run scripts/evaluate.py --split validation --mc-samples 10 (and --ensemble ...) on trained checkpoints."


def registry_path():
    return Path("outputs/experiments/registry.json")


def load_registry():
    p = registry_path()
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=list(EXPERIMENTS))
    ap.add_argument("--hardware", default=str(ROOT / "configs/train_mac.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    reg = load_registry()
    if a.experiment:
        for variant, over in EXPERIMENTS[a.experiment].items():
            over = dict(over)
            base = over.pop("_base", BASE)
            name = f"{a.experiment}__{variant}"
            cfg = compose_config(base, a.hardware)
            cfg = deep_merge(cfg, over)
            cfg["experiment"], cfg["seed"] = name, a.seed
            if a.epochs:
                cfg["train"]["epochs"] = a.epochs
            if a.experiment == "arch_comparison":
                cfg["model"]["dropout"] = 0.0  # identical regularisation for a fair comparison
            path = Path("configs/experiments") / a.experiment / f"{variant}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            entry = reg.get(name, {"experiment": a.experiment, "variant": variant})
            entry.update(config=str(path), seed=a.seed, status=entry.get("status", "planned"))
            reg[name] = entry
            print(f"[plan] {name}: python scripts/train.py --config {path}")
    if a.run and a.experiment:
        import subprocess
        import sys

        for variant in EXPERIMENTS[a.experiment]:
            name = f"{a.experiment}__{variant}"
            cfg_path = reg[name]["config"]
            r = subprocess.run([sys.executable, "scripts/train.py", "--config", cfg_path, "--resume"])
            rec_path = Path("outputs/checkpoints") / name / "run_record.json"
            if r.returncode == 0 and rec_path.exists():
                rec = json.loads(rec_path.read_text())
                reg[name].update(
                    status="completed",
                    best_val_mean_dice=rec["best_val_mean_dice"],
                    epochs_run=rec["epochs_run"],
                    checkpoint=rec["checkpoint_best"],
                    wall_seconds=rec["wall_seconds"],
                    stop_reason=rec["stop_reason"],
                )
            else:
                reg[name]["status"] = "failed"
            registry_path().parent.mkdir(parents=True, exist_ok=True)
            registry_path().write_text(json.dumps(reg, indent=2))
    registry_path().parent.mkdir(parents=True, exist_ok=True)
    registry_path().write_text(json.dumps(reg, indent=2))
    if a.report or a.run:
        write_table(reg)


def write_table(reg):
    rows = [
        "| Experiment | Variant | Seed | Status | Best val mean Dice (validation split) | Epochs | Checkpoint |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, e in sorted(reg.items()):
        done = e.get("status") == "completed"
        rows.append(
            f"| {e['experiment']} | {e['variant']} | {e['seed']} | {e['status']} | {e['best_val_mean_dice']:.4f} | {e['epochs_run']} | `{e['checkpoint']}` |"
            if done
            else f"| {e['experiment']} | {e['variant']} | {e['seed']} | {e['status']} | Not evaluated | - | - |"
        )
    p = Path("docs/EXPERIMENTS.md")
    txt = p.read_text()
    a, b = "<!-- TABLE:START -->", "<!-- TABLE:END -->"
    txt = txt.split(a)[0] + a + "\n" + "\n".join(rows) + "\n" + b + txt.split(b)[1]
    p.write_text(txt)
    print("updated docs/EXPERIMENTS.md")


if __name__ == "__main__":
    main()
