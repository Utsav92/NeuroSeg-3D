"""Logging, YAML config composition, CSV + TensorBoard metric sinks."""

from __future__ import annotations

import copy
import csv
import json
import logging
import sys
from pathlib import Path

import yaml


def get_logger(name: str = "neuroseg", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def compose_config(*paths: str | Path | None, overrides: list[str] | None = None) -> dict:
    """Merge YAML files left-to-right (later wins), then apply `a.b.c=value` overrides."""
    cfg: dict = {}
    for p in paths:
        if p:
            cfg = deep_merge(cfg, load_yaml(p))
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        val = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return cfg


class MetricLogger:
    """Append-only CSV (schema may grow) + optional TensorBoard."""

    def __init__(self, out_dir: str | Path, tensorboard: bool = True):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "metrics.csv"
        self.rows: list[dict] = []
        if self.csv_path.exists():
            with open(self.csv_path) as f:
                self.rows = list(csv.DictReader(f))
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(self.dir / "tb"))
            except Exception:  # noqa: BLE001
                self.tb = None

    def log(self, step: int, metrics: dict) -> None:
        row = {"epoch": step, **{k: v for k, v in metrics.items()}}
        self.rows.append(row)
        keys: list[str] = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        if self.tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
