"""Atomic checkpoint save/load. Checkpoints hold only tensors and plain Python types (weights_only-safe)."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written checkpoint


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=True)


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept a file, or an experiment directory (prefers best.pt, falls back to last.pt)."""
    p = Path(path)
    if p.is_dir():
        for n in ("best.pt", "last.pt"):
            if (p / n).exists():
                return p / n
        raise FileNotFoundError(f"no best.pt/last.pt in {p}")
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return p
