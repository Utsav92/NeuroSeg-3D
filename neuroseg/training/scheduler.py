from __future__ import annotations

import torch


def build_scheduler(optimizer, cfg: dict, epochs: int):
    """'cosine' steps once per epoch; 'plateau' steps on the validation metric (mode=max); 'none'."""
    name = cfg.get("name", "cosine").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=float(cfg.get("min_lr", 1e-6))
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 5)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
        )
    if name == "none":
        return None
    raise ValueError(f"unknown scheduler '{name}'")
