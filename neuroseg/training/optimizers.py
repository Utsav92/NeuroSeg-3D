from __future__ import annotations

import torch


def build_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("name", "adamw").lower()
    lr = float(cfg.get("lr", 2e-4))
    wd = float(cfg.get("weight_decay", 1e-5))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=tuple(cfg.get("betas", (0.9, 0.999))))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=float(cfg.get("momentum", 0.99)), nesterov=True, weight_decay=wd
        )
    raise ValueError(f"unknown optimizer '{name}'")
