from __future__ import annotations

from .attention_unet3d import AttentionUNet3D
from .resunet3d import ResUNet3D
from .unet3d import EncoderDecoder3D, UNet3D

ARCHS = {"unet3d": UNet3D, "resunet3d": ResUNet3D, "attention_unet3d": AttentionUNet3D}


def build_model(cfg: dict):
    """Instantiate from the `model:` config section (random init; no pretrained weights exist or are loaded)."""
    cfg = dict(cfg)
    arch = cfg.pop("arch")
    if arch not in ARCHS:
        raise ValueError(f"unknown arch '{arch}', choose from {list(ARCHS)}")
    return ARCHS[arch](**cfg)


__all__ = ["build_model", "ARCHS", "UNet3D", "ResUNet3D", "AttentionUNet3D", "EncoderDecoder3D"]
