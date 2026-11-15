"""3D Attention U-Net with additive attention gates on the skip connections."""

from __future__ import annotations

from .unet3d import EncoderDecoder3D


class AttentionUNet3D(EncoderDecoder3D):
    def __init__(self, **kw):
        kw.setdefault("block", "plain")
        kw["attention"] = True
        super().__init__(**kw)
