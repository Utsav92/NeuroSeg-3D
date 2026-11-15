"""3D Residual U-Net (primary research model)."""

from __future__ import annotations

from .unet3d import EncoderDecoder3D


class ResUNet3D(EncoderDecoder3D):
    """Residual blocks in encoder, bottleneck and decoder; long concat skips; optional dropout/checkpointing."""

    def __init__(self, **kw):
        kw["block"] = "res"
        kw["attention"] = False
        super().__init__(**kw)
