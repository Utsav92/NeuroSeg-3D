"""Generic configurable 3D encoder-decoder and the baseline 3D U-Net."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import AttentionGate, ConvBlock, Down, ResBlock, Up, run_block


class EncoderDecoder3D(nn.Module):
    """U-shaped network: `depth` encoder levels + bottleneck + `depth` decoder stages.

    block: 'plain' (ConvBlock) | 'res' (ResBlock); attention: gate encoder skips with decoder signal.
    Returns logits [B, out_channels, D, H, W]. Spatial dims must be divisible by 2**depth.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 16,
        depth: int = 4,
        block: str = "plain",
        attention: bool = False,
        norm: str = "instance",
        dropout: float = 0.0,
        down_mode: str = "maxpool",
        up_mode: str = "transpose",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gradient_checkpointing = gradient_checkpointing
        B = ResBlock if block == "res" else ConvBlock
        widths = [base_channels * 2**i for i in range(depth + 1)]
        self.widths = widths

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        c = in_channels
        for i in range(depth):
            self.enc.append(B(c, widths[i], norm, dropout=0.0))
            self.down.append(Down(down_mode, widths[i]))
            c = widths[i]
        self.bottleneck = B(widths[depth - 1], widths[depth], norm, dropout=dropout)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.gates = nn.ModuleList() if attention else None
        for i in reversed(range(depth)):
            self.up.append(Up(widths[i + 1], widths[i], up_mode))
            if attention:
                self.gates.append(AttentionGate(widths[i], widths[i], max(widths[i] // 2, 4), norm))
            self.dec.append(B(2 * widths[i], widths[i], norm, dropout=dropout))
        self.head = nn.Conv3d(widths[0], out_channels, 1)
        self._init_weights()
        self.last_attention: list[torch.Tensor] = []

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if any(s % (2**self.depth) for s in x.shape[2:]):
            raise ValueError(f"spatial dims {tuple(x.shape[2:])} must be divisible by {2**self.depth}")
        if self.training and min(x.shape[2:]) // (2**self.depth) < 2:
            raise ValueError(
                f"input {tuple(x.shape[2:])} gives a 1-voxel bottleneck at depth {self.depth}; normalisation is undefined in training. "
                f"Use spatial dims >= {2 ** (self.depth + 1)}."
            )
        ck = self.gradient_checkpointing
        skips = []
        for enc, down in zip(self.enc, self.down):
            x = run_block(enc, x, ck)
            skips.append(x)
            x = down(x)
        x = run_block(self.bottleneck, x, ck)
        atts = []
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            skip = skips[-(j + 1)]
            if self.gates is not None:
                skip, a = self.gates[j](skip, x)
                atts.append(a)
            x = run_block(dec, torch.cat([skip, x], dim=1), ck)
        out = self.head(x)
        if return_attention:
            return out, atts
        return out

    def set_dropout_mc(self, enabled: bool = True) -> None:
        """Enable only Dropout layers (for Monte-Carlo dropout) while the rest stays in eval mode."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.train(enabled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNet3D(EncoderDecoder3D):
    """Baseline 3D U-Net: double-conv blocks, max-pool down, transposed-conv up, concat skips."""

    def __init__(self, **kw):
        kw.setdefault("block", "plain")
        kw["attention"] = False
        super().__init__(**kw)
