"""Building blocks for the 3D segmentation networks (plain PyTorch primitives only)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def make_norm(kind: str, channels: int) -> nn.Module:
    kind = kind.lower()
    if kind == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if kind == "group":
        groups = 8 if channels % 8 == 0 else (4 if channels % 4 == 0 else (2 if channels % 2 == 0 else 1))
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unknown norm '{kind}'")


def make_act(slope: float = 0.01) -> nn.Module:
    return nn.LeakyReLU(slope, inplace=False)


class ConvBlock(nn.Module):
    """(Conv3d -> Norm -> LeakyReLU) x 2"""

    def __init__(self, cin: int, cout: int, norm: str = "instance", dropout: float = 0.0):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1, bias=False)
        self.n1 = make_norm(norm, cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.n2 = make_norm(norm, cout)
        self.act = make_act()
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.n1(self.c1(x)))
        x = self.act(self.n2(self.c2(x)))
        return self.drop(x)


class ResBlock(nn.Module):
    """Conv-Norm-Act -> Conv-Norm, plus (projected) identity shortcut, then Act."""

    def __init__(self, cin: int, cout: int, norm: str = "instance", dropout: float = 0.0):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1, bias=False)
        self.n1 = make_norm(norm, cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.n2 = make_norm(norm, cout)
        self.act = make_act()
        self.proj = (
            nn.Identity()
            if cin == cout
            else nn.Sequential(nn.Conv3d(cin, cout, 1, bias=False), make_norm(norm, cout))
        )
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.drop(self.act(h + self.proj(x)))


class AttentionGate(nn.Module):
    """Additive spatial attention gate (Oktay et al. 2018) applied to encoder skip features.

    skip x (Cx) is modulated by gating signal g (Cg, same spatial size after upsampling):
        a = sigmoid(psi(relu(Wx x + Wg g)));  out = a * x
    """

    def __init__(self, c_skip: int, c_gate: int, c_inter: int, norm: str = "instance"):
        super().__init__()
        self.wx = nn.Conv3d(c_skip, c_inter, 1, bias=False)
        self.wg = nn.Conv3d(c_gate, c_inter, 1, bias=True)
        self.psi = nn.Conv3d(c_inter, 1, 1, bias=True)
        self.norm = make_norm(norm, c_inter)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = F.relu(self.norm(self.wx(x) + self.wg(g)))
        a = torch.sigmoid(self.psi(a))
        return x * a, a


class Down(nn.Module):
    def __init__(self, mode: str = "maxpool", channels: int = 0):
        super().__init__()
        self.op: nn.Module
        if mode == "maxpool":
            self.op = nn.MaxPool3d(2)
        elif mode == "strided":
            self.op = nn.Conv3d(channels, channels, 2, stride=2)
        else:
            raise ValueError(mode)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, cin: int, cout: int, mode: str = "transpose"):
        super().__init__()
        self.op: nn.Module
        if mode == "transpose":
            self.op = nn.ConvTranspose3d(cin, cout, 2, stride=2)
        elif mode == "interp":
            self.op = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False), nn.Conv3d(cin, cout, 1)
            )
        else:
            raise ValueError(mode)

    def forward(self, x):
        return self.op(x)


def run_block(block: nn.Module, x: torch.Tensor, use_ckpt: bool) -> torch.Tensor:
    """Run a block with optional activation checkpointing (only active while training with grad)."""
    if use_ckpt and torch.is_grad_enabled() and x.requires_grad:
        return checkpoint(block, x, use_reentrant=False)
    return block(x)
