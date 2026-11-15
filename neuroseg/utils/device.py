"""Device discovery, operation probing, memory monitoring and config recommendations."""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _probe(device: torch.device) -> tuple[bool, str]:
    """Run a real forward+backward through every 3D op the models use."""
    try:
        conv = nn.Conv3d(2, 4, 3, padding=1).to(device)
        norm_in = nn.InstanceNorm3d(4, affine=True).to(device)
        norm_gn = nn.GroupNorm(2, 4).to(device)
        up = nn.ConvTranspose3d(4, 4, 2, stride=2).to(device)
        x = torch.randn(1, 2, 8, 8, 8, device=device, requires_grad=True)
        h = F.leaky_relu(norm_in(conv(x)), 0.01)
        h = F.max_pool3d(h, 2)
        h = norm_gn(up(h))
        h = F.interpolate(h, scale_factor=1.0, mode="trilinear", align_corners=False)
        h = F.dropout3d(h, 0.1, training=True)
        h.mean().backward()
        _sync(device)
        if not torch.isfinite(x.grad).all():
            return False, "non-finite gradient"
        return True, "ok"
    except Exception as e:  # noqa: BLE001 - report any backend failure
        return False, f"{type(e).__name__}: {e}"


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def select_device(preference: str = "auto", verify: bool = True) -> torch.device:
    """Pick cuda > mps > cpu. A backend is only used if a real fwd+bwd probe passes."""
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    candidates = []
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        candidates.append("cuda")
    if pref in ("auto", "mps") and mps_available():
        candidates.append("mps")
    for name in candidates:
        dev = torch.device(name)
        if not verify:
            return dev
        ok, msg = _probe(dev)
        if ok:
            return dev
        print(f"[device] {name} probe failed ({msg}); falling back")
    return torch.device("cpu")


def device_report() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "cuda_available": torch.cuda.is_available(),
        "mps_built": bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_built(),
        "mps_available": mps_available(),
    }
    probes = {}
    for name in (
        ["cpu"] + (["mps"] if mps_available() else []) + (["cuda"] if torch.cuda.is_available() else [])
    ):
        ok, msg = _probe(torch.device(name))
        probes[name] = {"fwd_bwd_3d_ops": ok, "detail": msg}
    info["probes"] = probes
    return info


def memory_usage(device: torch.device | None = None) -> dict:
    out = {"rss_gb": round(psutil.Process().memory_info().rss / 2**30, 3)}
    if device is not None:
        if device.type == "mps":
            try:
                out["mps_allocated_gb"] = round(torch.mps.current_allocated_memory() / 2**30, 3)
                out["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 2**30, 3)
            except Exception:  # noqa: BLE001
                pass
        elif device.type == "cuda":
            out["cuda_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
            out["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return out


def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


@dataclass
class Recommendation:
    patch_size: int
    base_channels: int
    batch_size: int
    grad_accum: int
    num_workers: int
    gradient_checkpointing: bool
    amp: bool
    notes: list[str] = field(default_factory=list)


def recommend_config(device: torch.device) -> Recommendation:
    """Heuristic starting point; confirm with scripts/benchmark_device.py (which measures)."""
    ram = psutil.virtual_memory().total / 2**30
    notes = []
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        patch, base, bs = (96, 32, 2) if vram >= 24 else (96, 16, 1) if vram >= 12 else (64, 16, 1)
        return Recommendation(patch, base, bs, 2 if bs == 1 else 1, 4, False, True, [f"CUDA {vram:.0f} GB"])
    if device.type == "mps":
        notes.append("fp16 autocast on MPS is not enabled by default (stability); checkpointing on")
        if ram >= 24:
            return Recommendation(64, 16, 1, 4, 2, True, False, notes)
        return Recommendation(48, 16, 1, 4, 1, True, False, notes + ["<24 GB RAM: smaller patch"])
    notes.append("CPU: use small patch for practical speed")
    return Recommendation(48 if ram >= 16 else 32, 8, 1, 4, 2, True, False, notes)


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0
