#!/usr/bin/env python
"""Measure (not assume) what this machine can do: real 3D forward+backward on CPU / MPS / CUDA."""

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from neuroseg.models import build_model
from neuroseg.training.losses import build_loss
from neuroseg.utils.device import device_report, empty_cache, memory_usage, recommend_config, select_device


def sync(d):
    if d.type == "mps":
        torch.mps.synchronize()
    elif d.type == "cuda":
        torch.cuda.synchronize()


def bench(arch, device, patch, base, ckpt, steps=3, bs=1):
    m = build_model(
        {
            "arch": arch,
            "in_channels": 4,
            "out_channels": 4,
            "base_channels": base,
            "depth": 4,
            "gradient_checkpointing": ckpt,
        }
    ).to(device)
    n_params = m.count_parameters()
    loss_fn = build_loss({"components": {"dice": 0.5, "ce": 0.5}}).to(device)
    opt = torch.optim.AdamW(m.parameters(), 2e-4)
    x = torch.randn(bs, 4, patch, patch, patch, device=device)
    y = torch.randint(0, 4, (bs, patch, patch, patch), device=device)
    res = {
        "arch": arch,
        "device": str(device),
        "patch": patch,
        "base": base,
        "grad_ckpt": ckpt,
        "batch": bs,
        "params": n_params,
        "param_mb": round(n_params * 4 / 2**20, 2),
    }
    try:
        m.eval()
        with torch.no_grad():
            m(x)
            sync(device)
            t = time.perf_counter()
            for _ in range(steps):
                m(x)
            sync(device)
            res["infer_s_per_patch"] = round((time.perf_counter() - t) / steps, 4)
        m.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for i in range(steps + 1):  # first iteration = warm-up
            if i == 1:
                sync(device)
                t = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(m(x), y)
            loss.backward()
            opt.step()
        sync(device)
        res["train_step_s"] = round((time.perf_counter() - t) / steps, 4)
        res["loss_finite"] = bool(torch.isfinite(loss))
        res["grads_finite"] = all(
            torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None
        )
        res["memory"] = memory_usage(device)
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    del m, opt
    empty_cache(device)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=["resunet3d"])
    ap.add_argument("--quick", action="store_true", help="only 64^3/base16 and 48^3/base16")
    ap.add_argument("--devices", nargs="*", default=None)
    a = ap.parse_args()
    rep = device_report()
    print(json.dumps(rep, indent=2))
    devs = a.devices or [d for d in ("cpu", "mps", "cuda") if d in rep["probes"]]
    grid = (
        [(64, 16, True), (64, 16, False), (48, 16, True)]
        if a.quick
        else [(64, 16, True), (64, 16, False), (48, 16, True), (48, 8, True), (96, 16, True)]
    )
    results = []
    for d in devs:
        if not rep["probes"][d]["fwd_bwd_3d_ops"]:
            results.append(
                {"device": d, "ok": False, "error": "3D op probe failed: " + rep["probes"][d]["detail"]}
            )
            continue
        for arch in a.archs:
            for patch, base, ck in grid:
                r = bench(arch, torch.device(d), patch, base, ck)
                results.append(r)
                print(json.dumps(r))
    sel = select_device("auto")
    rec = recommend_config(sel)
    out = {
        "report": rep,
        "selected_device": str(sel),
        "heuristic_recommendation": rec.__dict__,
        "results": results,
    }
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics/benchmark_device.json").write_text(json.dumps(out, indent=2))
    print(
        "selected device:",
        sel,
        "| heuristic recommendation (verify against measured rows above):",
        rec.__dict__,
    )


if __name__ == "__main__":
    main()
