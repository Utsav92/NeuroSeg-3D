"""Patient-level statistics. The unit of resampling is the patient, never the voxel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import METRIC_NAMES, REGIONS


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, stat=np.nanmean) -> dict:
    """Percentile bootstrap CI over patients. NaNs (undefined per-patient values) are dropped first."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = stat(v[idx], axis=1)
    return {
        "mean": float(stat(v)),
        "lo": float(np.percentile(boots, 100 * alpha / 2)),
        "hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "n": int(v.size),
    }


def paired_bootstrap_diff(a, b, n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean paired difference (a-b) over the same patients with bootstrap CI; resamples patients jointly."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    if d.size == 0:
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(1)
    return {
        "mean_diff": float(d.mean()),
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n": int(d.size),
    }


def results_to_frame(per_patient: dict[str, dict]) -> pd.DataFrame:
    """{patient: {region: {metric: v}}} -> long DataFrame (patient, region, metric columns...)."""
    rows = []
    for pid, regs in per_patient.items():
        for r, m in regs.items():
            rows.append({"patient": pid, "region": r, **m})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, metrics=None, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Aggregate per region with patient-level bootstrap CIs; also reports number of patients with undefined values."""
    metrics = metrics or METRIC_NAMES + [
        "pred_volume_ml",
        "gt_volume_ml",
        "abs_volume_error_ml",
        "rel_volume_error",
    ]
    rows = []
    for r in REGIONS:
        sub = df[df.region == r]
        for m in metrics:
            if m not in sub:
                continue
            ci = bootstrap_ci(sub[m].values, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "region": r,
                    "metric": m,
                    "mean": ci["mean"],
                    "ci_lo": ci["lo"],
                    "ci_hi": ci["hi"],
                    "median": float(np.nanmedian(sub[m].values)) if sub[m].notna().any() else float("nan"),
                    "n_defined": ci["n"],
                    "n_patients": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def subgroup_summary(
    df: pd.DataFrame, meta: pd.DataFrame, column: str, min_group: int = 20, metric: str = "dice"
) -> pd.DataFrame:
    """Per-subgroup means, only for groups with >= min_group patients (smaller groups are reported as 'insufficient')."""
    merged = df.merge(meta[["patient", column]], on="patient", how="left")
    rows = []
    for (g, r), sub in merged.groupby([column, "region"]):
        n = sub.patient.nunique()
        if n < min_group:
            rows.append(
                {
                    column: g,
                    "region": r,
                    "n_patients": n,
                    "mean": float("nan"),
                    "note": f"insufficient (<{min_group})",
                }
            )
        else:
            ci = bootstrap_ci(sub[metric].values)
            rows.append(
                {
                    column: g,
                    "region": r,
                    "n_patients": n,
                    "mean": ci["mean"],
                    "ci_lo": ci["lo"],
                    "ci_hi": ci["hi"],
                    "note": "",
                }
            )
    return pd.DataFrame(rows)
