"""Split creation + dataset preprocessing (shared by scripts and the synthetic smoke test)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from neuroseg.data import validation as V
from neuroseg.data.preprocessing import preprocess_case, save_processed


def make_splits(records, seed: int = 42, fractions=(0.7, 0.15, 0.15), strata: dict | None = None) -> dict:
    """Deterministic patient-level split. All cases of a patient share one partition.

    strata: optional {patient_id: group} from metadata that actually exists; splitting is then done within groups.
    """
    if abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("fractions must sum to 1")
    labelled = [r for r in records if r.label_path]
    patients = sorted({r.patient_id for r in labelled})
    rng = np.random.default_rng(seed)
    groups: dict = {}
    for p in patients:
        groups.setdefault((strata or {}).get(p, "_all"), []).append(p)
    parts: dict[str, list] = {"train": [], "validation": [], "test": []}
    for g in sorted(groups, key=str):
        ps = list(groups[g])
        rng.shuffle(ps)
        n = len(ps)
        n_tr, n_va = int(round(fractions[0] * n)), int(round(fractions[1] * n))
        parts["train"] += ps[:n_tr]
        parts["validation"] += ps[n_tr : n_tr + n_va]
        parts["test"] += ps[n_tr + n_va :]
    V.assert_no_leakage(parts)
    by_pat: dict = {}
    for r in labelled:
        by_pat.setdefault(r.patient_id, []).append(r.case_id)
    cases = {k: sorted(c for p in v for c in by_pat[p]) for k, v in parts.items()}
    c2p = {r.case_id: r.patient_id for r in labelled}
    V.assert_case_patient_consistency(c2p, cases)
    out = {}
    for k in parts:
        out[k] = {"patients": sorted(parts[k]), "cases": cases[k]}
    out["_info"] = {
        "seed": seed,
        "fractions": list(fractions),
        "n_patients": len(patients),
        "n_cases": len(labelled),
        "n_unlabelled_excluded": len(records) - len(labelled),
        "report": V.split_report({k: cases[k] for k in parts}),
        "stratified_by": "metadata" if strata else None,
    }
    return out


def save_splits(splits: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("train", "validation", "test"):
        (out / f"{k}.json").write_text(json.dumps({**splits[k], "split": k, **splits["_info"]}, indent=1))


def load_splits(split_dir: str | Path) -> dict:
    d = Path(split_dir)
    s = {k: json.loads((d / f"{k}.json").read_text()) for k in ("train", "validation", "test")}
    V.assert_no_leakage({k: v["patients"] for k, v in s.items()})  # re-verified on every load
    return s


def splits_hash(splits: dict) -> str:
    h = hashlib.sha256()
    for k in ("train", "validation", "test"):
        h.update(("|".join(splits[k]["cases"]) + k).encode())
    return h.hexdigest()[:16]


def _process_one(args):
    """Worker: preprocess one case and return its meta dict (top-level so it is picklable)."""
    adapter, rec, d, pre_cfg, overwrite = args
    d = Path(d)
    if (d / "meta.json").exists() and not overwrite:
        return json.loads((d / "meta.json").read_text())
    img, aff, lab, _hdr = adapter.load(rec)
    problems = V.check_nifti_geometry(
        [Path(p) for p in rec.image_paths],
        Path(rec.label_path) if rec.label_path else None,
        expect_channels=len(rec.modalities) if len(rec.image_paths) == 1 else None,
    )
    if problems:
        raise V.DatasetError(f"{rec.case_id}: geometry problems: {problems}")
    out = preprocess_case(
        img, aff, lab, adapter.label_map, list(rec.modalities),
        tuple(pre_cfg.get("clip_percentiles", (0.5, 99.5))), int(pre_cfg.get("crop_margin", 2)),
        tuple(pre_cfg["target_spacing"]) if pre_cfg.get("target_spacing") else None,
    )  # fmt: skip
    save_processed(d, out, rec.case_id, {"patient_id": rec.patient_id})
    return json.loads((d / "meta.json").read_text())


def preprocess_dataset(
    adapter, out_dir: str | Path, records, pre_cfg: dict, overwrite: bool = False, log=print, workers: int = 1
) -> dict:
    """Preprocess every record (resumable: existing cases are skipped unless overwrite). workers>1 uses processes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(adapter, r, out_dir / r.case_id, pre_cfg, overwrite) for r in records]
    metas = []
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers) as ex:
            for i, m in enumerate(ex.map(_process_one, jobs, chunksize=2)):
                metas.append(m)
                if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                    log(f"preprocessed {i + 1}/{len(jobs)}")
    else:
        for i, j in enumerate(jobs):
            metas.append(_process_one(j))
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                log(f"preprocessed {i + 1}/{len(jobs)}")
    hashes: dict = {}
    manifest = []
    for rec, m in zip(records, metas):
        hashes.setdefault(m["hash"], []).append(rec.case_id)
        manifest.append(
            {
                "case_id": rec.case_id,
                "patient_id": rec.patient_id,
                "dir": str(out_dir / rec.case_id),
                "spacing_mm": m["spacing_mm"],
                "label_counts": m.get("label_counts"),
            }
        )
    dups = {h: c for h, c in hashes.items() if len(c) > 1}
    info = {"n_cases": len(manifest), "duplicate_content_groups": list(dups.values()), "cases": manifest}
    (out_dir / "manifest.json").write_text(json.dumps(info, indent=1))
    return info


def resolve_dataset(datasets_cfg: dict, name: str | None = None) -> tuple[str, dict]:
    name = str(name or datasets_cfg.get("default_dataset"))
    if name not in datasets_cfg["datasets"]:
        raise KeyError(f"dataset '{name}' not in configs/datasets.yaml ({list(datasets_cfg['datasets'])})")
    ds = datasets_cfg["datasets"][name]
    if ds.get("access") == "manual" and not Path(ds.get("root", "")).exists():
        raise FileNotFoundError(
            f"dataset '{name}' requires manual, access-controlled download (see docs/DATASETS.md); expected data at {ds.get('root')}"
        )
    return name, ds
