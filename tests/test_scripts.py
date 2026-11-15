"""Runs the real CLI scripts (subprocess) on a synthetic dataset: verify -> inspect -> splits -> preprocess -> train -> evaluate -> predict."""

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(args, cwd):
    r = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{args}\nSTDOUT:\n{r.stdout[-2500:]}\nSTDERR:\n{r.stderr[-2500:]}"
    return r.stdout


@pytest.fixture(scope="module")
def workdir(tmp_path_factory, synth_root):
    w = tmp_path_factory.mktemp("cli")
    cfg = {
        "default_dataset": "synth",
        "preprocess": {"clip_percentiles": [0.5, 99.5], "crop_margin": 2, "target_spacing": None},
        "split": {"fractions": [0.7, 0.15, 0.15], "seed": 42},
        "datasets": {
            "synth": {
                "adapter": "msd_brain",
                "root": str(synth_root),
                "processed_dir": str(w / "proc"),
                "splits_dir": str(w / "splits"),
                "expected": {"n_training": 10, "modalities": ["flair", "t1", "t1ce", "t2"]},
            }
        },
    }
    (w / "datasets.yaml").write_text(yaml.safe_dump(cfg))
    model = {
        "experiment": "cli_test",
        "seed": 1,
        "output_dir": str(w / "outputs"),
        "data": {
            "dataset": "synth",
            "modalities": ["flair", "t1ce"],
            "patch_size": [32, 32, 32],
            "samples_per_epoch": 4,
            "augmentation": "basic",
        },
        "model": {"arch": "unet3d", "base_channels": 4, "depth": 4},
        "loss": {"components": {"dice": 0.5, "ce": 0.5}, "ce_class_weights": True},
        "optim": {"name": "adamw", "lr": 0.002},
        "scheduler": {"name": "plateau", "patience": 1},
        "train": {
            "epochs": 2,
            "device": "cpu",
            "batch_size": 1,
            "grad_accum": 2,
            "val_interval": 1,
            "val_max_cases": 2,
            "tensorboard": False,
        },
    }
    (w / "model.yaml").write_text(yaml.safe_dump(model))
    return w


def test_cli_pipeline(workdir, synth_root):
    ds = str(workdir / "datasets.yaml")
    out = run([str(ROOT / "scripts/verify_dataset.py"), "--config", ds], workdir)
    assert "VERIFICATION PASSED" in out
    assert (workdir / "data/manifests/synth_verification.json").exists()
    run([str(ROOT / "scripts/inspect_dataset.py"), "--config", ds, "--n", "5"], workdir)
    out = run([str(ROOT / "scripts/create_splits.py"), "--config", ds, "--seed", "42"], workdir)
    assert "leakage checks passed" in out
    for k in ("train", "validation", "test"):
        assert (workdir / "splits" / f"{k}.json").exists()
    run([str(ROOT / "scripts/preprocess.py"), "--config", ds, "--workers", "2"], workdir)
    assert (workdir / "proc/manifest.json").exists()
    out = run(
        [str(ROOT / "scripts/train.py"), "--config", str(workdir / "model.yaml"), "--datasets", ds], workdir
    )
    rec = json.loads((workdir / "outputs/checkpoints/cli_test/run_record.json").read_text())
    assert rec["epochs_run"] == 2 and rec["test_set_used"] is False
    ck = workdir / "outputs/checkpoints/cli_test"
    # test split stays sealed unless explicitly confirmed
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate.py"),
            "--checkpoint",
            str(ck),
            "--split",
            "test",
            "--datasets",
            ds,
            "--device",
            "cpu",
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "sealed" in (r.stderr + r.stdout)
    run(
        [
            str(ROOT / "scripts/evaluate.py"),
            "--checkpoint",
            str(ck),
            "--split",
            "validation",
            "--datasets",
            ds,
            "--device",
            "cpu",
            "--out",
            str(workdir / "ev"),
        ],
        workdir,
    )
    assert (workdir / "ev/validation_summary.csv").exists()
    # inference on a 2-modality model from a directory of NIfTI
    img = nib.load(str(synth_root / "imagesTr/SYNTH_000.nii.gz"))
    arr = np.asarray(img.dataobj)
    inp = workdir / "sample"
    inp.mkdir()
    for c, name in [(0, "flair"), (2, "t1ce")]:
        nib.save(nib.Nifti1Image(arr[..., c], img.affine), str(inp / f"case_{name}.nii.gz"))
    out = run(
        [
            str(ROOT / "scripts/predict.py"),
            "--checkpoint",
            str(ck / "best.pt"),
            "--input-dir",
            str(inp),
            "--output-dir",
            str(workdir / "pred"),
            "--device",
            "cpu",
        ],
        workdir,
    )
    assert (workdir / "pred/sample_seg.nii.gz").exists() and (workdir / "pred/sample_overview.png").exists()
    assert nib.load(str(workdir / "pred/sample_seg.nii.gz")).shape == img.shape[:3]


def test_predict_without_checkpoint_fails_clearly(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/predict.py"),
            "--checkpoint",
            str(tmp_path / "none.pt"),
            "--input-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "never returns synthetic" in (r.stderr + r.stdout)


def test_manual_access_dataset_refuses_download():
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/download_dataset.py"),
            "--dataset",
            "brats2023_glioma",
            "--config",
            str(ROOT / "configs/datasets.yaml"),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "registration" in (r.stderr + r.stdout)
