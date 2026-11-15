import json

import nibabel as nib
import numpy as np
import pytest
import torch
from conftest import TEST_AFFINE

from neuroseg.evaluation.evaluator import evaluate_split
from neuroseg.inference import sliding_window as SW
from neuroseg.inference.predictor import (
    InputError,
    NoCheckpointError,
    Predictor,
    save_result,
)
from neuroseg.inference.uncertainty import entropy, predict_with_uncertainty
from neuroseg.training.checkpointing import load_checkpoint


class Identity(torch.nn.Module):
    """Returns the first K input channels as 'logits' so aggregation can be verified exactly."""

    def forward(self, x):
        return x[:, :4]


def test_sliding_window_exact_reconstruction():
    torch.manual_seed(0)
    x = torch.randn(4, 50, 37, 45)
    for gaussian in (True, False):
        out = SW.sliding_window_predict(
            Identity(), x, (16, 16, 16), 0.5, 3, "cpu", gaussian=gaussian, output="logits"
        )
        assert out.shape == x.shape and torch.allclose(
            out, x, atol=1e-5
        )  # overlap-weighted average of identical values


def test_sliding_window_small_volume_padding_and_tta():
    x = torch.randn(4, 10, 12, 9)
    out = SW.sliding_window_predict(Identity(), x, (16, 16, 16), 0.25, 1, "cpu", output="logits", tta=True)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-5)  # flip-TTA of identity returns identity


def test_tiles_cover_volume_and_weights_positive():
    for size in (16, 17, 50, 100):
        st = SW.tile_starts(size, 16, 0.5)
        assert st[0] == 0 and st[-1] + 16 >= min(size, st[-1] + 16) and (size <= 16 or st[-1] == size - 16)
    w = SW.gaussian_importance((16, 16, 16))
    assert w.min() > 0 and w[8, 8, 8] == w.max() and w[0, 0, 0] < w[8, 8, 8]


def test_probabilities_sum_to_one(trained):
    p = Predictor(trained["ckpt"], "cpu")
    x = torch.randn(4, 36, 40, 33)
    probs = SW.sliding_window_predict(p.models[0], x, p.patch, 0.5, 2, "cpu")
    assert torch.allclose(probs.sum(0), torch.ones(36, 40, 33), atol=1e-4)


def test_smoke_training_ran_for_real(trained):
    rec = trained["record"]
    assert rec["epochs_run"] == 3 and rec["test_set_used"] is False
    assert (trained["out"] / "checkpoints" / "pytest_resunet" / "last.pt").exists()
    hist = load_checkpoint(trained["ckpt"])["history"]
    assert all(np.isfinite(h["train_loss"]) for h in hist) and all(0 <= h["val_mean_dice"] <= 1 for h in hist)
    import pandas as pd

    df = pd.read_csv(trained["out"] / "checkpoints" / "pytest_resunet" / "metrics.csv")
    for c in (
        "train_loss",
        "val_loss",
        "val_dice_WT",
        "val_dice_TC",
        "val_dice_ET",
        "lr",
        "epoch_seconds",
        "n_params",
        "rss_gb",
    ):
        assert c in df.columns


def test_checkpoint_roundtrip_and_resume(trained):
    from neuroseg.training.trainer import Trainer

    ck = load_checkpoint(trained["ckpt"])
    p = Predictor(trained["ckpt"], "cpu")
    for k, v in ck["model"].items():
        assert torch.equal(p.models[0].state_dict()[k].cpu(), v)
    cfg = json.loads(json.dumps(trained["cfg"]))
    cfg["train"]["epochs"] = 4
    t = Trainer(cfg, resume=True)
    assert t.start_epoch == 3
    rec = t.fit()
    assert rec["epochs_run"] == 4


def test_missing_checkpoint_gives_clear_error(tmp_path):
    with pytest.raises(NoCheckpointError, match="never returns synthetic"):
        Predictor(tmp_path / "nope.pt", "cpu")


def _write_case(tmp_path, synth_root, idx=0, affine=None):
    img = nib.load(str(synth_root / "imagesTr" / f"SYNTH_{idx:03d}.nii.gz"))
    arr = np.asarray(img.dataobj)
    d = tmp_path / "case"
    d.mkdir(exist_ok=True)
    for c, name in enumerate(["flair", "t1", "t1ce", "t2"]):
        nib.save(
            nib.Nifti1Image(arr[..., c], img.affine if affine is None else affine),
            str(d / f"sub1_{name}.nii.gz"),
        )
    return d, img


def test_end_to_end_predict_geometry_and_export(trained, synth_root, tmp_path):
    d, img = _write_case(tmp_path, synth_root)
    p = Predictor(trained["ckpt"], "cpu")
    res = p.predict_dir(d, overlap=0.5, mc_samples=3, return_probs=True)
    # exported mask lives in the ORIGINAL scan grid
    assert res.labels.shape == img.shape[:3] and np.allclose(res.affine, TEST_AFFINE)
    paths = save_result(res, tmp_path / "out", "sub1")
    seg = nib.load(str(paths["segmentation"]))
    assert (
        seg.shape == img.shape[:3]
        and np.allclose(seg.affine, img.affine)
        and seg.get_data_dtype() == np.uint8
    )
    assert nib.aff2axcodes(seg.affine) == nib.aff2axcodes(img.affine)
    assert set(np.unique(np.asarray(seg.dataobj))) <= {0, 1, 2, 3}
    ent = np.asarray(nib.load(str(paths["entropy"])).dataobj)
    assert ent.min() >= 0 and ent.max() <= np.log(4) + 1e-4
    assert (
        res.metadata["inference"]["mc_dropout_samples"] == 3
        and "not validated clinical risk" in res.metadata["uncertainty_note"]
    )
    # volumes: mL from physical voxel size (1.2*1.0*1.5 mm = 1.8 mm^3)
    n = int((res.labels > 0).sum())
    assert res.volumes["regions"]["WT"]["voxels"] == n
    assert res.volumes["regions"]["WT"]["volume_ml"] == pytest.approx(n * 1.8 / 1000)
    assert paths["volumes_csv"].exists() and json.loads(paths["metadata"].read_text())["research_only"]
    probs = np.asarray(nib.load(str(paths["probabilities"])).dataobj)
    assert probs.shape == (*img.shape[:3], 4) and np.allclose(probs.sum(-1), 1, atol=2e-2)
    assert np.array_equal(probs.argmax(-1), res.labels) or (probs.argmax(-1) == res.labels).mean() > 0.99


def test_inverse_transform_with_rotated_input(trained, synth_root, tmp_path):
    """Prediction for the same anatomy stored in a different orientation must map back into that orientation."""
    d, img = _write_case(tmp_path, synth_root)
    p = Predictor(trained["ckpt"], "cpu")
    base = p.predict_dir(d).labels
    flipped_aff = np.diag([1.2, 1.0, 1.5, 1.0])
    flipped_aff[:3, 3] = [-30, -20, 5]
    arr = [
        np.asarray(nib.load(str(d / f"sub1_{m}.nii.gz")).dataobj)[::-1] for m in ["flair", "t1", "t1ce", "t2"]
    ]  # flip axis 0 in the array
    d2 = tmp_path / "flip"
    d2.mkdir()
    aff2 = TEST_AFFINE.copy()
    aff2[0, 0], aff2[0, 3] = -TEST_AFFINE[0, 0], TEST_AFFINE[0, 3] + TEST_AFFINE[0, 0] * (arr[0].shape[0] - 1)
    for a, m in zip(arr, ["flair", "t1", "t1ce", "t2"]):
        nib.save(nib.Nifti1Image(np.ascontiguousarray(a), aff2), str(d2 / f"s_{m}.nii.gz"))
    r2 = p.predict_dir(d2).labels
    assert (
        r2[::-1] == base
    ).mean() > 0.995  # same physical anatomy -> same segmentation after undoing the array flip


@pytest.mark.parametrize("case", ["missing", "affine", "nan", "zeros", "spacing", "ambiguous"])
def test_input_validation_rejects(trained, synth_root, tmp_path, case):
    d, img = _write_case(tmp_path, synth_root)
    p = Predictor(trained["ckpt"], "cpu")
    if case == "missing":
        (d / "sub1_t2.nii.gz").unlink()
    elif case == "affine":
        a = TEST_AFFINE.copy()
        a[0, 3] += 4
        nib.save(
            nib.Nifti1Image(np.asarray(nib.load(str(d / "sub1_t1.nii.gz")).dataobj), a),
            str(d / "sub1_t1.nii.gz"),
        )
    elif case == "nan":
        arr = np.asarray(nib.load(str(d / "sub1_t1.nii.gz")).dataobj).copy()
        arr[5, 5, 5] = np.nan
        nib.save(nib.Nifti1Image(arr, TEST_AFFINE), str(d / "sub1_t1.nii.gz"))
    elif case == "zeros":
        nib.save(nib.Nifti1Image(np.zeros(img.shape[:3], np.float32), TEST_AFFINE), str(d / "sub1_t1.nii.gz"))
    elif case == "spacing":
        for m in ["flair", "t1", "t1ce", "t2"]:
            nib.save(
                nib.Nifti1Image(
                    np.asarray(nib.load(str(d / f"sub1_{m}.nii.gz")).dataobj), np.diag([2.0, 2.0, 2.0, 1.0])
                ),
                str(d / f"sub1_{m}.nii.gz"),
            )
    elif case == "ambiguous":
        nib.save(nib.Nifti1Image(np.ones(img.shape[:3], np.float32), TEST_AFFINE), str(d / "other_t1.nii.gz"))
    with pytest.raises(InputError):
        p.predict_dir(d)


def test_not_nifti_rejected(trained, tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "a_flair.nii.gz").write_text("not a nifti")
    p = Predictor(trained["ckpt"], "cpu")
    with pytest.raises(InputError):
        p.load_volumes({"flair": tmp_path / "x" / "a_flair.nii.gz"})


def test_uncertainty_ensemble_and_mc(trained):
    p = Predictor(trained["ckpt"], "cpu")
    x = torch.randn(4, 32, 32, 32)
    out = predict_with_uncertainty(p.models, x, p.patch, 0.25, "cpu", mc_samples=4)
    assert out["n_samples"] == 4 and out["mutual_information"].min() >= 0
    assert torch.allclose(out["mean_probs"].sum(0), torch.ones(32, 32, 32), atol=1e-4)
    assert (
        out["entropy"].max() <= np.log(4) + 1e-4 and out["mutual_information"].max() > 0
    )  # MC dropout produces spread
    ens = predict_with_uncertainty([p.models[0], p.models[0]], x, p.patch, 0.25, "cpu")
    assert (
        ens["n_samples"] == 2 and ens["mutual_information"].max() < 1e-5
    )  # identical members -> no epistemic spread
    nodrop = Predictor(trained["ckpt"], "cpu")
    for m in nodrop.models[0].modules():
        if isinstance(m, torch.nn.Dropout3d):
            m.p = 0.0
    with pytest.raises(ValueError, match="no dropout"):
        predict_with_uncertainty(nodrop.models, x, p.patch, 0.25, "cpu", mc_samples=2)
    assert torch.allclose(
        entropy(torch.tensor([1.0, 0, 0, 0]).view(4, 1, 1, 1)), torch.zeros(1, 1, 1), atol=1e-6
    )


def test_full_volume_evaluation(trained, processed, tmp_path):
    from neuroseg.data.pipeline import load_splits

    p = Predictor(trained["ckpt"], "cpu")
    dirs = [processed["processed"] / c for c in load_splits(processed["splits"])["validation"]["cases"]]
    res = evaluate_split(
        p, dirs, tmp_path / "ev", n_boot=50, save_figures=1, split_name="validation", log=lambda *_: None
    )
    assert res["n_patients"] == len(dirs)
    row = [r for r in res["summary"] if r["region"] == "WT" and r["metric"] == "dice"][0]
    assert 0 <= row["mean"] <= 1 and row["ci_lo"] <= row["mean"] <= row["ci_hi"]
    for f in (
        "validation_per_patient.csv",
        "validation_summary.csv",
        "figures/validation_dice_boxplot.png",
        "figures/validation_confusion.png",
        "figures/validation_reliability.png",
    ):
        assert (tmp_path / "ev" / f).exists()
