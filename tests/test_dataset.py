import numpy as np
import pytest

from neuroseg.data import validation as V
from neuroseg.data.dataset import FullVolumeDataset, PatchDataset, label_map_from_names, load_processed
from neuroseg.data.patch_sampler import PatchSampler, extract_patch
from neuroseg.data.pipeline import load_splits, make_splits
from neuroseg.data.preprocessing import remap_labels
from neuroseg.data.transforms import Augmenter, random_affine


def test_dataset_schema(synth_root):
    info = V.verify_dataset_json(synth_root)
    assert info["modalities"] == ["FLAIR", "T1w", "t1gd", "T2w"]
    assert info["num_training_listed"] == 10 and set(info["labels"]) == {0, 1, 2, 3}


def test_missing_files_detected(tmp_path, synth_root):
    import shutil

    shutil.copytree(synth_root, tmp_path / "d")
    next((tmp_path / "d" / "imagesTr").iterdir()).unlink()
    with pytest.raises(V.DatasetError):
        V.verify_dataset_json(tmp_path / "d")


def test_input_channel_ordering(adapter, processed):
    assert adapter.modalities == ["flair", "t1", "t1ce", "t2"]
    c = load_processed(processed["processed"] / "SYNTH_000", mmap=False)
    assert c["meta"]["modalities"] == adapter.modalities
    img, lab = c["image"].astype(np.float32), c["label"]
    # synthetic t1gd (channel 2) is brightest on enhancing tissue relative to edema; T1w (1) is darkest on core
    assert (
        img[2][lab == 3].mean() > img[2][lab == 2].mean() > 0
        or img[2][lab == 3].mean() > img[2][lab == 1].mean()
    )
    assert img[1][lab == 1].mean() < img[1][lab == 0].mean()  # core dark in the T1w channel only


def test_label_remap_from_names(adapter):
    assert adapter.label_map == {0: 0, 1: 2, 2: 1, 3: 3}  # MSD: 1 edema->2, 2 NET->1, 3 ET->3
    lab = np.array([[[0, 1, 2, 3]]])
    assert remap_labels(lab, adapter.label_map).ravel().tolist() == [0, 2, 1, 3]
    with pytest.raises(ValueError):
        remap_labels(np.array([[[7]]]), adapter.label_map)
    with pytest.raises(ValueError):
        label_map_from_names({0: "background", 1: "mystery"})


def test_patient_level_split_isolation(adapter):
    recs = adapter.list_cases()
    s = make_splits(recs, seed=5)
    sets = [set(s[k]["patients"]) for k in ("train", "validation", "test")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(x) for x in sets) == 10
    assert make_splits(recs, seed=5)["train"] == s["train"]  # deterministic
    assert make_splits(recs, seed=6)["train"] != s["train"]


def test_leakage_assertions_terminate():
    with pytest.raises(V.LeakageError):
        V.assert_no_leakage({"train": ["a", "b"], "validation": ["b"], "test": ["c"]})
    with pytest.raises(V.LeakageError):
        V.assert_no_leakage({"train": ["a", "a"], "validation": [], "test": []})
    with pytest.raises(V.LeakageError):  # two time points of one patient in different splits
        V.assert_case_patient_consistency(
            {"p1-000": "p1", "p1-001": "p1"}, {"train": ["p1-000"], "test": ["p1-001"]}
        )


def test_timepoints_share_partition():
    from neuroseg.data.dataset import CaseRecord

    recs = [CaseRecord(f"P{i}-00{t}", f"P{i}", ["x"], "y", []) for i in range(20) for t in range(2)]
    s = make_splits(recs, seed=1)
    for part in ("train", "validation", "test"):
        pats = {c.split("-")[0] for c in s[part]["cases"]}
        assert pats == set(s[part]["patients"])


def test_load_splits_reverifies(processed, tmp_path):
    import json

    s = load_splits(processed["splits"])
    d = tmp_path / "bad"
    d.mkdir()
    for k in ("train", "validation", "test"):
        v = dict(s[k])
        if k == "test":
            v["patients"] = s["train"]["patients"][:1]
        (d / f"{k}.json").write_text(json.dumps(v))
    with pytest.raises(V.LeakageError):
        load_splits(d)


def test_patch_extraction_padding():
    a = np.ones((1, 10, 10, 10), np.float32)
    p = extract_patch(a, (0, 0, 0), (8, 8, 8))
    assert p.shape == (1, 8, 8, 8) and p[0, 0, 0, 0] == 0 and p[0, 4, 4, 4] == 1


def test_sampler_modes_hit_targets():
    rng = np.random.default_rng(0)
    lab = np.zeros((40, 40, 40), np.uint8)
    lab[10:14, 10:14, 10:14] = 3
    img = np.ones((1, 40, 40, 40), np.float32)
    s = PatchSampler((16, 16, 16), jitter=0)
    for _ in range(10):
        c = s.sample_center(lab, img, rng, "tumor")
        assert lab[c] == 3
        b = s.sample_center(lab, img, rng, "background")
        assert lab[b] == 0
        e = s.sample_center(lab, img, rng, "boundary")
        assert lab[e] == 3


def test_patch_dataset_and_fullvolume(processed):
    dirs = [processed["processed"] / c for c in load_splits(processed["splits"])["train"]["cases"]]
    ds = PatchDataset(dirs, (32, 32, 32), 4, ["flair", "t1ce"], Augmenter("full"), seed=0)
    b = ds[0]
    assert (
        b["image"].shape == (2, 32, 32, 32)
        and b["label"].shape == (32, 32, 32)
        and b["label"].dtype.is_floating_point is False
    )
    fv = FullVolumeDataset(dirs[:1])[0]
    assert fv["image"].shape[0] == 4 and fv["label"].shape == fv["image"].shape[1:]


def test_augmentation_label_alignment():
    """Synchronised spatial transforms: a label defined as a function of the image must stay aligned."""
    rng = np.random.default_rng(0)
    z, y, x = np.meshgrid(*[np.arange(24)] * 3, indexing="ij")
    img = ((((z - 12) / 10) ** 2 + ((y - 12) / 6) ** 2 + ((x - 12) / 4) ** 2) < 1).astype(np.float32)[
        None
    ] * 5.0
    lab = (img[0] > 0).astype(np.uint8)
    for _ in range(5):
        xo, yo = random_affine(img, lab, rng, max_deg=20)
        overlap = ((xo[0] > 2.5) == (yo > 0)).mean()  # only boundary voxels differ (trilinear vs nearest)
        assert overlap > 0.98
        assert (
            (xo[0] > 2.5) == (lab > 0)
        ).mean() < overlap  # negative control: an un-transformed label is worse
    aug = Augmenter(
        "full",
        {
            "p_flip": 1,
            "p_affine": 1,
            "p_gamma": 1,
            "p_bias": 1,
            "p_noise": 1,
            "p_smooth": 1,
            "p_moddrop": 1,
            "p_scale": 1,
        },
    )
    xo, yo = aug(np.repeat(img, 3, 0), lab, rng)
    assert xo.shape == (3, 24, 24, 24) and set(np.unique(yo)) <= {0, 1}
    assert np.isfinite(xo).all()
