import numpy as np
import pytest
from conftest import TEST_AFFINE

from neuroseg.data.preprocessing import normalize_channel, preprocess_case
from neuroseg.utils import geometry as G


def rot_affine():
    # axes permuted + flipped: array axis0 -> -y, axis1 -> +x, axis2 -> -z
    a = np.zeros((4, 4))
    a[1, 0], a[0, 1], a[2, 2], a[3, 3] = -1.1, 0.9, -2.0, 1
    a[:3, 3] = [5, 6, 7]
    return a


@pytest.mark.parametrize("aff", [np.eye(4), TEST_AFFINE, rot_affine()])
def test_canonical_roundtrip_and_orientation(aff):
    rng = np.random.default_rng(0)
    arr = rng.random((7, 9, 11)).astype(np.float32)
    canon, caff = G.to_canonical(arr, aff)
    assert G.axcodes(caff) == ("R", "A", "S")
    assert np.array_equal(G.from_canonical(canon, aff), arr)  # exact inverse
    # a voxel keeps its physical (world) position under reorientation
    idx = (2, 3, 4)
    world = (aff @ np.array([*idx, 1]))[:3]
    cidx = np.argwhere(canon == arr[idx])[0]
    assert np.allclose((caff @ np.array([*cidx, 1]))[:3], world)


def test_normalization_properties():
    rng = np.random.default_rng(0)
    x = np.zeros((20, 20, 20), np.float32)
    x[5:15, 5:15, 5:15] = rng.normal(300, 40, (10, 10, 10)).clip(1)
    n, st = normalize_channel(x)
    assert (n[x == 0] == 0).all()
    fg = n[x != 0]
    assert abs(fg.mean()) < 0.05 and abs(fg.std() - 1) < 0.1
    n0, s0 = normalize_channel(np.zeros((5, 5, 5), np.float32))  # empty foreground
    assert (n0 == 0).all() and s0["empty_foreground"]
    c = np.zeros((5, 5, 5), np.float32)
    c[1:3, 1:3, 1:3] = 7.0  # zero variance
    nc, sc = normalize_channel(c)
    assert np.isfinite(nc).all() and sc["zero_variance"]


def test_outlier_clipping_is_robust():
    x = np.zeros((30, 30, 30), np.float32)
    x[5:25, 5:25, 5:25] = 100
    x[10, 10, 10] = 1e6
    n, _ = normalize_channel(x)
    assert n.max() < 10


def test_preprocessing_preserves_geometry(adapter):
    rec = adapter.list_cases()[0]
    img, aff, lab, _ = adapter.load(rec)
    out = preprocess_case(img, aff, lab, adapter.label_map, adapter.modalities, crop_margin=2)
    meta = G.SpatialMeta.from_dict(out["spatial"])
    assert tuple(meta.original_shape) == lab.shape and out["spacing_mm"] == pytest.approx((1.2, 1.0, 1.5))
    # inverse transform reproduces the ORIGINAL remapped label exactly, in the original grid
    back = G.to_original_space(out["label"], meta)
    expected = np.zeros_like(lab, dtype=np.uint8)
    for k, v in adapter.label_map.items():
        expected[lab == k] = v
    assert np.array_equal(back, expected)
    assert out["image"].dtype == np.float16 and out["image"].shape[0] == 4


def test_crop_uncrop():
    v = np.zeros((20, 20, 20), np.float32)
    v[4:9, 6:12, 3:7] = 1
    s, sh = G.foreground_bbox(v, margin=1)
    c = G.crop(v, s, sh)
    assert np.array_equal(G.uncrop(c, s, v.shape), v)


def test_alignment_checks_detect_mismatch():
    a2 = TEST_AFFINE.copy()
    a2[0, 3] += 5
    assert G.check_alignment([(5, 5, 5), (5, 5, 5)], [TEST_AFFINE, TEST_AFFINE]) == []
    assert G.check_alignment([(5, 5, 5), (5, 5, 5)], [TEST_AFFINE, a2])
    assert G.check_alignment([(5, 5, 5), (6, 5, 5)], [TEST_AFFINE, TEST_AFFINE])


def test_resample_interpolation_orders():
    lab = np.zeros((10, 10, 10), np.uint8)
    lab[2:6] = 3
    up = G.resample(lab, (2, 2, 2), (1, 1, 1), order=0)
    assert set(np.unique(up)) == {0, 3} and up.shape == (20, 20, 20)
    back = G.resample_to_shape(up, (10, 10, 10), 0)
    assert np.array_equal(back, lab)


def test_volume_ml():
    m = np.zeros((10, 10, 10), bool)
    m[:5, :5, :5] = True  # 125 voxels
    assert G.volume_ml(m, (2.0, 1.0, 1.0)) == pytest.approx(0.25)  # 250 mm^3
