import numpy as np
import pandas as pd
import pytest

from neuroseg.evaluation import calibration as cal
from neuroseg.evaluation import metrics as M
from neuroseg.evaluation import statistics as S


def cube(shape=(30, 30, 30), lo=5, hi=15, shift=0):
    m = np.zeros(shape, bool)
    m[lo + shift : hi + shift, lo:hi, lo:hi] = True
    return m


def test_dice_iou_values():
    a, b = cube(), cube(shift=5)  # 10x10x10 cubes, half overlap along axis 0
    r = M.overlap_metrics(a, b)
    assert r["dice"] == pytest.approx(0.5) and r["iou"] == pytest.approx(1 / 3)
    assert r["sensitivity"] == pytest.approx(0.5) and r["precision"] == pytest.approx(0.5)


def test_empty_mask_conventions():
    e = np.zeros((10, 10, 10), bool)
    both = M.overlap_metrics(e, e)
    assert both["dice"] == 1.0 and np.isnan(both["sensitivity"])
    assert M.surface_metrics(e, e)["hd95"] == 0.0
    one = M.overlap_metrics(cube((10, 10, 10), 2, 5), e)
    assert one["dice"] == 0.0
    assert np.isnan(M.surface_metrics(cube((10, 10, 10), 2, 5), e)["hd95"])
    assert M.surface_metrics(cube((10, 10, 10), 2, 5), e, empty_penalty=373.0)["hd95"] == 373.0


def test_hd95_uses_physical_spacing():
    a, b = cube(), cube(shift=3)
    h1 = M.surface_metrics(a, b, (1, 1, 1))["hd95"]
    h2 = M.surface_metrics(a, b, (2, 1, 1))["hd95"]  # shift along axis 0 with 2 mm voxels
    assert h1 == pytest.approx(3.0, abs=0.5) and h2 == pytest.approx(6.0, abs=1.0)
    assert M.surface_metrics(a, a)["hd95"] == 0 and M.surface_metrics(a, a)["assd"] == 0


def test_assd_symmetric():
    a, b = cube(), cube(shift=2)
    assert M.surface_metrics(a, b)["assd"] == pytest.approx(M.surface_metrics(b, a)["assd"])


def test_region_definitions_and_case_eval():
    lab = np.zeros((20, 20, 20), np.uint8)
    lab[2:6] = 1
    lab[6:10] = 2
    lab[10:12] = 3
    m = M.region_masks(lab)
    assert (
        m["WT"].sum() == (lab > 0).sum()
        and m["TC"].sum() == ((lab == 1) | (lab == 3)).sum()
        and m["ET"].sum() == (lab == 3).sum()
    )
    r = M.evaluate_case(lab, lab, (1, 1, 2))
    for reg in ("WT", "TC", "ET"):
        assert r[reg]["dice"] == 1.0 and r[reg]["abs_volume_error_ml"] == 0
    assert r["ET"]["gt_volume_ml"] == pytest.approx(2 * 400 * 2 / 1000)


def test_volume_error():
    gt, pr = np.zeros((10, 10, 10), np.uint8), np.zeros((10, 10, 10), np.uint8)
    gt[:5] = 3
    pr[:4] = 3
    r = M.evaluate_case(pr, gt, (1, 1, 1))["ET"]
    assert r["rel_volume_error"] == pytest.approx(-0.2) and r["abs_volume_error_ml"] == pytest.approx(0.1)


def test_confusion_matrix():
    gt = np.array([0, 1, 2, 3, 3])
    pr = np.array([0, 1, 1, 3, 2])
    cm = M.confusion_matrix(pr, gt)
    assert cm.sum() == 5 and cm[3, 3] == 1 and cm[3, 2] == 1 and cm[2, 1] == 1


def test_bootstrap_is_patient_level_and_reproducible():
    v = np.random.default_rng(0).uniform(0.6, 0.9, 40)
    a, b = S.bootstrap_ci(v, seed=1), S.bootstrap_ci(v, seed=1)
    assert a == b and a["lo"] < a["mean"] < a["hi"] and a["n"] == 40
    small = S.bootstrap_ci(v[:5], seed=1)
    assert (small["hi"] - small["lo"]) > (a["hi"] - a["lo"])  # fewer PATIENTS -> wider CI (not voxel counts)
    assert S.bootstrap_ci([np.nan, np.nan])["n"] == 0
    d = S.paired_bootstrap_diff(v, v - 0.05)
    assert d["mean_diff"] == pytest.approx(0.05) and d["lo"] > 0


def test_summary_and_subgroups():
    per = {
        f"p{i}": M.evaluate_case(*(lambda g: (g, g))(np.pad(np.ones((4, 4, 4), np.uint8) * 3, 3)), (1, 1, 1))
        for i in range(30)
    }
    df = S.results_to_frame(per)
    s = S.summarize(df, n_boot=100)
    assert s[(s.region == "WT") & (s.metric == "dice")]["mean"].iloc[0] == 1.0
    meta = pd.DataFrame({"patient": list(per), "site": ["A"] * 25 + ["B"] * 5})
    sg = S.subgroup_summary(df, meta, "site", min_group=10)
    assert (
        sg[sg.site == "B"].note.str.contains("insufficient").all()
        and not sg[sg.site == "A"]["mean"].isna().any()
    )


def test_calibration_perfect_vs_overconfident():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 5000)
    perfect = np.eye(4)[y].T * 0.97 + 0.01
    assert cal.reliability(perfect, y)["ece"] < 0.05
    over = np.eye(4)[(y + 1) % 4].T * 0.97 + 0.01
    assert cal.reliability(over, y)["ece"] > 0.9
    ent = np.where(rng.random(5000) < 0.5, 1.2, 0.1)
    pred = np.where(ent > 1, rng.integers(0, 4, 5000), y)
    pred, y2 = np.maximum(pred, 1), np.maximum(y, 1)
    out = cal.entropy_error_agreement(ent, pred, y2)
    assert out["auroc_entropy_vs_error"] > 0.7
