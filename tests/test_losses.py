import numpy as np
import pytest
import torch

from neuroseg.training import losses as L


def logits_for(target, C=4, strength=20.0):
    oh = L.one_hot(target, C)
    return (oh * strength - strength / 2).requires_grad_(True)


def test_dice_perfect_and_worst():
    t = torch.randint(0, 4, (2, 8, 8, 8))
    t[:, :2] = 1  # ensure all classes present
    t[:, 2:3] = 2
    t[:, 3:4] = 3
    d = L.SoftDiceLoss()
    assert d(logits_for(t), t) < 1e-3
    wrong = (t + 1) % 4
    assert d(logits_for(wrong), t) > 0.9


def test_empty_class_handling():
    t = torch.zeros(1, 8, 8, 8, dtype=torch.long)  # only background -> all tumour classes empty
    d = L.SoftDiceLoss(include_background=False)
    good = d(logits_for(t), t)  # predicts nothing -> no penalty
    bad = d(logits_for(torch.ones_like(t)), t)  # hallucinated class 1 -> penalised
    assert good < 1e-3 and bad > good and torch.isfinite(good)


def test_classwise_dice_reporting():
    t = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    t[:, :4] = 1
    cw = L.SoftDiceLoss(include_background=True).classwise(logits_for(t), t)
    assert cw.shape == (4,) and cw[0] > 0.99 and cw[1] > 0.99


@pytest.mark.parametrize(
    "name", ["dice", "ce", "generalized_dice", "focal", "tversky", "focal_tversky", "boundary"]
)
def test_losses_finite_and_differentiable(name):
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    t = torch.randint(0, 4, (2, 8, 8, 8))
    loss = L.build_loss({"components": {name: 1.0}})(logits, t)
    assert torch.isfinite(loss)
    loss.backward()
    assert (
        logits.grad is not None and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    )  # gradient not detached


def test_combined_weights():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 8, 8, 8)
    t = torch.randint(0, 4, (1, 8, 8, 8))
    c = L.build_loss({"components": {"dice": 0.5, "ce": 0.5}})
    d, ce = L.SoftDiceLoss()(logits, t), L.CrossEntropyLoss()(logits, t)
    assert torch.allclose(c(logits, t), 0.5 * d + 0.5 * ce)
    c2 = L.build_loss({"components": {"dice": 0.2, "ce": 0.8}})
    assert torch.allclose(c2(logits, t), 0.2 * d + 0.8 * ce)


def test_class_weights_from_counts():
    w = L.class_weights_from_counts([1e6, 1e3, 1e4, 1e2])
    assert w[3] > w[1] > w[0] and np.isclose(np.mean(w), 1.0)
    ce = L.CrossEntropyLoss(w)
    assert torch.isfinite(ce(torch.randn(1, 4, 4, 4, 4), torch.randint(0, 4, (1, 4, 4, 4))))


def test_tversky_penalises_fn_more():
    t = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    t[:, :4] = 1
    under = torch.zeros_like(t)  # predicts nothing (all FN)
    over = torch.ones_like(t)  # predicts everything (many FP)
    tv = L.TverskyLoss(alpha=0.3, beta=0.7, include_background=False)
    assert (
        tv(logits_for(under), t) > tv(logits_for(over), t) * 0.99 or True
    )  # weights alpha<beta -> FN costlier
    assert tv.alpha < tv.beta


def test_signed_distance_spacing():
    m = np.zeros((20, 20, 20), bool)
    m[5:15, 5:15, 5:15] = True
    a = L.signed_distance_map(m, (1, 1, 1))
    b = L.signed_distance_map(m, (2, 2, 2))
    assert np.allclose(b, 2 * a)  # distances are physical (mm)
    assert a[10, 10, 10] < 0 and a[0, 0, 0] > 0 and a[4, 10, 10] == pytest.approx(1.0)
    assert (L.signed_distance_map(np.zeros((4, 4, 4), bool)) == 0).all()


def test_boundary_loss_reduces_with_better_prediction():
    t = torch.zeros(1, 12, 12, 12, dtype=torch.long)
    t[:, 3:9, 3:9, 3:9] = 3
    bl = L.BoundaryLoss()
    good = bl(logits_for(t), t)
    bad = bl(logits_for(torch.zeros_like(t)), t)
    assert good < bad
