import pytest
import torch

from neuroseg.models import ARCHS, build_model


@pytest.mark.parametrize("arch", list(ARCHS))
def test_forward_shape_and_backprop(arch):
    m = build_model({"arch": arch, "in_channels": 4, "out_channels": 4, "base_channels": 4, "depth": 4})
    x = torch.randn(2, 4, 32, 32, 32)
    y = m(x)
    assert y.shape == (2, 4, 32, 32, 32) and torch.isfinite(y).all()
    y.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert sum(p.grad is not None for p in m.parameters()) == sum(
        1 for _ in m.parameters()
    )  # every param trained


def test_encoder_widths_default():
    m = build_model({"arch": "unet3d", "base_channels": 16})
    assert m.widths == [16, 32, 64, 128, 256]


def test_divisibility_error():
    m = build_model({"arch": "unet3d", "base_channels": 4})
    with pytest.raises(ValueError):
        m(torch.randn(1, 4, 20, 16, 16))


def test_random_init_no_pretrained():
    a = build_model({"arch": "resunet3d", "base_channels": 4})
    torch.manual_seed(1)
    b = build_model({"arch": "resunet3d", "base_channels": 4})
    assert not torch.equal(a.head.weight, b.head.weight)  # fresh random init each time


def test_attention_gates_modulate_skips():
    m = build_model({"arch": "attention_unet3d", "base_channels": 4})
    out, atts = m(torch.randn(1, 4, 32, 32, 32), return_attention=True)
    assert len(atts) == 4 and all(((a >= 0) & (a <= 1)).all() for a in atts)
    assert atts[-1].shape[2:] == (32, 32, 32)  # finest gate matches full resolution


def test_residual_block_projection():
    from neuroseg.models.blocks import ResBlock

    b = ResBlock(4, 8)
    assert b(torch.randn(1, 4, 8, 8, 8)).shape == (1, 8, 8, 8, 8) and not isinstance(
        b.proj, torch.nn.Identity
    )
    assert isinstance(ResBlock(8, 8).proj, torch.nn.Identity)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_gradient_checkpointing_equivalence(arch):
    torch.manual_seed(0)
    m = build_model({"arch": arch, "base_channels": 4, "gradient_checkpointing": False})
    m2 = build_model({"arch": arch, "base_channels": 4, "gradient_checkpointing": True})
    m2.load_state_dict(m.state_dict())
    x = torch.randn(1, 4, 32, 32, 32)
    m.train(), m2.train()
    m(x).square().mean().backward()
    m2(x).square().mean().backward()
    for p, q in zip(m.parameters(), m2.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-5, rtol=1e-4)


def test_group_norm_and_interp_options():
    m = build_model(
        {
            "arch": "resunet3d",
            "base_channels": 8,
            "norm": "group",
            "up_mode": "interp",
            "down_mode": "strided",
        }
    )
    assert m(torch.randn(1, 4, 32, 32, 32)).shape == (1, 4, 32, 32, 32)


def test_mc_dropout_stochastic_only_when_enabled():
    m = build_model({"arch": "resunet3d", "base_channels": 4, "dropout": 0.3}).eval()
    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        assert torch.equal(m(x), m(x))
        m.set_dropout_mc(True)
        assert not torch.equal(m(x), m(x))
        m.set_dropout_mc(False)
        assert torch.equal(m(x), m(x))


def test_device_backend_probe():
    """Real fwd+bwd on every available backend (reports which one the machine actually supports)."""
    from neuroseg.utils.device import device_report

    rep = device_report()
    assert rep["probes"]["cpu"]["fwd_bwd_3d_ops"]
