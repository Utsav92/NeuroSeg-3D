import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from components.viewer import aspect, render  # noqa: E402

from neuroseg.evaluation import visualization as viz  # noqa: E402


def blob_label():
    lab = np.zeros((40, 50, 30), np.uint8)
    lab[10:30, 15:35, 8:22] = 2
    lab[15:25, 20:30, 12:18] = 1
    lab[18:22, 23:27, 14:16] = 3
    return lab


def test_surface_mesh_uses_physical_spacing():
    m = blob_label() > 0
    sp = (1.2, 1.0, 2.5)
    v, f = viz.surface_mesh(m, sp, step=1, smooth_sigma=0)
    ext = v.max(0) - v.min(0)
    true = np.array([20, 20, 14]) * np.array(sp)  # voxel extent * spacing (mm)
    assert np.allclose(ext, true, atol=2.0)
    assert f.max() < len(v) and f.shape[1] == 3
    assert viz.surface_mesh(np.zeros((8, 8, 8), bool), sp) is None


def test_figure_3d_regions_and_aspect():
    fig = viz.figure_3d(blob_label(), (1.0, 1.0, 2.0))
    assert [t.name for t in fig.data] == ["WT", "TC", "ET"]
    assert fig.layout.scene.aspectmode == "data"
    only = viz.figure_3d(blob_label(), (1, 1, 1), show=("ET",))
    assert [t.name for t in only.data] == ["ET"]
    assert viz.mesh_to_obj(*viz.surface_mesh(blob_label() == 3, (1, 1, 1), smooth_sigma=0)).startswith("v ")


@pytest.mark.parametrize("plane", ["Axial", "Coronal", "Sagittal"])
def test_2d_viewer_renders_all_layers(plane):
    lab = blob_label()
    img = np.random.default_rng(0).random(lab.shape)
    ent = np.random.default_rng(1).random(lab.shape)
    fig = render(img, (1.2, 1.0, 2.5), plane, 15, lab, lab, ent, 0.5)
    assert fig is not None
    fig2 = render(img, (1, 1, 1), plane, 15, lab, np.roll(lab, 3, 0), None, 0.5, show_error=True)
    assert len(fig2.axes[0].images) >= 2


def test_viewer_aspect_follows_spacing():
    assert aspect((1.0, 1.0, 2.0), 2) == 1.0  # axial: in-plane isotropic
    assert aspect((1.0, 1.0, 2.0), 1) == pytest.approx(2.0)  # coronal rows follow z


def test_static_figures(tmp_path):
    lab = blob_label()
    img = np.random.default_rng(0).random(lab.shape)
    viz.case_figure(
        img, lab, np.roll(lab, 2, 1), np.random.default_rng(0).random(lab.shape), path=tmp_path / "c.png"
    )
    assert (tmp_path / "c.png").stat().st_size > 1000
