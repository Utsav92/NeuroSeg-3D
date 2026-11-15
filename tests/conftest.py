import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroseg.data.dataset import MSDBrainAdapter  # noqa: E402
from neuroseg.data.pipeline import make_splits, preprocess_dataset, save_splits  # noqa: E402
from neuroseg.data.synthetic import write_msd_like  # noqa: E402
from neuroseg.training.trainer import Trainer  # noqa: E402

# Rotated + flipped affine (LAS-like with anisotropic 1.2x1x1.5 mm voxels) so geometry bugs show up.
TEST_AFFINE = np.array([[-1.2, 0, 0, 30.0], [0, 1.0, 0, -20.0], [0, 0, 1.5, 5.0], [0, 0, 0, 1]])


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    return write_msd_like(
        tmp_path_factory.mktemp("raw") / "SYNTH", n_cases=10, seed=1, shape=(40, 44, 36), affine=TEST_AFFINE
    )


@pytest.fixture(scope="session")
def adapter(synth_root):
    return MSDBrainAdapter(synth_root)


@pytest.fixture(scope="session")
def processed(tmp_path_factory, adapter):
    d = tmp_path_factory.mktemp("proc")
    recs = adapter.list_cases()
    splits = make_splits(recs, seed=3)
    save_splits(splits, d / "splits")
    preprocess_dataset(
        adapter,
        d / "processed",
        recs,
        {"clip_percentiles": [0.5, 99.5], "crop_margin": 2},
        log=lambda *_: None,
    )
    return {"processed": d / "processed", "splits": d / "splits", "recs": recs}


@pytest.fixture(scope="session")
def trained(tmp_path_factory, processed):
    """A genuinely trained (3 epochs, synthetic data) tiny ResUNet. Meaningless as medical accuracy."""
    out = tmp_path_factory.mktemp("out")
    cfg = {
        "experiment": "pytest_resunet", "seed": 0, "output_dir": str(out),
        "data": {"dataset": "synthetic", "modalities": ["flair", "t1", "t1ce", "t2"], "patch_size": [32, 32, 32], "samples_per_epoch": 6, "augmentation": "full",
                 "processed_dir": str(processed["processed"]), "splits_dir": str(processed["splits"])},
        "model": {"arch": "resunet3d", "base_channels": 4, "depth": 4, "dropout": 0.1},
        "loss": {"components": {"dice": 0.5, "ce": 0.5}}, "optim": {"name": "adamw", "lr": 0.002}, "scheduler": {"name": "cosine"},
        "train": {"epochs": 3, "device": "cpu", "batch_size": 2, "grad_accum": 2, "num_workers": 0, "val_interval": 1, "val_max_cases": 2,
                  "gradient_checkpointing": True, "tensorboard": False},
    }  # fmt: skip
    rec = Trainer(cfg).fit()
    return {"record": rec, "ckpt": Path(rec["checkpoint_best"]), "cfg": cfg, "out": out}
