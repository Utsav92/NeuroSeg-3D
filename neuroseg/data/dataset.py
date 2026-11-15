"""Dataset adapters (raw layout -> records) and torch datasets over preprocessed cases."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from neuroseg.data.patch_sampler import PatchSampler, extract_patch
from neuroseg.data.preprocessing import CANONICAL_MODALITIES

# raw-name -> canonical modality name
_MODALITY_ALIASES = {
    "flair": "flair", "t2f": "flair", "t2-flair": "flair",
    "t1": "t1", "t1w": "t1", "t1n": "t1",
    "t1gd": "t1ce", "t1c": "t1ce", "t1ce": "t1ce", "t1gad": "t1ce",
    "t2": "t2", "t2w": "t2",
}  # fmt: skip


def canonical_modality(name: str) -> str:
    key = name.strip().lower()
    if key not in _MODALITY_ALIASES:
        raise ValueError(f"unrecognised modality name '{name}'")
    return _MODALITY_ALIASES[key]


def label_map_from_names(labels: dict[int, str]) -> dict[int, int]:
    """Derive raw->contiguous mapping from *label names* (never from assumed integer values).

    contiguous: 0 background, 1 necrotic/non-enhancing, 2 edema, 3 enhancing.
    """
    out = {}
    for k, name in labels.items():
        n = name.lower()
        if "background" in n:
            out[k] = 0
        elif "edema" in n or "oedema" in n:
            out[k] = 2
        elif "non-enhancing" in n or "non enhancing" in n or "necrotic" in n or "necrosis" in n:
            out[k] = 1
        elif "enhancing" in n:
            out[k] = 3
        else:
            raise ValueError(f"cannot map label {k}='{name}'")
    return out


@dataclass
class CaseRecord:
    case_id: str
    patient_id: str
    image_paths: list  # one 4D file (MSD) or one file per modality (BraTS)
    label_path: str | None
    modalities: list  # canonical names, in channel order of the loaded array
    meta: dict = field(default_factory=dict)


class MSDBrainAdapter:
    """Medical Segmentation Decathlon Task01_BrainTumour: 4D NIfTI [X,Y,Z,4], modality order from dataset.json."""

    name = "msd_brain"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        d = json.loads((self.root / "dataset.json").read_text())
        raw_mods = [d["modality"][str(i)] for i in range(len(d["modality"]))]
        self.raw_modalities = raw_mods
        self.modalities = [canonical_modality(m) for m in raw_mods]
        self.labels = {int(k): v for k, v in d["labels"].items()}
        self.label_map = label_map_from_names(self.labels)
        self._training = d["training"]

    def list_cases(self) -> list[CaseRecord]:
        recs = []
        for t in self._training:
            cid = Path(t["image"]).name.replace(".nii.gz", "")
            # MSD gives one identifier per case and no patient mapping; each case is treated as one patient.
            recs.append(
                CaseRecord(
                    cid, cid, [str(self.root / t["image"])], str(self.root / t["label"]), self.modalities
                )
            )
        return recs

    def load(self, rec: CaseRecord, load_label: bool = True):
        im = nib.load(rec.image_paths[0])
        arr = np.asarray(im.dataobj, dtype=np.float32)  # X,Y,Z,C
        arr = np.moveaxis(arr, -1, 0)
        lab = (
            np.asarray(nib.load(rec.label_path).dataobj).astype(np.int64)
            if (load_label and rec.label_path)
            else None
        )
        return arr, im.affine, lab, im.header


class BraTSAdapter:
    """BraTS-style layout: <root>/<case>/<case>-<suffix>.nii.gz, one file per modality (names set in config)."""

    name = "brats"

    def __init__(
        self,
        root: str | Path,
        modality_suffix: dict[str, str],
        seg_suffix: str = "seg",
        label_map: dict[int, int] | None = None,
        case_glob: str = "*",
    ):
        self.root = Path(root)
        self.suffix = {canonical_modality(k): v for k, v in modality_suffix.items()}
        self.modalities = list(CANONICAL_MODALITIES)
        self.seg_suffix = seg_suffix
        # BraTS 2021: 1 NCR, 2 ED, 4 ET.  BraTS 2023: 1 NCR, 2 ED, 3 ET.  Must be set explicitly per release.
        self.label_map = label_map or {0: 0, 1: 1, 2: 2, 3: 3}
        self.case_glob = case_glob

    def list_cases(self) -> list[CaseRecord]:
        recs = []
        for d in sorted(p for p in self.root.glob(self.case_glob) if p.is_dir()):
            paths = []
            for m in self.modalities:
                f = d / f"{d.name}-{self.suffix[m]}.nii.gz"
                if not f.exists():
                    f = d / f"{d.name}_{self.suffix[m]}.nii.gz"
                paths.append(str(f))
            seg = d / f"{d.name}-{self.seg_suffix}.nii.gz"
            if not seg.exists():
                seg = d / f"{d.name}_{self.seg_suffix}.nii.gz"
            if not all(Path(p).exists() for p in paths):
                continue
            pid = re.sub(
                r"-\d{3}$", "", d.name
            )  # BraTS-GLI-00000-000 -> BraTS-GLI-00000 (time points share a patient)
            recs.append(CaseRecord(d.name, pid, paths, str(seg) if seg.exists() else None, self.modalities))
        return recs

    def load(self, rec: CaseRecord, load_label: bool = True):
        imgs, aff0 = [], None
        for p in rec.image_paths:
            im = nib.load(p)
            if aff0 is None:
                aff0, hdr = im.affine, im.header
            elif not np.allclose(im.affine, aff0, atol=1e-3):
                raise ValueError(f"{rec.case_id}: modality affines differ")
            imgs.append(np.asarray(im.dataobj, dtype=np.float32))
        lab = (
            np.asarray(nib.load(rec.label_path).dataobj).astype(np.int64)
            if (load_label and rec.label_path)
            else None
        )
        return np.stack(imgs), aff0, lab, hdr


def get_adapter(name: str, cfg: dict):
    root = Path(cfg["root"])
    if name == "msd_brain":
        return MSDBrainAdapter(root)
    if name.startswith("brats"):
        return BraTSAdapter(
            root,
            cfg["modality_suffix"],
            cfg.get("seg_suffix", "seg"),
            {int(k): v for k, v in cfg["label_map"].items()},
        )
    raise ValueError(f"unknown dataset adapter '{name}'")


# ----------------------------------------------------------------------------- torch datasets


def load_processed(case_dir: str | Path, mmap: bool = True) -> dict:
    d = Path(case_dir)
    meta = json.loads((d / "meta.json").read_text())
    mode = "r" if mmap else None
    out = {"image": np.load(d / "image.npy", mmap_mode=mode), "meta": meta}
    if (d / "label.npy").exists():
        out["label"] = np.load(d / "label.npy", mmap_mode=mode)
    return out


def channel_indices(meta_modalities: list[str], wanted: list[str] | None) -> list[int]:
    if not wanted:
        return list(range(len(meta_modalities)))
    miss = [m for m in wanted if m not in meta_modalities]
    if miss:
        raise ValueError(f"modalities {miss} not present in processed data {meta_modalities}")
    return [meta_modalities.index(m) for m in wanted]


class PatchDataset(Dataset):
    """Random 3D patches from *training* cases only. One 'epoch' = `samples_per_epoch` patches."""

    def __init__(
        self,
        case_dirs: list,
        patch_size,
        samples_per_epoch: int,
        modalities: list[str] | None = None,
        augmenter=None,
        sampler: PatchSampler | None = None,
        seed: int = 0,
        cache_cases: int = 0,
    ):
        self.case_dirs = [Path(c) for c in case_dirs]
        self.patch = tuple(patch_size) if not isinstance(patch_size, int) else (patch_size,) * 3
        self.n = samples_per_epoch
        self.modalities = modalities
        self.augmenter = augmenter
        self.sampler = sampler or PatchSampler(self.patch)
        self.seed = seed
        self.cache_cases = cache_cases
        self._cache: dict = {}
        self.epoch = 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self) -> int:
        return self.n

    def _get_case(self, i: int) -> dict:
        if i in self._cache:
            return self._cache[i]
        c = load_processed(self.case_dirs[i], mmap=True)
        if self.cache_cases and len(self._cache) < self.cache_cases:
            c = {"image": np.asarray(c["image"]), "label": np.asarray(c["label"]), "meta": c["meta"]}
            self._cache[i] = c
        return c

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(
            [
                self.seed,
                self.epoch,
                idx,
                torch.initial_seed() % 2**31 if torch.utils.data.get_worker_info() else 0,
            ]
        )
        ci = int(rng.integers(len(self.case_dirs)))
        c = self._get_case(ci)
        img, lab = c["image"], c["label"]
        center = self.sampler.sample_center(lab, img, rng)
        chans = channel_indices(c["meta"]["modalities"], self.modalities)
        x = extract_patch(
            img[chans] if len(chans) != img.shape[0] or chans != list(range(img.shape[0])) else img,
            center,
            self.patch,
        ).astype(np.float32)
        y = extract_patch(lab, center, self.patch).astype(np.uint8)
        if self.augmenter is not None:
            x, y = self.augmenter(x, y, rng)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "label": torch.from_numpy(np.ascontiguousarray(y)).long(),
        }


class FullVolumeDataset(Dataset):
    """Whole cropped canonical volumes for sliding-window validation/testing."""

    def __init__(self, case_dirs: list, modalities: list[str] | None = None):
        self.case_dirs = [Path(c) for c in case_dirs]
        self.modalities = modalities

    def __len__(self) -> int:
        return len(self.case_dirs)

    def __getitem__(self, i: int) -> dict:
        c = load_processed(self.case_dirs[i], mmap=False)
        chans = channel_indices(c["meta"]["modalities"], self.modalities)
        return {
            "image": torch.from_numpy(c["image"][chans].astype(np.float32)),
            "label": torch.from_numpy(c["label"].astype(np.int64)) if "label" in c else None,
            "meta": c["meta"],
            "case_id": c["meta"]["case_id"],
        }
