"""High-level inference: validated 4-modality input -> label map in ORIGINAL scan coordinates + volumes + uncertainty."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from neuroseg import RESEARCH_ONLY_NOTICE
from neuroseg.data import validation as V
from neuroseg.data.dataset import canonical_modality
from neuroseg.data.preprocessing import preprocess_for_inference
from neuroseg.inference.postprocess import (
    probs_to_labels,
    region_probability_maps,
    remove_small_components,
    volume_summary,
)
from neuroseg.inference.sliding_window import sliding_window_predict
from neuroseg.inference.uncertainty import predict_with_uncertainty
from neuroseg.models import build_model
from neuroseg.training.checkpointing import load_checkpoint, resolve_checkpoint
from neuroseg.utils import geometry as G
from neuroseg.utils.device import select_device


class InputError(ValueError):
    """Invalid user-supplied imaging input (reported to the user rather than silently 'fixed')."""


class NoCheckpointError(FileNotFoundError):
    pass


def find_modality_files(directory: str | Path, required: list[str]) -> dict[str, Path]:
    """Match NIfTI files to modalities from filename tokens (flair/t2f, t1ce/t1c/t1gd, t1/t1n/t1w, t2/t2w)."""
    found: dict[str, list[Path]] = {}
    for p in sorted(Path(directory).iterdir()):
        if not re.search(r"\.nii(\.gz)?$", p.name):
            continue
        stem = re.sub(r"\.nii(\.gz)?$", "", p.name).lower()
        for tok in re.split(r"[-_.\s]+", stem):
            try:
                found.setdefault(canonical_modality(tok), []).append(p)
                break
            except ValueError:
                continue
    out = {}
    for m in required:
        c = found.get(m, [])
        if len(c) > 1:
            raise InputError(f"ambiguous files for modality '{m}': {[x.name for x in c]}")
        if c:
            out[m] = c[0]
    return out


@dataclass
class PredictionResult:
    labels: np.ndarray  # uint8 in ORIGINAL grid (contiguous label convention)
    affine: np.ndarray
    header: object
    spacing_mm: tuple
    volumes: dict
    metadata: dict
    probs: np.ndarray | None = None  # [4,...] original grid float16
    region_probs: dict | None = None
    entropy: np.ndarray | None = None  # original grid float32
    mutual_information: np.ndarray | None = None
    canonical: dict = field(default_factory=dict)  # cropped canonical-space arrays for visualisation/analysis


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto", extra_checkpoints: list | None = None):
        try:
            path = resolve_checkpoint(checkpoint)
        except FileNotFoundError as e:
            raise NoCheckpointError(
                f"{e}. A trained checkpoint is required; NeuroSeg-3D never returns synthetic or random predictions. "
                "Train one with scripts/train.py first."
            ) from e
        self.checkpoint_path = path
        self.device = select_device(device)
        self.models, self.ckpts = [], []
        for p in [path] + [resolve_checkpoint(x) for x in (extra_checkpoints or [])]:
            ck = load_checkpoint(p)
            m = build_model(ck["model_cfg"])
            try:
                m.load_state_dict(ck["model"])
            except RuntimeError as e:
                raise InputError(f"checkpoint {p} is incompatible with its declared architecture: {e}") from e
            m.to(self.device).eval()
            self.models.append(m)
            self.ckpts.append(ck)
        ck0 = self.ckpts[0]
        self.modalities: list[str] = ck0["modalities"]
        self.patch = tuple(ck0["patch_size"])
        self.train_spacing = tuple(ck0["spacing_mm"])
        self.norm_cfg = ck0.get("normalization", {}) or {}
        self.info = {
            "architecture": ck0["model_cfg"].get("arch"),
            "model_cfg": ck0["model_cfg"],
            "modalities": self.modalities,
            "patch_size": list(self.patch),
            "n_params": ck0.get("n_params"),
            "epoch": ck0.get("epoch"),
            "best_val_mean_dice": ck0.get("best"),
            "dataset": ck0.get("dataset"),
            "splits_hash": ck0.get("splits_hash"),
            "versions": ck0.get("versions"),
            "checkpoint": str(path),
            "ensemble_size": len(self.models),
            "device": str(self.device),
            "training_history": ck0.get("history", []),
            "research_only": RESEARCH_ONLY_NOTICE,
        }

    # -------------------------------------------------------------- loading
    def load_volumes(self, files: dict[str, str | Path]) -> dict:
        vols = {}
        for m, p in files.items():
            try:
                im = nib.load(str(p))
            except Exception as e:  # noqa: BLE001
                raise InputError(f"{m}: cannot read '{Path(p).name}' as NIfTI ({e})") from e
            if len(im.shape) != 3:
                raise InputError(f"{m}: expected a 3D volume, got shape {im.shape}")
            vols[m] = (np.asarray(im.dataobj, dtype=np.float32), im.affine, im.header)
        return vols

    # -------------------------------------------------------------- predict
    def predict(
        self,
        volumes: dict[str, tuple],
        overlap: float = 0.5,
        tta: bool = False,
        mc_samples: int = 0,
        return_probs: bool = False,
        allow_resample: bool = False,
        min_component_voxels: int = 0,
    ) -> PredictionResult:
        """volumes: {modality: (array3d, affine[, header])}."""
        simple = {m: (v[0], v[1]) for m, v in volumes.items()}
        problems = V.validate_inference_inputs(
            simple, self.modalities, None if allow_resample else self.train_spacing
        )
        if problems:
            raise InputError("; ".join(problems))
        ref = volumes[self.modalities[0]]
        affine, header = np.asarray(ref[1], float), (ref[2] if len(ref) > 2 else None)
        images = np.stack([volumes[m][0] for m in self.modalities])
        pre = preprocess_for_inference(
            images, affine, self.modalities, tuple(self.norm_cfg.get("clip_percentiles", (0.5, 99.5))),
            int(self.norm_cfg.get("crop_margin", 2)), self.train_spacing if allow_resample else None,
        )  # fmt: skip
        meta = G.SpatialMeta.from_dict(pre["spatial"])
        x = torch.from_numpy(pre["image"].astype(np.float32))
        unc = None
        if mc_samples > 0 or len(self.models) > 1:
            unc = predict_with_uncertainty(self.models, x, self.patch, overlap, self.device, mc_samples, tta)
            probs = unc["mean_probs"].numpy()
        else:
            probs = sliding_window_predict(
                self.models[0], x, self.patch, overlap, 1, self.device, tta=tta
            ).numpy()
            from neuroseg.inference.uncertainty import entropy

            unc = {"entropy": entropy(torch.from_numpy(probs)), "mutual_information": None, "n_samples": 1}
        labels_c = remove_small_components(probs_to_labels(probs), min_component_voxels)

        def to_orig(a, order, bg_one=False):
            canon = G.uncrop(a, meta.crop_start, meta.canonical_shape)
            if bg_one:  # outside the foreground crop the model was never run: background probability = 1
                inside = G.uncrop(np.ones(a.shape[1:], bool), meta.crop_start, meta.canonical_shape)
                canon[0] = np.where(inside, canon[0], 1).astype(canon.dtype)
            if pre["resampled"]:
                canon = G.resample_to_shape(canon, tuple(pre["canonical_pre_resample_shape"]), order).astype(
                    a.dtype
                )
            if canon.ndim == 4:  # channel-first: reorient each channel (never the class axis)
                return np.stack([G.from_canonical(c, affine) for c in canon])
            return G.from_canonical(canon, affine)

        labels_o = to_orig(labels_c, 0).astype(np.uint8)
        spacing_o = G.voxel_spacing(affine)
        vols = volume_summary(labels_o, spacing_o)
        res = PredictionResult(
            labels=labels_o, affine=affine, header=header, spacing_mm=spacing_o, volumes=vols,
            metadata={}, canonical={"image": pre["image"], "labels": labels_c, "spatial": meta, "spacing_mm": meta.spacing_mm},
        )  # fmt: skip
        res.entropy = to_orig(unc["entropy"].numpy().astype(np.float32), 1)
        if unc.get("mutual_information") is not None:
            res.mutual_information = to_orig(unc["mutual_information"].numpy().astype(np.float32), 1)
        if return_probs:
            res.probs = to_orig(probs.astype(np.float16), 1, bg_one=True)
            res.region_probs = {
                k: to_orig(v.astype(np.float16), 1) for k, v in region_probability_maps(probs).items()
            }
        res.metadata = {
            "research_only": RESEARCH_ONLY_NOTICE,
            "model": {
                k: self.info[k]
                for k in (
                    "architecture",
                    "n_params",
                    "epoch",
                    "best_val_mean_dice",
                    "dataset",
                    "checkpoint",
                    "ensemble_size",
                    "splits_hash",
                    "versions",
                )
            },
            "inference": {
                "device": str(self.device),
                "overlap": overlap,
                "tta": tta,
                "mc_dropout_samples": mc_samples,
                "n_prob_samples": unc["n_samples"],
                "gaussian_importance": True,
                "resampled_to_training_spacing": bool(pre["resampled"]),
                "min_component_voxels": min_component_voxels,
            },
            "input": {
                "modalities": self.modalities,
                "shape": list(labels_o.shape),
                "spacing_mm": list(spacing_o),
                "axcodes": list(G.axcodes(affine)),
            },
            "label_convention": {
                "0": "background",
                "1": "necrotic/non-enhancing tumor",
                "2": "edema",
                "3": "enhancing tumor",
            },
            "uncertainty_note": "Predictive entropy / mutual information describe model output spread, not validated clinical risk.",
        }
        return res

    def predict_dir(self, input_dir: str | Path, **kw) -> PredictionResult:
        files = find_modality_files(input_dir, self.modalities)
        missing = [m for m in self.modalities if m not in files]
        if missing:
            raise InputError(f"missing modality files {missing} in {input_dir}")
        return self.predict(self.load_volumes(files), **kw)


def save_result(res: PredictionResult, out_dir: str | Path, case_id: str = "case") -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    def nii(arr, name, dtype):
        p = out / f"{case_id}_{name}.nii.gz"
        img = nib.Nifti1Image(arr.astype(dtype), res.affine, res.header)
        img.set_data_dtype(dtype)
        nib.save(img, str(p))
        return p

    paths["segmentation"] = nii(res.labels, "seg", np.uint8)
    if res.entropy is not None:
        paths["entropy"] = nii(res.entropy, "entropy", np.float32)
    if res.mutual_information is not None:
        paths["mutual_information"] = nii(res.mutual_information, "mutual_information", np.float32)
    if res.probs is not None:
        probs4d = np.moveaxis(res.probs, 0, -1)
        p = out / f"{case_id}_probs.nii.gz"
        nib.save(nib.Nifti1Image(probs4d.astype(np.float32), res.affine), str(p))
        paths["probabilities"] = p
    import csv

    p = out / f"{case_id}_volumes.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "name", "voxels", "volume_ml"])
        for kind in ("regions", "classes"):
            for k, v in res.volumes[kind].items():
                w.writerow([kind, k, v["voxels"], f"{v['volume_ml']:.4f}"])
    paths["volumes_csv"] = p
    p = out / f"{case_id}_metadata.json"
    p.write_text(json.dumps({**res.metadata, "volumes": res.volumes}, indent=2, default=str))
    paths["metadata"] = p
    return paths
