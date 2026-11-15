"""Full-volume, physical-space evaluation of a checkpoint on a split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from neuroseg.data.dataset import FullVolumeDataset
from neuroseg.evaluation import calibration as cal
from neuroseg.evaluation import statistics as S
from neuroseg.evaluation import visualization as viz
from neuroseg.evaluation.metrics import confusion_matrix, evaluate_case
from neuroseg.inference.sliding_window import sliding_window_predict
from neuroseg.inference.uncertainty import entropy, predict_with_uncertainty
from neuroseg.utils.logging import write_json


def evaluate_split(
    predictor,
    case_dirs,
    out_dir: str | Path,
    overlap: float = 0.5,
    tta: bool = False,
    mc_samples: int = 0,
    n_boot: int = 2000,
    save_figures: int = 3,
    split_name: str = "test",
    seed: int = 0,
    log=print,
) -> dict:
    """Evaluates in the cropped canonical space (all brain voxels are inside the crop; voxels outside are
    background in both prediction and ground truth). Distances/volumes use physical spacing in mm / mL."""
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ds = FullVolumeDataset(case_dirs, predictor.modalities)
    per_patient, cm = {}, np.zeros((4, 4), np.int64)
    rel_probs, rel_labels, ent_stats = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        if s["label"] is None:
            raise ValueError(f"{s['case_id']} has no label; refusing to compute Dice on unlabeled data")
        spacing = tuple(s["meta"]["spacing_mm"])
        if mc_samples > 0 or len(predictor.models) > 1:
            u = predict_with_uncertainty(
                predictor.models, s["image"], predictor.patch, overlap, predictor.device, mc_samples, tta
            )
            probs, ent = u["mean_probs"].numpy(), u["entropy"].numpy()
        else:
            probs = sliding_window_predict(
                predictor.models[0], s["image"], predictor.patch, overlap, 1, predictor.device, tta=tta
            ).numpy()
            ent = entropy(torch.from_numpy(probs)).numpy()
        pred, gt = probs.argmax(0).astype(np.uint8), s["label"].numpy().astype(np.uint8)
        per_patient[s["case_id"]] = evaluate_case(pred, gt, spacing)
        cm += confusion_matrix(pred, gt)
        ent_stats.append({"case": s["case_id"], **cal.entropy_error_agreement(ent, pred, gt)})
        if i < 2:  # keep small subsample for reliability (memory)
            rel_probs.append(probs[:, ::2, ::2, ::2].copy())
            rel_labels.append(gt[::2, ::2, ::2].copy())
        if i < save_figures:
            img = s["image"][min(2, s["image"].shape[0] - 1)].numpy()
            viz.case_figure(img, pred, gt, ent, path=out / "figures" / f"{split_name}_{s['case_id']}.png")
        log(
            f"[{split_name}] {i + 1}/{len(ds)} {s['case_id']} WT={per_patient[s['case_id']]['WT']['dice']:.3f} TC={per_patient[s['case_id']]['TC']['dice']:.3f} ET={per_patient[s['case_id']]['ET']['dice']:.3f}"
        )
    df = S.results_to_frame(per_patient)
    summary = S.summarize(df, n_boot=n_boot, seed=seed)
    df.to_csv(out / f"{split_name}_per_patient.csv", index=False)
    summary.to_csv(out / f"{split_name}_summary.csv", index=False)
    viz.dice_boxplot(df, out / "figures" / f"{split_name}_dice_boxplot.png")
    viz.confusion_figure(cm, out / "figures" / f"{split_name}_confusion.png")
    rel = cal.reliability(np.concatenate(rel_probs, 1), np.concatenate(rel_labels, 0))
    viz.reliability_figure(rel["bins"], rel["ece"], out / "figures" / f"{split_name}_reliability.png")
    res = {
        "split": split_name,
        "n_patients": len(per_patient),
        "checkpoint": str(predictor.checkpoint_path),
        "settings": {
            "overlap": overlap,
            "tta": tta,
            "mc_samples": mc_samples,
            "ensemble": len(predictor.models),
            "n_boot": n_boot,
            "ci": "patient-level percentile bootstrap, 95%",
        },
        "summary": summary.to_dict(orient="records"),
        "confusion_matrix_voxels": cm.tolist(),
        "calibration": {k: v for k, v in rel.items() if k != "bins"}
        | {"note": "voxel-level, computed on a 2x-subsampled subset of the first 2 cases"},
        "uncertainty_vs_error": ent_stats,
        "empty_mask_convention": "both empty: Dice=1, HD95=0; one empty: Dice=0, HD95=NaN (excluded, count in n_defined)",
    }
    write_json(out / f"{split_name}_results.json", res)
    return res
