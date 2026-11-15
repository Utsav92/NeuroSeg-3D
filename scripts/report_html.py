#!/usr/bin/env python
"""One-scan HTML report: prediction vs ground truth (research use only)."""
import _bootstrap  # noqa: F401
import argparse, base64, io
from pathlib import Path
import nibabel as nib, numpy as np
import matplotlib.pyplot as plt
from neuroseg import RESEARCH_ONLY_NOTICE
from neuroseg.data.preprocessing import remap_labels
from neuroseg.evaluation.metrics import evaluate_case
from neuroseg.evaluation.visualization import case_figure, figure_3d
from neuroseg.inference.predictor import Predictor, find_modality_files
from neuroseg.utils import geometry as G

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True); ap.add_argument("--case-dir", required=True)
ap.add_argument("--out", default="outputs/predictions/report.html"); ap.add_argument("--mc", type=int, default=5)
a = ap.parse_args()
p = Predictor(a.checkpoint)
files = find_modality_files(a.case_dir, p.modalities)
vols = p.load_volumes(files)
res = p.predict(vols, mc_samples=a.mc)
gtp = next(Path(a.case_dir).glob("*_GT_msd_labels.nii.gz"))
gt = remap_labels(np.asarray(nib.load(str(gtp)).dataobj).astype(np.int64), {0: 0, 1: 2, 2: 1, 3: 3})
C = lambda x: G.to_canonical(x, res.affine)[0]
sp = G.voxel_spacing(G.to_canonical(res.labels, res.affine)[1])
pred, gtc, ent, img = C(res.labels), C(gt), C(res.entropy), C(vols["t1ce"][0])
m = evaluate_case(pred, gtc, sp)
def b64(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=110); plt.close(fig); return base64.b64encode(b.getvalue()).decode()
imgs = []
for ax in (2, 1, 0):
    other = tuple(i for i in range(3) if i != ax)
    idx = int(np.argmax(((pred > 0) | (gtc > 0)).sum(other)))
    imgs.append(b64(case_figure(img, pred, gtc, ent, axis=ax, idx=idx)))
rows = "".join(f"<tr><td>{r}</td>" + "".join(f"<td>{m[r][k]:.3f}</td>" for k in ("dice","iou","sensitivity","precision","hd95","assd","pred_volume_ml","gt_volume_ml")) + "</tr>" for r in m)
html = f"""<!doctype html><meta charset=utf-8><title>NeuroSeg-3D report</title>
<body style="font-family:system-ui;max-width:1100px;margin:2em auto;background:#fafafa">
<h1>NeuroSeg-3D - single-scan check</h1><p style="background:#fff3cd;padding:8px"><b>{RESEARCH_ONLY_NOTICE}</b></p>
<p>Case: <code>{Path(a.case_dir).name}</code> (validation split) | model {p.info['architecture']} | device {p.device} | MC-dropout samples {a.mc}</p>
<table border=1 cellpadding=6 style="border-collapse:collapse"><tr><th>region</th><th>Dice</th><th>IoU</th><th>Sens</th><th>Prec</th><th>HD95 mm</th><th>ASSD mm</th><th>pred mL</th><th>GT mL</th></tr>{rows}</table>
<p>Panels: T1ce | prediction | ground truth | error map (green TP, red FP, blue FN) | predictive entropy. Axial, coronal, sagittal.</p>
{''.join(f'<img style="width:100%" src="data:image/png;base64,{i}">' for i in imgs)}
<h2>3D surfaces (mm)</h2>{figure_3d(pred, sp).to_html(full_html=False, include_plotlyjs='cdn')}
<p style="color:#666">Entropy is predictive spread, not clinical risk. Single scan, no progression inference.</p></body>"""
Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(html)
print(a.out); print({r: {k: round(m[r][k], 3) for k in ("dice", "hd95", "pred_volume_ml", "gt_volume_ml")} for r in m})
