"""NeuroSeg-3D Research Studio (Streamlit). RESEARCH USE ONLY - not a medical device."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from components.viewer import AXES, render

from neuroseg import RESEARCH_ONLY_NOTICE
from neuroseg.data.preprocessing import remap_labels
from neuroseg.evaluation.metrics import evaluate_case
from neuroseg.evaluation.visualization import CLASS_COLORS, CLASS_LABELS, figure_3d, mesh_to_obj, surface_mesh
from neuroseg.inference.predictor import InputError, NoCheckpointError, Predictor, save_result
from neuroseg.utils import geometry as G

st.set_page_config(page_title="NeuroSeg-3D Research Studio", layout="wide", page_icon="🧠")
GT_CONVENTIONS = {
    "Contiguous (1 NCR/NET, 2 edema, 3 ET)": {0: 0, 1: 1, 2: 2, 3: 3},
    "MSD Task01 (1 edema, 2 non-enhancing, 3 enhancing)": {0: 0, 1: 2, 2: 1, 3: 3},
    "BraTS 2021 (1 NCR, 2 edema, 4 ET)": {0: 0, 1: 1, 2: 2, 4: 3},
}


@st.cache_resource(show_spinner="Loading checkpoint...")
def load_predictor(path: str, device: str):
    return Predictor(path, device)


def nii_bytes(arr, affine, header, dtype) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.nii.gz"
        img = nib.Nifti1Image(arr.astype(dtype), affine, header)
        img.set_data_dtype(dtype)
        nib.save(img, str(p))
        return p.read_bytes()


def to_canon(a, affine):
    return G.to_canonical(a, affine)[0]


st.title("🧠 NeuroSeg-3D Research Studio")
st.warning(
    RESEARCH_ONLY_NOTICE
    + " Predicted masks and uncertainty are research outputs and must not inform patient care. Use de-identified data only."
)

with st.sidebar:
    st.header("Model")
    ckpt = st.text_input(
        "Checkpoint (file or experiment dir)",
        os.environ.get("NEUROSEG_CHECKPOINT", "outputs/checkpoints/resunet3d_msd/best.pt"),
    )
    device = st.selectbox("Device", ["auto", "cpu", "mps", "cuda"])
    try:
        pred = load_predictor(ckpt, device)
    except (NoCheckpointError, InputError) as e:
        st.error(f"No usable trained model: {e}")
        st.info(
            "Train a checkpoint first (`python scripts/train.py ...`). This app never fabricates predictions."
        )
        st.stop()
    st.success(f"{pred.info['architecture']} on {pred.device}")
    st.header("Inference")
    overlap = st.slider("Window overlap", 0.0, 0.9, 0.5, 0.05)
    tta = st.checkbox("Flip test-time augmentation")
    mc = st.slider("MC-dropout samples", 0, 20, 0, help="needs a model trained with dropout > 0")
    want_probs = st.checkbox("Compute probability maps", value=False)
    allow_resample = st.checkbox("Allow resampling to training spacing", value=False)

tab_dash, tab_up, tab_view, tab_3d, tab_perf, tab_exp = st.tabs(
    ["Dashboard", "Upload & predict", "2D viewer", "3D tumor", "Performance", "Export"]
)

with tab_dash:
    i = pred.info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Architecture", i["architecture"])
    c2.metric("Parameters", f"{(i['n_params'] or 0):,}")
    c3.metric("Inference device", i["device"])
    c4.metric("Trained epochs", (i["epoch"] or 0) + 1)
    st.caption(
        f"Checkpoint: `{i['checkpoint']}` · dataset: `{i['dataset']}` · splits hash `{i['splits_hash']}` · best validation mean Dice {i['best_val_mean_dice']:.4f}"
        if i["best_val_mean_dice"] is not None
        else f"Checkpoint: `{i['checkpoint']}`"
    )
    st.caption(
        f"Input modalities (channel order): {i['modalities']} · training patch {i['patch_size']} · versions {i['versions']}"
    )
    h = pd.DataFrame(i["training_history"])
    if len(h):
        long = h.melt(
            id_vars="epoch", value_vars=[c for c in ("train_loss", "val_loss") if c in h], var_name="series"
        ).dropna()
        st.plotly_chart(
            px.line(long, x="epoch", y="value", color="series", title="Training / validation loss"),
            width="stretch",
        )
        dc = [c for c in ("val_dice_WT", "val_dice_TC", "val_dice_ET") if c in h]
        if dc:
            st.plotly_chart(
                px.line(
                    h.melt(id_vars="epoch", value_vars=dc).dropna(),
                    x="epoch",
                    y="value",
                    color="variable",
                    title="Validation Dice (whole-volume sliding window, validation split)",
                ),
                width="stretch",
            )
    st.info(
        "Dice values above are from the validation split used for model selection. They are not a held-out test result and not evidence of clinical performance."
    )

with tab_up:
    st.subheader("Upload the four co-registered MRI modalities")
    cols = st.columns(4)
    ups = {
        m: cols[k].file_uploader(m.upper(), type=["nii", "gz"], key=f"up_{m}")
        for k, m in enumerate(pred.modalities)
    }
    gt_up = st.file_uploader(
        "Optional ground-truth segmentation (research cases only)", type=["nii", "gz"], key="gt"
    )
    gt_conv = st.selectbox("Ground-truth label convention", list(GT_CONVENTIONS))
    if st.button("Run segmentation", type="primary", disabled=not all(ups.values())):
        try:
            with tempfile.TemporaryDirectory() as d:
                paths = {}
                for m, f in ups.items():
                    p = Path(d) / (f"{m}.nii.gz" if f.name.lower().endswith(".gz") else f"{m}.nii")
                    p.write_bytes(f.getvalue())
                    paths[m] = p
                vols = pred.load_volumes(paths)
                with st.spinner("Sliding-window inference..."):
                    res = pred.predict(
                        vols,
                        overlap=overlap,
                        tta=tta,
                        mc_samples=mc,
                        return_probs=want_probs,
                        allow_resample=allow_resample,
                    )
                gt = None
                if gt_up is not None:
                    gp = Path(d) / ("gt.nii.gz" if gt_up.name.lower().endswith(".gz") else "gt.nii")
                    gp.write_bytes(gt_up.getvalue())
                    gi = nib.load(str(gp))
                    prob = G.check_alignment(
                        [vols[pred.modalities[0]][0].shape, gi.shape[:3]],
                        [vols[pred.modalities[0]][1], gi.affine],
                        names=["MRI", "ground truth"],
                    )
                    if prob:
                        raise InputError("ground truth is not aligned with the MRI: " + "; ".join(prob))
                    gt = remap_labels(np.asarray(gi.dataobj).astype(np.int64), GT_CONVENTIONS[gt_conv])
            st.session_state["res"] = res
            st.session_state["imgs"] = {m: to_canon(vols[m][0], res.affine) for m in pred.modalities}
            st.session_state["gt"] = gt
            st.success("Segmentation complete.")
        except InputError as e:
            st.error(f"Input rejected (nothing was guessed or auto-corrected): {e}")
    elif not all(ups.values()):
        st.info("All four modalities are required; incomplete uploads are rejected rather than guessed.")

res = st.session_state.get("res")
if res is None:
    for t in (tab_view, tab_3d, tab_perf, tab_exp):
        with t:
            st.info("Run a segmentation on the *Upload & predict* tab first.")
else:
    imgs, gt = st.session_state["imgs"], st.session_state.get("gt")
    sp_c = G.voxel_spacing(G.to_canonical(res.labels, res.affine)[1])
    lab_c, ent_c = to_canon(res.labels, res.affine), (
        to_canon(res.entropy, res.affine) if res.entropy is not None else None
    )
    gt_c = to_canon(gt, res.affine) if gt is not None else None
    with tab_view:
        c1, c2, c3 = st.columns(3)
        mod = c1.selectbox("Modality", list(imgs))
        opacity = c2.slider("Overlay opacity", 0.0, 1.0, 0.45)
        layers = c3.multiselect(
            "Layers",
            ["Prediction", "Ground truth (contour)", "Uncertainty heatmap", "Error map"],
            default=["Prediction"],
        )
        planes = st.columns(3)
        for col, plane in zip(planes, AXES):
            ax = AXES[plane]
            n = imgs[mod].shape[ax]
            other = tuple(a for a in range(3) if a != ax)
            default = int(np.argmax((lab_c > 0).sum(other))) if (lab_c > 0).any() else n // 2
            idx = col.slider(f"{plane} slice", 0, n - 1, default, key=f"s_{plane}")
            fig = render(
                imgs[mod],
                sp_c,
                plane,
                idx,
                lab_c if "Prediction" in layers or "Error map" in layers else None,
                gt_c if ("Ground truth (contour)" in layers or "Error map" in layers) else None,
                ent_c if "Uncertainty heatmap" in layers else None,
                opacity,
                "Error map" in layers and gt_c is not None,
            )
            col.pyplot(fig)
            if col.button("PNG", key=f"png_{plane}"):
                b = io.BytesIO()
                fig.savefig(b, format="png", dpi=150, facecolor=fig.get_facecolor())
                st.session_state[f"png_{plane}"] = b.getvalue()
            if f"png_{plane}" in st.session_state:
                col.download_button(
                    f"Download {plane} PNG",
                    st.session_state[f"png_{plane}"],
                    f"{plane.lower()}_{mod}.png",
                    "image/png",
                    key=f"dl_{plane}",
                )
        st.caption(
            "Legend — "
            + " · ".join(f"{CLASS_LABELS[k]}: {v}" for k, v in CLASS_COLORS.items())
            + ". Uncertainty = predictive entropy (model output spread), not clinical risk. Slices are displayed in RAS orientation with physical voxel aspect ratio."
        )
    with tab_3d:
        show = st.multiselect("Regions", ["WT", "TC", "ET"], default=["WT", "TC", "ET"])
        step = st.select_slider("Mesh coarseness (marching-cubes step)", [1, 2, 3, 4], value=2)
        st.plotly_chart(figure_3d(lab_c, sp_c, tuple(show), step=step), width="stretch")
        st.caption(
            "Marching-cubes surfaces in physical millimetres, for visualisation only — not a diagnostic-grade reconstruction. Drag to rotate, scroll to zoom, click legend entries to toggle regions."
        )
        vt = pd.DataFrame(
            [
                {"region": r, "voxels": v["voxels"], "volume_mL": round(v["volume_ml"], 3)}
                for r, v in res.volumes["regions"].items()
            ]
        )
        st.dataframe(vt, hide_index=True)
        st.caption("Single-scan volumes only; no inference about tumor progression is made.")
    with tab_perf:
        if gt_c is None:
            st.info(
                "No ground truth was supplied, so no performance metrics are shown (none are invented for unlabeled uploads)."
            )
        else:
            m = evaluate_case(lab_c, gt_c, sp_c)
            st.dataframe(
                pd.DataFrame(m)
                .T[
                    [
                        "dice",
                        "iou",
                        "sensitivity",
                        "specificity",
                        "precision",
                        "hd95",
                        "assd",
                        "pred_volume_ml",
                        "gt_volume_ml",
                        "abs_volume_error_ml",
                        "rel_volume_error",
                    ]
                ]
                .round(3)
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("WT Dice", f"{m['WT']['dice']:.3f}")
            c2.metric("TC Dice", f"{m['TC']['dice']:.3f}")
            c3.metric("ET Dice", f"{m['ET']['dice']:.3f}")
            c4.metric("WT HD95 (mm)", f"{m['WT']['hd95']:.1f}")
            vol = pd.DataFrame(
                [{"region": r, "kind": k, "mL": m[r][f"{k}_volume_ml"]} for r in m for k in ("pred", "gt")]
            )
            st.plotly_chart(
                px.bar(
                    vol,
                    x="region",
                    y="mL",
                    color="kind",
                    barmode="group",
                    title="Predicted vs ground-truth volume",
                ),
                width="stretch",
            )
            idx = int(np.argmax(((lab_c > 0) | (gt_c > 0)).sum((0, 1))))
            st.pyplot(
                render(
                    imgs[list(imgs)[0]],
                    sp_c,
                    "Axial",
                    idx,
                    lab_c,
                    gt_c,
                    None,
                    0.5,
                    True,
                    "Error map (green TP, red FP, blue FN)",
                )
            )
    with tab_exp:
        with tempfile.TemporaryDirectory() as d:
            paths = save_result(res, d, "case")
            c1, c2, c3 = st.columns(3)
            c1.download_button(
                "Segmentation NIfTI", paths["segmentation"].read_bytes(), "segmentation.nii.gz"
            )
            c2.download_button(
                "Region volumes CSV", paths["volumes_csv"].read_bytes(), "volumes.csv", "text/csv"
            )
            c3.download_button(
                "Model + inference metadata JSON",
                paths["metadata"].read_bytes(),
                "metadata.json",
                "application/json",
            )
            if "entropy" in paths:
                st.download_button(
                    "Uncertainty (entropy) NIfTI", paths["entropy"].read_bytes(), "entropy.nii.gz"
                )
        r = st.selectbox("Surface mesh region (OBJ, mm coordinates)", ["WT", "TC", "ET"])
        from neuroseg.evaluation.metrics import region_masks

        mesh = surface_mesh(region_masks(lab_c)[r], sp_c)
        if mesh is not None:
            st.download_button(f"{r} surface (.obj)", mesh_to_obj(*mesh), f"{r}.obj")
        st.caption(
            "Exports contain research outputs only. Use de-identified data; do not upload identifiable patient data."
        )
