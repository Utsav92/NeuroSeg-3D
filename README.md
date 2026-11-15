# NeuroSeg-3D

Research-only 3D brain-tumour segmentation from multimodal MRI, built from scratch in PyTorch (no pretrained
segmentation weights) and developed on an Apple-Silicon Mac mini (MPS) with an optional CUDA path.

> **Research, engineering and educational use only.** NeuroSeg-3D is not a medical device, not a diagnostic or
> clinical decision-support tool, and not a substitute for radiologist interpretation. It makes no patient-specific
> recommendations. See [docs/RESEARCH_LIMITATIONS.md](docs/RESEARCH_LIMITATIONS.md).

## Explainer video
[`docs/media/NeuroSeg3D_explainer.mp4`](docs/media/NeuroSeg3D_explainer.mp4) (~3 min, voiceover) explains the MRI input, deep learning, the 3D U-Net and the measured results on one unseen scan. It is generated from this repo's real outputs by `scripts/make_video.py`. Scan frames come from MSD Task01_BrainTumour (CC-BY-SA 4.0; Simpson et al., arXiv:1902.09063); the video is research-only and not a diagnostic demonstration.

## 1. Research problem
Segment glioma sub-regions in co-registered FLAIR / T1 / T1ce / T2 volumes: multiclass labels (background,
necrotic/non-enhancing, edema, enhancing), composite regions WT / TC / ET, physical tumour volumes (mL),
predictive uncertainty (MC-dropout, ensembles, entropy), and controlled comparisons of architectures, losses,
patch sizes, augmentation and input modalities.

## 2. Dataset provenance and licensing
Primary: **Medical Segmentation Decathlon Task01_BrainTumour** (CC-BY-SA 4.0), downloaded from the official AWS
Open Data bucket `s3://msd-for-monai`. Endpoint, size and licence checks are in [docs/DATASETS.md](docs/DATASETS.md);
`data/manifests/` records URL, size, SHA-256 and verification results after you run the scripts. BraTS 2021/2023 and
BraTS-PEDs need registration and are **manual-access only** (adapters exist but are untested on real BraTS data).
Do not redistribute data.

## 3. System architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): adapters → canonical RAS + per-volume z-score + crop (reversible) → patient-level
splits → patch training → whole-volume sliding-window validation → Predictor (NIfTI in original grid) → FastAPI + Streamlit.

## 4. Models
`unet3d` (baseline), `resunet3d` (primary research model, optional dropout + gradient checkpointing), `attention_unet3d`
(additive attention gates). Widths 16→32→64→128→256. Parameter counts at base 16 / 4 levels: U-Net 5.65 M, Attention U-Net 5.67 M
(Residual U-Net is reported by the trainer).

## 5. Label mapping
Contiguous training labels: 0 background, 1 necrotic/non-enhancing, 2 edema, 3 enhancing. MSD raw (1 edema, 2 non-enhancing,
3 enhancing) is mapped **from the label names in `dataset.json`**. WT={1,2,3}, TC={1,3}, ET={3}.

## 6. Installation (Apple Silicon)
```bash
python3.11 -m venv .venv && source .venv/bin/activate     # or: uv venv --python 3.11 .venv
pip install -r requirements-macos.txt                      # or: uv pip install -r requirements-macos.txt
pip install -e .
python scripts/benchmark_device.py --quick                 # real fwd+bwd on CPU / MPS, writes outputs/metrics/benchmark_device.json
pytest -q                                                  # runs without any MRI download (synthetic NIfTI fixtures)
```

## 7. Dataset acquisition
```bash
python scripts/download_dataset.py --dataset msd_brain    # probes endpoints, resumable, ~7.6 GB, SHA-256 + manifest, extracts read-only
python scripts/verify_dataset.py  --dataset msd_brain     # dataset.json, modality order, labels, geometry
python scripts/inspect_dataset.py --dataset msd_brain     # spacing, orientation, volumes, intensities
python scripts/create_splits.py --config configs/datasets.yaml --seed 42
```

## 8. Preprocessing
`python scripts/preprocess.py --config configs/datasets.yaml` → `data/processed/msd_brain/<case>/{image.npy (float16), label.npy, meta.json}`.
Canonical RAS orientation (lossless), per-volume per-modality robust z-score on the foreground (clip 0.5–99.5 %, background = 0,
guards for empty/zero-variance), foreground crop. `meta.json` keeps the affine/crop needed to return predictions to the original
grid exactly (tested). Optional resampling: linear for MRI, nearest for labels.

## 9. Training
Configs compose explicitly: `configs/datasets.yaml` < `--config` < `--hardware` < `--set key=value`.
```bash
python scripts/train.py --config configs/smoke_test.yaml                                           # synthetic CPU smoke test
python scripts/train.py --config configs/model_resunet.yaml --hardware configs/train_mac.yaml      # primary model
python scripts/train.py --config configs/model_unet.yaml    --hardware configs/train_mac.yaml
python scripts/train.py --config configs/model_attention.yaml --hardware configs/train_mac.yaml
python scripts/train.py --config configs/model_resunet.yaml --hardware configs/train_mac.yaml --resume
```
Checkpoints: `outputs/checkpoints/<experiment>/{best.pt,last.pt}` (the experiment name replaces the generic `outputs/checkpoints/best.pt`).
`best.pt` is chosen by **validation** mean Dice (whole-volume sliding window); the test split is never used for selection.
Tensorboard: `tensorboard --logdir outputs/checkpoints`.

## 10. Evaluation
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/resunet3d_msd --split validation
python scripts/evaluate.py --checkpoint outputs/checkpoints/resunet3d_msd --split test --confirm-final-test   # once, at the end
```
Outputs in `outputs/metrics/<experiment>/`: per-patient CSV, summary with **patient-level bootstrap CIs**, confusion matrix,
Dice box-plot, reliability diagram, prediction/error/uncertainty figures. Protocol and empty-mask conventions: [docs/EVALUATION.md](docs/EVALUATION.md).

## 11. Inference
```bash
python scripts/predict.py --checkpoint outputs/checkpoints/resunet3d_msd/best.pt --input-dir sample_mri/ --output-dir outputs/predictions/ [--mc-samples 10] [--tta] [--save-probs]
```
`sample_mri/` holds one NIfTI per modality (filenames containing `flair`, `t1`, `t1ce`, `t2`). Inputs are validated (file type, all
modalities, shapes, affines, spacing, NaN/Inf, empty volumes); invalid input is rejected, never guessed. Output NIfTI is aligned with the input grid.

## 12. Web application
```bash
export NEUROSEG_CHECKPOINT=outputs/checkpoints/resunet3d_msd/best.pt
uvicorn neuroseg.api.main:app --host 127.0.0.1 --port 8000
streamlit run app/streamlit_app.py
```
Studio: dashboard, 4-modality uploader, axial/coronal/sagittal viewer (opacity, overlays, GT contour, error map, uncertainty heatmap),
Plotly 3D surfaces in mm, performance panel (only when GT is supplied), exports (NIfTI, CSV, JSON, PNG, OBJ). No checkpoint → clear message, never fake output. Local-only; see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## 13. Reproducibility
Fixed seeds (split seed 42; training seed in config), deterministic patient-level splits with hard leakage assertions re-checked on every load,
the splits hash stored in each checkpoint and verified by `evaluate.py`, every run writes `run_record.json` with its resolved config, and
`outputs/configs/<experiment>.yaml`. GPU/MPS kernels are not bit-wise deterministic. Experiments: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) (`scripts/run_experiments.py`).

## 14. Cloud acceleration
`configs/train_cloud.yaml` (CUDA, AMP+GradScaler, 96³, batch 2) with `requirements-cloud.txt`. Same code on Linux/macOS. Untested here (no NVIDIA GPU). No paid service is required.

## 15. Results and limitations
See "Status" below for what has and has not been executed. **No performance numbers are quoted unless measured in this repository.**
Quality gates: `ruff check`, `black --check`, `mypy neuroseg` (some noisy nibabel/torch-stub error codes are disabled, see `pyproject.toml`), `pytest`.

## Status (what was actually run)
<!-- STATUS:START -->
**Executed and verified in this repository (2026-11-15, Apple Silicon, MPS):**
* 100 pytest tests pass (synthetic fixtures: geometry round-trips, leakage, losses, metrics, sliding window, API, CLI end-to-end); `ruff`, `black --check`, `mypy neuroseg` clean.
* Device benchmark (real 3D fwd+bwd): ResUNet 64³/base 16/batch 1 trains at ~0.21 s/step on MPS vs ~0.46 s on CPU, ~1.2 GB MPS memory (`outputs/metrics/benchmark_device.json`). Gradient checkpointing gave no memory saving at this size (it is implemented and tested; off in `train_mac.yaml`).
* MSD Task01_BrainTumour downloaded (7,608,266,240 bytes, SHA-256 in `data/manifests/msd_brain_download.json`), verified (484 cases, all 1 mm isotropic RAS, 240×240×155×4), split 339/73/72 by patient with leakage checks, preprocessed (no duplicate-content groups).
* **One real training run**: Residual U-Net (5.74 M params, dropout 0.1), random init, 40 epochs x 300 patches (reduced budget vs. the 100-epoch target), seed 42, 57 min on MPS. Best checkpoint chosen on the validation split only.

| Residual U-Net, 40-epoch budget | WT Dice | TC Dice | ET Dice | WT HD95 mm | TC HD95 mm | ET HD95 mm |
|---|---|---|---|---|---|---|
| Validation (73 pts, 95% patient-bootstrap CI) | 0.845 [0.817, 0.870] | 0.765 [0.721, 0.804] | 0.725 [0.673, 0.771] | 42.1 [36.1, 48.2] | 35.9 [29.0, 43.1] | 34.1 [25.7, 42.4] |
| Test, evaluated once (72 pts) | 0.832 [0.806, 0.855] | 0.746 [0.695, 0.789] | 0.689 [0.627, 0.743] | 45.7 [39.4, 52.1] | 37.0 [30.3, 43.9] | 32.7 [25.2, 40.2] |

HD95 is large because no post-processing is applied, so isolated false-positive islands far from the tumour dominate it (an observation, not yet tested). Dice is still rising slowly at epoch 40; these are short-budget, single-seed, single-dataset (MSD) results, not a tuned model. The ET test CI is wider because 2 test cases have no defined ET HD95.

**Not executed (commands provided):** U-Net and Attention U-Net training, experiments 1-6 comparisons (`scripts/run_experiments.py`), MC-dropout/ensemble calibration study, BraTS 2021/2023 (access-controlled), CUDA path, Docker build, public deployment. The 100-epoch schedule was cut to 40 epochs for session time.
<!-- STATUS:END -->
