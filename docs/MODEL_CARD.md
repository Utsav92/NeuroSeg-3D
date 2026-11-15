# Model cards

Per-model cards are generated from real run records so they can never contain invented numbers:

```bash
python scripts/make_model_card.py --experiment resunet3d_msd     # -> docs/model_cards/resunet3d_msd.md
```

| Model | Config | Status |
|---|---|---|
| 3D U-Net | `configs/model_unet.yaml` | Not evaluated |
| 3D Residual U-Net (primary) | `configs/model_resunet.yaml` | Not evaluated |
| 3D Attention U-Net | `configs/model_attention.yaml` | Not evaluated |

**Common statements for all models:** research use only; trained from random initialisation (no pretrained segmentation weights); input = 4 co-registered, skull-stripped MRI modalities in order FLAIR, T1, T1ce, T2; output = 4 classes (background, necrotic/non-enhancing, edema, enhancing); trained and validated on MSD Task01_BrainTumour only; no external validation; not a clinical tool. See `docs/RESEARCH_LIMITATIONS.md`.
