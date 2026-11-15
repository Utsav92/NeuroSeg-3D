# Model card — resunet3d_msd

**Research use only. Not a medical device.** See docs/RESEARCH_LIMITATIONS.md.

* Architecture: `resunet3d` (base channels 16, depth 4, dropout 0.1), random init, **no pretrained weights**
* Parameters: 5,736,308
* Inputs: ['flair', 't1', 't1ce', 't2'], patch [64, 64, 64]
* Training data: `msd_brain`, splits hash `fabf645af2a37053`, seed 42
* Device: mps; epochs run 40 (max_epochs)
* Selection metric: val_mean_dice (validation split, whole-volume sliding window); best = 0.8187760915752728
* Checkpoint: `outputs/checkpoints/resunet3d_msd/best.pt`

## Measured performance

### validation (patient-level bootstrap 95% CI)

| region | Dice | HD95 (mm) |
|---|---|---|
| WT | 0.845 [0.817, 0.870] | 42.1 [36.1, 48.2] |
| TC | 0.765 [0.721, 0.804] | 35.9 [29.0, 43.1] |
| ET | 0.725 [0.673, 0.771] | 34.1 [25.7, 42.4] |

### test (patient-level bootstrap 95% CI)

| region | Dice | HD95 (mm) |
|---|---|---|
| WT | 0.832 [0.806, 0.855] | 45.7 [39.4, 52.1] |
| TC | 0.746 [0.695, 0.789] | 37.0 [30.3, 43.9] |
| ET | 0.689 [0.627, 0.743] | 32.7 [25.2, 40.2] |

## Limitations

Curated-challenge data only; no external or multi-site validation; uncertainty is predictive spread, not clinical risk.