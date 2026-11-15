# Evaluation protocol

* Whole-volume, Gaussian-weighted sliding window (overlap 0.5), evaluated in the cropped canonical space; distances in **mm** via voxel spacing (`scipy.ndimage.distance_transform_edt(sampling=spacing)`), volumes in **mL**.
* Regions: WT {1,2,3}, TC {1,3}, ET {3}. Metrics per region: Dice, IoU, sensitivity, specificity, precision, HD95, ASSD, predicted/GT volume, absolute and relative volume error.
* HD95 = max of the two directed 95th-percentile surface distances; surfaces = mask minus its 6-connected erosion. ASSD pools both directed distance sets.
* Specificity is computed inside the crop (voxels outside the foreground crop are background in both prediction and truth), so it is optimistic relative to full-volume specificity.

## Empty-mask conventions
| GT | Pred | Dice/IoU | HD95/ASSD | sensitivity / precision |
|---|---|---|---|---|
| empty | empty | 1 | 0 | NaN (undefined) |
| non-empty | empty (or reverse) | 0 | NaN (undefined; optional fixed penalty via `empty_penalty`) | 0 / NaN |

NaNs are excluded from aggregates; each summary row reports `n_defined` and `n_patients`.

## Statistics
Aggregates use **patient-level percentile bootstrap** (2000 resamples, 95%). Voxels are never treated as independent. `paired_bootstrap_diff` compares models on the same patients. Subgroup tables are produced only when metadata exists and a group has ≥ 20 patients; smaller groups are labelled "insufficient".

## Calibration & uncertainty
Reliability (ECE/MCE/Brier on a voxel subsample, diagnostic only), predictive entropy, mutual information (MC-dropout / ensemble), and AUROC of entropy for detecting misclassified voxels. These measure the model's predictive spread, **not clinical risk**.

## What is *not* claimed
No cross-hospital generalization is claimed: MSD carries no site metadata and no external cohort has been evaluated. Robustness to missing modalities is only addressed by separately trained models (Experiment 2) and modality-dropout augmentation.
