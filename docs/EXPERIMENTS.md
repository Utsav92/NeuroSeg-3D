# Experiments

All experiments use the same seed (42), the same patient-level splits, the same preprocessing, the same
patch budget (`samples_per_epoch`) and the same epochs unless the experiment varies it. Model selection uses
the **validation** split only; the test split is evaluated once, at the end, with `--confirm-final-test`.

Generate and run (configs are written to `configs/experiments/<exp>/<variant>.yaml`):

```bash
python scripts/run_experiments.py --experiment arch_comparison  --hardware configs/train_mac.yaml --run
python scripts/run_experiments.py --experiment modality_ablation --hardware configs/train_mac.yaml --run
python scripts/run_experiments.py --experiment loss_comparison  --hardware configs/train_mac.yaml --run
python scripts/run_experiments.py --experiment patch_size       --hardware configs/train_mac.yaml --run
python scripts/run_experiments.py --experiment augmentation     --hardware configs/train_mac.yaml --run
python scripts/run_experiments.py --report
```

| # | Experiment | Variants | Notes |
|---|---|---|---|
| 1 | Architecture comparison | U-Net, Residual U-Net, Attention U-Net | dropout fixed to 0 for all three |
| 2 | MRI modality ablation | FLAIR; T1ce; T1+T1ce; all four | a **separate model is trained** for each input set |
| 3 | Loss comparison | Dice; Dice+CE; Tversky+CE | Tversky α=0.3, β=0.7 |
| 4 | Patch size | 48³, 64³, 96³ | 96³ may not fit; check `scripts/benchmark_device.py` |
| 5 | Augmentation | basic (flips + intensity scale) vs full | |
| 6 | Calibration & uncertainty | evaluation-only | `scripts/evaluate.py --mc-samples 10`, `--ensemble ...` |

## Results registry

Rows are filled automatically from `outputs/experiments/registry.json` after a run completes. "Not evaluated"
means no completed run exists. Validation Dice is the model-selection metric, not a held-out test result.

<!-- TABLE:START -->
| Experiment | Variant | Seed | Status | Best val mean Dice (validation split) | Epochs | Checkpoint |
|---|---|---|---|---|---|---|
| arch_comparison | attention_unet3d | 42 | planned | Not evaluated | - | - |
| arch_comparison | resunet3d | 42 | planned | Not evaluated | - | - |
| arch_comparison | unet3d | 42 | planned | Not evaluated | - | - |
<!-- TABLE:END -->
