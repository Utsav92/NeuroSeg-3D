# Architecture

```
raw NIfTI --adapter--> canonical RAS + per-volume z-score + foreground crop --> data/processed/<case>/{image.npy,label.npy,meta.json}
                                                                                  │ (SpatialMeta: affine, shapes, crop → reversible)
splits (patient-level, hard assertions) ──> PatchDataset (train only) ──> Trainer ──> checkpoints (best by validation Dice)
                                                       FullVolumeDataset (val/test) ──> sliding-window ──> metrics / figures
Predictor: validate 4 modalities → preprocess → sliding window (Gaussian) → [MC-dropout/ensemble] → inverse transform → NIfTI in ORIGINAL grid
FastAPI (bounded queue, 1 worker)  ·  Streamlit Studio (in-process Predictor)
```

## Networks (all from scratch, random init, PyTorch primitives only; `neuroseg/models`)
* **UNet3D** – (Conv3d–InstanceNorm–LeakyReLU)×2 blocks, MaxPool3d, ConvTranspose3d, concatenation skips, 1×1×1 head. Widths 16→32→64→128→256.
* **ResUNet3D** (primary) – residual blocks (conv-norm-act, conv-norm, + identity or 1×1 projection, act), 4 encoder levels, bottleneck, 4 decoder stages, optional `Dropout3d`, optional per-block gradient checkpointing.
* **AttentionUNet3D** – additive attention gates (Oktay et al.) modulate each encoder skip with the up-sampled decoder signal before concatenation.

Input `[B,4,D,H,W]` → logits `[B,4,D,H,W]`; softmax at inference. Spatial dims must be divisible by 16 and ≥ 32 in training (instance-norm needs >1 bottleneck voxel; enforced with a clear error).

## Losses (`training/losses.py`)
Soft Dice (per-sample/class, background excluded by default, absent classes use a larger smoothing constant so they don't dominate), cross-entropy (optional train-split class weights), generalized Dice, focal, Tversky, focal Tversky, boundary loss (physical-mm signed distance maps from labels; gradients flow through the softmax). Default `0.5·Dice + 0.5·CE`.

## Training (`training/trainer.py`)
AdamW, gradient accumulation, clipping, cosine or ReduceLROnPlateau, whole-volume sliding-window validation on validation subjects, best checkpoint by **validation mean(WT,TC,ET) Dice**, early stopping, atomic `last.pt`/`best.pt`, `--resume`, emergency `crash.pt`, CSV + TensorBoard logs, `run_record.json` (config, seed, splits hash, checkpoint reference). CUDA-only AMP + GradScaler; MPS runs fp32.

## Device selection
`select_device` only picks CUDA/MPS if a real 3D forward+backward probe (Conv3d, InstanceNorm3d, GroupNorm, ConvTranspose3d, trilinear interpolate, dropout3d) passes, otherwise CPU. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set.
