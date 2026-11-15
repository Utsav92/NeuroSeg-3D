#!/usr/bin/env python
"""Explainer video built from this project's REAL outputs (scan, prediction, curves, metrics) + macOS TTS narration."""
import _bootstrap  # noqa: F401
import json, shutil, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
import pandas as pd
from skimage.measure import marching_cubes
from neuroseg.data.preprocessing import remap_labels
from neuroseg.evaluation.visualization import LABEL_CMAP
from neuroseg.inference.predictor import Predictor, find_modality_files
from neuroseg.utils import geometry as G
import nibabel as nib

OUT = Path("outputs/video"); shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
BG, FG, ACC = "#0e1117", "#f2f2f2", "#4cc9f0"
W, H = 12.8, 7.2

# ---------------------------------------------------------------- real data
case = sorted(Path("sample_mri").iterdir())[0]
P = Predictor("outputs/checkpoints/resunet3d_msd/best.pt")
vols = P.load_volumes(find_modality_files(case, P.modalities))
res = P.predict(vols)
gt = remap_labels(np.asarray(nib.load(str(next(case.glob("*_GT_msd_labels.nii.gz")))).dataobj).astype(np.int64), {0: 0, 1: 2, 2: 1, 3: 3})
C = lambda x: G.to_canonical(x, res.affine)[0]
imgs = {m: C(vols[m][0]) for m in P.modalities}
pred, gtc = C(res.labels), C(gt)
sp = G.voxel_spacing(G.to_canonical(res.labels, res.affine)[1])
z0 = int(np.argmax(((pred > 0) | (gtc > 0)).sum((0, 1))))
hist = pd.read_csv("outputs/checkpoints/resunet3d_msd/metrics.csv")
tests = pd.read_csv("outputs/metrics/resunet3d_msd/test_summary.csv")
vals = pd.read_csv("outputs/metrics/resunet3d_msd/validation_summary.csv")
rec = json.load(open("outputs/checkpoints/resunet3d_msd/run_record.json"))
sl = lambda v, z: np.rot90(v[:, :, z])
asp = sp[1] / sp[0]
def dice(df, r):
    d = df[(df.region == r) & (df.metric == "dice")].iloc[0]; return d["mean"], d.ci_lo, d.ci_hi

def canvas(title, sub=None):
    fig = plt.figure(figsize=(W, H), dpi=100, facecolor=BG)
    fig.text(0.05, 0.92, title, color=FG, fontsize=30, weight="bold", va="center")
    if sub: fig.text(0.05, 0.855, sub, color=ACC, fontsize=16, va="center")
    return fig
def ax_(fig, rect, off=True):
    a = fig.add_axes(rect, facecolor=BG)
    if off: a.set_axis_off()
    else:
        for s in a.spines.values(): s.set_color("#666")
        a.tick_params(colors=FG)
    return a
import textwrap
def txt(fig, x, y, s, size=17, color=FG, **k):
    s = "\n".join(textwrap.fill(l, int(1500 / size), subsequent_indent="    ") if l.strip() else l for l in s.split("\n"))
    fig.text(x, y, s, color=color, fontsize=size, va="top", linespacing=1.5, **k)

scenes = []  # (name, narration, [fig,...])
def add(name, narr, figs): scenes.append((name, narr, figs))

# 1 title
f = canvas("NeuroSeg-3D", "How a deep-learning model outlines brain tumours in MRI")
txt(f, 0.05, 0.7, "A research project: 3D U-Net networks trained from scratch\non the Medical Segmentation Decathlon brain-tumour dataset.", 22)
a = ax_(f, [0.55, 0.08, 0.4, 0.5]); a.imshow(sl(imgs["t1ce"], z0), cmap="gray", aspect=asp); a.imshow(sl(pred, z0), cmap=LABEL_CMAP, vmin=0, vmax=3, alpha=0.5, aspect=asp, interpolation="nearest")
f.text(0.05, 0.08, "RESEARCH USE ONLY. Not a medical device. Not a diagnostic tool.", color="#ffd166", fontsize=15)
add("title", "This is NeuroSeg-3D, a research project that teaches a computer to outline brain tumours in M R I scans. Everything you will see comes from this project's own scan, model, and measured results. It is for research only. It is not a medical device, and it does not diagnose anyone.", [f])

# 2 MRI modalities
f = canvas("Step 1 - The input: four MRI views of one brain", "Each patient scan has four co-registered 3D volumes")
for i, (m, name) in enumerate(zip(P.modalities, ["FLAIR: shows swelling (edema)", "T1: basic anatomy", "T1 + contrast: bright where the tumour is active", "T2: fluid and swelling"])):
    a = ax_(f, [0.03 + i * 0.24, 0.2, 0.22, 0.5]); a.imshow(sl(imgs[m], z0), cmap="gray", aspect=asp); a.set_title(name, color=FG, fontsize=11, wrap=True)
txt(f, 0.05, 0.14, f"A volume is a 3D grid of numbers: {imgs['t1ce'].shape[0]} x {imgs['t1ce'].shape[1]} x {imgs['t1ce'].shape[2]} voxels, each 1 millimetre wide. The network reads all four at once as four input channels.", 15)
add("mri", "An M R I scan is not one picture. It is a three dimensional grid of numbers, and each patient has four of them: FLAIR, T1, T1 with contrast, and T2. Each highlights different tissue. The network looks at all four at the same time, as four input channels.", [f])

# 3 what is deep learning (network + learning loop)
f = canvas("Step 2 - What is deep learning?", "A function with millions of adjustable numbers, tuned from examples")
a = ax_(f, [0.03, 0.15, 0.55, 0.65]); a.set_xlim(0, 10); a.set_ylim(0, 6)
rng = np.random.default_rng(0); layers = [4, 6, 6, 4]; xs = [1, 3.3, 5.6, 8]
pos = [[(x, 3 + (j - (n - 1) / 2) * 0.9) for j in range(n)] for x, n in zip(xs, layers)]
for l in range(3):
    for p in pos[l]:
        for q in pos[l + 1]: a.plot([p[0], q[0]], [p[1], q[1]], color="#335", lw=0.6, alpha=0.8)
for l, ps in enumerate(pos):
    for p in ps: a.add_patch(mp.Circle(p, 0.22, color=[ACC, "#7b9", "#7b9", "#f94144"][l]))
for x, s in zip(xs, ["input\n(MRI voxels)", "layer", "layer", "output\n(tissue class)"]): a.text(x, 0.4, s, color=FG, ha="center", fontsize=11)
txt(f, 0.62, 0.78, "Each line is a weight (a number).\n\n1. Show the network a scan\n2. It guesses a label for every voxel\n3. Compare with the expert outline\n    -> the 'loss' (how wrong)\n4. Nudge all weights to reduce the loss\n5. Repeat thousands of times\n\nNo one writes rules for 'what a tumour\nlooks like' - the network finds them.", 16)
add("dl", "Deep learning is a function with millions of adjustable numbers, called weights. We show it a scan, it guesses a label for every voxel, and we compare the guess with an expert's outline. The difference is the loss. We nudge every weight a little to reduce the loss, and repeat thousands of times. Nobody writes rules for what a tumour looks like. The network discovers them from examples.", [f])

# 4 U-Net
f = canvas("Step 3 - The 3D U-Net", f"{rec['n_params']/1e6:.2f} million weights; random start, no pretrained weights")
a = ax_(f, [0.03, 0.1, 0.94, 0.7]); a.set_xlim(0, 14); a.set_ylim(0, 8)
widths = [16, 32, 64, 128, 256]; cols = ["#4cc9f0", "#4895ef", "#4361ee", "#3f37c9", "#7209b7"]
bx = [(1 + i * 1.5, 7 - i * 1.4) for i in range(4)]
for i, (x, y) in enumerate(bx):
    a.add_patch(mp.FancyBboxPatch((x, y), 1.0, 0.9 + (3 - i) * 0.15, boxstyle="round,pad=0.05", color=cols[i])); a.text(x + 0.5, y + 0.45, str(widths[i]), ha="center", color="white", fontsize=12, weight="bold")
    xr = 13 - x; a.add_patch(mp.FancyBboxPatch((xr - 0.6, y), 1.0, 0.9 + (3 - i) * 0.15, boxstyle="round,pad=0.05", color=cols[i])); a.text(xr - 0.1, y + 0.45, str(widths[i]), ha="center", color="white", fontsize=12, weight="bold")
    a.plot([x + 1.1, xr - 0.7], [y + 0.5, y + 0.5], "--", color="#ffd166", lw=1.2)
a.add_patch(mp.FancyBboxPatch((6.2, 1.3), 1.6, 0.9, boxstyle="round,pad=0.05", color=cols[4])); a.text(7, 1.75, "256", ha="center", color="white", fontsize=12, weight="bold")
a.text(1.0, 7.9, "ENCODER: shrink, see more context", color=FG, fontsize=12); a.text(8.0, 7.9, "DECODER: grow back to full detail", color=FG, fontsize=12)
a.text(3.2, 0.5, "yellow dashes = skip connections keep fine detail", color="#ffd166", fontsize=12)
txt(f, 0.05, 0.12, "Residual blocks (shortcut connections) make this version easier to train. Output: 4 numbers per voxel -> background, necrotic/non-enhancing core, edema, enhancing tumour.", 14)
add("unet", "The model is a three dimensional U-Net. The encoder side shrinks the volume so each layer sees more context, from small edges up to whole-brain structure. The decoder side grows it back to full detail, and skip connections carry fine detail across. We trained this one from random numbers. There are no pretrained weights. At every voxel it outputs four scores: background, necrotic core, swelling, and enhancing tumour.", [f])

# 5 scan window animation
fr = []
n = 40; xs_ = np.linspace(8, 180, n)
for i, x in enumerate(xs_):
    f = canvas("Step 4 - Reading the whole scan, patch by patch", "Sliding 64 x 64 x 64 window, overlapping, blended with Gaussian weights")
    a = ax_(f, [0.2, 0.05, 0.6, 0.76]); s = sl(imgs["t1ce"], z0); a.imshow(s, cmap="gray", aspect=asp)
    rows = s.shape[0]; yy = 30 + 90 * (0.5 + 0.5 * np.sin(i / 6)); a.add_patch(mp.Rectangle((x, yy), 64, 64, fill=False, ec="#ffd166", lw=3))
    fr.append(f)
add("window", "The whole scan is too big to process at once, so the network reads it through a sliding window of sixty four voxels per side. Windows overlap by half, and their predictions are blended with Gaussian weights so there are no seams. Stitched together, they give a label for every voxel in the brain.", fr)

# 6 sweep: prediction vs expert
fr = []
zs = list(range(max(z0 - 36, 5), min(z0 + 37, pred.shape[2] - 5), 2)) + list(range(min(z0 + 36, pred.shape[2] - 6), max(z0 - 36, 5), -2))
zs = zs[: len(zs)]
for z in zs:
    f = canvas("Step 5 - The result on one unseen scan", f"Case {case.name} (validation split, never used for training) - slice {z}")
    for k, (lab, t) in enumerate([(pred, "Model prediction"), (gtc, "Expert outline (ground truth)")]):
        a = ax_(f, [0.04 + k * 0.32, 0.12, 0.3, 0.64]); a.imshow(sl(imgs["t1ce"], z), cmap="gray", aspect=asp); a.imshow(sl(lab, z), cmap=LABEL_CMAP, vmin=0, vmax=3, alpha=0.5, aspect=asp, interpolation="nearest"); a.set_title(t, color=FG, fontsize=14)
    p_, g_ = sl(pred, z) > 0, sl(gtc, z) > 0
    err = np.zeros((*p_.shape, 4)); err[p_ & g_] = (0.1, 0.8, 0.2, 0.8); err[p_ & ~g_] = (0.95, 0.1, 0.1, 0.9); err[~p_ & g_] = (0.2, 0.4, 1, 0.9)
    a = ax_(f, [0.68, 0.12, 0.3, 0.64]); a.imshow(sl(imgs["t1ce"], z), cmap="gray", aspect=asp); a.imshow(err, aspect=asp, interpolation="nearest"); a.set_title("Agreement", color=FG, fontsize=14)
    f.text(0.05, 0.06, "red = necrotic core    green = edema    yellow = enhancing tumour          agreement: green match, red extra, blue missed", color=FG, fontsize=12)
    fr.append(f)
add("sweep", "Here is a scan the model never saw during training. On the left is the model's outline, in the middle the expert's outline, and on the right where they agree. Red is the necrotic core, green is swelling, and yellow is the enhancing tumour. The sweep moves through the brain slice by slice.", fr)

# 7 3D rotation
def mesh(mask):
    m = np.pad(mask.astype(np.float32), 1)
    from scipy.ndimage import gaussian_filter
    v, f_, _, _ = marching_cubes(gaussian_filter(m, 0.8), 0.5, spacing=sp, step_size=3); return v, f_
M = {"WT": (mesh(np.isin(pred, (1, 2, 3))), "#2ca02c", 0.18), "TC": (mesh(np.isin(pred, (1, 3))), "#d62728", 0.4), "ET": (mesh(pred == 3), "#ffd400", 0.9)}
fr = []
for az in np.linspace(0, 360, 48, endpoint=False):
    f = canvas("Step 6 - The tumour in 3D", "Surfaces from the predicted mask, in real millimetres (visualisation only)")
    a = f.add_axes([0.0, 0.08, 1.0, 0.8], projection="3d", facecolor=BG); a.set_axis_off()
    allv = M["WT"][0][0]; lo_, hi_ = np.percentile(allv, 3, axis=0), np.percentile(allv, 97, axis=0); mid = (lo_ + hi_) / 2; r = (hi_ - lo_).max() / 2 * 0.9
    for k, ((v, f_), c, al) in M.items():
        a.plot_trisurf(v[:, 0], v[:, 1], f_, v[:, 2], color=c, alpha=al, linewidth=0, shade=True)
    a.set_xlim(mid[0] - r, mid[0] + r); a.set_ylim(mid[1] - r, mid[1] + r); a.set_zlim(mid[2] - r, mid[2] + r); a.view_init(20, az)
    wt = res.volumes["regions"]["WT"]["volume_ml"]; f.text(0.05, 0.12, f"Predicted whole-tumour volume: {wt:.1f} mL   (expert outline: {np.isin(gtc,(1,2,3)).sum()*np.prod(sp)/1000:.1f} mL)", color=FG, fontsize=16); f.text(0.05, 0.075, "Tiny isolated specks away from the main tumour are false positives - the cause of the large surface-distance errors.", color="#ffd166", fontsize=12)
    fr.append(f)
add("3d", "Stacking the outlines gives a three dimensional shape. Because we know the physical size of every voxel, the model can report volume in millilitres. For this scan it predicted a whole tumour volume close to the expert's. This is a single scan, so it says nothing about how the tumour is changing over time. Small stray specks elsewhere in the brain are mistakes the model still makes.", fr)

# 8 training curves
f = canvas("How good is it? Training and measured results", f"{rec['epochs_run']} epochs on Apple Silicon (MPS), {rec['wall_seconds']/60:.0f} minutes, 339 training patients")
a = ax_(f, [0.07, 0.14, 0.4, 0.58], off=False); a.plot(hist.epoch, hist.train_loss, color="#f94144", label="training loss"); v = hist.dropna(subset=["val_loss"]); a.plot(v.epoch, v.val_loss, "o-", color=ACC, label="validation loss"); a.set_xlabel("epoch", color=FG); a.legend(facecolor=BG, labelcolor=FG)
a2 = ax_(f, [0.55, 0.14, 0.4, 0.58], off=False)
for r_, c in [("WT", "#2ca02c"), ("TC", "#d62728"), ("ET", "#ffd400")]: a2.plot(v.epoch, v[f"val_dice_{r_}"], "o-", color=c, label=f"{r_} Dice")
a2.set_xlabel("epoch", color=FG); a2.set_ylim(0, 1); a2.legend(facecolor=BG, labelcolor=FG, loc="lower right")
txt(f, 0.05, 0.08, "Loss falls and overlap (Dice, 1.0 = perfect) rises as the weights are tuned. WT = whole tumour, TC = tumour core, ET = enhancing tumour.", 14)
add("curves", "During training the loss falls, and the overlap score, called Dice, rises. A Dice of one would be a perfect match with the expert. After forty epochs, about an hour on this Mac, the curves were flattening.", [f])

# 9 numbers + honesty
f = canvas("Measured results and honest limits", "Held-out test set: 72 patients, evaluated once")
rows = []
for r_, n in [("WT", "Whole tumour"), ("TC", "Tumour core"), ("ET", "Enhancing")]:
    m, lo, hi = dice(tests, r_); rows.append(f"{n:<14} Dice {m:.2f}   (95% CI {lo:.2f} - {hi:.2f})")
txt(f, 0.05, 0.78, "\n".join(rows), 22, family="monospace")
txt(f, 0.05, 0.5, "What this is NOT:\n  - It does not diagnose cancer. It outlines tumours in scans already known to contain a glioma.\n  - One dataset, one short training run, one random seed.\n  - Surface errors (HD95 ~ 33-46 mm) show stray false-positive islands: not yet fixed.\n  - Never tested on other hospitals, scanners, children or treated tumours.\n  - Uncertainty maps show model doubt, not clinical risk.", 17, color="#ffd166")
add("results", "On a held out test set of seventy two patients, evaluated once, the model reaches Dice of about point eight three for the whole tumour, point seven five for the core, and point six nine for the enhancing part. These are honest research numbers, with wide error bars. This system does not detect cancer in the clinical sense. It outlines tumours in scans that already contain one. It was trained on one dataset, once, and it still makes stray errors far from the tumour. It has never been tested on other hospitals, so it must not be used for patient care.", [f])

# ---------------------------------------------------------------- render + assemble
clips = []
for si, (name, narr, figs) in enumerate(scenes):
    d = OUT / f"s{si:02d}_{name}"; d.mkdir()
    for i, fg in enumerate(figs): fg.savefig(d / f"f{i:03d}.png", facecolor=BG); plt.close(fg)
    aud = d / "a.aiff"; subprocess.run(["say", "-v", "Samantha", "-r", "165", "-o", str(aud), narr], check=True)
    dur = float([l for l in subprocess.check_output(["afinfo", str(aud)]).decode().splitlines() if "estimated duration" in l][0].split()[2]) + 0.6
    clip = OUT / f"clip{si:02d}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{len(figs)/dur}", "-i", str(d / "f%03d.png"), "-i", str(aud), "-af", "apad", "-t", f"{dur}", "-vf", "scale=1280:720,format=yuv420p", "-r", "30", "-c:v", "libx264", "-c:a", "aac", "-ar", "44100", str(clip)], check=True)
    clips.append(clip); print(name, f"{dur:.1f}s", len(figs), "frames")
(OUT / "list.txt").write_text("".join(f"file '{c.name}'\n" for c in clips))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(OUT / "list.txt"), "-c", "copy", str(OUT / "NeuroSeg3D_explainer.mp4")], check=True)
print("DONE", OUT / "NeuroSeg3D_explainer.mp4")
