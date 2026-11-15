# Datasets

## Primary: Medical Segmentation Decathlon — Task01_BrainTumour

| Item | Value (verified 2026-11-15 unless noted) |
|---|---|
| Project site | https://medicaldecathlon.com/ |
| Registry | https://registry.opendata.aws/msd/ — buckets `s3://msd-for-monai` (us-west-2) and mirror `s3://msd-for-monai-eu` (eu-west-2) |
| Endpoint used | `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar` (HTTP 200, `Accept-Ranges: bytes`) |
| Archive size | 7,608,266,240 bytes (matched `expected_size_bytes` before download) |
| S3 ETag | `"dfd1b5a5df62dadb48cad3d347148c70-907"` — a **multipart** ETag, *not* an MD5 of the file, so it cannot be used as a checksum. `download_dataset.py` records a local SHA-256 instead (`data/manifests/msd_brain_download.json`). |
| Last-Modified | 14 Aug 2020 |
| Content | 484 labelled training cases (`imagesTr`/`labelsTr`), 266 unlabelled test images (`imagesTs`) |
| Licence | CC-BY-SA 4.0. The AWS registry text says "CC-BY-SA 4.0" but its link points at the plain CC-BY page; the `licence` string inside `dataset.json` is recorded in `data/manifests/msd_brain_verification.json`. Treat it as **share-alike + attribution**. |
| Citation | Simpson et al., *A large annotated medical image dataset for the development and evaluation of segmentation algorithms*, arXiv:1902.09063; Antonelli et al., Nat Commun 2022 |

The dataset is a subset of BraTS 2016/2017 data. It may therefore **overlap with later BraTS releases** (2021, 2023);
this repository cannot verify non-overlap without those datasets, so no external-cohort claim is made.

### Verification performed by the scripts
`download_dataset.py` probes every endpoint, checks `Content-Length` against the expected size, resumes with HTTP
Range, checks size + SHA-256 + tar index, extracts safely (path-traversal guard, `filter="data"`), keeps the original
archive untouched and makes the extracted tree read-only. `verify_dataset.py` then reads `dataset.json` and checks:
file presence, 4 modalities and their **order from the file itself** (FLAIR, T1w, t1gd, T2w → canonical flair, t1, t1ce, t2),
label names (derived mapping, never assumed integers), shape/affine agreement of every modality and mask, spacing,
orientation, and that label values in the masks are declared. Results: `data/manifests/msd_brain_verification.json`.

### Label mapping (derived from label *names* in dataset.json)
| MSD raw | name | contiguous training label |
|---|---|---|
| 0 | background | 0 |
| 1 | edema | 2 |
| 2 | non-enhancing tumor | 1 (necrotic/non-enhancing) |
| 3 | enhancing tumour | 3 |

Regions: **WT** = {1,2,3}, **TC** = {1,3}, **ET** = {3}. (MSD does not separate necrosis from non-enhancing tumor.)

### Splits
`scripts/create_splits.py --seed 42` → `data/splits/{train,validation,test}.json` (70/15/15 by patient). MSD supplies no
patient identifiers beyond the case id, so each case is treated as one patient; a duplicate-content hash is recorded during
preprocessing. Only labelled training cases are split; the unlabelled `imagesTs` are never used for Dice.

## Secondary: BraTS 2023 adult glioma (Synapse syn51156910) — manual access only
Requires Synapse registration and acceptance of the data-use terms. **Not downloaded by this repository.** After you obtain it
legitimately, place it at `data/raw/BraTS2023_GLI/training/` and use `--dataset brats2023_glioma`. The adapter
(`BraTSAdapter`) was written from the published file-naming and label conventions (1 NCR, 2 ED, 3 ET) but has **not been run
on real BraTS data** in this repository. Timepoints (`-000`, `-001`) of a patient share a split.

## Other references
* BraTS 2021 (UPenn CBICA, registration): label map `{1:1, 2:2, 4:3}`; adapter config provided, untested on real data.
* BraTS-PEDs (TCIA): pediatric domain. **Never mix with adult data** without an explicit domain-shift protocol; no config is enabled.

## Redistribution
Do not commit or redistribute raw/processed data (`.gitignore` excludes `data/`). Models trained on CC-BY-SA data may carry
share-alike obligations — review before publishing weights.
