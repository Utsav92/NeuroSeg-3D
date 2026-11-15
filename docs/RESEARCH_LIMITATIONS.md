# Research limitations and safety

NeuroSeg-3D is **not** a medical device, diagnostic tool, clinical decision-support system, treatment tool, or substitute for radiologist interpretation. It makes no patient-specific recommendations and does not infer tumour progression from a single scan.

* **Intended use**: reproducible engineering/educational research on public, de-identified, curated data.
* **Dataset limits**: MSD brain data derive from BraTS 2016/2017 — pre-operative glioma, skull-stripped, co-registered, curated multi-institutional scans. Site/scanner metadata are not provided, so subgroup and cross-site analyses are impossible here. Overlap with later BraTS releases is unverified.
* **Label uncertainty**: ground truth is expert-refined consensus with inter-rater variability; MSD merges necrosis and non-enhancing tumor into one class. Dice against these labels is not a measure of clinical correctness.
* **Domain shift**: no validation on post-treatment, pediatric, non-glioma, low-quality or differently preprocessed MRI. Inputs must already be co-registered and skull-stripped in the BraTS style; the pipeline does not do this.
* **Failure modes to expect**: small/multifocal lesions, ET over/under-segmentation, motion/artefacts, missing or swapped modalities, different intensity statistics, tumours at the volume edge.
* **Calibration**: softmax confidence, entropy and MC-dropout spread are model-output statistics; they are not validated risk estimates and can be miscalibrated.
* **Bias**: demographics of the training cohorts are not controlled or reported by MSD; performance across groups is unknown.
* **Privacy**: use only de-identified research data. Do not upload identifiable patient data to the Studio/API; exports contain no identifiers beyond what the input headers carry (headers are copied from the input NIfTI — strip them first if needed).
* **Curated-challenge ≠ clinical readiness**: results on a curated split say nothing about deployment readiness, which would require prospective multi-site validation and regulatory review.
* **Redistribution**: do not redistribute the data; weights trained on CC-BY-SA data may carry share-alike obligations.
