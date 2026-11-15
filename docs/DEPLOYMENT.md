# Deployment (research only)

## Mode A — Local macOS (tested)
```bash
uvicorn neuroseg.api.main:app --host 127.0.0.1 --port 8000     # needs NEUROSEG_CHECKPOINT
streamlit run app/streamlit_app.py
```
Without a checkpoint the API answers `503 No trained model loaded` and the Studio shows a clear message — it never returns synthetic or random predictions.

API: `GET /health`, `GET /model/info`, `POST /predict` (multipart fields `flair,t1,t1ce,t2` + `overlap,tta,mc_samples,return_probs`; returns 202 + job id), `GET /jobs/{id}`, `GET /results/{id}`, `GET /results/{id}/files/{artifact}`.
Controls: header-level geometry validation (422), per-file size limit (413), bounded queue (429), one inference worker, soft job timeout (result discarded; a running thread cannot be killed), TTL cleanup of artifacts, inputs deleted after inference, model loaded at startup, work dir removed at shutdown.

## Mode B — CPU Docker (Linux)
`docker compose up --build` (mount a checkpoint at `/models/best.pt`). Docker on macOS cannot expose Apple MPS to containers. **The Dockerfile has not been built or run** in the authoring environment (no Docker installed).

## Mode C — CUDA server
`pip install torch --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements-cloud.txt`, train with `--hardware configs/train_cloud.yaml` (AMP, 96³, batch 2), serve with `NEUROSEG_DEVICE=cuda`. Keep training, API and frontend as separate processes/hosts. Not tested here (no NVIDIA GPU).

## Before any public exposure (not implemented — local only)
Authentication and authorisation, HTTPS/TLS termination, malware/format scanning beyond header validation, upload and rate limits at the proxy, retention and deletion policy, audit logging, de-identification verification, and a documented security and privacy review (e.g. institutional/IRB and data-protection approval). Never expose raw clinical data or unapproved uploads through public endpoints.
