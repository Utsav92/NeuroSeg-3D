"""Local-only FastAPI inference service (research use only).

Run:  uvicorn neuroseg.api.main:app --host 127.0.0.1 --port 8000
There is deliberately NO authentication: this service must only bind to localhost. Before any public exposure
you would need authentication, HTTPS, strict upload validation/scanning, retention policy and a privacy review
(docs/DEPLOYMENT.md).
"""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import neuroseg
from neuroseg import RESEARCH_ONLY_NOTICE
from neuroseg.api.model_service import ModelService, QueueFull
from neuroseg.api.schemas import HealthResponse, JobStatus
from neuroseg.inference.predictor import InputError

_NII = re.compile(r"\.nii(\.gz)?$", re.I)
CHUNK = 1 << 20


def create_app(
    checkpoint: str | None = None,
    device: str | None = None,
    work_dir: str | None = None,
    max_upload_mb: int | None = None,
    max_queue: int | None = None,
) -> FastAPI:
    ckpt = checkpoint if checkpoint is not None else os.environ.get("NEUROSEG_CHECKPOINT")
    max_mb = int(
        max_upload_mb if max_upload_mb is not None else os.environ.get("NEUROSEG_MAX_UPLOAD_MB", 800)
    )
    svc = ModelService(
        ckpt, device or os.environ.get("NEUROSEG_DEVICE", "auto"), work_dir or os.environ.get("NEUROSEG_WORK_DIR"),
        int(max_queue if max_queue is not None else os.environ.get("NEUROSEG_MAX_QUEUE", 4)), float(os.environ.get("NEUROSEG_JOB_TIMEOUT_S", 1800)), float(os.environ.get("NEUROSEG_JOB_TTL_S", 3600)),
    )  # fmt: skip

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc.startup()  # model initialised once at startup
        yield
        svc.shutdown()  # graceful: stop workers, remove temp artifacts

    app = FastAPI(
        title="NeuroSeg-3D Research API",
        version=neuroseg.__version__,
        description=RESEARCH_ONLY_NOTICE,
        lifespan=lifespan,
    )
    app.state.service = svc

    @app.exception_handler(InputError)
    async def _input_error(_, exc: InputError):
        return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": str(exc)})

    def need_model():
        if svc.predictor is None:
            raise HTTPException(
                503,
                detail=f"No trained model loaded: {svc.load_error}. Train a checkpoint first; the API never returns synthetic predictions.",
            )
        return svc.predictor

    @app.get("/health", response_model=HealthResponse)
    def health():
        loaded = svc.predictor is not None
        return HealthResponse(
            status="ok" if loaded else "no_model",
            model_loaded=loaded,
            device=str(svc.predictor.device) if loaded else None,
            message=None if loaded else svc.load_error,
            research_only=RESEARCH_ONLY_NOTICE,
        )

    @app.get("/model/info")
    def model_info():
        p = need_model()
        info = {k: v for k, v in p.info.items() if k != "training_history"}
        info["training_history_epochs"] = len(p.info.get("training_history", []))
        return info

    async def save_upload(up: UploadFile, dest: Path) -> None:
        total = 0
        with open(dest, "wb") as f:
            while chunk := await up.read(CHUNK):
                total += len(chunk)
                if total > max_mb * 2**20:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, detail=f"upload '{up.filename}' exceeds {max_mb} MB limit")
                f.write(chunk)

    @app.post("/predict", status_code=202)
    async def predict(
        flair: UploadFile | None = File(None), t1: UploadFile | None = File(None), t1ce: UploadFile | None = File(None), t2: UploadFile | None = File(None),
        overlap: float = Form(0.5), tta: bool = Form(False), mc_samples: int = Form(0), return_probs: bool = Form(False),
    ):  # fmt: skip
        p = need_model()
        if not (0.0 <= overlap <= 0.9) or not (0 <= mc_samples <= 30):
            raise HTTPException(422, detail="overlap must be in [0,0.9] and mc_samples in [0,30]")
        given = {"flair": flair, "t1": t1, "t1ce": t1ce, "t2": t2}
        missing = [m for m in p.modalities if given.get(m) is None]
        if missing:
            raise HTTPException(
                422, detail=f"missing required modalities: {missing} (model expects {p.modalities})"
            )
        for m, up in given.items():
            if up is not None and not _NII.search(up.filename or ""):
                raise HTTPException(422, detail=f"{m}: file must be .nii or .nii.gz")
        job_dir = svc.new_job_dir()
        paths = {}
        try:
            for m in p.modalities:
                dest = (
                    job_dir / f"{m}.nii.gz"
                    if (given[m].filename or "").lower().endswith(".gz")
                    else job_dir / f"{m}.nii"
                )
                await save_upload(given[m], dest)
                paths[m] = dest
            svc.validate_headers(paths)
            job = svc.submit(
                job_dir,
                paths,
                {"overlap": overlap, "tta": tta, "mc_samples": mc_samples, "return_probs": return_probs},
            )
        except QueueFull as e:
            _rm(job_dir)
            raise HTTPException(429, detail=str(e)) from e
        except Exception:
            _rm(job_dir)
            raise
        return {
            "job_id": job.job_id,
            "status": job.status,
            "status_url": f"/jobs/{job.job_id}",
            "result_url": f"/results/{job.job_id}",
        }

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    def job_status(job_id: str):
        j = svc.get(job_id)
        if j is None:
            raise HTTPException(404, detail="unknown job (it may have expired)")
        return JobStatus(
            job_id=j.job_id,
            status=j.status,
            created=j.created,
            started=j.started,
            finished=j.finished,
            error=j.error,
            error_type=j.error_type,
        )

    @app.get("/results/{job_id}")
    def results(job_id: str):
        j = svc.get(job_id)
        if j is None:
            raise HTTPException(404, detail="unknown job (it may have expired)")
        if j.status in ("queued", "running"):
            raise HTTPException(409, detail=f"job is {j.status}")
        if j.status != "succeeded":
            raise HTTPException(422 if j.error_type == "input_validation" else 500, detail=j.error)
        return {**j.result, "artifacts": {k: f"/results/{job_id}/files/{k}" for k in j.artifacts}}

    @app.get("/results/{job_id}/files/{name}")
    def result_file(job_id: str, name: str):
        j = svc.get(job_id)
        if j is None or j.status != "succeeded" or name not in j.artifacts:
            raise HTTPException(404, detail="artifact not found")
        return FileResponse(j.artifacts[name], filename=Path(j.artifacts[name]).name)

    return app


def _rm(d: Path) -> None:
    import shutil

    shutil.rmtree(d, ignore_errors=True)


app = create_app()
