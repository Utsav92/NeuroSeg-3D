"""Model lifecycle + bounded asynchronous job execution for the local inference API."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib

from neuroseg.inference.predictor import InputError, NoCheckpointError, Predictor, save_result
from neuroseg.utils import geometry as G


class QueueFull(RuntimeError):
    pass


@dataclass
class Job:
    job_id: str
    dir: Path
    params: dict
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    error_type: str | None = None
    result: dict | None = None
    artifacts: dict = field(default_factory=dict)


class ModelService:
    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "auto",
        work_dir: str | None = None,
        max_queue: int = 4,
        timeout_s: float = 1800,
        ttl_s: float = 3600,
    ):
        self.checkpoint, self.device_pref = checkpoint, device
        self.work_dir = Path(work_dir or tempfile.mkdtemp(prefix="neuroseg_jobs_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_queue, self.timeout_s, self.ttl_s = max_queue, timeout_s, ttl_s
        self.predictor: Predictor | None = None
        self.load_error: str | None = None
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="neuroseg-infer"
        )  # one inference at a time

    # -- lifecycle
    def startup(self) -> None:
        if not self.checkpoint:
            self.load_error = "no checkpoint configured (set NEUROSEG_CHECKPOINT)"
            return
        try:
            self.predictor = Predictor(self.checkpoint, self.device_pref)
        except (NoCheckpointError, InputError, Exception) as e:  # noqa: BLE001
            self.load_error = f"{type(e).__name__}: {e}"

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(self.work_dir, ignore_errors=True)

    # -- jobs
    def validate_headers(self, paths: dict[str, Path]) -> None:
        shapes, affines, names = [], [], []
        for m, p in paths.items():
            try:
                im = nib.load(str(p))
            except Exception as e:  # noqa: BLE001
                raise InputError(f"{m}: not a readable NIfTI file ({type(e).__name__})") from e
            if len(im.shape) != 3:
                raise InputError(f"{m}: expected 3D volume, got shape {im.shape}")
            shapes.append(im.shape)
            affines.append(im.affine)
            names.append(m)
        problems = G.check_alignment(shapes, affines, names=names)
        if problems:
            raise InputError("; ".join(problems))

    def submit(self, job_dir: Path, files: dict[str, Path], params: dict) -> Job:
        self.cleanup()
        with self.lock:
            active = sum(1 for j in self.jobs.values() if j.status in ("queued", "running"))
            if active >= self.max_queue:
                raise QueueFull(f"queue full ({active}/{self.max_queue} active jobs)")
            job = Job(job_dir.name, job_dir, params)
            self.jobs[job.job_id] = job
        self.pool.submit(self._run, job, files)
        return job

    def new_job_dir(self) -> Path:
        d = self.work_dir / uuid.uuid4().hex
        d.mkdir(parents=True)
        return d

    def _run(self, job: Job, files: dict[str, Path]) -> None:
        job.status, job.started = "running", time.time()
        try:
            vols = self.predictor.load_volumes(files)
            res = self.predictor.predict(
                vols,
                overlap=job.params["overlap"],
                tta=job.params["tta"],
                mc_samples=job.params["mc_samples"],
                return_probs=job.params["return_probs"],
            )
            if time.time() - job.started > self.timeout_s:
                job.status, job.error, job.error_type = (
                    "timeout",
                    f"exceeded {self.timeout_s}s; result discarded",
                    "timeout",
                )
                return
            paths = save_result(res, job.dir / "out", "case")
            job.artifacts = {k: str(v) for k, v in paths.items()}
            job.result = {"volumes": res.volumes, "metadata": res.metadata}
            job.status = "succeeded"
        except InputError as e:
            job.status, job.error, job.error_type = "failed", str(e), "input_validation"
        except Exception as e:  # noqa: BLE001
            job.status, job.error, job.error_type = "failed", f"{type(e).__name__}: {e}", "internal"
        finally:
            job.finished = time.time()
            for p in files.values():  # uploaded inputs are not retained after inference
                try:
                    os.remove(p)
                except OSError:
                    pass

    def get(self, job_id: str) -> Job | None:
        j = self.jobs.get(job_id)
        if j and j.status == "running" and j.started and time.time() - j.started > self.timeout_s:
            j.status, j.error, j.error_type = "timeout", f"exceeded {self.timeout_s}s", "timeout"
        return j

    def cleanup(self) -> int:
        """Delete artifacts of finished jobs older than the retention period."""
        now, n = time.time(), 0
        with self.lock:
            for jid in [k for k, j in self.jobs.items() if j.finished and now - j.finished > self.ttl_s]:
                shutil.rmtree(self.jobs[jid].dir, ignore_errors=True)
                del self.jobs[jid]
                n += 1
        return n
