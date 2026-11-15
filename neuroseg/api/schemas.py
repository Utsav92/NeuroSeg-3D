from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None = None
    message: str | None = None
    research_only: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | succeeded | failed | timeout
    created: float
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    error_type: str | None = None


class PredictParams(BaseModel):
    overlap: float = Field(0.5, ge=0.0, le=0.9)
    tta: bool = False
    mc_samples: int = Field(0, ge=0, le=30)
    return_probs: bool = False


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
