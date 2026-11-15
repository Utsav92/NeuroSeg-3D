# CPU-only research deployment (Linux). Docker on macOS does NOT expose Apple MPS to containers.
# Not built/tested in the authoring environment (no Docker available) - see docs/DEPLOYMENT.md.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd -m neuroseg
COPY requirements.txt requirements-cloud.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY neuroseg ./neuroseg
COPY app ./app
COPY configs ./configs
RUN pip install --no-deps -e . && mkdir -p /models /work && chown neuroseg /work
USER neuroseg
# A trained checkpoint MUST be mounted at /models; without one the API reports "no_model" and refuses to predict.
ENV NEUROSEG_CHECKPOINT=/models/best.pt NEUROSEG_DEVICE=cpu NEUROSEG_WORK_DIR=/work
EXPOSE 8000 8501
CMD ["uvicorn", "neuroseg.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
