FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/multirocket-new

COPY pyproject.toml README.md PROJECT.md OPERATIONS.md ./
COPY configs ./configs
COPY kubeflow ./kubeflow
COPY src ./src

RUN python -m pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 && \
    pip install .

ENTRYPOINT ["python", "-m", "multirocket_new.cli.train"]

