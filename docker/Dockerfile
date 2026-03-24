FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

ARG MAMBA_WHEEL_URL=https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

WORKDIR /workspace

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/*

# Pin transformers to avoid mamba_ssm import incompatibility
# (mamba_ssm 2.2.2 imports GreedySearchDecoderOnlyOutput removed in transformers>=4.45)
# Install mamba_ssm from pre-built wheel (CPU fallback via selective_scan_ref)
RUN pip install --upgrade pip setuptools wheel ninja packaging \
 && pip install "transformers>=4.40,<4.45" \
 && pip install "${MAMBA_WHEEL_URL}"

# Install kfp-workflow
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY configs /workspace/configs
RUN pip install ".[serving]"

# Install mambasl-new model package
COPY models/mambasl-new/pyproject.toml models/mambasl-new/README.md /workspace/mambasl-new/
COPY models/mambasl-new/src /workspace/mambasl-new/src
RUN pip install ./mambasl-new

# Install multirocket-new model package
COPY models/multirocket-new/pyproject.toml models/multirocket-new/README.md /workspace/multirocket-new/
COPY models/multirocket-new/src /workspace/multirocket-new/src
RUN pip install ./multirocket-new

ENTRYPOINT ["python", "-m", "kfp_workflow.cli.main"]
