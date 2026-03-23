# kfp-workflow

KFP v2 training pipeline and KServe serving workflow manager.

## Scope

Two-part ML workflow on Kubeflow:

1. **Training pipeline** — Compile and submit a KFP v2 pipeline that loads data from PVC, preprocesses, trains a model, evaluates, and saves weights.
2. **Inference serving** — Deploy any trained model as a KServe InferenceService (custom or standard runtime).

Both parts use file-backed registries for model and dataset management, with PVC-based storage.

## Model Plugin System

Models are integrated via a **modular plugin system**. Each model implements the `ModelPlugin` ABC (`plugins/base.py`) with one method per pipeline stage:

```
load_data → preprocess → train → evaluate → save_model → predict
```

Plugins are discovered via an explicit registry dict in `plugins/__init__.py`. To add a new model:

1. Create `plugins/my_model.py` implementing `ModelPlugin`
2. Register it in `plugins/__init__.py`'s `_build_registry()`
3. Reference it by name in pipeline spec: `model.name: my-model`

**Current plugins:**
- `mambasl-cmapss` — MambaSL state-space model for C-MAPSS turbofan RUL prediction

## Quick Start

```bash
make venv
make install
make test
make spec-validate
make compile-pipeline
```

## CLI Commands

```
kfp-workflow [--json]                                       # Global: JSON output mode

# Pipeline lifecycle
kfp-workflow pipeline compile   --spec <path> --output <path> [--set key=value ...]
kfp-workflow pipeline submit    --spec <path> [--namespace] [--host] [--user] [--existing-token] [--cookies] [--set key=value ...]

# Pipeline run monitoring
kfp-workflow pipeline run get       <run_id>   [--namespace] [--host] [--user]
kfp-workflow pipeline run list                 [--namespace] [--experiment-id] [--page-size] [--sort-by]
kfp-workflow pipeline run wait      <run_id>   [--timeout] [--namespace] [--host] [--user]
kfp-workflow pipeline run terminate <run_id>   [--namespace] [--host] [--user]
kfp-workflow pipeline run logs      <run_id>   [--step] [--namespace]

# Experiment management
kfp-workflow pipeline experiment list          [--namespace] [--page-size] [--host] [--user]

# Serving
kfp-workflow serve create       --spec <path> [--dry-run]
kfp-workflow serve delete       --name <name> [--namespace]
kfp-workflow serve list         [--namespace]
kfp-workflow serve get          --name <name> [--namespace]

# Registries
kfp-workflow registry model register    --name <n> --version <v> --uri <u> [--registry-path]
kfp-workflow registry model get         --name <n> [--version] [--registry-path]
kfp-workflow registry model list        [--registry-path]
kfp-workflow registry dataset register  --name <n> --pvc-name <p> --subpath <s> [--registry-path]
kfp-workflow registry dataset get       --name <n> [--version] [--registry-path]
kfp-workflow registry dataset list      [--registry-path]

# Infrastructure
kfp-workflow cluster bootstrap  --spec <path> [--dry-run]
kfp-workflow spec validate      --spec <path> [--type {pipeline,serving}] [--set key=value ...]
```

## Pipeline DAG

```
load_data → preprocess → train → evaluate → save_model
```

All components receive a serialised `PipelineSpec` JSON and communicate via JSON strings. PVCs for data and model weights are mounted on every task. Each component delegates to the model plugin identified by `spec.model.name`.

## Configuration

Pipeline and serving behaviour is driven by YAML specs under `configs/`:
- `configs/pipelines/` — Training pipeline specs (validated as `PipelineSpec`)
- `configs/serving/` — Serving specs (validated as `ServingSpec`)

### Spec Fields

**PipelineSpec** supports plugin-specific config via:
- `model.config` — Architecture params (e.g., d_model, d_state for MambaSL)
- `dataset.config` — Dataset params (e.g., fd_name, feature_mode for C-MAPSS)
- `train` — Generic hyperparams (batch_size, lr, epochs, patience, seed)

**ServingSpec** supports:
- `runtime: custom` — Uses custom container predictor with `predictor_image`
- `runtime: kserve-torchserve` — Standard KServe runtime with PVC storage

### CLI Config Overrides (`--set`)

Override any spec value from the command line without editing YAML files. Uses Helm-style dotted-path syntax:

```bash
# Override training hyperparams
kfp-workflow pipeline compile --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --output pipelines/experiment.yaml \
    --set train.max_epochs=100 \
    --set train.learning_rate=0.0005

# Override model architecture
kfp-workflow pipeline submit --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --set model.config.d_model=128 \
    --set model.config.d_state=32

# Switch dataset variant
kfp-workflow pipeline compile --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --output pipelines/fd003.yaml \
    --set dataset.config.fd_name=FD003
```

**Precedence:** CLI `--set` > YAML spec > plugin defaults.

Values are auto-coerced: `128` → int, `0.001` → float, `true`/`false` → bool, `[1,2,3]` → list.

## Container Image

Single base image for all pipeline components:

```bash
make docker-build
# or
./scripts/build_image.sh [image-name:tag]
```

Base: `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`
Includes: `mamba_ssm` (pre-built wheel), `mambasl-new` package (from `models/mambasl-new/`), `kserve` SDK

## Cluster Monitoring

Energy and infrastructure monitoring is deployed on the cluster:

- **Grafana** — `http://155.230.34.51:30090` (admin/admin)
- **Prometheus** — Metrics collection with 7-day retention
- **Kepler** — Per-container energy consumption via Intel RAPL (CPU) and NVML (GPU)

The Kepler dashboard in Grafana shows real-time power consumption at node, pod, and container granularity.

## Registries

- **Model Registry** — `FileModelRegistry`: JSON file on model PVC at `.model_registry.json`
- **Dataset Registry** — `PVCDatasetRegistry`: JSON file on data PVC at `.dataset_registry.json`

Both support register (upsert), get, and list operations via CLI or Python API.
