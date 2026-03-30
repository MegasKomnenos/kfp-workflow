# kfp-workflow

KFP v2 training, benchmark, and KServe serving workflow manager.

## Scope

Two-part ML workflow on Kubeflow:

1. **Training pipeline** — Compile and submit a KFP v2 pipeline that loads data from PVC, preprocesses, trains a model, evaluates, and saves weights.
2. **Benchmark workflow** — Deploy a model as a temporary KServe `InferenceService`, replay a scenario against it, collect metrics, and persist benchmark results to PVC.
3. **Inference serving** — Deploy any trained model as a KServe InferenceService (custom or standard runtime).

Both parts use file-backed registries for model and dataset management, with PVC-based storage.

## Model Plugin System

Models are integrated via a **modular plugin system**. Each model implements the `ModelPlugin` ABC (`plugins/base.py`) with one method per pipeline stage:

```
load_data → preprocess → train → evaluate → save_model → predict
```

Plugins can optionally implement HPO hooks (`hpo_search_space`, `hpo_base_config`, `hpo_objective`) to enable hyperparameter tuning via the project-owned Optuna engine.
Katib-backed distributed HPO reuses the same plugin HPO contract by launching an internal `kfp-workflow tune trial` command inside each trial pod and collecting `objective=<value>` from stdout.

Plugins are discovered via an explicit registry dict in `plugins/__init__.py`. To add a new model:

1. Create `plugins/my_model.py` implementing `ModelPlugin`
2. Register it in `plugins/__init__.py`'s `_build_registry()`
3. Reference it by name in pipeline spec: `model.name: my-model`

**Current plugins:**
- `mambasl-cmapss` — MambaSL state-space model for C-MAPSS turbofan RUL prediction (PyTorch)
- `mrhysp-cmapss` — MR-HY-SP ensemble (MultiRocket + HYDRA + SPRocket + RidgeCV) for C-MAPSS turbofan RUL prediction (sklearn/aeon)
- `softs-cmapss` — SOFTS (STAR Aggregate-Redistribute) transformer for C-MAPSS turbofan RUL prediction (PyTorch)

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

# Benchmark lifecycle
kfp-workflow benchmark compile  --spec <path> --output <path> [--set key=value ...]
kfp-workflow benchmark submit   --spec <path> [--namespace] [--host] [--user] [--existing-token] [--cookies] [--set key=value ...]
kfp-workflow benchmark list                 [--namespace] [--page-size] [--sort-by] [--host] [--user]
kfp-workflow benchmark get      <run_id>   [--namespace] [--host] [--user]
kfp-workflow benchmark download <run_id>   [--output <path>] [--namespace] [--host] [--user]

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

# Hyperparameter tuning
kfp-workflow tune run            --spec <path> [--set key=value ...] [--data-mount-path] [--output]
kfp-workflow tune katib          --spec <path> [--set key=value ...] [--dry-run] [--output]
kfp-workflow tune show-space     --spec <path> [--set key=value ...]

# Infrastructure
kfp-workflow cluster bootstrap  --spec <path> [--type {pipeline,benchmark}] [--dry-run]
kfp-workflow spec validate      --spec <path> [--type {pipeline,serving,tune,benchmark}] [--set key=value ...]
```

## Pipeline DAG

```
load_data → preprocess → train → evaluate → save_model
```

All components receive a serialised `PipelineSpec` JSON and communicate via JSON strings. PVCs for data and model weights are mounted on every task. Each component delegates to the model plugin identified by `spec.model.name`.

## Configuration

Pipeline, benchmark, serving, and tuning behaviour is driven by YAML specs under `configs/`:
- `configs/pipelines/` — Training pipeline specs (validated as `PipelineSpec`)
- `configs/benchmarks/` — Benchmark specs, scenario refs, and metric refs (validated as `BenchmarkSpec`)
- `configs/tuning/` — HPO tuning specs (validated as `TuneSpec`)
- `configs/serving/` — Serving specs (validated as `ServingSpec`)

### Spec Fields

**PipelineSpec** supports plugin-specific config via:
- `model.config` — Architecture params (e.g., d_model, d_state for MambaSL)
- `dataset.config` — Dataset params (e.g., `fd[]`, feature_mode for C-MAPSS)
- `train` — Generic hyperparams (batch_size, lr, epochs, patience, seed)

**ServingSpec** supports:
- `runtime: custom` — Uses custom container predictor with `predictor_image`
- `runtime: kserve-torchserve` — Standard KServe runtime with PVC storage

**BenchmarkSpec** supports:
- `model` — Temporary serving target, deployed as a KServe `InferenceService`
- `scenario` — Dataset + replay pipeline, defined inline, by YAML ref, or by Python symbol
- `metrics` — One or more collectors, defined inline, by YAML ref, or by Python symbol
- `storage.results_pvc` — Dedicated PVC for benchmark result payloads

### Benchmark Runtime Model

A benchmark is a bundle of:
- `model` — The deployable inference target
- `scenario` — The dataset source plus the activity pipeline that drives requests
- `metrics` — Collectors that observe the benchmark target while the scenario runs

Built-in benchmark pieces included in this repo:

**Dataset sources:**
- `cmapss-timeseries` — All sliding-window sections from the C-MAPSS test set (for streaming/latency benchmarks)
- `cmapss-test-set` — One last-window section per test unit from the independent C-MAPSS test set, using `make_last_windows` semantics (for accuracy evaluation)

**Pipelines:**
- `sequential-replay` — Sends sections at a configurable rate (Hz) to the benchmark `InferenceService`
- `test-eval` — Sends each section once with no rate limiting; suited for test-set accuracy evaluation

**Metrics:**
- `kepler-energy` — Reads `kepler_container_joules_total` from Prometheus/Kepler for the predictor container
- `cmapss-test` — Evaluates per-unit RUL predictions against `RUL_FDxxx.txt` ground truth; applies a configurable threshold (default 30 cycles) to convert RUL regression to binary classification and returns `f1_score`, `precision`, `recall`, `accuracy`

**Shipped benchmarks:**

`configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml` — deploys `mambasl-cmapss`, replays FD001 units [1,2,3] with `max_sections: 5` at 1 Hz, collects Kepler energy.

`configs/benchmarks/mambasl_cmapss_test.yaml` — deploys `mambasl-cmapss`, evaluates the full FD001 test set (one request per unit), and returns F1/precision/recall/accuracy in `results.json`.

```bash
kfp-workflow cluster bootstrap \
  --type benchmark \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml

kfp-workflow benchmark compile \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml \
  --output pipelines/mambasl_cmapss_kepler_smoke.yaml

kfp-workflow benchmark submit \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml

kfp-workflow benchmark list
kfp-workflow benchmark get <run_id>
kfp-workflow benchmark download <run_id>
```

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
    --set dataset.config.fd[0].fd_name=FD003

# Add a second FD entry
kfp-workflow pipeline submit --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
    --set dataset.config.fd[1].fd_name=FD003
```

**Precedence:** CLI `--set` > YAML spec > plugin defaults.

### Hyperparameter Tuning

The project owns the HPO orchestration engine (Optuna-based). Plugins provide search spaces and single-trial objective functions; the engine handles study creation, parameter suggestion, and result aggregation.

**TuneSpec** extends `PipelineSpec` with an `hpo` section:
- `hpo.algorithm` — `tpe`, `random`, or `grid`
- `hpo.max_trials` — Maximum number of trials
- `hpo.builtin_profile` — `default` or `aggressive` (plugin-provided search spaces)
- `hpo.search_space` — Custom search space (overrides builtin profile)

```bash
# Preview the resolved search space
kfp-workflow tune show-space --spec configs/tuning/mambasl_cmapss_tune.yaml

# Run local HPO with Optuna
kfp-workflow tune run --spec configs/tuning/mambasl_cmapss_tune.yaml \
    --set hpo.algorithm=tpe --set hpo.max_trials=20

# Generate Katib manifest for distributed HPO
kfp-workflow tune katib --spec configs/tuning/mambasl_cmapss_tune.yaml --dry-run
```

`kfp-workflow tune katib` renders a Katib `Experiment` whose trial Job mounts the workflow PVCs, calls the hidden shared `tune trial` entrypoint, and reports the objective through Katib's `StdOut` collector.

Values are auto-coerced: `128` → int, `0.001` → float, `true`/`false` → bool, `[1,2,3]` → list.

## Container Image

Single base image for all pipeline components:

```bash
make docker-build
# or
./scripts/build_image.sh [image-name:tag]
```

Base: `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`
Includes: `mamba_ssm` (pre-built wheel), `mambasl-new` (from `models/mambasl-new/`), `multirocket-new` (from `models/multirocket-new/`), `kserve` SDK

## Cluster Monitoring

Energy and infrastructure monitoring is deployed on the cluster:

- **Grafana** — `http://155.230.34.51:30090` (admin/admin)
- **Prometheus** — Metrics collection with 7-day retention
- **Kepler** — Per-container energy consumption via Intel RAPL (CPU) and NVML (GPU)

The Kepler dashboard in Grafana shows real-time power consumption at node, pod, and container granularity.
The benchmark workflow queries Kepler metrics through Prometheus and persists the benchmark result payload to the benchmark PVC.

## Registries

- **Model Registry** — `FileModelRegistry`: JSON file on model PVC at `.model_registry.json`
- **Dataset Registry** — `PVCDatasetRegistry`: JSON file on data PVC at `.dataset_registry.json`

Both support register (upsert), get, and list operations via CLI or Python API.
