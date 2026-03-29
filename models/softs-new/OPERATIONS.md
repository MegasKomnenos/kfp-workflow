# Operations — softs-new

## Defaults

| Key | Value |
|-----|-------|
| Python | `>=3.10` |
| Entry point | `softs-new` |
| Container image | `softs-new:latest` |
| Base framework | PyTorch `>=2.4,<2.6` |
| C-MAPSS data source | `mambasl-new` (re-export) |
| Default `d_core` | `32` |
| Default `window_size` | `50` |
| Default `max_rul` | `125.0` |

## Development Workflow

### Setup
```bash
cd models/softs-new
make venv
source .venv/bin/activate
make install
```

### Run tests
```bash
make test
# or
python -m pytest tests/ -v
```

### Validate experiment spec
```bash
make spec-validate
# equivalent to:
softs-new spec validate --spec configs/experiments/fd001_smoke.yaml
```

### Compile a KFP pipeline
```bash
make compile-pipeline
# equivalent to:
mkdir -p pipelines
softs-new pipeline compile \
  --spec configs/experiments/fd001_smoke.yaml \
  --output pipelines/fd001_smoke.yaml
```

## Local Training

### Smoke run (CPU, 2 epochs, fixed params)
```bash
softs-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001 \
  --run-hpo false
```

### Full HPO run (all 4 datasets, default profile)
```bash
softs-new train run \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001
```

## Hyperparameter Tuning

### Generate Katib manifest
```bash
softs-new katib render \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001 \
  --output kubeflow/katib/fd001_default_katib.yaml
```

### Submit to Kubeflow
```bash
softs-new katib submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001
```

## Kubeflow End-to-End Deployment

```bash
# 1. Build Docker image (from repo root)
docker build -t softs-new:latest \
  -f docker/Dockerfile.softs \
  models/softs-new/

# Or use the unified image that installs all model packages
docker build -t kfp-workflow:latest -f docker/Dockerfile .
docker save kfp-workflow:latest | sudo k3s ctr images import -

# 2. Compile pipeline for the full workflow
softs-new pipeline compile \
  --spec configs/experiments/fd_all_core_default.yaml \
  --output pipelines/fd_all_default.yaml

# 3. Submit pipeline run
kfp run create \
  --experiment-name softs-cmapss \
  --pipeline-package-path pipelines/fd_all_default.yaml

# 4. Monitor run
kfp-workflow pipeline run list --namespace kubeflow-user-example-com
kfp-workflow pipeline run wait <run_id>

# 5. Serve the trained model
kfp-workflow serve create \
  --spec configs/serving/softs_cmapss_serve.yaml
```

## Key Architectural Decisions

- **C-MAPSS re-export**: `softs_new.cmapss.{constants,data,preprocess,windowing}` re-export from `mambasl_new.cmapss.*`. This avoids duplication while keeping `softs-new` importable without MambaSL model weights.
- **pred_len=1**: SOFTS is a forecasting model; RUL adaptation hardcodes `pred_len=1` and adds `nn.Linear(c_in, 1)` head.
- **use_norm=False default**: C-MAPSS data is already normalized during preprocessing; instance norm adds noise.
- **d_core tuning**: The STAR core dimension `d_core` is SOFTS-specific and must always be present in the search space. `d_core` controls the shared attention bottleneck — smaller values reduce parameters but hurt multi-condition datasets (FD002, FD004).
