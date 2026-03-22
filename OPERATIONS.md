# Operations

## Defaults

| Key | Value |
|-----|-------|
| Python | `>=3.9` (Docker image: 3.11.9) |
| Namespace | `kubeflow-user-example-com` |
| KFP SDK | `2.15.0` |
| Container image | `kfp-workflow:latest` |
| Storage class | `local-path` |
| Base image | `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` |
| Serving runtime | `custom` (plugin-based predictor) |
| mamba_ssm | `2.2.2+cu122torch2.4` (pre-built wheel) |

## Development Workflow

### Setup
```bash
make venv && make install
```

### Run tests
```bash
make test
# or
python -m pytest tests/ -v
```

### Validate a spec
```bash
kfp-workflow spec validate --spec configs/pipelines/mambasl_cmapss_smoke.yaml
kfp-workflow spec validate --spec configs/serving/mambasl_cmapss_serve.yaml --type serving
```

### Compile a pipeline
```bash
kfp-workflow pipeline compile \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --output pipelines/mambasl_cmapss_smoke.yaml
```

## Docker Build

```bash
# Build image with mamba_ssm + mambasl-new
docker build -t kfp-workflow:latest -f docker/Dockerfile .

# Import into containerd for k8s
docker save kfp-workflow:latest | sudo ctr -n k8s.io images import -
```

The Dockerfile installs:
1. `mamba_ssm` from pre-built GitHub wheel (CPU fallback via `selective_scan_ref`)
2. `kfp-workflow[serving]` (main package + kserve SDK)
3. `mambasl-new` (ML logic package, installed from local directory)

## End-to-End Deployment (MambaSL C-MAPSS Smoke Test)

### 1. Build and import Docker image
```bash
docker build -t kfp-workflow:latest -f docker/Dockerfile .
docker save kfp-workflow:latest | sudo ctr -n k8s.io images import -
```

### 2. Bootstrap cluster storage
```bash
kfp-workflow cluster bootstrap \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml

# Or preview manifests first:
kfp-workflow cluster bootstrap \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml --dry-run
```

### 3. Register dataset
```bash
kfp-workflow registry dataset register \
  --name cmapss --pvc-name dataset-store --subpath CMAPSSData
```

### 4. Compile pipeline
```bash
kfp-workflow pipeline compile \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --output pipelines/mambasl_cmapss_smoke.yaml
```

### 5. Submit pipeline
```bash
kfp-workflow pipeline submit \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml
```

### 6. Monitor pipeline runs
```bash
# List recent runs
kfp-workflow pipeline run list

# Get details of a specific run
kfp-workflow pipeline run get <run_id>

# Wait for a run to complete (blocks with spinner)
kfp-workflow pipeline run wait <run_id> --timeout 3600

# View component logs from a run
kfp-workflow pipeline run logs <run_id>

# View logs for a specific step
kfp-workflow pipeline run logs <run_id> --step train

# Terminate a running pipeline
kfp-workflow pipeline run terminate <run_id>

# List experiments
kfp-workflow pipeline experiment list

# JSON output for scripting
kfp-workflow --json pipeline run list
```

### 7. Verify pipeline completion
```bash
# Check model saved
kubectl exec -it <pod> -n kubeflow-user-example-com -- \
  ls /mnt/models/mambasl-cmapss/v1/

# Check model registry
kubectl exec -it <pod> -n kubeflow-user-example-com -- \
  cat /mnt/models/.model_registry.json
```

### 8. Deploy serving
```bash
kfp-workflow serve create \
  --spec configs/serving/mambasl_cmapss_serve.yaml

# Or dry run:
kfp-workflow serve create \
  --spec configs/serving/mambasl_cmapss_serve.yaml --dry-run
```

### 9. Test inference
```bash
# Check InferenceService status
kfp-workflow serve list
kfp-workflow serve get --name mambasl-cmapss-serving

# Send test request
curl -X POST http://<isvc-url>/v1/models/mambasl-cmapss-serving:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [<windowed_data>]}'
```

### 10. Cleanup
```bash
kfp-workflow serve delete \
  --name mambasl-cmapss-serving --namespace kubeflow-user-example-com
```

## Registry Operations

### Model registry (file-backed)
```bash
kfp-workflow registry model register \
  --name mambasl-cmapss --version v1 --uri /mnt/models/mambasl-cmapss/v1/model.pt

kfp-workflow registry model list
kfp-workflow registry model get --name mambasl-cmapss --version v1
```

### Dataset registry (file-backed)
```bash
kfp-workflow registry dataset register \
  --name cmapss --pvc-name dataset-store --subpath CMAPSSData

kfp-workflow registry dataset list
kfp-workflow registry dataset get --name cmapss
```

Custom registry paths:
```bash
kfp-workflow registry model list --registry-path /tmp/models.json
```

## Architecture Notes

- All pipeline components use a single shared Docker image
- Components communicate via JSON-serialised strings
- PipelineSpec is passed as `spec_json` to every component for self-contained configuration
- Each component delegates to a `ModelPlugin` identified by `spec["model"]["name"]`
- Data PVC is mounted read-only; model PVC is mounted read-write
- Training is single-node only (no distributed/PyTorchJob)
- Heavy data (numpy arrays) saved as `.npy` on PVC; only paths passed between stages
- Custom KServe predictor runs `kfp_workflow.serving.predictor` — loads model via plugin's `predict()` method
- Pipeline submission uses `kubectl port-forward` to KFP API, then `kfp.Client`

## Adding a New Model Plugin

1. Create `src/kfp_workflow/plugins/my_model.py` implementing `ModelPlugin` ABC
2. Implement: `name()`, `load_data()`, `preprocess()`, `train()`, `evaluate()`, `save_model()`, `predict()`
3. Add import to `_build_registry()` in `src/kfp_workflow/plugins/__init__.py`
4. Create pipeline config in `configs/pipelines/` with `model.name: my-model`
5. Create serving config in `configs/serving/` with `model_name: my-model`
6. Update Docker image to include any new dependencies
