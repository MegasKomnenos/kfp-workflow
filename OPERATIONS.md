# Operations

## Defaults

| Key | Value |
|-----|-------|
| Python | `>=3.9` |
| Namespace | `kubeflow-user-example-com` |
| KFP SDK | `2.15.0` |
| Container image | `kfp-workflow:latest` |
| Storage class | `local-path` |
| Base image | `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` |
| Serving runtime | `kserve-torchserve` |

## Development Workflow

### Setup
```bash
make venv && make install
```

### Validate a spec
```bash
kfp-workflow spec validate --spec configs/pipelines/sample_train.yaml
kfp-workflow spec validate --spec configs/serving/sample_serve.yaml --type serving
```

### Compile a pipeline
```bash
kfp-workflow pipeline compile --spec configs/pipelines/sample_train.yaml --output pipelines/sample_train.yaml
```

### Submit a pipeline (requires cluster)
```bash
kfp-workflow pipeline submit --spec configs/pipelines/sample_train.yaml
```

### Deploy an inference service (requires cluster)
```bash
kfp-workflow serve create --spec configs/serving/sample_serve.yaml
kfp-workflow serve create --spec configs/serving/sample_serve.yaml --dry-run
```

### Registry operations (requires cluster)
```bash
kfp-workflow registry model register --name my-model --version v1 --uri models/my-model/v1
kfp-workflow registry model list
kfp-workflow registry dataset register --name my-dataset --pvc-name dataset-store --subpath datasets/my-dataset
kfp-workflow registry dataset list
```

### Bootstrap cluster storage (requires cluster)
```bash
kfp-workflow cluster bootstrap --spec configs/pipelines/sample_train.yaml --dry-run
```

### Build Docker image
```bash
make docker-build
```

### Run tests
```bash
make test
```

## Architecture Notes

- All pipeline components use a single shared Docker image
- Components communicate via JSON-serialised strings
- PipelineSpec is passed as `spec_json` to every component for self-contained configuration
- Data PVC is mounted read-only; model PVC is mounted read-write
- Training is single-node only (no distributed/PyTorchJob)
- KServe InferenceService uses TorchServe runtime with PVC storage source
