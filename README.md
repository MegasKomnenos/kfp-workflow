# kfp-workflow

KFP v2 training pipeline and KServe serving workflow manager.

## Scope

Two-part ML workflow on Kubeflow:

1. **Training pipeline** — Compile and submit a KFP v2 pipeline that loads data from PVC, preprocesses, trains a PyTorch model, evaluates, and saves weights.
2. **Inference serving** — Deploy any trained model as a KServe InferenceService with TorchServe runtime.

Both parts use Kubeflow Model Registry for model management and PVC-based dataset storage.

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
kfp-workflow pipeline compile   --spec <path> --output <path>
kfp-workflow pipeline submit    --spec <path> [--namespace] [--host] [--existing-token] [--cookies]
kfp-workflow serve create       --spec <path> [--dry-run]
kfp-workflow serve delete       --name <name> --namespace <ns>
kfp-workflow registry model register  --name <n> --version <v> --uri <u> [--framework] [--description]
kfp-workflow registry model get       --name <n> [--version]
kfp-workflow registry model list
kfp-workflow registry dataset register  --name <n> --pvc-name <p> --subpath <s> [--version] [--description]
kfp-workflow registry dataset get       --name <n> [--version]
kfp-workflow registry dataset list
kfp-workflow cluster bootstrap  --spec <path> [--dry-run]
kfp-workflow spec validate      --spec <path> [--type {pipeline,serving}]
```

## Implementation Status

**Implemented** (testable without a cluster):
- Pydantic specs + loaders + validators
- CLI wiring (all Typer commands with parameter declarations)
- Pipeline DAG assembly (`build_pipeline`, `compile_pipeline`)
- KServe manifest builder (`build_inference_service_manifest`)
- Registry ABCs and data models

**Stubbed** (NotImplementedError — requires cluster/ML code):
- Pipeline component bodies (load_data, preprocess, train, evaluate, save_model)
- Registry concrete implementations (KubeflowModelRegistry, PVCDatasetRegistry)
- Pipeline submission (`submit_pipeline`)
- KServe create/delete

## Pipeline DAG

```
load_data → preprocess → train → evaluate → save_model
```

All components receive a serialised `PipelineSpec` JSON and communicate via JSON strings. PVCs for data and model weights are mounted on every task.

## Configuration

Pipeline and serving behaviour is driven by YAML specs under `configs/`:
- `configs/pipelines/` — Training pipeline specs (validated as `PipelineSpec`)
- `configs/serving/` — Serving specs (validated as `ServingSpec`)

## Container Image

Single base image for all pipeline components:

```bash
make docker-build
# or
./scripts/build_image.sh [image-name:tag]
```

Base: `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`
