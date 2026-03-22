# Project Structure

```
test/
├── .gitignore
├── CLAUDE.md                          # AI agent directives
├── Makefile                           # venv, install, test, compile, docker-build
├── OPERATIONS.md                      # Operational patterns and procedures
├── PROJECT.md                         # This file — directory inventory
├── README.md                          # Project overview and quick start
├── pyproject.toml                     # Build config, dependencies, entry points
│
├── configs/
│   ├── pipelines/
│   │   └── sample_train.yaml          # Example training pipeline spec
│   └── serving/
│       └── sample_serve.yaml          # Example KServe serving spec
│
├── docker/
│   └── Dockerfile                     # Single base image (PyTorch + CUDA)
│
├── kubeflow/
│   └── pvc/
│       ├── dataset-pvc.yaml           # Dataset PVC manifest
│       └── model-pvc.yaml             # Model weights PVC manifest
│
├── pipelines/
│   └── README.md                      # Compiled YAML output (git-ignored)
│
├── scripts/
│   └── build_image.sh                 # Docker image build script
│
├── src/kfp_workflow/
│   ├── __init__.py                    # Package root, __version__
│   ├── specs.py                       # Pydantic config models
│   ├── utils.py                       # YAML/JSON I/O helpers
│   ├── cli/
│   │   └── main.py                    # Typer CLI (pipeline, serve, registry, cluster, spec)
│   ├── components/
│   │   ├── load_data.py               # KFP component: load data from PVC
│   │   ├── preprocess.py              # KFP component: data preprocessing
│   │   ├── train.py                   # KFP component: PyTorch training
│   │   ├── evaluate.py                # KFP component: model evaluation
│   │   └── save_model.py              # KFP component: save weights to PVC
│   ├── pipeline/
│   │   ├── compiler.py                # Pipeline DAG assembly + KFP compilation
│   │   └── client.py                  # kfp.Client submission (stub)
│   ├── registry/
│   │   ├── base.py                    # ABCs: ModelRegistryBase, DatasetRegistryBase
│   │   ├── model_registry.py          # Kubeflow Model Registry client (stub)
│   │   └── dataset_registry.py        # PVC dataset registry (stub)
│   └── serving/
│       └── kserve.py                  # KServe InferenceService management
│
└── tests/
    ├── test_specs.py                  # Spec loading and validation
    ├── test_cli_protocol.py           # CLI command existence checks
    ├── test_pipeline_compile.py       # Pipeline compilation verification
    └── test_registry.py               # Registry ABC contract tests
```
