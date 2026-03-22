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
│   │   ├── sample_train.yaml          # Example training pipeline spec
│   │   └── mambasl_cmapss_smoke.yaml  # MambaSL C-MAPSS smoke test (2 epochs, CPU)
│   └── serving/
│       ├── sample_serve.yaml          # Example KServe serving spec
│       └── mambasl_cmapss_serve.yaml  # MambaSL C-MAPSS custom predictor spec
│
├── docker/
│   └── Dockerfile                     # Base image (PyTorch + CUDA + mamba_ssm + mambasl-new)
│
├── kubeflow/
│   └── pvc/
│       ├── dataset-pvc.yaml           # Dataset PVC manifest
│       └── model-pvc.yaml             # Model weights PVC manifest
│
├── mambasl-new/                       # Sibling package — ML logic (installed in Docker image)
│   └── src/mambasl_new/
│       ├── cmapss/                    # C-MAPSS data, preprocessing, model, training
│       ├── mamba_layers/              # Mamba_TimeVariant, PositionalEmbedding
│       └── ...
│
├── pipelines/
│   └── README.md                      # Compiled YAML output (git-ignored)
│
├── scripts/
│   └── build_image.sh                 # Docker image build script
│
├── src/kfp_workflow/
│   ├── __init__.py                    # Package root, __version__
│   ├── specs.py                       # Pydantic config models (PipelineSpec, ServingSpec)
│   ├── utils.py                       # YAML/JSON I/O helpers
│   ├── cli/
│   │   └── main.py                    # Typer CLI (pipeline, serve, registry, cluster, spec)
│   ├── components/
│   │   ├── __init__.py                # Re-exports all 5 component functions
│   │   ├── load_data.py               # KFP component: load data via plugin
│   │   ├── preprocess.py              # KFP component: preprocess via plugin
│   │   ├── train.py                   # KFP component: train via plugin
│   │   ├── evaluate.py                # KFP component: evaluate via plugin
│   │   └── save_model.py              # KFP component: save weights via plugin
│   ├── pipeline/
│   │   ├── compiler.py                # Pipeline DAG assembly + KFP compilation
│   │   └── client.py                  # kfp.Client port-forward + submission
│   ├── plugins/
│   │   ├── __init__.py                # Plugin registry dict + get_plugin()
│   │   ├── base.py                    # ModelPlugin ABC + result dataclasses
│   │   └── mambasl_cmapss.py          # MambaSL C-MAPSS adapter plugin
│   ├── registry/
│   │   ├── base.py                    # ABCs: ModelRegistryBase, DatasetRegistryBase
│   │   ├── model_registry.py          # FileModelRegistry (JSON on PVC)
│   │   └── dataset_registry.py        # PVCDatasetRegistry (JSON on PVC)
│   └── serving/
│       ├── __init__.py
│       ├── kserve.py                  # KServe InferenceService management (custom + standard)
│       └── predictor.py               # Custom KServe predictor (plugin dispatch)
│
└── tests/
    ├── test_specs.py                  # Spec loading and validation
    ├── test_cli_protocol.py           # CLI command existence checks
    ├── test_pipeline_compile.py       # Pipeline compilation verification
    ├── test_registry.py               # File-backed registry CRUD tests
    └── test_plugin_system.py          # Plugin ABC, registry, _build_cfg tests
```
