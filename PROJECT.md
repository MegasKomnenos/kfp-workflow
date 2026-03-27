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
│   │   ├── mambasl_cmapss_smoke.yaml  # MambaSL C-MAPSS smoke test (2 epochs, CPU)
│   │   ├── mrhysp_cmapss_smoke.yaml   # MR-HY-SP C-MAPSS smoke test (minimal kernels, CPU)
│   │   └── softs_cmapss_smoke.yaml    # SOFTS C-MAPSS smoke test (2 epochs, CPU, d_core=16)
│   ├── tuning/
│   │   ├── mambasl_cmapss_tune.yaml   # MambaSL C-MAPSS HPO tuning spec
│   │   ├── mrhysp_cmapss_tune.yaml    # MR-HY-SP C-MAPSS HPO tuning spec
│   │   └── softs_cmapss_tune.yaml     # SOFTS C-MAPSS HPO tuning spec (tpe, 20 trials)
│   └── serving/
│       ├── sample_serve.yaml          # Example KServe serving spec
│       ├── mambasl_cmapss_serve.yaml  # MambaSL C-MAPSS custom predictor spec
│       ├── mrhysp_cmapss_serve.yaml   # MR-HY-SP C-MAPSS custom predictor spec
│       └── softs_cmapss_serve.yaml    # SOFTS C-MAPSS custom predictor spec
│
├── docker/
│   └── Dockerfile                     # Base image (PyTorch + CUDA + mamba_ssm + mambasl-new + multirocket-new + softs-new)
│
├── examples/                          # Korean-language tutorials and usage guides
│   ├── README.md                      # Table of contents and learning order
│   ├── 00_프로젝트_개요.md            # Project overview, philosophy, architecture
│   ├── 01_설치_및_설정.md             # Installation and environment setup
│   ├── 02_스펙_파일_작성_및_검증.md   # Pipeline/serving spec authoring and validation
│   ├── 03_파이프라인_컴파일_및_제출.md # Pipeline compile, submit, and DAG overview
│   ├── 04_파이프라인_실행_모니터링.md  # Run get/list/wait/terminate/logs
│   ├── 05_CLI_설정_오버라이드.md       # --set flag, type coercion, plugin schemas
│   ├── 06_서빙_배포_및_추론.md        # KServe InferenceService create/list/get/delete
│   ├── 07_레지스트리_관리.md          # Model and dataset registry management
│   ├── 08_클러스터_부트스트랩.md       # PVC provisioning via cluster bootstrap
│   ├── 09_Docker_이미지_빌드.md       # Docker image build and optimization
│   ├── 10_새_모델_플러그인_개발.md    # ModelPlugin ABC full implementation guide (architecture, stages, serving, HPO, testing)
│   └── 11_하이퍼파라미터_튜닝.md     # Optuna/Katib HPO execution and management
│
├── kubeflow/
│   └── pvc/
│       ├── dataset-pvc.yaml           # Dataset PVC manifest
│       └── model-pvc.yaml             # Model weights PVC manifest
│
├── models/
│   ├── mambasl-new/                   # MambaSL model package (installed in Docker image)
│   │   └── src/mambasl_new/
│   │       ├── cmapss/                # C-MAPSS data, preprocessing, model, training
│   │       ├── mamba_layers/          # Mamba_TimeVariant, PositionalEmbedding
│   │       ├── kubeflow/              # Katib manifest construction, HPO pipeline
│   │       └── cli/                   # Container-internal CLI (train, katib-trial)
│   ├── multirocket-new/              # MR-HY-SP model package (installed in Docker image)
│   │   └── src/multirocket_new/
│   │       ├── cmapss.py              # C-MAPSS data loading, windowing, scaling
│   │       ├── model.py               # MRHySPRegressor (HYDRA + MultiRocket + SPRocket + RidgeCV)
│   │       ├── runner.py              # Experiment runner, metrics, batch prediction
│   │       └── config.py              # ExperimentConfig, mr_num_kernels constraint
│   └── softs-new/                    # SOFTS model package (installed in Docker image)
│       ├── configs/
│       │   ├── experiments/           # fd001_smoke, fd_all_core_{default,aggressive}.yaml
│       │   └── search_spaces/         # default.yaml, aggressive.yaml (13 SOFTS params)
│       └── src/softs_new/
│           ├── softs_layers/          # Embed (DataEmbedding_inverted), Transformer_EncDec, softs
│           ├── cmapss/                # model (SOFTSForRUL), train, search_space, experiment, re-exports
│           ├── kubeflow/              # pipeline.py, katib.py
│           ├── cli/                   # main.py (softs-new CLI)
│           ├── specs.py               # Pydantic ExperimentSpec hierarchy
│           └── utils.py               # YAML/JSON helpers
│
├── pipelines/
│   └── README.md                      # Compiled YAML output (git-ignored)
│
├── scripts/
│   └── build_image.sh                 # Docker image build script
│
├── src/kfp_workflow/
│   ├── __init__.py                    # Package root, __version__
│   ├── config_override.py             # CLI --set override utilities (coerce, merge, validate)
│   ├── specs.py                       # Pydantic config models (PipelineSpec, ServingSpec, TuneSpec)
│   ├── utils.py                       # YAML/JSON I/O helpers
│   ├── cli/
│   │   ├── main.py                    # Typer CLI (pipeline, serve, registry, cluster, spec, tune)
│   │   └── output.py                  # Rich-based structured output (tables, colors, JSON)
│   ├── components/
│   │   ├── __init__.py                # Re-exports all 5 component functions
│   │   ├── load_data.py               # KFP component: load data via plugin
│   │   ├── preprocess.py              # KFP component: preprocess via plugin
│   │   ├── train.py                   # KFP component: train via plugin
│   │   ├── evaluate.py                # KFP component: evaluate via plugin
│   │   └── save_model.py              # KFP component: save weights via plugin
│   ├── pipeline/
│   │   ├── compiler.py                # Pipeline DAG assembly + KFP compilation
│   │   ├── client.py                  # Pipeline compilation + submission
│   │   └── connection.py              # Reusable kfp_connection() context manager
│   ├── plugins/
│   │   ├── __init__.py                # Plugin registry dict + get_plugin()
│   │   ├── base.py                    # ModelPlugin ABC + result dataclasses + HPO contract
│   │   ├── mambasl_cmapss.py          # MambaSL C-MAPSS adapter plugin (incl. HPO)
│   │   ├── mrhysp_cmapss.py           # MR-HY-SP C-MAPSS adapter plugin (incl. HPO)
│   │   └── softs_cmapss.py            # SOFTS C-MAPSS adapter plugin (incl. HPO)
│   ├── tune/
│   │   ├── __init__.py
│   │   ├── exceptions.py              # TrialPruned exception (project-level, no Optuna leak)
│   │   ├── engine.py                  # Optuna HPO engine (study, trial loop, suggest)
│   │   └── katib.py                   # Katib Experiment CRD manifest builder
│   ├── registry/
│   │   ├── base.py                    # ABCs: ModelRegistryBase, DatasetRegistryBase
│   │   ├── model_registry.py          # FileModelRegistry (JSON on PVC)
│   │   └── dataset_registry.py        # PVCDatasetRegistry (JSON on PVC)
│   └── serving/
│       ├── __init__.py
│       ├── kserve.py                  # KServe InferenceService CRUD + status (custom + standard)
│       └── predictor.py               # Custom KServe predictor (plugin dispatch)
│
└── tests/
    ├── test_specs.py                  # Spec loading and validation
    ├── test_cli_protocol.py           # CLI command existence checks (all commands)
    ├── test_cli_run_commands.py       # Mocked functional tests for run/serve/experiment commands
    ├── test_config_override.py        # CLI --set override system tests
    ├── test_output.py                 # Output formatting unit tests
    ├── test_pipeline_compile.py       # Pipeline compilation verification
    ├── test_registry.py               # File-backed registry CRUD tests
    └── test_plugin_system.py          # Plugin ABC, registry, _build_cfg tests
```
