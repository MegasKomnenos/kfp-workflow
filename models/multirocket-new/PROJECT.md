# PROJECT

## Purpose
Provide a standalone Kubeflow-native workflow repository for `aeon` MR-HY-SP experiments on NASA C-MAPSS FD001-FD004 with the same operator protocol used by `mambasl-new`.

## Current structure
```text
.
├── Dockerfile
├── Makefile
├── OPERATIONS.md
├── PROJECT.md
├── README.md
├── UI_WORKFLOW_UNIFICATION.md
├── pyproject.toml
├── configs/
│   ├── experiments/
│   │   ├── fd001_smoke.yaml
│   │   ├── fd_all_core_default.yaml
│   │   └── fd_all_core_aggressive.yaml
│   └── search_spaces/
│       ├── default.yaml
│       └── aggressive.yaml
├── kubeflow/
│   ├── katib/
│   │   └── README.md
│   └── pvc/
│       ├── cmapss-data-pvc.yaml
│       └── cmapss-results-pvc.yaml
├── pipelines/
│   └── README.md
├── scripts/
│   └── build_image.sh
├── src/
│   └── multirocket_new/
│       ├── __init__.py
│       ├── cmapss.py
│       ├── config.py
│       ├── experiment.py
│       ├── model.py
│       ├── runner.py
│       ├── search_space.py
│       ├── specs.py
│       ├── utils.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py
│       └── kubeflow/
│           ├── __init__.py
│           ├── bootstrap.py
│           ├── client.py
│           ├── katib.py
│           └── pipeline.py
└── tests/
    ├── test_cli_protocol.py
    ├── test_compile_and_render.py
    └── test_specs.py
```

## Component notes
- `UI_WORKFLOW_UNIFICATION.md`: explicit shared-operator checklist and current status.
- `src/multirocket_new/specs.py`: canonical experiment schema, runtime/storage protocol, and ablation expansion.
- `src/multirocket_new/experiment.py`: spec-driven final training, Katib trial handling, and ablation orchestration.
- `src/multirocket_new/config.py`: low-level trainer config normalization used by the MR-HY-SP runner.
- `src/multirocket_new/kubeflow/`: PVC bootstrap, Katib manifest generation, KFP pipeline compilation, and submission helpers.
- `configs/experiments/`: smoke, default, and aggressive operator-ready workflows.
- `tests/`: schema, CLI-protocol, bootstrap, manifest-render, and pipeline-compile verification.
