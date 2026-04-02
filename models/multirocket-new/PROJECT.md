# Project Structure

Maintained tree for `multirocket-new`.

```text
models/multirocket-new/
├── README.md
├── PROJECT.md
├── OPERATIONS.md
├── UI_WORKFLOW_UNIFICATION.md
├── Dockerfile
├── Makefile
├── pyproject.toml
├── configs/
│   ├── experiments/
│   └── search_spaces/
├── kubeflow/
│   ├── katib/
│   │   └── README.md
│   └── pvc/
├── pipelines/
│   └── README.md
├── scripts/
│   └── build_image.sh
├── src/multirocket_new/
│   ├── cli/
│   ├── kubeflow/
│   ├── cmapss.py
│   ├── config.py
│   ├── experiment.py
│   ├── model.py
│   ├── runner.py
│   ├── search_space.py
│   ├── specs.py
│   └── utils.py
└── tests/
```

## Notes

- This package exposes more package-local operational commands than the other `*-new` packages.
- `UI_WORKFLOW_UNIFICATION.md` is a design/status document, not the primary user entrypoint.
