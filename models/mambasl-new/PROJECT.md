# Project Structure

Maintained tree for the `mambasl-new` package.

```text
models/mambasl-new/
├── README.md
├── PROJECT.md
├── OPERATIONS.md
├── Makefile
├── pyproject.toml
├── configs/
│   ├── experiments/         # canonical package-local experiment specs
│   └── search_spaces/       # human-readable search-space references
├── kubeflow/
│   └── katib/
│       └── README.md        # package-local Katib notes
├── src/mambasl_new/
│   ├── cli/                 # argparse CLI entrypoint
│   ├── cmapss/              # data, preprocessing, model, train, report, search space
│   ├── kubeflow/            # Katib manifest and package-local KFP helpers
│   ├── mamba_layers/        # Mamba-specific model layers
│   ├── specs.py             # experiment schema
│   └── utils.py
└── tests/                   # package-local validation and protocol tests
```

## Notes

- This package is standalone, but it is also consumed by the root plugin adapter for `mambasl-cmapss`.
- The maintained user-facing docs for this package are `README.md` and `OPERATIONS.md`.
