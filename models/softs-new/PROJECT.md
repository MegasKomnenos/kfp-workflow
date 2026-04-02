# Project Structure

Maintained tree for `softs-new`.

```text
models/softs-new/
├── README.md
├── PROJECT.md
├── OPERATIONS.md
├── Makefile
├── pyproject.toml
├── configs/
│   ├── experiments/
│   └── search_spaces/
├── kubeflow/
│   └── katib/
│       └── README.md
├── pipelines/
│   └── README.md
├── src/softs_new/
│   ├── cli/
│   ├── cmapss/              # package-local SOFTS adaptation plus re-exported data utilities
│   ├── kubeflow/
│   ├── softs_layers/
│   ├── specs.py
│   └── utils.py
└── tests/
```

## Notes

- `softs_new.cmapss` mixes package-local model logic with re-exported C-MAPSS utilities from `mambasl-new`.
- The package is root-integrated through the `softs-cmapss` plugin, not through its standalone CLI.
