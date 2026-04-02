# Project Structure

Maintained tree for `timemixer-new`.

```text
models/timemixer-new/
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
├── src/timemixer_new/
│   ├── cli/
│   ├── cmapss/
│   ├── kubeflow/
│   ├── timemixer_layers/
│   ├── specs.py
│   └── utils.py
└── tests/
```

## Notes

- This package is standalone today.
- Root-project integration would require a new plugin adapter and registry entry in `src/kfp_workflow/plugins/`.
