# Project Structure

## Tree

```text
.
├── Makefile
├── PROJECT.md
├── README.md
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
│   └── katib/
│       └── README.md
├── src/
│   └── timemixer_new/
│       ├── __init__.py
│       ├── specs.py
│       ├── utils.py
│       ├── cli/
│       │   └── main.py              # Container-internal CLI (train, katib-trial, compile)
│       ├── cmapss/
│       │   ├── constants.py          # FD_CONFIGS, column definitions
│       │   ├── data.py               # C-MAPSS download + loading
│       │   ├── experiment.py         # HPO, training, ablation pipeline
│       │   ├── model.py              # TimeMixerForRUL model definition
│       │   ├── preprocess.py         # Normalization + feature selection
│       │   ├── report.py             # Literature comparison + summaries
│       │   ├── search_space.py       # HPO search space definitions
│       │   ├── train.py              # Training loop + metrics
│       │   └── windowing.py          # Sliding window construction
│       ├── kubeflow/
│       │   ├── katib.py              # Katib experiment manifest construction
│       │   └── pipeline.py           # HPO+ablation KFP pipeline topology
│       └── timemixer_layers/
│           ├── Autoformer_EncDec.py   # Moving-average series decomposition
│           ├── Embed.py               # Token embedding without positional encoding
│           ├── StandardNorm.py        # Reversible instance normalization (RevIN)
│           └── TimeMixer.py           # Core TimeMixer model (PDM blocks, multi-scale mixing)
└── tests/
    ├── test_kubeflow_protocol.py     # Katib manifest + execution spec tests
    ├── test_pipeline_compile.py      # KFP pipeline compilation
    └── test_specs.py                 # Spec loading + search space tests
```

## Roles

- `configs/experiments/`: YAML specs for smoke, default, and aggressive workflows.
- `configs/search_spaces/`: Human-readable mirrors of builtin HPO spaces.
- `src/timemixer_new/cmapss/`: Core ML — data, preprocessing, model, training, evaluation, reporting.
- `src/timemixer_new/kubeflow/`: Katib experiment construction and HPO+ablation KFP pipeline.
- `src/timemixer_new/timemixer_layers/`: TimeMixer architecture — multi-scale decomposition, PDM blocks, embedding, normalization.
- `src/timemixer_new/cli/`: Container-internal CLI for Katib trials and local training.
- `tests/`: Schema, search-space, Katib manifest, and compile-path verification.
