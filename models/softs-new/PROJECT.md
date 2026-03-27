# Project Structure — softs-new

```
models/softs-new/
├── .gitignore
├── Makefile                              # venv, install, test, spec-validate, compile-pipeline
├── PROJECT.md                            # This file — directory inventory
├── README.md                             # Package overview, quickstart, HPO params
├── OPERATIONS.md                         # Development and deployment procedures
├── pyproject.toml                        # Build config, dependencies (incl. mambasl-new), entry point
│
├── configs/
│   ├── experiments/
│   │   ├── fd001_smoke.yaml              # FD001 smoke test (CPU, 2 epochs, fixed params)
│   │   ├── fd_all_core_default.yaml      # FD001-FD004 default HPO workflow (random, 12 trials)
│   │   └── fd_all_core_aggressive.yaml   # FD001-FD004 aggressive HPO workflow (TPE, 30 trials)
│   └── search_spaces/
│       ├── default.yaml                  # Default profile: 7 architecture params (reference)
│       └── aggressive.yaml               # Aggressive profile: wider ranges for d_core, d_ff, etc.
│
├── kubeflow/
│   └── katib/
│       └── README.md                     # Katib HPO manifest generation and submission guide
│
├── pipelines/
│   └── README.md                         # Compiled YAML output directory (git-ignored)
│
├── src/softs_new/
│   ├── __init__.py                       # Package root, __version__ = "0.1.0"
│   ├── specs.py                          # Pydantic ExperimentSpec hierarchy (15 models)
│   ├── utils.py                          # YAML/JSON I/O helpers (load_yaml, save_json, etc.)
│   │
│   ├── softs_layers/
│   │   ├── __init__.py
│   │   ├── Embed.py                      # DataEmbedding_inverted: [B,T,N] → [B,N,d_model]
│   │   ├── Transformer_EncDec.py         # EncoderLayer (STAR attention + FFN), Encoder
│   │   └── softs.py                      # SOFTS.Model backbone (STAR + encoder stack)
│   │
│   ├── cmapss/
│   │   ├── __init__.py
│   │   ├── constants.py                  # re-export from mambasl_new.cmapss.constants
│   │   ├── data.py                       # re-export: ensure_cmapss_downloaded, load_fd
│   │   ├── preprocess.py                 # re-export: add_train_rul, preprocess_frames, etc.
│   │   ├── windowing.py                  # re-export: make_windows, make_last_windows
│   │   ├── model.py                      # SOFTSForRUL, build_model, configure_device
│   │   ├── train.py                      # train_model, predict_array, rmse, nasa_score, mae
│   │   ├── search_space.py               # builtin_search_space (default/aggressive), katib_parameter_specs
│   │   ├── experiment.py                 # run_experiment, hpo_phase, final_train_eval, ablation_sweep
│   │   └── report.py                     # aggregate_reports, write_summary_csv
│   │
│   ├── kubeflow/
│   │   ├── __init__.py
│   │   ├── pipeline.py                   # compile_pipeline, submit_pipeline (KFP v2)
│   │   └── katib.py                      # render_katib_experiment, submit_katib_experiment
│   │
│   └── cli/
│       ├── __init__.py
│       └── main.py                       # softs-new CLI (spec, train, report, pipeline, katib)
│
└── tests/
    ├── __init__.py
    ├── test_specs.py                     # Spec loading, d_core presence, Katib spec generation
    ├── test_pipeline_compile.py          # KFP pipeline compile smoke tests
    └── test_kubeflow_protocol.py         # PVC path substitution and Katib manifest tests
```
