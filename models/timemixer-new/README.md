# TimeMixer-New

ML model package for TimeMixer multi-scale mixing model on NASA C-MAPSS turbofan RUL prediction.

TimeMixer (ICLR 2024) decomposes time series into seasonal and trend components at multiple scales, then applies Past Decomposable Mixing (PDM) blocks that perform bottom-up seasonal mixing and top-down trend mixing. This package adapts TimeMixer for Remaining Useful Life regression on the C-MAPSS benchmark.

Consumed by the kfp-workflow plugin system via `kfp_workflow.plugins.timemixer_cmapss`. Pipeline submission and cluster bootstrap are handled by the parent kfp-workflow CLI.

## Core Modules

- `src/timemixer_new/cmapss/` — Data loading, preprocessing, windowing, model definition, training, evaluation, reporting, HPO search spaces
- `src/timemixer_new/timemixer_layers/` — TimeMixer architecture: multi-scale decomposition, embedding, PDM blocks
- `src/timemixer_new/kubeflow/` — Katib experiment manifest construction, HPO+ablation KFP pipeline
- `src/timemixer_new/cli/` — Container-internal CLI for Katib trial execution and local training
- `src/timemixer_new/specs.py` — Pydantic experiment spec models

## Datasets

NASA C-MAPSS FD001, FD002, FD003, FD004 — turbofan engine degradation simulation.

## CLI Commands (container-internal / local development)

```
timemixer-new spec validate       --spec <path>
timemixer-new train run           --spec <path> [--dataset] [--output-dir] [--params-json] [--run-hpo] [--run-ablations]
timemixer-new train katib-trial   --spec <path> --dataset <fd> --trial-params-json <json>
timemixer-new report summarize    --run-dir <path>
timemixer-new pipeline compile    --spec <path> --output <path>
timemixer-new katib render        --spec <path> --dataset <fd> [--output]
timemixer-new katib submit        --spec <path> --dataset <fd> [--output] [--dry-run]
```

## Development

```bash
make venv && make install && make test
```
