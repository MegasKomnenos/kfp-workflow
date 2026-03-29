# MambaSL-New

ML model package for MambaSL state-space model on NASA C-MAPSS turbofan RUL prediction.

Consumed by the kfp-workflow plugin system via `kfp_workflow.plugins.mambasl_cmapss`. Pipeline submission and cluster bootstrap are handled by the parent kfp-workflow CLI.

## Core Modules

- `src/mambasl_new/cmapss/` — Data loading, preprocessing, windowing, model definition, training, evaluation, reporting, HPO search spaces
- `src/mambasl_new/mamba_layers/` — Mamba time-variant block and positional embedding
- `src/mambasl_new/kubeflow/` — Katib experiment manifest construction, HPO+ablation KFP pipeline
- `src/mambasl_new/cli/` — Container-internal CLI for Katib trial execution and local training
- `src/mambasl_new/specs.py` — Pydantic experiment spec models

## Datasets

NASA C-MAPSS FD001, FD002, FD003, FD004 — turbofan engine degradation simulation.

## CLI Commands (container-internal / local development)

```
mambasl-new spec validate       --spec <path>
mambasl-new train run           --spec <path> [--dataset] [--output-dir] [--params-json] [--run-hpo] [--run-ablations]
mambasl-new train katib-trial   --spec <path> --dataset <fd> --trial-params-json <json>
mambasl-new report summarize    --run-dir <path>
mambasl-new pipeline compile    --spec <path> --output <path>
mambasl-new katib render        --spec <path> --dataset <fd> [--output]
mambasl-new katib submit        --spec <path> --dataset <fd> [--output] [--dry-run]
```

## Development

```bash
make venv && make install && make test
```
