# Operations

This document covers package-local workflows for `mambasl-new`.

## Defaults

| Key | Value |
|-----|-------|
| Python | `>=3.10` |
| Entry point | `mambasl-new` |
| Package role | standalone research package consumed by the root MambaSL plugin |
| Package Make targets | `venv`, `install`, `test` |

## Setup

```bash
cd models/mambasl-new
make venv
make install
```

## Tests

```bash
make test
```

## Validate Specs

```bash
mambasl-new spec validate --spec configs/experiments/fd001_smoke.yaml
```

## Local Training

Single dataset run:

```bash
mambasl-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001
```

Run all datasets declared in the spec:

```bash
mambasl-new train run \
  --spec configs/experiments/fd_all_core_default.yaml
```

Inject explicit parameter values:

```bash
mambasl-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001 \
  --params-json '{"d_model": 64, "lr": 0.001}'
```

## Katib Trial and Manifest Workflow

Render a package-local Katib manifest:

```bash
mambasl-new katib render \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001
```

Submit the manifest:

```bash
mambasl-new katib submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001
```

Run a single trial command directly:

```bash
mambasl-new train katib-trial \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001 \
  --trial-params-json '{"d_model": 64, "lr": 0.001}'
```

## Pipeline Compilation

```bash
mambasl-new pipeline compile \
  --spec configs/experiments/fd_all_core_default.yaml \
  --output compiled/fd_all_core_default.yaml
```

This is package-local compilation only. Root-cluster submission is handled by `kfp-workflow`.

## Maintenance Notes

- Keep this file aligned with [src/mambasl_new/cli/main.py](/home/scouter/proj_2026_1_etri/test/models/mambasl-new/src/mambasl_new/cli/main.py).
- Keep [PROJECT.md](/home/scouter/proj_2026_1_etri/test/models/mambasl-new/PROJECT.md) synchronized with the maintained package tree.
