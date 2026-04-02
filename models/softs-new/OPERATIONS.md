# Operations

This document covers package-local workflows for `softs-new`.

## Defaults

| Key | Value |
|-----|-------|
| Python | `>=3.9` |
| Entry point | `softs-new` |
| Package role | standalone SOFTS package used by the root `softs-cmapss` plugin |

## Setup

```bash
cd models/softs-new
make venv
make install
```

## Tests and Helpers

```bash
make test
make spec-validate
make compile-pipeline
```

Validate a package-local spec directly:

```bash
softs-new spec validate --spec configs/experiments/fd001_smoke.yaml
```

## Local Training

```bash
softs-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001
```

## Katib Workflow

```bash
softs-new katib render \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001

softs-new katib submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001
```

## Pipeline Compilation

```bash
softs-new pipeline compile \
  --spec configs/experiments/fd001_smoke.yaml \
  --output pipelines/fd001_smoke.yaml
```

## Integration Notes

- Root-cluster submission, serving, and benchmarking are handled by `kfp-workflow`, not by this package.
- Integrated root examples default to the image built from [docker/Dockerfile](/home/scouter/proj_2026_1_etri/test/docker/Dockerfile); use that wording rather than implying a package-local image contract.
