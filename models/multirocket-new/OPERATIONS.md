# Operations

This document covers package-local workflows for `multirocket-new`.

## Defaults

| Key | Value |
|-----|-------|
| Python | `>=3.10` |
| Entry point | `multirocket-new` |
| KFP SDK | `2.15.0` |
| Default namespace in examples | `kubeflow-user-example-com` |

## Setup

```bash
cd models/multirocket-new
make venv
make install
```

## Tests

```bash
make test
```

## Validation and Compilation

```bash
make spec-validate
make compile-pipeline
```

Equivalent direct commands:

```bash
multirocket-new spec validate --spec configs/experiments/fd_all_core_default.yaml

multirocket-new pipeline compile \
  --spec configs/experiments/fd_all_core_default.yaml \
  --output compiled/fd_all_core_default.yaml
```

## Local Training

```bash
multirocket-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001
```

## Package-Local Cluster Flow

Bootstrap storage:

```bash
multirocket-new cluster bootstrap \
  --spec configs/experiments/fd_all_core_default.yaml
```

Submit:

```bash
multirocket-new pipeline submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --namespace kubeflow-user-example-com
```

Render or submit Katib:

```bash
multirocket-new katib render \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001

multirocket-new katib submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001
```

## Maintenance Notes

- Keep this file aligned with [src/multirocket_new/cli/main.py](/home/scouter/proj_2026_1_etri/test/models/multirocket-new/src/multirocket_new/cli/main.py).
- The integrated root workflow is documented at [README.md](/home/scouter/proj_2026_1_etri/test/README.md) and [OPERATIONS.md](/home/scouter/proj_2026_1_etri/test/OPERATIONS.md).
