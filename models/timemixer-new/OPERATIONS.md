# Operations

This document covers package-local workflows for `timemixer-new`.

## Defaults

| Key | Value |
|-----|-------|
| Python | `>=3.10` |
| Entry point | `timemixer-new` |
| Package role | standalone TimeMixer package, not root-integrated yet |

## Setup

```bash
cd models/timemixer-new
make venv
make install
```

## Tests

```bash
make test
```

## Validate Specs

```bash
timemixer-new spec validate --spec configs/experiments/fd001_smoke.yaml
```

## Local Training

```bash
timemixer-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001
```

## Katib and Pipeline Helpers

```bash
timemixer-new katib render \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001

timemixer-new katib submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001

timemixer-new pipeline compile \
  --spec configs/experiments/fd_all_core_default.yaml \
  --output compiled/fd_all_core_default.yaml
```

## Maintenance Notes

- Keep package docs honest about integration status: this package is not currently wired into the root plugin registry.
- Keep this file aligned with [src/timemixer_new/cli/main.py](/home/scouter/proj_2026_1_etri/test/models/timemixer-new/src/timemixer_new/cli/main.py).
