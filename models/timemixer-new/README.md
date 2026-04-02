# timemixer-new

`timemixer-new` is a standalone TimeMixer-based C-MAPSS workflow package.

## Current Status

This package is maintained as a standalone package under `models/`, but it is not registered in the root `kfp-workflow` plugin registry in `src/kfp_workflow/plugins/__init__.py`.

That means:

- you can use `timemixer-new` directly as a package-local CLI
- you cannot currently reference `timemixer-new` from a root `kfp-workflow` pipeline, serving, tune, or benchmark spec without adding a root plugin adapter first

## Quick Start

```bash
make venv
make install
make test
```

Validate a spec:

```bash
timemixer-new spec validate --spec configs/experiments/fd001_smoke.yaml
```

Run local training:

```bash
timemixer-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001
```

Compile a package-local pipeline:

```bash
timemixer-new pipeline compile \
  --spec configs/experiments/fd_all_core_default.yaml \
  --output compiled/fd_all_core_default.yaml
```

## CLI Surface

Supported commands from `timemixer_new.cli.main`:

- `timemixer-new spec validate`
- `timemixer-new train run`
- `timemixer-new train katib-trial`
- `timemixer-new report summarize`
- `timemixer-new pipeline compile`
- `timemixer-new katib render`
- `timemixer-new katib submit`

## Reference Docs

- [PROJECT.md](/home/scouter/proj_2026_1_etri/test/models/timemixer-new/PROJECT.md)
- [OPERATIONS.md](/home/scouter/proj_2026_1_etri/test/models/timemixer-new/OPERATIONS.md)

The legacy `TimeMixer/` directory at the repo root is separate from this package and should not be confused with the maintained `models/timemixer-new` workflow package.
