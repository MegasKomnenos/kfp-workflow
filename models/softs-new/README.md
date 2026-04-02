# softs-new

`softs-new` is a standalone SOFTS-based C-MAPSS workflow package. The root project consumes it through the `softs-cmapss` plugin adapter in `kfp_workflow.plugins.softs_cmapss`.

## Scope

This package owns:

- SOFTS model layers and C-MAPSS adaptation
- package-local experiment specs and search spaces
- package-local Katib manifest generation and KFP pipeline compilation
- package-local training and trial execution CLI

The integrated root project owns:

- root `kfp-workflow` specs and submission
- root serving, benchmark, and registry workflows

## Quick Start

```bash
make venv
make install
make test
make spec-validate
```

Validate a spec:

```bash
softs-new spec validate --spec configs/experiments/fd001_smoke.yaml
```

Run package-local training:

```bash
softs-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001
```

Compile a package-local pipeline:

```bash
softs-new pipeline compile \
  --spec configs/experiments/fd001_smoke.yaml \
  --output pipelines/fd001_smoke.yaml
```

## CLI Surface

Supported commands from `softs_new.cli.main`:

- `softs-new spec validate`
- `softs-new train run`
- `softs-new train katib-trial`
- `softs-new report summarize`
- `softs-new pipeline compile`
- `softs-new katib render`
- `softs-new katib submit`

This package does not currently expose standalone `pipeline submit` or `cluster bootstrap` commands.

## Package Notes

- The package depends on `mambasl-new` for shared C-MAPSS data utilities.
- Integrated root examples default to the root image built from `docker/Dockerfile`; package docs should not claim a package-local `docker/Dockerfile.softs` because that file does not exist here.

## Reference Docs

- [PROJECT.md](/home/scouter/proj_2026_1_etri/test/models/softs-new/PROJECT.md)
- [OPERATIONS.md](/home/scouter/proj_2026_1_etri/test/models/softs-new/OPERATIONS.md)
