# mambasl-new

`mambasl-new` is a standalone research package for MambaSL-based C-MAPSS workflows. The root project consumes it indirectly through the `mambasl-cmapss` plugin adapter in `kfp_workflow.plugins.mambasl_cmapss`.

## Scope

This package owns:

- C-MAPSS data handling, preprocessing, windowing, and training
- MambaSL model definition and package-local reporting helpers
- package-local Katib manifest generation and KFP pipeline compilation
- a standalone CLI for package-local training and trial execution

The root project owns:

- end-to-end `kfp-workflow` specs
- root pipeline submission
- root benchmark workflows
- root KServe deployment and registry management

## Quick Start

```bash
make venv
make install
make test
```

Validate an experiment spec:

```bash
mambasl-new spec validate --spec configs/experiments/fd001_smoke.yaml
```

Run package-local training:

```bash
mambasl-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001
```

Compile a package-local Kubeflow pipeline:

```bash
mambasl-new pipeline compile \
  --spec configs/experiments/fd_all_core_default.yaml \
  --output compiled/fd_all_core_default.yaml
```

## CLI Surface

Supported commands from `mambasl_new.cli.main`:

- `mambasl-new spec validate`
- `mambasl-new train run`
- `mambasl-new train katib-trial`
- `mambasl-new report summarize`
- `mambasl-new pipeline compile`
- `mambasl-new katib render`
- `mambasl-new katib submit`

This package does not currently expose standalone `pipeline submit` or `cluster bootstrap` commands.

## Important Paths

- `configs/experiments/`: canonical experiment specs
- `configs/search_spaces/`: human-readable search-space references
- `src/mambasl_new/cmapss/`: dataset, preprocessing, model, train, report
- `src/mambasl_new/mamba_layers/`: model building blocks
- `src/mambasl_new/kubeflow/`: package-local Katib and KFP helpers
- `tests/`: package-local validation and compile coverage

## Relationship to the Root Project

If you are operating the integrated stack, start from the root docs:

- [README.md](/home/scouter/proj_2026_1_etri/test/README.md)
- [OPERATIONS.md](/home/scouter/proj_2026_1_etri/test/OPERATIONS.md)

If you are modifying this package itself, use:

- [PROJECT.md](/home/scouter/proj_2026_1_etri/test/models/mambasl-new/PROJECT.md)
- [OPERATIONS.md](/home/scouter/proj_2026_1_etri/test/models/mambasl-new/OPERATIONS.md)
