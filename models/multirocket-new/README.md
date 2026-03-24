# multirocket-new

`multirocket-new` is a standalone Kubeflow-native repository for `aeon` MR-HY-SP experiments on NASA C-MAPSS FD001-FD004.

It now follows the same operator protocol as `mambasl-new`:
- one canonical experiment spec under `configs/experiments/`
- one top-level CLI with shared subcommands
- the same stage names for HPO, final training, ablations, and report aggregation
- the same hybrid runtime model for object-store pipeline roots plus optional PVC-backed data/results mounts

## Quick start

```bash
make venv
make install
make spec-validate
make compile-pipeline
```

Validate a spec:

```bash
.venv/bin/multirocket-new spec validate --spec configs/experiments/fd001_smoke.yaml
```

Run one fixed-configuration local training pass:

```bash
.venv/bin/multirocket-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001
```

Compile the Kubeflow pipeline:

```bash
.venv/bin/multirocket-new pipeline compile \
  --spec configs/experiments/fd_all_core_default.yaml \
  --output compiled/fd_all_core_default.yaml
```

Submit to Kubeflow:

```bash
.venv/bin/multirocket-new pipeline submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --namespace kubeflow-user-example-com
```

## Main interfaces

- `configs/experiments/*.yaml`: canonical experiment specs
- `configs/search_spaces/*.yaml`: human-readable HPO profiles
- `multirocket-new spec validate`: schema and consistency checks
- `multirocket-new train run`: local fixed-config training/evaluation
- `multirocket-new train katib-trial`: one Katib trial execution with concrete parameters
- `multirocket-new pipeline compile|submit`: KFP v2 packaging and submission
- `multirocket-new katib render|submit`: Katib manifest rendering and submission
- `multirocket-new cluster bootstrap`: PVC creation and optional dataset seeding from the spec
- `multirocket-new report summarize`: aggregate collected per-dataset results

## Repo layout

- `src/multirocket_new/config.py`: low-level single-run trainer config normalization
- `src/multirocket_new/experiment.py`: spec-driven dataset execution and ablation orchestration
- `src/multirocket_new/kubeflow/`: shared KFP, Katib, bootstrap, and submit helpers
- `configs/experiments/`: smoke, default, and aggressive workflow specs
- `kubeflow/pvc/`: PVC templates for data and results
- `pipelines/`: compiled pipeline landing area
- `tests/`: schema and Kubeflow protocol coverage

## Notes

- `selected_sensors` still coerces scaling mode to `global`.
- MultiRocket kernel counts are still canonicalized to `aeon`'s effective 84-kernel blocks.
- Katib metrics now follow the shared `objective`, `rmse`, `score`, `mae` stdout protocol.

## Workflow Contract

- The canonical workflow input is a versioned experiment spec under `configs/experiments/`.
- Recommended compile output for parity with `mambasl-new` is `compiled/<spec-name>.yaml`.
