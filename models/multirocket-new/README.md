# multirocket-new

`multirocket-new` is a standalone MR-HY-SP workflow package for NASA C-MAPSS experiments. Its architecture is intentionally close to the other `*-new` packages, but it also exposes package-local `pipeline submit` and `cluster bootstrap` commands.

The root project consumes the MR-HY-SP model through the `mrhysp-cmapss` plugin adapter, not by shelling out to this package CLI.

## Quick Start

```bash
make venv
make install
make test
make spec-validate
```

Validate an experiment:

```bash
multirocket-new spec validate --spec configs/experiments/fd_all_core_default.yaml
```

Run one dataset locally:

```bash
multirocket-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001
```

Compile and submit:

```bash
multirocket-new pipeline compile \
  --spec configs/experiments/fd_all_core_default.yaml \
  --output compiled/fd_all_core_default.yaml

multirocket-new pipeline submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --namespace kubeflow-user-example-com
```

## CLI Surface

Supported commands from `multirocket_new.cli.main`:

- `multirocket-new spec validate`
- `multirocket-new train run`
- `multirocket-new train katib-trial`
- `multirocket-new report summarize`
- `multirocket-new pipeline compile`
- `multirocket-new pipeline submit`
- `multirocket-new katib render`
- `multirocket-new katib submit`
- `multirocket-new cluster bootstrap`

## Important Paths

- `configs/experiments/`: canonical workflow specs
- `configs/search_spaces/`: package-local search-space definitions
- `src/multirocket_new/experiment.py`: run orchestration
- `src/multirocket_new/model.py`: MR-HY-SP model composition
- `src/multirocket_new/kubeflow/`: client, bootstrap, Katib, pipeline helpers
- `kubeflow/pvc/`: PVC manifests for package-local workflow storage

## Integration Status

- Root integration exists through `kfp_workflow.plugins.mrhysp_cmapss`.
- The package CLI is useful when working on the standalone package itself.
- Root documentation remains the canonical source for integrated `kfp-workflow` operation.
