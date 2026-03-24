# OPERATIONS

## Defaults

- Python target: `3.9+`
- Default namespace: `kubeflow-user-example-com`
- Expected control plane:
  - Kubeflow Pipelines `2.15.0`
  - Katib `0.19.0`
- Canonical workflow input: a versioned experiment spec under `configs/experiments/`

## Common commands

```bash
make install
make test
make spec-validate
make compile-pipeline
```

Validate an experiment:

```bash
.venv/bin/multirocket-new spec validate --spec configs/experiments/fd_all_core_default.yaml
```

Bootstrap PVC-backed storage from the spec:

```bash
.venv/bin/multirocket-new cluster bootstrap \
  --spec configs/experiments/fd_all_core_default.yaml
```

Run one dataset locally from the canonical spec:

```bash
.venv/bin/multirocket-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001
```

Compile and submit:

```bash
.venv/bin/multirocket-new pipeline compile \
  --spec configs/experiments/fd_all_core_default.yaml \
  --output compiled/fd_all_core_default.yaml
.venv/bin/multirocket-new pipeline submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --namespace kubeflow-user-example-com
```

Render and optionally submit Katib:

```bash
.venv/bin/multirocket-new katib render \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001
.venv/bin/multirocket-new katib submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001 \
  --dry-run
```

## Runtime notes

- The default shared runtime model supports both `pipeline_root`-backed KFP execution and PVC-backed `/mnt/data` plus `/mnt/results` mounts.
- If the KFP API is behind Dex or auth headers, use `pipeline submit --existing-token ...` or `pipeline submit --cookies ...`.
- Katib trials emit the shared stdout metrics protocol: `objective`, `rmse`, `score`, and `mae`.
- Local smoke specs can run from downloaded local data, while the default and aggressive specs assume PVC-backed cluster execution.

## Maintenance

- Keep `configs/experiments/` and `configs/search_spaces/` aligned with the code-backed schema.
- Keep `PROJECT.md` synchronized with the live tree after structural edits.
- Prefer adding new control surfaces through the experiment spec instead of one-off CLI flags.
