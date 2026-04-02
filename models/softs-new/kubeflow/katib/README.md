# Katib Manifests

Use `softs-new katib render` to materialize reviewable Katib `Experiment` manifests into this directory when you need a saved snapshot.

## Example

```bash
softs-new katib render \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001 \
  --output kubeflow/katib/fd001_default_katib.yaml
```

Submit directly from the CLI:

```bash
softs-new katib submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001
```

Preview the manifest without applying it:

```bash
softs-new katib submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001 \
  --dry-run
```

## Trial Metrics

SOFTS Katib trials emit the shared stdout metric contract used by the maintained packages:

- `objective=<float>`
- `rmse=<float>`
- `score=<float>`
- `mae=<float>`

For integrated root workflows, use the root [README.md](/home/scouter/proj_2026_1_etri/test/README.md) and [OPERATIONS.md](/home/scouter/proj_2026_1_etri/test/OPERATIONS.md) rather than this package-local flow.
