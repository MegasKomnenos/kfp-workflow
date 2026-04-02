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

## Trial Metrics

SOFTS Katib trials emit the shared stdout metric contract used by the maintained packages:

- `objective=<float>`
- `rmse=<float>`
- `score=<float>`
- `mae=<float>`
