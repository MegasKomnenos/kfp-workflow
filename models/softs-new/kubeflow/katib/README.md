# Katib HPO — SOFTS New

This directory holds rendered Katib `Experiment` CRD manifests for
hyperparameter optimisation of the SOFTS C-MAPSS model.

## Generating manifests

```bash
# Render a Katib experiment manifest for FD001
softs-new katib render \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001 \
  --output kubeflow/katib/fd001_smoke_katib.yaml

# Render for the full four-dataset default workflow
softs-new katib render \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001 \
  --output kubeflow/katib/fd001_default_katib.yaml
```

## Submitting to the cluster

```bash
# Submit pre-rendered manifest
softs-new katib submit \
  --manifest kubeflow/katib/fd001_default_katib.yaml

# Or render + submit in one step
softs-new katib submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001
```

## Parameter space

The Katib search covers 13 SOFTS-specific parameters:

| Parameter    | Role                            |
|--------------|---------------------------------|
| `d_model`    | Embedding dimension             |
| `d_core`     | STAR core dimension             |
| `d_ff`       | FFN hidden dimension            |
| `e_layers`   | Number of encoder layers        |
| `dropout`    | Dropout rate                    |
| `activation` | Activation function             |
| `use_norm`   | Instance normalisation toggle   |
| `batch_size` | Mini-batch size                 |
| `lr`         | Learning rate (log scale)       |
| `weight_decay` | L2 regularisation (log scale) |
| `huber_delta` | Huber loss delta               |
| `window_size` | Input sequence length          |
| `max_rul`    | RUL clamp ceiling               |

Profiles `default` and `aggressive` are defined in
`configs/search_spaces/`.

## Metrics collection

Katib reads the objective metric from **stdout** using the pattern:

```
softs-metric-val-loss=<float>
```

The trial container exits with code 0 on success; Katib marks it
`Succeeded`. Failed trials count against `max_failed_trial_count`.
