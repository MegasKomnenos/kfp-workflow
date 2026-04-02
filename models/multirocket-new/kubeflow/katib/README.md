# Katib Manifests

Use `multirocket-new katib render` to write versioned Katib manifests into this directory when you need a checked-in snapshot for review or cluster submission.

Preview a submission without applying it:

```bash
multirocket-new katib submit \
  --spec configs/experiments/fd_all_core_default.yaml \
  --dataset FD001 \
  --dry-run
```

This is the standalone package Katib flow. For the integrated root workflow, use the root [README.md](/home/scouter/proj_2026_1_etri/test/README.md) and [OPERATIONS.md](/home/scouter/proj_2026_1_etri/test/OPERATIONS.md).
