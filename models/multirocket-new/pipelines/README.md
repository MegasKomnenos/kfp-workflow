# Compiled Pipelines

This directory records one possible package-local output location for `multirocket-new pipeline compile`.

The current package examples and Makefile convention write compiled YAML to `compiled/`, not to this directory by default.

Example:

```bash
multirocket-new pipeline compile \
  --spec configs/experiments/fd_all_core_default.yaml \
  --output compiled/fd_all_core_default.yaml
```

Use package-local `pipeline submit` if you are operating the standalone package. Use the root `kfp-workflow` CLI when operating the integrated root workflow.
