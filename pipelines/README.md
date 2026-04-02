# Compiled Pipelines

This directory is a conventional manual output location used by explicit compile examples in the root project.

It is not the implicit output path used by `pipeline submit` or `benchmark submit`; those flows auto-compile into `compiled/<spec-name>.yaml`.

Typical commands:

```bash
make compile-pipeline
```

```bash
kfp-workflow pipeline compile \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --output pipelines/mambasl_cmapss_smoke.yaml
```

```bash
kfp-workflow benchmark compile \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml \
  --output pipelines/mambasl_cmapss_kepler_smoke.yaml
```

YAML written here should be treated as build output rather than hand-maintained source.
