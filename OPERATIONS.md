# Operations

This document is the canonical command and procedure reference for the integrated root project.

## Defaults

| Key | Value |
|-----|-------|
| Python | `>=3.10` |
| Root entry point | `kfp-workflow` |
| KFP SDK | `2.15.0` |
| KServe SDK extra | `0.17.0` |
| Default namespace in examples | `kubeflow-user-example-com` |
| Default local KFP API host | `http://127.0.0.1:8888` |
| Root image name in examples | `kfp-workflow:latest` |
| Supported root plugins | `mambasl-cmapss`, `mrhysp-cmapss`, `softs-cmapss` |

## Local Development

### Setup

```bash
make venv
make install
```

### Tests

```bash
make test
```

### Root helper targets

```bash
make spec-validate
make compile-pipeline
make docker-build
```

## Spec Validation

Validate the four supported root spec types:

```bash
kfp-workflow spec validate --spec configs/pipelines/mambasl_cmapss_smoke.yaml
kfp-workflow spec validate --spec configs/serving/mambasl_cmapss_serve.yaml --type serving
kfp-workflow spec validate --spec configs/tuning/mambasl_cmapss_tune.yaml --type tune
kfp-workflow spec validate --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml --type benchmark
```

Machine-readable output:

```bash
kfp-workflow --json spec validate \
  --spec configs/tuning/mambasl_cmapss_tune.yaml \
  --type tune
```

Override values without editing YAML:

```bash
kfp-workflow spec validate \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --set train.max_epochs=10 \
  --set dataset.config.fd[0].fd_name=FD003
```

## Training Pipeline Workflow

Compile:

```bash
kfp-workflow pipeline compile \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --output pipelines/mambasl_cmapss_smoke.yaml
```

Submit:

```bash
kfp-workflow pipeline submit \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml
```

Common run management:

```bash
kfp-workflow pipeline run list
kfp-workflow pipeline run get <run-id-or-prefix>
kfp-workflow pipeline run wait <run-id-or-prefix>
kfp-workflow pipeline run logs <run-id-or-prefix>
kfp-workflow pipeline run terminate <run-id-or-prefix>
```

Experiment listing:

```bash
kfp-workflow pipeline experiment list
```

## Serving Workflow

Create:

```bash
kfp-workflow serve create \
  --spec configs/serving/mambasl_cmapss_serve.yaml \
  --wait
```

Preview only:

```bash
kfp-workflow serve create \
  --spec configs/serving/mambasl_cmapss_serve.yaml \
  --dry-run
```

Inspect and remove:

```bash
kfp-workflow serve list
kfp-workflow serve get --name mambasl-cmapss-serving
kfp-workflow serve delete --name mambasl-cmapss-serving
```

Notes:

- `serve create` supports `--wait` and `--timeout`.
- The command prints the manifest JSON and, when waiting, prints readiness diagnostics.
- The service name comes from `metadata.name` inside the serving spec.

## Benchmark Workflow

Compile:

```bash
kfp-workflow benchmark compile \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml \
  --output pipelines/mambasl_cmapss_kepler_smoke.yaml
```

Submit:

```bash
kfp-workflow benchmark submit \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml
```

Inspect results:

```bash
kfp-workflow benchmark list
kfp-workflow benchmark get <run-id-or-prefix>
kfp-workflow benchmark download <run-id-or-prefix>
```

Notes:

- Benchmark specs may reference reusable scenario and metric YAML under `configs/benchmarks/`.
- Results are stored on the benchmark results PVC and can be downloaded locally through the CLI.

## Tuning Workflow

Show the resolved search space:

```bash
kfp-workflow tune space --spec configs/tuning/mambasl_cmapss_tune.yaml
```

Preview the generated Katib manifest:

```bash
kfp-workflow tune \
  --spec configs/tuning/mambasl_cmapss_tune.yaml \
  --dry-run
```

Submit and wait:

```bash
kfp-workflow tune \
  --spec configs/tuning/mambasl_cmapss_tune.yaml \
  --wait
```

Inspect progress and fetch results:

```bash
kfp-workflow tune status
kfp-workflow tune status <experiment-id-or-prefix>
kfp-workflow tune logs <experiment-id-or-prefix>
kfp-workflow tune results <experiment-id-or-prefix>
```

Apply best parameters into an existing pipeline spec:

```bash
kfp-workflow tune results <experiment-id-or-prefix> \
  --apply-best configs/pipelines/mambasl_cmapss_smoke.yaml
```

Important behavior:

- The supported user-facing entrypoint is `kfp-workflow tune --spec ...`.
- Each submission gets a generated Katib experiment ID; the logical tune name is preserved in annotations and result payloads.
- Hidden commands such as `tune run`, `tune katib`, and `tune show-space` remain compatibility paths and are not the preferred public workflow.

## Cluster Bootstrap

Create PVCs derived from a pipeline, benchmark, or tune spec:

```bash
kfp-workflow cluster bootstrap \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml

kfp-workflow cluster bootstrap \
  --type benchmark \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml

kfp-workflow cluster bootstrap \
  --type tune \
  --spec configs/tuning/mambasl_cmapss_tune.yaml
```

Preview manifests without applying:

```bash
kfp-workflow cluster bootstrap \
  --type tune \
  --spec configs/tuning/mambasl_cmapss_tune.yaml \
  --dry-run
```

Behavior:

- The command emits JSON manifests first.
- Without `--dry-run`, it creates the namespace if needed and then applies each PVC manifest.

## Docker Image

Build the unified root image:

```bash
docker build -t kfp-workflow:latest -f docker/Dockerfile .
```

Or use the helper script:

```bash
./scripts/build_image.sh
```

Local-cluster import example:

```bash
docker save kfp-workflow:latest -o /tmp/kfp-workflow-latest.tar
```

If the cluster cannot pull from a registry, use the helper-pod import flow already documented by the project and avoid inventing alternate ad hoc paths.

## Auth and Connection Notes

- Root `pipeline submit`, `benchmark submit`, and run listing commands can use `--host`, `--user`, `--existing-token`, and `--cookies`.
- When `host` is left at the default local address, the client code uses the port-forward settings in the spec runtime block.
- The default spec runtime values are defined in [src/kfp_workflow/specs.py](/home/scouter/proj_2026_1_etri/test/src/kfp_workflow/specs.py).

## Maintenance Rules

- Keep this file aligned with the actual CLI in [src/kfp_workflow/cli/main.py](/home/scouter/proj_2026_1_etri/test/src/kfp_workflow/cli/main.py).
- When adding or removing maintained directories, update [PROJECT.md](/home/scouter/proj_2026_1_etri/test/PROJECT.md).
- When changing the supported root model plugins, update both this file and [README.md](/home/scouter/proj_2026_1_etri/test/README.md).
