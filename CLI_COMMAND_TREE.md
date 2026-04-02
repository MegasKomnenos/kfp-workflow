# `kfp-workflow` Command Tree

This file is the canonical public command hierarchy for the root `kfp-workflow` CLI.

Source of truth:

- [src/kfp_workflow/cli/main.py](/home/scouter/proj_2026_1_etri/test/src/kfp_workflow/cli/main.py)

This reference documents the supported public surface only. Hidden compatibility or internal commands under `tune` are intentionally excluded from the canonical tree.

## Global Shape

```text
kfp-workflow [--json]
├── spec
│   └── validate
├── pipeline
│   ├── compile
│   ├── submit
│   ├── run
│   │   ├── get
│   │   ├── list
│   │   ├── wait
│   │   ├── terminate
│   │   └── logs
│   └── experiment
│       └── list
├── benchmark
│   ├── compile
│   ├── submit
│   ├── list
│   ├── get
│   └── download
├── serve
│   ├── create
│   ├── delete
│   ├── list
│   └── get
├── registry
│   ├── model
│   │   ├── register
│   │   ├── get
│   │   └── list
│   └── dataset
│       ├── register
│       ├── get
│       └── list
├── cluster
│   └── bootstrap
└── tune
    ├── [callback submit mode]
    ├── status
    ├── results
    ├── space
    └── logs
```

## Global Behavior

`kfp-workflow [--json]`

- Purpose: root entrypoint for training, benchmark, serving, registry, bootstrap, spec validation, and tuning workflows.
- Global option: `--json` switches supported commands to machine-readable JSON output.
- Public top-level command groups: `pipeline`, `benchmark`, `serve`, `registry`, `cluster`, `spec`, `tune`.

## `spec`

### `kfp-workflow spec validate`

- Purpose: load and validate one spec file.
- Required: `--spec <path>`
- Key options:
  - `--type pipeline|serving|tune|benchmark`
  - `--set key=value`
- Behavior:
  - `pipeline`, `serving`, and `tune` validate the typed spec and plugin-specific config.
  - `benchmark` validates the fully materialized benchmark payload.
  - With global `--json`, the validated payload is emitted directly.

## `pipeline`

### `kfp-workflow pipeline compile`

- Purpose: compile a training pipeline into a KFP v2 YAML package.
- Required:
  - `--spec <path>`
  - `--output <path>`
- Key options:
  - `--set key=value`

### `kfp-workflow pipeline submit`

- Purpose: compile and submit a training pipeline to Kubeflow.
- Required: `--spec <path>`
- Key options:
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`
  - `--set key=value`

### `kfp-workflow pipeline run`

Public nested run-management layer for KFP runs.

#### `kfp-workflow pipeline run get <run_id>`

- Purpose: show detailed information for a pipeline run.
- Required argument: `<run_id>` or a unique visible prefix
- Key options:
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

#### `kfp-workflow pipeline run list`

- Purpose: list visible pipeline runs.
- Key options:
  - `--namespace <ns>`
  - `--experiment-id <id-or-prefix>`
  - `--page-size <n>`
  - `--sort-by "created_at desc"`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

#### `kfp-workflow pipeline run wait <run_id>`

- Purpose: block until a run reaches a terminal state.
- Required argument: `<run_id>` or a unique visible prefix
- Key options:
  - `--timeout <seconds>`
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

#### `kfp-workflow pipeline run terminate <run_id>`

- Purpose: cancel a running pipeline.
- Required argument: `<run_id>` or a unique visible prefix
- Key options:
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

#### `kfp-workflow pipeline run logs <run_id>`

- Purpose: view component pod logs associated with a run.
- Required argument: `<run_id>` or a unique visible prefix
- Key options:
  - `--step <substring>`
  - `--namespace <ns>`
- Note: this command looks up Kubernetes pods directly and does not expose host/auth flags.

### `kfp-workflow pipeline experiment`

Public nested experiment-management layer.

#### `kfp-workflow pipeline experiment list`

- Purpose: list visible pipeline experiments.
- Key options:
  - `--namespace <ns>`
  - `--page-size <n>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

## `benchmark`

### `kfp-workflow benchmark compile`

- Purpose: compile a benchmark workflow into a KFP v2 YAML package.
- Required:
  - `--spec <path>`
  - `--output <path>`
- Key options:
  - `--set key=value`

### `kfp-workflow benchmark submit`

- Purpose: compile and submit a benchmark workflow to Kubeflow.
- Required: `--spec <path>`
- Key options:
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`
  - `--set key=value`

### `kfp-workflow benchmark list`

- Purpose: list visible benchmark runs.
- Key options:
  - `--namespace <ns>`
  - `--page-size <n>`
  - `--sort-by "created_at desc"`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

### `kfp-workflow benchmark get <run_id>`

- Purpose: show detailed benchmark run state and resolved result metadata.
- Required argument: `<run_id>` or a unique visible prefix
- Key options:
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

### `kfp-workflow benchmark download <run_id>`

- Purpose: download the benchmark run payload to a local JSON file.
- Required argument: `<run_id>` or a unique visible prefix
- Key options:
  - `--output <path>`
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

## `serve`

### `kfp-workflow serve create`

- Purpose: create a KServe `InferenceService` from a serving spec.
- Required: `--spec <path>`
- Key options:
  - `--dry-run`
  - `--wait`
  - `--timeout <seconds>`
- Behavior:
  - always prints the generated manifest JSON
  - with `--wait`, also prints readiness diagnostics and exits non-zero if not ready

### `kfp-workflow serve delete`

- Purpose: delete a KServe `InferenceService`.
- Required: `--name <service-name>`
- Key options:
  - `--namespace <ns>`

### `kfp-workflow serve list`

- Purpose: list KServe `InferenceService` objects in one namespace.
- Key options:
  - `--namespace <ns>`

### `kfp-workflow serve get`

- Purpose: show detailed status, conditions, and warning events for one `InferenceService`.
- Required: `--name <service-name>`
- Key options:
  - `--namespace <ns>`

## `registry`

### `kfp-workflow registry model`

#### `kfp-workflow registry model register`

- Purpose: register a model in the file-backed model registry.
- Required:
  - `--name <model-name>`
  - `--version <version>`
  - `--uri <artifact-uri-or-pvc-subpath>`
- Key options:
  - `--framework <name>` default `pytorch`
  - `--description <text>`
  - `--registry-path <json-path>`

#### `kfp-workflow registry model get`

- Purpose: retrieve one model entry.
- Required: `--name <model-name>`
- Key options:
  - `--version <version>`
  - `--registry-path <json-path>`

#### `kfp-workflow registry model list`

- Purpose: list all model entries.
- Key options:
  - `--registry-path <json-path>`

### `kfp-workflow registry dataset`

#### `kfp-workflow registry dataset register`

- Purpose: register a dataset in the PVC-backed dataset registry.
- Required:
  - `--name <dataset-name>`
  - `--pvc-name <pvc>`
  - `--subpath <path-within-pvc>`
- Key options:
  - `--version <version>` default `v1`
  - `--description <text>`
  - `--registry-path <json-path>`

#### `kfp-workflow registry dataset get`

- Purpose: retrieve one dataset entry.
- Required: `--name <dataset-name>`
- Key options:
  - `--version <version>`
  - `--registry-path <json-path>`

#### `kfp-workflow registry dataset list`

- Purpose: list all dataset entries.
- Key options:
  - `--registry-path <json-path>`

## `cluster`

### `kfp-workflow cluster bootstrap`

- Purpose: generate and optionally apply PVC manifests for a pipeline, benchmark, or tune spec.
- Required: `--spec <path>`
- Key options:
  - `--type pipeline|benchmark|tune`
  - `--dry-run`
- Behavior:
  - prints manifest JSON before any apply step
  - applies PVCs and creates the namespace if needed when not in dry-run mode

## `tune`

`tune` is the only public command group with callback-style submission behavior in addition to subcommands.

### Callback submit mode

Invocation:

```bash
kfp-workflow tune --spec <tune-spec.yaml> [--set ...] [--output <path>] [--dry-run] [--wait]
```

- Purpose: submit a Katib experiment directly when `tune` is called with `--spec` and no subcommand.
- Required for submission mode: `--spec <path>`
- Key options:
  - `--set key=value`
  - `--output <manifest-path>`
  - `--dry-run`
  - `--wait`
- Behavior:
  - builds a Katib manifest from the tune spec
  - submits it with `kubectl create -f -` unless `--dry-run` is used
  - generates a unique experiment ID per submission

### `kfp-workflow tune status [experiment_name_or_prefix]`

- Purpose: list all managed tune experiments or show one resolved experiment.
- Optional argument:
  - `[experiment_name_or_prefix]`
- Key options:
  - `--namespace <ns>`

### `kfp-workflow tune results <experiment_name_or_prefix>`

- Purpose: download one tune result payload and optionally merge best params into a pipeline spec.
- Required argument:
  - `<experiment_name_or_prefix>`
- Key options:
  - `--output <path>`
  - `--namespace <ns>`
  - `--from-pvc`
  - `--apply-best <pipeline-spec-path>`

### `kfp-workflow tune space`

- Purpose: show the resolved plugin search space for a tune spec.
- Required: `--spec <path>`
- Key options:
  - `--set key=value`

### `kfp-workflow tune logs <experiment_name_or_prefix>`

- Purpose: show stored trial logs for one tune experiment.
- Required argument:
  - `<experiment_name_or_prefix>`
- Key options:
  - `--namespace <ns>`
  - `--all`
  - `--trial <trial-name>`
  - `--tail <n>`

## Public Surface Notes

- `pipeline run` and `pipeline experiment` are nested under `pipeline`, not top-level groups.
- `registry` is split into `model` and `dataset` subtrees.
- `cluster` exposes only `bootstrap`.
- Hidden `tune` compatibility or internal commands exist in code but are not part of this canonical public reference.
