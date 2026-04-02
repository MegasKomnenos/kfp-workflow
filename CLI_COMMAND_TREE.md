# `kfp-workflow` Command Tree

This file is the canonical public command hierarchy for the root `kfp-workflow` CLI.

Source of truth:

- [src/kfp_workflow/cli/main.py](/home/scouter/proj_2026_1_etri/test/src/kfp_workflow/cli/main.py)

This reference documents the supported public surface only.

## Global Shape

```text
kfp-workflow [--json]
├── spec
│   └── validate
├── pipeline
│   ├── compile
│   ├── submit
│   ├── get
│   ├── list
│   ├── wait
│   ├── terminate
│   ├── logs
│   └── list-experiments
├── benchmark
│   ├── compile
│   ├── submit
│   ├── list
│   ├── get
│   └── download
├── tune
│   ├── submit
│   ├── list
│   ├── get
│   ├── download
│   ├── space
│   └── logs
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
└── cluster
    └── bootstrap
```

## Interface Rules

- `--json` switches supported commands to machine-readable payloads.
- Workflow-style domains use backend IDs as the primary identifiers.
- Pipeline and benchmark commands accept full IDs or unique visible prefixes where noted.
- Tune commands use generated opaque Katib experiment IDs. The logical tune spec name is returned separately as `name`.
- `serve get` and `serve delete` take the service name as a positional argument, not `--name`.
- Hidden internal commands and internal HPO engine details are not part of this contract.

## `spec`

### `kfp-workflow spec validate`

- Purpose: load and validate one spec file.
- Required: `--spec <path>`
- Key options:
  - `--type pipeline|serving|tune|benchmark`
  - `--set key=value`
- Notes:
  - `pipeline`, `serving`, and `tune` validate the typed spec plus plugin-specific config.
  - `benchmark` validates the materialized benchmark payload.
  - Benchmark validation currently accepts YAML specs and Python benchmark definitions.

## `pipeline`

### `kfp-workflow pipeline compile`

- Purpose: compile a training pipeline into a KFP v2 YAML package.
- Required:
  - `--spec <path>`
  - `--output <path>`
- Key options:
  - `--set key=value`
- Notes:
  - The output path is always explicit; the command does not choose a default directory for you.

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
- Result:
  - human output prints `Submitted pipeline run: <run_id>`
  - JSON output uses `id` for the backend run ID
  - the command auto-compiles to `compiled/<spec-name>.yaml` before submission

### `kfp-workflow pipeline get <run_id>`

- Purpose: show detailed information for a pipeline run.
- Required argument: `<run_id>` or unique visible prefix.
- Key options:
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`
- JSON core fields:
  - `id`
  - `name`
  - `state`
  - `created_at`
  - `finished_at`
  - `namespace`
  - `workflow`

### `kfp-workflow pipeline list`

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

### `kfp-workflow pipeline wait <run_id>`

- Purpose: wait for a run to reach a terminal state.
- Required argument: `<run_id>` or unique visible prefix.
- Key options:
  - `--timeout <seconds>`
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

### `kfp-workflow pipeline terminate <run_id>`

- Purpose: cancel a running pipeline.
- Required argument: `<run_id>` or unique visible prefix.
- Key options:
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

### `kfp-workflow pipeline logs <run_id>`

- Purpose: view component pod logs associated with a pipeline run.
- Required argument: `<run_id>` or unique visible prefix.
- Key options:
  - `--step <substring>`
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

### `kfp-workflow pipeline list-experiments`

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
- Notes:
  - The output path is explicit.
  - Benchmark compile currently accepts YAML specs and Python benchmark definitions.

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
- Notes:
  - Benchmark submit currently accepts YAML specs and Python benchmark definitions.
  - The command auto-compiles to `compiled/<spec-name>.yaml` before submission.

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
- Required argument: `<run_id>` or unique visible prefix.
- Key options:
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

### `kfp-workflow benchmark download <run_id>`

- Purpose: download the benchmark result payload to a local JSON file.
- Required argument: `<run_id>` or unique visible prefix.
- Key options:
  - `--output <path>`
  - `--namespace <ns>`
  - `--host <url>`
  - `--user <identity>`
  - `--existing-token <token>`
  - `--cookies <cookie-header>`

## `tune`

### `kfp-workflow tune submit`

- Purpose: submit a Katib hyperparameter tuning experiment.
- Required: `--spec <path>`
- Key options:
  - `--set key=value`
  - `--output <path>`
  - `--dry-run`
  - `--wait`
- Behavior:
  - each submission gets a generated opaque Katib experiment ID
  - human output prints `Submitted tune experiment: <experiment_id>`
  - JSON output uses `id` for the experiment ID and `name` for the logical tune spec name
  - `--dry-run` prints or returns the generated manifest without creating the experiment
  - internal HPO engine details and the hidden `tune trial` runner are intentionally excluded from the public root contract

### `kfp-workflow tune list`

- Purpose: list managed Katib tune experiments.
- Key options:
  - `--namespace <ns>`

### `kfp-workflow tune get <experiment_id>`

- Purpose: show detailed info for one managed Katib tune experiment.
- Required argument: `<experiment_id>` or unique visible prefix.
- Key options:
  - `--namespace <ns>`
- JSON core fields:
  - `id`
  - `name`
  - `state`
  - `created_at`
  - `finished_at`
  - `namespace`
  - `best_value`
  - `best_params`
  - `trials`
  - `results`

### `kfp-workflow tune download <experiment_id>`

- Purpose: download the resolved tune result payload to a local JSON file.
- Required argument: `<experiment_id>` or unique visible prefix.
- Key options:
  - `--output <path>`
  - `--namespace <ns>`
  - `--from-pvc`
  - `--apply-best <pipeline-spec-path>`

### `kfp-workflow tune space`

- Purpose: display the resolved search space for a tune spec.
- Required: `--spec <path>`
- Key options:
  - `--set key=value`

### `kfp-workflow tune logs <experiment_id>`

- Purpose: show Katib trial logs for one tune experiment.
- Required argument: `<experiment_id>` or unique visible prefix.
- Key options:
  - `--namespace <ns>`
  - `--all`
  - `--trial <trial-name>`
  - `--tail <n>`

## `serve`

### `kfp-workflow serve create`

- Purpose: create a KServe `InferenceService` from a serving spec.
- Required: `--spec <path>`
- Key options:
  - `--dry-run`
  - `--wait`
  - `--timeout <seconds>`
- Notes:
  - The resulting service name is the serving spec's `metadata.name`.

### `kfp-workflow serve delete <name>`

- Purpose: delete a KServe `InferenceService`.
- Required argument: `<name>`
- Key options:
  - `--namespace <ns>`

### `kfp-workflow serve list`

- Purpose: list KServe `InferenceService` objects in one namespace.
- Key options:
  - `--namespace <ns>`

### `kfp-workflow serve get <name>`

- Purpose: show detailed service status, conditions, and events.
- Required argument: `<name>`
- Key options:
  - `--namespace <ns>`

## `registry`

### `kfp-workflow registry model register`

- Purpose: register a model in the file-backed model registry.
- Required:
  - `--name <model>`
  - `--version <version>`
  - `--uri <artifact-uri>`
- Key options:
  - `--framework <framework>`
  - `--description <text>`
  - `--registry-path <path>`

### `kfp-workflow registry model get`

- Purpose: retrieve a model from the file-backed model registry.
- Required:
  - `--name <model>`
- Key options:
  - `--version <version>`
  - `--registry-path <path>`

### `kfp-workflow registry model list`

- Purpose: list all models in the file-backed model registry.
- Key options:
  - `--registry-path <path>`

### `kfp-workflow registry dataset register`

- Purpose: register a dataset in the PVC-backed dataset registry.
- Required:
  - `--name <dataset>`
  - `--pvc-name <pvc>`
  - `--subpath <path>`
- Key options:
  - `--version <version>`
  - `--description <text>`
  - `--registry-path <path>`

### `kfp-workflow registry dataset get`

- Purpose: retrieve a dataset from the PVC-backed dataset registry.
- Required:
  - `--name <dataset>`
- Key options:
  - `--version <version>`
  - `--registry-path <path>`

### `kfp-workflow registry dataset list`

- Purpose: list all datasets in the PVC-backed dataset registry.
- Key options:
  - `--registry-path <path>`

## `cluster`

### `kfp-workflow cluster bootstrap`

- Purpose: create storage PVC manifests from a pipeline, benchmark, or tune spec and optionally apply them.
- Required: `--spec <path>`
- Key options:
  - `--type pipeline|benchmark|tune`
  - `--dry-run`
- Notes:
  - Pipeline and tune currently use YAML specs.
  - Benchmark bootstrap currently accepts YAML specs and Python benchmark definitions.
