# kfp-workflow

`kfp-workflow` is a Kubeflow-oriented orchestration project for three related jobs:

- compile and submit KFP v2 training pipelines
- deploy trained models with KServe
- run benchmark workflows against temporary KServe deployments

The repo also contains standalone model packages under `models/` that supply the model-specific training and HPO logic used by the root project.

## What Is Integrated Today

The root plugin registry currently supports these model plugins:

- `mambasl-cmapss`
- `mrhysp-cmapss`
- `softs-cmapss`

`models/timemixer-new` is present as a standalone package, but it is not registered in the root `kfp-workflow` plugin registry as of this revision.

## Repository Map

- `src/kfp_workflow/`: root Python package, CLI, specs, pipeline compiler, serving, benchmark, tuning, registries
- `configs/`: example pipeline, serving, tuning, and benchmark specs
- `examples/`: Korean-language tutorial track
- `models/`: standalone model packages and research workflows
- `docker/`: unified container image for the root workflow
- `pipelines/`: landing area for compiled KFP YAML

For the concise tree and documentation inventory, see [PROJECT.md](/home/scouter/proj_2026_1_etri/test/PROJECT.md).
For day-to-day commands and operational procedures, see [OPERATIONS.md](/home/scouter/proj_2026_1_etri/test/OPERATIONS.md).
For the exact public CLI hierarchy, see [CLI_COMMAND_TREE.md](/home/scouter/proj_2026_1_etri/test/CLI_COMMAND_TREE.md).

## Quick Start

```bash
make venv
make install
make test
make spec-validate
make compile-pipeline
```

Common first validation commands:

```bash
kfp-workflow spec validate --spec configs/pipelines/mambasl_cmapss_smoke.yaml
kfp-workflow spec validate --spec configs/serving/mambasl_cmapss_serve.yaml --type serving
kfp-workflow spec validate --spec configs/tuning/mambasl_cmapss_tune.yaml --type tune
kfp-workflow spec validate --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml --type benchmark
```

## CLI Reference

The canonical public command tree is maintained in [CLI_COMMAND_TREE.md](/home/scouter/proj_2026_1_etri/test/CLI_COMMAND_TREE.md).

Use that file for:

- exact command nesting
- callback-style `tune` submission behavior
- public versus intentionally undocumented hidden commands
- key required args and high-signal options per command

## Core Workflows

### 1. Training pipelines

Training specs live under `configs/pipelines/` and compile into a fixed five-stage DAG:

```text
load_data -> preprocess -> train -> evaluate -> save_model
```

Compile or submit:

```bash
kfp-workflow pipeline compile \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --output pipelines/mambasl_cmapss_smoke.yaml

kfp-workflow pipeline submit \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml
```

### 2. Serving

Serving specs live under `configs/serving/` and create KServe `InferenceService` objects.

```bash
kfp-workflow serve create --spec configs/serving/mambasl_cmapss_serve.yaml --wait
kfp-workflow serve list
kfp-workflow serve get --name mambasl-cmapss-serving
```

### 3. Hyperparameter tuning

Tune specs live under `configs/tuning/`. The default user-facing flow is Katib submission via `kfp-workflow tune --spec ...`.

```bash
kfp-workflow tune space --spec configs/tuning/mambasl_cmapss_tune.yaml
kfp-workflow tune --spec configs/tuning/mambasl_cmapss_tune.yaml --dry-run
kfp-workflow tune --spec configs/tuning/mambasl_cmapss_tune.yaml --wait
kfp-workflow tune status
kfp-workflow tune results <experiment-id-prefix>
```

The hidden legacy aliases `tune run`, `tune katib`, and `tune show-space` still exist for compatibility, but the documentation treats them as internal or deprecated.

### 4. Benchmarks

Benchmark specs live under `configs/benchmarks/` and deploy a temporary inference service, replay a scenario, collect metrics, then clean up.

```bash
kfp-workflow benchmark compile \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml \
  --output pipelines/mambasl_cmapss_kepler_smoke.yaml

kfp-workflow benchmark submit \
  --spec configs/benchmarks/mambasl_cmapss_kepler_smoke.yaml
```

## Specs and Overrides

The root CLI works from YAML specs:

- `configs/pipelines/`: `PipelineSpec`
- `configs/serving/`: `ServingSpec`
- `configs/tuning/`: `TuneSpec`
- `configs/benchmarks/`: `BenchmarkSpec`

All of these support Helm-style `--set` overrides at validation or submission time where implemented:

```bash
kfp-workflow pipeline compile \
  --spec configs/pipelines/mambasl_cmapss_smoke.yaml \
  --output pipelines/fd003.yaml \
  --set dataset.config.fd[0].fd_name=FD003 \
  --set train.max_epochs=10
```

## Operational Notes

- Default local KFP API host: `http://127.0.0.1:8888`
- Default namespace in many examples: `kubeflow-user-example-com`
- Root package requires Python `>=3.10`
- The unified Docker image is built from [docker/Dockerfile](/home/scouter/proj_2026_1_etri/test/docker/Dockerfile)
- `cluster bootstrap` creates PVCs for pipeline, benchmark, or tune storage from a spec

## Tutorials

The guided learning path is in Korean under [examples/README.md](/home/scouter/proj_2026_1_etri/test/examples/README.md).

Recommended order:

1. [00_프로젝트_개요.md](/home/scouter/proj_2026_1_etri/test/examples/00_%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8_%EA%B0%9C%EC%9A%94.md)
2. [01_설치_및_설정.md](/home/scouter/proj_2026_1_etri/test/examples/01_%EC%84%A4%EC%B9%98_%EB%B0%8F_%EC%84%A4%EC%A0%95.md)
3. [02_스펙_파일_작성_및_검증.md](/home/scouter/proj_2026_1_etri/test/examples/02_%EC%8A%A4%ED%8E%99_%ED%8C%8C%EC%9D%BC_%EC%9E%91%EC%84%B1_%EB%B0%8F_%EA%B2%80%EC%A6%9D.md)
4. [03_파이프라인_컴파일_및_제출.md](/home/scouter/proj_2026_1_etri/test/examples/03_%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8_%EC%BB%B4%ED%8C%8C%EC%9D%BC_%EB%B0%8F_%EC%A0%9C%EC%B6%9C.md)
5. [11_하이퍼파라미터_튜닝.md](/home/scouter/proj_2026_1_etri/test/examples/11_%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0_%ED%8A%9C%EB%8B%9D.md)
6. [12_벤치마크_워크플로우.md](/home/scouter/proj_2026_1_etri/test/examples/12_%EB%B2%A4%EC%B9%98%EB%A7%88%ED%81%AC_%EC%9B%8C%ED%81%AC%ED%94%8C%EB%A1%9C%EC%9A%B0.md)

## Model Packages

Maintained package-local docs:

- [models/mambasl-new/README.md](/home/scouter/proj_2026_1_etri/test/models/mambasl-new/README.md)
- [models/multirocket-new/README.md](/home/scouter/proj_2026_1_etri/test/models/multirocket-new/README.md)
- [models/softs-new/README.md](/home/scouter/proj_2026_1_etri/test/models/softs-new/README.md)
- [models/timemixer-new/README.md](/home/scouter/proj_2026_1_etri/test/models/timemixer-new/README.md)

The legacy upstream-style `TimeMixer/` directory is not part of the maintained root workflow documentation set.
