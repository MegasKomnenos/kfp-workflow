# Project Structure

This file is the maintained directory map for the root project. It is intentionally concise and focuses on the directories and files that a maintainer or operator is expected to use.

## Root Tree

```text
test/
├── AGENTS.md                # Agent instructions for this repo
├── CLAUDE.md                # Alternate agent guidance
├── GEMINI.md                # Alternate agent guidance
├── README.md                # Root overview, supported workflows, navigation hub
├── CLI_COMMAND_TREE.md      # Canonical public root CLI hierarchy and synopsis
├── PROJECT.md               # This maintained tree and component map
├── OPERATIONS.md            # Canonical local/cluster procedures
├── Makefile                 # Root setup, test, validation, compile helpers
├── pyproject.toml           # Root package metadata and dependencies
├── docker/
│   └── Dockerfile           # Unified workflow image used by root pipelines and serving
├── configs/
│   ├── pipelines/           # Training pipeline specs
│   ├── serving/             # KServe serving specs
│   ├── tuning/              # Katib tuning specs
│   └── benchmarks/          # Benchmark specs plus reusable scenarios and metrics
├── examples/
│   ├── README.md            # Korean tutorial index and learning path
│   └── *.md                 # Guided tutorials for install, specs, serving, tuning, benchmarks
├── kubeflow/
│   └── pvc/                 # Example PVC manifests
├── pipelines/
│   └── README.md            # Generated pipeline output landing area
├── scripts/
│   └── build_image.sh       # Helper for container image builds
├── src/
│   └── kfp_workflow/
│       ├── cli/             # Typer CLI, shared workflow helpers, and output formatting
│       ├── pipeline/        # Training pipeline compile and submit logic
│       ├── benchmark/       # Benchmark compile, runtime, history, result handling
│       ├── components/      # KFP pipeline component entrypoints
│       ├── plugins/         # Root-integrated model plugin adapters
│       ├── tune/            # Search-space resolution, Katib manifests, result handling
│       ├── serving/         # KServe CRUD and custom predictor
│       ├── registry/        # Model and dataset registry implementations
│       ├── config_override.py
│       ├── specs.py
│       └── utils.py
├── models/
│   ├── mambasl-new/         # Standalone MambaSL research package
│   ├── multirocket-new/     # Standalone MR-HY-SP research package
│   ├── softs-new/           # Standalone SOFTS research package
│   └── timemixer-new/       # Standalone TimeMixer research package, not root-integrated yet
├── results/                 # Example or collected result artifacts
├── tests/                   # Root regression suite
└── TimeMixer/               # Legacy upstream-style source snapshot, not root-maintained docs
```

## Documentation Inventory

- `README.md`: entry point for the integrated root project
- `CLI_COMMAND_TREE.md`: canonical public `kfp-workflow` command hierarchy
- `OPERATIONS.md`: repeatable procedures, command patterns, and deployment notes
- `examples/README.md`: tutorial navigation in Korean
- `models/*/README.md`: package-local overviews and current integration status
- `models/*/PROJECT.md`: package-local tree summaries
- `models/*/OPERATIONS.md`: package-local workflow guidance where maintained

## Runtime and Code Boundaries

- Root code under `src/kfp_workflow/` is the orchestration layer.
- Model packages under `models/` own model-specific training logic, search spaces, and some package-local Kubeflow helpers.
- Only the plugins explicitly registered in `src/kfp_workflow/plugins/__init__.py` are available to root specs.
- Tutorial docs under `examples/` describe the root workflow, not every standalone package feature.

## Generated or Secondary Areas

- `pipelines/` holds compiled KFP YAML and should be treated as generated output.
- `results/` contains result artifacts and examples, not source-of-truth configuration.
- `TimeMixer/` is a separate legacy code drop and is not the maintained entrypoint for `models/timemixer-new`.
