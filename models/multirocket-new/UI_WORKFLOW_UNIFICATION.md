# UI and Workflow Unification Audit

This document records the operator-facing unification checklist between `multirocket-new` and `mambasl-new`.

## Evaluation scale

- `Pass`: same operator-facing contract
- `Partial`: same intent, but different visible behavior, wording, or artifact convention
- `Fail`: missing or incompatible operator-facing behavior

## Checklist and current status

- `Pass`: top-level CLI families match: `spec`, `train`, `report`, `pipeline`, `katib`, `cluster`
- `Pass`: shared subcommands match: `spec validate`, `train run`, `train katib-trial`, `report summarize`, `pipeline compile`, `pipeline submit`, `katib render`, `katib submit`, `cluster bootstrap`
- `Pass`: shared flag families match for workflow control: `--spec`, `--dataset`, `--output`, `--namespace`, `--host`, `--existing-token`, `--cookies`, `--dry-run`
- `Pass`: canonical config layout matches: `configs/experiments/` and `configs/search_spaces/`
- `Pass`: workflow stage names match: `hpo`, `final_train_eval`, `ablation_sweep`, `aggregate_reports`
- `Pass`: pipeline submit flow matches: compile from spec, port-forward via runtime fields, optional auth overrides
- `Pass`: Katib render and submit flow matches: render from spec, optional dry-run submit, shared stdout metrics contract
- `Pass`: PVC bootstrap flow matches: spec-driven PVC names, sizes, mounts, optional seeding, dry-run manifest inspection
- `Pass`: documented runbook order matches: validate, local run, compile and submit, bootstrap when PVC-backed, render or submit Katib
- `Pass`: recommended compile artifact convention matches: `compiled/<spec-name>.yaml`
- `Pass`: automated protocol coverage exists for schema loading, pipeline compilation, Katib manifest generation, and bootstrap dry-run

## Allowed differences

- Model-specific training fields under `train_defaults` and search-space contents differ by algorithm and are not counted against UI or workflow unification.
- Resource defaults can differ when they do not change the operator workflow shape.
