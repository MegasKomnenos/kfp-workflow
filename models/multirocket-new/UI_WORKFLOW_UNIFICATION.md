# UI and Workflow Unification Audit

This note records the current operator-facing relationship between `multirocket-new` and the other maintained `*-new` packages.

## Current Status

The packages are intentionally similar, but they are not fully uniform today.

Shared package-local CLI families:

- `spec`
- `train`
- `report`
- `pipeline`
- `katib`

Shared package-local commands across all maintained `*-new` packages:

- `spec validate`
- `train run`
- `train katib-trial`
- `report summarize`
- `pipeline compile`
- `katib render`
- `katib submit`

`multirocket-new` currently exposes two additional standalone operational commands that the other maintained packages do not expose:

- `pipeline submit`
- `cluster bootstrap`

## What Is Actually Unified

- Experiment-spec layout is aligned around `configs/experiments/` and `configs/search_spaces/`.
- The package-local Katib flow is aligned around `katib render` and `katib submit`.
- The package-local compile artifact convention is aligned around explicit `--output` paths, typically under `compiled/`.
- The root integrated workflow still uses the root `kfp-workflow` docs as the canonical source for integrated operation.

## What Is Intentionally Different

- `multirocket-new` can submit pipelines and bootstrap PVCs directly as a standalone package.
- `mambasl-new`, `softs-new`, and `timemixer-new` stop at package-local compilation and Katib manifest handling.
- Root integration is handled through root plugin adapters, not by trying to normalize all standalone package CLIs into one identical surface.

## Maintenance Rule

Treat this file as a factual status note, not as a target-state guarantee.

If package-local CLIs converge or diverge further, update:

- [README.md](/home/scouter/proj_2026_1_etri/test/models/multirocket-new/README.md)
- [OPERATIONS.md](/home/scouter/proj_2026_1_etri/test/models/multirocket-new/OPERATIONS.md)
- this file
