# Documentation Rules

This file records the documentation lessons learned from repeated drift audits in this repo.

It is a maintenance policy for all maintained docs, tutorials, agent guidance files, and user-facing CLI/help text.

## Scope and Source of Truth

- Treat current code and live CLI/help output as the primary sources of truth.
- Never treat older markdown as authoritative just because multiple docs repeat the same claim.
- When runtime help is unavailable for a package because of missing dependencies, verify claims from source.
- Root docs must describe the integrated `kfp-workflow` surface. Package docs must describe package-local truth.

## Public vs Internal Boundaries

- Document the public operator interface, not hidden implementation helpers.
- Hidden commands are not part of the public contract unless they are explicitly promoted.
- Do not leak internal tuning implementation details into root operator docs.
- For root tuning docs, the public UX is Katib experiment oriented: `submit`, `list`, `get`, `download`, `space`, `logs`.
- Internal HPO engine details such as Optuna usage, hidden `tune trial`, or plugin-internal trial orchestration belong in source or developer-focused internal notes, not root operator docs.

## Command and Path Accuracy

- Every documented command example must be checked against live help or the exact command implementation.
- Distinguish explicit example output paths from actual defaults.
- Do not call a directory the “default landing area” unless code writes there by default.
- If `submit` auto-generates artifacts in `compiled/` while examples write to `pipelines/`, document both roles explicitly.
- If a command accepts more than the help text implies, fix the help text or document the broader accepted input exactly.

## Behavior Claims

- Document configurable defaults as defaults, not universal behavior.
- If a benchmark example uses `cleanup: true`, say the example cleans up; do not claim benchmark workflows always clean up.
- If the project examples default to `kfp-workflow:latest`, say they default to it; do not claim the entire system always uses exactly one image.
- If a service endpoint depends on `metadata.name`, document the real name-resolution rule rather than a plugin-name shortcut.

## Storage and Registry Semantics

- Describe storage backends exactly.
- The model registry is file-backed.
- The dataset registry is PVC-backed JSON storage.
- Do not collapse those into one generic “file-based registry” description.

## Spec Format Semantics

- State accepted spec forms exactly.
- Pipeline, serving, and tune docs currently use YAML specs.
- Benchmark docs must acknowledge that benchmark commands currently accept YAML or Python benchmark definitions where the code supports both.
- If examples only show YAML, say they are examples, not the entire accepted surface.

## Procedure Claims

- Never say a procedure is “already documented” unless the repo contains that maintained procedure and links to it directly.
- If a cluster import or deployment path is environment-specific and not standardized in repo docs, say that plainly.

## Maintenance Workflow

- When changing code that affects user-visible behavior, update:
  - `README.md`
  - `OPERATIONS.md`
  - `PROJECT.md`
  - `CLI_COMMAND_TREE.md`
  - affected tutorials under `examples/`
  - affected package docs under `models/*/`
  - user-facing CLI help/docstrings when they are part of the drift
- When root documentation rules change, update this file.

## Audit Checklist

- Re-read the relevant command implementation before editing docs.
- Re-check live `--help` output for every changed command family.
- Search maintained markdown for stale command forms and stale conceptual terms.
- Verify all referenced paths and linked procedures actually exist.
- Keep root docs task-oriented and contract-focused.
- Keep package docs honest about integration status and package-local-only behavior.
