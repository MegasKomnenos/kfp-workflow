from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ..experiment import emit_katib_metrics, load_params_json, run_dataset_pipeline
from ..kubeflow.bootstrap import bootstrap_storage
from ..kubeflow.client import submit_pipeline
from ..kubeflow.katib import build_experiment_manifest, create_experiment, dump_manifest
from ..kubeflow.pipeline import compile_pipeline
from ..specs import ExperimentSpec, load_spec
from ..utils import dump_json


def _coerce_values(payload):
    out = {}
    for key, value in payload.items():
        if isinstance(value, (bool, int, float)):
            out[key] = value
            continue
        if value == "True":
            out[key] = True
            continue
        if value == "False":
            out[key] = False
            continue
        try:
            if "." in str(value):
                out[key] = float(value)
            else:
                out[key] = int(value)
            continue
        except (TypeError, ValueError):
            out[key] = value
    return out


def cmd_spec_validate(args) -> None:
    spec = load_spec(args.spec)
    print(spec.metadata.name)


def cmd_train_run(args) -> None:
    spec = load_spec(args.spec) if args.spec else ExperimentSpec.model_validate_json(args.spec_json)
    datasets = [args.dataset] if args.dataset else list(spec.datasets.items)
    explicit_params = _coerce_values(load_params_json(args.params_json)) if args.params_json else None
    for dataset in datasets:
        out_dir = Path(args.output_dir or spec.outputs.local_results_dir) / spec.metadata.name / dataset.lower()
        result = run_dataset_pipeline(
            spec,
            dataset,
            out_dir,
            explicit_params=explicit_params,
            run_hpo_stage=args.run_hpo,
            run_ablation_stage=args.run_ablations,
            kubeflow=False,
        )
        print(json.dumps({"dataset": dataset, "metrics": result["final"]["test_metrics"]}, indent=2))


def cmd_train_katib_trial(args) -> None:
    spec = load_spec(args.spec) if args.spec else ExperimentSpec.model_validate_json(args.spec_json)
    explicit_params = _coerce_values(load_params_json(args.trial_params_json))
    out_dir = Path(args.output_dir or spec.outputs.local_results_dir) / "katib-trials" / args.dataset.lower()
    result = run_dataset_pipeline(
        spec,
        args.dataset,
        out_dir,
        explicit_params=explicit_params,
        run_hpo_stage=False,
        run_ablation_stage=False,
        kubeflow=True,
    )
    emit_katib_metrics(
        result["final"]["val_metrics"],
        selection_metric=spec.train_defaults.selection_metric,
        score_weight=spec.train_defaults.score_weight,
    )


def cmd_report_summarize(args) -> None:
    run_dir = Path(args.run_dir)
    payload = {}
    for path in sorted(run_dir.glob("*_result.json")):
        payload[path.stem] = json.loads(path.read_text())
    out_path = run_dir / "collected_results.json"
    dump_json(out_path, payload)
    print(out_path)


def cmd_pipeline_compile(args) -> None:
    spec = load_spec(args.spec)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    compile_pipeline(spec, str(output))
    print(output)


def cmd_pipeline_submit(args) -> None:
    spec = load_spec(args.spec)
    run_id = submit_pipeline(
        spec,
        namespace=args.namespace,
        host=args.host,
        existing_token=args.existing_token,
        cookies=args.cookies,
    )
    print(run_id)


def cmd_katib_render(args) -> None:
    spec = load_spec(args.spec)
    manifest = build_experiment_manifest(spec, args.dataset)
    output = Path(args.output or f"kubeflow/katib/{spec.metadata.name}-{args.dataset.lower()}.yaml")
    dump_manifest(output, manifest)
    print(output)


def cmd_katib_submit(args) -> None:
    spec = load_spec(args.spec)
    manifest = build_experiment_manifest(spec, args.dataset)
    output = Path(args.output or f"kubeflow/katib/{spec.metadata.name}-{args.dataset.lower()}.yaml")
    dump_manifest(output, manifest)
    print(output)
    if not args.dry_run:
        subprocess.run(["kubectl", "apply", "-f", "-"], check=True, text=True, input=json.dumps(manifest))


def cmd_cluster_bootstrap(args) -> None:
    spec = load_spec(args.spec)
    manifests = bootstrap_storage(spec, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(manifests, indent=2))
    else:
        print(spec.runtime.namespace)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multirocket-new")
    subparsers = parser.add_subparsers(dest="command", required=True)

    spec_parser = subparsers.add_parser("spec")
    spec_sub = spec_parser.add_subparsers(dest="spec_command", required=True)
    spec_validate = spec_sub.add_parser("validate")
    spec_validate.add_argument("--spec", required=True)
    spec_validate.set_defaults(func=cmd_spec_validate)

    train_parser = subparsers.add_parser("train")
    train_sub = train_parser.add_subparsers(dest="train_command", required=True)

    train_run = train_sub.add_parser("run")
    train_run.add_argument("--spec")
    train_run.add_argument("--spec-json")
    train_run.add_argument("--dataset")
    train_run.add_argument("--output-dir", default=None)
    train_run.add_argument("--params-json", default=None)
    train_run.add_argument("--run-hpo", action="store_true")
    train_run.add_argument("--run-ablations", action="store_true")
    train_run.set_defaults(func=cmd_train_run)

    katib_trial = train_sub.add_parser("katib-trial")
    katib_trial.add_argument("--spec")
    katib_trial.add_argument("--spec-json")
    katib_trial.add_argument("--dataset", required=True)
    katib_trial.add_argument("--trial-params-json", required=True)
    katib_trial.add_argument("--output-dir", default=None)
    katib_trial.set_defaults(func=cmd_train_katib_trial)

    report_parser = subparsers.add_parser("report")
    report_sub = report_parser.add_subparsers(dest="report_command", required=True)
    report_summarize = report_sub.add_parser("summarize")
    report_summarize.add_argument("--run-dir", required=True)
    report_summarize.set_defaults(func=cmd_report_summarize)

    pipeline_parser = subparsers.add_parser("pipeline")
    pipeline_sub = pipeline_parser.add_subparsers(dest="pipeline_command", required=True)
    pipeline_compile = pipeline_sub.add_parser("compile")
    pipeline_compile.add_argument("--spec", required=True)
    pipeline_compile.add_argument("--output", required=True)
    pipeline_compile.set_defaults(func=cmd_pipeline_compile)

    pipeline_submit = pipeline_sub.add_parser("submit")
    pipeline_submit.add_argument("--spec", required=True)
    pipeline_submit.add_argument("--namespace", default=None)
    pipeline_submit.add_argument("--host", default=None)
    pipeline_submit.add_argument("--existing-token", default="")
    pipeline_submit.add_argument("--cookies", default="")
    pipeline_submit.set_defaults(func=cmd_pipeline_submit)

    katib_parser = subparsers.add_parser("katib")
    katib_sub = katib_parser.add_subparsers(dest="katib_command", required=True)
    katib_render = katib_sub.add_parser("render")
    katib_render.add_argument("--spec", required=True)
    katib_render.add_argument("--dataset", required=True)
    katib_render.add_argument("--output", default=None)
    katib_render.set_defaults(func=cmd_katib_render)

    katib_submit = katib_sub.add_parser("submit")
    katib_submit.add_argument("--spec", required=True)
    katib_submit.add_argument("--dataset", required=True)
    katib_submit.add_argument("--output", default=None)
    katib_submit.add_argument("--dry-run", action="store_true")
    katib_submit.set_defaults(func=cmd_katib_submit)

    cluster_parser = subparsers.add_parser("cluster")
    cluster_sub = cluster_parser.add_subparsers(dest="cluster_command", required=True)
    cluster_bootstrap = cluster_sub.add_parser("bootstrap")
    cluster_bootstrap.add_argument("--spec", required=True)
    cluster_bootstrap.add_argument("--dry-run", action="store_true")
    cluster_bootstrap.set_defaults(func=cmd_cluster_bootstrap)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
