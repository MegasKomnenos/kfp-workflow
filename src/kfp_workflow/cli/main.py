"""Typer CLI application for kfp-workflow."""

from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from kfp_server_api.exceptions import ApiException

# Suppress noisy third-party deprecation warnings that are not actionable
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", category=FutureWarning, module="kfp")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_json_output = False


def _coerce_json_scalar_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce JSON scalar strings produced by Katib back into Python types."""
    coerced: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (bool, int, float)) or value is None:
            coerced[key] = value
            continue
        if value == "True":
            coerced[key] = True
            continue
        if value == "False":
            coerced[key] = False
            continue
        try:
            if "." in str(value):
                coerced[key] = float(value)
            else:
                coerced[key] = int(value)
            continue
        except (TypeError, ValueError):
            coerced[key] = value
    return coerced


def _run_state_str(state: object) -> str:
    """Extract run state as a plain string (handles enum or str)."""
    if state is None:
        return "UNKNOWN"
    return state.value if hasattr(state, "value") else str(state)


def _validate_plugin_config_or_exit(spec_dict: dict) -> None:
    """Abort CLI execution when plugin-specific config validation fails."""
    from kfp_workflow.config_override import validate_plugin_config

    errors = validate_plugin_config(spec_dict)
    if not errors:
        return

    for error in errors:
        typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1)


def _validate_serving_plugin_config_or_exit(spec_dict: dict) -> None:
    """Abort CLI execution when serving plugin-specific validation fails."""
    from kfp_workflow.config_override import validate_serving_plugin_config

    errors = validate_serving_plugin_config(spec_dict)
    if not errors:
        return

    for error in errors:
        typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1)


def _emit_validated_spec_output(name: str, payload: dict) -> None:
    """Emit validated spec output honoring the CLI JSON flag."""
    if _json_output:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    typer.echo(f"Spec '{name}' validated successfully.")
    typer.echo(json.dumps(payload, indent=2, default=str))


def _find_workflow_for_run(run_id: str, namespace: str) -> Optional[Dict[str, Any]]:
    """Return the Argo Workflow object for a KFP run, if present."""
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    try:
        k8s_config.load_kube_config()
        api = k8s_client.CustomObjectsApi()
        result = api.list_namespaced_custom_object(
            group="argoproj.io",
            version="v1alpha1",
            namespace=namespace,
            plural="workflows",
            label_selector=f"pipeline/runid={run_id}",
        )
    except Exception:
        return None
    items = result.get("items", [])
    if not items:
        return None
    items.sort(key=lambda item: item.get("metadata", {}).get("creationTimestamp", ""))
    return items[-1]


def _workflow_summary(workflow: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize a workflow object for CLI output."""
    if not workflow:
        return None

    status = workflow.get("status", {})
    nodes = status.get("nodes", {}) or {}
    pending = []
    failed = []
    for node in nodes.values():
        phase = node.get("phase", "")
        display = node.get("displayName") or node.get("name") or ""
        if phase in {"Pending", "Running"} and display:
            pending.append(display)
        elif phase in {"Failed", "Error"} and display:
            failed.append(display)

    return {
        "name": workflow.get("metadata", {}).get("name", ""),
        "phase": status.get("phase", ""),
        "progress": status.get("progress", ""),
        "finished_at": status.get("finishedAt", ""),
        "message": status.get("message", ""),
        "pending_nodes": sorted(set(pending))[:10],
        "failed_nodes": sorted(set(failed))[:10],
    }


def _augment_run_payload(run: Any, workflow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the JSON payload returned by run-get and reused by wait."""
    payload = {
        "run_id": run.run_id,
        "display_name": run.display_name,
        "state": _run_state_str(run.state),
        "created_at": str(run.created_at),
        "finished_at": str(run.finished_at),
        "experiment_id": run.experiment_id,
        "error": str(run.error) if run.error else None,
    }
    workflow_summary = _workflow_summary(workflow)
    if workflow_summary:
        payload.update({
            "workflow_name": workflow_summary["name"],
            "workflow_phase": workflow_summary["phase"],
            "workflow_progress": workflow_summary["progress"],
            "workflow_finished_at": workflow_summary["finished_at"],
            "workflow_message": workflow_summary["message"],
            "pending_nodes": workflow_summary["pending_nodes"],
            "failed_nodes": workflow_summary["failed_nodes"],
        })
    return payload


def _iter_runs(
    client: Any,
    *,
    namespace: str,
    experiment_id: Optional[str] = None,
    page_size: int = 100,
    sort_by: str = "created_at desc",
) -> List[Any]:
    """Return all visible runs for ID-prefix resolution."""
    runs: List[Any] = []
    page_token = ""
    while True:
        response = client.list_runs(
            page_size=page_size,
            sort_by=sort_by,
            experiment_id=experiment_id,
            namespace=namespace,
            page_token=page_token,
        )
        batch = list(response.runs or [])
        runs.extend(batch)
        raw_next_page_token = getattr(response, "next_page_token", "")
        page_token = raw_next_page_token if isinstance(raw_next_page_token, str) else ""
        if not page_token:
            return runs


def _iter_experiments(
    client: Any,
    *,
    namespace: str,
    page_size: int = 100,
    sort_by: str = "created_at desc",
) -> List[Any]:
    """Return all visible experiments for ID-prefix resolution."""
    experiments: List[Any] = []
    page_token = ""
    while True:
        response = client.list_experiments(
            page_size=page_size,
            sort_by=sort_by,
            namespace=namespace,
            page_token=page_token,
        )
        batch = list(response.experiments or [])
        experiments.extend(batch)
        raw_next_page_token = getattr(response, "next_page_token", "")
        page_token = raw_next_page_token if isinstance(raw_next_page_token, str) else ""
        if not page_token:
            return experiments


def _resolve_unique_id_prefix(
    raw_id: str,
    candidates: List[str],
    *,
    kind: str,
) -> str:
    """Resolve a user-supplied full ID or unique prefix."""
    exact_matches = [candidate for candidate in candidates if candidate == raw_id]
    if exact_matches:
        return exact_matches[0]

    prefix_matches = [candidate for candidate in candidates if candidate.startswith(raw_id)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if not prefix_matches:
        raise typer.BadParameter(f"No {kind} found matching ID or prefix '{raw_id}'.")
    preview = ", ".join(match[:8] for match in prefix_matches[:5])
    raise typer.BadParameter(
        f"ID prefix '{raw_id}' matches multiple {kind}s: {preview}"
    )


def _resolve_run(client: Any, *, run_id: str, namespace: str) -> Any:
    """Resolve a full run ID or unique prefix and return the run object."""
    try:
        run = client.get_run(run_id=run_id)
        resolved = getattr(run, "run_id", None)
        if isinstance(resolved, str) and resolved:
            return run
    except ApiException as exc:
        if getattr(exc, "status", None) != 404:
            raise
    candidates = [str(run.run_id) for run in _iter_runs(client, namespace=namespace) if getattr(run, "run_id", None)]
    resolved_run_id = _resolve_unique_id_prefix(run_id, candidates, kind="run")
    return client.get_run(run_id=resolved_run_id)


def _resolve_experiment_id(client: Any, *, experiment_id: str, namespace: str) -> str:
    """Resolve a full experiment ID or unique prefix against visible experiments."""
    candidates = [
        str(experiment.experiment_id)
        for experiment in _iter_experiments(client, namespace=namespace)
        if getattr(experiment, "experiment_id", None)
    ]
    return _resolve_unique_id_prefix(experiment_id, candidates, kind="experiment")


def _log_for_pod(v1: Any, pod: Any, namespace: str) -> str:
    """Read logs from a pod, preferring the main container when present."""
    container_names = [c.name for c in (pod.spec.containers or [])]
    container = "main" if "main" in container_names else (container_names[0] if container_names else None)
    kwargs = {
        "name": pod.metadata.name,
        "namespace": namespace,
    }
    if container:
        kwargs["container"] = container
    return v1.read_namespaced_pod_log(**kwargs)

# ---------------------------------------------------------------------------
# App and sub-apps
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="kfp-workflow",
    help="KFP v2 training pipeline and KServe serving workflow manager.",
    add_completion=False,
)

pipeline_app = typer.Typer(help="Compile and submit training pipelines.")
benchmark_app = typer.Typer(help="Compile and submit benchmark workflows.")
run_app = typer.Typer(help="Monitor and manage pipeline runs.")
experiment_app = typer.Typer(help="Manage pipeline experiments.")
serve_app = typer.Typer(help="Create and manage KServe InferenceServices.")
registry_app = typer.Typer(help="Register and retrieve models and datasets.")
model_reg_app = typer.Typer(help="Model registry operations.")
dataset_reg_app = typer.Typer(help="Dataset registry operations.")
cluster_app = typer.Typer(help="Cluster bootstrapping operations.")
spec_app = typer.Typer(help="Spec validation.")
tune_app = typer.Typer(help="Hyperparameter tuning operations.")

app.add_typer(pipeline_app, name="pipeline")
app.add_typer(benchmark_app, name="benchmark")
pipeline_app.add_typer(run_app, name="run")
pipeline_app.add_typer(experiment_app, name="experiment")
app.add_typer(serve_app, name="serve")
app.add_typer(registry_app, name="registry")
registry_app.add_typer(model_reg_app, name="model")
registry_app.add_typer(dataset_reg_app, name="dataset")
app.add_typer(cluster_app, name="cluster")
app.add_typer(spec_app, name="spec")
app.add_typer(tune_app, name="tune")


@app.callback()
def main_callback(
    json_mode: bool = typer.Option(
        False, "--json", help="Output in JSON format.",
    ),
) -> None:
    """KFP v2 training pipeline and KServe serving workflow manager."""
    global _json_output
    _json_output = json_mode


# ---------------------------------------------------------------------------
# spec commands
# ---------------------------------------------------------------------------

@spec_app.command("validate")
def cmd_spec_validate(
    spec: Path = typer.Option(..., help="Path to a pipeline or serving YAML spec."),
    spec_type: str = typer.Option(
        "pipeline",
        "--type",
        help="Spec type: 'pipeline', 'serving', 'tune', or 'benchmark'.",
    ),
    set_values: List[str] = typer.Option(
        [], "--set",
        help="Override spec values before validation (e.g., --set train.batch_size=128).",
    ),
) -> None:
    """Load and validate a spec file."""
    from kfp_workflow.benchmark.materialize import (
        load_materialized_benchmark_spec,
    )
    from kfp_workflow.specs import (
        load_pipeline_spec_with_overrides,
        load_serving_spec_with_overrides,
    )

    try:
        materialized: Optional[Dict[str, Any]] = None
        if spec_type == "pipeline":
            loaded = load_pipeline_spec_with_overrides(spec, set_values or None)
            _validate_plugin_config_or_exit(loaded.model_dump())
        elif spec_type == "serving":
            loaded = load_serving_spec_with_overrides(spec, set_values or None)
            _validate_serving_plugin_config_or_exit(loaded.model_dump())
        elif spec_type == "tune":
            from kfp_workflow.specs import load_tune_spec_with_overrides
            loaded = load_tune_spec_with_overrides(spec, set_values or None)
            _validate_plugin_config_or_exit(loaded.model_dump())
        elif spec_type == "benchmark":
            loaded, materialized = load_materialized_benchmark_spec(spec, set_values or None)
        else:
            typer.echo(f"Unknown spec type: {spec_type}", err=True)
            raise typer.Exit(code=1)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    payload = materialized if spec_type == "benchmark" else loaded.model_dump()
    _emit_validated_spec_output(loaded.metadata.name, payload)


# ---------------------------------------------------------------------------
# benchmark commands
# ---------------------------------------------------------------------------


@benchmark_app.command("compile")
def cmd_benchmark_compile(
    spec: Path = typer.Option(..., help="Path to a benchmark YAML spec."),
    output: Path = typer.Option(..., help="Output path for compiled YAML."),
    set_values: List[str] = typer.Option(
        [], "--set",
        help="Override spec values (e.g., --set scenario.pipeline.config.interval_hz=2).",
    ),
) -> None:
    """Compile a benchmark workflow to a KFP v2 YAML package."""
    from kfp_workflow.benchmark.compiler import compile_benchmark
    from kfp_workflow.benchmark.materialize import load_materialized_benchmark_spec

    loaded, materialized = load_materialized_benchmark_spec(spec, set_values or None)
    result = compile_benchmark(loaded, materialized, output)
    typer.echo(f"Benchmark compiled to {result}")


@benchmark_app.command("submit")
def cmd_benchmark_submit(
    spec: Path = typer.Option(..., help="Path to a benchmark YAML spec."),
    namespace: Optional[str] = typer.Option(None, help="Kubernetes namespace override."),
    host: Optional[str] = typer.Option(None, help="KFP API host override."),
    user: Optional[str] = typer.Option(None, help="Kubeflow user identity header."),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
    set_values: List[str] = typer.Option(
        [], "--set",
        help="Override spec values (e.g., --set model.service_name=bench-smoke).",
    ),
) -> None:
    """Compile and submit a benchmark workflow to Kubeflow."""
    from kfp_workflow.benchmark.client import submit_benchmark
    from kfp_workflow.benchmark.materialize import load_materialized_benchmark_spec

    loaded, _ = load_materialized_benchmark_spec(spec, set_values or None)
    run_id = submit_benchmark(
        loaded,
        spec_path=spec,
        overrides=set_values or None,
        namespace=namespace,
        host=host,
        existing_token=existing_token,
        cookies=cookies,
        user=user,
    )
    typer.echo(f"Benchmark submitted. Run ID: {run_id}")


@benchmark_app.command("list")
def cmd_benchmark_list(
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace.",
    ),
    page_size: int = typer.Option(20, help="Number of runs to return."),
    sort_by: str = typer.Option(
        "created_at desc", help="Sort order (e.g. 'created_at desc').",
    ),
    host: str = typer.Option(
        "http://127.0.0.1:8888", help="KFP API host.",
    ),
    user: str = typer.Option(
        "user@example.com", help="Kubeflow user identity header.",
    ),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """List past benchmark runs."""
    from kfp_workflow.benchmark.history import (
        extract_benchmark_spec,
        find_workflow_for_run,
        is_benchmark_workflow,
    )
    from kfp_workflow.cli.output import print_json, print_table, style_run_state
    from kfp_workflow.pipeline.connection import kfp_connection

    with kfp_connection(
        namespace=namespace, host=host, user=user,
        existing_token=existing_token, cookies=cookies,
    ) as client:
        response = client.list_runs(
            page_size=page_size,
            sort_by=sort_by,
            namespace=namespace,
        )

    runs = response.runs or []
    items: List[Dict[str, Any]] = []
    for run in runs:
        workflow = find_workflow_for_run(run.run_id, namespace)
        if not workflow or not is_benchmark_workflow(workflow):
            continue
        benchmark_spec = extract_benchmark_spec(workflow) or {}
        workflow_summary = _workflow_summary(workflow) or {}
        items.append({
            "run_id": run.run_id,
            "display_name": run.display_name,
            "benchmark_name": benchmark_spec.get("metadata", {}).get("name", run.display_name),
            "state": _run_state_str(run.state),
            "created_at": str(run.created_at),
            "finished_at": str(run.finished_at),
            "workflow_name": workflow_summary.get("name", ""),
            "workflow_phase": workflow_summary.get("phase", ""),
        })

    if _json_output:
        print_json(items)
        return

    rows = []
    for item in items:
        rows.append((
            (item["run_id"] or "")[:8],
            item["benchmark_name"] or "",
            style_run_state(item["state"]),
            item["workflow_name"] or "",
            item["created_at"] or "",
            item["finished_at"] or "",
        ))
    print_table(
        title="Benchmark Runs",
        columns=["ID", "BENCHMARK", "STATE", "WORKFLOW", "CREATED", "FINISHED"],
        rows=rows,
    )


@benchmark_app.command("get")
def cmd_benchmark_get(
    run_id: str = typer.Argument(..., help="Benchmark run ID."),
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace.",
    ),
    host: str = typer.Option(
        "http://127.0.0.1:8888", help="KFP API host.",
    ),
    user: str = typer.Option(
        "user@example.com", help="Kubeflow user identity header.",
    ),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """Show detailed info for a benchmark run."""
    from kfp_workflow.benchmark.history import (
        extract_benchmark_spec,
        find_workflow_for_run,
        is_benchmark_workflow,
        resolve_results,
    )
    from kfp_workflow.cli.output import print_json, print_kv, style_run_state
    from kfp_workflow.pipeline.connection import kfp_connection

    with kfp_connection(
        namespace=namespace, host=host, user=user,
        existing_token=existing_token, cookies=cookies,
    ) as client:
        run = _resolve_run(client, run_id=run_id, namespace=namespace)
        resolved_run_id = run.run_id

    workflow = find_workflow_for_run(resolved_run_id, namespace)
    if not workflow or not is_benchmark_workflow(workflow):
        typer.echo(f"Run '{resolved_run_id}' is not a benchmark workflow.", err=True)
        raise typer.Exit(code=1)

    benchmark_spec = extract_benchmark_spec(workflow) or {}
    payload = _augment_run_payload(run, workflow)
    payload["benchmark_name"] = benchmark_spec.get("metadata", {}).get("name", run.display_name)

    try:
        results = resolve_results(
            workflow=workflow,
            benchmark_spec=benchmark_spec,
            namespace=namespace,
        )
        payload["results_path"] = results["results_path"]
        payload["results_summary"] = results["summary"]
    except Exception as exc:
        payload["results_error"] = str(exc)

    if _json_output:
        print_json(payload)
        return

    pairs = [
        ("Benchmark", payload["benchmark_name"] or ""),
        ("Run ID", run.run_id or ""),
        ("Name", run.display_name or ""),
        ("State", style_run_state(_run_state_str(run.state))),
        ("Created", str(run.created_at or "")),
        ("Finished", str(run.finished_at or "")),
        ("Experiment ID", run.experiment_id or ""),
    ]
    workflow_summary = _workflow_summary(workflow)
    if workflow_summary:
        pairs.extend([
            ("Workflow", workflow_summary["name"]),
            ("Workflow Phase", workflow_summary["phase"] or "(unknown)"),
            ("Workflow Progress", workflow_summary["progress"] or "(unknown)"),
        ])
    if "results_path" in payload:
        pairs.append(("Results Path", payload["results_path"]))
    if "results_summary" in payload:
        summary = payload["results_summary"]
        pairs.append(("Result Status", str(summary.get("status", ""))))
        pairs.append(("Request Count", str(summary.get("request_count", ""))))
        if "delta_joules" in summary:
            pairs.append(("Delta Joules", str(summary["delta_joules"])))
    if "results_error" in payload:
        pairs.append(("Results Error", payload["results_error"]))
    print_kv(pairs)


@benchmark_app.command("download")
def cmd_benchmark_download(
    run_id: str = typer.Argument(..., help="Benchmark run ID."),
    output: Optional[Path] = typer.Option(None, help="Write results JSON to this path."),
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace.",
    ),
    host: str = typer.Option(
        "http://127.0.0.1:8888", help="KFP API host.",
    ),
    user: str = typer.Option(
        "user@example.com", help="Kubeflow user identity header.",
    ),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """Download a benchmark run's results.json to the local machine."""
    from kfp_workflow.benchmark.history import (
        extract_benchmark_spec,
        find_workflow_for_run,
        is_benchmark_workflow,
        resolve_results,
    )
    from kfp_workflow.cli.output import print_json
    from kfp_workflow.pipeline.connection import kfp_connection
    from kfp_workflow.utils import dump_json, ensure_parent

    with kfp_connection(
        namespace=namespace, host=host, user=user,
        existing_token=existing_token, cookies=cookies,
    ) as client:
        run = _resolve_run(client, run_id=run_id, namespace=namespace)
        resolved_run_id = run.run_id

    workflow = find_workflow_for_run(resolved_run_id, namespace)
    if not workflow or not is_benchmark_workflow(workflow):
        typer.echo(f"Run '{resolved_run_id}' is not a benchmark workflow.", err=True)
        raise typer.Exit(code=1)

    benchmark_spec = extract_benchmark_spec(workflow) or {}
    benchmark_name = benchmark_spec.get("metadata", {}).get("name", run.display_name or "benchmark")
    results = resolve_results(
        workflow=workflow,
        benchmark_spec=benchmark_spec,
        namespace=namespace,
    )

    destination = output or Path.cwd() / f"{benchmark_name}-{resolved_run_id}.json"
    ensure_parent(destination)
    dump_json(results["payload"], destination)

    payload = {
        "run_id": resolved_run_id,
        "benchmark_name": benchmark_name,
        "results_path": results["results_path"],
        "output_path": str(destination),
    }
    if _json_output:
        print_json(payload)
        return

    typer.echo(f"Benchmark results downloaded to {destination}")


# ---------------------------------------------------------------------------
# pipeline commands
# ---------------------------------------------------------------------------

@pipeline_app.command("compile")
def cmd_pipeline_compile(
    spec: Path = typer.Option(..., help="Path to a pipeline YAML spec."),
    output: Path = typer.Option(..., help="Output path for compiled YAML."),
    set_values: List[str] = typer.Option(
        [], "--set",
        help="Override spec values (e.g., --set train.batch_size=128).",
    ),
) -> None:
    """Compile a training pipeline to a KFP v2 YAML package."""
    from kfp_workflow.pipeline.compiler import compile_pipeline
    from kfp_workflow.specs import load_pipeline_spec_with_overrides

    loaded = load_pipeline_spec_with_overrides(spec, set_values or None)
    _validate_plugin_config_or_exit(loaded.model_dump())
    result = compile_pipeline(loaded, output)
    typer.echo(f"Pipeline compiled to {result}")


@pipeline_app.command("submit")
def cmd_pipeline_submit(
    spec: Path = typer.Option(..., help="Path to a pipeline YAML spec."),
    namespace: Optional[str] = typer.Option(None, help="Kubernetes namespace override."),
    host: Optional[str] = typer.Option(None, help="KFP API host override."),
    user: Optional[str] = typer.Option(None, help="Kubeflow user identity header."),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
    set_values: List[str] = typer.Option(
        [], "--set",
        help="Override spec values (e.g., --set train.batch_size=128).",
    ),
) -> None:
    """Compile and submit a training pipeline to Kubeflow."""
    from kfp_workflow.pipeline.client import submit_pipeline
    from kfp_workflow.specs import load_pipeline_spec_with_overrides

    loaded = load_pipeline_spec_with_overrides(spec, set_values or None)
    _validate_plugin_config_or_exit(loaded.model_dump())
    run_id = submit_pipeline(
        loaded,
        namespace=namespace,
        host=host,
        existing_token=existing_token,
        cookies=cookies,
        user=user,
    )
    typer.echo(f"Pipeline submitted. Run ID: {run_id}")


# ---------------------------------------------------------------------------
# pipeline run commands
# ---------------------------------------------------------------------------

@run_app.command("get")
def cmd_run_get(
    run_id: str = typer.Argument(..., help="Pipeline run ID."),
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace.",
    ),
    host: str = typer.Option(
        "http://127.0.0.1:8888", help="KFP API host.",
    ),
    user: str = typer.Option(
        "user@example.com", help="Kubeflow user identity header.",
    ),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """Get details of a pipeline run by ID."""
    from kfp_workflow.cli.output import print_json, print_kv, style_run_state
    from kfp_workflow.pipeline.connection import kfp_connection

    with kfp_connection(
        namespace=namespace, host=host, user=user,
        existing_token=existing_token, cookies=cookies,
    ) as client:
        run = _resolve_run(client, run_id=run_id, namespace=namespace)
        resolved_run_id = run.run_id

    state = _run_state_str(run.state)
    workflow = _find_workflow_for_run(run_id=resolved_run_id, namespace=namespace)
    payload = _augment_run_payload(run, workflow)

    if _json_output:
        print_json(payload)
        return

    pairs = [
        ("Run ID", run.run_id or ""),
        ("Name", run.display_name or ""),
        ("State", style_run_state(state)),
        ("Created", str(run.created_at or "")),
        ("Finished", str(run.finished_at or "")),
        ("Experiment ID", run.experiment_id or ""),
    ]
    workflow_summary = _workflow_summary(workflow)
    if workflow_summary:
        pairs.extend([
            ("Workflow", workflow_summary["name"]),
            ("Workflow Phase", workflow_summary["phase"] or "(unknown)"),
            ("Workflow Progress", workflow_summary["progress"] or "(unknown)"),
        ])
        if workflow_summary["message"]:
            pairs.append(("Workflow Message", workflow_summary["message"]))
        if workflow_summary["pending_nodes"]:
            pairs.append(("Pending Nodes", ", ".join(workflow_summary["pending_nodes"])))
        if workflow_summary["failed_nodes"]:
            pairs.append(("Failed Nodes", ", ".join(workflow_summary["failed_nodes"])))
    if run.error:
        pairs.append(("Error", str(run.error)))
    print_kv(pairs)


@run_app.command("list")
def cmd_run_list(
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace.",
    ),
    experiment_id: Optional[str] = typer.Option(
        None, help="Filter by experiment ID.",
    ),
    page_size: int = typer.Option(20, help="Number of runs to return."),
    sort_by: str = typer.Option(
        "created_at desc", help="Sort order (e.g. 'created_at desc').",
    ),
    host: str = typer.Option(
        "http://127.0.0.1:8888", help="KFP API host.",
    ),
    user: str = typer.Option(
        "user@example.com", help="Kubeflow user identity header.",
    ),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """List pipeline runs."""
    from kfp_workflow.cli.output import print_json, print_table, style_run_state
    from kfp_workflow.pipeline.connection import kfp_connection

    with kfp_connection(
        namespace=namespace, host=host, user=user,
        existing_token=existing_token, cookies=cookies,
    ) as client:
        resolved_experiment_id = (
            _resolve_experiment_id(client, experiment_id=experiment_id, namespace=namespace)
            if experiment_id else None
        )
        response = client.list_runs(
            page_size=page_size,
            sort_by=sort_by,
            experiment_id=resolved_experiment_id,
            namespace=namespace,
        )

    runs = response.runs or []

    if _json_output:
        print_json([{
            "run_id": r.run_id,
            "display_name": r.display_name,
            "state": _run_state_str(r.state),
            "created_at": str(r.created_at),
            "finished_at": str(r.finished_at),
        } for r in runs])
        return

    rows = []
    for r in runs:
        state = _run_state_str(r.state)
        rows.append((
            (r.run_id or "")[:8],
            r.display_name or "",
            style_run_state(state),
            str(r.created_at or ""),
            str(r.finished_at or ""),
        ))
    print_table(
        title="Pipeline Runs",
        columns=["ID", "NAME", "STATE", "CREATED", "FINISHED"],
        rows=rows,
    )


@run_app.command("wait")
def cmd_run_wait(
    run_id: str = typer.Argument(..., help="Pipeline run ID."),
    timeout: int = typer.Option(3600, help="Timeout in seconds."),
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace.",
    ),
    host: str = typer.Option(
        "http://127.0.0.1:8888", help="KFP API host.",
    ),
    user: str = typer.Option(
        "user@example.com", help="Kubeflow user identity header.",
    ),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """Wait for a pipeline run to reach a terminal state."""
    from kfp_workflow.cli.output import console, style_run_state
    from kfp_workflow.pipeline.connection import kfp_connection

    try:
        with kfp_connection(
            namespace=namespace, host=host, user=user,
            existing_token=existing_token, cookies=cookies,
        ) as client:
            resolved_run = _resolve_run(client, run_id=run_id, namespace=namespace)
            resolved_run_id = resolved_run.run_id
            with console.status(f"Waiting for run {resolved_run_id[:8]}..."):
                run = client.wait_for_run_completion(resolved_run_id, timeout=timeout)
    except TimeoutError:
        workflow_summary = _workflow_summary(
            _find_workflow_for_run(run_id=resolved_run_id, namespace=namespace)
        )
        console.print(
            f"Run {resolved_run_id[:8]} did not reach a terminal KFP state within {timeout}s.",
            style="yellow",
        )
        if workflow_summary:
            console.print(
                "Workflow: "
                f"{workflow_summary['name']} "
                f"phase={workflow_summary['phase'] or 'Unknown'} "
                f"progress={workflow_summary['progress'] or 'Unknown'}"
            )
            if workflow_summary["message"]:
                console.print(f"Workflow message: {workflow_summary['message']}")
            if workflow_summary["pending_nodes"]:
                console.print(
                    "Pending nodes: " + ", ".join(workflow_summary["pending_nodes"])
                )
            if workflow_summary["failed_nodes"]:
                console.print(
                    "Failed nodes: " + ", ".join(workflow_summary["failed_nodes"]),
                    style="red",
                )
        raise typer.Exit(code=1)

    state = _run_state_str(run.state)
    console.print(f"Run {resolved_run_id[:8]} finished: {style_run_state(state)}")

    workflow_summary = _workflow_summary(
        _find_workflow_for_run(run_id=resolved_run_id, namespace=namespace)
    )
    if workflow_summary and workflow_summary["phase"]:
        console.print(
            "Workflow: "
            f"{workflow_summary['name']} "
            f"phase={workflow_summary['phase']} "
            f"progress={workflow_summary['progress'] or '(unknown)'}"
        )
        if workflow_summary["failed_nodes"]:
            console.print(
                "Failed nodes: " + ", ".join(workflow_summary["failed_nodes"]),
                style="red",
            )

    if state in ("FAILED", "CANCELED"):
        if run.error:
            console.print(f"Error: {run.error}", style="red")
        raise typer.Exit(code=1)


@run_app.command("terminate")
def cmd_run_terminate(
    run_id: str = typer.Argument(..., help="Pipeline run ID."),
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace.",
    ),
    host: str = typer.Option(
        "http://127.0.0.1:8888", help="KFP API host.",
    ),
    user: str = typer.Option(
        "user@example.com", help="Kubeflow user identity header.",
    ),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """Terminate (cancel) a running pipeline."""
    from kfp_workflow.pipeline.connection import kfp_connection

    with kfp_connection(
        namespace=namespace, host=host, user=user,
        existing_token=existing_token, cookies=cookies,
    ) as client:
        resolved_run_id = _resolve_run(client, run_id=run_id, namespace=namespace).run_id
        client.terminate_run(resolved_run_id)

    typer.echo(f"Run {resolved_run_id[:8]} terminated.")


@run_app.command("logs")
def cmd_run_logs(
    run_id: str = typer.Argument(..., help="Pipeline run ID."),
    step: Optional[str] = typer.Option(
        None, help="Filter pods by step name substring.",
    ),
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace.",
    ),
) -> None:
    """View logs from a pipeline run's component pods."""
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    from kfp_workflow.cli.output import console
    from kfp_workflow.pipeline.connection import kfp_connection

    with kfp_connection(
        namespace=namespace,
        user="user@example.com",
    ) as client:
        resolved_run_id = _resolve_run(client, run_id=run_id, namespace=namespace).run_id

    k8s_config.load_kube_config()
    v1 = k8s_client.CoreV1Api()
    pods = v1.list_namespaced_pod(
        namespace=namespace,
        label_selector=f"pipeline/runid={resolved_run_id}",
    )

    # Filter to implementation pods (contain actual component output)
    impl_pods = [
        p for p in pods.items
        if "system-container-impl" in p.metadata.name
    ]

    if step:
        impl_pods = [p for p in impl_pods if step in p.metadata.name]

    pods_to_show = impl_pods
    if not pods_to_show:
        driver_pods = [
            p for p in pods.items
            if "driver" in p.metadata.name
        ]
        if step:
            driver_pods = [p for p in driver_pods if step in p.metadata.name]
        pods_to_show = driver_pods

    if not pods_to_show:
        workflow_summary = _workflow_summary(
            _find_workflow_for_run(run_id=resolved_run_id, namespace=namespace)
        )
        if workflow_summary:
            typer.echo(
                "No matching pods found. "
                f"Workflow {workflow_summary['name']} "
                f"is {workflow_summary['phase'] or 'Unknown'} "
                f"with progress {workflow_summary['progress'] or 'Unknown'}.",
                err=True,
            )
            if workflow_summary["message"]:
                typer.echo(f"Workflow message: {workflow_summary['message']}", err=True)
            if workflow_summary["pending_nodes"]:
                typer.echo(
                    "Pending nodes: " + ", ".join(workflow_summary["pending_nodes"]),
                    err=True,
                )
            if workflow_summary["failed_nodes"]:
                typer.echo(
                    "Failed nodes: " + ", ".join(workflow_summary["failed_nodes"]),
                    err=True,
                )
        else:
            typer.echo("No matching pods found.", err=True)
        raise typer.Exit(code=1)

    for pod in sorted(pods_to_show, key=lambda p: p.metadata.name):
        console.rule(f"[bold]{pod.metadata.name}[/bold]")
        try:
            log = _log_for_pod(v1=v1, pod=pod, namespace=namespace)
            typer.echo(log)
        except k8s_client.ApiException as exc:
            typer.echo(f"  (error reading logs: {exc.reason})", err=True)


# ---------------------------------------------------------------------------
# pipeline experiment commands
# ---------------------------------------------------------------------------

@experiment_app.command("list")
def cmd_experiment_list(
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace.",
    ),
    page_size: int = typer.Option(20, help="Number of experiments to return."),
    host: str = typer.Option(
        "http://127.0.0.1:8888", help="KFP API host.",
    ),
    user: str = typer.Option(
        "user@example.com", help="Kubeflow user identity header.",
    ),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """List pipeline experiments."""
    from kfp_workflow.cli.output import print_json, print_table
    from kfp_workflow.pipeline.connection import kfp_connection

    with kfp_connection(
        namespace=namespace, host=host, user=user,
        existing_token=existing_token, cookies=cookies,
    ) as client:
        response = client.list_experiments(
            page_size=page_size, namespace=namespace,
        )

    experiments = response.experiments or []

    if _json_output:
        print_json([{
            "experiment_id": e.experiment_id,
            "display_name": e.display_name,
            "created_at": str(e.created_at),
            "last_run_created_at": str(e.last_run_created_at),
        } for e in experiments])
        return

    rows = []
    for e in experiments:
        rows.append((
            (e.experiment_id or "")[:8],
            e.display_name or "",
            str(e.created_at or ""),
            str(e.last_run_created_at or ""),
        ))
    print_table(
        title="Experiments",
        columns=["ID", "NAME", "CREATED", "LAST RUN"],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# serve commands
# ---------------------------------------------------------------------------

@serve_app.command("create")
def cmd_serve_create(
    spec: Path = typer.Option(..., help="Path to a serving YAML spec."),
    dry_run: bool = typer.Option(False, help="Print manifest without applying."),
    wait: bool = typer.Option(False, help="Wait for Ready=True after apply."),
    timeout: int = typer.Option(300, help="Wait timeout in seconds."),
) -> None:
    """Create a KServe InferenceService from a serving spec."""
    from kfp_workflow.serving import kserve
    from kfp_workflow.specs import load_serving_spec

    loaded = load_serving_spec(spec)
    result = kserve.create_inference_service(
        name=loaded.metadata.name,
        namespace=loaded.namespace,
        model_pvc_name=loaded.model_pvc,
        model_subpath=loaded.model_subpath,
        runtime=loaded.runtime,
        predictor_image=loaded.predictor_image,
        model_name=loaded.model_name,
        replicas=loaded.replicas,
        resources=loaded.resources.model_dump(),
        dry_run=dry_run,
    )
    typer.echo(json.dumps(result, indent=2))

    if wait and not dry_run:
        diagnostics = kserve.wait_for_inference_service_ready(
            name=loaded.metadata.name,
            namespace=loaded.namespace,
            timeout=timeout,
        )
        typer.echo(
            json.dumps(
                {
                    "name": loaded.metadata.name,
                    "namespace": loaded.namespace,
                    "ready": diagnostics["ready"],
                    "conditions": diagnostics["conditions"],
                    "events": diagnostics["events"],
                },
                indent=2,
            )
        )
        if diagnostics["ready"] != "True":
            raise typer.Exit(code=1)


@serve_app.command("delete")
def cmd_serve_delete(
    name: str = typer.Option(..., help="InferenceService name."),
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace."
    ),
) -> None:
    """Delete a KServe InferenceService."""
    from kfp_workflow.serving import kserve

    kserve.delete_inference_service(name=name, namespace=namespace)
    typer.echo(f"InferenceService '{name}' deleted.")


@serve_app.command("list")
def cmd_serve_list(
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace.",
    ),
) -> None:
    """List KServe InferenceServices in a namespace."""
    from kfp_workflow.cli.output import (
        print_json,
        print_table,
        style_isvc_ready,
    )
    from kfp_workflow.serving import kserve

    items = kserve.list_inference_services(namespace=namespace)

    if _json_output:
        print_json([{
            "name": svc["metadata"]["name"],
            "ready": _isvc_ready(svc),
            "url": svc.get("status", {}).get("url", ""),
        } for svc in items])
        return

    rows = []
    for svc in items:
        name = svc["metadata"]["name"]
        ready = _isvc_ready(svc)
        url = svc.get("status", {}).get("url", "")
        created = svc["metadata"].get("creationTimestamp", "")
        rows.append((name, style_isvc_ready(ready), url, created))

    print_table(
        title="InferenceServices",
        columns=["NAME", "READY", "URL", "CREATED"],
        rows=rows,
    )


@serve_app.command("get")
def cmd_serve_get(
    name: str = typer.Option(..., help="InferenceService name."),
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace.",
    ),
) -> None:
    """Get detailed status of a KServe InferenceService."""
    from kfp_workflow.cli.output import print_json, print_kv, style_isvc_ready
    from kfp_workflow.serving import kserve

    diagnostics = kserve.get_inference_service_diagnostics(
        name=name,
        namespace=namespace,
    )
    svc = diagnostics["service"]
    ready = diagnostics["ready"]
    url = svc.get("status", {}).get("url", "")
    created = svc["metadata"].get("creationTimestamp", "")
    conditions = diagnostics["conditions"]
    events = diagnostics["events"]

    if _json_output:
        print_json({
            "name": name,
            "namespace": namespace,
            "ready": ready,
            "url": url,
            "created": created,
            "conditions": conditions,
            "events": events,
        })
        return

    pairs = [
        ("Name", name),
        ("Namespace", namespace),
        ("Ready", style_isvc_ready(ready)),
        ("URL", url or "(none)"),
        ("Created", created),
    ]
    for cond in conditions:
        ctype = cond.get("type", "")
        cstatus = cond.get("status", "")
        detail = cstatus
        if cond.get("reason"):
            detail += f" ({cond['reason']})"
        if cond.get("message"):
            detail += f" - {cond['message']}"
        pairs.append((f"  {ctype}", detail))
    for idx, event in enumerate(events, start=1):
        detail = event.get("reason", "") or "(event)"
        if event.get("message"):
            detail += f" - {event['message']}"
        pairs.append((f"  Warning {idx}", detail))

    print_kv(pairs)


def _isvc_ready(svc: dict) -> str:
    """Extract the Ready condition status from an InferenceService dict."""
    conditions = svc.get("status", {}).get("conditions", [])
    for cond in conditions:
        if cond.get("type") == "Ready":
            return cond.get("status", "Unknown")
    return "Unknown"


# ---------------------------------------------------------------------------
# registry model commands
# ---------------------------------------------------------------------------

@model_reg_app.command("register")
def cmd_model_register(
    name: str = typer.Option(..., help="Model name."),
    version: str = typer.Option(..., help="Model version."),
    uri: str = typer.Option(..., help="Model artifact URI (PVC subpath)."),
    framework: str = typer.Option("pytorch", help="ML framework."),
    description: str = typer.Option("", help="Model description."),
    registry_path: str = typer.Option(
        "/mnt/models/.model_registry.json", help="Path to registry JSON file."
    ),
) -> None:
    """Register a model in the file-backed model registry."""
    from kfp_workflow.registry.model_registry import FileModelRegistry

    registry = FileModelRegistry(registry_path=registry_path)
    info = registry.register_model(
        name=name,
        version=version,
        uri=uri,
        framework=framework,
        description=description,
    )
    typer.echo(json.dumps(info.model_dump(), indent=2))


@model_reg_app.command("get")
def cmd_model_get(
    name: str = typer.Option(..., help="Model name."),
    version: Optional[str] = typer.Option(None, help="Model version."),
    registry_path: str = typer.Option(
        "/mnt/models/.model_registry.json", help="Path to registry JSON file."
    ),
) -> None:
    """Retrieve a model from the file-backed model registry."""
    from kfp_workflow.registry.model_registry import FileModelRegistry

    registry = FileModelRegistry(registry_path=registry_path)
    info = registry.get_model(name=name, version=version)
    typer.echo(json.dumps(info.model_dump(), indent=2))


@model_reg_app.command("list")
def cmd_model_list(
    registry_path: str = typer.Option(
        "/mnt/models/.model_registry.json", help="Path to registry JSON file."
    ),
) -> None:
    """List all models in the file-backed model registry."""
    from kfp_workflow.cli.output import print_json, print_table
    from kfp_workflow.registry.model_registry import FileModelRegistry

    registry = FileModelRegistry(registry_path=registry_path)
    models = registry.list_models()

    if _json_output:
        print_json([m.model_dump() for m in models])
        return

    rows = [(m.name, m.version, m.framework, m.uri) for m in models]
    print_table(
        title="Models",
        columns=["NAME", "VERSION", "FRAMEWORK", "URI"],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# registry dataset commands
# ---------------------------------------------------------------------------

@dataset_reg_app.command("register")
def cmd_dataset_register(
    name: str = typer.Option(..., help="Dataset name."),
    pvc_name: str = typer.Option(..., help="PVC name."),
    subpath: str = typer.Option(..., help="Path within the PVC."),
    version: str = typer.Option("v1", help="Dataset version."),
    description: str = typer.Option("", help="Dataset description."),
    registry_path: str = typer.Option(
        "/mnt/data/.dataset_registry.json", help="Path to registry JSON file."
    ),
) -> None:
    """Register a dataset in the PVC dataset registry."""
    from kfp_workflow.registry.dataset_registry import PVCDatasetRegistry

    registry = PVCDatasetRegistry(registry_path=registry_path)
    info = registry.register_dataset(
        name=name,
        pvc_name=pvc_name,
        subpath=subpath,
        version=version,
        description=description,
    )
    typer.echo(json.dumps(info.model_dump(), indent=2))


@dataset_reg_app.command("get")
def cmd_dataset_get(
    name: str = typer.Option(..., help="Dataset name."),
    version: Optional[str] = typer.Option(None, help="Dataset version."),
    registry_path: str = typer.Option(
        "/mnt/data/.dataset_registry.json", help="Path to registry JSON file."
    ),
) -> None:
    """Retrieve a dataset from the PVC dataset registry."""
    from kfp_workflow.registry.dataset_registry import PVCDatasetRegistry

    registry = PVCDatasetRegistry(registry_path=registry_path)
    info = registry.get_dataset(name=name, version=version)
    typer.echo(json.dumps(info.model_dump(), indent=2))


@dataset_reg_app.command("list")
def cmd_dataset_list(
    registry_path: str = typer.Option(
        "/mnt/data/.dataset_registry.json", help="Path to registry JSON file."
    ),
) -> None:
    """List all datasets in the PVC dataset registry."""
    from kfp_workflow.cli.output import print_json, print_table
    from kfp_workflow.registry.dataset_registry import PVCDatasetRegistry

    registry = PVCDatasetRegistry(registry_path=registry_path)
    datasets = registry.list_datasets()

    if _json_output:
        print_json([d.model_dump() for d in datasets])
        return

    rows = [
        (d.name, d.version, d.pvc_name, d.subpath) for d in datasets
    ]
    print_table(
        title="Datasets",
        columns=["NAME", "VERSION", "PVC", "SUBPATH"],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# cluster commands
# ---------------------------------------------------------------------------

def _run_kubectl(args: list[str], *, input_text: str | None = None) -> None:
    """Run a kubectl command, raising on failure."""
    subprocess.run(args, check=True, text=True, input=input_text)


@cluster_app.command("bootstrap")
def cmd_cluster_bootstrap(
    spec: Path = typer.Option(..., help="Path to a pipeline YAML spec."),
    spec_type: str = typer.Option(
        "pipeline",
        "--type",
        help="Spec type: 'pipeline' or 'benchmark'.",
    ),
    dry_run: bool = typer.Option(False, help="Print manifests without applying."),
) -> None:
    """Create PVCs for data and model storage."""
    from kfp_workflow.benchmark.materialize import load_materialized_benchmark_spec
    from kfp_workflow.specs import load_pipeline_spec

    if spec_type == "pipeline":
        loaded = load_pipeline_spec(spec)
    elif spec_type == "benchmark":
        loaded, _ = load_materialized_benchmark_spec(spec)
    else:
        typer.echo(f"Unknown spec type: {spec_type}", err=True)
        raise typer.Exit(code=1)

    storage = loaded.storage
    namespace = loaded.runtime.namespace

    data_pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": storage.data_pvc,
            "namespace": namespace,
        },
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "storageClassName": storage.storage_class,
            "resources": {"requests": {"storage": storage.data_size}},
        },
    }
    model_pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": storage.model_pvc,
            "namespace": namespace,
        },
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "storageClassName": storage.storage_class,
            "resources": {"requests": {"storage": storage.model_size}},
        },
    }

    manifests = [data_pvc, model_pvc]
    results_pvc = getattr(storage, "results_pvc", None)
    if results_pvc:
        manifests.append(
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {
                    "name": results_pvc,
                    "namespace": namespace,
                },
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "storageClassName": storage.storage_class,
                    "resources": {
                        "requests": {
                            "storage": getattr(storage, "results_size", storage.model_size),
                        }
                    },
                },
            }
        )
    typer.echo(json.dumps(manifests, indent=2))

    if dry_run:
        typer.echo("Dry run — manifests printed above, nothing applied.")
        return

    # Ensure namespace exists (idempotent)
    subprocess.run(
        ["kubectl", "create", "namespace", namespace],
        check=False, text=True, capture_output=True,
    )

    for manifest in manifests:
        _run_kubectl(
            ["kubectl", "apply", "-n", namespace, "-f", "-"],
            input_text=json.dumps(manifest),
        )
        typer.echo(f"Applied {manifest['kind']} '{manifest['metadata']['name']}'")

    typer.echo("Cluster bootstrap complete.")


# ---------------------------------------------------------------------------
# tune commands
# ---------------------------------------------------------------------------

@tune_app.command("run")
def cmd_tune_run(
    spec: Path = typer.Option(..., help="Path to a tune YAML spec."),
    set_values: List[str] = typer.Option(
        [], "--set",
        help="Override spec values (e.g., --set hpo.algorithm=tpe).",
    ),
    data_mount_path: Optional[str] = typer.Option(
        None, help="Override data mount path (default: from spec).",
    ),
    output: Optional[Path] = typer.Option(
        None, help="Write best params JSON to this file.",
    ),
) -> None:
    """Run local hyperparameter optimisation using Optuna."""
    from kfp_workflow.plugins import get_plugin
    from kfp_workflow.specs import load_tune_spec_with_overrides
    from kfp_workflow.tune.engine import run_hpo

    loaded = load_tune_spec_with_overrides(spec, set_values or None)
    _validate_plugin_config_or_exit(loaded.model_dump())
    plugin = get_plugin(loaded.model.name)
    mount = data_mount_path or loaded.storage.data_mount_path

    result = run_hpo(plugin, loaded, mount)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.model_dump(), indent=2, default=str))
        typer.echo(f"Results written to {output}")

    if _json_output:
        from kfp_workflow.cli.output import print_json
        print_json(result.model_dump())
    else:
        from kfp_workflow.cli.output import print_kv
        print_kv([
            ("Best value", f"{result.best_value:.4f}"),
            ("Trials", f"{result.n_completed}/{result.n_trials} completed"),
            ("Pruned", str(result.n_pruned)),
            ("Failed", str(result.n_failed)),
            ("Wall time", f"{result.wall_time_seconds:.1f}s"),
        ])
        typer.echo("\nBest parameters:")
        typer.echo(json.dumps(result.best_params, indent=2, default=str))


@tune_app.command("katib")
def cmd_tune_katib(
    spec: Path = typer.Option(..., help="Path to a tune YAML spec."),
    set_values: List[str] = typer.Option(
        [], "--set",
        help="Override spec values (e.g., --set hpo.max_trials=30).",
    ),
    output: Optional[Path] = typer.Option(
        None, help="Write Katib manifest YAML to this file.",
    ),
    dry_run: bool = typer.Option(
        False, help="Print manifest without applying to the cluster.",
    ),
) -> None:
    """Generate or apply a Katib Experiment manifest for distributed HPO."""
    import yaml as pyyaml

    from kfp_workflow.plugins import get_plugin
    from kfp_workflow.specs import load_tune_spec_with_overrides
    from kfp_workflow.tune.engine import resolve_search_space
    from kfp_workflow.tune.katib import build_katib_experiment

    loaded = load_tune_spec_with_overrides(spec, set_values or None)
    _validate_plugin_config_or_exit(loaded.model_dump())
    plugin = get_plugin(loaded.model.name)
    search_space = resolve_search_space(plugin, loaded.model_dump())

    trial_command = [
        "python", "-m", "kfp_workflow.cli.main",
        "tune", "trial",
        "--spec-json", loaded.model_dump_json(),
        "--data-mount-path", loaded.storage.data_mount_path,
    ]
    manifest = build_katib_experiment(
        loaded,
        search_space,
        trial_image=loaded.runtime.image,
        trial_command=trial_command,
    )

    manifest_yaml = pyyaml.dump(manifest, default_flow_style=False, sort_keys=False)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(manifest_yaml)
        typer.echo(f"Katib manifest written to {output}")

    if dry_run or _json_output:
        if _json_output:
            from kfp_workflow.cli.output import print_json
            print_json(manifest)
        else:
            typer.echo(manifest_yaml)
        if dry_run:
            return

    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        check=True, text=True, input=json.dumps(manifest),
    )
    typer.echo(
        f"Katib experiment '{loaded.metadata.name}' submitted "
        f"to namespace '{loaded.runtime.namespace}'."
    )


@tune_app.command("trial", hidden=True)
def cmd_tune_trial(
    spec_json: str = typer.Option(..., help="Serialized tune spec JSON."),
    trial_params_json: str = typer.Option(
        ..., help="Serialized Katib trial parameter assignments.",
    ),
    data_mount_path: str = typer.Option(
        "/mnt/data", help="Path where the data PVC is mounted.",
    ),
) -> None:
    """Run one Katib trial by delegating to the selected model plugin."""
    from kfp_workflow.plugins import get_plugin
    from kfp_workflow.specs import TuneSpec

    loaded = TuneSpec.model_validate_json(spec_json)
    _validate_plugin_config_or_exit(loaded.model_dump())
    plugin = get_plugin(loaded.model.name)
    trial_params = _coerce_json_scalar_values(json.loads(trial_params_json))
    base_params = plugin.hpo_base_config(loaded.model_dump())
    objective_value = plugin.hpo_objective(
        loaded.model_dump(),
        {**base_params, **trial_params},
        data_mount_path,
    )
    typer.echo(f"objective={float(objective_value)}")


@tune_app.command("show-space")
def cmd_tune_show_space(
    spec: Path = typer.Option(..., help="Path to a tune YAML spec."),
    set_values: List[str] = typer.Option(
        [], "--set",
        help="Override spec values (e.g., --set hpo.builtin_profile=aggressive).",
    ),
) -> None:
    """Display the resolved search space for a tune spec."""
    from kfp_workflow.cli.output import print_json, print_table
    from kfp_workflow.plugins import get_plugin
    from kfp_workflow.specs import load_tune_spec_with_overrides
    from kfp_workflow.tune.engine import resolve_search_space

    loaded = load_tune_spec_with_overrides(spec, set_values or None)
    _validate_plugin_config_or_exit(loaded.model_dump())
    plugin = get_plugin(loaded.model.name)
    space = resolve_search_space(plugin, loaded.model_dump())

    if _json_output:
        print_json([p.model_dump() for p in space])
        return

    rows = []
    for p in space:
        if p.type == "categorical":
            detail = ", ".join(str(v) for v in (p.values or []))
        else:
            detail = f"[{p.low}, {p.high}]"
            if p.step:
                detail += f" step={p.step}"
            if p.type == "log_float":
                detail += " (log)"
        rows.append((p.name, p.type, detail))
    print_table("Search Space", ["NAME", "TYPE", "RANGE / VALUES"], rows)


# ---------------------------------------------------------------------------
# Entry point for `python -m kfp_workflow.cli.main`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
