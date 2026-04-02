"""Typer CLI application for kfp-workflow."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

# Suppress noisy third-party deprecation warnings before importing modules that
# transitively load google-auth / KFP.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google(\..*)?$")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"kfp(\..*)?$")

from kfp_workflow.cli.workflows import (
    DEFAULT_HOST,
    DEFAULT_NAMESPACE,
    DEFAULT_USER,
    build_run_payload,
    find_workflow_for_run,
    resolve_experiment_id,
    resolve_run,
    run_state_str,
    short_id,
    workflow_summary,
)

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
    help="Integrated Kubeflow workflow manager for pipelines, serving, tuning, benchmarks, registry, and storage bootstrap.",
    add_completion=False,
)

pipeline_app = typer.Typer(help="Compile and submit training pipelines.")
benchmark_app = typer.Typer(help="Compile and submit benchmark workflows.")
serve_app = typer.Typer(help="Create and manage KServe InferenceServices.")
registry_app = typer.Typer(help="Register and retrieve models and datasets.")
model_reg_app = typer.Typer(help="Model registry operations.")
dataset_reg_app = typer.Typer(help="Dataset registry operations.")
cluster_app = typer.Typer(help="Cluster bootstrapping operations.")
spec_app = typer.Typer(help="Spec validation.")
tune_app = typer.Typer(help="Hyperparameter tuning operations.")

app.add_typer(pipeline_app, name="pipeline")
app.add_typer(benchmark_app, name="benchmark")
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
    """Integrated Kubeflow workflow manager for pipelines, serving, tuning, benchmarks, registry, and storage bootstrap."""
    global _json_output
    _json_output = json_mode


# ---------------------------------------------------------------------------
# spec commands
# ---------------------------------------------------------------------------

@spec_app.command("validate")
def cmd_spec_validate(
    spec: Path = typer.Option(
        ...,
        help="Path to a pipeline/serving/tune YAML spec or benchmark spec file.",
    ),
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
    spec: Path = typer.Option(
        ...,
        help="Path to a benchmark spec file (YAML or Python).",
    ),
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
    spec: Path = typer.Option(
        ...,
        help="Path to a benchmark spec file (YAML or Python).",
    ),
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
    typer.echo(f"Submitted benchmark run: {run_id}")


@benchmark_app.command("list")
def cmd_benchmark_list(
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
    page_size: int = typer.Option(20, help="Number of runs to return."),
    sort_by: str = typer.Option(
        "created_at desc", help="Sort order (e.g. 'created_at desc').",
    ),
    host: str = typer.Option(
        DEFAULT_HOST, help="KFP API host.",
    ),
    user: str = typer.Option(
        DEFAULT_USER, help="Kubeflow user identity header.",
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
        workflow_meta = workflow_summary(workflow) or {}
        items.append({
            "id": run.run_id,
            "name": benchmark_spec.get("metadata", {}).get("name", run.display_name),
            "state": run_state_str(run.state),
            "created_at": str(run.created_at),
            "finished_at": str(run.finished_at),
            "namespace": namespace,
            "workflow": {
                "name": workflow_meta.get("name", ""),
                "phase": workflow_meta.get("phase", ""),
            },
        })

    if _json_output:
        print_json(items)
        return

    rows = []
    for item in items:
        rows.append((
            short_id(item["id"] or ""),
            item["name"] or "",
            style_run_state(item["state"]),
            item["workflow"]["name"] or "",
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
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
    host: str = typer.Option(
        DEFAULT_HOST, help="KFP API host.",
    ),
    user: str = typer.Option(
        DEFAULT_USER, help="Kubeflow user identity header.",
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
        run = resolve_run(client, run_id=run_id, namespace=namespace)
        resolved_run_id = run.run_id

    workflow = find_workflow_for_run(resolved_run_id, namespace)
    if not workflow or not is_benchmark_workflow(workflow):
        typer.echo(f"Run '{resolved_run_id}' is not a benchmark workflow.", err=True)
        raise typer.Exit(code=1)

    benchmark_spec = extract_benchmark_spec(workflow) or {}
    payload = build_run_payload(
        run,
        workflow,
        namespace=namespace,
        name=benchmark_spec.get("metadata", {}).get("name", run.display_name),
    )

    try:
        results = resolve_results(
            workflow=workflow,
            benchmark_spec=benchmark_spec,
            namespace=namespace,
        )
        payload["results"] = {
            "path": results["results_path"],
            "summary": results["summary"],
        }
    except Exception as exc:
        payload["results"] = {"error": str(exc)}

    if _json_output:
        print_json(payload)
        return

    pairs = [
        ("Benchmark", payload["name"] or ""),
        ("Run ID", payload["id"] or ""),
        ("State", style_run_state(payload["state"])),
        ("Created", str(run.created_at or "")),
        ("Finished", str(run.finished_at or "")),
        ("Experiment ID", run.experiment_id or ""),
    ]
    workflow_meta = payload.get("workflow", {})
    if workflow_meta.get("name"):
        pairs.extend([
            ("Workflow", workflow_meta["name"]),
            ("Workflow Phase", workflow_meta.get("phase") or "(unknown)"),
            ("Workflow Progress", workflow_meta.get("progress") or "(unknown)"),
        ])
    if payload.get("results", {}).get("path"):
        pairs.append(("Results Path", payload["results"]["path"]))
    if payload.get("results", {}).get("summary"):
        summary = payload["results"]["summary"]
        pairs.append(("Result Status", str(summary.get("status", ""))))
        pairs.append(("Request Count", str(summary.get("request_count", ""))))
        if "delta_joules" in summary:
            pairs.append(("Delta Joules", str(summary["delta_joules"])))
    if payload.get("results", {}).get("error"):
        pairs.append(("Results Error", payload["results"]["error"]))
    print_kv(pairs)


@benchmark_app.command("download")
def cmd_benchmark_download(
    run_id: str = typer.Argument(..., help="Benchmark run ID."),
    output: Optional[Path] = typer.Option(None, help="Write results JSON to this path."),
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
    host: str = typer.Option(
        DEFAULT_HOST, help="KFP API host.",
    ),
    user: str = typer.Option(
        DEFAULT_USER, help="Kubeflow user identity header.",
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
        run = resolve_run(client, run_id=run_id, namespace=namespace)
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
        "id": resolved_run_id,
        "name": benchmark_name,
        "results_path": results["results_path"],
        "output_path": str(destination),
        "namespace": namespace,
    }
    if _json_output:
        print_json(payload)
        return

    typer.echo(f"Downloaded benchmark results to {destination}")


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
    typer.echo(f"Compiled pipeline package: {result}")


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
    typer.echo(f"Submitted pipeline run: {run_id}")


# ---------------------------------------------------------------------------
# pipeline commands
# ---------------------------------------------------------------------------


@pipeline_app.command("get")
def cmd_pipeline_get(
    run_id: str = typer.Argument(..., help="Pipeline run ID."),
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
    host: str = typer.Option(
        DEFAULT_HOST, help="KFP API host.",
    ),
    user: str = typer.Option(
        DEFAULT_USER, help="Kubeflow user identity header.",
    ),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """Show detailed info for a pipeline run."""
    from kfp_workflow.cli.output import print_json, print_kv, style_run_state
    from kfp_workflow.pipeline.connection import kfp_connection

    with kfp_connection(
        namespace=namespace, host=host, user=user,
        existing_token=existing_token, cookies=cookies,
    ) as client:
        run = resolve_run(client, run_id=run_id, namespace=namespace)
        resolved_run_id = run.run_id

    workflow = find_workflow_for_run(run_id=resolved_run_id, namespace=namespace)
    payload = build_run_payload(run, workflow, namespace=namespace)

    if _json_output:
        print_json(payload)
        return

    pairs = [
        ("Run ID", payload["id"] or ""),
        ("Name", payload["name"] or ""),
        ("State", style_run_state(payload["state"])),
        ("Created", payload["created_at"] or ""),
        ("Finished", payload["finished_at"] or ""),
        ("Experiment ID", payload["workflow"].get("experiment_id", "")),
    ]
    workflow_meta = payload.get("workflow", {})
    if workflow_meta.get("name"):
        pairs.extend([
            ("Workflow", workflow_meta["name"]),
            ("Workflow Phase", workflow_meta.get("phase") or "(unknown)"),
            ("Workflow Progress", workflow_meta.get("progress") or "(unknown)"),
        ])
        if workflow_meta.get("message"):
            pairs.append(("Workflow Message", workflow_meta["message"]))
        if workflow_meta.get("pending_nodes"):
            pairs.append(("Pending Nodes", ", ".join(workflow_meta["pending_nodes"])))
        if workflow_meta.get("failed_nodes"):
            pairs.append(("Failed Nodes", ", ".join(workflow_meta["failed_nodes"])))
    if payload["workflow"].get("error"):
        pairs.append(("Error", payload["workflow"]["error"]))
    print_kv(pairs)


@pipeline_app.command("list")
def cmd_pipeline_list(
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
    experiment_id: Optional[str] = typer.Option(
        None, help="Filter by experiment ID.",
    ),
    page_size: int = typer.Option(20, help="Number of runs to return."),
    sort_by: str = typer.Option(
        "created_at desc", help="Sort order (e.g. 'created_at desc').",
    ),
    host: str = typer.Option(
        DEFAULT_HOST, help="KFP API host.",
    ),
    user: str = typer.Option(
        DEFAULT_USER, help="Kubeflow user identity header.",
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
            resolve_experiment_id(client, experiment_id=experiment_id, namespace=namespace)
            if experiment_id else None
        )
        response = client.list_runs(
            page_size=page_size,
            sort_by=sort_by,
            experiment_id=resolved_experiment_id,
            namespace=namespace,
        )

    items = [{
        "id": r.run_id,
        "name": r.display_name,
        "state": run_state_str(r.state),
        "created_at": str(r.created_at),
        "finished_at": str(r.finished_at),
        "namespace": namespace,
    } for r in (response.runs or [])]

    if _json_output:
        print_json(items)
        return

    rows = [
        (
            short_id(item["id"] or ""),
            item["name"] or "",
            style_run_state(item["state"]),
            item["created_at"] or "",
            item["finished_at"] or "",
        )
        for item in items
    ]
    print_table(
        title="Pipeline Runs",
        columns=["ID", "NAME", "STATE", "CREATED", "FINISHED"],
        rows=rows,
    )


@pipeline_app.command("wait")
def cmd_pipeline_wait(
    run_id: str = typer.Argument(..., help="Pipeline run ID."),
    timeout: int = typer.Option(3600, help="Timeout in seconds."),
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
    host: str = typer.Option(
        DEFAULT_HOST, help="KFP API host.",
    ),
    user: str = typer.Option(
        DEFAULT_USER, help="Kubeflow user identity header.",
    ),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """Wait for a pipeline run to reach a terminal state."""
    from kfp_workflow.cli.output import console, print_json, style_run_state
    from kfp_workflow.pipeline.connection import kfp_connection

    resolved_run_id = run_id
    try:
        with kfp_connection(
            namespace=namespace, host=host, user=user,
            existing_token=existing_token, cookies=cookies,
        ) as client:
            resolved_run = resolve_run(client, run_id=run_id, namespace=namespace)
            resolved_run_id = resolved_run.run_id
            with console.status(f"Waiting for run {short_id(resolved_run_id)}..."):
                run = client.wait_for_run_completion(resolved_run_id, timeout=timeout)
    except TimeoutError:
        workflow_meta = workflow_summary(
            find_workflow_for_run(run_id=resolved_run_id, namespace=namespace)
        )
        if _json_output:
            print_json({
                "id": resolved_run_id,
                "namespace": namespace,
                "state": "TIMEOUT",
                "workflow": workflow_meta or {},
                "timeout_seconds": timeout,
            })
        else:
            console.print(
                f"Run {short_id(resolved_run_id)} did not reach a terminal KFP state within {timeout}s.",
                style="yellow",
            )
            if workflow_meta:
                console.print(
                    "Workflow: "
                    f"{workflow_meta['name']} "
                    f"phase={workflow_meta['phase'] or 'Unknown'} "
                    f"progress={workflow_meta['progress'] or 'Unknown'}"
                )
                if workflow_meta["message"]:
                    console.print(f"Workflow message: {workflow_meta['message']}")
                if workflow_meta["pending_nodes"]:
                    console.print(
                        "Pending nodes: " + ", ".join(workflow_meta["pending_nodes"])
                    )
                if workflow_meta["failed_nodes"]:
                    console.print(
                        "Failed nodes: " + ", ".join(workflow_meta["failed_nodes"]),
                        style="red",
                    )
        raise typer.Exit(code=1)

    state = run_state_str(run.state)
    workflow = find_workflow_for_run(run_id=resolved_run_id, namespace=namespace)
    payload = build_run_payload(run, workflow, namespace=namespace)

    if _json_output:
        print_json(payload)
    else:
        console.print(f"Run {short_id(resolved_run_id)} finished: {style_run_state(state)}")
        workflow_meta = payload.get("workflow", {})
        if workflow_meta.get("phase"):
            console.print(
                "Workflow: "
                f"{workflow_meta['name']} "
                f"phase={workflow_meta['phase']} "
                f"progress={workflow_meta.get('progress') or '(unknown)'}"
            )
            if workflow_meta.get("failed_nodes"):
                console.print(
                    "Failed nodes: " + ", ".join(workflow_meta["failed_nodes"]),
                    style="red",
                )

        if state in ("FAILED", "CANCELED") and payload["workflow"].get("error"):
            console.print(f"Error: {payload['workflow']['error']}", style="red")

    if state in ("FAILED", "CANCELED"):
        raise typer.Exit(code=1)


@pipeline_app.command("terminate")
def cmd_pipeline_terminate(
    run_id: str = typer.Argument(..., help="Pipeline run ID."),
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
    host: str = typer.Option(
        DEFAULT_HOST, help="KFP API host.",
    ),
    user: str = typer.Option(
        DEFAULT_USER, help="Kubeflow user identity header.",
    ),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """Terminate (cancel) a running pipeline."""
    from kfp_workflow.cli.output import print_json
    from kfp_workflow.pipeline.connection import kfp_connection

    with kfp_connection(
        namespace=namespace, host=host, user=user,
        existing_token=existing_token, cookies=cookies,
    ) as client:
        resolved_run_id = resolve_run(client, run_id=run_id, namespace=namespace).run_id
        client.terminate_run(resolved_run_id)

    payload = {
        "id": resolved_run_id,
        "namespace": namespace,
        "state": "CANCELING",
    }
    if _json_output:
        print_json(payload)
        return

    typer.echo(f"Terminated pipeline run: {resolved_run_id}")


@pipeline_app.command("logs")
def cmd_pipeline_logs(
    run_id: str = typer.Argument(..., help="Pipeline run ID."),
    step: Optional[str] = typer.Option(
        None, help="Filter pods by step name substring.",
    ),
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
    host: str = typer.Option(
        DEFAULT_HOST, help="KFP API host.",
    ),
    user: str = typer.Option(
        DEFAULT_USER, help="Kubeflow user identity header.",
    ),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """View logs from a pipeline run's component pods."""
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    from kfp_workflow.cli.output import console, print_json
    from kfp_workflow.pipeline.connection import kfp_connection

    with kfp_connection(
        namespace=namespace,
        host=host,
        user=user,
        existing_token=existing_token,
        cookies=cookies,
    ) as client:
        resolved_run_id = resolve_run(client, run_id=run_id, namespace=namespace).run_id

    k8s_config.load_kube_config()
    v1 = k8s_client.CoreV1Api()
    pods = v1.list_namespaced_pod(
        namespace=namespace,
        label_selector=f"pipeline/runid={resolved_run_id}",
    )

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
        workflow_meta = workflow_summary(
            find_workflow_for_run(run_id=resolved_run_id, namespace=namespace)
        )
        if workflow_meta:
            typer.echo(
                "No matching pods found. "
                f"Workflow {workflow_meta['name']} "
                f"is {workflow_meta['phase'] or 'Unknown'} "
                f"with progress {workflow_meta['progress'] or 'Unknown'}.",
                err=True,
            )
            if workflow_meta["message"]:
                typer.echo(f"Workflow message: {workflow_meta['message']}", err=True)
            if workflow_meta["pending_nodes"]:
                typer.echo(
                    "Pending nodes: " + ", ".join(workflow_meta["pending_nodes"]),
                    err=True,
                )
            if workflow_meta["failed_nodes"]:
                typer.echo(
                    "Failed nodes: " + ", ".join(workflow_meta["failed_nodes"]),
                    err=True,
                )
        else:
            typer.echo("No matching pods found.", err=True)
        raise typer.Exit(code=1)

    entries = []
    for pod in sorted(pods_to_show, key=lambda p: p.metadata.name):
        try:
            log = _log_for_pod(v1=v1, pod=pod, namespace=namespace)
        except k8s_client.ApiException as exc:
            log = f"(error reading logs: {exc.reason})"
        entries.append({
            "id": resolved_run_id,
            "name": pod.metadata.name,
            "namespace": namespace,
            "state": "LOGS",
            "logs": log,
        })

    if _json_output:
        print_json(entries)
        return

    for entry in entries:
        console.rule(f"[bold]{entry['name']}[/bold]")
        typer.echo(entry["logs"])


@pipeline_app.command("list-experiments")
def cmd_pipeline_list_experiments(
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
    page_size: int = typer.Option(20, help="Number of experiments to return."),
    host: str = typer.Option(
        DEFAULT_HOST, help="KFP API host.",
    ),
    user: str = typer.Option(
        DEFAULT_USER, help="Kubeflow user identity header.",
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

    items = [{
        "id": e.experiment_id,
        "name": e.display_name,
        "created_at": str(e.created_at),
        "last_run_created_at": str(e.last_run_created_at),
        "namespace": namespace,
    } for e in (response.experiments or [])]

    if _json_output:
        print_json(items)
        return

    rows = [
        (
            short_id(item["id"] or ""),
            item["name"] or "",
            item["created_at"] or "",
            item["last_run_created_at"] or "",
        )
        for item in items
    ]
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
    payload = {
        "id": loaded.metadata.name,
        "name": loaded.metadata.name,
        "namespace": loaded.namespace,
        "state": "READY" if dry_run else "CREATED",
        "service": result,
    }

    if wait and not dry_run:
        diagnostics = kserve.wait_for_inference_service_ready(
            name=loaded.metadata.name,
            namespace=loaded.namespace,
            timeout=timeout,
        )
        payload.update({
            "state": diagnostics["ready"],
            "conditions": diagnostics["conditions"],
            "events": diagnostics["events"],
        })
        if diagnostics["ready"] != "True":
            typer.echo(json.dumps(payload, indent=2))
            raise typer.Exit(code=1)
    if _json_output:
        from kfp_workflow.cli.output import print_json
        print_json(payload)
        return
    typer.echo(json.dumps(payload, indent=2))


@serve_app.command("delete")
def cmd_serve_delete(
    name: str = typer.Argument(..., help="InferenceService name."),
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace."
    ),
) -> None:
    """Delete a KServe InferenceService."""
    from kfp_workflow.serving import kserve

    kserve.delete_inference_service(name=name, namespace=namespace)
    payload = {"id": name, "name": name, "namespace": namespace, "state": "DELETED"}
    if _json_output:
        from kfp_workflow.cli.output import print_json
        print_json(payload)
        return
    typer.echo(f"Deleted inference service: {name}")


@serve_app.command("list")
def cmd_serve_list(
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
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
            "id": svc["metadata"]["name"],
            "name": svc["metadata"]["name"],
            "state": _isvc_ready(svc),
            "url": svc.get("status", {}).get("url", ""),
            "created_at": svc["metadata"].get("creationTimestamp", ""),
            "namespace": namespace,
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
    name: str = typer.Argument(..., help="InferenceService name."),
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
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
            "id": name,
            "name": name,
            "namespace": namespace,
            "state": ready,
            "url": url,
            "created_at": created,
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


def _kubectl_completed(args: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run kubectl and return the completed process for custom error handling."""
    return subprocess.run(
        args,
        check=False,
        text=True,
        input=input_text,
        capture_output=True,
    )


@cluster_app.command("bootstrap")
def cmd_cluster_bootstrap(
    spec: Path = typer.Option(
        ...,
        help="Path to a pipeline/tune YAML spec or benchmark spec file.",
    ),
    spec_type: str = typer.Option(
        "pipeline",
        "--type",
        help="Spec type: 'pipeline', 'benchmark', or 'tune'.",
    ),
    dry_run: bool = typer.Option(False, help="Print manifests without applying."),
) -> None:
    """Create PVC manifests derived from a pipeline, benchmark, or tune spec."""
    from kfp_workflow.benchmark.materialize import load_materialized_benchmark_spec
    from kfp_workflow.specs import load_pipeline_spec, load_tune_spec

    if spec_type == "pipeline":
        loaded = load_pipeline_spec(spec)
    elif spec_type == "benchmark":
        loaded, _ = load_materialized_benchmark_spec(spec)
    elif spec_type == "tune":
        loaded = load_tune_spec(spec)
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


def _generate_experiment_id() -> str:
    """Generate an opaque 16-character Katib experiment ID.

    The returned ID stays lowercase hex-like for UX consistency with other
    backend-generated identifiers while always satisfying the Katib name regex
    by forcing the first character to be alphabetic.
    """
    raw = secrets.token_hex(8)
    if raw[0].isdigit():
        raw = f"a{raw[1:]}"
    return raw


def _build_tune_manifest(
    spec_path: Path,
    set_values: List[str],
) -> tuple[Any, str, dict, str]:
    """Load, validate, and materialize a Katib experiment manifest."""
    import yaml as pyyaml

    from kfp_workflow.plugins import get_plugin
    from kfp_workflow.specs import load_tune_spec_with_overrides
    from kfp_workflow.tune.engine import resolve_search_space
    from kfp_workflow.tune.katib import (
        _trial_parameters_json,
        build_katib_experiment,
    )

    loaded = load_tune_spec_with_overrides(spec_path, set_values or None)
    _validate_plugin_config_or_exit(loaded.model_dump())
    plugin = get_plugin(loaded.model.name)
    search_space = resolve_search_space(plugin, loaded.model_dump())
    experiment_id = _generate_experiment_id()
    trial_command = [
        "python", "-m", "kfp_workflow.cli.main",
        "tune", "trial",
        "--data-mount-path", loaded.storage.data_mount_path,
        "--experiment-name", experiment_id,
        "--namespace", loaded.runtime.namespace,
        "--results-mount-path", loaded.storage.results_mount_path,
    ]
    trial_env = {
        "KFP_WORKFLOW_TUNE_SPEC_JSON": loaded.model_dump_json(),
        "KFP_WORKFLOW_TUNE_TRIAL_PARAMS_JSON": _trial_parameters_json(search_space),
    }
    manifest = build_katib_experiment(
        loaded,
        search_space,
        trial_image=loaded.runtime.image,
        trial_command=trial_command,
        trial_env=trial_env,
        experiment_name=experiment_id,
    )
    manifest_yaml = pyyaml.dump(manifest, default_flow_style=False, sort_keys=False)
    return loaded, experiment_id, manifest, manifest_yaml


def _tune_detail_payload(experiment: Dict[str, Any], namespace: str) -> Dict[str, Any]:
    """Build a normalized detail payload for one Katib experiment."""
    from kfp_workflow.tune.history import (
        extract_tune_spec,
        resolve_results,
        summarize_experiment,
    )

    tune_spec = extract_tune_spec(experiment) or {}
    summary = summarize_experiment(experiment)
    payload = {
        "id": summary["id"],
        "name": summary["name"],
        "namespace": namespace,
        "state": summary["state"],
        "created_at": summary["created_at"],
        "finished_at": summary["finished_at"],
        "best_value": summary.get("best_value"),
        "best_params": summary.get("best_params", {}),
        "trials": {
            "total": summary.get("n_trials"),
            "completed": summary.get("n_completed"),
            "pruned": summary.get("n_pruned"),
            "failed": summary.get("n_failed"),
        },
    }
    try:
        results = resolve_results(
            experiment=experiment,
            tune_spec=tune_spec,
            namespace=namespace,
        )
        payload["results"] = {
            "path": results["results_path"],
            "summary": results["summary"],
            "payload": results["payload"],
        }
        if results["payload"].get("best_params"):
            payload["best_params"] = results["payload"]["best_params"]
        if results["payload"].get("best_value") is not None:
            payload["best_value"] = results["payload"]["best_value"]
    except Exception as exc:
        payload["results"] = {"error": str(exc)}
    return payload


def _tune_wait(experiment_id: str, namespace: str) -> None:
    """Poll a Katib experiment until completion and display results."""
    from kfp_workflow.tune.history import watch_experiment

    typer.echo(f"Waiting for tune experiment '{experiment_id}'...")

    def on_update(summary: dict) -> None:
        state = summary.get("state", "PENDING")
        completed = summary.get("n_completed", 0)
        total = summary.get("n_trials", 0)
        best = summary.get("best_value")
        best_str = f"  best={best:.4f}" if best is not None else ""
        typer.echo(f"  [{state}] {completed}/{total} trials completed{best_str}")

    try:
        final = watch_experiment(
            experiment_id, namespace, on_update=on_update,
        )
    except TimeoutError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    state = final.get("state", "UNKNOWN")
    if state == "SUCCEEDED":
        typer.echo(f"\nTune experiment '{experiment_id}' completed successfully.")
        if final.get("best_value") is not None:
            typer.echo(f"Best objective value: {final['best_value']}")
        if final.get("best_params"):
            typer.echo("Best parameters:")
            typer.echo(json.dumps(final["best_params"], indent=2, default=str))
    else:
        typer.echo(f"\nTune experiment '{experiment_id}' finished with state: {state}")
        raise typer.Exit(code=1)


@tune_app.command("submit")
def cmd_tune_submit(
    spec: Path = typer.Option(..., help="Path to a tune YAML spec."),
    set_values: List[str] = typer.Option(
        [], "--set",
        help="Override spec values (e.g., --set hpo.max_trials=30).",
    ),
    output: Optional[Path] = typer.Option(
        None, help="Write Katib manifest YAML to this file.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print manifest without applying to the cluster.",
    ),
    wait: bool = typer.Option(
        False, "--wait",
        help="Wait for experiment completion and display results.",
    ),
) -> None:
    """Submit a Katib hyperparameter tuning experiment."""
    from kfp_workflow.cli.output import print_json

    loaded, experiment_id, manifest, manifest_yaml = _build_tune_manifest(spec, set_values)
    payload = {
        "id": experiment_id,
        "name": loaded.metadata.name,
        "namespace": loaded.runtime.namespace,
        "state": "SUBMITTED",
    }

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(manifest_yaml)
        payload["output_path"] = str(output)

    if dry_run:
        payload["state"] = "DRY_RUN"
        if _json_output:
            payload["manifest"] = manifest
            print_json(payload)
        else:
            typer.echo(manifest_yaml)
        return

    _run_kubectl(
        ["kubectl", "create", "-f", "-"],
        input_text=json.dumps(manifest),
    )

    if _json_output:
        print_json(payload)
    else:
        typer.echo(f"Submitted tune experiment: {experiment_id}")

    if wait:
        _tune_wait(experiment_id, loaded.runtime.namespace)


@tune_app.command("list")
def cmd_tune_list(
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
) -> None:
    """List managed Katib tune experiments."""
    from kfp_workflow.cli.output import print_json, print_table, style_run_state
    from kfp_workflow.tune.history import (
        is_tune_experiment,
        list_tune_experiments,
        summarize_experiment,
    )

    items = []
    for experiment in list_tune_experiments(namespace):
        if not is_tune_experiment(experiment):
            continue
        summary = summarize_experiment(experiment)
        items.append(summary)

    if _json_output:
        print_json(items)
        return

    rows = []
    for item in items:
        best = ""
        if item.get("best_value") is not None:
            best = f"{float(item['best_value']):.4f}"
        rows.append((
            short_id(item["id"]),
            item.get("name", "") or "",
            style_run_state(item["state"] or "UNKNOWN"),
            best,
            f"{item.get('n_completed', 0)}/{item.get('n_trials', 0)}",
            item.get("created_at", "") or "",
        ))
    print_table(
        title="Tune Experiments",
        columns=["ID", "NAME", "STATE", "BEST", "TRIALS", "CREATED"],
        rows=rows,
    )


@tune_app.command("get")
def cmd_tune_get(
    experiment_id: str = typer.Argument(..., help="Tune experiment ID or unique prefix."),
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
) -> None:
    """Show detailed info for one managed Katib tune experiment."""
    from kfp_workflow.cli.output import print_json, print_kv, style_run_state
    from kfp_workflow.tune.history import (
        resolve_tune_experiment,
    )

    try:
        experiment = resolve_tune_experiment(experiment_id, namespace)
    except LookupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    payload = _tune_detail_payload(experiment, namespace)

    if _json_output:
        print_json(payload)
        return

    pairs = [
        ("Tune", payload["name"] or ""),
        ("Experiment", payload["id"] or ""),
        ("Namespace", payload["namespace"] or ""),
        ("State", style_run_state(payload["state"] or "UNKNOWN")),
        ("Created", payload["created_at"] or ""),
        ("Finished", payload["finished_at"] or ""),
        ("Best Value", "" if payload.get("best_value") is None else str(payload["best_value"])),
        ("Trials", f"{payload['trials'].get('completed', 0)}/{payload['trials'].get('total', 0)} completed"),
        ("Pruned", str(payload["trials"].get("pruned", 0))),
        ("Failed", str(payload["trials"].get("failed", 0))),
    ]
    if payload.get("results", {}).get("path"):
        pairs.append(("Results Path", payload["results"]["path"]))
    if payload.get("results", {}).get("error"):
        pairs.append(("Results Error", payload["results"]["error"]))
    print_kv(pairs)
    if payload.get("best_params"):
        typer.echo("\nBest parameters:")
        typer.echo(json.dumps(payload["best_params"], indent=2, default=str))


@tune_app.command("download")
def cmd_tune_download(
    experiment_id: str = typer.Argument(..., help="Tune experiment ID or unique prefix."),
    output: Optional[Path] = typer.Option(None, help="Write results JSON to this path."),
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
    from_pvc: bool = typer.Option(
        False, "--from-pvc",
        help="Force PVC-based retrieval instead of Katib API.",
    ),
    apply_best: Optional[Path] = typer.Option(
        None, "--apply-best",
        help="Merge best params into a pipeline spec YAML and write it.",
    ),
) -> None:
    """Download a tune experiment's results to the local machine."""
    from kfp_workflow.cli.output import print_json
    from kfp_workflow.tune.history import (
        extract_tune_name,
        extract_tune_spec,
        resolve_results,
        resolve_tune_experiment,
    )
    from kfp_workflow.utils import dump_json, ensure_parent

    try:
        experiment = resolve_tune_experiment(experiment_id, namespace)
    except LookupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    resolved_name = experiment.get("metadata", {}).get("name", experiment_id)
    tune_spec = extract_tune_spec(experiment) or {}
    tune_name = extract_tune_name(experiment)
    results = resolve_results(
        experiment=experiment,
        tune_spec=tune_spec,
        namespace=namespace,
        from_pvc=from_pvc,
    )
    destination = output or Path.cwd() / f"{tune_name}-{resolved_name}.json"
    ensure_parent(destination)
    dump_json(results["payload"], destination)

    if apply_best:
        best_params = results["payload"].get("best_params", {})
        if best_params:
            import yaml as pyyaml
            from kfp_workflow.specs import merge_best_params
            from kfp_workflow.utils import load_yaml
            pipeline_raw = load_yaml(apply_best)
            merged = merge_best_params(pipeline_raw, best_params)
            apply_best.write_text(pyyaml.dump(merged, default_flow_style=False, sort_keys=False))
            typer.echo(f"Best params merged into {apply_best}")
        else:
            typer.echo("No best params to apply.", err=True)

    payload = {
        "id": resolved_name,
        "name": tune_name,
        "namespace": namespace,
        "results_path": results["results_path"],
        "output_path": str(destination),
    }
    if _json_output:
        print_json(payload)
        return

    typer.echo(f"Downloaded tune results to {destination}")


@tune_app.command("space")
def cmd_tune_space(
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


@tune_app.command("logs")
def cmd_tune_logs(
    experiment_id: str = typer.Argument(..., help="Tune experiment ID or unique prefix."),
    namespace: str = typer.Option(
        DEFAULT_NAMESPACE, help="Kubernetes namespace.",
    ),
    all_trials: bool = typer.Option(
        False, "--all", help="Show logs for all trials, not just failed ones.",
    ),
    trial: Optional[str] = typer.Option(
        None, "--trial", help="Show logs for a specific trial only.",
    ),
    tail: int = typer.Option(
        50, "--tail", help="Number of log lines per trial.",
    ),
) -> None:
    """Show trial logs for a tune experiment."""
    from kfp_workflow.cli.output import print_json
    from kfp_workflow.tune.history import (
        get_trial_logs,
        resolve_tune_experiment,
    )

    try:
        experiment = resolve_tune_experiment(experiment_id, namespace)
    except LookupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    resolved_name = experiment.get("metadata", {}).get("name", experiment_id)
    failed_only = not all_trials and trial is None

    entries = get_trial_logs(
        resolved_name,
        namespace,
        trial_name=trial,
        failed_only=failed_only,
        tail_lines=tail,
    )

    if _json_output:
        print_json([{
            "id": resolved_name,
            "name": entry["trial_name"],
            "namespace": namespace,
            "state": entry["phase"],
            "pod_name": entry["pod_name"],
            "logs": entry["logs"],
        } for entry in entries])
        return

    if not entries:
        label = "failed " if failed_only else ""
        typer.echo(f"No {label}trial logs found for experiment '{resolved_name}'.")
        return

    for entry in entries:
        phase = entry["phase"]
        color = "red" if phase == "Failed" else "green" if phase == "Succeeded" else "yellow"
        typer.echo(typer.style(
            f"── {entry['trial_name']} ({phase}) ──",
            fg=color, bold=True,
        ))
        typer.echo(entry["logs"])
        typer.echo("")


@tune_app.command("trial", hidden=True)
def cmd_tune_trial(
    spec_json: str = typer.Option(
        "", envvar="KFP_WORKFLOW_TUNE_SPEC_JSON",
        help="Serialized tune spec JSON.",
    ),
    trial_params_json: str = typer.Option(
        "", envvar="KFP_WORKFLOW_TUNE_TRIAL_PARAMS_JSON",
        help="Serialized Katib trial parameter assignments.",
    ),
    data_mount_path: str = typer.Option(
        "/mnt/data", help="Path where the data PVC is mounted.",
    ),
    experiment_name: Optional[str] = typer.Option(
        None, help="Katib experiment name for result persistence.",
    ),
    namespace: Optional[str] = typer.Option(
        None, help="Kubernetes namespace for result persistence.",
    ),
    results_mount_path: Optional[str] = typer.Option(
        None, help="Path where the tune results PVC is mounted.",
    ),
    trial_name: Optional[str] = typer.Option(
        None, help="Katib trial/job name. Defaults to env metadata.name.",
    ),
) -> None:
    """Run one Katib trial by delegating to the selected model plugin."""
    from kfp_workflow.plugins import get_plugin
    from kfp_workflow.specs import TuneSpec
    from kfp_workflow.tune.exceptions import TrialPruned
    from kfp_workflow.tune.results import persist_trial_result

    if not spec_json:
        typer.echo("Error: --spec-json or KFP_WORKFLOW_TUNE_SPEC_JSON required.", err=True)
        raise typer.Exit(code=1)
    if not trial_params_json:
        typer.echo("Error: --trial-params-json or KFP_WORKFLOW_TUNE_TRIAL_PARAMS_JSON required.", err=True)
        raise typer.Exit(code=1)

    loaded = TuneSpec.model_validate_json(spec_json)
    _validate_plugin_config_or_exit(loaded.model_dump())
    plugin = get_plugin(loaded.model.name)
    trial_params = _coerce_json_scalar_values(json.loads(trial_params_json))
    base_params = plugin.hpo_base_config(loaded.model_dump())
    merged_params = {**base_params, **trial_params}
    experiment_name = experiment_name or os.environ.get(
        "KFP_WORKFLOW_TUNE_EXPERIMENT_NAME",
        loaded.metadata.name,
    )
    namespace = namespace or os.environ.get(
        "KFP_WORKFLOW_TUNE_NAMESPACE",
        loaded.runtime.namespace,
    )
    trial_name = trial_name or os.environ.get(
        "KFP_WORKFLOW_TUNE_TRIAL_NAME",
        f"{experiment_name}-trial",
    )
    spec_dict = loaded.model_dump()
    if results_mount_path:
        spec_dict["storage"]["results_mount_path"] = results_mount_path

    def _persist(status: str, *, objective_value: Optional[float] = None, error: Optional[str] = None) -> None:
        try:
            persist_trial_result(
                spec=spec_dict,
                experiment_name=experiment_name,
                namespace=namespace,
                trial_name=trial_name,
                params=merged_params,
                status=status,
                objective_value=objective_value,
                error=error,
            )
        except OSError:
            # Local/manual trial execution may not have the PVC mounted.
            pass

    try:
        objective_value = plugin.hpo_objective(
            spec_dict,
            merged_params,
            data_mount_path,
        )
    except TrialPruned as exc:
        _persist("pruned", error=str(exc) or "Trial pruned.")
        raise
    except Exception as exc:
        _persist("failed", error=str(exc))
        raise

    _persist("completed", objective_value=float(objective_value))
    typer.echo(f"objective={float(objective_value)}")

# ---------------------------------------------------------------------------
# Entry point for `python -m kfp_workflow.cli.main`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
