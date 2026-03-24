"""Typer CLI application for kfp-workflow."""

from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path
from typing import List, Optional

import typer

# Suppress noisy third-party deprecation warnings that are not actionable
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", category=FutureWarning, module="kfp")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_json_output = False


def _run_state_str(state: object) -> str:
    """Extract run state as a plain string (handles enum or str)."""
    if state is None:
        return "UNKNOWN"
    return state.value if hasattr(state, "value") else str(state)

# ---------------------------------------------------------------------------
# App and sub-apps
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="kfp-workflow",
    help="KFP v2 training pipeline and KServe serving workflow manager.",
    add_completion=False,
)

pipeline_app = typer.Typer(help="Compile and submit training pipelines.")
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
        help="Spec type: 'pipeline' or 'serving'.",
    ),
    set_values: List[str] = typer.Option(
        [], "--set",
        help="Override spec values (e.g., --set train.batch_size=128). Pipeline specs only.",
    ),
) -> None:
    """Load and validate a spec file."""
    from kfp_workflow.specs import load_pipeline_spec_with_overrides, load_serving_spec

    if spec_type == "pipeline":
        loaded = load_pipeline_spec_with_overrides(spec, set_values or None)
        if set_values:
            from kfp_workflow.config_override import validate_plugin_config
            for w in validate_plugin_config(loaded.model_dump()):
                typer.echo(f"Warning: {w}", err=True)
    elif spec_type == "serving":
        loaded = load_serving_spec(spec)
    elif spec_type == "tune":
        from kfp_workflow.specs import load_tune_spec_with_overrides
        loaded = load_tune_spec_with_overrides(spec, set_values or None)
    else:
        typer.echo(f"Unknown spec type: {spec_type}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Spec '{loaded.metadata.name}' validated successfully.")
    typer.echo(json.dumps(loaded.model_dump(), indent=2, default=str))


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
        run = client.get_run(run_id=run_id)

    state = _run_state_str(run.state)

    if _json_output:
        print_json({
            "run_id": run.run_id,
            "display_name": run.display_name,
            "state": state,
            "created_at": str(run.created_at),
            "finished_at": str(run.finished_at),
            "experiment_id": run.experiment_id,
            "error": str(run.error) if run.error else None,
        })
        return

    pairs = [
        ("Run ID", run.run_id or ""),
        ("Name", run.display_name or ""),
        ("State", style_run_state(state)),
        ("Created", str(run.created_at or "")),
        ("Finished", str(run.finished_at or "")),
        ("Experiment ID", run.experiment_id or ""),
    ]
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
        response = client.list_runs(
            page_size=page_size,
            sort_by=sort_by,
            experiment_id=experiment_id,
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

    with kfp_connection(
        namespace=namespace, host=host, user=user,
        existing_token=existing_token, cookies=cookies,
    ) as client:
        with console.status(f"Waiting for run {run_id[:8]}..."):
            run = client.wait_for_run_completion(run_id, timeout=timeout)

    state = _run_state_str(run.state)
    console.print(f"Run {run_id[:8]} finished: {style_run_state(state)}")

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
        client.terminate_run(run_id)

    typer.echo(f"Run {run_id[:8]} terminated.")


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

    k8s_config.load_kube_config()
    v1 = k8s_client.CoreV1Api()

    pods = v1.list_namespaced_pod(
        namespace=namespace,
        label_selector=f"pipeline/runid={run_id}",
    )

    # Filter to implementation pods (contain actual component output)
    impl_pods = [
        p for p in pods.items
        if "system-container-impl" in p.metadata.name
    ]

    if step:
        impl_pods = [p for p in impl_pods if step in p.metadata.name]

    if not impl_pods:
        typer.echo("No matching pods found.", err=True)
        raise typer.Exit(code=1)

    for pod in sorted(impl_pods, key=lambda p: p.metadata.name):
        console.rule(f"[bold]{pod.metadata.name}[/bold]")
        try:
            log = v1.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=namespace,
                container="main",
            )
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
) -> None:
    """Create a KServe InferenceService from a serving spec."""
    from kfp_workflow.serving.kserve import create_inference_service
    from kfp_workflow.specs import load_serving_spec

    loaded = load_serving_spec(spec)
    result = create_inference_service(
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


@serve_app.command("delete")
def cmd_serve_delete(
    name: str = typer.Option(..., help="InferenceService name."),
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace."
    ),
) -> None:
    """Delete a KServe InferenceService."""
    from kfp_workflow.serving.kserve import delete_inference_service

    delete_inference_service(name=name, namespace=namespace)
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
    from kfp_workflow.serving.kserve import list_inference_services

    items = list_inference_services(namespace=namespace)

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
    from kfp_workflow.serving.kserve import get_inference_service

    svc = get_inference_service(name=name, namespace=namespace)

    ready = _isvc_ready(svc)
    url = svc.get("status", {}).get("url", "")
    created = svc["metadata"].get("creationTimestamp", "")
    conditions = svc.get("status", {}).get("conditions", [])

    if _json_output:
        print_json({
            "name": name,
            "namespace": namespace,
            "ready": ready,
            "url": url,
            "created": created,
            "conditions": conditions,
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
        pairs.append((f"  {ctype}", cstatus))

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
    dry_run: bool = typer.Option(False, help="Print manifests without applying."),
) -> None:
    """Create PVCs for data and model storage."""
    from kfp_workflow.specs import load_pipeline_spec

    loaded = load_pipeline_spec(spec)
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
    plugin = get_plugin(loaded.model.name)
    search_space = resolve_search_space(plugin, loaded.model_dump())

    trial_command = [
        "python", "-m", "kfp_workflow.cli.main",
        "pipeline", "submit",
        "--spec", "/mnt/config/spec.yaml",
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
