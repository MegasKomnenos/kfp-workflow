"""Shared workflow-oriented CLI helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from kfp_server_api.exceptions import ApiException

from kfp_workflow.pipeline.connection import kfp_connection

DEFAULT_NAMESPACE = "kubeflow-user-example-com"
DEFAULT_HOST = "http://127.0.0.1:8888"
DEFAULT_USER = "user@example.com"
DISPLAY_ID_WIDTH = 12


def short_id(raw_id: str, *, width: int = DISPLAY_ID_WIDTH) -> str:
    """Return a stable short display form for a backend identifier."""
    return raw_id if len(raw_id) <= width else raw_id[:width]


def run_state_str(state: object) -> str:
    """Extract run state as a plain string (handles enum or str)."""
    if state is None:
        return "UNKNOWN"
    return state.value if hasattr(state, "value") else str(state)


def find_workflow_for_run(run_id: str, namespace: str) -> Optional[Dict[str, Any]]:
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


def workflow_summary(workflow: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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


def build_run_payload(
    run: Any,
    workflow: Optional[Dict[str, Any]],
    *,
    namespace: str,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a normalized run payload shared by pipeline and benchmark."""
    payload = {
        "id": run.run_id,
        "name": name or run.display_name,
        "state": run_state_str(run.state),
        "created_at": str(run.created_at),
        "finished_at": str(run.finished_at),
        "namespace": namespace,
        "workflow": {
            "run_id": run.run_id,
            "display_name": run.display_name,
            "experiment_id": run.experiment_id,
            "error": str(run.error) if run.error else None,
        },
    }
    summary = workflow_summary(workflow)
    if summary:
        payload["workflow"].update({
            "name": summary["name"],
            "phase": summary["phase"],
            "progress": summary["progress"],
            "finished_at": summary["finished_at"],
            "message": summary["message"],
            "pending_nodes": summary["pending_nodes"],
            "failed_nodes": summary["failed_nodes"],
        })
    return payload


def iter_runs(
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


def iter_experiments(
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


def resolve_unique_id_prefix(
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
    preview = ", ".join(short_id(match) for match in prefix_matches[:5])
    raise typer.BadParameter(
        f"ID prefix '{raw_id}' matches multiple {kind}s: {preview}"
    )


def resolve_run(client: Any, *, run_id: str, namespace: str) -> Any:
    """Resolve a full run ID or unique prefix and return the run object."""
    try:
        run = client.get_run(run_id=run_id)
        resolved = getattr(run, "run_id", None)
        if isinstance(resolved, str) and resolved:
            return run
    except ApiException as exc:
        if getattr(exc, "status", None) != 404:
            raise
    candidates = [str(run.run_id) for run in iter_runs(client, namespace=namespace) if getattr(run, "run_id", None)]
    resolved_run_id = resolve_unique_id_prefix(run_id, candidates, kind="run")
    return client.get_run(run_id=resolved_run_id)


def resolve_experiment_id(client: Any, *, experiment_id: str, namespace: str) -> str:
    """Resolve a full experiment ID or unique prefix against visible experiments."""
    candidates = [
        str(experiment.experiment_id)
        for experiment in iter_experiments(client, namespace=namespace)
        if getattr(experiment, "experiment_id", None)
    ]
    return resolve_unique_id_prefix(experiment_id, candidates, kind="experiment")


def compiled_package_path(name: str) -> Path:
    """Return the canonical compiled package path for a workflow spec name."""
    compiled_dir = Path("compiled")
    compiled_dir.mkdir(parents=True, exist_ok=True)
    return compiled_dir / f"{name}.yaml"


def submit_pipeline_package(
    *,
    package_path: Path,
    run_name: str,
    experiment_name: str,
    namespace: str,
    runtime_host: Optional[str],
    port_forward_namespace: Optional[str],
    port_forward_service: Optional[str],
    host: Optional[str] = None,
    existing_token: Optional[str] = None,
    cookies: Optional[str] = None,
    user: Optional[str] = None,
) -> str:
    """Submit a compiled pipeline package to Kubeflow and return the backend run ID."""
    with kfp_connection(
        namespace=namespace,
        host=host or runtime_host,
        port_forward_namespace=port_forward_namespace,
        port_forward_service=port_forward_service,
        user=user or DEFAULT_USER,
        existing_token=existing_token,
        cookies=cookies,
    ) as client:
        try:
            client.create_experiment(
                name=experiment_name,
                namespace=namespace,
            )
        except Exception:
            pass

        try:
            run = client.create_run_from_pipeline_package(
                pipeline_file=str(package_path),
                arguments={},
                run_name=run_name,
                experiment_name=experiment_name,
                namespace=namespace,
            )
        except ApiException as exc:
            if exc.status == 401:
                raise SystemExit(
                    "KFP submission was unauthorized. Re-run with --existing-token or --cookies."
                ) from exc
            raise
        return run.run_id
