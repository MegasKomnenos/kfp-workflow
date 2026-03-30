"""Tune experiment discovery and result retrieval helpers."""

from __future__ import annotations

import ast
import base64
import json
import shlex
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.exceptions import ApiException
from kubernetes.stream import stream as k8s_stream

from kfp_workflow.tune.results import (
    aggregate_experiment_results,
    experiment_result_dir,
    experiment_results_path,
)

_SPEC_ANNOTATION = "kfp-workflow/spec-json"
_TYPE_LABEL = "kfp-workflow/type"
_MANAGED_BY_LABEL = "app.kubernetes.io/managed-by"


def _load_config() -> None:
    """Load in-cluster config when available, otherwise local kubeconfig."""
    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()


def _custom_objects_api() -> Any:
    _load_config()
    return k8s_client.CustomObjectsApi()


def _core_v1_api() -> Any:
    _load_config()
    return k8s_client.CoreV1Api()


def list_tune_experiments(namespace: str) -> List[Dict[str, Any]]:
    """Return all Katib Experiment objects in a namespace."""
    api = _custom_objects_api()
    result = api.list_namespaced_custom_object(
        group="kubeflow.org",
        version="v1beta1",
        namespace=namespace,
        plural="experiments",
    )
    items = result.get("items", [])
    items.sort(
        key=lambda item: item.get("metadata", {}).get("creationTimestamp", ""),
        reverse=True,
    )
    return items


def get_tune_experiment(name: str, namespace: str) -> Optional[Dict[str, Any]]:
    """Return one Katib Experiment object by name, if it exists."""
    api = _custom_objects_api()
    try:
        return api.get_namespaced_custom_object(
            group="kubeflow.org",
            version="v1beta1",
            namespace=namespace,
            plural="experiments",
            name=name,
        )
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise


def extract_tune_spec(experiment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the embedded tune spec JSON from a Katib Experiment object."""
    metadata = experiment.get("metadata", {})
    annotations = metadata.get("annotations", {}) or {}
    raw = annotations.get(_SPEC_ANNOTATION)
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    if _looks_like_tune_spec(parsed):
        return parsed
    return None


def _looks_like_tune_spec(value: Any) -> bool:
    return isinstance(value, dict) and {
        "metadata", "runtime", "storage", "model", "dataset", "hpo",
    }.issubset(value.keys())


def is_tune_experiment(experiment: Dict[str, Any]) -> bool:
    """Identify kfp-workflow-managed Katib tune experiments."""
    metadata = experiment.get("metadata", {})
    labels = metadata.get("labels", {}) or {}
    if labels.get(_MANAGED_BY_LABEL) != "kfp-workflow":
        return False
    if labels.get(_TYPE_LABEL) != "tune":
        return False
    return extract_tune_spec(experiment) is not None


def summarize_result_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact summary from a stored tune result payload."""
    return {
        "status": str(payload.get("status", "")),
        "best_value": payload.get("best_value"),
        "best_trial_name": payload.get("best_trial_name", ""),
        "n_trials": payload.get("n_trials"),
        "n_completed": payload.get("n_completed"),
        "n_pruned": payload.get("n_pruned"),
        "n_failed": payload.get("n_failed"),
        "objective_metric_name": payload.get("objective_metric_name", "objective"),
    }


def summarize_experiment(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact status summary from the Katib Experiment CRD."""
    metadata = experiment.get("metadata", {})
    status = experiment.get("status", {}) or {}
    best = _current_optimal_trial(status)
    counts = _status_counts(status)
    return {
        "experiment_name": metadata.get("name", ""),
        "state": _experiment_state(experiment),
        "created_at": metadata.get("creationTimestamp", ""),
        "finished_at": status.get("completionTime", ""),
        "best_value": best.get("objective_value"),
        "best_params": best.get("params", {}),
        "n_trials": counts["n_trials"],
        "n_completed": counts["n_completed"],
        "n_pruned": counts["n_pruned"],
        "n_failed": counts["n_failed"],
    }


def resolve_results(
    *,
    experiment: Dict[str, Any],
    tune_spec: Dict[str, Any],
    namespace: str,
) -> Dict[str, Any]:
    """Locate, lazily aggregate, and read the stored tune results payload."""
    experiment_name = experiment.get("metadata", {}).get("name", "")
    pvc_name = tune_spec.get("storage", {}).get("results_pvc", "tune-store")
    result_path = str(experiment_results_path(tune_spec, experiment_name))

    try:
        raw = _read_result_file(
            pvc_name=pvc_name,
            namespace=namespace,
            result_path=result_path,
        )
        payload = _parse_result_payload(raw)
    except Exception:
        trial_payloads = _read_trial_payloads(
            pvc_name=pvc_name,
            namespace=namespace,
            tune_spec=tune_spec,
            experiment_name=experiment_name,
        )
        if not trial_payloads:
            raise FileNotFoundError(
                f"No tune results found for experiment '{experiment_name}'."
            )
        payload = aggregate_experiment_results(
            spec=tune_spec,
            experiment_name=experiment_name,
            namespace=namespace,
            experiment_status=_experiment_state(experiment),
            created_at=experiment.get("metadata", {}).get("creationTimestamp", ""),
            completed_at=(experiment.get("status", {}) or {}).get("completionTime", ""),
            trial_payloads=trial_payloads,
        )
        _write_result_file(
            pvc_name=pvc_name,
            namespace=namespace,
            result_path=result_path,
            payload=payload,
        )

    return {
        "results_path": result_path,
        "payload": payload,
        "summary": summarize_result_payload(payload),
    }


def _read_trial_payloads(
    *,
    pvc_name: str,
    namespace: str,
    tune_spec: Dict[str, Any],
    experiment_name: str,
) -> List[Dict[str, Any]]:
    trial_dir = experiment_result_dir(tune_spec, experiment_name) / "trials"
    command = [
        "sh",
        "-lc",
        f"find {shlex.quote(str(trial_dir))} -name '*.json' -print 2>/dev/null | sort",
    ]
    output = _exec_with_results_pvc(
        pvc_name=pvc_name,
        namespace=namespace,
        command=command,
    )
    paths = [line.strip() for line in output.splitlines() if line.strip()]
    payloads: List[Dict[str, Any]] = []
    for path in paths:
        payloads.append(
            _parse_result_payload(
                _read_result_file(
                    pvc_name=pvc_name,
                    namespace=namespace,
                    result_path=path,
                )
            )
        )
    return payloads


def _parse_result_payload(raw: str) -> Dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = ast.literal_eval(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Tune results payload must be a dict, got {type(payload)!r}.")
    return payload


def _current_optimal_trial(status: Dict[str, Any]) -> Dict[str, Any]:
    optimal = status.get("currentOptimalTrial", {}) or {}
    observation = optimal.get("observation", {}) or {}
    metrics = observation.get("metrics", []) or []
    objective_value = None
    for metric in metrics:
        if metric.get("name") == "objective" and metric.get("value") is not None:
            try:
                objective_value = float(metric["value"])
            except (TypeError, ValueError):
                objective_value = metric["value"]
            break
    params = {
        item.get("name"): item.get("value")
        for item in (optimal.get("parameterAssignments", []) or [])
        if item.get("name")
    }
    return {
        "objective_value": objective_value,
        "params": params,
    }


def _status_counts(status: Dict[str, Any]) -> Dict[str, int]:
    completed = _coerce_int(status.get("trialsSucceeded"))
    failed = _coerce_int(status.get("trialsFailed"))
    running = _coerce_int(status.get("trialsRunning"))
    pending = _coerce_int(status.get("trialsPending"))
    created = _coerce_int(status.get("trialsCreated"))
    n_trials = created or (completed + failed + running + pending)
    return {
        "n_trials": n_trials,
        "n_completed": completed,
        "n_pruned": 0,
        "n_failed": failed,
    }


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _experiment_state(experiment: Dict[str, Any]) -> str:
    status = experiment.get("status", {}) or {}
    for condition in reversed(status.get("conditions", []) or []):
        cond_type = str(condition.get("type", "")).upper()
        cond_status = str(condition.get("status", "")).lower()
        if cond_status != "true":
            continue
        if cond_type in {"SUCCEEDED", "SUCCESSFUL", "COMPLETE", "COMPLETED"}:
            return "SUCCEEDED"
        if cond_type in {"FAILED", "FAILURE"}:
            return "FAILED"
        if cond_type in {"RUNNING"}:
            return "RUNNING"
        if cond_type in {"CREATED", "STARTED"}:
            return "PENDING"
    if status.get("completionTime"):
        return "SUCCEEDED"
    if status.get("startTime"):
        return "RUNNING"
    return "PENDING"


def _read_result_file(
    *,
    pvc_name: str,
    namespace: str,
    result_path: str,
) -> str:
    return _exec_with_results_pvc(
        pvc_name=pvc_name,
        namespace=namespace,
        command=["cat", result_path],
    )


def _write_result_file(
    *,
    pvc_name: str,
    namespace: str,
    result_path: str,
    payload: Dict[str, Any],
) -> None:
    encoded = base64.b64encode(json.dumps(payload, indent=2, default=str).encode("utf-8")).decode("ascii")
    parent = str(Path(result_path).parent)
    command = [
        "sh",
        "-lc",
        (
            f"mkdir -p {shlex.quote(parent)} && "
            f"printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(result_path)}"
        ),
    ]
    _exec_with_results_pvc(
        pvc_name=pvc_name,
        namespace=namespace,
        command=command,
    )


def _exec_with_results_pvc(
    *,
    pvc_name: str,
    namespace: str,
    command: List[str],
) -> str:
    pod_name = f"tune-results-reader-{uuid.uuid4().hex[:8]}"
    v1 = _core_v1_api()
    pod = k8s_client.V1Pod(
        metadata=k8s_client.V1ObjectMeta(name=pod_name, namespace=namespace),
        spec=k8s_client.V1PodSpec(
            restart_policy="Never",
            containers=[
                k8s_client.V1Container(
                    name="reader",
                    image="busybox:1.36",
                    command=["sh", "-c", "sleep 300"],
                    volume_mounts=[
                        k8s_client.V1VolumeMount(
                            name="results",
                            mount_path="/mnt/results",
                        )
                    ],
                )
            ],
            volumes=[
                k8s_client.V1Volume(
                    name="results",
                    persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=pvc_name,
                    ),
                )
            ],
        ),
    )

    try:
        v1.create_namespaced_pod(namespace=namespace, body=pod)
        for _ in range(60):
            current = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            if current.status.phase == "Running":
                break
            if current.status.phase in {"Failed", "Succeeded"}:
                raise RuntimeError(
                    f"Helper pod '{pod_name}' terminated before exec ({current.status.phase})."
                )
            time.sleep(1)
        else:
            raise TimeoutError(f"Timed out waiting for helper pod '{pod_name}' to start.")

        return str(
            k8s_stream(
                v1.connect_get_namespaced_pod_exec,
                pod_name,
                namespace,
                command=command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
            )
        )
    finally:
        try:
            v1.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace,
                grace_period_seconds=0,
            )
        except Exception:
            pass
