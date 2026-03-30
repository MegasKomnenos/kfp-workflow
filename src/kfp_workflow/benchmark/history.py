"""Benchmark run discovery and result retrieval helpers."""

from __future__ import annotations

import ast
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.exceptions import ApiException
from kubernetes.stream import stream as k8s_stream


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


def find_workflow_for_run(run_id: str, namespace: str) -> Optional[Dict[str, Any]]:
    """Return the Argo workflow object for a KFP run, if present."""
    api = _custom_objects_api()
    result = api.list_namespaced_custom_object(
        group="argoproj.io",
        version="v1alpha1",
        namespace=namespace,
        plural="workflows",
        label_selector=f"pipeline/runid={run_id}",
    )
    items = result.get("items", [])
    if not items:
        return None
    items.sort(key=lambda item: item.get("metadata", {}).get("creationTimestamp", ""))
    return items[-1]


def _iter_spec_json_values(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        if node.get("name") == "spec_json" and isinstance(node.get("value"), str):
            yield node["value"]
        spec_node = node.get("spec_json")
        if isinstance(spec_node, str):
            yield spec_node
        elif isinstance(spec_node, dict):
            constant = spec_node.get("constant")
            if isinstance(constant, str):
                yield constant
            runtime_value = spec_node.get("runtimeValue", {})
            if isinstance(runtime_value, dict) and isinstance(runtime_value.get("constant"), str):
                yield runtime_value["constant"]
        for value in node.values():
            yield from _iter_spec_json_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_spec_json_values(item)
    elif isinstance(node, str):
        stripped = node.strip()
        if not stripped.startswith(("{", "[")):
            return
        try:
            parsed = json.loads(node)
        except Exception:
            return
        yield from _iter_spec_json_values(parsed)


def extract_benchmark_spec(workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the embedded benchmark spec JSON from a workflow object."""
    for raw in _iter_spec_json_values(workflow):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if _looks_like_benchmark_spec(parsed):
            return parsed
    return None


def _looks_like_benchmark_spec(value: Any) -> bool:
    return isinstance(value, dict) and {
        "metadata", "runtime", "storage", "model", "scenario",
    }.issubset(value.keys())


def _has_benchmark_component_in_spec(workflow: Dict[str, Any]) -> bool:
    """Check spec.templates for the run-benchmark-component task definition."""
    templates = workflow.get("spec", {}).get("templates") or []
    for template in templates:
        if template.get("name") == "run-benchmark-component":
            return True
        for task in (template.get("dag") or {}).get("tasks") or []:
            if task.get("name") == "run-benchmark-component":
                return True
    return False


def is_benchmark_workflow(workflow: Dict[str, Any]) -> bool:
    """Identify benchmark workflows from their template spec, node graph, and embedded spec."""
    nodes = workflow.get("status", {}).get("nodes", {}) or {}
    display_names = {
        node.get("displayName") or node.get("name") or ""
        for node in nodes.values()
    }
    found_in_nodes = "run-benchmark-component" in display_names
    found_in_spec = _has_benchmark_component_in_spec(workflow)
    if not found_in_nodes and not found_in_spec:
        return False
    return extract_benchmark_spec(workflow) is not None


def summarize_result_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact summary from a stored benchmark result payload."""
    scenario = payload.get("scenario", {}) if isinstance(payload.get("scenario"), dict) else {}
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    summary: Dict[str, Any] = {
        "status": payload.get("status", ""),
        "request_count": scenario.get("request_count"),
        "duration_seconds": scenario.get("duration_seconds"),
        "metric_keys": sorted(metrics.keys()),
    }
    metric_0 = metrics.get("metric_0", {}) if isinstance(metrics.get("metric_0"), dict) else {}
    if "delta_joules" in metric_0:
        summary["delta_joules"] = metric_0["delta_joules"]
    for metric_val in metrics.values():
        if isinstance(metric_val, dict) and "f1_score" in metric_val:
            summary["f1_score"] = metric_val["f1_score"]
            break
    return summary


def resolve_results(
    *,
    workflow: Dict[str, Any],
    benchmark_spec: Dict[str, Any],
    namespace: str,
) -> Dict[str, Any]:
    """Locate and read the stored benchmark results payload from the PVC."""
    benchmark_name = benchmark_spec["metadata"]["name"]
    workflow_name = workflow.get("metadata", {}).get("name", "")
    storage = benchmark_spec.get("storage", {})
    pvc_name = storage.get("results_pvc", "benchmark-store")
    candidates = _list_result_candidates(
        pvc_name=pvc_name,
        namespace=namespace,
        benchmark_name=benchmark_name,
    )
    filtered = [
        candidate for candidate in candidates
        if workflow_name and workflow_name in Path(candidate).parent.name
    ]
    if not filtered and len(candidates) == 1:
        filtered = candidates
    if not filtered:
        raise FileNotFoundError(
            f"No results.json found for benchmark '{benchmark_name}' and workflow '{workflow_name}'."
        )
    if len(filtered) > 1:
        raise RuntimeError(
            f"Multiple results.json candidates found for workflow '{workflow_name}': {filtered}"
        )

    result_path = filtered[0]
    payload = _parse_result_payload(
        _read_result_file(
            pvc_name=pvc_name,
            namespace=namespace,
            result_path=result_path,
        )
    )
    return {
        "results_path": result_path,
        "payload": payload,
        "summary": summarize_result_payload(payload),
    }


def _list_result_candidates(
    *,
    pvc_name: str,
    namespace: str,
    benchmark_name: str,
) -> List[str]:
    root = f"/mnt/results/benchmark-results/{benchmark_name}"
    command = ["sh", "-lc", f"find {root} -name results.json -print 2>/dev/null | sort"]
    output = _exec_with_results_pvc(
        pvc_name=pvc_name,
        namespace=namespace,
        command=command,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def _read_result_file(
    *,
    pvc_name: str,
    namespace: str,
    result_path: str,
) -> str:
    command = ["cat", result_path]
    return _exec_with_results_pvc(
        pvc_name=pvc_name,
        namespace=namespace,
        command=command,
    )


def _parse_result_payload(raw: str) -> Dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = ast.literal_eval(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Benchmark results payload must be a dict, got {type(payload)!r}.")
    return payload


def _exec_with_results_pvc(
    *,
    pvc_name: str,
    namespace: str,
    command: List[str],
) -> str:
    pod_name = f"benchmark-results-reader-{uuid.uuid4().hex[:8]}"
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
        _wait_for_pod_ready(v1, namespace=namespace, pod_name=pod_name)
        return k8s_stream(
            v1.connect_get_namespaced_pod_exec,
            pod_name,
            namespace,
            command=command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
        )
    finally:
        try:
            v1.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace,
                body=k8s_client.V1DeleteOptions(),
            )
        except ApiException as exc:
            if exc.status != 404:
                raise


def _wait_for_pod_ready(v1: Any, *, namespace: str, pod_name: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        if pod.status.phase == "Running":
            conditions = pod.status.conditions or []
            if any(cond.type == "Ready" and cond.status == "True" for cond in conditions):
                return
        if pod.status.phase in {"Failed", "Succeeded"}:
            raise RuntimeError(f"Reader pod '{pod_name}' terminated before becoming Ready.")
        time.sleep(1.0)
    raise TimeoutError(f"Reader pod '{pod_name}' did not become Ready within {timeout} seconds.")
