"""KServe InferenceService creation and management."""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Dict, List, Optional


def _ready_status(svc: Dict[str, Any]) -> str:
    """Return the Ready condition status for an InferenceService."""
    conditions = svc.get("status", {}).get("conditions", [])
    for cond in conditions:
        if cond.get("type") == "Ready":
            return cond.get("status", "Unknown")
    return "Unknown"


def _condition_payloads(svc: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize InferenceService conditions for CLI output."""
    payloads: List[Dict[str, str]] = []
    for cond in svc.get("status", {}).get("conditions", []):
        payloads.append({
            "type": cond.get("type", ""),
            "status": cond.get("status", ""),
            "reason": cond.get("reason", ""),
            "message": cond.get("message", ""),
            "lastTransitionTime": cond.get("lastTransitionTime", ""),
        })
    return payloads


def _event_timestamp(event: Any) -> Any:
    """Best-effort sort key for Kubernetes events."""
    return (
        getattr(event, "last_timestamp", None)
        or getattr(event, "event_time", None)
        or getattr(event, "first_timestamp", None)
        or getattr(getattr(event, "metadata", None), "creation_timestamp", None)
    )


def _event_payload(event: Any) -> Dict[str, str]:
    """Normalize a Kubernetes event for CLI output."""
    return {
        "type": getattr(event, "type", "") or "",
        "reason": getattr(event, "reason", "") or "",
        "message": getattr(event, "message", "") or "",
        "count": str(getattr(event, "count", "") or ""),
        "lastTimestamp": str(_event_timestamp(event) or ""),
    }


def build_inference_service_manifest(
    name: str,
    namespace: str,
    model_pvc_name: str,
    model_subpath: str,
    runtime: str = "custom",
    predictor_image: str = "",
    model_name: str = "",
    replicas: int = 1,
    resources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a KServe InferenceService manifest dict.

    Supports both standard KServe runtimes and custom container predictors.
    When ``runtime="custom"``, a custom container spec is generated using
    ``predictor_image`` with environment variables for the plugin predictor.
    """
    manifest: Dict[str, Any] = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "annotations": {
                # Use RawDeployment mode to support PVC mounts
                # (Knative serverless mode does not allow PVCs)
                "serving.kserve.io/deploymentMode": "RawDeployment",
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": replicas,
            },
        },
    }

    resource_spec = {}
    if resources:
        resource_spec = {
            "requests": {
                "cpu": resources.get("cpu_request", "2"),
                "memory": resources.get("memory_request", "4Gi"),
            },
            "limits": {
                "cpu": resources.get("cpu_limit", "2"),
                "memory": resources.get("memory_limit", "4Gi"),
            },
        }

    if runtime == "custom":
        # Custom container predictor — runs the plugin predictor entrypoint
        container = {
            "name": "kserve-container",
            "image": predictor_image,
            "imagePullPolicy": "IfNotPresent",
            "command": [
                "python", "-m", "kfp_workflow.serving.predictor",
            ],
            "env": [
                {"name": "MODEL_PLUGIN_NAME", "value": model_name},
                {"name": "MODEL_DIR", "value": f"/mnt/models/{model_subpath}"},
                {"name": "MODEL_NAME", "value": name},
            ],
            "volumeMounts": [
                {
                    "name": "model-store",
                    "mountPath": "/mnt/models",
                    "readOnly": True,
                },
            ],
        }
        if resource_spec:
            container["resources"] = resource_spec

        manifest["spec"]["predictor"]["containers"] = [container]
        manifest["spec"]["predictor"]["volumes"] = [
            {
                "name": "model-store",
                "persistentVolumeClaim": {
                    "claimName": model_pvc_name,
                },
            },
        ]
    else:
        # Standard KServe model runtime (e.g., kserve-torchserve)
        model_spec: Dict[str, Any] = {
            "modelFormat": {"name": "pytorch"},
            "runtime": runtime,
            "storageUri": f"pvc://{model_pvc_name}/{model_subpath}",
        }
        if resource_spec:
            model_spec["resources"] = resource_spec

        manifest["spec"]["predictor"]["model"] = model_spec

    return manifest


def create_inference_service(
    name: str,
    namespace: str,
    model_pvc_name: str,
    model_subpath: str,
    runtime: str = "custom",
    predictor_image: str = "",
    model_name: str = "",
    replicas: int = 1,
    resources: Optional[Dict[str, Any]] = None,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Create a KServe InferenceService via kubectl apply.

    When *dry_run* is ``True``, returns the manifest without applying.
    """
    manifest = build_inference_service_manifest(
        name=name,
        namespace=namespace,
        model_pvc_name=model_pvc_name,
        model_subpath=model_subpath,
        runtime=runtime,
        predictor_image=predictor_image,
        model_name=model_name,
        replicas=replicas,
        resources=resources,
    )

    if dry_run:
        return manifest

    subprocess.run(
        ["kubectl", "apply", "-n", namespace, "-f", "-"],
        input=json.dumps(manifest),
        check=True,
        text=True,
        capture_output=True,
    )
    return manifest


def delete_inference_service(name: str, namespace: str) -> None:
    """Delete a KServe InferenceService by name."""
    subprocess.run(
        [
            "kubectl", "delete", "inferenceservice", name,
            "-n", namespace,
        ],
        check=True,
        text=True,
        capture_output=True,
    )


def list_inference_services(namespace: str) -> List[Dict[str, Any]]:
    """List InferenceServices in a namespace via the Kubernetes API."""
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    k8s_config.load_kube_config()
    api = k8s_client.CustomObjectsApi()
    result = api.list_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=namespace,
        plural="inferenceservices",
    )
    return result.get("items", [])


def get_inference_service(name: str, namespace: str) -> Dict[str, Any]:
    """Get a single InferenceService by name."""
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    k8s_config.load_kube_config()
    api = k8s_client.CustomObjectsApi()
    return api.get_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=namespace,
        plural="inferenceservices",
        name=name,
    )


def get_inference_service_events(
    name: str,
    namespace: str,
    *,
    limit: int = 5,
    event_type: Optional[str] = "Warning",
) -> List[Dict[str, str]]:
    """Return recent Kubernetes events for an InferenceService."""
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    k8s_config.load_kube_config()
    v1 = k8s_client.CoreV1Api()
    selector = (
        f"involvedObject.kind=InferenceService,"
        f"involvedObject.name={name}"
    )
    result = v1.list_namespaced_event(namespace=namespace, field_selector=selector)
    items = result.items or []
    if event_type:
        items = [event for event in items if getattr(event, "type", None) == event_type]
    items = sorted(items, key=_event_timestamp)
    if limit > 0:
        items = items[-limit:]
    return [_event_payload(event) for event in items]


def get_inference_service_diagnostics(
    name: str,
    namespace: str,
    *,
    event_limit: int = 5,
) -> Dict[str, Any]:
    """Return an InferenceService plus normalized conditions and warnings."""
    svc = get_inference_service(name=name, namespace=namespace)
    return {
        "service": svc,
        "ready": _ready_status(svc),
        "conditions": _condition_payloads(svc),
        "events": get_inference_service_events(
            name=name,
            namespace=namespace,
            limit=event_limit,
        ),
    }


def wait_for_inference_service_ready(
    name: str,
    namespace: str,
    *,
    timeout: int = 300,
    poll_interval: float = 2.0,
) -> Dict[str, Any]:
    """Poll an InferenceService until Ready=True or timeout expires."""
    deadline = time.time() + timeout
    latest: Dict[str, Any] = {
        "service": {},
        "ready": "Unknown",
        "conditions": [],
        "events": [],
    }
    while time.time() < deadline:
        try:
            latest = get_inference_service_diagnostics(name=name, namespace=namespace)
        except Exception:
            time.sleep(poll_interval)
            continue
        if latest["ready"] == "True":
            return latest
        time.sleep(poll_interval)
    return latest
