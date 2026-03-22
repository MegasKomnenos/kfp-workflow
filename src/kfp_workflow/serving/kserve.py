"""KServe InferenceService creation and management."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_inference_service_manifest(
    name: str,
    namespace: str,
    model_pvc_name: str,
    model_subpath: str,
    runtime: str = "kserve-torchserve",
    replicas: int = 1,
    resources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a KServe InferenceService manifest dict.

    Returns a dict matching the KServe InferenceService CRD structure
    with a TorchServe runtime and PVC storage source.
    """
    manifest: Dict[str, Any] = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "predictor": {
                "model": {
                    "modelFormat": {"name": "pytorch"},
                    "runtime": runtime,
                    "storageUri": f"pvc://{model_pvc_name}/{model_subpath}",
                },
                "minReplicas": replicas,
            },
        },
    }

    if resources:
        manifest["spec"]["predictor"]["model"]["resources"] = {
            "requests": {
                "cpu": resources.get("cpu_request", "2"),
                "memory": resources.get("memory_request", "4Gi"),
            },
            "limits": {
                "cpu": resources.get("cpu_limit", "2"),
                "memory": resources.get("memory_limit", "4Gi"),
            },
        }

    return manifest


def create_inference_service(
    name: str,
    namespace: str,
    model_pvc_name: str,
    model_subpath: str,
    runtime: str = "kserve-torchserve",
    replicas: int = 1,
    resources: Optional[Dict[str, Any]] = None,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Create a KServe InferenceService via the KServe Python SDK.

    When *dry_run* is ``True``, returns the manifest without applying.
    """
    manifest = build_inference_service_manifest(
        name=name,
        namespace=namespace,
        model_pvc_name=model_pvc_name,
        model_subpath=model_subpath,
        runtime=runtime,
        replicas=replicas,
        resources=resources,
    )

    if dry_run:
        return manifest

    raise NotImplementedError(
        "KServe InferenceService creation not yet implemented. "
        "Requires the kserve Python SDK and cluster access."
    )


def delete_inference_service(name: str, namespace: str) -> None:
    """Delete a KServe InferenceService by name."""
    raise NotImplementedError(
        "KServe InferenceService deletion not yet implemented."
    )
