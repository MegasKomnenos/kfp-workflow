"""KServe InferenceService creation and management."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, Optional


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
