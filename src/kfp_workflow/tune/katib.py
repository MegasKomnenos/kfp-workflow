"""Generate Katib Experiment CRD manifests from project-owned types."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from kfp_workflow.specs import SearchParamSpec, TuneSpec


# ---------------------------------------------------------------------------
# Parameter conversion
# ---------------------------------------------------------------------------

def search_param_to_katib(param: SearchParamSpec) -> Dict[str, Any]:
    """Convert one ``SearchParamSpec`` to a Katib parameter dict."""
    if param.type == "categorical":
        return {
            "name": param.name,
            "parameterType": "categorical",
            "feasibleSpace": {
                "list": [str(v) for v in (param.values or [])],
            },
        }
    if param.type == "int":
        space: Dict[str, str] = {
            "min": str(int(param.low)),  # type: ignore[arg-type]
            "max": str(int(param.high)),  # type: ignore[arg-type]
        }
        if param.step is not None:
            space["step"] = str(int(param.step))
        return {
            "name": param.name,
            "parameterType": "int",
            "feasibleSpace": space,
        }
    if param.type in {"float", "log_float"}:
        space = {
            "min": str(float(param.low)),  # type: ignore[arg-type]
            "max": str(float(param.high)),  # type: ignore[arg-type]
        }
        if param.step is not None and param.type == "float":
            space["step"] = str(float(param.step))
        return {
            "name": param.name,
            "parameterType": "double",
            "feasibleSpace": space,
        }
    raise ValueError(f"Unknown search-param type: {param.type}")


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

_ALGORITHM_MAP = {"tpe": "tpe", "random": "random", "grid": "grid"}


def _trial_parameters_json(search_space: List[SearchParamSpec]) -> str:
    """Return a Katib placeholder JSON object passed to the trial command."""
    return json.dumps(
        {
            param.name: f"${{trialParameters.{param.name}}}"
            for param in search_space
        },
        separators=(",", ":"),
    )


def build_katib_experiment(
    spec: TuneSpec,
    search_space: List[SearchParamSpec],
    trial_image: str,
    trial_command: List[str],
    trial_env: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Build a complete Katib Experiment CRD manifest.

    Parameters
    ----------
    spec:
        The validated ``TuneSpec``.
    search_space:
        Resolved search space (from engine or plugin).
    trial_image:
        Container image used for each trial Job.
    trial_command:
        Command + args for the trial container.
    trial_env:
        Extra environment variables for the trial container.  JSON
        payloads should be passed here rather than on the command line
        because the Katib metrics-collector webhook wraps the container
        command with ``sh -c``, which destroys un-quoted JSON.
    """
    env: List[Dict[str, Any]] = [
        {
            "name": "KFP_WORKFLOW_TUNE_TRIAL_NAME",
            "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
        },
    ]
    for key, value in (trial_env or {}).items():
        env.append({"name": key, "value": value})

    container: Dict[str, Any] = {
        "name": "training-container",
        "image": trial_image,
        "imagePullPolicy": spec.runtime.image_pull_policy,
        "command": list(trial_command),
        "resources": {
            "requests": {
                "cpu": spec.runtime.resources.cpu_request,
                "memory": spec.runtime.resources.memory_request,
            },
            "limits": {
                "cpu": spec.runtime.resources.cpu_limit,
                "memory": spec.runtime.resources.memory_limit,
            },
        },
        "volumeMounts": [
            {
                "name": "workflow-data",
                "mountPath": spec.storage.data_mount_path,
                "readOnly": True,
            },
            {
                "name": "workflow-models",
                "mountPath": spec.storage.model_mount_path,
            },
            {
                "name": "workflow-results",
                "mountPath": spec.storage.results_mount_path,
            },
        ],
        "env": env,
    }
    if spec.runtime.use_gpu:
        container["resources"]["requests"]["nvidia.com/gpu"] = (
            spec.runtime.resources.gpu_request
        )
        container["resources"]["limits"]["nvidia.com/gpu"] = (
            spec.runtime.resources.gpu_limit
        )

    return {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {
            "name": spec.metadata.name,
            "namespace": spec.runtime.namespace,
            "labels": {
                "app.kubernetes.io/managed-by": "kfp-workflow",
                "kfp-workflow/type": "tune",
            },
            "annotations": {
                "sidecar.istio.io/inject": "false",
                "kfp-workflow/spec-json": spec.model_dump_json(),
            },
        },
        "spec": {
            "resumePolicy": "Never",
            "objective": {
                "type": "minimize",
                "objectiveMetricName": "objective",
            },
            "algorithm": {
                "algorithmName": _ALGORITHM_MAP.get(
                    spec.hpo.algorithm, "random"
                ),
            },
            "maxTrialCount": spec.hpo.max_trials,
            "maxFailedTrialCount": spec.hpo.max_failed_trials,
            "parallelTrialCount": spec.hpo.parallel_trials,
            "parameters": [
                search_param_to_katib(p) for p in search_space
            ],
            "metricsCollectorSpec": {"collector": {"kind": "StdOut"}},
            "trialTemplate": {
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {
                        "name": p.name,
                        "description": p.name,
                        "reference": p.name,
                    }
                    for p in search_space
                ],
                "successCondition": 'status.conditions.#(type=="Complete")#',
                "failureCondition": 'status.conditions.#(type=="Failed")#',
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "backoffLimit": 0,
                        "template": {
                            "metadata": {
                                "annotations": {"sidecar.istio.io/inject": "false"}
                            },
                            "spec": {
                                "serviceAccountName": spec.runtime.service_account,
                                "containers": [container],
                                "volumes": [
                                    {
                                        "name": "workflow-data",
                                        "persistentVolumeClaim": {
                                            "claimName": spec.storage.data_pvc,
                                        },
                                    },
                                    {
                                        "name": "workflow-models",
                                        "persistentVolumeClaim": {
                                            "claimName": spec.storage.model_pvc,
                                        },
                                    },
                                    {
                                        "name": "workflow-results",
                                        "persistentVolumeClaim": {
                                            "claimName": spec.storage.results_pvc,
                                        },
                                    },
                                ],
                                "restartPolicy": "Never",
                            }
                        }
                    },
                },
            },
        },
    }
