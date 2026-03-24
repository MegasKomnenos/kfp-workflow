"""Generate Katib Experiment CRD manifests from project-owned types.

This module is a pure function of ``TuneSpec`` and ``SearchParamSpec`` — it
has zero plugin imports.
"""

from __future__ import annotations

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


def build_katib_experiment(
    spec: TuneSpec,
    search_space: List[SearchParamSpec],
    trial_image: str,
    trial_command: List[str],
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
        Command + args for the trial container.  Katib will append
        ``--<name>=<value>`` flags for each search parameter.
    """
    return {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {
            "name": spec.metadata.name,
            "namespace": spec.runtime.namespace,
        },
        "spec": {
            "objective": {
                "type": "minimize",
                "goal": 0.0,
                "objectiveMetricName": "objective",
                "additionalMetricNames": ["rmse", "score", "mae"],
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
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "training-container",
                                        "image": trial_image,
                                        "command": trial_command,
                                    }
                                ],
                                "restartPolicy": "Never",
                            }
                        }
                    },
                },
            },
        },
    }
