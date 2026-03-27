from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from ..cmapss.search_space import katib_parameter_specs, resolve_search_space
from ..specs import ExperimentSpec, execution_spec


GROUP = "kubeflow.org"
VERSION = "v1beta1"
PLURAL = "experiments"


def load_kube_config() -> None:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def katib_algorithm_name(name: str) -> str:
    mapping = {"random": "random", "grid": "grid", "tpe": "tpe"}
    return mapping[name]


def _trial_parameters(search_space) -> List[Dict[str, str]]:
    out = []
    for param in search_space:
        out.append({"name": param.name, "reference": param.name})
    return out


def build_experiment_manifest(spec: ExperimentSpec, dataset: str) -> Dict[str, Any]:
    spec = execution_spec(spec, kubeflow=True)
    search_space = resolve_search_space(spec.hpo)
    metric_names = ["rmse", "score", "mae"]
    experiment_name = f"{spec.metadata.name.lower()}-{dataset.lower()}-{int(time.time())}"[:63]
    trial_args = [
        "python",
        "-m",
        "softs_new.cli.main",
        "train",
        "katib-trial",
        "--spec-json",
        json.dumps(spec.model_dump(mode="json")),
        "--dataset",
        dataset,
        "--trial-params-json",
        "{"
        + ",".join(f'\\"{param.name}\\":\\"${{trialParameters.{param.name}}}\\"' for param in search_space)
        + "}",
    ]
    container: Dict[str, Any] = {
        "name": "training-container",
        "image": spec.runtime.image,
        "imagePullPolicy": spec.runtime.image_pull_policy,
        "command": trial_args,
        "resources": {
            "requests": {
                "cpu": spec.runtime.resources.cpu_request,
                "memory": spec.runtime.resources.memory_request,
                "nvidia.com/gpu": spec.runtime.resources.gpu_request,
            },
            "limits": {
                "cpu": spec.runtime.resources.cpu_limit,
                "memory": spec.runtime.resources.memory_limit,
                "nvidia.com/gpu": spec.runtime.resources.gpu_limit,
            },
        },
    }
    pod_spec: Dict[str, Any] = {
        "serviceAccountName": spec.runtime.service_account,
        "restartPolicy": "Never",
        "containers": [container],
    }
    if spec.storage.mode == "pvc":
        container["volumeMounts"] = [
            {"name": "cmapss-data", "mountPath": spec.storage.data_mount_path, "readOnly": True},
            {"name": "cmapss-results", "mountPath": spec.storage.results_mount_path},
        ]
        pod_spec["volumes"] = [
            {"name": "cmapss-data", "persistentVolumeClaim": {"claimName": spec.storage.data_pvc}},
            {"name": "cmapss-results", "persistentVolumeClaim": {"claimName": spec.storage.results_pvc}},
        ]

    return {
        "apiVersion": f"{GROUP}/{VERSION}",
        "kind": "Experiment",
        "metadata": {
            "name": experiment_name,
            "namespace": spec.runtime.namespace,
            "annotations": {"sidecar.istio.io/inject": "false"},
        },
        "spec": {
            "maxTrialCount": spec.hpo.max_trial_count,
            "parallelTrialCount": spec.hpo.parallel_trial_count,
            "maxFailedTrialCount": spec.hpo.max_failed_trial_count,
            "objective": {
                "type": "minimize",
                "objectiveMetricName": "objective",
                "additionalMetricNames": metric_names,
            },
            "algorithm": {"algorithmName": katib_algorithm_name(spec.hpo.algorithm)},
            "parameters": katib_parameter_specs(search_space),
            "metricsCollectorSpec": {"collector": {"kind": "StdOut"}},
            "trialTemplate": {
                "primaryContainerName": "training-container",
                "trialParameters": _trial_parameters(search_space),
                "successCondition": 'status.conditions.#(type=="Complete")#',
                "failureCondition": 'status.conditions.#(type=="Failed")#',
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "backoffLimit": 0,
                        "template": {
                            "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
                            "spec": pod_spec,
                        },
                    },
                },
            },
        },
    }


def dump_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def create_experiment(manifest: Dict[str, Any]) -> Dict[str, Any]:
    load_kube_config()
    api = client.CustomObjectsApi()
    return api.create_namespaced_custom_object(
        group=GROUP,
        version=VERSION,
        namespace=manifest["metadata"]["namespace"],
        plural=PLURAL,
        body=manifest,
    )


def get_experiment(namespace: str, name: str) -> Dict[str, Any]:
    load_kube_config()
    api = client.CustomObjectsApi()
    return api.get_namespaced_custom_object(group=GROUP, version=VERSION, namespace=namespace, plural=PLURAL, name=name)


def delete_experiment(namespace: str, name: str) -> None:
    load_kube_config()
    api = client.CustomObjectsApi()
    try:
        api.delete_namespaced_custom_object(group=GROUP, version=VERSION, namespace=namespace, plural=PLURAL, name=name)
    except ApiException as exc:
        if exc.status != 404:
            raise


def experiment_finished(payload: Dict[str, Any]) -> tuple[bool, bool]:
    conditions = payload.get("status", {}).get("conditions", [])
    for condition in conditions:
        if condition.get("type") in {"Succeeded", "Completed"} and condition.get("status") in {"True", True}:
            return True, True
        if condition.get("type") in {"Failed"} and condition.get("status") in {"True", True}:
            return True, False
    return False, False


def extract_best_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    status = payload.get("status", {})
    current = status.get("currentOptimalTrial") or status.get("optimalTrial") or {}
    assignments = current.get("parameterAssignments") or current.get("parameter_assignments") or []
    result: Dict[str, Any] = {}
    for item in assignments:
        name = item.get("name")
        value = item.get("value")
        if name is None:
            continue
        result[name] = _coerce_value(value)
    if not result:
        raise RuntimeError(f"unable to extract best trial parameters from Katib status: {json.dumps(status, indent=2)}")
    return result


def _coerce_value(value: Any) -> Any:
    if isinstance(value, (int, float, bool)):
        return value
    if value in {"True", "False"}:
        return value == "True"
    try:
        if "." in str(value):
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        return value


def wait_for_best_params(namespace: str, name: str, poll_seconds: int = 15, timeout_seconds: int = 60 * 60 * 24) -> Dict[str, Any]:
    started = time.time()
    while time.time() - started < timeout_seconds:
        payload = get_experiment(namespace, name)
        finished, succeeded = experiment_finished(payload)
        if finished:
            if not succeeded:
                raise RuntimeError(f"Katib experiment failed: {name}")
            return extract_best_params(payload)
        time.sleep(poll_seconds)
    raise TimeoutError(f"timed out waiting for Katib experiment {name}")


def launch_and_wait(spec: ExperimentSpec, dataset: str) -> Dict[str, Any]:
    manifest = build_experiment_manifest(spec, dataset)
    create_experiment(manifest)
    return wait_for_best_params(spec.runtime.namespace, manifest["metadata"]["name"])
