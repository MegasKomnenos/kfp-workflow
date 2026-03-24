from __future__ import annotations

import json
import subprocess
import time
from typing import Optional

from ..specs import ExperimentSpec


def _run(cmd: list[str], *, input_text: Optional[str] = None) -> None:
    subprocess.run(cmd, check=True, text=True, input=input_text)


def _ensure_namespace(namespace: str) -> None:
    subprocess.run(["kubectl", "create", "namespace", namespace], check=False, text=True, capture_output=True)


def pvc_manifest(name: str, storage_class: str, size: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": name},
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": size}},
            "storageClassName": storage_class,
        },
    }


def seed_job_manifest(namespace: str, job_name: str, data_pvc: str, source_dir: str) -> dict:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"annotations": {"sidecar.istio.io/inject": "false"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "seed",
                            "image": "python:3.10-slim",
                            "command": ["bash", "-lc", "set -euo pipefail && cp -av /seed-source/. /data/"],
                            "volumeMounts": [
                                {"name": "data", "mountPath": "/data"},
                                {"name": "seed-source", "mountPath": "/seed-source", "readOnly": True},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "data", "persistentVolumeClaim": {"claimName": data_pvc}},
                        {"name": "seed-source", "hostPath": {"path": source_dir, "type": "Directory"}},
                    ],
                },
            },
        },
    }


def bootstrap_storage(spec: ExperimentSpec, *, dry_run: bool = False) -> list[dict]:
    manifests = []
    if spec.storage.mode != "pvc":
        return manifests
    manifests.extend(
        [
            pvc_manifest(spec.storage.data_pvc, spec.storage.storage_class, spec.storage.data_size),
            pvc_manifest(spec.storage.results_pvc, spec.storage.storage_class, spec.storage.results_size),
        ]
    )
    if spec.storage.skip_seed or not spec.storage.seed_source_dir:
        return manifests
    manifests.append(seed_job_manifest(spec.runtime.namespace, f"seed-cmapss-{int(time.time())}", spec.storage.data_pvc, spec.storage.seed_source_dir))
    if dry_run:
        return manifests

    _ensure_namespace(spec.runtime.namespace)
    for manifest in manifests[:2]:
        _run(["kubectl", "apply", "-n", spec.runtime.namespace, "-f", "-"], input_text=json.dumps(manifest))
    if len(manifests) == 3:
        _run(["kubectl", "apply", "-f", "-"], input_text=json.dumps(manifests[2]))
        _run(["kubectl", "-n", spec.runtime.namespace, "wait", "--for=condition=complete", f"job/{manifests[2]['metadata']['name']}", "--timeout=30m"])
    return manifests
