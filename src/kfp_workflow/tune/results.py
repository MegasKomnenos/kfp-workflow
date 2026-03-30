"""Tune result artifact helpers."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from kfp_workflow.utils import dump_json

_RESULTS_ROOT = "tune-results"


def experiment_result_dir(spec: Dict[str, Any], experiment_name: str) -> Path:
    """Return the canonical result directory for one tune experiment."""
    return (
        Path(spec["storage"]["results_mount_path"])
        / _RESULTS_ROOT
        / spec["metadata"]["name"]
        / experiment_name
    )


def experiment_results_path(spec: Dict[str, Any], experiment_name: str) -> Path:
    """Return the canonical aggregated results.json path."""
    return experiment_result_dir(spec, experiment_name) / "results.json"


def trial_results_dir(spec: Dict[str, Any], experiment_name: str) -> Path:
    """Return the directory containing per-trial result payloads."""
    return experiment_result_dir(spec, experiment_name) / "trials"


def trial_results_path(spec: Dict[str, Any], experiment_name: str, trial_name: str) -> Path:
    """Return the canonical per-trial payload path."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", trial_name).strip("-") or "trial"
    return trial_results_dir(spec, experiment_name) / f"{safe_name}.json"


def trial_number_from_name(trial_name: str) -> Optional[int]:
    """Best-effort parse of a trial number from a Katib trial or pod name."""
    match = re.search(r"(\d+)(?!.*\d)", trial_name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def persist_trial_result(
    *,
    spec: Dict[str, Any],
    experiment_name: str,
    namespace: str,
    trial_name: str,
    params: Dict[str, Any],
    status: str,
    objective_value: Optional[float] = None,
    error: Optional[str] = None,
) -> Tuple[Dict[str, Any], Path]:
    """Persist one per-trial tune result payload to the mounted PVC path."""
    path = trial_results_path(spec, experiment_name, trial_name)
    payload: Dict[str, Any] = {
        "tune_name": spec["metadata"]["name"],
        "experiment_name": experiment_name,
        "namespace": namespace,
        "trial_name": trial_name,
        "trial_number": trial_number_from_name(trial_name),
        "status": status,
        "params": params,
        "objective_value": objective_value,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results_path": str(path),
    }
    if error:
        payload["error"] = error
    dump_json(payload, path)
    return payload, path


def aggregate_experiment_results(
    *,
    spec: Dict[str, Any],
    experiment_name: str,
    namespace: str,
    experiment_status: str,
    created_at: str,
    completed_at: str,
    trial_payloads: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the aggregated experiment-level result payload."""
    trials: List[Dict[str, Any]] = list(trial_payloads)
    best_trial: Optional[Dict[str, Any]] = None
    for trial in trials:
        if trial.get("status") != "completed":
            continue
        if trial.get("objective_value") is None:
            continue
        if best_trial is None or float(trial["objective_value"]) < float(best_trial["objective_value"]):
            best_trial = trial

    n_completed = sum(1 for trial in trials if trial.get("status") == "completed")
    n_pruned = sum(1 for trial in trials if trial.get("status") == "pruned")
    n_failed = sum(1 for trial in trials if trial.get("status") == "failed")
    results_path = experiment_results_path(spec, experiment_name)

    return {
        "tune_name": spec["metadata"]["name"],
        "experiment_name": experiment_name,
        "namespace": namespace,
        "status": experiment_status,
        "objective_metric_name": "objective",
        "objective_type": "minimize",
        "best_value": best_trial.get("objective_value") if best_trial else None,
        "best_trial_name": best_trial.get("trial_name") if best_trial else None,
        "best_trial_number": best_trial.get("trial_number") if best_trial else None,
        "best_params": best_trial.get("params", {}) if best_trial else {},
        "n_trials": len(trials),
        "n_completed": n_completed,
        "n_pruned": n_pruned,
        "n_failed": n_failed,
        "trials": trials,
        "created_at": created_at,
        "completed_at": completed_at,
        "results_path": str(results_path),
        "spec": spec,
    }
