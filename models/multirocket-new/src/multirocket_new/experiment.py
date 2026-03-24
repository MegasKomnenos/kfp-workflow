from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ExperimentConfig
from .runner import run_experiment
from .specs import ExperimentSpec, expand_ablation_cases
from .utils import dump_json


def _execution_data_dir(spec: ExperimentSpec, *, kubeflow: bool) -> str:
    if kubeflow and spec.storage.mode == "pvc":
        return spec.storage.data_mount_path
    return spec.data.data_root


def _execution_output_dir(spec: ExperimentSpec, *, kubeflow: bool) -> str:
    if kubeflow and spec.storage.mode == "pvc":
        return spec.storage.results_mount_path
    return spec.outputs.local_results_dir


def base_config_mapping(spec: ExperimentSpec, dataset: str, *, kubeflow: bool) -> Dict[str, Any]:
    return {
        "subset": dataset,
        "data_dir": _execution_data_dir(spec, kubeflow=kubeflow),
        "output_dir": _execution_output_dir(spec, kubeflow=kubeflow),
        "feature_mode": spec.train_defaults.feature_mode,
        "scaling_mode": spec.train_defaults.scaling_mode,
        "seq_len": spec.train_defaults.seq_len,
        "train_stride": spec.train_defaults.train_stride,
        "max_rul": spec.train_defaults.max_rul,
        "val_frac": spec.train_defaults.val_frac,
        "val_mode": spec.train_defaults.val_mode,
        "val_samples_per_unit": spec.train_defaults.val_samples_per_unit,
        "n_jobs": spec.train_defaults.n_jobs,
        "predict_batch_size": spec.train_defaults.predict_batch_size,
        "seed": spec.train_defaults.seed,
        "limit_train_units": spec.train_defaults.limit_train_units,
        "limit_val_units": spec.train_defaults.limit_val_units,
        "limit_test_units": spec.train_defaults.limit_test_units,
        "download_if_missing": spec.data.download_policy in {"if_missing", "always"},
        **spec.train_defaults.fixed_params,
    }


def build_experiment_config(
    spec: ExperimentSpec,
    dataset: str,
    *,
    overrides: Optional[Dict[str, Any]] = None,
    kubeflow: bool = False,
    run_name: str = "",
) -> ExperimentConfig:
    payload = base_config_mapping(spec, dataset, kubeflow=kubeflow)
    if overrides:
        payload.update(overrides)
    if run_name:
        payload["run_name"] = run_name
    return ExperimentConfig.from_mapping(payload)


def emit_katib_metrics(metrics: Dict[str, float], selection_metric: str, score_weight: float) -> None:
    objective = metrics["rmse"]
    if selection_metric == "score":
        objective = metrics["nasa_score"]
    elif selection_metric == "hybrid":
        objective = metrics["rmse"] + score_weight * metrics["nasa_score"]
    print(f"objective={objective}", flush=True)
    print(f"rmse={metrics['rmse']}", flush=True)
    print(f"score={metrics['nasa_score']}", flush=True)
    print(f"mae={metrics['mae']}", flush=True)


def run_ablation_only(
    spec: ExperimentSpec,
    dataset: str,
    best_params: Dict[str, Any],
    output_dir: Path,
    *,
    kubeflow: bool,
) -> list[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in expand_ablation_cases(spec):
        merged = dict(best_params)
        merged.update(case["overrides"])
        payload = run_experiment(
            build_experiment_config(
                spec,
                dataset,
                overrides=merged,
                kubeflow=kubeflow,
                run_name=f"{dataset.lower()}__{case['name']}",
            )
        )
        rows.append(
            {
                "condition": case["name"],
                "overrides": case["overrides"],
                "val_metrics": payload["val_metrics"],
                "test_metrics": payload["test_metrics"],
                "result_dir": payload["result_dir"],
            }
        )
    dump_json(output_dir / f"{dataset.lower()}_ablations.json", rows)
    return rows


def run_dataset_pipeline(
    spec: ExperimentSpec,
    dataset: str,
    output_dir: Path,
    *,
    explicit_params: Optional[Dict[str, Any]] = None,
    run_hpo_stage: bool = False,
    run_ablation_stage: bool = False,
    kubeflow: bool = False,
) -> Dict[str, Any]:
    best_params = dict(spec.train_defaults.fixed_params)
    best_objective = None
    if explicit_params is not None:
        best_params.update(explicit_params)
    elif run_hpo_stage and spec.hpo.enabled:
        from .kubeflow.katib import launch_and_wait

        best_params = launch_and_wait(spec, dataset)
        best_objective = "katib"

    payload = run_experiment(
        build_experiment_config(
            spec,
            dataset,
            overrides=best_params,
            kubeflow=kubeflow,
            run_name=f"{dataset.lower()}__final",
        )
    )
    result = {
        "experiment": spec.metadata.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "settings": spec.model_dump(mode="json"),
        "hpo": {
            "objective_metric": spec.train_defaults.selection_metric,
            "best_val_objective": best_objective,
            "best_params": best_params,
        },
        "final": {
            "val_metrics": payload["val_metrics"],
            "test_metrics": payload["test_metrics"],
            "val_fit_s": payload["val_fit_s"],
            "final_fit_s": payload["final_fit_s"],
            "result_dir": payload["result_dir"],
        },
    }
    if run_ablation_stage and spec.ablations.enabled:
        result["ablations"] = run_ablation_only(spec, dataset, best_params, output_dir, kubeflow=kubeflow)
    dump_json(output_dir / f"{dataset.lower()}_result.json", result)
    dump_json(output_dir / f"{dataset.lower()}_metrics.json", payload["test_metrics"])
    dump_json(output_dir / f"{dataset.lower()}_config.json", best_params)
    return result


def load_params_json(text: str) -> Dict[str, Any]:
    return json.loads(text)
