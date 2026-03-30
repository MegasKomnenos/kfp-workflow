"""Tests for Katib manifest generation and shared trial execution."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from kfp_workflow.cli.main import app
from kfp_workflow.specs import SearchParamSpec, TuneSpec
from kfp_workflow.tune.katib import build_katib_experiment


runner = CliRunner()


def _sample_tune_spec(**runtime_overrides) -> TuneSpec:
    runtime = {
        "namespace": "kubeflow-user-example-com",
        "image": "kfp-workflow:latest",
        "image_pull_policy": "IfNotPresent",
        "service_account": "default-editor",
        "use_gpu": False,
        "resources": {
            "cpu_request": "8",
            "cpu_limit": "8",
            "memory_request": "16Gi",
            "memory_limit": "16Gi",
            "gpu_request": "1",
            "gpu_limit": "1",
        },
    }
    runtime.update(runtime_overrides)
    return TuneSpec.model_validate(
        {
            "metadata": {"name": "mambasl-cmapss-hpo", "description": "test"},
            "runtime": runtime,
            "storage": {
                "data_pvc": "dataset-store",
                "model_pvc": "model-store",
                "data_mount_path": "/mnt/data",
                "model_mount_path": "/mnt/models",
            },
            "model": {"name": "mambasl-cmapss", "config": {}},
            "dataset": {
                "name": "cmapss",
                "config": {"fd": [{"fd_name": "FD001"}]},
            },
            "train": {"selection_metric": "rmse", "score_weight": 0.01},
            "hpo": {
                "algorithm": "tpe",
                "max_trials": 80,
                "max_failed_trials": 3,
                "parallel_trials": 4,
            },
        }
    )


def test_build_katib_experiment_references_trial_parameters():
    spec = _sample_tune_spec()
    search_space = [
        SearchParamSpec(name="d_model", type="categorical", values=[32, 64]),
        SearchParamSpec(name="lr", type="log_float", low=1e-4, high=1e-2),
    ]

    manifest = build_katib_experiment(
        spec,
        search_space,
        trial_image=spec.runtime.image,
        trial_command=[
            "python",
            "-m",
            "kfp_workflow.cli.main",
            "tune",
            "trial",
            "--spec-json",
            spec.model_dump_json(),
            "--data-mount-path",
            spec.storage.data_mount_path,
        ],
    )

    container = manifest["spec"]["trialTemplate"]["trialSpec"]["spec"]["template"]["spec"]["containers"][0]
    command = container["command"]
    params_json = command[command.index("--trial-params-json") + 1]

    assert manifest["spec"]["metricsCollectorSpec"] == {"collector": {"kind": "StdOut"}}
    assert manifest["spec"]["trialTemplate"]["successCondition"] == 'status.conditions.#(type=="Complete")#'
    assert manifest["metadata"]["annotations"]["sidecar.istio.io/inject"] == "false"
    assert json.loads(params_json) == {
        "d_model": "${trialParameters.d_model}",
        "lr": "${trialParameters.lr}",
    }
    assert container["volumeMounts"][0]["mountPath"] == "/mnt/data"
    assert "nvidia.com/gpu" not in container["resources"]["requests"]


def test_tune_trial_coerces_katib_params_and_emits_objective(monkeypatch):
    captured: dict[str, object] = {}

    class DummyPlugin:
        def hpo_base_config(self, spec):
            captured["base_spec_name"] = spec["metadata"]["name"]
            return {"base_only": 3, "toggle": False}

        def hpo_objective(self, spec, params, data_mount_path):
            captured["objective_spec_name"] = spec["metadata"]["name"]
            captured["params"] = params
            captured["data_mount_path"] = data_mount_path
            assert isinstance(params["width"], int)
            assert isinstance(params["enabled"], bool)
            assert isinstance(params["dropout"], float)
            return params["width"] + params["dropout"] + params["base_only"]

    monkeypatch.setattr("kfp_workflow.cli.main._validate_plugin_config_or_exit", lambda spec_dict: None)
    monkeypatch.setattr("kfp_workflow.plugins.get_plugin", lambda model_name: DummyPlugin())

    spec = _sample_tune_spec()
    result = runner.invoke(
        app,
        [
            "tune",
            "trial",
            "--spec-json",
            spec.model_dump_json(),
            "--trial-params-json",
            json.dumps({"width": "7", "enabled": "True", "dropout": "0.25"}),
            "--data-mount-path",
            "/mnt/data",
        ],
    )

    assert result.exit_code == 0
    assert "objective=10.25" in result.output
    assert captured["base_spec_name"] == "mambasl-cmapss-hpo"
    assert captured["objective_spec_name"] == "mambasl-cmapss-hpo"
    assert captured["params"] == {
        "base_only": 3,
        "toggle": False,
        "width": 7,
        "enabled": True,
        "dropout": 0.25,
    }
    assert captured["data_mount_path"] == "/mnt/data"
