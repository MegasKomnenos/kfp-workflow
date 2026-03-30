"""Tests for tune history helpers and CLI commands."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from kfp_workflow.cli.main import app
from kfp_workflow.tune.history import (
    extract_tune_spec,
    is_tune_experiment,
    resolve_results,
    summarize_result_payload,
)

runner = CliRunner()


def _tune_spec() -> dict:
    return {
        "metadata": {"name": "mambasl-cmapss-hpo"},
        "runtime": {"namespace": "kubeflow-user-example-com"},
        "storage": {
            "data_pvc": "dataset-store",
            "model_pvc": "model-store",
            "results_pvc": "tune-store",
            "results_mount_path": "/mnt/tune-results",
        },
        "model": {"name": "mambasl-cmapss"},
        "dataset": {"name": "cmapss"},
        "train": {"seed": 42},
        "hpo": {"algorithm": "tpe"},
    }


def _experiment() -> dict:
    spec = _tune_spec()
    return {
        "metadata": {
            "name": "mambasl-cmapss-hpo",
            "namespace": "kubeflow-user-example-com",
            "creationTimestamp": "2026-03-30T00:00:00Z",
            "labels": {
                "app.kubernetes.io/managed-by": "kfp-workflow",
                "kfp-workflow/type": "tune",
            },
            "annotations": {
                "kfp-workflow/spec-json": json.dumps(spec),
            },
        },
        "status": {
            "completionTime": "2026-03-30T01:00:00Z",
            "trialsCreated": "4",
            "trialsSucceeded": "3",
            "trialsFailed": "1",
            "conditions": [{"type": "Succeeded", "status": "True"}],
            "currentOptimalTrial": {
                "parameterAssignments": [{"name": "lr", "value": "0.001"}],
                "observation": {"metrics": [{"name": "objective", "value": "1.25"}]},
            },
        },
    }


def test_extract_tune_spec():
    experiment = _experiment()
    assert extract_tune_spec(experiment) == _tune_spec()


def test_is_tune_experiment_requires_markers():
    experiment = _experiment()
    assert is_tune_experiment(experiment) is True
    experiment["metadata"]["labels"]["kfp-workflow/type"] = "benchmark"
    assert is_tune_experiment(experiment) is False


def test_summarize_result_payload():
    summary = summarize_result_payload(
        {
            "status": "SUCCEEDED",
            "best_value": 1.2,
            "best_trial_name": "trial-1",
            "n_trials": 5,
            "n_completed": 4,
            "n_pruned": 0,
            "n_failed": 1,
        }
    )
    assert summary["best_value"] == pytest.approx(1.2)
    assert summary["n_failed"] == 1


@patch("kfp_workflow.tune.history._write_result_file")
@patch("kfp_workflow.tune.history._read_trial_payloads")
@patch("kfp_workflow.tune.history._read_result_file")
def test_resolve_results_aggregates_from_trials(mock_read, mock_trials, mock_write):
    mock_read.side_effect = FileNotFoundError("missing results")
    mock_trials.return_value = [
        {
            "trial_name": "trial-1",
            "trial_number": 1,
            "status": "completed",
            "params": {"lr": 0.001},
            "objective_value": 1.25,
        },
        {
            "trial_name": "trial-2",
            "trial_number": 2,
            "status": "failed",
            "params": {"lr": 0.002},
            "objective_value": None,
        },
    ]

    resolved = resolve_results(
        experiment=_experiment(),
        tune_spec=_tune_spec(),
        namespace="kubeflow-user-example-com",
    )

    assert resolved["payload"]["best_value"] == pytest.approx(1.25)
    assert resolved["summary"]["n_completed"] == 1
    mock_write.assert_called_once()


@patch("kfp_workflow.utils.dump_json")
@patch("kfp_workflow.tune.history.resolve_results")
@patch("kfp_workflow.tune.history.is_tune_experiment")
@patch("kfp_workflow.tune.history.get_tune_experiment")
@patch("kfp_workflow.tune.history.extract_tune_spec")
def test_tune_download_cli(mock_extract, mock_get, mock_is_tune, mock_resolve, mock_dump):
    mock_extract.return_value = _tune_spec()
    mock_get.return_value = _experiment()
    mock_is_tune.return_value = True
    mock_resolve.return_value = {
        "results_path": "/mnt/tune-results/tune-results/mambasl-cmapss-hpo/mambasl-cmapss-hpo/results.json",
        "payload": {"status": "SUCCEEDED"},
        "summary": {"status": "SUCCEEDED"},
    }

    result = runner.invoke(app, ["tune", "download", "mambasl-cmapss-hpo"])

    assert result.exit_code == 0
    assert "Tune results downloaded" in result.output
    mock_dump.assert_called_once()


@patch("kfp_workflow.tune.history.resolve_results")
@patch("kfp_workflow.tune.history.is_tune_experiment")
@patch("kfp_workflow.tune.history.get_tune_experiment")
@patch("kfp_workflow.tune.history.extract_tune_spec")
def test_tune_get_cli_json(mock_extract, mock_get, mock_is_tune, mock_resolve):
    mock_extract.return_value = _tune_spec()
    mock_get.return_value = _experiment()
    mock_is_tune.return_value = True
    mock_resolve.return_value = {
        "results_path": "/mnt/tune-results/tune-results/mambasl-cmapss-hpo/mambasl-cmapss-hpo/results.json",
        "payload": {"best_params": {"lr": 0.001}, "best_value": 1.25},
        "summary": {"status": "SUCCEEDED", "best_value": 1.25},
    }

    result = runner.invoke(app, ["--json", "tune", "get", "mambasl-cmapss-hpo"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["experiment_name"] == "mambasl-cmapss-hpo"
    assert data["best_value"] == pytest.approx(1.25)
    assert data["best_params"]["lr"] == pytest.approx(0.001)


@patch("kfp_workflow.tune.history.list_tune_experiments")
@patch("kfp_workflow.tune.history.is_tune_experiment")
@patch("kfp_workflow.tune.history.extract_tune_spec")
@patch("kfp_workflow.tune.history.summarize_experiment")
def test_tune_list_cli_json(mock_summary, mock_extract, mock_is_tune, mock_list):
    mock_list.return_value = [_experiment()]
    mock_is_tune.return_value = True
    mock_extract.return_value = _tune_spec()
    mock_summary.return_value = {
        "experiment_name": "mambasl-cmapss-hpo",
        "state": "SUCCEEDED",
        "created_at": "2026-03-30T00:00:00Z",
        "finished_at": "2026-03-30T01:00:00Z",
        "best_value": 1.25,
        "best_params": {"lr": 0.001},
        "n_trials": 4,
        "n_completed": 3,
        "n_pruned": 0,
        "n_failed": 1,
    }

    result = runner.invoke(app, ["--json", "tune", "list"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["experiment_name"] == "mambasl-cmapss-hpo"
    assert data[0]["tune_name"] == "mambasl-cmapss-hpo"


def test_cluster_bootstrap_tune_dry_run(tmp_path):
    spec_path = tmp_path / "tune.yaml"
    spec_path.write_text(
        """\
metadata:
  name: tune-smoke
runtime:
  namespace: test-ns
model:
  name: m
dataset:
  name: d
"""
    )

    result = runner.invoke(
        app,
        ["cluster", "bootstrap", "--spec", str(spec_path), "--type", "tune", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "tune-store" in result.output
