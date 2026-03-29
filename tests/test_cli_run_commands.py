"""Mocked functional tests for new CLI run/serve/experiment commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from kfp_server_api.exceptions import ApiException
from typer.testing import CliRunner

from kfp_workflow.cli.main import app

runner = CliRunner()


def _mock_run(
    run_id: str = "abc-12345-def",
    display_name: str = "test-run",
    state_value: str = "SUCCEEDED",
    error: object = None,
) -> MagicMock:
    """Build a mock V2beta1Run object."""
    run = MagicMock()
    run.run_id = run_id
    run.display_name = display_name
    run.state = MagicMock(value=state_value)
    run.created_at = "2026-03-22T00:00:00Z"
    run.finished_at = "2026-03-22T01:00:00Z"
    run.experiment_id = "exp-001"
    run.error = error
    return run


def _mock_experiment(
    experiment_id: str = "exp-001",
    display_name: str = "test-experiment",
) -> MagicMock:
    """Build a mock V2beta1Experiment object."""
    exp = MagicMock()
    exp.experiment_id = experiment_id
    exp.display_name = display_name
    exp.created_at = "2026-03-22T00:00:00Z"
    exp.last_run_created_at = "2026-03-22T01:00:00Z"
    return exp


def _not_found() -> ApiException:
    return ApiException(status=404, reason="Not Found")


# ---------------------------------------------------------------------------
# pipeline run get
# ---------------------------------------------------------------------------

@patch("kfp_workflow.cli.main._find_workflow_for_run")
@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_get(mock_conn, mock_workflow):
    mock_client = MagicMock()
    mock_client.get_run.return_value = _mock_run()
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)
    mock_workflow.return_value = None

    result = runner.invoke(app, ["pipeline", "run", "get", "abc-12345-def"])
    assert result.exit_code == 0
    assert "abc-12345-def" in result.output
    assert "SUCCEEDED" in result.output
    mock_client.get_run.assert_called_once_with(run_id="abc-12345-def")


@patch("kfp_workflow.cli.main._find_workflow_for_run")
@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_get_json(mock_conn, mock_workflow):
    mock_client = MagicMock()
    mock_client.get_run.return_value = _mock_run()
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)
    mock_workflow.return_value = {
        "metadata": {"name": "workflow-123"},
        "status": {"phase": "Succeeded", "progress": "5/5"},
    }

    result = runner.invoke(app, ["--json", "pipeline", "run", "get", "abc-12345-def"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["run_id"] == "abc-12345-def"
    assert data["state"] == "SUCCEEDED"
    assert data["workflow_name"] == "workflow-123"


@patch("kfp_workflow.cli.main._find_workflow_for_run")
@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_get_resolves_unique_short_prefix(mock_conn, mock_workflow):
    mock_client = MagicMock()
    full_run = _mock_run(run_id="abc-12345-def", display_name="test-run")
    mock_client.get_run.side_effect = [_not_found(), full_run]
    response = MagicMock()
    response.runs = [full_run]
    response.next_page_token = ""
    mock_client.list_runs.return_value = response
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)
    mock_workflow.return_value = None

    result = runner.invoke(app, ["pipeline", "run", "get", "abc"])
    assert result.exit_code == 0
    assert "abc-12345-def" in result.output
    assert mock_client.get_run.call_args_list[-1].kwargs == {"run_id": "abc-12345-def"}


# ---------------------------------------------------------------------------
# pipeline run list
# ---------------------------------------------------------------------------

@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_list(mock_conn):
    mock_client = MagicMock()
    response = MagicMock()
    response.runs = [_mock_run(), _mock_run(run_id="xyz-999", display_name="run-2")]
    mock_client.list_runs.return_value = response
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = runner.invoke(app, ["pipeline", "run", "list"])
    assert result.exit_code == 0
    assert "abc-1234" in result.output
    assert "xyz-999" in result.output


@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_list_empty(mock_conn):
    mock_client = MagicMock()
    response = MagicMock()
    response.runs = None  # KFP returns None for empty
    mock_client.list_runs.return_value = response
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = runner.invoke(app, ["pipeline", "run", "list"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# pipeline run terminate
# ---------------------------------------------------------------------------

@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_terminate(mock_conn):
    mock_client = MagicMock()
    mock_client.get_run.return_value = _mock_run()
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = runner.invoke(app, ["pipeline", "run", "terminate", "abc-12345-def"])
    assert result.exit_code == 0
    assert "terminated" in result.output
    mock_client.terminate_run.assert_called_once_with("abc-12345-def")


@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_terminate_resolves_unique_short_prefix(mock_conn):
    mock_client = MagicMock()
    full_run = _mock_run(run_id="abc-12345-def")
    mock_client.get_run.side_effect = [_not_found(), full_run]
    response = MagicMock()
    response.runs = [full_run]
    response.next_page_token = ""
    mock_client.list_runs.return_value = response
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = runner.invoke(app, ["pipeline", "run", "terminate", "abc"])
    assert result.exit_code == 0
    mock_client.terminate_run.assert_called_once_with("abc-12345-def")


# ---------------------------------------------------------------------------
# pipeline run wait
# ---------------------------------------------------------------------------

@patch("kfp_workflow.cli.main._find_workflow_for_run")
@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_wait_success(mock_conn, mock_workflow):
    mock_client = MagicMock()
    mock_client.get_run.return_value = _mock_run()
    mock_client.wait_for_run_completion.return_value = _mock_run(state_value="SUCCEEDED")
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)
    mock_workflow.return_value = {
        "metadata": {"name": "workflow-123"},
        "status": {"phase": "Succeeded", "progress": "5/5"},
    }

    result = runner.invoke(app, ["pipeline", "run", "wait", "abc-12345-def"])
    assert result.exit_code == 0
    assert "SUCCEEDED" in result.output
    assert "workflow-123" in result.output


@patch("kfp_workflow.cli.main._find_workflow_for_run")
@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_wait_failure(mock_conn, mock_workflow):
    mock_client = MagicMock()
    mock_client.get_run.return_value = _mock_run()
    mock_client.wait_for_run_completion.return_value = _mock_run(state_value="FAILED")
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)
    mock_workflow.return_value = None

    result = runner.invoke(app, ["pipeline", "run", "wait", "abc-12345-def"])
    assert result.exit_code == 1


@patch("kfp_workflow.cli.main._find_workflow_for_run")
@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_wait_timeout_shows_workflow_diagnostics(mock_conn, mock_workflow):
    mock_client = MagicMock()
    mock_client.get_run.return_value = _mock_run()
    mock_client.wait_for_run_completion.side_effect = TimeoutError("timed out")
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)
    mock_workflow.return_value = {
        "metadata": {"name": "workflow-123"},
        "status": {
            "phase": "Running",
            "progress": "4/5",
            "message": "save-model pending",
            "nodes": {
                "n1": {"displayName": "save-model", "phase": "Pending"},
            },
        },
    }

    result = runner.invoke(app, ["pipeline", "run", "wait", "abc-12345-def", "--timeout", "1"])
    assert result.exit_code == 1
    assert "workflow-123" in result.output
    assert "save-model" in result.output


@patch("kfp_workflow.cli.main._find_workflow_for_run")
@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_wait_resolves_unique_short_prefix(mock_conn, mock_workflow):
    mock_client = MagicMock()
    full_run = _mock_run(run_id="abc-12345-def", state_value="SUCCEEDED")
    mock_client.get_run.side_effect = [_not_found(), full_run]
    response = MagicMock()
    response.runs = [full_run]
    response.next_page_token = ""
    mock_client.list_runs.return_value = response
    mock_client.wait_for_run_completion.return_value = full_run
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)
    mock_workflow.return_value = None

    result = runner.invoke(app, ["pipeline", "run", "wait", "abc"])
    assert result.exit_code == 0
    mock_client.wait_for_run_completion.assert_called_once_with("abc-12345-def", timeout=3600)


# ---------------------------------------------------------------------------
# pipeline experiment list
# ---------------------------------------------------------------------------

@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_experiment_list(mock_conn):
    mock_client = MagicMock()
    response = MagicMock()
    response.experiments = [_mock_experiment()]
    mock_client.list_experiments.return_value = response
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = runner.invoke(app, ["pipeline", "experiment", "list"])
    assert result.exit_code == 0
    assert "test-experiment" in result.output


@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_list_resolves_experiment_short_prefix(mock_conn):
    mock_client = MagicMock()
    response = MagicMock()
    response.experiments = [_mock_experiment(experiment_id="exp-001-full")]
    response.next_page_token = ""
    mock_client.list_experiments.return_value = response
    runs_response = MagicMock()
    runs_response.runs = [_mock_run()]
    mock_client.list_runs.return_value = runs_response
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = runner.invoke(app, ["pipeline", "run", "list", "--experiment-id", "exp-001"])
    assert result.exit_code == 0
    assert "test-run" in result.output
    assert mock_client.list_runs.call_args.kwargs["experiment_id"] == "exp-001-full"


# ---------------------------------------------------------------------------
# benchmark list / get / download
# ---------------------------------------------------------------------------

@patch("kfp_workflow.benchmark.history.extract_benchmark_spec")
@patch("kfp_workflow.benchmark.history.is_benchmark_workflow")
@patch("kfp_workflow.benchmark.history.find_workflow_for_run")
@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_benchmark_list(mock_conn, mock_find, mock_is_benchmark, mock_extract):
    mock_client = MagicMock()
    response = MagicMock()
    response.runs = [_mock_run(display_name="ignored"), _mock_run(run_id="bench-123", display_name="bench-run")]
    mock_client.list_runs.return_value = response
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    workflow = {"metadata": {"name": "workflow-123"}, "status": {"phase": "Succeeded"}}
    mock_find.side_effect = [None, workflow]
    mock_is_benchmark.side_effect = [True]
    mock_extract.return_value = {"metadata": {"name": "mambasl-cmapss-benchmark-smoke"}}

    result = runner.invoke(app, ["--json", "benchmark", "list"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["benchmark_name"] == "mambasl-cmapss-benchmark-smoke"
    assert payload[0]["run_id"] == "bench-123"


@patch("kfp_workflow.benchmark.history.resolve_results")
@patch("kfp_workflow.benchmark.history.extract_benchmark_spec")
@patch("kfp_workflow.benchmark.history.is_benchmark_workflow")
@patch("kfp_workflow.benchmark.history.find_workflow_for_run")
@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_benchmark_get(mock_conn, mock_find, mock_is_benchmark, mock_extract, mock_resolve):
    mock_client = MagicMock()
    mock_client.get_run.return_value = _mock_run(run_id="bench-123", display_name="bench-run")
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    workflow = {
        "metadata": {"name": "workflow-123"},
        "status": {"phase": "Succeeded", "progress": "4/4"},
    }
    mock_find.return_value = workflow
    mock_is_benchmark.return_value = True
    mock_extract.return_value = {"metadata": {"name": "mambasl-cmapss-benchmark-smoke"}}
    mock_resolve.return_value = {
        "results_path": "/mnt/results/bench/results.json",
        "summary": {"status": "succeeded", "request_count": 5, "delta_joules": 2.5},
        "payload": {"status": "succeeded"},
    }

    result = runner.invoke(app, ["benchmark", "get", "bench-123"])
    assert result.exit_code == 0
    assert "mambasl-cmapss-benchmark-smoke" in result.output
    assert "/mnt/results/bench/results.json" in result.output
    assert "2.5" in result.output


@patch("kfp_workflow.benchmark.history.resolve_results")
@patch("kfp_workflow.benchmark.history.extract_benchmark_spec")
@patch("kfp_workflow.benchmark.history.is_benchmark_workflow")
@patch("kfp_workflow.benchmark.history.find_workflow_for_run")
@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_benchmark_get_resolves_unique_short_prefix(mock_conn, mock_find, mock_is_benchmark, mock_extract, mock_resolve):
    mock_client = MagicMock()
    full_run = _mock_run(run_id="bench-12345-full", display_name="bench-run")
    mock_client.get_run.side_effect = [_not_found(), full_run]
    response = MagicMock()
    response.runs = [full_run]
    response.next_page_token = ""
    mock_client.list_runs.return_value = response
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    workflow = {"metadata": {"name": "workflow-123"}, "status": {"phase": "Succeeded", "progress": "4/4"}}
    mock_find.return_value = workflow
    mock_is_benchmark.return_value = True
    mock_extract.return_value = {"metadata": {"name": "mambasl-cmapss-benchmark-smoke"}}
    mock_resolve.return_value = {
        "results_path": "/mnt/results/bench/results.json",
        "summary": {"status": "succeeded", "request_count": 5},
        "payload": {"status": "succeeded"},
    }

    result = runner.invoke(app, ["benchmark", "get", "bench-123"])
    assert result.exit_code == 0
    assert "bench-12345-full" in result.output


@patch("kfp_workflow.benchmark.history.resolve_results")
@patch("kfp_workflow.benchmark.history.extract_benchmark_spec")
@patch("kfp_workflow.benchmark.history.is_benchmark_workflow")
@patch("kfp_workflow.benchmark.history.find_workflow_for_run")
@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_benchmark_download(mock_conn, mock_find, mock_is_benchmark, mock_extract, mock_resolve, tmp_path: Path):
    mock_client = MagicMock()
    mock_client.get_run.return_value = _mock_run(run_id="bench-123", display_name="bench-run")
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    workflow = {"metadata": {"name": "workflow-123"}, "status": {"phase": "Succeeded"}}
    mock_find.return_value = workflow
    mock_is_benchmark.return_value = True
    mock_extract.return_value = {"metadata": {"name": "mambasl-cmapss-benchmark-smoke"}}
    mock_resolve.return_value = {
        "results_path": "/mnt/results/bench/results.json",
        "summary": {"status": "succeeded", "request_count": 5},
        "payload": {"status": "succeeded", "scenario": {"request_count": 5}},
    }

    output = tmp_path / "downloaded.json"
    result = runner.invoke(app, ["benchmark", "download", "bench-123", "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    saved = json.loads(output.read_text("utf-8"))
    assert saved["status"] == "succeeded"
    assert "downloaded" in result.output.lower()


@patch("kfp_workflow.benchmark.history.resolve_results")
@patch("kfp_workflow.benchmark.history.extract_benchmark_spec")
@patch("kfp_workflow.benchmark.history.is_benchmark_workflow")
@patch("kfp_workflow.benchmark.history.find_workflow_for_run")
@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_benchmark_download_resolves_unique_short_prefix(mock_conn, mock_find, mock_is_benchmark, mock_extract, mock_resolve, tmp_path: Path):
    mock_client = MagicMock()
    full_run = _mock_run(run_id="bench-12345-full", display_name="bench-run")
    mock_client.get_run.side_effect = [_not_found(), full_run]
    response = MagicMock()
    response.runs = [full_run]
    response.next_page_token = ""
    mock_client.list_runs.return_value = response
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    workflow = {"metadata": {"name": "workflow-123"}, "status": {"phase": "Succeeded"}}
    mock_find.return_value = workflow
    mock_is_benchmark.return_value = True
    mock_extract.return_value = {"metadata": {"name": "mambasl-cmapss-benchmark-smoke"}}
    mock_resolve.return_value = {
        "results_path": "/mnt/results/bench/results.json",
        "summary": {"status": "succeeded"},
        "payload": {"status": "succeeded"},
    }

    result = runner.invoke(app, ["benchmark", "download", "bench-123", "--output", str(tmp_path / "out.json")])
    assert result.exit_code == 0
    assert mock_client.get_run.call_args_list[-1].kwargs == {"run_id": "bench-12345-full"}


# ---------------------------------------------------------------------------
# serve list / get
# ---------------------------------------------------------------------------

@patch("kfp_workflow.serving.kserve.list_inference_services")
def test_serve_list(mock_list):
    mock_list.return_value = [
        {
            "metadata": {"name": "my-isvc", "creationTimestamp": "2026-03-22"},
            "status": {
                "url": "http://my-isvc.example.com",
                "conditions": [{"type": "Ready", "status": "True"}],
            },
        },
    ]

    result = runner.invoke(app, ["serve", "list"])
    assert result.exit_code == 0
    assert "my-isvc" in result.output


@patch("kfp_workflow.serving.kserve.get_inference_service_diagnostics")
def test_serve_get(mock_get):
    mock_get.return_value = {
        "service": {
            "metadata": {"name": "my-isvc", "creationTimestamp": "2026-03-22"},
            "status": {
                "url": "http://my-isvc.example.com",
                "conditions": [
                    {"type": "Ready", "status": "True"},
                    {"type": "PredictorReady", "status": "True"},
                ],
            },
        },
        "ready": "True",
        "conditions": [
            {"type": "Ready", "status": "True", "reason": "", "message": ""},
            {"type": "PredictorReady", "status": "True", "reason": "", "message": ""},
        ],
        "events": [],
    }

    result = runner.invoke(app, ["serve", "get", "--name", "my-isvc"])
    assert result.exit_code == 0
    assert "my-isvc" in result.output
    assert "True" in result.output


@patch("kfp_workflow.serving.kserve.get_inference_service_diagnostics")
def test_serve_get_json(mock_get):
    mock_get.return_value = {
        "service": {
            "metadata": {"name": "my-isvc", "creationTimestamp": "2026-03-22"},
            "status": {
                "url": "http://my-isvc.example.com",
                "conditions": [{"type": "Ready", "status": "True"}],
            },
        },
        "ready": "True",
        "conditions": [{"type": "Ready", "status": "True", "reason": "", "message": ""}],
        "events": [{"type": "Warning", "reason": "InternalError", "message": "bad host"}],
    }

    result = runner.invoke(app, ["--json", "serve", "get", "--name", "my-isvc"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["name"] == "my-isvc"
    assert data["ready"] == "True"
    assert data["events"][0]["reason"] == "InternalError"


@patch("kfp_workflow.serving.kserve.wait_for_inference_service_ready")
@patch("kfp_workflow.serving.kserve.create_inference_service")
def test_serve_create_wait_failure(mock_create, mock_wait):
    mock_create.return_value = {"metadata": {"name": "sample-serving"}}
    mock_wait.return_value = {
        "ready": "Unknown",
        "conditions": [],
        "events": [{"reason": "InternalError", "message": "name too long"}],
    }

    result = runner.invoke(
        app,
        [
            "serve", "create",
            "--spec", "configs/serving/sample_serve.yaml",
            "--wait",
            "--timeout", "1",
        ],
    )
    assert result.exit_code == 1
    assert "name too long" in result.output
