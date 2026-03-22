"""Mocked functional tests for new CLI run/serve/experiment commands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# pipeline run get
# ---------------------------------------------------------------------------

@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_get(mock_conn):
    mock_client = MagicMock()
    mock_client.get_run.return_value = _mock_run()
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = runner.invoke(app, ["pipeline", "run", "get", "abc-12345-def"])
    assert result.exit_code == 0
    assert "abc-12345-def" in result.output
    assert "SUCCEEDED" in result.output
    mock_client.get_run.assert_called_once_with(run_id="abc-12345-def")


@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_get_json(mock_conn):
    mock_client = MagicMock()
    mock_client.get_run.return_value = _mock_run()
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = runner.invoke(app, ["--json", "pipeline", "run", "get", "abc-12345-def"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["run_id"] == "abc-12345-def"
    assert data["state"] == "SUCCEEDED"


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
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = runner.invoke(app, ["pipeline", "run", "terminate", "abc-12345-def"])
    assert result.exit_code == 0
    assert "terminated" in result.output
    mock_client.terminate_run.assert_called_once_with("abc-12345-def")


# ---------------------------------------------------------------------------
# pipeline run wait
# ---------------------------------------------------------------------------

@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_wait_success(mock_conn):
    mock_client = MagicMock()
    mock_client.wait_for_run_completion.return_value = _mock_run(state_value="SUCCEEDED")
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = runner.invoke(app, ["pipeline", "run", "wait", "abc-12345-def"])
    assert result.exit_code == 0
    assert "SUCCEEDED" in result.output


@patch("kfp_workflow.pipeline.connection.kfp_connection")
def test_run_wait_failure(mock_conn):
    mock_client = MagicMock()
    mock_client.wait_for_run_completion.return_value = _mock_run(state_value="FAILED")
    mock_conn.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_conn.return_value.__exit__ = MagicMock(return_value=False)

    result = runner.invoke(app, ["pipeline", "run", "wait", "abc-12345-def"])
    assert result.exit_code == 1


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


@patch("kfp_workflow.serving.kserve.get_inference_service")
def test_serve_get(mock_get):
    mock_get.return_value = {
        "metadata": {"name": "my-isvc", "creationTimestamp": "2026-03-22"},
        "status": {
            "url": "http://my-isvc.example.com",
            "conditions": [
                {"type": "Ready", "status": "True"},
                {"type": "PredictorReady", "status": "True"},
            ],
        },
    }

    result = runner.invoke(app, ["serve", "get", "--name", "my-isvc"])
    assert result.exit_code == 0
    assert "my-isvc" in result.output
    assert "True" in result.output


@patch("kfp_workflow.serving.kserve.get_inference_service")
def test_serve_get_json(mock_get):
    mock_get.return_value = {
        "metadata": {"name": "my-isvc", "creationTimestamp": "2026-03-22"},
        "status": {
            "url": "http://my-isvc.example.com",
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }

    result = runner.invoke(app, ["--json", "serve", "get", "--name", "my-isvc"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["name"] == "my-isvc"
    assert data["ready"] == "True"
