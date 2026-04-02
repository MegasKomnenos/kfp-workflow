"""Tests that CLI command groups and subcommands exist."""

from typer.testing import CliRunner

from kfp_workflow.cli.main import app

runner = CliRunner()


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_pipeline_compile_help():
    result = runner.invoke(app, ["pipeline", "compile", "--help"])
    assert result.exit_code == 0
    assert "--spec" in result.output


def test_pipeline_submit_help():
    result = runner.invoke(app, ["pipeline", "submit", "--help"])
    assert result.exit_code == 0
    assert "--spec" in result.output


def test_benchmark_compile_help():
    result = runner.invoke(app, ["benchmark", "compile", "--help"])
    assert result.exit_code == 0
    assert "--spec" in result.output


def test_benchmark_submit_help():
    result = runner.invoke(app, ["benchmark", "submit", "--help"])
    assert result.exit_code == 0
    assert "--spec" in result.output


def test_benchmark_list_help():
    result = runner.invoke(app, ["benchmark", "list", "--help"])
    assert result.exit_code == 0
    assert "--namespace" in result.output


def test_benchmark_get_help():
    result = runner.invoke(app, ["benchmark", "get", "--help"])
    assert result.exit_code == 0
    assert "run-id" in result.output.lower() or "RUN_ID" in result.output


def test_benchmark_download_help():
    result = runner.invoke(app, ["benchmark", "download", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output


def test_serve_create_help():
    result = runner.invoke(app, ["serve", "create", "--help"])
    assert result.exit_code == 0
    assert "--spec" in result.output


def test_serve_delete_help():
    result = runner.invoke(app, ["serve", "delete", "--help"])
    assert result.exit_code == 0
    assert "NAME" in result.output


def test_registry_model_register_help():
    result = runner.invoke(app, ["registry", "model", "register", "--help"])
    assert result.exit_code == 0
    assert "--name" in result.output


def test_registry_dataset_register_help():
    result = runner.invoke(app, ["registry", "dataset", "register", "--help"])
    assert result.exit_code == 0
    assert "--pvc-name" in result.output


def test_spec_validate_help():
    result = runner.invoke(app, ["spec", "validate", "--help"])
    assert result.exit_code == 0
    assert "--spec" in result.output


def test_cluster_bootstrap_help():
    result = runner.invoke(app, ["cluster", "bootstrap", "--help"])
    assert result.exit_code == 0
    assert "--spec" in result.output


# ---------------------------------------------------------------------------
# New CLI commands — help tests
# ---------------------------------------------------------------------------

def test_json_flag_in_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--json" in result.output


def test_pipeline_submit_has_user_option():
    result = runner.invoke(app, ["pipeline", "submit", "--help"])
    assert result.exit_code == 0
    assert "--user" in result.output


def test_pipeline_get_help():
    result = runner.invoke(app, ["pipeline", "get", "--help"])
    assert result.exit_code == 0
    assert "run-id" in result.output.lower() or "RUN_ID" in result.output


def test_pipeline_list_help():
    result = runner.invoke(app, ["pipeline", "list", "--help"])
    assert result.exit_code == 0
    assert "--namespace" in result.output


def test_pipeline_wait_help():
    result = runner.invoke(app, ["pipeline", "wait", "--help"])
    assert result.exit_code == 0
    assert "--timeout" in result.output


def test_pipeline_terminate_help():
    result = runner.invoke(app, ["pipeline", "terminate", "--help"])
    assert result.exit_code == 0
    assert "run-id" in result.output.lower() or "RUN_ID" in result.output


def test_pipeline_logs_help():
    result = runner.invoke(app, ["pipeline", "logs", "--help"])
    assert result.exit_code == 0
    assert "--step" in result.output


def test_pipeline_list_experiments_help():
    result = runner.invoke(app, ["pipeline", "list-experiments", "--help"])
    assert result.exit_code == 0
    assert "--namespace" in result.output


def test_tune_list_help():
    result = runner.invoke(app, ["tune", "list", "--help"])
    assert result.exit_code == 0
    assert "--namespace" in result.output


def test_tune_get_help():
    result = runner.invoke(app, ["tune", "get", "--help"])
    assert result.exit_code == 0
    assert "experiment-id" in result.output.lower() or "EXPERIMENT_ID" in result.output


def test_tune_download_help():
    result = runner.invoke(app, ["tune", "download", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output


def test_tune_submit_help():
    result = runner.invoke(app, ["tune", "submit", "--help"])
    assert result.exit_code == 0
    assert "--spec" in result.output


def test_serve_list_help():
    result = runner.invoke(app, ["serve", "list", "--help"])
    assert result.exit_code == 0
    assert "--namespace" in result.output


def test_serve_get_help():
    result = runner.invoke(app, ["serve", "get", "--help"])
    assert result.exit_code == 0
    assert "NAME" in result.output
