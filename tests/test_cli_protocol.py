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


def test_serve_create_help():
    result = runner.invoke(app, ["serve", "create", "--help"])
    assert result.exit_code == 0
    assert "--spec" in result.output


def test_serve_delete_help():
    result = runner.invoke(app, ["serve", "delete", "--help"])
    assert result.exit_code == 0
    assert "--name" in result.output


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
