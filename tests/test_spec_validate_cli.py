"""CLI regression tests for spec validation behavior."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from kfp_workflow.cli.main import app


runner = CliRunner()


def test_spec_validate_json_output_is_machine_safe_for_tune():
    result = runner.invoke(
        app,
        [
            "--json",
            "spec",
            "validate",
            "--spec",
            "configs/tuning/mambasl_cmapss_tune.yaml",
            "--type",
            "tune",
        ],
    )

    assert result.exit_code == 0
    assert "validated successfully" not in result.output
    payload = json.loads(result.output)
    assert payload["metadata"]["name"] == "mambasl-cmapss-hpo"
    assert payload["hpo"]["algorithm"] == "tpe"


def test_spec_validate_json_output_is_machine_safe_for_benchmark():
    result = runner.invoke(
        app,
        [
            "--json",
            "spec",
            "validate",
            "--spec",
            "configs/benchmarks/sample_benchmark.yaml",
            "--type",
            "benchmark",
        ],
    )

    assert result.exit_code == 0
    assert "validated successfully" not in result.output
    payload = json.loads(result.output)
    assert payload["metadata"]["name"] == "sample-benchmark"
    assert payload["scenario"]["dataset"]["kind"] == "cmapss-timeseries"
    assert payload["_spec_source"].endswith("configs/benchmarks/sample_benchmark.yaml")


def test_spec_validate_serving_applies_set_overrides(tmp_path):
    spec_yaml = tmp_path / "serve.yaml"
    spec_yaml.write_text("""\
metadata:
  name: test-serving
namespace: test-ns
model_name: mambasl-cmapss
model_subpath: mambasl-cmapss/v1
runtime: custom
predictor_image: kfp-workflow:latest
replicas: 1
serving_model_config:
  d_model: 64
""")

    result = runner.invoke(
        app,
        [
            "spec",
            "validate",
            "--spec",
            str(spec_yaml),
            "--type",
            "serving",
            "--set",
            "metadata.name=serving-overridden",
            "--set",
            "replicas=3",
            "--set",
            "serving_model_config.d_model=128",
        ],
    )

    assert result.exit_code == 0
    assert "validated successfully" in result.output
    payload = json.loads(result.output.split("\n", 1)[1])
    assert payload["metadata"]["name"] == "serving-overridden"
    assert payload["replicas"] == 3
    assert payload["serving_model_config"]["d_model"] == 128


def test_spec_validate_serving_rejects_malformed_set(tmp_path):
    spec_yaml = tmp_path / "serve.yaml"
    spec_yaml.write_text("""\
metadata:
  name: test-serving
namespace: test-ns
model_name: mambasl-cmapss
model_subpath: mambasl-cmapss/v1
runtime: custom
predictor_image: kfp-workflow:latest
""")

    result = runner.invoke(
        app,
        [
            "spec",
            "validate",
            "--spec",
            str(spec_yaml),
            "--type",
            "serving",
            "--set",
            "malformed_override",
        ],
    )

    assert result.exit_code != 0
    assert "Malformed override" in result.output


def test_spec_validate_rejects_unknown_type():
    result = runner.invoke(
        app,
        [
            "spec",
            "validate",
            "--spec",
            "configs/pipelines/sample_train.yaml",
            "--type",
            "unknown",
        ],
    )

    assert result.exit_code == 1
    assert "Unknown spec type" in result.output
