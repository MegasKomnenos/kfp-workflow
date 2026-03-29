"""Tests for CLI config override system (--set flag)."""

import pytest
from typer.testing import CliRunner

from kfp_workflow.config_override import (
    apply_overrides,
    coerce_value,
    set_nested,
    validate_plugin_config,
)
from kfp_workflow.cli.main import app


# ---------------------------------------------------------------------------
# coerce_value
# ---------------------------------------------------------------------------

class TestCoerceValue:
    def test_integer(self):
        assert coerce_value("42") == 42

    def test_float(self):
        assert coerce_value("3.14") == 3.14

    def test_bool_true(self):
        assert coerce_value("true") is True

    def test_bool_false(self):
        assert coerce_value("false") is False

    def test_null(self):
        assert coerce_value("null") is None

    def test_string(self):
        assert coerce_value("hello") == "hello"

    def test_json_list(self):
        assert coerce_value("[1, 2, 3]") == [1, 2, 3]

    def test_json_dict(self):
        assert coerce_value('{"a": 1}') == {"a": 1}

    def test_string_with_spaces(self):
        assert coerce_value("hello world") == "hello world"

    def test_negative_number(self):
        assert coerce_value("-5") == -5

    def test_scientific_notation(self):
        assert coerce_value("1e-3") == 1e-3


# ---------------------------------------------------------------------------
# set_nested
# ---------------------------------------------------------------------------

class TestSetNested:
    def test_top_level(self):
        d = {"a": 1}
        set_nested(d, "b", 2)
        assert d == {"a": 1, "b": 2}

    def test_nested_existing(self):
        d = {"model": {"config": {"d_model": 64}}}
        set_nested(d, "model.config.d_model", 128)
        assert d["model"]["config"]["d_model"] == 128

    def test_creates_intermediate_dicts(self):
        d = {}
        set_nested(d, "a.b.c", "val")
        assert d == {"a": {"b": {"c": "val"}}}

    def test_overwrites_non_dict_intermediate(self):
        d = {"a": "scalar"}
        set_nested(d, "a.b", "val")
        assert d == {"a": {"b": "val"}}

    def test_preserves_siblings(self):
        d = {"model": {"config": {"d_model": 64, "d_state": 16}}}
        set_nested(d, "model.config.d_model", 128)
        assert d["model"]["config"]["d_state"] == 16

    def test_sets_list_index_path(self):
        d = {"dataset": {"config": {"fd": [{"fd_name": "FD001"}]}}}
        set_nested(d, "dataset.config.fd[0].fd_name", "FD003")
        assert d["dataset"]["config"]["fd"][0]["fd_name"] == "FD003"

    def test_creates_list_index_path(self):
        d = {}
        set_nested(d, "dataset.config.fd[0].fd_name", "FD001")
        assert d == {"dataset": {"config": {"fd": [{"fd_name": "FD001"}]}}}


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------

class TestApplyOverrides:
    def test_single_override(self):
        d = {"train": {"batch_size": 32}}
        apply_overrides(d, ["train.batch_size=128"])
        assert d["train"]["batch_size"] == 128

    def test_multiple_overrides(self):
        d = {"model": {"config": {}}, "train": {}}
        apply_overrides(d, [
            "model.config.d_model=128",
            "train.batch_size=64",
        ])
        assert d["model"]["config"]["d_model"] == 128
        assert d["train"]["batch_size"] == 64

    def test_override_wins_over_yaml(self):
        d = {"train": {"learning_rate": 0.001}}
        apply_overrides(d, ["train.learning_rate=0.0005"])
        assert d["train"]["learning_rate"] == 0.0005

    def test_malformed_no_equals(self):
        with pytest.raises(ValueError, match="Malformed"):
            apply_overrides({}, ["no_equals_sign"])

    def test_malformed_empty_key(self):
        with pytest.raises(ValueError, match="Empty key"):
            apply_overrides({}, ["=value"])

    def test_value_with_equals(self):
        d = {}
        apply_overrides(d, ["key=a=b"])
        assert d["key"] == "a=b"

    def test_boolean_override(self):
        d = {"model": {"config": {}}}
        apply_overrides(d, ["model.config.tv_dt=false"])
        assert d["model"]["config"]["tv_dt"] is False

    def test_list_override(self):
        d = {}
        apply_overrides(d, ['features=[1,2,3]'])
        assert d["features"] == [1, 2, 3]

    def test_list_index_override(self):
        d = {"dataset": {"config": {"fd": [{"fd_name": "FD001"}]}}}
        apply_overrides(d, ["dataset.config.fd[0].unit_ids=[1,2,3]"])
        assert d["dataset"]["config"]["fd"][0]["unit_ids"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# load_pipeline_spec_with_overrides (integration)
# ---------------------------------------------------------------------------

class TestLoadPipelineSpecWithOverrides:
    def test_overrides_train_field(self, tmp_path):
        spec_yaml = tmp_path / "spec.yaml"
        spec_yaml.write_text("""\
metadata:
  name: test-pipeline
runtime:
  namespace: test-ns
  host: http://localhost:8888
model:
  name: mambasl-cmapss
dataset:
  name: cmapss
train:
  batch_size: 32
  max_epochs: 10
""")
        from kfp_workflow.specs import load_pipeline_spec_with_overrides

        loaded = load_pipeline_spec_with_overrides(
            spec_yaml, ["train.batch_size=128", "train.max_epochs=50"]
        )
        assert loaded.train.batch_size == 128
        assert loaded.train.max_epochs == 50

    def test_overrides_model_config(self, tmp_path):
        spec_yaml = tmp_path / "spec.yaml"
        spec_yaml.write_text("""\
metadata:
  name: test-pipeline
runtime:
  namespace: test-ns
  host: http://localhost:8888
model:
  name: mambasl-cmapss
  config:
    d_model: 64
dataset:
  name: cmapss
""")
        from kfp_workflow.specs import load_pipeline_spec_with_overrides

        loaded = load_pipeline_spec_with_overrides(
            spec_yaml, ["model.config.d_model=256"]
        )
        assert loaded.model.config["d_model"] == 256

    def test_no_overrides_unchanged(self, tmp_path):
        spec_yaml = tmp_path / "spec.yaml"
        spec_yaml.write_text("""\
metadata:
  name: test-pipeline
runtime:
  namespace: test-ns
  host: http://localhost:8888
model:
  name: mambasl-cmapss
dataset:
  name: cmapss
train:
  batch_size: 32
""")
        from kfp_workflow.specs import load_pipeline_spec_with_overrides

        loaded = load_pipeline_spec_with_overrides(spec_yaml)
        assert loaded.train.batch_size == 32


# ---------------------------------------------------------------------------
# validate_plugin_config
# ---------------------------------------------------------------------------

class TestValidatePluginConfig:
    def test_valid_config_no_warnings(self):
        spec = {
            "model": {"name": "mambasl-cmapss", "config": {"d_model": 128}},
            "dataset": {"name": "cmapss", "config": {"fd": [{"fd_name": "FD001"}]}},
            "train": {},
        }
        warnings = validate_plugin_config(spec)
        assert warnings == []

    def test_invalid_type_produces_warning(self):
        spec = {
            "model": {"name": "mambasl-cmapss", "config": {"d_model": "not_an_int"}},
            "dataset": {"name": "cmapss", "config": {"fd": [{"fd_name": "FD001"}]}},
            "train": {},
        }
        warnings = validate_plugin_config(spec)
        assert len(warnings) == 1
        assert "model.config" in warnings[0]

    def test_unknown_plugin_no_warnings(self):
        spec = {
            "model": {"name": "nonexistent-plugin", "config": {}},
            "dataset": {"name": "test"},
            "train": {},
        }
        warnings = validate_plugin_config(spec)
        assert warnings == []

    def test_missing_model_name_no_warnings(self):
        warnings = validate_plugin_config({"dataset": {}, "train": {}})
        assert warnings == []


# ---------------------------------------------------------------------------
# CLI --set flag (help text presence)
# ---------------------------------------------------------------------------

class TestCLISetFlag:
    def test_compile_has_set_option(self):
        from typer.testing import CliRunner
        from kfp_workflow.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["pipeline", "compile", "--help"])
        assert "--set" in result.output

    def test_submit_has_set_option(self):
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["pipeline", "submit", "--help"])
        assert "--set" in result.output


class TestCLIPluginValidation:
    def test_spec_validate_fails_on_plugin_config_error(self, tmp_path):
        spec_yaml = tmp_path / "spec.yaml"
        spec_yaml.write_text("""\
metadata:
  name: test-pipeline
runtime:
  namespace: test-ns
  host: http://localhost:8888
model:
  name: mambasl-cmapss
dataset:
  name: cmapss
  config:
    fd_name: FD001
""")
        runner = CliRunner()

        result = runner.invoke(app, ["spec", "validate", "--spec", str(spec_yaml)])

        assert result.exit_code == 1
        assert "dataset.config validation" in result.output

    def test_pipeline_compile_fails_on_plugin_config_error(self, tmp_path):
        spec_yaml = tmp_path / "spec.yaml"
        spec_yaml.write_text("""\
metadata:
  name: test-pipeline
runtime:
  namespace: test-ns
  host: http://localhost:8888
model:
  name: mambasl-cmapss
  config:
    d_model: not_an_int
dataset:
  name: cmapss
""")
        runner = CliRunner()
        output_yaml = tmp_path / "compiled.yaml"

        result = runner.invoke(
            app,
            [
                "pipeline",
                "compile",
                "--spec",
                str(spec_yaml),
                "--output",
                str(output_yaml),
            ],
        )

        assert result.exit_code == 1
        assert "model.config validation" in result.output
        assert not output_yaml.exists()

    def test_validate_has_set_option(self):
        from typer.testing import CliRunner
        from kfp_workflow.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["spec", "validate", "--help"])
        assert "--set" in result.output
