"""Tests for plugin-backed serving artifact resolution."""

from __future__ import annotations

import importlib
import json
import sys
import types
from unittest.mock import patch

class _DummyPtPlugin:
    def __init__(self):
        self.loaded = None
        self.warmed = None

    @classmethod
    def serving_model_filenames(cls):
        return ["model.pt"]

    def load_serving_artifact(self, model_path, model_config):
        self.loaded = (model_path, model_config)
        return {"artifact_path": model_path}

    def warmup_serving_artifact(self, artifact, model_config):
        self.warmed = (artifact, model_config)

    def predict_loaded(self, artifact, input_data, model_config):
        return [1.0]

    def predict(self, model_path, input_data, model_config):
        raise NotImplementedError


class _DummyJoblibPlugin:
    def __init__(self):
        self.loaded = None

    @classmethod
    def serving_model_filenames(cls):
        return ["model.joblib", "model.pt"]

    def load_serving_artifact(self, model_path, model_config):
        self.loaded = (model_path, model_config)
        return {"artifact_path": model_path}

    def warmup_serving_artifact(self, artifact, model_config):
        return None

    def predict_loaded(self, artifact, input_data, model_config):
        return [2.0]

    def predict(self, model_path, input_data, model_config):
        raise NotImplementedError


def _load_predictor_class():
    fake_kserve = types.ModuleType("kserve")

    class _FakeModel:
        def __init__(self, name):
            self.name = name
            self.ready = False

    class _FakeModelServer:
        def start(self, models):
            self.models = models

    fake_kserve.Model = _FakeModel
    fake_kserve.ModelServer = _FakeModelServer

    sys.modules.pop("kfp_workflow.serving.predictor", None)
    with patch.dict(sys.modules, {"kserve": fake_kserve}):
        module = importlib.import_module("kfp_workflow.serving.predictor")
    return module.PluginPredictor


def test_predictor_loads_default_pt_artifact(tmp_path):
    PluginPredictor = _load_predictor_class()
    (tmp_path / "model.pt").write_text("weights")
    (tmp_path / "model_config.json").write_text(json.dumps({"a": 1}))

    predictor = PluginPredictor("model", str(tmp_path), "dummy")
    plugin = _DummyPtPlugin()

    with patch("kfp_workflow.plugins.get_plugin", return_value=plugin):
        predictor.load()

    assert predictor.ready is True
    assert predictor._model_path.endswith("model.pt")
    assert predictor._model_config == {"a": 1}
    assert plugin.loaded[0].endswith("model.pt")
    assert plugin.warmed[0]["artifact_path"].endswith("model.pt")


def test_predictor_loads_joblib_artifact_for_mrhysp(tmp_path):
    PluginPredictor = _load_predictor_class()
    (tmp_path / "model.joblib").write_text("serialized")
    (tmp_path / "model_config.json").write_text(json.dumps({"b": 2}))

    predictor = PluginPredictor("model", str(tmp_path), "dummy")
    plugin = _DummyJoblibPlugin()

    with patch("kfp_workflow.plugins.get_plugin", return_value=plugin):
        predictor.load()

    assert predictor.ready is True
    assert predictor._model_path.endswith("model.joblib")
    assert predictor._model_config == {"b": 2}
    assert plugin.loaded[0].endswith("model.joblib")
