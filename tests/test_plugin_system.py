"""Tests for the model plugin system."""

import pytest

from kfp_workflow.plugins import get_plugin, get_plugin_registry
from kfp_workflow.plugins.base import (
    EvalResult,
    LoadDataResult,
    ModelPlugin,
    PreprocessResult,
    SaveResult,
    TrainResult,
    result_to_dict,
)


def test_plugin_registry_contains_mambasl():
    registry = get_plugin_registry()
    assert "mambasl-cmapss" in registry


def test_get_plugin_returns_instance():
    plugin = get_plugin("mambasl-cmapss")
    assert isinstance(plugin, ModelPlugin)
    assert plugin.name() == "mambasl-cmapss"


def test_get_plugin_unknown_raises():
    with pytest.raises(KeyError, match="Unknown model plugin"):
        get_plugin("nonexistent-model")


def test_load_data_result_serialisation():
    result = LoadDataResult(
        data_dir="/tmp/data",
        dataset_name="FD001",
        num_train_samples=100,
        num_test_samples=50,
    )
    d = result_to_dict(result)
    assert d["data_dir"] == "/tmp/data"
    assert d["dataset_name"] == "FD001"
    assert d["num_train_samples"] == 100


def test_preprocess_result_serialisation():
    result = PreprocessResult(
        artifacts_dir="/tmp/artifacts",
        x_train_path="/tmp/x_train.npy",
        y_train_path="/tmp/y_train.npy",
        x_val_path="/tmp/x_val.npy",
        y_val_path="/tmp/y_val.npy",
        x_test_path="/tmp/x_test.npy",
        y_test_path="/tmp/y_test.npy",
        feature_dim=14,
        seq_len=30,
        num_train=500,
        num_val=100,
        num_test=50,
    )
    d = result_to_dict(result)
    assert d["feature_dim"] == 14
    assert d["seq_len"] == 30


def test_train_result_serialisation():
    result = TrainResult(
        model_path="/tmp/model.pt",
        best_epoch=5,
        train_loss=0.1,
        val_loss=0.2,
    )
    d = result_to_dict(result)
    assert d["model_path"] == "/tmp/model.pt"
    assert d["best_epoch"] == 5


def test_eval_result_serialisation():
    result = EvalResult(
        metrics={"rmse": 15.0, "score": 300.0},
        model_path="/tmp/model.pt",
    )
    d = result_to_dict(result)
    assert d["metrics"]["rmse"] == 15.0


def test_save_result_serialisation():
    result = SaveResult(
        saved_path="/mnt/models/mambasl-cmapss/v1/model.pt",
        model_name="mambasl-cmapss",
        model_version="v1",
    )
    d = result_to_dict(result)
    assert d["model_name"] == "mambasl-cmapss"


def test_build_cfg_merges_config():
    """Test that _build_cfg merges model config with train params."""
    from kfp_workflow.plugins.mambasl_cmapss import _build_cfg

    spec = {
        "model": {
            "name": "mambasl-cmapss",
            "config": {"d_model": 32, "d_state": 8},
        },
        "train": {
            "batch_size": 128,
            "learning_rate": 0.01,
            "weight_decay": 0.001,
        },
    }
    cfg = _build_cfg(spec)
    # Explicit values from config
    assert cfg["d_model"] == 32
    assert cfg["d_state"] == 8
    # Values from train
    assert cfg["batch_size"] == 128
    assert cfg["lr"] == 0.01
    # Defaults
    assert cfg["d_conv"] == 3
    assert cfg["expand"] == 2
    assert cfg["window_size"] == 50


def test_build_cfg_defaults():
    """Test _build_cfg with minimal spec gives sensible defaults."""
    from kfp_workflow.plugins.mambasl_cmapss import _build_cfg

    spec = {"model": {}, "train": {}}
    cfg = _build_cfg(spec)
    assert cfg["d_model"] == 64
    assert cfg["batch_size"] == 64
    assert cfg["lr"] == 1e-3
    assert cfg["max_rul"] == 125.0
