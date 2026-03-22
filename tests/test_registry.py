"""Tests for registry ABC contracts and data models."""

import pytest

from kfp_workflow.registry.base import DatasetInfo, ModelInfo
from kfp_workflow.registry.dataset_registry import PVCDatasetRegistry
from kfp_workflow.registry.model_registry import KubeflowModelRegistry


def test_model_info_creation():
    info = ModelInfo(name="test-model", version="v1", uri="/models/test")
    assert info.framework == "pytorch"
    assert info.name == "test-model"


def test_dataset_info_creation():
    info = DatasetInfo(name="test-ds", pvc_name="data-pvc", subpath="datasets/test")
    assert info.version == "v1"
    assert info.pvc_name == "data-pvc"


def test_model_registry_stubs_raise():
    registry = KubeflowModelRegistry()
    with pytest.raises(NotImplementedError):
        registry.register_model(name="m", version="v1", uri="/x")
    with pytest.raises(NotImplementedError):
        registry.get_model(name="m")
    with pytest.raises(NotImplementedError):
        registry.list_models()


def test_dataset_registry_stubs_raise():
    registry = PVCDatasetRegistry()
    with pytest.raises(NotImplementedError):
        registry.register_dataset(name="d", pvc_name="pvc", subpath="/x")
    with pytest.raises(NotImplementedError):
        registry.get_dataset(name="d")
    with pytest.raises(NotImplementedError):
        registry.list_datasets()
