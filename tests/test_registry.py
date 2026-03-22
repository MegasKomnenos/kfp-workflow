"""Tests for file-backed model and dataset registries."""

import pytest

from kfp_workflow.registry.base import DatasetInfo, ModelInfo
from kfp_workflow.registry.dataset_registry import PVCDatasetRegistry
from kfp_workflow.registry.model_registry import FileModelRegistry


def test_model_info_creation():
    info = ModelInfo(name="test-model", version="v1", uri="/models/test")
    assert info.framework == "pytorch"
    assert info.name == "test-model"


def test_dataset_info_creation():
    info = DatasetInfo(name="test-ds", pvc_name="data-pvc", subpath="datasets/test")
    assert info.version == "v1"
    assert info.pvc_name == "data-pvc"


def test_file_model_registry_register_get(tmp_path):
    registry = FileModelRegistry(registry_path=str(tmp_path / "models.json"))
    info = registry.register_model(name="m1", version="v1", uri="/x/m1.pt")
    assert info.name == "m1"
    assert info.version == "v1"

    retrieved = registry.get_model(name="m1", version="v1")
    assert retrieved.uri == "/x/m1.pt"


def test_file_model_registry_list(tmp_path):
    registry = FileModelRegistry(registry_path=str(tmp_path / "models.json"))
    registry.register_model(name="m1", version="v1", uri="/x/m1.pt")
    registry.register_model(name="m2", version="v1", uri="/x/m2.pt")
    models = registry.list_models()
    assert len(models) == 2
    names = {m.name for m in models}
    assert names == {"m1", "m2"}


def test_file_model_registry_upsert(tmp_path):
    registry = FileModelRegistry(registry_path=str(tmp_path / "models.json"))
    registry.register_model(name="m1", version="v1", uri="/x/old.pt")
    registry.register_model(name="m1", version="v1", uri="/x/new.pt")
    models = registry.list_models()
    assert len(models) == 1
    assert models[0].uri == "/x/new.pt"


def test_file_model_registry_get_not_found(tmp_path):
    registry = FileModelRegistry(registry_path=str(tmp_path / "models.json"))
    with pytest.raises(KeyError):
        registry.get_model(name="nonexistent")


def test_dataset_registry_register_get(tmp_path):
    registry = PVCDatasetRegistry(registry_path=str(tmp_path / "datasets.json"))
    info = registry.register_dataset(
        name="ds1", pvc_name="data-pvc", subpath="data/ds1",
    )
    assert info.name == "ds1"

    retrieved = registry.get_dataset(name="ds1")
    assert retrieved.pvc_name == "data-pvc"
    assert retrieved.subpath == "data/ds1"


def test_dataset_registry_list(tmp_path):
    registry = PVCDatasetRegistry(registry_path=str(tmp_path / "datasets.json"))
    registry.register_dataset(name="ds1", pvc_name="pvc", subpath="a")
    registry.register_dataset(name="ds2", pvc_name="pvc", subpath="b")
    datasets = registry.list_datasets()
    assert len(datasets) == 2


def test_dataset_registry_upsert(tmp_path):
    registry = PVCDatasetRegistry(registry_path=str(tmp_path / "datasets.json"))
    registry.register_dataset(name="ds1", pvc_name="pvc", subpath="old")
    registry.register_dataset(name="ds1", pvc_name="pvc", subpath="new")
    datasets = registry.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].subpath == "new"


def test_dataset_registry_get_not_found(tmp_path):
    registry = PVCDatasetRegistry(registry_path=str(tmp_path / "datasets.json"))
    with pytest.raises(KeyError):
        registry.get_dataset(name="nonexistent")
