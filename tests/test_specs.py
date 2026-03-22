"""Tests for Pydantic spec loading and validation."""

from pathlib import Path

import pytest

from kfp_workflow.specs import (
    PipelineSpec,
    ServingSpec,
    load_pipeline_spec,
    load_serving_spec,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_load_pipeline_spec():
    spec = load_pipeline_spec(CONFIGS / "pipelines" / "sample_train.yaml")
    assert isinstance(spec, PipelineSpec)
    assert spec.metadata.name == "sample-train-pipeline"
    assert spec.model.name == "sample-pytorch-model"
    assert spec.dataset.name == "sample-dataset"
    assert spec.train.max_epochs == 50


def test_load_serving_spec():
    spec = load_serving_spec(CONFIGS / "serving" / "sample_serve.yaml")
    assert isinstance(spec, ServingSpec)
    assert spec.metadata.name == "sample-serving"
    assert spec.model_name == "sample-pytorch-model"
    assert spec.runtime == "kserve-torchserve"


def test_pipeline_spec_defaults():
    spec = PipelineSpec(
        metadata={"name": "test"},
        runtime={"namespace": "default"},
        model={"name": "m"},
        dataset={"name": "d"},
    )
    assert spec.train.seed == 42
    assert spec.storage.data_pvc == "dataset-store"
    assert spec.runtime.torch_num_threads == 4


def test_pipeline_spec_rejects_missing_fields():
    with pytest.raises(Exception):
        PipelineSpec(metadata={"name": "test"})
