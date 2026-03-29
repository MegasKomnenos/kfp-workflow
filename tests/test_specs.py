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
    assert spec.storage.data_subpath == ""
    assert spec.runtime.torch_num_threads == 4


def test_pipeline_spec_rejects_missing_fields():
    with pytest.raises(Exception):
        PipelineSpec(metadata={"name": "test"})


def test_serving_spec_rejects_long_rawdeployment_name():
    with pytest.raises(Exception, match="too long for KServe RawDeployment"):
        ServingSpec(
            metadata={"name": "mambasl-cmapss-serving-20260329-r1"},
            namespace="kubeflow-user-example-com",
            model_name="mambasl-cmapss",
            model_subpath="mambasl-cmapss/v1",
            runtime="custom",
            predictor_image="kfp-workflow:latest",
        )


def test_serving_spec_rejects_file_like_model_subpath_for_custom_runtime():
    with pytest.raises(Exception, match="model_subpath must point to a model directory"):
        ServingSpec(
            metadata={"name": "short-name"},
            namespace="kubeflow-user-example-com",
            model_name="mambasl-cmapss",
            model_subpath="mambasl-cmapss/v1/model.pt",
            runtime="custom",
            predictor_image="kfp-workflow:latest",
        )
