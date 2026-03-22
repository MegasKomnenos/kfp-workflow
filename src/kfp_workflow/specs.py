"""Pydantic configuration models for training pipelines and inference serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from kfp_workflow.utils import load_yaml


# ---------------------------------------------------------------------------
# Shared sub-specs
# ---------------------------------------------------------------------------

class MetadataSpec(BaseModel):
    """Name, description, and version tag for any spec."""
    name: str
    description: str = ""
    version: str = "v1"


class ResourceSpec(BaseModel):
    """Kubernetes resource requests and limits."""
    cpu_request: str = "4"
    cpu_limit: str = "4"
    memory_request: str = "16Gi"
    memory_limit: str = "16Gi"
    gpu_request: str = "1"
    gpu_limit: str = "1"


class RuntimeSpec(BaseModel):
    """Execution environment for pipeline components."""
    namespace: str = "kubeflow-user-example-com"
    pipeline_root: str = ""
    image: str = "kfp-workflow:latest"
    image_pull_policy: str = "IfNotPresent"
    service_account: str = "default-editor"
    host: str = "http://127.0.0.1:8888"
    port_forward_namespace: str = "kubeflow"
    port_forward_service: str = "svc/ml-pipeline"
    use_gpu: bool = True
    torch_num_threads: int = 4
    resources: ResourceSpec = Field(default_factory=ResourceSpec)


class StorageSpec(BaseModel):
    """PVC-based storage configuration for data and model weights."""
    data_pvc: str = "dataset-store"
    model_pvc: str = "model-store"
    storage_class: str = "local-path"
    data_size: str = "32Gi"
    model_size: str = "32Gi"
    data_mount_path: str = "/mnt/data"
    model_mount_path: str = "/mnt/models"
    seed_source_dir: str = ""
    skip_seed: bool = False


# ---------------------------------------------------------------------------
# Training pipeline spec
# ---------------------------------------------------------------------------

class DatasetRef(BaseModel):
    """Reference to a registered dataset."""
    name: str
    version: str = "v1"
    config: Dict[str, Any] = Field(default_factory=dict)


class ModelRef(BaseModel):
    """Reference to a model architecture and its plugin-specific config."""
    name: str
    version: str = "v1"
    config: Dict[str, Any] = Field(default_factory=dict)


class TrainSpec(BaseModel):
    """Training hyper-parameters."""
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 8
    val_split: float = 0.2
    selection_metric: str = "rmse"
    score_weight: float = 0.01


class PipelineSpec(BaseModel):
    """Top-level spec for a training pipeline run."""
    metadata: MetadataSpec
    runtime: RuntimeSpec
    storage: StorageSpec = Field(default_factory=StorageSpec)
    model: ModelRef
    dataset: DatasetRef
    train: TrainSpec = Field(default_factory=TrainSpec)


# ---------------------------------------------------------------------------
# Serving spec
# ---------------------------------------------------------------------------

class ServingSpec(BaseModel):
    """Spec for creating a KServe InferenceService."""
    metadata: MetadataSpec
    namespace: str = "kubeflow-user-example-com"
    model_name: str
    model_version: str = "v1"
    model_pvc: str = "model-store"
    model_subpath: str
    runtime: str = "custom"
    predictor_image: str = ""
    replicas: int = 1
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    serving_model_config: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pipeline_spec(path: str | Path) -> PipelineSpec:
    """Load and validate a training pipeline spec from a YAML file."""
    return PipelineSpec.model_validate(load_yaml(Path(path)))


def load_serving_spec(path: str | Path) -> ServingSpec:
    """Load and validate a serving spec from a YAML file."""
    return ServingSpec.model_validate(load_yaml(Path(path)))
