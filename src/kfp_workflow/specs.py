"""Pydantic configuration models for training, serving, tuning, and benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kfp_workflow.utils import load_yaml


# ---------------------------------------------------------------------------
# Shared sub-specs
# ---------------------------------------------------------------------------

class SpecModel(BaseModel):
    """Shared Pydantic settings for user-authored workflow specs."""

    model_config = ConfigDict(protected_namespaces=())


class MetadataSpec(SpecModel):
    """Name, description, and version tag for any spec."""
    name: str
    description: str = ""
    version: str = "v1"


class ResourceSpec(SpecModel):
    """Kubernetes resource requests and limits."""
    cpu_request: str = "4"
    cpu_limit: str = "4"
    memory_request: str = "16Gi"
    memory_limit: str = "16Gi"
    gpu_request: str = "1"
    gpu_limit: str = "1"


class RuntimeSpec(SpecModel):
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


class StorageSpec(SpecModel):
    """PVC-based storage configuration for data and model weights."""
    data_pvc: str = "dataset-store"
    model_pvc: str = "model-store"
    storage_class: str = "local-path"
    data_size: str = "32Gi"
    model_size: str = "32Gi"
    data_mount_path: str = "/mnt/data"
    data_subpath: str = ""
    model_mount_path: str = "/mnt/models"
    seed_source_dir: str = ""
    skip_seed: bool = False


class BenchmarkStorageSpec(StorageSpec):
    """PVC-backed storage for benchmark inputs and results."""

    results_pvc: str = "benchmark-store"
    results_size: str = "8Gi"
    results_mount_path: str = "/mnt/benchmarks"


# ---------------------------------------------------------------------------
# Training pipeline spec
# ---------------------------------------------------------------------------

class DatasetRef(SpecModel):
    """Reference to a registered dataset."""
    name: str
    version: str = "v1"
    config: Dict[str, Any] = Field(default_factory=dict)


class ModelRef(SpecModel):
    """Reference to a model architecture and its plugin-specific config."""
    name: str
    version: str = "v1"
    config: Dict[str, Any] = Field(default_factory=dict)


class TrainSpec(SpecModel):
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


class PipelineSpec(SpecModel):
    """Top-level spec for a training pipeline run."""
    metadata: MetadataSpec
    runtime: RuntimeSpec
    storage: StorageSpec = Field(default_factory=StorageSpec)
    model: ModelRef
    dataset: DatasetRef
    train: TrainSpec = Field(default_factory=TrainSpec)


# ---------------------------------------------------------------------------
# Hyperparameter tuning spec
# ---------------------------------------------------------------------------

SearchParamType = Literal["categorical", "int", "float", "log_float"]


class SearchParamSpec(SpecModel):
    """A single hyperparameter search dimension."""

    name: str
    type: SearchParamType
    values: Optional[List[Any]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "SearchParamSpec":
        if self.type == "categorical" and not self.values:
            raise ValueError("categorical param requires 'values'")
        if self.type in {"int", "float", "log_float"} and (
            self.low is None or self.high is None
        ):
            raise ValueError(f"{self.type} param requires 'low' and 'high'")
        return self


class HpoSpec(SpecModel):
    """Hyperparameter optimisation configuration."""

    algorithm: Literal["random", "tpe", "grid"] = "tpe"
    max_trials: int = 20
    max_failed_trials: int = 3
    parallel_trials: int = 1
    builtin_profile: Literal["default", "aggressive", "custom"] = "default"
    search_space: List[SearchParamSpec] = Field(default_factory=list)


class TuneSpec(SpecModel):
    """Top-level spec for a hyperparameter tuning run."""

    metadata: MetadataSpec
    runtime: RuntimeSpec
    storage: StorageSpec = Field(default_factory=StorageSpec)
    model: ModelRef
    dataset: DatasetRef
    train: TrainSpec = Field(default_factory=TrainSpec)
    hpo: HpoSpec = Field(default_factory=HpoSpec)


class HpoTrialResult(SpecModel):
    """Result of a single HPO trial."""

    trial_number: int
    params: Dict[str, Any]
    objective_value: float
    status: Literal["completed", "pruned", "failed"]
    user_attrs: Dict[str, Any] = Field(default_factory=dict)


class HpoResult(SpecModel):
    """Aggregated result of an HPO run."""

    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    n_completed: int
    n_pruned: int
    n_failed: int
    trials: List[HpoTrialResult] = Field(default_factory=list)
    wall_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Serving spec
# ---------------------------------------------------------------------------

class ServingSpec(SpecModel):
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

    @model_validator(mode="after")
    def _validate_serving_contract(self) -> "ServingSpec":
        effective_name = f"{self.metadata.name}-predictor-{self.namespace}"
        if len(effective_name) > 63:
            raise ValueError(
                "metadata.name is too long for KServe RawDeployment. "
                f"Effective host label '{effective_name}' has length {len(effective_name)} "
                "but the limit is 63."
            )

        if self.runtime == "custom":
            if not self.predictor_image:
                raise ValueError(
                    "predictor_image is required when runtime='custom'."
                )

            artifact_like_suffixes = {
                ".pt", ".pth", ".joblib", ".pkl", ".pickle", ".onnx", ".bin",
            }
            suffix = Path(self.model_subpath).suffix.lower()
            if suffix in artifact_like_suffixes:
                raise ValueError(
                    "model_subpath must point to a model directory for runtime='custom', "
                    f"not a file-like path ending in '{suffix}'."
                )

        return self


# ---------------------------------------------------------------------------
# Benchmark spec
# ---------------------------------------------------------------------------


class BenchmarkModelSpec(SpecModel):
    """Model deployment configuration for a benchmark run."""

    model_name: str
    model_version: str = "v1"
    model_pvc: str = "model-store"
    model_subpath: str
    runtime: str = "custom"
    predictor_image: str = "kfp-workflow:latest"
    service_name: str = ""
    replicas: int = 1
    cleanup: bool = True
    wait_timeout: int = 300
    resources: ResourceSpec = Field(default_factory=ResourceSpec)


class BenchmarkSpec(SpecModel):
    """Top-level spec for a benchmark workflow run."""

    metadata: MetadataSpec
    runtime: RuntimeSpec
    storage: BenchmarkStorageSpec = Field(default_factory=BenchmarkStorageSpec)
    model: BenchmarkModelSpec
    scenario: Dict[str, Any]
    metrics: List[Dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_contract(self) -> "BenchmarkSpec":
        service_name = self.model.service_name or self.metadata.name
        effective_name = f"{service_name}-predictor-{self.runtime.namespace}"
        if len(effective_name) > 63:
            raise ValueError(
                "benchmark model service_name is too long for KServe RawDeployment. "
                f"Effective host label '{effective_name}' has length {len(effective_name)} "
                "but the limit is 63."
            )
        return self


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pipeline_spec(path: str | Path) -> PipelineSpec:
    """Load and validate a training pipeline spec from a YAML file."""
    return PipelineSpec.model_validate(load_yaml(Path(path)))


def load_pipeline_spec_with_overrides(
    path: str | Path,
    overrides: list[str] | None = None,
) -> PipelineSpec:
    """Load a pipeline spec from YAML, apply CLI overrides, then validate.

    Overrides are applied to the raw dict *before* Pydantic validation,
    so known fields get type-checked and open dicts pass through.
    """
    raw = load_yaml(Path(path))
    if overrides:
        from kfp_workflow.config_override import apply_overrides
        raw = apply_overrides(raw, overrides)
    return PipelineSpec.model_validate(raw)


def load_serving_spec(path: str | Path) -> ServingSpec:
    """Load and validate a serving spec from a YAML file."""
    return ServingSpec.model_validate(load_yaml(Path(path)))


def load_serving_spec_with_overrides(
    path: str | Path,
    overrides: list[str] | None = None,
) -> ServingSpec:
    """Load a serving spec from YAML, apply CLI overrides, then validate."""
    raw = load_yaml(Path(path))
    if overrides:
        from kfp_workflow.config_override import apply_overrides
        raw = apply_overrides(raw, overrides)
    return ServingSpec.model_validate(raw)


def load_tune_spec(path: str | Path) -> TuneSpec:
    """Load and validate a tuning spec from a YAML file."""
    return TuneSpec.model_validate(load_yaml(Path(path)))


def load_tune_spec_with_overrides(
    path: str | Path,
    overrides: list[str] | None = None,
) -> TuneSpec:
    """Load a tuning spec from YAML, apply CLI overrides, then validate."""
    raw = load_yaml(Path(path))
    if overrides:
        from kfp_workflow.config_override import apply_overrides
        raw = apply_overrides(raw, overrides)
    return TuneSpec.model_validate(raw)


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    """Load and validate a benchmark spec from YAML or Python."""
    raw = load_yaml(Path(path))
    return BenchmarkSpec.model_validate(raw)


def load_benchmark_spec_with_overrides(
    path: str | Path,
    overrides: list[str] | None = None,
) -> BenchmarkSpec:
    """Load a benchmark spec from YAML, apply CLI overrides, then validate."""
    raw = load_yaml(Path(path))
    if overrides:
        from kfp_workflow.config_override import apply_overrides

        raw = apply_overrides(raw, overrides)
    return BenchmarkSpec.model_validate(raw)
