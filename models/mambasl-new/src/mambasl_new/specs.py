from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .utils import load_yaml


DatasetName = Literal["FD001", "FD002", "FD003", "FD004"]
StageName = Literal["hpo", "final_train_eval", "ablation_sweep", "aggregate_reports"]
StorageMode = Literal["download", "pvc"]
FeatureMode = Literal["settings_plus_sensors", "sensors_only"]
NormMode = Literal["global_standard", "condition_standard", "condition_minmax"]
SelectionMetric = Literal["rmse", "score", "hybrid"]
ValMode = Literal["all_windows", "pseudo_test"]
SearchParamType = Literal["categorical", "int", "float", "log_float"]


class MetadataSpec(BaseModel):
    name: str
    description: str = ""
    version: str = "v1"


class ResourceSpec(BaseModel):
    cpu_request: str = "4"
    cpu_limit: str = "4"
    memory_request: str = "16Gi"
    memory_limit: str = "16Gi"
    gpu_request: str = "1"
    gpu_limit: str = "1"


class RuntimeSpec(BaseModel):
    namespace: str = "kubeflow-user-example-com"
    pipeline_root: str
    image: str = "mambasl-new:latest"
    image_pull_policy: str = "IfNotPresent"
    service_account: str = "default-editor"
    host: str = "http://127.0.0.1:8888"
    port_forward_namespace: str = "kubeflow"
    port_forward_service: str = "svc/ml-pipeline"
    use_gpu: bool = True
    torch_num_threads: int = 4
    resources: ResourceSpec = Field(default_factory=ResourceSpec)


class StorageSpec(BaseModel):
    mode: StorageMode = "download"
    data_pvc: str = "cmapss-data"
    results_pvc: str = "cmapss-results"
    storage_class: str = "local-path"
    data_size: str = "8Gi"
    results_size: str = "32Gi"
    data_mount_path: str = "/mnt/data"
    results_mount_path: str = "/mnt/results"
    seed_source_dir: str = ""
    skip_seed: bool = False


class DataSpec(BaseModel):
    data_root: str = "data/cmapss"
    download_policy: Literal["if_missing", "never", "always"] = "if_missing"
    nasa_url: str = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
    nasa_md5: str = "79a22f36e80606c69d0e9e4da5bb2b7a"


class DatasetsSpec(BaseModel):
    items: List[DatasetName]

    @field_validator("items")
    @classmethod
    def non_empty(cls, value: List[DatasetName]) -> List[DatasetName]:
        if not value:
            raise ValueError("at least one dataset is required")
        return value


class TrainDefaultsSpec(BaseModel):
    seed: int = 42
    feature_mode: FeatureMode = "settings_plus_sensors"
    norm_mode: NormMode = "condition_minmax"
    selection_metric: SelectionMetric = "rmse"
    score_weight: float = 0.01
    val_mode: ValMode = "all_windows"
    val_pseudo_samples: int = 3
    val_min_history: int = 20
    refit_full_train: bool = True
    hpo_train_stride: int = 2
    hpo_val_stride: int = 2
    hpo_max_epochs: int = 25
    hpo_patience: int = 5
    final_max_epochs: int = 50
    final_patience: int = 8
    fixed_params: Dict[str, Any] = Field(default_factory=dict)


class SearchParamSpec(BaseModel):
    name: str
    type: SearchParamType
    values: Optional[List[Any]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None

    @model_validator(mode="after")
    def validate_shape(self) -> "SearchParamSpec":
        if self.type == "categorical" and not self.values:
            raise ValueError("categorical search param requires values")
        if self.type in {"int", "float", "log_float"} and (self.low is None or self.high is None):
            raise ValueError(f"{self.type} search param requires low/high")
        return self


class HpoSpec(BaseModel):
    enabled: bool = True
    builtin_profile: Literal["default", "aggressive", "custom"] = "default"
    algorithm: Literal["random", "tpe", "grid"] = "random"
    max_trial_count: int = 12
    parallel_trial_count: int = 2
    max_failed_trial_count: int = 3
    search_space: List[SearchParamSpec] = Field(default_factory=list)


class AblationSpec(BaseModel):
    enabled: bool = True
    axes: Dict[str, List[Any]] = Field(default_factory=dict)
    base_overrides: Dict[str, Any] = Field(default_factory=dict)


class OutputsSpec(BaseModel):
    local_results_dir: str = "results"
    retain_model_state: bool = False


class ExperimentSpec(BaseModel):
    metadata: MetadataSpec
    runtime: RuntimeSpec
    storage: StorageSpec = Field(default_factory=StorageSpec)
    data: DataSpec = Field(default_factory=DataSpec)
    datasets: DatasetsSpec
    stages: List[StageName] = Field(default_factory=lambda: ["hpo", "final_train_eval", "ablation_sweep", "aggregate_reports"])
    train_defaults: TrainDefaultsSpec = Field(default_factory=TrainDefaultsSpec)
    hpo: HpoSpec = Field(default_factory=HpoSpec)
    ablations: AblationSpec = Field(default_factory=AblationSpec)
    outputs: OutputsSpec = Field(default_factory=OutputsSpec)

    @field_validator("stages")
    @classmethod
    def unique_stages(cls, value: List[StageName]) -> List[StageName]:
        seen = set()
        out = []
        for item in value:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out


def load_spec(path: str | Path) -> ExperimentSpec:
    return ExperimentSpec.model_validate(load_yaml(Path(path)))


def execution_spec(spec: ExperimentSpec, *, kubeflow: bool) -> ExperimentSpec:
    payload = spec.model_copy(deep=True)
    if kubeflow and payload.storage.mode == "pvc":
        payload.data.data_root = payload.storage.data_mount_path
        payload.outputs.local_results_dir = payload.storage.results_mount_path
    return payload


def expand_ablation_cases(spec: ExperimentSpec) -> List[Dict[str, Any]]:
    if not spec.ablations.enabled or not spec.ablations.axes:
        return []
    keys = sorted(spec.ablations.axes)
    cases = []
    for values in product(*(spec.ablations.axes[key] for key in keys)):
        overrides = dict(spec.ablations.base_overrides)
        name_parts = []
        for key, value in zip(keys, values):
            overrides[key] = value
            name_parts.append(f"{key}-{value}")
        cases.append({"name": "__".join(name_parts), "overrides": overrides})
    return cases
