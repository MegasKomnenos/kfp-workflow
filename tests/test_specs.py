"""Tests for Pydantic spec loading and validation."""

from pathlib import Path

import pytest

from kfp_workflow.specs import (
    BenchmarkSpec,
    PipelineSpec,
    SearchParamSpec,
    ServingSpec,
    TuneSpec,
    load_benchmark_spec,
    load_pipeline_spec,
    load_serving_spec,
    load_serving_spec_with_overrides,
    merge_best_params,
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


def test_load_serving_spec_with_overrides(tmp_path):
    spec_yaml = tmp_path / "serve.yaml"
    spec_yaml.write_text("""\
metadata:
  name: serve-test
namespace: test-ns
model_name: mambasl-cmapss
model_subpath: mambasl-cmapss/v1
runtime: custom
predictor_image: kfp-workflow:latest
replicas: 1
""")
    spec = load_serving_spec_with_overrides(
        spec_yaml,
        ["metadata.name=serve-overridden", "replicas=3"],
    )
    assert spec.metadata.name == "serve-overridden"
    assert spec.replicas == 3


def test_load_benchmark_spec():
    spec = load_benchmark_spec(CONFIGS / "benchmarks" / "sample_benchmark.yaml")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.metadata.name == "sample-benchmark"
    assert spec.model.model_name == "mambasl-cmapss"
    assert spec.storage.results_pvc == "benchmark-store"


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


def test_benchmark_spec_defaults():
    spec = BenchmarkSpec(
        metadata={"name": "bench"},
        runtime={"namespace": "default"},
        model={
            "model_name": "mambasl-cmapss",
            "model_subpath": "mambasl-cmapss/v1",
        },
        scenario={"kind": "inline"},
    )
    assert spec.storage.results_pvc == "benchmark-store"
    assert spec.model.predictor_image == "kfp-workflow:latest"


def test_tune_spec_defaults():
    spec = TuneSpec(
        metadata={"name": "tune"},
        runtime={"namespace": "default"},
        model={"name": "m"},
        dataset={"name": "d"},
    )
    assert spec.storage.results_pvc == "tune-store"
    assert spec.storage.results_mount_path == "/mnt/tune-results"


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


def test_benchmark_spec_rejects_long_service_name():
    with pytest.raises(Exception, match="too long for KServe RawDeployment"):
        BenchmarkSpec(
            metadata={"name": "benchmark-name"},
            runtime={"namespace": "kubeflow-user-example-com"},
            model={
                "model_name": "mambasl-cmapss",
                "model_subpath": "mambasl-cmapss/v1",
                "service_name": "benchmark-service-name-20260329-very-long-suffix",
            },
            scenario={"kind": "inline"},
        )


# ---------------------------------------------------------------------------
# SearchParamSpec shorthand syntax
# ---------------------------------------------------------------------------


def test_shorthand_log_float():
    p = SearchParamSpec.model_validate({"lr": "log_float(1e-4, 1e-2)"})
    assert p.name == "lr"
    assert p.type == "log_float"
    assert p.low == pytest.approx(1e-4)
    assert p.high == pytest.approx(1e-2)


def test_shorthand_categorical():
    p = SearchParamSpec.model_validate({"d_model": "categorical(32, 64, 128)"})
    assert p.name == "d_model"
    assert p.type == "categorical"
    assert p.values == [32, 64, 128]


def test_shorthand_int():
    p = SearchParamSpec.model_validate({"n_layers": "int(1, 6)"})
    assert p.name == "n_layers"
    assert p.type == "int"
    assert p.low == pytest.approx(1.0)
    assert p.high == pytest.approx(6.0)


def test_shorthand_float_with_step():
    p = SearchParamSpec.model_validate({"dropout": "float(0.0, 0.5, step=0.05)"})
    assert p.name == "dropout"
    assert p.type == "float"
    assert p.low == pytest.approx(0.0)
    assert p.high == pytest.approx(0.5)
    assert p.step == pytest.approx(0.05)


def test_shorthand_categorical_strings():
    p = SearchParamSpec.model_validate({"activation": "categorical(relu, gelu, silu)"})
    assert p.name == "activation"
    assert p.type == "categorical"
    assert p.values == ["relu", "gelu", "silu"]


def test_shorthand_categorical_booleans():
    p = SearchParamSpec.model_validate({"use_norm": "categorical(True, False)"})
    assert p.name == "use_norm"
    assert p.values == [True, False]


def test_verbose_form_still_works():
    p = SearchParamSpec.model_validate({
        "name": "lr", "type": "log_float", "low": 1e-4, "high": 1e-2
    })
    assert p.name == "lr"
    assert p.type == "log_float"


def test_shorthand_invalid_type():
    with pytest.raises(Exception):
        SearchParamSpec.model_validate({"lr": "unknown_type(1, 2)"})


def test_shorthand_missing_args():
    with pytest.raises(Exception):
        SearchParamSpec.model_validate({"lr": "float(1.0)"})


# ---------------------------------------------------------------------------
# HpoSpec composable search spaces
# ---------------------------------------------------------------------------


def test_hpo_spec_overrides_exclude_extra():
    """HpoSpec accepts the composable search space fields."""
    spec = TuneSpec.model_validate({
        "metadata": {"name": "test"},
        "runtime": {"namespace": "default"},
        "model": {"name": "m"},
        "dataset": {"name": "d"},
        "hpo": {
            "overrides": {"lr": {"low": 1e-5, "high": 1e-1}},
            "exclude": ["weight_decay"],
            "extra": [{"dropout": "float(0.0, 0.5)"}],
        },
    })
    assert spec.hpo.overrides == {"lr": {"low": 1e-5, "high": 1e-1}}
    assert spec.hpo.exclude == ["weight_decay"]
    assert len(spec.hpo.extra) == 1
    assert spec.hpo.extra[0].name == "dropout"


# ---------------------------------------------------------------------------
# resolve_search_space composable logic
# ---------------------------------------------------------------------------

from kfp_workflow.tune.engine import resolve_search_space


class _StubPlugin:
    """Minimal plugin that returns a fixed search space."""

    def hpo_search_space(self, spec_dict, profile="default"):
        return [
            SearchParamSpec(name="lr", type="log_float", low=1e-4, high=1e-2),
            SearchParamSpec(name="weight_decay", type="log_float", low=1e-5, high=1e-2),
            SearchParamSpec(name="d_model", type="categorical", values=[32, 64, 128]),
        ]


def test_resolve_search_space_no_composition():
    """Without overrides/exclude/extra, delegates to plugin as before."""
    plugin = _StubPlugin()
    space = resolve_search_space(plugin, {"hpo": {}})
    assert [p.name for p in space] == ["lr", "weight_decay", "d_model"]


def test_resolve_search_space_override():
    plugin = _StubPlugin()
    space = resolve_search_space(plugin, {
        "hpo": {
            "overrides": {"lr": {"low": 1e-5, "high": 1e-1}},
        },
    })
    names = {p.name: p for p in space}
    assert names["lr"].low == pytest.approx(1e-5)
    assert names["lr"].high == pytest.approx(1e-1)
    assert names["lr"].type == "log_float"  # unchanged
    assert "weight_decay" in names


def test_resolve_search_space_exclude():
    plugin = _StubPlugin()
    space = resolve_search_space(plugin, {
        "hpo": {"exclude": ["weight_decay"]},
    })
    names = [p.name for p in space]
    assert "weight_decay" not in names
    assert "lr" in names
    assert "d_model" in names


def test_resolve_search_space_extra():
    plugin = _StubPlugin()
    space = resolve_search_space(plugin, {
        "hpo": {
            "extra": [{"dropout": "float(0.0, 0.5)"}],
        },
    })
    names = [p.name for p in space]
    assert "dropout" in names
    assert len(space) == 4


def test_resolve_search_space_full_composition():
    """Override + exclude + extra together."""
    plugin = _StubPlugin()
    space = resolve_search_space(plugin, {
        "hpo": {
            "overrides": {"lr": {"low": 1e-5, "high": 1e-1}},
            "exclude": ["weight_decay"],
            "extra": [{"dropout": "float(0.0, 0.5)"}],
        },
    })
    names = {p.name: p for p in space}
    assert "weight_decay" not in names
    assert names["lr"].low == pytest.approx(1e-5)
    assert names["dropout"].type == "float"
    assert len(space) == 3  # lr, d_model, dropout


def test_resolve_search_space_custom_takes_precedence():
    """Explicit search_space overrides plugin profile."""
    plugin = _StubPlugin()
    space = resolve_search_space(plugin, {
        "hpo": {
            "search_space": [
                {"name": "custom_param", "type": "int", "low": 1, "high": 10},
            ],
        },
    })
    assert len(space) == 1
    assert space[0].name == "custom_param"


def test_resolve_search_space_custom_with_shorthand():
    """Shorthand syntax works inside search_space too."""
    plugin = _StubPlugin()
    space = resolve_search_space(plugin, {
        "hpo": {
            "search_space": [
                {"lr": "log_float(1e-4, 1e-2)"},
                {"d_model": "categorical(32, 64)"},
            ],
        },
    })
    assert space[0].name == "lr"
    assert space[1].name == "d_model"
    assert space[1].values == [32, 64]


# ---------------------------------------------------------------------------
# merge_best_params
# ---------------------------------------------------------------------------


def test_merge_best_params_routes_to_train_and_config():
    pipeline_raw = {
        "metadata": {"name": "test"},
        "runtime": {"namespace": "default"},
        "model": {"name": "m", "config": {"existing": 1}},
        "dataset": {"name": "d"},
        "train": {"seed": 42},
    }
    best = {
        "learning_rate": 0.001,  # TrainSpec field
        "batch_size": 128,       # TrainSpec field
        "d_model": 64,           # model.config
        "n_layers": 4,           # model.config
    }

    merged = merge_best_params(pipeline_raw, best)

    # TrainSpec fields
    assert merged["train"]["learning_rate"] == pytest.approx(0.001)
    assert merged["train"]["batch_size"] == 128
    assert merged["train"]["seed"] == 42  # unchanged

    # model.config fields
    assert merged["model"]["config"]["d_model"] == 64
    assert merged["model"]["config"]["n_layers"] == 4
    assert merged["model"]["config"]["existing"] == 1  # preserved

    # Original not mutated
    assert "learning_rate" not in pipeline_raw["train"]


def test_merge_best_params_empty():
    pipeline_raw = {"metadata": {"name": "t"}, "model": {"name": "m"}}
    merged = merge_best_params(pipeline_raw, {})
    assert merged == {"metadata": {"name": "t"}, "model": {"name": "m", "config": {}}, "train": {}}
