"""Tests for benchmark spec materialization and compilation."""

from pathlib import Path

from kfp_workflow.benchmark.compiler import compile_benchmark
from kfp_workflow.benchmark.materialize import load_materialized_benchmark_spec

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_materialized_benchmark_loads():
    spec, materialized = load_materialized_benchmark_spec(
        CONFIGS / "benchmarks" / "sample_benchmark.yaml"
    )
    assert spec.metadata.name == "sample-benchmark"
    assert materialized["scenario"]["dataset"]["kind"] == "cmapss-timeseries"
    assert materialized["metrics"][0]["kind"] == "kepler-energy"


def test_compile_benchmark_produces_yaml(tmp_path):
    spec, materialized = load_materialized_benchmark_spec(
        CONFIGS / "benchmarks" / "sample_benchmark.yaml"
    )
    output = tmp_path / "benchmark.yaml"
    result = compile_benchmark(spec, materialized, output)

    content = result.read_text(encoding="utf-8")
    assert "pipelineSpec" in content or "PIPELINE" in content.upper()
    assert spec.metadata.name in content


def test_python_benchmark_spec_loads(tmp_path):
    spec_py = tmp_path / "benchmark_spec.py"
    spec_py.write_text(
        """\
def build_benchmark_spec():
    return {
        "metadata": {"name": "python-benchmark"},
        "runtime": {"namespace": "kubeflow-user-example-com"},
        "model": {
            "model_name": "mambasl-cmapss",
            "model_subpath": "mambasl-cmapss/v1",
        },
        "scenario": {
            "dataset": {"kind": "cmapss-timeseries", "config": {"fd_name": "FD001"}},
            "pipeline": {"kind": "sequential-replay", "config": {"interval_hz": 1.0}},
        },
        "metrics": [{"kind": "kepler-energy", "config": {"mode": "dynamic"}}],
    }
""",
        encoding="utf-8",
    )

    spec, materialized = load_materialized_benchmark_spec(spec_py)

    assert spec.metadata.name == "python-benchmark"
    assert materialized["scenario"]["pipeline"]["kind"] == "sequential-replay"
