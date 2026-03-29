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
