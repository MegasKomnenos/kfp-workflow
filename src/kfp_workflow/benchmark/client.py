"""KFP submission for benchmark workflows."""

from __future__ import annotations

from typing import Optional

from kfp_workflow.benchmark.compiler import compile_benchmark
from kfp_workflow.benchmark.materialize import load_materialized_benchmark_spec
from kfp_workflow.cli.workflows import compiled_package_path, submit_pipeline_package
from kfp_workflow.specs import BenchmarkSpec


def submit_benchmark(
    spec: BenchmarkSpec,
    *,
    spec_path: str | Path,
    overrides: list[str] | None = None,
    namespace: Optional[str] = None,
    host: Optional[str] = None,
    existing_token: Optional[str] = None,
    cookies: Optional[str] = None,
    user: Optional[str] = None,
) -> str:
    """Compile and submit a benchmark workflow to Kubeflow."""
    namespace = namespace or spec.runtime.namespace

    package_path = compiled_package_path(spec.metadata.name)
    _, materialized = load_materialized_benchmark_spec(spec_path, overrides)
    compile_benchmark(spec, materialized, package_path)

    return submit_pipeline_package(
        package_path=package_path,
        run_name=spec.metadata.name,
        experiment_name=spec.metadata.name,
        namespace=namespace,
        runtime_host=spec.runtime.host,
        port_forward_namespace=spec.runtime.port_forward_namespace,
        port_forward_service=spec.runtime.port_forward_service,
        host=host,
        existing_token=existing_token,
        cookies=cookies,
        user=user,
    )
