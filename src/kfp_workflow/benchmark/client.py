"""KFP submission for benchmark workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from kfp_server_api.exceptions import ApiException

from kfp_workflow.benchmark.compiler import compile_benchmark
from kfp_workflow.benchmark.materialize import load_materialized_benchmark_spec
from kfp_workflow.pipeline.connection import kfp_connection
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

    compiled_dir = Path("compiled")
    compiled_dir.mkdir(parents=True, exist_ok=True)
    package_path = compiled_dir / f"{spec.metadata.name}.yaml"
    _, materialized = load_materialized_benchmark_spec(spec_path, overrides)
    compile_benchmark(spec, materialized, package_path)

    with kfp_connection(
        namespace=namespace,
        host=host or spec.runtime.host,
        port_forward_namespace=spec.runtime.port_forward_namespace,
        port_forward_service=spec.runtime.port_forward_service,
        user=user or "user@example.com",
        existing_token=existing_token,
        cookies=cookies,
    ) as client:
        try:
            client.create_experiment(
                name=spec.metadata.name,
                namespace=namespace,
            )
        except Exception:
            pass

        try:
            run = client.create_run_from_pipeline_package(
                pipeline_file=str(package_path),
                arguments={},
                run_name=spec.metadata.name,
                experiment_name=spec.metadata.name,
                namespace=namespace,
            )
        except ApiException as exc:
            if exc.status == 401:
                raise SystemExit(
                    "KFP submission was unauthorized. Re-run with --existing-token or --cookies."
                ) from exc
            raise
        return run.run_id
