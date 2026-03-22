"""KFP client for programmatic pipeline submission."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from kfp_server_api.exceptions import ApiException

from kfp_workflow.pipeline.compiler import compile_pipeline
from kfp_workflow.pipeline.connection import kfp_connection
from kfp_workflow.specs import PipelineSpec


def submit_pipeline(
    spec: PipelineSpec,
    namespace: Optional[str] = None,
    host: Optional[str] = None,
    existing_token: Optional[str] = None,
    cookies: Optional[str] = None,
    user: Optional[str] = None,
) -> str:
    """Compile and submit a training pipeline to Kubeflow.

    Steps:
    1. Compile the pipeline to a temporary YAML package.
    2. Open a ``kfp_connection`` (port-forward + Client).
    3. Create an experiment (idempotent) and submit a run.

    Parameters
    ----------
    spec:
        Validated ``PipelineSpec``.
    namespace:
        Kubernetes namespace override.
    host:
        KFP API host override.
    existing_token:
        Bearer token for authentication.
    cookies:
        Cookie header for authentication.
    user:
        Kubeflow user identity header value.

    Returns
    -------
    str
        The ``run_id`` of the submitted pipeline run.
    """
    namespace = namespace or spec.runtime.namespace

    # 1. Compile pipeline to temp YAML
    compiled_dir = Path("compiled")
    compiled_dir.mkdir(parents=True, exist_ok=True)
    package_path = compiled_dir / f"{spec.metadata.name}.yaml"
    compile_pipeline(spec, package_path)

    # 2. Connect to KFP API and submit
    with kfp_connection(
        namespace=namespace,
        host=host or spec.runtime.host,
        port_forward_namespace=spec.runtime.port_forward_namespace,
        port_forward_service=spec.runtime.port_forward_service,
        user=user or "user@example.com",
        existing_token=existing_token,
        cookies=cookies,
    ) as client:
        # Create experiment (idempotent)
        try:
            client.create_experiment(
                name=spec.metadata.name, namespace=namespace,
            )
        except Exception:
            pass  # experiment already exists

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
                    "KFP submission was unauthorized. "
                    "Re-run with --existing-token or --cookies."
                ) from exc
            raise

        return run.run_id
