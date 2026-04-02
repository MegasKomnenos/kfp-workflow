"""KFP client for programmatic pipeline submission."""

from __future__ import annotations

from typing import Optional

from kfp_workflow.cli.workflows import compiled_package_path, submit_pipeline_package
from kfp_workflow.pipeline.compiler import compile_pipeline
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

    package_path = compiled_package_path(spec.metadata.name)
    compile_pipeline(spec, package_path)

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
