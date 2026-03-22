"""KFP client for programmatic pipeline submission."""

from __future__ import annotations

from typing import Optional

from kfp_workflow.specs import PipelineSpec


def submit_pipeline(
    spec: PipelineSpec,
    namespace: Optional[str] = None,
    host: Optional[str] = None,
    existing_token: Optional[str] = None,
    cookies: Optional[str] = None,
) -> str:
    """Compile and submit a training pipeline to Kubeflow.

    This involves:
    1. Compiling the pipeline to a temporary YAML package.
    2. Establishing a port-forward to the KFP API (if needed).
    3. Creating a ``kfp.Client`` connection.
    4. Creating an experiment (idempotent) and submitting a run.

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

    Returns
    -------
    str
        The ``run_id`` of the submitted pipeline run.
    """
    raise NotImplementedError(
        "Pipeline submission not yet implemented. "
        "Requires a live Kubeflow cluster with KFP API access."
    )
