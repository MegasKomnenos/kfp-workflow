"""KFP client for programmatic pipeline submission."""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

from kfp import Client
from kfp_server_api.exceptions import ApiException

from kfp_workflow.pipeline.compiler import compile_pipeline
from kfp_workflow.specs import PipelineSpec


def _inject_user_header(client: Client, user: str) -> None:
    """Set the ``kubeflow-userid`` header on all internal KFP API clients.

    When accessing the ml-pipeline service via port-forward (bypassing
    Istio auth), the identity header must be injected manually.
    """
    for attr in (
        "_experiment_api", "_run_api", "_pipelines_api",
        "_upload_api", "_recurring_run_api", "_healthz_api",
    ):
        api = getattr(client, attr, None)
        if api is not None and hasattr(api, "api_client"):
            api.api_client.default_headers["kubeflow-userid"] = user


def _wait_port(host: str, port: int, timeout: float = 15.0) -> None:
    """Poll a TCP socket until it accepts connections."""
    started = time.time()
    while time.time() - started < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"port-forward did not come up on {host}:{port}")


def submit_pipeline(
    spec: PipelineSpec,
    namespace: Optional[str] = None,
    host: Optional[str] = None,
    existing_token: Optional[str] = None,
    cookies: Optional[str] = None,
) -> str:
    """Compile and submit a training pipeline to Kubeflow.

    Steps:
    1. Compile the pipeline to a temporary YAML package.
    2. Start ``kubectl port-forward`` to the KFP API service.
    3. Create a ``kfp.Client`` connection.
    4. Create an experiment (idempotent) and submit a run.

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
    namespace = namespace or spec.runtime.namespace

    # 1. Compile pipeline to temp YAML
    compiled_dir = Path("compiled")
    compiled_dir.mkdir(parents=True, exist_ok=True)
    package_path = compiled_dir / f"{spec.metadata.name}.yaml"
    compile_pipeline(spec, package_path)

    # 2. Start port-forward to KFP API
    port = 8888
    forward = subprocess.Popen(
        [
            "kubectl", "-n", spec.runtime.port_forward_namespace,
            "port-forward", spec.runtime.port_forward_service,
            f"{port}:8888",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _wait_port("127.0.0.1", port)

        # 3. Create KFP client
        client = Client(
            host=host or spec.runtime.host,
            existing_token=existing_token or None,
            cookies=cookies or None,
            namespace=namespace,
        )

        # Inject kubeflow-userid header for port-forward auth bypass
        _inject_user_header(client, "user@example.com")

        # 4. Create experiment (idempotent) and submit run
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
    finally:
        forward.terminate()
        try:
            forward.wait(timeout=5)
        except subprocess.TimeoutExpired:
            forward.kill()
