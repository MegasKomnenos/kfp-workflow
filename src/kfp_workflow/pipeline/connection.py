"""Reusable KFP API connection via kubectl port-forward."""

from __future__ import annotations

import contextlib
import socket
import subprocess
import time
from typing import Generator, Optional

from kfp import Client


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


@contextlib.contextmanager
def kfp_connection(
    *,
    namespace: str = "kubeflow-user-example-com",
    host: str = "http://127.0.0.1:8888",
    port_forward_namespace: str = "kubeflow",
    port_forward_service: str = "svc/ml-pipeline",
    user: str = "user@example.com",
    existing_token: Optional[str] = None,
    cookies: Optional[str] = None,
) -> Generator[Client, None, None]:
    """Context manager that yields a configured ``kfp.Client``.

    Starts a ``kubectl port-forward`` subprocess, waits for it to become
    ready, creates and configures the Client with the user identity header,
    yields it, and tears down the port-forward on exit.
    """
    port = 8888
    forward = subprocess.Popen(
        [
            "kubectl", "-n", port_forward_namespace,
            "port-forward", port_forward_service,
            f"{port}:8888",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _wait_port("127.0.0.1", port)

        client = Client(
            host=host,
            existing_token=existing_token or None,
            cookies=cookies or None,
            namespace=namespace,
        )
        _inject_user_header(client, user)

        yield client
    finally:
        forward.terminate()
        try:
            forward.wait(timeout=5)
        except subprocess.TimeoutExpired:
            forward.kill()
