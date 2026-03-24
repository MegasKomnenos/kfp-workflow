from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

from kfp import Client
from kfp_server_api.exceptions import ApiException

from ..specs import ExperimentSpec
from .pipeline import compile_pipeline


def _wait_port(host: str, port: int, timeout: float = 15.0) -> None:
    started = time.time()
    while time.time() - started < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"port-forward did not come up on {host}:{port}")


def submit_pipeline(
    spec: ExperimentSpec,
    *,
    namespace: str | None = None,
    host: str | None = None,
    existing_token: str | None = None,
    cookies: str | None = None,
) -> str:
    namespace = namespace or spec.runtime.namespace
    compiled_dir = Path("compiled")
    compiled_dir.mkdir(parents=True, exist_ok=True)
    package_path = compiled_dir / f"{spec.metadata.name}.yaml"
    compile_pipeline(spec, str(package_path))

    port = 8888
    forward = subprocess.Popen(
        ["kubectl", "-n", spec.runtime.port_forward_namespace, "port-forward", spec.runtime.port_forward_service, f"{port}:8888"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _wait_port("127.0.0.1", port)
        client = Client(host=host or spec.runtime.host, existing_token=existing_token or None, cookies=cookies or None, namespace=namespace)
        try:
            client.create_experiment(name=spec.metadata.name, namespace=namespace)
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
                raise SystemExit("KFP submission was unauthorized. Re-run with --existing-token or --cookies.") from exc
            raise
        return str(run.run_id)
    finally:
        forward.terminate()
        try:
            forward.wait(timeout=5)
        except subprocess.TimeoutExpired:
            forward.kill()
