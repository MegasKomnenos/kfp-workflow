"""KFP compilation for benchmark workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from kfp import compiler, dsl, kubernetes

from kfp_workflow.benchmark.components import (
    cleanup_benchmark_model_component,
    deploy_benchmark_model_component,
    run_benchmark_component,
    wait_for_benchmark_model_component,
)
from kfp_workflow.specs import BenchmarkSpec
from kfp_workflow.utils import ensure_parent


def _set_image_pull_policy(task: dsl.PipelineTask, policy: str) -> dsl.PipelineTask:
    kube_config = dict(task.platform_config.get("kubernetes", {}))
    kube_config["imagePullPolicy"] = policy
    task.platform_config["kubernetes"] = kube_config
    return task


def _mount_pvc(task: dsl.PipelineTask, pvc_name: str, mount_path: str) -> dsl.PipelineTask:
    """Mount a PVC using the supported KFP Kubernetes helper."""
    return kubernetes.mount_pvc(task, pvc_name=pvc_name, mount_path=mount_path)


def _set_pod_annotation(
    task: dsl.PipelineTask,
    key: str,
    value: str,
) -> dsl.PipelineTask:
    """Apply a Pod annotation using the supported KFP Kubernetes helper."""
    return kubernetes.add_pod_annotation(
        task,
        annotation_key=key,
        annotation_value=value,
    )


def _configure_task(task: dsl.PipelineTask, spec: BenchmarkSpec) -> dsl.PipelineTask:
    res = spec.runtime.resources
    task = (
        task.set_cpu_request(res.cpu_request)
        .set_cpu_limit(res.cpu_limit)
        .set_memory_request(res.memory_request)
        .set_memory_limit(res.memory_limit)
    )
    if spec.runtime.use_gpu:
        task = task.set_gpu_limit(res.gpu_limit)

    _mount_pvc(task, pvc_name=spec.storage.data_pvc, mount_path=spec.storage.data_mount_path)
    _mount_pvc(task, pvc_name=spec.storage.model_pvc, mount_path=spec.storage.model_mount_path)
    _mount_pvc(task, pvc_name=spec.storage.results_pvc, mount_path=spec.storage.results_mount_path)
    task.set_caching_options(False)
    return _set_image_pull_policy(task, spec.runtime.image_pull_policy)


def build_benchmark_pipeline(spec: BenchmarkSpec, materialized_spec: dict):
    """Construct a KFP benchmark workflow."""
    spec_json = json.dumps(materialized_spec, sort_keys=True)

    @dsl.pipeline(name=spec.metadata.name, description=spec.metadata.description)
    def benchmark_pipeline():
        cleanup_task = cleanup_benchmark_model_component(spec_json=spec_json)
        _configure_task(cleanup_task, spec)

        with dsl.ExitHandler(cleanup_task):
            deploy_task = deploy_benchmark_model_component(spec_json=spec_json)
            _configure_task(deploy_task, spec)

            wait_task = wait_for_benchmark_model_component(spec_json=spec_json)
            wait_task.after(deploy_task)
            _configure_task(wait_task, spec)

            run_task = run_benchmark_component(
                spec_json=spec_json,
                target_json=wait_task.output,
            )
            run_task.after(wait_task)
            _configure_task(run_task, spec)
            _set_pod_annotation(run_task, "sidecar.istio.io/inject", "true")

    return benchmark_pipeline


def compile_benchmark(
    spec: BenchmarkSpec,
    materialized_spec: dict,
    output_path: str | Path,
) -> Path:
    """Compile a materialized benchmark spec into a KFP package."""
    output = Path(output_path)
    ensure_parent(output)
    compiler.Compiler().compile(
        pipeline_func=build_benchmark_pipeline(spec, materialized_spec),
        package_path=str(output),
    )
    return output
