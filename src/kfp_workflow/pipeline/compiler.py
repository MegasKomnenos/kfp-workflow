"""KFP v2 pipeline assembly and compilation."""

from __future__ import annotations

from pathlib import Path

from kfp import compiler, dsl
from kfp.kubernetes import mount_pvc

from kfp_workflow.components import (
    evaluate_component,
    load_data_component,
    preprocess_component,
    save_model_component,
    train_component,
)
from kfp_workflow.specs import PipelineSpec
from kfp_workflow.utils import ensure_parent


def _configure_task(task: dsl.PipelineTask, spec: PipelineSpec) -> dsl.PipelineTask:
    """Apply resource requests/limits and PVC mounts to a pipeline task."""
    res = spec.runtime.resources
    task = (
        task.set_cpu_request(res.cpu_request)
        .set_cpu_limit(res.cpu_limit)
        .set_memory_request(res.memory_request)
        .set_memory_limit(res.memory_limit)
    )
    if spec.runtime.use_gpu:
        task = task.set_gpu_limit(res.gpu_limit)

    # Mount data PVC (read-only for all steps)
    mount_pvc(
        task,
        pvc_name=spec.storage.data_pvc,
        mount_path=spec.storage.data_mount_path,
    )
    # Mount model PVC (read-write for saving weights)
    mount_pvc(
        task,
        pvc_name=spec.storage.model_pvc,
        mount_path=spec.storage.model_mount_path,
    )
    return task


def build_pipeline(spec: PipelineSpec):
    """Construct a KFP v2 pipeline function for the given spec.

    The returned pipeline wires 5 components in sequence:
    ``load_data -> preprocess -> train -> evaluate -> save_model``
    """
    spec_json = spec.model_dump_json()

    @dsl.pipeline(
        name=spec.metadata.name,
        description=spec.metadata.description,
    )
    def training_pipeline():
        # Step 1: Load data
        load_task = load_data_component(spec_json=spec_json)
        _configure_task(load_task, spec)

        # Step 2: Preprocess
        preprocess_task = preprocess_component(
            spec_json=spec_json,
            load_result_json=load_task.output,
        )
        _configure_task(preprocess_task, spec)

        # Step 3: Train
        train_task = train_component(
            spec_json=spec_json,
            preprocess_result_json=preprocess_task.output,
        )
        _configure_task(train_task, spec)

        # Step 4: Evaluate
        eval_task = evaluate_component(
            spec_json=spec_json,
            train_result_json=train_task.output,
            preprocess_result_json=preprocess_task.output,
        )
        _configure_task(eval_task, spec)

        # Step 5: Save model
        save_task = save_model_component(
            spec_json=spec_json,
            train_result_json=train_task.output,
            eval_result_json=eval_task.output,
        )
        _configure_task(save_task, spec)

    return training_pipeline


def compile_pipeline(spec: PipelineSpec, output_path: str | Path) -> Path:
    """Compile a training pipeline to a KFP v2 YAML package.

    Parameters
    ----------
    spec:
        Validated ``PipelineSpec``.
    output_path:
        Destination path for the compiled YAML.

    Returns
    -------
    Path
        The path the YAML was written to.
    """
    output = Path(output_path)
    ensure_parent(output)
    pipeline_func = build_pipeline(spec)
    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=str(output),
    )
    return output
