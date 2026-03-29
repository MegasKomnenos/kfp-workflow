import json
from typing import Dict, List

from kfp import compiler, dsl, kubernetes

from ..specs import ExperimentSpec, execution_spec


def _configure_task(task, spec: ExperimentSpec):
    if spec.storage.mode == "pvc":
        kubernetes.mount_pvc(task, pvc_name=spec.storage.data_pvc, mount_path=spec.storage.data_mount_path)
        kubernetes.mount_pvc(task, pvc_name=spec.storage.results_pvc, mount_path=spec.storage.results_mount_path)
    task.set_cpu_request(spec.runtime.resources.cpu_request)
    task.set_cpu_limit(spec.runtime.resources.cpu_limit)
    task.set_memory_request(spec.runtime.resources.memory_request)
    task.set_memory_limit(spec.runtime.resources.memory_limit)
    if spec.runtime.resources.gpu_limit not in {"0", "", None}:
        task.set_accelerator_limit(int(spec.runtime.resources.gpu_limit))
        task.set_accelerator_type("nvidia.com/gpu")
    return task


def build_pipeline(spec: ExperimentSpec):
    spec = execution_spec(spec, kubeflow=True)
    base_image = spec.runtime.image
    spec_json = json.dumps(spec.model_dump(mode="json"))
    datasets = list(spec.datasets.items)
    run_hpo = "hpo" in spec.stages and spec.hpo.enabled
    run_ablations = "ablation_sweep" in spec.stages and spec.ablations.enabled
    run_aggregate = "aggregate_reports" in spec.stages

    @dsl.component(base_image=base_image)
    def validate_spec_component(spec_json: str) -> str:
        from mambasl_new.specs import ExperimentSpec

        payload = ExperimentSpec.model_validate_json(spec_json)
        return payload.metadata.name

    @dsl.component(base_image=base_image)
    def katib_search_component(spec_json: str, dataset: str) -> str:
        from mambasl_new.kubeflow.katib import launch_and_wait
        from mambasl_new.specs import ExperimentSpec

        spec = ExperimentSpec.model_validate_json(spec_json)
        return json.dumps(launch_and_wait(spec, dataset), sort_keys=True)

    @dsl.component(base_image=base_image)
    def final_train_component(spec_json: str, dataset: str, best_params_json: str) -> str:
        from mambasl_new.cmapss.experiment import run_dataset_pipeline
        from mambasl_new.specs import ExperimentSpec

        spec = ExperimentSpec.model_validate_json(spec_json)
        explicit = json.loads(best_params_json) if best_params_json else None
        from pathlib import Path

        out_dir = Path(spec.outputs.local_results_dir) / spec.metadata.name / dataset.lower()
        result = run_dataset_pipeline(
            spec,
            dataset,
            out_dir,
            explicit_params=explicit,
            run_hpo_stage=False,
            run_ablation_stage=False,
        )
        summary = {
            "dataset": dataset,
            "best_params": result["hpo"]["best_params"],
            "final": {
                "best_epoch": result["final"]["best_epoch"],
                "test_metrics": result["final"]["test_metrics"],
                "val_rmse": result["final"]["val_rmse"],
                "val_score": result["final"]["val_score"],
                "val_selection_metric": result["final"]["val_selection_metric"],
            },
            "literature_comparison": result["literature_comparison"],
        }
        return json.dumps(summary, sort_keys=True)

    @dsl.component(base_image=base_image)
    def ablation_component(spec_json: str, dataset: str, best_params_json: str) -> str:
        from mambasl_new.cmapss.experiment import run_ablation_only
        from mambasl_new.specs import ExperimentSpec

        spec = ExperimentSpec.model_validate_json(spec_json)
        from pathlib import Path

        out_dir = Path(spec.outputs.local_results_dir) / spec.metadata.name / dataset.lower()
        results = run_ablation_only(spec, dataset, json.loads(best_params_json), out_dir)
        return json.dumps({"dataset": dataset, "ablations": results}, sort_keys=True)

    @dsl.component(base_image=base_image)
    def aggregate_component(
        spec_json: str,
        fd001_final: str = "",
        fd002_final: str = "",
        fd003_final: str = "",
        fd004_final: str = "",
        fd001_ablation: str = "",
        fd002_ablation: str = "",
        fd003_ablation: str = "",
        fd004_ablation: str = "",
    ) -> str:
        from mambasl_new.specs import ExperimentSpec

        spec = ExperimentSpec.model_validate_json(spec_json)
        finals = [value for value in [fd001_final, fd002_final, fd003_final, fd004_final] if value]
        ablations = [value for value in [fd001_ablation, fd002_ablation, fd003_ablation, fd004_ablation] if value]
        payload = {
            "experiment": spec.metadata.name,
            "datasets": {json.loads(item)["dataset"]: json.loads(item) for item in finals},
            "ablations": {json.loads(item)["dataset"]: json.loads(item)["ablations"] for item in ablations},
        }
        return json.dumps(payload, sort_keys=True)

    @dsl.pipeline(name=spec.metadata.name, pipeline_root=spec.runtime.pipeline_root)
    def pipeline():
        validate_task = _configure_task(validate_spec_component(spec_json=spec_json), spec)
        final_outputs: Dict[str, dsl.PipelineChannel] = {}
        ablation_outputs: Dict[str, dsl.PipelineChannel] = {}

        for dataset in datasets:
            if run_hpo:
                best_params_task = _configure_task(katib_search_component(spec_json=spec_json, dataset=dataset), spec)
                best_params_task.after(validate_task)
                best_params_output = best_params_task.output
            else:
                best_params_output = json.dumps(spec.train_defaults.fixed_params or {})

            final_task = _configure_task(final_train_component(spec_json=spec_json, dataset=dataset, best_params_json=best_params_output), spec)
            if run_hpo:
                final_task.after(best_params_task)
            else:
                final_task.after(validate_task)
            final_outputs[dataset] = final_task.output

            if run_ablations:
                ablation_task = _configure_task(ablation_component(spec_json=spec_json, dataset=dataset, best_params_json=best_params_output), spec)
                ablation_task.after(final_task)
                ablation_outputs[dataset] = ablation_task.output

        if run_aggregate:
            aggregate_component(
                spec_json=spec_json,
                fd001_final=final_outputs.get("FD001", ""),
                fd002_final=final_outputs.get("FD002", ""),
                fd003_final=final_outputs.get("FD003", ""),
                fd004_final=final_outputs.get("FD004", ""),
                fd001_ablation=ablation_outputs.get("FD001", ""),
                fd002_ablation=ablation_outputs.get("FD002", ""),
                fd003_ablation=ablation_outputs.get("FD003", ""),
                fd004_ablation=ablation_outputs.get("FD004", ""),
            )

    return pipeline


def compile_pipeline(spec: ExperimentSpec, output_path: str) -> None:
    compiler.Compiler().compile(build_pipeline(spec), package_path=output_path)
