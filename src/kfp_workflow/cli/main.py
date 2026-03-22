"""Typer CLI application for kfp-workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

# ---------------------------------------------------------------------------
# App and sub-apps
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="kfp-workflow",
    help="KFP v2 training pipeline and KServe serving workflow manager.",
    add_completion=False,
)

pipeline_app = typer.Typer(help="Compile and submit training pipelines.")
serve_app = typer.Typer(help="Create and manage KServe InferenceServices.")
registry_app = typer.Typer(help="Register and retrieve models and datasets.")
model_reg_app = typer.Typer(help="Model registry operations.")
dataset_reg_app = typer.Typer(help="Dataset registry operations.")
cluster_app = typer.Typer(help="Cluster bootstrapping operations.")
spec_app = typer.Typer(help="Spec validation.")

app.add_typer(pipeline_app, name="pipeline")
app.add_typer(serve_app, name="serve")
app.add_typer(registry_app, name="registry")
registry_app.add_typer(model_reg_app, name="model")
registry_app.add_typer(dataset_reg_app, name="dataset")
app.add_typer(cluster_app, name="cluster")
app.add_typer(spec_app, name="spec")


# ---------------------------------------------------------------------------
# spec commands
# ---------------------------------------------------------------------------

@spec_app.command("validate")
def cmd_spec_validate(
    spec: Path = typer.Option(..., help="Path to a pipeline or serving YAML spec."),
    spec_type: str = typer.Option(
        "pipeline",
        "--type",
        help="Spec type: 'pipeline' or 'serving'.",
    ),
) -> None:
    """Load and validate a spec file."""
    from kfp_workflow.specs import load_pipeline_spec, load_serving_spec

    if spec_type == "pipeline":
        loaded = load_pipeline_spec(spec)
    elif spec_type == "serving":
        loaded = load_serving_spec(spec)
    else:
        typer.echo(f"Unknown spec type: {spec_type}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Spec '{loaded.metadata.name}' validated successfully.")
    typer.echo(json.dumps(loaded.model_dump(), indent=2, default=str))


# ---------------------------------------------------------------------------
# pipeline commands
# ---------------------------------------------------------------------------

@pipeline_app.command("compile")
def cmd_pipeline_compile(
    spec: Path = typer.Option(..., help="Path to a pipeline YAML spec."),
    output: Path = typer.Option(..., help="Output path for compiled YAML."),
) -> None:
    """Compile a training pipeline to a KFP v2 YAML package."""
    from kfp_workflow.pipeline.compiler import compile_pipeline
    from kfp_workflow.specs import load_pipeline_spec

    loaded = load_pipeline_spec(spec)
    result = compile_pipeline(loaded, output)
    typer.echo(f"Pipeline compiled to {result}")


@pipeline_app.command("submit")
def cmd_pipeline_submit(
    spec: Path = typer.Option(..., help="Path to a pipeline YAML spec."),
    namespace: Optional[str] = typer.Option(None, help="Kubernetes namespace override."),
    host: Optional[str] = typer.Option(None, help="KFP API host override."),
    existing_token: Optional[str] = typer.Option(None, help="Bearer token for auth."),
    cookies: Optional[str] = typer.Option(None, help="Cookie header for auth."),
) -> None:
    """Compile and submit a training pipeline to Kubeflow."""
    from kfp_workflow.pipeline.client import submit_pipeline
    from kfp_workflow.specs import load_pipeline_spec

    loaded = load_pipeline_spec(spec)
    run_id = submit_pipeline(
        loaded,
        namespace=namespace,
        host=host,
        existing_token=existing_token,
        cookies=cookies,
    )
    typer.echo(f"Pipeline submitted. Run ID: {run_id}")


# ---------------------------------------------------------------------------
# serve commands
# ---------------------------------------------------------------------------

@serve_app.command("create")
def cmd_serve_create(
    spec: Path = typer.Option(..., help="Path to a serving YAML spec."),
    dry_run: bool = typer.Option(False, help="Print manifest without applying."),
) -> None:
    """Create a KServe InferenceService from a serving spec."""
    from kfp_workflow.serving.kserve import create_inference_service
    from kfp_workflow.specs import load_serving_spec

    loaded = load_serving_spec(spec)
    result = create_inference_service(
        name=loaded.metadata.name,
        namespace=loaded.namespace,
        model_pvc_name=loaded.model_pvc,
        model_subpath=loaded.model_subpath,
        runtime=loaded.runtime,
        replicas=loaded.replicas,
        resources=loaded.resources.model_dump(),
        dry_run=dry_run,
    )
    typer.echo(json.dumps(result, indent=2))


@serve_app.command("delete")
def cmd_serve_delete(
    name: str = typer.Option(..., help="InferenceService name."),
    namespace: str = typer.Option(
        "kubeflow-user-example-com", help="Kubernetes namespace."
    ),
) -> None:
    """Delete a KServe InferenceService."""
    from kfp_workflow.serving.kserve import delete_inference_service

    delete_inference_service(name=name, namespace=namespace)
    typer.echo(f"InferenceService '{name}' deleted.")


# ---------------------------------------------------------------------------
# registry model commands
# ---------------------------------------------------------------------------

@model_reg_app.command("register")
def cmd_model_register(
    name: str = typer.Option(..., help="Model name."),
    version: str = typer.Option(..., help="Model version."),
    uri: str = typer.Option(..., help="Model artifact URI (PVC subpath)."),
    framework: str = typer.Option("pytorch", help="ML framework."),
    description: str = typer.Option("", help="Model description."),
) -> None:
    """Register a model in the Kubeflow Model Registry."""
    from kfp_workflow.registry.model_registry import KubeflowModelRegistry

    registry = KubeflowModelRegistry()
    info = registry.register_model(
        name=name,
        version=version,
        uri=uri,
        framework=framework,
        description=description,
    )
    typer.echo(json.dumps(info.model_dump(), indent=2))


@model_reg_app.command("get")
def cmd_model_get(
    name: str = typer.Option(..., help="Model name."),
    version: Optional[str] = typer.Option(None, help="Model version."),
) -> None:
    """Retrieve a model from the Kubeflow Model Registry."""
    from kfp_workflow.registry.model_registry import KubeflowModelRegistry

    registry = KubeflowModelRegistry()
    info = registry.get_model(name=name, version=version)
    typer.echo(json.dumps(info.model_dump(), indent=2))


@model_reg_app.command("list")
def cmd_model_list() -> None:
    """List all models in the Kubeflow Model Registry."""
    from kfp_workflow.registry.model_registry import KubeflowModelRegistry

    registry = KubeflowModelRegistry()
    models = registry.list_models()
    for m in models:
        typer.echo(f"{m.name} v{m.version} ({m.framework})")


# ---------------------------------------------------------------------------
# registry dataset commands
# ---------------------------------------------------------------------------

@dataset_reg_app.command("register")
def cmd_dataset_register(
    name: str = typer.Option(..., help="Dataset name."),
    pvc_name: str = typer.Option(..., help="PVC name."),
    subpath: str = typer.Option(..., help="Path within the PVC."),
    version: str = typer.Option("v1", help="Dataset version."),
    description: str = typer.Option("", help="Dataset description."),
) -> None:
    """Register a dataset in the PVC dataset registry."""
    from kfp_workflow.registry.dataset_registry import PVCDatasetRegistry

    registry = PVCDatasetRegistry()
    info = registry.register_dataset(
        name=name,
        pvc_name=pvc_name,
        subpath=subpath,
        version=version,
        description=description,
    )
    typer.echo(json.dumps(info.model_dump(), indent=2))


@dataset_reg_app.command("get")
def cmd_dataset_get(
    name: str = typer.Option(..., help="Dataset name."),
    version: Optional[str] = typer.Option(None, help="Dataset version."),
) -> None:
    """Retrieve a dataset from the PVC dataset registry."""
    from kfp_workflow.registry.dataset_registry import PVCDatasetRegistry

    registry = PVCDatasetRegistry()
    info = registry.get_dataset(name=name, version=version)
    typer.echo(json.dumps(info.model_dump(), indent=2))


@dataset_reg_app.command("list")
def cmd_dataset_list() -> None:
    """List all datasets in the PVC dataset registry."""
    from kfp_workflow.registry.dataset_registry import PVCDatasetRegistry

    registry = PVCDatasetRegistry()
    datasets = registry.list_datasets()
    for d in datasets:
        typer.echo(f"{d.name} v{d.version} @ {d.pvc_name}:{d.subpath}")


# ---------------------------------------------------------------------------
# cluster commands
# ---------------------------------------------------------------------------

@cluster_app.command("bootstrap")
def cmd_cluster_bootstrap(
    spec: Path = typer.Option(..., help="Path to a pipeline YAML spec."),
    dry_run: bool = typer.Option(False, help="Print manifests without applying."),
) -> None:
    """Create PVCs for data and model storage."""
    from kfp_workflow.specs import load_pipeline_spec

    loaded = load_pipeline_spec(spec)
    storage = loaded.storage

    data_pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": storage.data_pvc,
            "namespace": loaded.runtime.namespace,
        },
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "storageClassName": storage.storage_class,
            "resources": {"requests": {"storage": storage.data_size}},
        },
    }
    model_pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": storage.model_pvc,
            "namespace": loaded.runtime.namespace,
        },
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "storageClassName": storage.storage_class,
            "resources": {"requests": {"storage": storage.model_size}},
        },
    }

    typer.echo(json.dumps([data_pvc, model_pvc], indent=2))

    if dry_run:
        typer.echo("Dry run — manifests printed above, nothing applied.")
        return

    raise NotImplementedError(
        "Cluster bootstrapping (kubectl apply) not yet implemented."
    )


# ---------------------------------------------------------------------------
# Entry point for `python -m kfp_workflow.cli.main`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
