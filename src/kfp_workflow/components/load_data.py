"""KFP component: load raw data from PVC via model plugin."""

from pathlib import Path

from kfp import dsl

BASE_IMAGE = "kfp-workflow:latest"


def _resolve_data_mount_path_from_spec(spec: dict) -> str:
    """Resolve the effective dataset path from storage config and registry."""
    from kfp_workflow.registry.dataset_registry import PVCDatasetRegistry

    storage = spec.get("storage", {})
    base_mount = Path(storage["data_mount_path"])
    explicit_subpath = str(storage.get("data_subpath", "") or "").strip("/")
    if explicit_subpath:
        return str(base_mount / explicit_subpath)

    registry_path = base_mount / ".dataset_registry.json"
    if not registry_path.exists():
        return str(base_mount)

    registry = PVCDatasetRegistry(registry_path=str(registry_path))
    dataset_ref = spec.get("dataset", {})
    try:
        info = registry.get_dataset(
            name=dataset_ref.get("name", ""),
            version=dataset_ref.get("version"),
        )
    except KeyError:
        return str(base_mount)

    expected_pvc = storage.get("data_pvc", "")
    if info.pvc_name != expected_pvc:
        raise ValueError(
            "Dataset registry entry points to PVC "
            f"'{info.pvc_name}', but the pipeline mounts "
            f"'{expected_pvc}'. Cross-PVC dataset resolution is not "
            "supported by this pipeline path."
        )

    return str(base_mount / info.subpath.strip("/"))


@dsl.component(base_image=BASE_IMAGE)
def load_data_component(spec_json: str) -> str:
    """Resolve dataset and load raw data using the model plugin.

    Parameters
    ----------
    spec_json:
        JSON-serialised ``PipelineSpec``.

    Returns
    -------
    str
        JSON with load result fields (data_dir, dataset_name, counts, metadata).
    """
    import json
    from kfp_workflow.plugins import get_plugin
    from kfp_workflow.plugins.base import result_to_dict

    spec = json.loads(spec_json)
    data_mount_path = _resolve_data_mount_path_from_spec(spec)

    plugin = get_plugin(spec["model"]["name"])
    result = plugin.load_data(
        spec=spec,
        data_mount_path=data_mount_path,
    )
    return json.dumps(result_to_dict(result))
