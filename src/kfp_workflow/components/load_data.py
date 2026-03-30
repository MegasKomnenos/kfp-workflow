"""KFP component: load raw data from PVC via model plugin."""

from kfp import dsl

BASE_IMAGE = "kfp-workflow:latest"


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
    from kfp_workflow.registry.dataset_registry import resolve_data_mount_path

    spec = json.loads(spec_json)
    data_mount_path = resolve_data_mount_path(spec)

    plugin = get_plugin(spec["model"]["name"])
    result = plugin.load_data(
        spec=spec,
        data_mount_path=data_mount_path,
    )
    return json.dumps(result_to_dict(result))
