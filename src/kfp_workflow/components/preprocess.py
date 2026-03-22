"""KFP component: preprocess raw data into training-ready arrays."""

from kfp import dsl

BASE_IMAGE = "kfp-workflow:latest"


@dsl.component(base_image=BASE_IMAGE)
def preprocess_component(spec_json: str, load_result_json: str) -> str:
    """Transform raw data into training-ready numpy arrays via model plugin.

    Parameters
    ----------
    spec_json:
        JSON-serialised ``PipelineSpec``.
    load_result_json:
        JSON output from ``load_data_component``.

    Returns
    -------
    str
        JSON with preprocess result fields (array paths, shapes, metadata).
    """
    import json
    from kfp_workflow.plugins import get_plugin
    from kfp_workflow.plugins.base import LoadDataResult, result_to_dict

    spec = json.loads(spec_json)
    load_raw = json.loads(load_result_json)
    load_result = LoadDataResult(**load_raw)

    plugin = get_plugin(spec["model"]["name"])
    model_name = spec["model"]["name"]
    model_version = spec["model"].get("version", "v1")
    artifacts_dir = (
        f"{spec['storage']['model_mount_path']}/artifacts/{model_name}/{model_version}"
    )

    result = plugin.preprocess(
        spec=spec,
        load_result=load_result,
        artifacts_dir=artifacts_dir,
    )
    return json.dumps(result_to_dict(result))
