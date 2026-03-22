"""KFP component: run model training via model plugin."""

from kfp import dsl

BASE_IMAGE = "kfp-workflow:latest"


@dsl.component(base_image=BASE_IMAGE)
def train_component(spec_json: str, preprocess_result_json: str) -> str:
    """Build and train model using the model plugin.

    Parameters
    ----------
    spec_json:
        JSON-serialised ``PipelineSpec``.
    preprocess_result_json:
        JSON output from ``preprocess_component``.

    Returns
    -------
    str
        JSON with train result fields (model_path, metrics, metadata).
    """
    import json
    from kfp_workflow.plugins import get_plugin
    from kfp_workflow.plugins.base import PreprocessResult, result_to_dict

    spec = json.loads(spec_json)
    pp_raw = json.loads(preprocess_result_json)
    preprocess_result = PreprocessResult(**pp_raw)

    plugin = get_plugin(spec["model"]["name"])
    model_name = spec["model"]["name"]
    model_version = spec["model"].get("version", "v1")
    model_dir = (
        f"{spec['storage']['model_mount_path']}/{model_name}/{model_version}"
    )

    result = plugin.train(
        spec=spec,
        preprocess_result=preprocess_result,
        model_dir=model_dir,
    )
    return json.dumps(result_to_dict(result))
