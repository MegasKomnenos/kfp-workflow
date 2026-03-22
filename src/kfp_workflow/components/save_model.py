"""KFP component: save model weights and register via model plugin."""

from kfp import dsl

BASE_IMAGE = "kfp-workflow:latest"


@dsl.component(base_image=BASE_IMAGE)
def save_model_component(
    spec_json: str,
    train_result_json: str,
    eval_result_json: str,
) -> str:
    """Save model weights to PVC and register in model registry.

    Parameters
    ----------
    spec_json:
        JSON-serialised ``PipelineSpec``.
    train_result_json:
        JSON output from ``train_component``.
    eval_result_json:
        JSON output from ``evaluate_component``.

    Returns
    -------
    str
        JSON with saved path, model name, and version.
    """
    import json
    from kfp_workflow.plugins import get_plugin
    from kfp_workflow.plugins.base import (
        EvalResult,
        TrainResult,
        result_to_dict,
    )

    spec = json.loads(spec_json)
    train_result = TrainResult(**json.loads(train_result_json))
    eval_result = EvalResult(**json.loads(eval_result_json))

    plugin = get_plugin(spec["model"]["name"])
    model_name = spec["model"]["name"]
    model_version = spec["model"].get("version", "v1")
    final_model_dir = (
        f"{spec['storage']['model_mount_path']}/{model_name}/{model_version}"
    )

    result = plugin.save_model(
        spec=spec,
        train_result=train_result,
        eval_result=eval_result,
        final_model_dir=final_model_dir,
    )
    return json.dumps(result_to_dict(result))
