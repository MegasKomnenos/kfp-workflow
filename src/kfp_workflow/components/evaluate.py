"""KFP component: evaluate trained model via model plugin."""

from kfp import dsl

BASE_IMAGE = "kfp-workflow:latest"


@dsl.component(base_image=BASE_IMAGE)
def evaluate_component(
    spec_json: str,
    train_result_json: str,
    preprocess_result_json: str,
) -> str:
    """Evaluate model on test set and compute metrics via model plugin.

    Parameters
    ----------
    spec_json:
        JSON-serialised ``PipelineSpec``.
    train_result_json:
        JSON output from ``train_component``.
    preprocess_result_json:
        JSON output from ``preprocess_component`` (for data paths).

    Returns
    -------
    str
        JSON with evaluation metrics and model path.
    """
    import json
    from kfp_workflow.plugins import get_plugin
    from kfp_workflow.plugins.base import (
        PreprocessResult,
        TrainResult,
        result_to_dict,
    )

    spec = json.loads(spec_json)
    train_result = TrainResult(**json.loads(train_result_json))
    preprocess_result = PreprocessResult(**json.loads(preprocess_result_json))

    plugin = get_plugin(spec["model"]["name"])
    result = plugin.evaluate(
        spec=spec,
        train_result=train_result,
        preprocess_result=preprocess_result,
    )
    return json.dumps(result_to_dict(result))
