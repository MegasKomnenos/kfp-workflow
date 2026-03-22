"""KFP component: save model weights to PVC and optionally register."""

from kfp import dsl

BASE_IMAGE = "kfp-workflow:latest"


@dsl.component(base_image=BASE_IMAGE)
def save_model_component(
    spec_json: str,
    train_result_json: str,
    eval_result_json: str,
) -> str:
    """Save model weights to PVC. Optionally register in model registry.

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
        JSON: ``{"saved_path": str, "model_name": str, "model_version": str}``
    """
    raise NotImplementedError("save_model_component not yet implemented")
