"""KFP component: evaluate trained model on validation/test set."""

from kfp import dsl

BASE_IMAGE = "kfp-workflow:latest"


@dsl.component(base_image=BASE_IMAGE)
def evaluate_component(
    spec_json: str,
    train_result_json: str,
    preprocess_result_json: str,
) -> str:
    """Evaluate model on validation/test set and compute metrics.

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
        JSON: ``{"metrics": {"loss": float, "accuracy": float, ...},
        "model_path": str}``
    """
    raise NotImplementedError("evaluate_component not yet implemented")
