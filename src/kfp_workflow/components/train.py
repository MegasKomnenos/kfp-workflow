"""KFP component: run PyTorch training loop (single-node)."""

from kfp import dsl

BASE_IMAGE = "kfp-workflow:latest"


@dsl.component(base_image=BASE_IMAGE)
def train_component(spec_json: str, preprocess_result_json: str) -> str:
    """Run PyTorch training loop on preprocessed data.

    Parameters
    ----------
    spec_json:
        JSON-serialised ``PipelineSpec``.
    preprocess_result_json:
        JSON output from ``preprocess_component``.

    Returns
    -------
    str
        JSON: ``{"model_path": str, "best_epoch": int,
        "train_loss": float, "val_loss": float}``
    """
    raise NotImplementedError("train_component not yet implemented")
