"""KFP component: preprocess raw data into training-ready tensors."""

from kfp import dsl

BASE_IMAGE = "kfp-workflow:latest"


@dsl.component(base_image=BASE_IMAGE)
def preprocess_component(spec_json: str, load_result_json: str) -> str:
    """Transform raw data into training-ready tensors.

    Parameters
    ----------
    spec_json:
        JSON-serialised ``PipelineSpec``.
    load_result_json:
        JSON output from ``load_data_component``.

    Returns
    -------
    str
        JSON: ``{"train_path": str, "val_path": str, "feature_dim": int,
        "num_train": int, "num_val": int}``
    """
    raise NotImplementedError("preprocess_component not yet implemented")
