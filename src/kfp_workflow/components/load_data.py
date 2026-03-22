"""KFP component: load raw data from PVC via dataset registry."""

from kfp import dsl

BASE_IMAGE = "kfp-workflow:latest"


@dsl.component(base_image=BASE_IMAGE)
def load_data_component(spec_json: str) -> str:
    """Resolve dataset from registry and verify data exists on PVC.

    Parameters
    ----------
    spec_json:
        JSON-serialised ``PipelineSpec``.

    Returns
    -------
    str
        JSON: ``{"data_path": str, "dataset_name": str, "num_samples": int}``
    """
    raise NotImplementedError("load_data_component not yet implemented")
