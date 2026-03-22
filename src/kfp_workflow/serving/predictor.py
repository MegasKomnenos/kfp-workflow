"""Custom KServe predictor that delegates to the model plugin system.

Reads env vars:
- ``MODEL_PLUGIN_NAME``: model name for plugin lookup (e.g. ``mambasl-cmapss``)
- ``MODEL_DIR``: path to model directory containing ``model.pt`` + ``model_config.json``
- ``MODEL_NAME``: KServe model name for registration

Entrypoint: ``python -m kfp_workflow.serving.predictor``
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import kserve
import numpy as np


class PluginPredictor(kserve.Model):
    """KServe custom predictor backed by a ``ModelPlugin``."""

    def __init__(self, name: str, model_dir: str, plugin_name: str) -> None:
        super().__init__(name)
        self._model_dir = Path(model_dir)
        self._plugin_name = plugin_name
        self._plugin = None
        self._model_config: dict = {}
        self._model_path: str = ""

    def load(self) -> None:
        """Load model config and prepare plugin for inference."""
        from kfp_workflow.plugins import get_plugin

        self._plugin = get_plugin(self._plugin_name)

        config_path = self._model_dir / "model_config.json"
        if config_path.exists():
            self._model_config = json.loads(config_path.read_text("utf-8"))

        model_path = self._model_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self._model_path = str(model_path)

        self.ready = True

    def predict(
        self,
        payload: dict,
        headers: dict | None = None,
    ) -> dict:
        """Run inference on input data via the plugin's predict method.

        Expects payload in the KServe V1 format::

            {"instances": [[[...], [...], ...], ...]}

        where ``instances`` is a list of 2-D windows (seq_len x features).
        """
        instances = payload.get("instances", [])
        input_array = np.array(instances, dtype=np.float32)

        predictions = self._plugin.predict(
            model_path=self._model_path,
            input_data=input_array,
            model_config=self._model_config,
        )

        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()

        return {"predictions": predictions}


if __name__ == "__main__":
    plugin_name = os.environ.get("MODEL_PLUGIN_NAME", "")
    model_dir = os.environ.get("MODEL_DIR", "/mnt/models")
    model_name = os.environ.get("MODEL_NAME", "model")

    if not plugin_name:
        raise ValueError("MODEL_PLUGIN_NAME environment variable is required")

    predictor = PluginPredictor(
        name=model_name,
        model_dir=model_dir,
        plugin_name=plugin_name,
    )
    predictor.load()
    kserve.ModelServer().start([predictor])
