"""File-backed model registry stored as JSON on the model PVC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from kfp_workflow.registry.base import ModelInfo, ModelRegistryBase


class FileModelRegistry(ModelRegistryBase):
    """JSON-file-backed model registry.

    Stores model metadata at *registry_path*.  Suitable for single-node
    clusters where Kubeflow Model Registry is not deployed.
    """

    def __init__(self, registry_path: str = "/mnt/models/.model_registry.json"):
        self._path = Path(registry_path)

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            return json.loads(self._path.read_text("utf-8"))
        return {"models": []}

    def _save(self, data: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, default=str), "utf-8")

    def register_model(
        self,
        name: str,
        version: str,
        uri: str,
        framework: str = "pytorch",
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        data = self._load()
        entry = ModelInfo(
            name=name,
            version=version,
            framework=framework,
            description=description,
            uri=uri,
            parameters=parameters or {},
        )
        # Upsert: remove existing entry with same name+version
        data["models"] = [
            m for m in data["models"]
            if not (m["name"] == name and m["version"] == version)
        ]
        data["models"].append(entry.model_dump())
        self._save(data)
        return entry

    def get_model(self, name: str, version: Optional[str] = None) -> ModelInfo:
        data = self._load()
        for m in data["models"]:
            if m["name"] == name:
                if version is None or m["version"] == version:
                    return ModelInfo.model_validate(m)
        raise KeyError(f"Model '{name}' (version={version}) not found")

    def list_models(self) -> List[ModelInfo]:
        data = self._load()
        return [ModelInfo.model_validate(m) for m in data["models"]]
