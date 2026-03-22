"""PVC-based dataset registry stored as JSON on the data PVC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from kfp_workflow.registry.base import DatasetInfo, DatasetRegistryBase


class PVCDatasetRegistry(DatasetRegistryBase):
    """Simple file-backed registry mapping dataset names to PVC locations.

    Stores metadata in a JSON file on the PVC itself at
    ``<mount_path>/.dataset_registry.json``.
    """

    def __init__(self, registry_path: str = "/mnt/data/.dataset_registry.json") -> None:
        self._path = Path(registry_path)

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            return json.loads(self._path.read_text("utf-8"))
        return {"datasets": []}

    def _save(self, data: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, default=str), "utf-8")

    def register_dataset(
        self,
        name: str,
        pvc_name: str,
        subpath: str,
        version: str = "v1",
        description: str = "",
    ) -> DatasetInfo:
        data = self._load()
        entry = DatasetInfo(
            name=name,
            version=version,
            pvc_name=pvc_name,
            subpath=subpath,
            description=description,
        )
        # Upsert: remove existing entry with same name+version
        data["datasets"] = [
            d for d in data["datasets"]
            if not (d["name"] == name and d["version"] == version)
        ]
        data["datasets"].append(entry.model_dump())
        self._save(data)
        return entry

    def get_dataset(self, name: str, version: Optional[str] = None) -> DatasetInfo:
        data = self._load()
        for d in data["datasets"]:
            if d["name"] == name:
                if version is None or d["version"] == version:
                    return DatasetInfo.model_validate(d)
        raise KeyError(f"Dataset '{name}' (version={version}) not found")

    def list_datasets(self) -> List[DatasetInfo]:
        data = self._load()
        return [DatasetInfo.model_validate(d) for d in data["datasets"]]
