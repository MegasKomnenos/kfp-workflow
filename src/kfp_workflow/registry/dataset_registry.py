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


def resolve_data_mount_path(spec: dict) -> str:
    """Resolve the effective dataset path from a pipeline storage spec.

    Checks explicit ``data_subpath`` first, then the PVC dataset registry,
    then falls back to the bare mount path.

    Parameters
    ----------
    spec:
        Pipeline spec dict with ``storage`` and ``dataset`` sections.

    Returns
    -------
    str
        Absolute path to the dataset directory on the mounted PVC.

    Raises
    ------
    ValueError
        If the registry entry references a different PVC than the one mounted.
    """
    storage = spec.get("storage", {})
    base_mount = Path(storage["data_mount_path"])
    explicit_subpath = str(storage.get("data_subpath", "") or "").strip("/")
    if explicit_subpath:
        return str(base_mount / explicit_subpath)

    registry_path = base_mount / ".dataset_registry.json"
    if not registry_path.exists():
        return str(base_mount)

    registry = PVCDatasetRegistry(registry_path=str(registry_path))
    dataset_ref = spec.get("dataset", {})
    try:
        info = registry.get_dataset(
            name=dataset_ref.get("name", ""),
            version=dataset_ref.get("version"),
        )
    except KeyError:
        return str(base_mount)

    expected_pvc = storage.get("data_pvc", "")
    if info.pvc_name != expected_pvc:
        raise ValueError(
            "Dataset registry entry points to PVC "
            f"'{info.pvc_name}', but the pipeline mounts "
            f"'{expected_pvc}'. Cross-PVC dataset resolution is not "
            "supported by this pipeline path."
        )

    return str(base_mount / info.subpath.strip("/"))
