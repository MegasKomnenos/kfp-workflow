"""PVC-based dataset registry implementation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from kfp_workflow.registry.base import DatasetInfo, DatasetRegistryBase


class PVCDatasetRegistry(DatasetRegistryBase):
    """Simple file-backed registry mapping dataset names to PVC locations.

    Stores metadata in a JSON file on the PVC itself at
    ``<mount_path>/.registry.json``.  All methods are currently stubs.
    """

    def __init__(self, registry_path: str = "/mnt/data/.registry.json") -> None:
        self._registry_path = Path(registry_path)

    def register_dataset(
        self,
        name: str,
        pvc_name: str,
        subpath: str,
        version: str = "v1",
        description: str = "",
    ) -> DatasetInfo:
        raise NotImplementedError(
            "PVC dataset registration not yet implemented"
        )

    def get_dataset(self, name: str, version: Optional[str] = None) -> DatasetInfo:
        raise NotImplementedError(
            "PVC dataset retrieval not yet implemented"
        )

    def list_datasets(self) -> List[DatasetInfo]:
        raise NotImplementedError(
            "PVC dataset listing not yet implemented"
        )
