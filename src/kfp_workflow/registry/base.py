"""Abstract base classes and data models for model and dataset registries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """Metadata returned from model registry lookups."""
    name: str
    version: str
    framework: str = "pytorch"
    description: str = ""
    uri: str = ""
    parameters: Dict[str, Any] = {}


class DatasetInfo(BaseModel):
    """Metadata returned from dataset registry lookups."""
    name: str
    version: str = "v1"
    pvc_name: str
    subpath: str
    description: str = ""


# ---------------------------------------------------------------------------
# Abstract contracts
# ---------------------------------------------------------------------------

class ModelRegistryBase(ABC):
    """Contract for model registry clients."""

    @abstractmethod
    def register_model(
        self,
        name: str,
        version: str,
        uri: str,
        framework: str = "pytorch",
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        ...

    @abstractmethod
    def get_model(self, name: str, version: Optional[str] = None) -> ModelInfo:
        ...

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        ...


class DatasetRegistryBase(ABC):
    """Contract for dataset registry clients."""

    @abstractmethod
    def register_dataset(
        self,
        name: str,
        pvc_name: str,
        subpath: str,
        version: str = "v1",
        description: str = "",
    ) -> DatasetInfo:
        ...

    @abstractmethod
    def get_dataset(self, name: str, version: Optional[str] = None) -> DatasetInfo:
        ...

    @abstractmethod
    def list_datasets(self) -> List[DatasetInfo]:
        ...
