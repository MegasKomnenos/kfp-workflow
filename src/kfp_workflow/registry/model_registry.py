"""Kubeflow Model Registry client implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from kfp_workflow.registry.base import ModelInfo, ModelRegistryBase


class KubeflowModelRegistry(ModelRegistryBase):
    """Client for the Kubeflow Model Registry REST API.

    All methods are currently stubs. Implement by calling the Model Registry
    REST endpoints (typically at ``http://model-registry-service:8080``).
    """

    def __init__(
        self,
        host: str = "http://model-registry-service:8080",
        namespace: str = "kubeflow",
    ) -> None:
        self._host = host
        self._namespace = namespace

    def register_model(
        self,
        name: str,
        version: str,
        uri: str,
        framework: str = "pytorch",
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        raise NotImplementedError(
            "Kubeflow Model Registry registration not yet implemented"
        )

    def get_model(self, name: str, version: Optional[str] = None) -> ModelInfo:
        raise NotImplementedError(
            "Kubeflow Model Registry retrieval not yet implemented"
        )

    def list_models(self) -> List[ModelInfo]:
        raise NotImplementedError(
            "Kubeflow Model Registry listing not yet implemented"
        )
