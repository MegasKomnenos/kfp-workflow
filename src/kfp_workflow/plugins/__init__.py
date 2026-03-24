"""Model plugin registry.

Central explicit dict mapping model name strings to plugin classes.
To add a new model, import its class in ``_build_registry`` and add an entry.
"""

from __future__ import annotations

from typing import Dict, Type

from kfp_workflow.plugins.base import ModelPlugin


_REGISTRY: Dict[str, Type[ModelPlugin]] | None = None


def _build_registry() -> Dict[str, Type[ModelPlugin]]:
    """Build the plugin registry.

    Uses lazy imports so that heavy dependencies (torch, mamba_ssm) are
    not loaded until a plugin is actually requested.
    """
    registry: Dict[str, Type[ModelPlugin]] = {}

    # --- MambaSL C-MAPSS ---
    from kfp_workflow.plugins.mambasl_cmapss import MambaSLCmapssPlugin
    registry[MambaSLCmapssPlugin.name()] = MambaSLCmapssPlugin

    # --- MR-HY-SP C-MAPSS ---
    from kfp_workflow.plugins.mrhysp_cmapss import MRHySPCmapssPlugin
    registry[MRHySPCmapssPlugin.name()] = MRHySPCmapssPlugin

    # --- Add future model plugins here ---
    # from kfp_workflow.plugins.some_other import SomeOtherPlugin
    # registry[SomeOtherPlugin.name()] = SomeOtherPlugin

    return registry


def get_plugin_registry() -> Dict[str, Type[ModelPlugin]]:
    """Return the model plugin registry (built lazily on first call)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_plugin(model_name: str) -> ModelPlugin:
    """Instantiate and return a plugin by its registered name.

    Raises
    ------
    KeyError
        If *model_name* is not in the registry.
    """
    registry = get_plugin_registry()
    if model_name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(
            f"Unknown model plugin '{model_name}'. Available: {available}"
        )
    return registry[model_name]()
