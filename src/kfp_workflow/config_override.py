"""CLI config override utilities for Helm-style --set flag support.

Applies dotted-path key=value overrides to a raw spec dict before
Pydantic validation, so CLI values take highest precedence over YAML.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def coerce_value(raw: str) -> Any:
    """Smart type coercion for CLI string values.

    Tries JSON parse first (handles numbers, booleans, lists, dicts, null).
    Falls back to the raw string.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key path.

    Example: ``set_nested(d, "model.config.d_model", 128)``
    creates intermediate dicts as needed.
    """
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def apply_overrides(spec_dict: dict, overrides: List[str]) -> dict:
    """Parse ``key=value`` strings and apply them to a spec dict.

    Returns the modified dict (mutated in place).
    Raises ``ValueError`` on malformed override strings.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Malformed override (expected key=value): {override!r}"
            )
        key, _, raw_value = override.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in override: {override!r}")
        set_nested(spec_dict, key, coerce_value(raw_value))
    return spec_dict


def validate_plugin_config(spec_dict: dict) -> List[str]:
    """Validate model.config and dataset.config against plugin schemas.

    Returns a list of warning strings (empty if all valid or no schemas).
    Non-blocking — schemas that use ``extra="allow"`` will only warn
    about type mismatches, not unknown keys.
    """
    from kfp_workflow.plugins import get_plugin

    warnings: List[str] = []
    model_name = spec_dict.get("model", {}).get("name")
    if not model_name:
        return warnings

    try:
        plugin = get_plugin(model_name)
    except KeyError:
        return warnings

    cls = plugin.__class__

    model_schema = cls.model_config_schema()
    if model_schema is not None:
        try:
            model_schema.model_validate(
                spec_dict.get("model", {}).get("config", {})
            )
        except Exception as exc:
            warnings.append(f"model.config validation: {exc}")

    dataset_schema = cls.dataset_config_schema()
    if dataset_schema is not None:
        try:
            dataset_schema.model_validate(
                spec_dict.get("dataset", {}).get("config", {})
            )
        except Exception as exc:
            warnings.append(f"dataset.config validation: {exc}")

    train_schema = cls.train_config_schema()
    if train_schema is not None:
        try:
            train_schema.model_validate(spec_dict.get("train", {}))
        except Exception as exc:
            warnings.append(f"train validation: {exc}")

    return warnings
