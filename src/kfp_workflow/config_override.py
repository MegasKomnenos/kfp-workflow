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


def _parse_path(dotted_key: str) -> List[Any]:
    """Parse dotted keys with optional list indexes.

    Supported examples:
    - ``model.config.d_model``
    - ``dataset.config.fd[0].fd_name``
    """
    tokens: List[Any] = []
    buf = ""
    index_buf = ""
    in_index = False

    for char in dotted_key:
        if in_index:
            if char == "]":
                if not index_buf:
                    raise ValueError(f"Empty list index in override key: {dotted_key!r}")
                tokens.append(int(index_buf))
                index_buf = ""
                in_index = False
                continue
            if not char.isdigit():
                raise ValueError(f"Invalid list index in override key: {dotted_key!r}")
            index_buf += char
            continue

        if char == ".":
            if buf:
                tokens.append(buf)
                buf = ""
            continue
        if char == "[":
            if buf:
                tokens.append(buf)
                buf = ""
            in_index = True
            continue
        if char == "]":
            raise ValueError(f"Unexpected ']' in override key: {dotted_key!r}")
        buf += char

    if in_index:
        raise ValueError(f"Unclosed list index in override key: {dotted_key!r}")
    if buf:
        tokens.append(buf)
    if not tokens:
        raise ValueError(f"Empty override key: {dotted_key!r}")
    return tokens


def set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested structure using dotted keys and list indexes."""
    tokens = _parse_path(dotted_key)
    current: Any = d

    for index, token in enumerate(tokens[:-1]):
        next_token = tokens[index + 1]
        if isinstance(token, str):
            if not isinstance(current, dict):
                raise ValueError(
                    f"Cannot descend into non-dict at '{token}' while setting {dotted_key!r}"
                )
            if token not in current or not isinstance(current[token], (dict, list)):
                current[token] = [] if isinstance(next_token, int) else {}
            current = current[token]
            continue

        if token < 0:
            raise ValueError(f"Negative list index not allowed in override key: {dotted_key!r}")
        if not isinstance(current, list):
            raise ValueError(
                f"Cannot index non-list with [{token}] while setting {dotted_key!r}"
            )
        while len(current) <= token:
            current.append([] if isinstance(next_token, int) else {})
        current = current[token]

    last = tokens[-1]
    if isinstance(last, str):
        if not isinstance(current, dict):
            raise ValueError(
                f"Cannot assign key '{last}' into non-dict while setting {dotted_key!r}"
            )
        current[last] = value
        return

    if last < 0:
        raise ValueError(f"Negative list index not allowed in override key: {dotted_key!r}")
    if not isinstance(current, list):
        raise ValueError(
            f"Cannot index non-list with [{last}] while setting {dotted_key!r}"
        )
    while len(current) <= last:
        current.append(None)
    current[last] = value


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
