"""Pure utility functions: YAML/JSON I/O and path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dump_json(data: Any, path: Path) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def dump_yaml(data: Any, path: Path) -> None:
    """Write *data* as YAML to *path*."""
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


def ensure_parent(path: Path) -> None:
    """Create parent directories of *path* if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
