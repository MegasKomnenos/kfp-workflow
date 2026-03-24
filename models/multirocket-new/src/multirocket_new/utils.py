from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text()) or {}
