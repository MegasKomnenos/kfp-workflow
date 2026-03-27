from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import yaml


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2))


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def b64_encode_text(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def b64_decode_text(text: str) -> str:
    return base64.b64decode(text.encode("utf-8")).decode("utf-8")
