"""Shared helpers for locating C-MAPSS data on mounted PVCs."""

from __future__ import annotations

from pathlib import Path


def cmapss_storage_root(data_mount_path: str) -> Path:
    """Return the directory under which C-MAPSS should be stored/downloaded."""
    base = Path(data_mount_path)
    if base.name == "CMAPSSData":
        return base.parent
    if base.name == "cmapss":
        return base
    return base / "cmapss"


def resolve_cmapss_data_dir(data_mount_path: str) -> Path:
    """Return the directory containing extracted C-MAPSS text files."""
    base = Path(data_mount_path)
    candidates = [
        base,
        base / "CMAPSSData",
        base / "cmapss",
        base / "cmapss" / "CMAPSSData",
    ]
    for candidate in candidates:
        if (candidate / "train_FD001.txt").exists():
            return candidate
    return cmapss_storage_root(data_mount_path)
