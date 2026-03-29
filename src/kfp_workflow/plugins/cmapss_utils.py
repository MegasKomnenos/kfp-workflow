"""Shared helpers for locating and normalizing C-MAPSS data/configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field


_LEGACY_CMAPSS_KEYS = {"fd_name", "unit_ids", "max_sections"}


class CmapssFdEntry(BaseModel):
    """One FD selection entry shared by pipeline, tune, and benchmark flows."""

    fd_name: str = Field(min_length=1)
    unit_ids: Optional[List[int]] = None
    max_sections: Optional[int] = Field(default=None, gt=0)

    model_config = ConfigDict(extra="forbid")


class CmapssDatasetSelection(BaseModel):
    """Root selector shared by all C-MAPSS dataset consumers."""

    fd: List[CmapssFdEntry] = Field(min_length=1)

    # Dataset consumers keep their own extra root-level options such as
    # feature/scaling/download settings.
    model_config = ConfigDict(extra="allow")


def normalize_cmapss_fd_entries(
    config: Dict[str, Any],
    *,
    context: str,
) -> List[Dict[str, Any]]:
    """Validate and normalize the canonical ``fd[]`` selector list."""
    legacy_keys = sorted(_LEGACY_CMAPSS_KEYS.intersection(config))
    if legacy_keys:
        joined = ", ".join(legacy_keys)
        raise ValueError(
            f"{context} uses legacy C-MAPSS fields ({joined}). "
            "Use dataset.config.fd with a list of {fd_name, unit_ids?, max_sections?} entries."
        )

    validated = CmapssDatasetSelection.model_validate(config)
    normalized: List[Dict[str, Any]] = []
    for entry in validated.fd:
        unit_ids = [int(uid) for uid in entry.unit_ids] if entry.unit_ids else None
        normalized.append(
            {
                "fd_name": entry.fd_name,
                "unit_ids": unit_ids,
                "max_sections": int(entry.max_sections) if entry.max_sections is not None else None,
            }
        )
    return normalized


def cmapss_fd_signature(entries: Sequence[Dict[str, Any]]) -> tuple:
    """Return a hashable signature for the normalized FD plan."""
    return tuple(
        (
            str(entry["fd_name"]),
            tuple(int(uid) for uid in (entry.get("unit_ids") or [])),
            None if entry.get("max_sections") is None else int(entry["max_sections"]),
        )
        for entry in entries
    )


def cmapss_fd_summary(entries: Sequence[Dict[str, Any]]) -> str:
    """Return a short human-readable summary of the selected FDs."""
    labels = [str(entry["fd_name"]) for entry in entries]
    if not labels:
        return "cmapss"
    if len(labels) == 1:
        return labels[0]
    return "+".join(labels)


def filter_cmapss_unit_ids(
    unit_ids: Iterable[int],
    requested_unit_ids: Sequence[int] | None,
) -> List[int]:
    """Filter a unit collection while preserving sorted deterministic order."""
    available = sorted(int(uid) for uid in unit_ids)
    if not requested_unit_ids:
        return available

    requested = {int(uid) for uid in requested_unit_ids}
    return [uid for uid in available if uid in requested]


def cap_array_splits(*arrays: Any, max_sections: int | None) -> tuple:
    """Cap split arrays deterministically for smoke-style dataset reduction."""
    if max_sections is None:
        return tuple(arrays)
    return tuple(arr[:max_sections] for arr in arrays)


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
