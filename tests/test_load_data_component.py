"""Tests for dataset mount-path resolution in load-data component."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kfp_workflow.components.load_data import _resolve_data_mount_path_from_spec


def _base_spec(tmp_path: Path) -> dict:
    return {
        "storage": {
            "data_mount_path": str(tmp_path),
            "data_pvc": "dataset-store",
            "data_subpath": "",
        },
        "dataset": {
            "name": "cmapss",
            "version": "v1",
        },
    }


def test_explicit_data_subpath_wins(tmp_path: Path):
    spec = _base_spec(tmp_path)
    spec["storage"]["data_subpath"] = "seeded/cmapss"

    resolved = _resolve_data_mount_path_from_spec(spec)
    assert resolved == str(tmp_path / "seeded" / "cmapss")


def test_registry_subpath_used_when_present(tmp_path: Path):
    spec = _base_spec(tmp_path)
    registry_path = tmp_path / ".dataset_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "cmapss",
                        "version": "v1",
                        "pvc_name": "dataset-store",
                        "subpath": "cmapss/CMAPSSData",
                        "description": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_data_mount_path_from_spec(spec)
    assert resolved == str(tmp_path / "cmapss" / "CMAPSSData")


def test_registry_pvc_mismatch_fails(tmp_path: Path):
    spec = _base_spec(tmp_path)
    registry_path = tmp_path / ".dataset_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "cmapss",
                        "version": "v1",
                        "pvc_name": "other-pvc",
                        "subpath": "cmapss/CMAPSSData",
                        "description": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Cross-PVC dataset resolution"):
        _resolve_data_mount_path_from_spec(spec)
