"""Validate checked-in config fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from kfp_workflow.specs import (
    load_pipeline_spec,
    load_serving_spec,
    load_tune_spec,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize(
    "path",
    sorted((CONFIGS / "pipelines").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_pipeline_configs_validate(path: Path):
    load_pipeline_spec(path)


@pytest.mark.parametrize(
    "path",
    sorted((CONFIGS / "serving").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_serving_configs_validate(path: Path):
    load_serving_spec(path)


@pytest.mark.parametrize(
    "path",
    sorted((CONFIGS / "tuning").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_tuning_configs_validate(path: Path):
    load_tune_spec(path)
