"""Tests for pipeline compilation."""

import tempfile
from pathlib import Path

from kfp_workflow.pipeline.compiler import compile_pipeline
from kfp_workflow.specs import load_pipeline_spec

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_compile_produces_yaml():
    spec = load_pipeline_spec(CONFIGS / "pipelines" / "sample_train.yaml")
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        output = compile_pipeline(spec, tmp.name)

    content = output.read_text()
    assert "pipelineSpec" in content or "PIPELINE" in content.upper()
    assert spec.metadata.name in content
