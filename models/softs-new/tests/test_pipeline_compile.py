"""Tests for KFP pipeline compilation from experiment specs."""

from pathlib import Path
import pytest

SMOKE_SPEC = Path(__file__).parent.parent / "configs" / "experiments" / "fd001_smoke.yaml"


def test_pipeline_compiles_without_error(tmp_path):
    """Pipeline compile should produce a YAML file with no exceptions."""
    from softs_new.specs import load_spec
    from softs_new.kubeflow.pipeline import compile_pipeline

    spec = load_spec(SMOKE_SPEC)
    out_path = tmp_path / "smoke_pipeline.yaml"
    compile_pipeline(spec, output_path=str(out_path))
    assert out_path.exists(), "compile_pipeline must write an output file"
    assert out_path.stat().st_size > 0, "compiled pipeline YAML must be non-empty"


def test_compiled_pipeline_is_valid_yaml(tmp_path):
    """Compiled pipeline must be parseable YAML."""
    import yaml
    from softs_new.specs import load_spec
    from softs_new.kubeflow.pipeline import compile_pipeline

    spec = load_spec(SMOKE_SPEC)
    out_path = tmp_path / "smoke_pipeline.yaml"
    compile_pipeline(spec, output_path=str(out_path))

    with open(out_path) as f:
        doc = yaml.safe_load(f)
    assert isinstance(doc, dict), "compiled pipeline YAML must be a dict"


def test_compiled_pipeline_references_softs_image(tmp_path):
    """Compiled pipeline YAML must reference the softs-new image."""
    import yaml
    from softs_new.specs import load_spec
    from softs_new.kubeflow.pipeline import compile_pipeline

    spec = load_spec(SMOKE_SPEC)
    out_path = tmp_path / "smoke_pipeline.yaml"
    compile_pipeline(spec, output_path=str(out_path))

    raw = out_path.read_text()
    assert "softs-new" in raw, "pipeline YAML must reference softs-new image"
