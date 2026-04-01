from pathlib import Path

from timemixer_new.kubeflow.pipeline import compile_pipeline
from timemixer_new.specs import load_spec


def test_pipeline_compile(tmp_path: Path):
    spec = load_spec("configs/experiments/fd001_smoke.yaml")
    out = tmp_path / "pipeline.yaml"
    compile_pipeline(spec, str(out))
    text = out.read_text()
    assert "PIPELINE DEFINITION" in text
    assert spec.metadata.name in text
