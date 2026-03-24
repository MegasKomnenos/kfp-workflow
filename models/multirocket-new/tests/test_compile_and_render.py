from pathlib import Path

from multirocket_new.kubeflow.bootstrap import bootstrap_storage
from multirocket_new.kubeflow.katib import build_experiment_manifest
from multirocket_new.kubeflow.pipeline import compile_pipeline
from multirocket_new.specs import execution_spec, load_spec


def test_compile_pipeline(tmp_path: Path):
    spec = load_spec("configs/experiments/fd001_smoke.yaml")
    out = tmp_path / "pipeline.yaml"
    compile_pipeline(spec, str(out))
    assert out.exists()
    content = out.read_text()
    assert "PIPELINE DEFINITION" in content
    assert spec.metadata.name in content


def test_render_katib_manifest_uses_shared_protocol():
    spec = execution_spec(load_spec("configs/experiments/fd_all_core_default.yaml"), kubeflow=True)
    manifest = build_experiment_manifest(spec, "FD001")
    container = manifest["spec"]["trialTemplate"]["trialSpec"]["spec"]["template"]["spec"]["containers"][0]
    assert manifest["kind"] == "Experiment"
    assert manifest["spec"]["trialTemplate"]["trialParameters"][0]["name"] == "mr_num_kernels"
    assert container["volumeMounts"][0]["mountPath"] == "/mnt/data"


def test_bootstrap_storage_dry_run_returns_pvc_manifests():
    spec = load_spec("configs/experiments/fd_all_core_default.yaml")
    manifests = bootstrap_storage(spec, dry_run=True)
    assert len(manifests) == 2
    assert manifests[0]["kind"] == "PersistentVolumeClaim"
