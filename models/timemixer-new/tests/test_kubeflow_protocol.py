from timemixer_new.kubeflow.katib import build_experiment_manifest
from timemixer_new.specs import execution_spec, load_spec


def test_execution_spec_uses_mount_paths_for_pvc():
    spec = load_spec("configs/experiments/fd_all_core_default.yaml")
    runtime_spec = execution_spec(spec, kubeflow=True)
    assert runtime_spec.data.data_root == "/mnt/data"
    assert runtime_spec.outputs.local_results_dir == "/mnt/results"


def test_render_katib_manifest_for_pvc_spec_mounts_storage():
    spec = load_spec("configs/experiments/fd_all_core_default.yaml")
    manifest = build_experiment_manifest(spec, "FD001")
    container = manifest["spec"]["trialTemplate"]["trialSpec"]["spec"]["template"]["spec"]["containers"][0]
    assert container["volumeMounts"][0]["mountPath"] == "/mnt/data"
    assert manifest["metadata"]["namespace"] == "kubeflow-user-example-com"
