from multirocket_new.search_space import builtin_search_space, katib_parameter_specs
from multirocket_new.specs import execution_spec, expand_ablation_cases, load_spec


def test_load_default_spec():
    spec = load_spec("configs/experiments/fd_all_core_default.yaml")
    assert spec.metadata.name == "multirocket-new-fd-all-default"
    assert spec.datasets.items == ["FD001", "FD002", "FD003", "FD004"]
    assert spec.storage.mode == "pvc"


def test_execution_spec_uses_mount_paths_for_pvc():
    spec = load_spec("configs/experiments/fd_all_core_default.yaml")
    runtime_spec = execution_spec(spec, kubeflow=True)
    assert runtime_spec.data.data_root == "/mnt/data"
    assert runtime_spec.outputs.local_results_dir == "/mnt/results"


def test_expand_ablation_cases():
    spec = load_spec("configs/experiments/fd_all_core_default.yaml")
    cases = expand_ablation_cases(spec)
    assert len(cases) == 4
    assert any("seq_len-40" in case["name"] for case in cases)


def test_katib_parameter_specs():
    params = builtin_search_space("default")
    katib = katib_parameter_specs(params)
    assert any(item["name"] == "mr_num_kernels" for item in katib)
    assert any(item["name"] == "seed" for item in katib)
