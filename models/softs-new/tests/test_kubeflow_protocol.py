"""Tests for Kubeflow PVC-mode path substitution and Katib manifest generation."""

from pathlib import Path
import pytest

SMOKE_SPEC = Path(__file__).parent.parent / "configs" / "experiments" / "fd001_smoke.yaml"
PVC_SPEC = Path(__file__).parent.parent / "configs" / "experiments" / "fd_all_core_default.yaml"


# ---------------------------------------------------------------------------
# PVC path substitution (execution_spec)
# ---------------------------------------------------------------------------

def test_execution_spec_pvc_substitutes_data_root():
    """In PVC mode, data_root must be replaced with data_mount_path."""
    from softs_new.specs import execution_spec, load_spec

    spec = load_spec(PVC_SPEC)
    assert spec.storage.mode == "pvc"

    exec_spec = execution_spec(spec, kubeflow=True)
    assert exec_spec.data.data_root == spec.storage.data_mount_path


def test_execution_spec_pvc_substitutes_results_dir():
    """In PVC mode, local_results_dir must be replaced with results_mount_path."""
    from softs_new.specs import execution_spec, load_spec

    spec = load_spec(PVC_SPEC)
    exec_spec = execution_spec(spec, kubeflow=True)
    assert exec_spec.outputs.local_results_dir == spec.storage.results_mount_path


def test_execution_spec_no_kubeflow_leaves_paths_unchanged():
    """Non-Kubeflow execution must leave paths unchanged."""
    from softs_new.specs import execution_spec, load_spec

    spec = load_spec(PVC_SPEC)
    exec_spec = execution_spec(spec, kubeflow=False)
    assert exec_spec.data.data_root == spec.data.data_root
    assert exec_spec.outputs.local_results_dir == spec.outputs.local_results_dir


def test_execution_spec_download_mode_skips_substitution():
    """Download-mode spec must not have paths substituted even in Kubeflow mode."""
    from softs_new.specs import execution_spec, load_spec

    spec = load_spec(SMOKE_SPEC)
    assert spec.storage.mode == "download"
    exec_spec = execution_spec(spec, kubeflow=True)
    # data_root should remain as specified (not replaced by mount path)
    assert exec_spec.data.data_root == spec.data.data_root


# ---------------------------------------------------------------------------
# Katib manifest rendering
# ---------------------------------------------------------------------------

def test_katib_render_contains_algorithm():
    """Rendered Katib manifest must include the configured algorithm."""
    from softs_new.specs import load_spec
    from softs_new.kubeflow.katib import render_katib_experiment

    spec = load_spec(PVC_SPEC)
    manifest = render_katib_experiment(spec, dataset="FD001")
    assert "algorithm" in str(manifest).lower() or "AlgorithmName" in str(manifest), (
        "Katib manifest must declare the search algorithm"
    )


def test_katib_render_contains_d_core_parameter():
    """Rendered Katib experiment manifest must include d_core in parameters."""
    from softs_new.specs import load_spec
    from softs_new.kubeflow.katib import render_katib_experiment

    spec = load_spec(PVC_SPEC)
    manifest_str = str(render_katib_experiment(spec, dataset="FD001"))
    assert "d_core" in manifest_str


def test_katib_render_trial_count():
    """Rendered Katib manifest must reflect max_trial_count from the spec."""
    from softs_new.specs import load_spec
    from softs_new.kubeflow.katib import render_katib_experiment

    spec = load_spec(PVC_SPEC)
    manifest_str = str(render_katib_experiment(spec, dataset="FD001"))
    assert str(spec.hpo.max_trial_count) in manifest_str
