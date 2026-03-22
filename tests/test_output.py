"""Unit tests for CLI output formatting module."""

from kfp_workflow.cli.output import style_isvc_ready, style_run_state


def test_style_run_state_succeeded():
    styled = style_run_state("SUCCEEDED")
    assert "green" in styled
    assert "SUCCEEDED" in styled


def test_style_run_state_failed():
    styled = style_run_state("FAILED")
    assert "red" in styled
    assert "FAILED" in styled


def test_style_run_state_running():
    styled = style_run_state("RUNNING")
    assert "blue" in styled
    assert "RUNNING" in styled


def test_style_run_state_pending():
    styled = style_run_state("PENDING")
    assert "yellow" in styled
    assert "PENDING" in styled


def test_style_run_state_unknown():
    styled = style_run_state("SOMETHING_ELSE")
    assert styled == "SOMETHING_ELSE"


def test_style_isvc_ready_true():
    styled = style_isvc_ready("True")
    assert "green" in styled
    assert "True" in styled


def test_style_isvc_ready_false():
    styled = style_isvc_ready("False")
    assert "red" in styled
    assert "False" in styled


def test_style_isvc_ready_unknown():
    styled = style_isvc_ready("Unknown")
    assert "yellow" in styled
    assert "Unknown" in styled
