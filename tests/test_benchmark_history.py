"""Unit tests for benchmark history helpers."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from kfp_workflow.benchmark.history import (
    extract_benchmark_spec,
    is_benchmark_workflow,
    resolve_results,
    summarize_result_payload,
)


def _workflow_with_spec(spec: dict, *, workflow_name: str = "bench-workflow") -> dict:
    return {
        "metadata": {"name": workflow_name},
        "status": {
            "nodes": {
                "n1": {"displayName": "run-benchmark-component"},
            }
        },
        "spec": {
            "arguments": {
                "parameters": [
                    {"name": "spec_json", "value": json.dumps(spec)},
                ]
            }
        },
    }


def test_extract_benchmark_spec():
    spec = {
        "metadata": {"name": "bench"},
        "runtime": {"namespace": "ns"},
        "storage": {"results_pvc": "benchmark-store"},
        "model": {"model_name": "m", "model_subpath": "m/v1"},
        "scenario": {"kind": "inline"},
    }
    workflow = _workflow_with_spec(spec)

    extracted = extract_benchmark_spec(workflow)

    assert extracted == spec


def test_extract_benchmark_spec_from_encoded_json_string():
    spec = {
        "metadata": {"name": "bench"},
        "runtime": {"namespace": "ns"},
        "storage": {"results_pvc": "benchmark-store"},
        "model": {"model_name": "m", "model_subpath": "m/v1"},
        "scenario": {"kind": "inline"},
    }
    workflow = {
        "metadata": {"name": "bench-workflow"},
        "status": {"nodes": {"n1": {"displayName": "run-benchmark-component"}}},
        "spec": {
            "arguments": {
                "parameters": [
                    {
                        "name": "component-json",
                        "value": json.dumps(
                            {
                                "inputs": {
                                    "parameters": {
                                        "spec_json": {"runtimeValue": {"constant": json.dumps(spec)}}
                                    }
                                }
                            }
                        ),
                    }
                ]
            }
        },
    }

    extracted = extract_benchmark_spec(workflow)

    assert extracted == spec


def test_is_benchmark_workflow_requires_run_component():
    workflow = {
        "metadata": {"name": "wf"},
        "status": {"nodes": {"n1": {"displayName": "train"}}},
        "spec": {"arguments": {"parameters": []}},
    }
    assert is_benchmark_workflow(workflow) is False


def test_summarize_result_payload():
    payload = {
        "status": "succeeded",
        "scenario": {"request_count": 5, "duration_seconds": 5.8},
        "metrics": {"metric_0": {"delta_joules": 12.3}},
    }
    summary = summarize_result_payload(payload)
    assert summary["status"] == "succeeded"
    assert summary["request_count"] == 5
    assert summary["delta_joules"] == 12.3


@patch("kfp_workflow.benchmark.history._read_result_file")
@patch("kfp_workflow.benchmark.history._list_result_candidates")
def test_resolve_results(mock_list, mock_read):
    spec = {
        "metadata": {"name": "bench"},
        "storage": {"results_pvc": "benchmark-store"},
    }
    workflow = _workflow_with_spec(spec, workflow_name="bench-workflow-123")
    mock_list.return_value = [
        "/mnt/results/bench/20260329T000000-bench-workflow-123-metadata/results.json",
    ]
    mock_read.return_value = json.dumps(
        {"status": "succeeded", "scenario": {"request_count": 5}, "metrics": {}}
    )

    resolved = resolve_results(workflow=workflow, benchmark_spec=spec, namespace="ns")

    assert resolved["results_path"].endswith("results.json")
    assert resolved["summary"]["request_count"] == 5


@patch("kfp_workflow.benchmark.history._read_result_file")
@patch("kfp_workflow.benchmark.history._list_result_candidates")
def test_resolve_results_accepts_python_literal_payload(mock_list, mock_read):
    spec = {
        "metadata": {"name": "bench"},
        "storage": {"results_pvc": "benchmark-store"},
    }
    workflow = _workflow_with_spec(spec, workflow_name="bench-workflow-123")
    mock_list.return_value = [
        "/mnt/results/bench/20260329T000000-bench-workflow-123-metadata/results.json",
    ]
    mock_read.return_value = "{'status': 'succeeded', 'scenario': {'request_count': 5}, 'metrics': {}}"

    resolved = resolve_results(workflow=workflow, benchmark_spec=spec, namespace="ns")

    assert resolved["payload"]["status"] == "succeeded"
    assert resolved["summary"]["request_count"] == 5


@patch("kfp_workflow.benchmark.history._list_result_candidates")
def test_resolve_results_fails_on_multiple_matches(mock_list):
    spec = {
        "metadata": {"name": "bench"},
        "storage": {"results_pvc": "benchmark-store"},
    }
    workflow = _workflow_with_spec(spec, workflow_name="bench-workflow-123")
    mock_list.return_value = [
        "/mnt/results/bench/20260329T000000-bench-workflow-123-a/results.json",
        "/mnt/results/bench/20260329T000100-bench-workflow-123-b/results.json",
    ]

    with pytest.raises(RuntimeError):
        resolve_results(workflow=workflow, benchmark_spec=spec, namespace="ns")
