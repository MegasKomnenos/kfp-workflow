"""Tests for Kubernetes-API-backed KServe operations."""

from unittest.mock import MagicMock

from kubernetes.client.exceptions import ApiException

from kfp_workflow.serving.kserve import (
    create_inference_service,
    delete_inference_service,
    wait_for_inference_service_ready,
)


def test_create_inference_service_creates_when_missing(monkeypatch):
    api = MagicMock()
    api.get_namespaced_custom_object.side_effect = ApiException(status=404)
    monkeypatch.setattr("kfp_workflow.serving.kserve._custom_objects_api", lambda: api)

    create_inference_service(
        name="bench-svc",
        namespace="ns",
        model_pvc_name="model-store",
        model_subpath="mambasl-cmapss/v1",
        predictor_image="kfp-workflow:latest",
        model_name="mambasl-cmapss",
    )

    api.create_namespaced_custom_object.assert_called_once()
    api.replace_namespaced_custom_object.assert_not_called()


def test_create_inference_service_replaces_when_present(monkeypatch):
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = {
        "metadata": {"resourceVersion": "123"},
    }
    monkeypatch.setattr("kfp_workflow.serving.kserve._custom_objects_api", lambda: api)

    create_inference_service(
        name="bench-svc",
        namespace="ns",
        model_pvc_name="model-store",
        model_subpath="mambasl-cmapss/v1",
        predictor_image="kfp-workflow:latest",
        model_name="mambasl-cmapss",
    )

    api.replace_namespaced_custom_object.assert_called_once()
    body = api.replace_namespaced_custom_object.call_args.kwargs["body"]
    assert body["metadata"]["resourceVersion"] == "123"


def test_delete_inference_service_ignores_missing(monkeypatch):
    api = MagicMock()
    api.delete_namespaced_custom_object.side_effect = ApiException(status=404)
    monkeypatch.setattr("kfp_workflow.serving.kserve._custom_objects_api", lambda: api)

    delete_inference_service(name="bench-svc", namespace="ns")

    api.delete_namespaced_custom_object.assert_called_once()


def test_wait_for_inference_service_ready_polls_between_non_ready_reads(monkeypatch):
    diagnostics = iter(
        [
            {"ready": "False", "conditions": [{"type": "Ready", "status": "False"}]},
            {"ready": "True", "conditions": [{"type": "Ready", "status": "True"}]},
        ]
    )
    sleeps = []

    monkeypatch.setattr(
        "kfp_workflow.serving.kserve.get_inference_service_diagnostics",
        lambda name, namespace: next(diagnostics),
    )
    monkeypatch.setattr("kfp_workflow.serving.kserve.time.sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        "kfp_workflow.serving.kserve.time.time",
        iter([100.0, 100.0, 101.0]).__next__,
    )

    result = wait_for_inference_service_ready(
        name="bench-svc",
        namespace="ns",
        timeout=30,
        poll_interval=2.0,
    )

    assert result["ready"] == "True"
    assert sleeps == [2.0]
