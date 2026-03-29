"""KFP components for benchmark workflows."""

from kfp import dsl

BASE_IMAGE = "kfp-workflow:latest"


@dsl.component(base_image=BASE_IMAGE)
def deploy_benchmark_model_component(spec_json: str) -> str:
    """Deploy the benchmark target model as an InferenceService."""
    import json

    from kfp_workflow.serving import kserve

    spec = json.loads(spec_json)
    model = spec["model"]
    namespace = spec["runtime"]["namespace"]
    service_name = model.get("service_name") or spec["metadata"]["name"]

    kserve.create_inference_service(
        name=service_name,
        namespace=namespace,
        model_pvc_name=model["model_pvc"],
        model_subpath=model["model_subpath"],
        runtime=model.get("runtime", "custom"),
        predictor_image=model.get("predictor_image", ""),
        model_name=model["model_name"],
        replicas=int(model.get("replicas", 1)),
        resources=model.get("resources", {}),
        dry_run=False,
    )
    return json.dumps({"service_name": service_name, "namespace": namespace})


@dsl.component(base_image=BASE_IMAGE)
def wait_for_benchmark_model_component(spec_json: str) -> str:
    """Wait until the benchmark InferenceService is Ready."""
    import json

    from kfp_workflow.serving import kserve

    spec = json.loads(spec_json)
    model = spec["model"]
    namespace = spec["runtime"]["namespace"]
    service_name = model.get("service_name") or spec["metadata"]["name"]

    diagnostics = kserve.wait_for_inference_service_ready(
        name=service_name,
        namespace=namespace,
        timeout=int(model.get("wait_timeout", 300)),
    )
    if diagnostics["ready"] != "True":
        raise RuntimeError(
            f"InferenceService '{service_name}' did not become Ready=True: "
            f"{diagnostics.get('conditions', [])}"
        )
    pod_name = kserve.get_predictor_pod_name(service_name, namespace)
    return json.dumps(
        {
            "service_name": service_name,
            "namespace": namespace,
            "service_url": f"http://{service_name}-predictor.{namespace}.svc.cluster.local",
            "predictor_pod_name": pod_name,
            "predictor_container_name": "kserve-container",
        }
    )


@dsl.component(base_image=BASE_IMAGE)
def run_benchmark_component(spec_json: str, target_json: str) -> str:
    """Execute the scenario and collect metrics."""
    import json

    from kfp_workflow.benchmark.runtime import execute_benchmark

    spec = json.loads(spec_json)
    target = json.loads(target_json)
    result = execute_benchmark(spec, target)
    return json.dumps(result)


@dsl.component(base_image=BASE_IMAGE)
def cleanup_benchmark_model_component(spec_json: str) -> str:
    """Delete the benchmark InferenceService when cleanup is enabled."""
    import json

    from kfp_workflow.serving import kserve

    spec = json.loads(spec_json)
    model = spec["model"]
    if not model.get("cleanup", True):
        return json.dumps({"cleanup": "skipped"})

    namespace = spec["runtime"]["namespace"]
    service_name = model.get("service_name") or spec["metadata"]["name"]
    kserve.delete_inference_service(service_name, namespace)
    return json.dumps({"cleanup": "deleted", "service_name": service_name})
