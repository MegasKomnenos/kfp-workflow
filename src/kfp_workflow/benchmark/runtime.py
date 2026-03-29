"""Runtime resolution and execution for benchmark workflows."""

from __future__ import annotations

import inspect
import itertools
import json
import socket
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Type

import requests

from kfp_workflow.benchmark.interfaces import (
    DatasetSource,
    InlineScenarioDefinition,
    MetricCollector,
    ScenarioDefinition,
    ScenarioPipeline,
    ensure_metric_collectors,
)
from kfp_workflow.plugins.cmapss_utils import resolve_cmapss_data_dir


_EMBEDDED_COUNTER = itertools.count()


def _module_from_source(source_code: str, source_path: str) -> types.ModuleType:
    module_name = f"kfp_workflow_benchmark_embedded_{next(_EMBEDDED_COUNTER)}"
    module = types.ModuleType(module_name)
    module.__file__ = source_path
    exec(compile(source_code, source_path, "exec"), module.__dict__)
    return module


def _instantiate_symbol(
    node: Dict[str, Any],
    expected_type: Type,
) -> Any:
    module = _module_from_source(node["source_code"], node["source_path"])
    symbol_name = node.get("symbol")
    if not symbol_name:
        raise ValueError("Python benchmark refs must include a symbol name.")
    symbol = getattr(module, symbol_name)
    config = dict(node.get("config", {}))

    if isinstance(symbol, expected_type):
        return symbol
    if inspect.isclass(symbol) and issubclass(symbol, expected_type):
        try:
            return symbol(config=config)
        except TypeError:
            return symbol()
    if callable(symbol):
        try:
            instance = symbol(config=config)
        except TypeError:
            instance = symbol()
        if not isinstance(instance, expected_type):
            raise TypeError(
                f"Callable '{symbol_name}' did not return {expected_type.__name__}, "
                f"got {type(instance)!r}."
            )
        return instance
    raise TypeError(
        f"Python benchmark symbol '{symbol_name}' must resolve to "
        f"{expected_type.__name__} or a factory returning it."
    )


class CmapssTimeseriesDatasetSource(DatasetSource):
    """Replay-ready C-MAPSS sections for channels-last predictors."""

    def __init__(self, spec: Dict[str, Any], config: Dict[str, Any]) -> None:
        self._spec = spec
        self._config = config

    def iter_sections(self) -> Iterable[Dict[str, Any]]:
        import numpy as np
        _ensure_model_package_on_path("mambasl-new")
        from mambasl_new.cmapss.constants import FD_CONFIGS
        from mambasl_new.cmapss.data import load_fd
        from mambasl_new.cmapss.preprocess import get_feature_cols, preprocess_frames

        storage = self._spec["storage"]
        configured_subpath = str(self._config.get("data_subpath", "") or storage.get("data_subpath", "")).strip("/")
        base_mount = Path(storage["data_mount_path"])
        dataset_root = base_mount / configured_subpath if configured_subpath else base_mount
        data_dir = resolve_cmapss_data_dir(str(dataset_root))

        model_dir = Path(storage["model_mount_path"]) / self._spec["model"]["model_subpath"]
        model_config_path = model_dir / "model_config.json"
        model_config = json.loads(model_config_path.read_text("utf-8")) if model_config_path.exists() else {}
        model_cfg = model_config.get("cfg", model_config)

        fd_name = self._config.get("fd_name", "FD001")
        feature_mode = self._config.get("feature_mode", "settings_plus_sensors")
        norm_mode = self._config.get("norm_mode", "condition_minmax")
        seed = int(self._config.get("seed", 42))
        window_size = int(
            self._config.get(
                "window_size",
                model_config.get("seq_len", model_cfg.get("window_size", 30)),
            )
        )
        section_stride = int(self._config.get("section_stride", 1))
        max_sections = int(self._config.get("max_sections", 5))

        train_df, test_df, _ = load_fd(Path(data_dir), fd_name)
        _, _, te_df = preprocess_frames(
            train_df,
            train_df.copy(),
            test_df.copy(),
            feature_mode=feature_mode,
            norm_mode=norm_mode,
            n_conditions=FD_CONFIGS[fd_name]["n_conditions"],
            seed=seed,
        )
        feature_cols = get_feature_cols(feature_mode)

        unit_ids = self._config.get("unit_ids")
        if unit_ids:
            selected_units = [int(uid) for uid in unit_ids]
        else:
            selected_units = sorted(te_df["unit"].unique().tolist())[:1]

        count = 0
        for uid in selected_units:
            sub = te_df[te_df["unit"] == uid].copy()
            x = sub[feature_cols].to_numpy(np.float32)
            if len(x) == 0:
                continue
            if len(x) < window_size:
                pad = np.repeat(x[:1], window_size - len(x), axis=0)
                window = np.concatenate([pad, x], axis=0)
                yield {
                    "payload": window.tolist(),
                    "unit": int(uid),
                    "start_index": 0,
                    "end_index": int(len(x) - 1),
                }
                return

            for start in range(0, len(x) - window_size + 1, section_stride):
                end = start + window_size
                yield {
                    "payload": x[start:end].tolist(),
                    "unit": int(uid),
                    "start_index": int(start),
                    "end_index": int(end - 1),
                }
                count += 1
                if count >= max_sections:
                    return


class SequentialReplayPipeline(ScenarioPipeline):
    """Replay sections against a KServe predictor at a fixed rate."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def run(
        self,
        dataset: DatasetSource,
        *,
        target: Dict[str, Any],
        results_dir: str,
        spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        interval_hz = float(self._config.get("interval_hz", 1.0))
        interval_seconds = 0.0 if interval_hz <= 0 else 1.0 / interval_hz
        timeout = int(self._config.get("request_timeout", 30))
        service_url = target["service_url"].rstrip("/")
        service_name = target["service_name"]
        endpoint = f"{service_url}/v1/models/{service_name}:predict"

        started = time.time()
        records: List[Dict[str, Any]] = []
        for index, section in enumerate(dataset.iter_sections(), start=1):
            response = requests.post(
                endpoint,
                json={"instances": [section["payload"]]},
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            prediction = body.get("predictions", [])
            records.append(
                {
                    "index": index,
                    "unit": section.get("unit"),
                    "start_index": section.get("start_index"),
                    "end_index": section.get("end_index"),
                    "prediction": prediction[0] if prediction else None,
                }
            )
            if interval_seconds > 0:
                time.sleep(interval_seconds)
        return {
            "request_count": len(records),
            "duration_seconds": time.time() - started,
            "interval_hz": interval_hz,
            "endpoint": endpoint,
            "requests": records,
        }


class KeplerEnergyMetricCollector(MetricCollector):
    """Collect Kepler container energy through Prometheus."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def start(
        self,
        *,
        target: Dict[str, Any],
        spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        value = self._query_scalar(
            target=target,
            spec=spec,
            wait_seconds=float(self._config.get("series_wait_seconds", 30.0)),
        )
        return {
            "start_joules": value,
            "start_time": time.time(),
            "target_pod_name": target["predictor_pod_name"],
        }

    def finish(
        self,
        *,
        target: Dict[str, Any],
        spec: Dict[str, Any],
        start_state: Dict[str, Any],
        scenario_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        settle_seconds = float(self._config.get("settle_seconds", 5.0))
        poll_interval = float(self._config.get("poll_interval_seconds", 1.0))
        deadline = time.time() + settle_seconds
        values = [self._query_scalar(target=target, spec=spec)]
        while time.time() < deadline:
            time.sleep(max(poll_interval, 0.1))
            values.append(self._query_scalar(target=target, spec=spec))
        end_value = max(values)
        return {
            "metric_name": self._config.get("metric_name", "kepler_container_joules_total"),
            "mode": self._config.get("mode", "dynamic"),
            "container_name": self._config.get("container_name", "kserve-container"),
            "pod_name": target["predictor_pod_name"],
            "start_joules": start_state["start_joules"],
            "end_joules": end_value,
            "delta_joules": end_value - start_state["start_joules"],
            "duration_seconds": time.time() - float(start_state["start_time"]),
            "request_count": int(scenario_result.get("request_count", 0)),
        }

    def _query_scalar(
        self,
        *,
        target: Dict[str, Any],
        spec: Dict[str, Any],
        wait_seconds: float = 0.0,
    ) -> float:
        prometheus_url = self._config.get(
            "prometheus_url",
            "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090",
        ).rstrip("/")
        metric_name = self._config.get("metric_name", "kepler_container_joules_total")
        mode = self._config.get("mode", "dynamic")
        container_name = self._config.get("container_name", "kserve-container")
        namespace = spec["runtime"]["namespace"]
        poll_interval = max(float(self._config.get("poll_interval_seconds", 1.0)), 0.1)
        deadline = time.time() + max(wait_seconds, 0.0)

        while True:
            result = self._query_result_series(
                prometheus_url=prometheus_url,
                metric_name=metric_name,
                namespace=namespace,
                container_name=container_name,
                mode=mode,
                target=target,
            )
            if len(result) == 1:
                metric = result[0].get("metric", {})
                target["predictor_pod_name"] = metric.get(
                    "pod_name",
                    target.get("predictor_pod_name"),
                )
                return float(result[0]["value"][1])
            if len(result) > 1 or time.time() >= deadline:
                raise RuntimeError(
                    f"Expected 1 Prometheus series for benchmark energy query, got {len(result)}."
                )
            time.sleep(poll_interval)

    def _query_result_series(
        self,
        *,
        prometheus_url: str,
        metric_name: str,
        namespace: str,
        container_name: str,
        mode: str,
        target: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        query_timeout = int(self._config.get("query_timeout", 10))
        pod_name = target.get("predictor_pod_name")
        if pod_name:
            exact = self._run_query(
                prometheus_url=prometheus_url,
                query=(
                    f'{metric_name}{{container_namespace="{namespace}",'
                    f'pod_name="{pod_name}",container_name="{container_name}",mode="{mode}"}}'
                ),
                query_timeout=query_timeout,
            )
            if exact:
                return exact

        service_name = target.get("service_name")
        if service_name:
            return self._run_query(
                prometheus_url=prometheus_url,
                query=(
                    f'{metric_name}{{container_namespace="{namespace}",'
                    f'pod_name=~"{service_name}-predictor-.*",'
                    f'container_name="{container_name}",mode="{mode}"}}'
                ),
                query_timeout=query_timeout,
            )
        return []

    def _run_query(
        self,
        *,
        prometheus_url: str,
        query: str,
        query_timeout: int,
    ) -> List[Dict[str, Any]]:
        response = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=query_timeout,
        )
        response.raise_for_status()
        return response.json()["data"]["result"]


def resolve_scenario_definition(node: Dict[str, Any], spec: Dict[str, Any]) -> ScenarioDefinition:
    """Resolve an inline or embedded scenario definition."""
    if node.get("type") == "python-ref":
        return _instantiate_symbol(node, ScenarioDefinition)
    dataset = resolve_dataset(node["dataset"], spec)
    pipeline = resolve_pipeline(node["pipeline"], spec)
    return InlineScenarioDefinition(dataset, pipeline)


def resolve_dataset(node: Dict[str, Any], spec: Dict[str, Any]) -> DatasetSource:
    """Resolve a dataset source."""
    if node.get("type") == "python-ref":
        return _instantiate_symbol(node, DatasetSource)
    kind = node.get("kind")
    config = dict(node.get("config", {}))
    if kind == "cmapss-timeseries":
        return CmapssTimeseriesDatasetSource(spec, config)
    raise KeyError(f"Unknown benchmark dataset kind '{kind}'.")


def resolve_pipeline(node: Dict[str, Any], spec: Dict[str, Any]) -> ScenarioPipeline:
    """Resolve a scenario pipeline."""
    if node.get("type") == "python-ref":
        return _instantiate_symbol(node, ScenarioPipeline)
    kind = node.get("kind")
    config = dict(node.get("config", {}))
    if kind == "sequential-replay":
        return SequentialReplayPipeline(config)
    raise KeyError(f"Unknown benchmark pipeline kind '{kind}'.")


def resolve_metric_collectors(nodes: List[Dict[str, Any]], spec: Dict[str, Any]) -> List[MetricCollector]:
    """Resolve metric collectors."""
    collectors: List[Any] = []
    for node in nodes:
        if node.get("type") == "python-ref":
            collectors.append(_instantiate_symbol(node, MetricCollector))
            continue
        kind = node.get("kind")
        config = dict(node.get("config", {}))
        if kind == "kepler-energy":
            collectors.append(KeplerEnergyMetricCollector(config))
            continue
        raise KeyError(f"Unknown benchmark metric kind '{kind}'.")
    return ensure_metric_collectors(collectors)


def validate_materialized_benchmark(spec: Dict[str, Any]) -> None:
    """Resolve benchmark runtime objects to fail fast during validation."""
    resolve_scenario_definition(spec["scenario"], spec)
    resolve_metric_collectors(spec.get("metrics", []), spec)


def execute_benchmark(
    spec: Dict[str, Any],
    target: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a fully materialized benchmark and persist results."""
    target = _refresh_target(dict(target), spec)
    scenario = resolve_scenario_definition(spec["scenario"], spec)
    collectors = resolve_metric_collectors(spec.get("metrics", []), spec)

    run_dir = _build_run_dir(spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"

    metric_states: Dict[str, Dict[str, Any]] = {}
    scenario_result: Dict[str, Any] = {}
    status = "succeeded"
    error: str | None = None

    try:
        for index, collector in enumerate(collectors):
            metric_states[f"metric_{index}"] = collector.start(target=target, spec=spec)
        scenario_result = scenario.pipeline().run(
            scenario.dataset(),
            target=target,
            results_dir=str(run_dir),
            spec=spec,
        )
    except Exception as exc:
        status = "failed"
        error = str(exc)
    finally:
        metrics_payload: Dict[str, Any] = {}
        for index, collector in enumerate(collectors):
            key = f"metric_{index}"
            try:
                metrics_payload[key] = collector.finish(
                    target=target,
                    spec=spec,
                    start_state=metric_states.get(key, {}),
                    scenario_result=scenario_result,
                )
            except Exception as exc:
                metrics_payload[key] = {"status": "failed", "error": str(exc)}

        payload = {
            "benchmark_name": spec["metadata"]["name"],
            "status": status,
            "target": target,
            "scenario": scenario_result,
            "metrics": metrics_payload,
            "results_path": str(results_path),
        }
        if error:
            payload["error"] = error
        results_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    if status != "succeeded":
        raise RuntimeError(error or "Benchmark execution failed.")
    return payload


def _build_run_dir(spec: Dict[str, Any]) -> Path:
    root = Path(spec["storage"]["results_mount_path"])
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    run_name = f"{stamp}-{socket.gethostname()}"
    return root / spec["metadata"]["name"] / run_name


def _refresh_target(target: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh the predictor pod name so metrics follow the live replica."""
    service_name = target.get("service_name")
    namespace = target.get("namespace") or spec["runtime"]["namespace"]
    if not service_name:
        return target

    try:
        from kfp_workflow.serving import kserve

        target["predictor_pod_name"] = kserve.get_predictor_pod_name(service_name, namespace)
    except Exception:
        pass
    return target


def _ensure_model_package_on_path(package_dir_name: str) -> None:
    """Add the local model package source tree during repository-local tests."""
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "models" / package_dir_name / "src"
    if candidate.exists():
        sys.path.insert(0, str(candidate))
