"""Tests for benchmark runtime builtins."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kfp_workflow.benchmark.runtime import (
    CmapssTimeseriesDatasetSource,
    KeplerEnergyMetricCollector,
    SequentialReplayPipeline,
    execute_benchmark,
)


def _write_cmapss_rows(path: Path, rows: list[list[float]]) -> None:
    path.write_text(
        "\n".join(" ".join(str(value) for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _cmapss_row(unit: int, cycle: int, sensor_offset: float) -> list[float]:
    ops = [0.1, 0.2, 0.3]
    sensors = [sensor_offset + float(i) for i in range(21)]
    return [float(unit), float(cycle), *ops, *sensors]


def _runtime_spec(tmp_path: Path) -> dict:
    data_dir = tmp_path / "data" / "cmapss" / "CMAPSSData"
    model_dir = tmp_path / "models" / "mambasl-cmapss" / "v1"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    train_rows = []
    test_rows = []
    for unit in (1, 2):
        for cycle in range(1, 8):
            train_rows.append(_cmapss_row(unit, cycle, sensor_offset=float(unit)))
        for cycle in range(1, 7):
            test_rows.append(_cmapss_row(unit, cycle, sensor_offset=float(unit) + 10.0))

    _write_cmapss_rows(data_dir / "train_FD001.txt", train_rows)
    _write_cmapss_rows(data_dir / "test_FD001.txt", test_rows)
    (data_dir / "RUL_FD001.txt").write_text("10\n20\n", encoding="utf-8")
    (model_dir / "model_config.json").write_text(
        json.dumps({"cfg": {"window_size": 4, "max_rul": 125.0}, "seq_len": 4}),
        encoding="utf-8",
    )

    return {
        "metadata": {"name": "bench"},
        "runtime": {"namespace": "kubeflow-user-example-com"},
        "storage": {
            "data_mount_path": str(tmp_path / "data"),
            "data_subpath": "",
            "model_mount_path": str(tmp_path / "models"),
            "results_mount_path": str(tmp_path / "results"),
        },
        "model": {
            "model_subpath": "mambasl-cmapss/v1",
        },
        "scenario": {
            "dataset": {
                "kind": "cmapss-timeseries",
                "config": {
                    "fd_name": "FD001",
                    "feature_mode": "settings_plus_sensors",
                    "norm_mode": "condition_minmax",
                    "max_sections": 3,
                },
            }
        },
    }


def test_cmapss_timeseries_dataset_source_yields_sections(tmp_path: Path):
    spec = _runtime_spec(tmp_path)
    dataset = CmapssTimeseriesDatasetSource(spec, spec["scenario"]["dataset"]["config"])

    sections = list(dataset.iter_sections())

    assert len(sections) == 3
    assert len(sections[0]["payload"]) == 4
    assert len(sections[0]["payload"][0]) == 17


def test_cmapss_timeseries_dataset_source_has_no_default_section_cap(tmp_path: Path):
    spec = _runtime_spec(tmp_path)
    spec["scenario"]["dataset"]["config"].pop("max_sections", None)
    spec["scenario"]["dataset"]["config"]["unit_ids"] = [1, 2]
    dataset = CmapssTimeseriesDatasetSource(spec, spec["scenario"]["dataset"]["config"])

    sections = list(dataset.iter_sections())

    assert len(sections) == 6
    assert sections[-1]["unit"] == 2
    assert sections[-1]["start_index"] == 2


def test_sequential_replay_pipeline_replays_sections(monkeypatch, tmp_path: Path):
    class _Dataset:
        def iter_sections(self):
            for idx in range(2):
                yield {"payload": [[float(idx)]], "unit": 1, "start_index": idx, "end_index": idx}

    calls = []
    sleeps = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"predictions": [3.14]}

    def _post(url, json, timeout):
        calls.append((url, json, timeout))
        return _Response()

    monkeypatch.setattr("requests.post", _post)
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

    pipeline = SequentialReplayPipeline({"interval_hz": 2.0, "request_timeout": 7})
    result = pipeline.run(
        _Dataset(),
        target={"service_url": "http://svc", "service_name": "bench-svc"},
        results_dir=str(tmp_path),
        spec={"metadata": {"name": "bench"}},
    )

    assert result["request_count"] == 2
    assert calls[0][0] == "http://svc/v1/models/bench-svc:predict"
    assert calls[0][2] == 7
    assert sleeps == [0.5, 0.5]


def test_kepler_energy_metric_collector_queries_prometheus(monkeypatch):
    responses = iter([1.0, 3.5, 4.0, 3.8, 4.2, 4.2])

    class _Response:
        def __init__(self, value: float):
            self._value = value

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": {
                    "result": [
                        {
                            "value": [0, str(self._value)],
                        }
                    ]
                }
            }

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: _Response(next(responses)))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    collector = KeplerEnergyMetricCollector(
        {
            "prometheus_url": "http://prom",
            "settle_seconds": 0.0,
            "mode": "dynamic",
        }
    )
    target = {"predictor_pod_name": "pod-1"}
    spec = {"runtime": {"namespace": "kubeflow-user-example-com"}}

    start = collector.start(target=target, spec=spec)
    end = collector.finish(
        target=target,
        spec=spec,
        start_state=start,
        scenario_result={"request_count": 2},
    )

    assert start["start_joules"] == 1.0
    assert end["delta_joules"] == pytest.approx(2.5)


def test_kepler_energy_metric_collector_waits_for_series(monkeypatch):
    current_time = [100.0]
    sleeps = []
    responses = iter(
        [
            [],
            [],
            [{"value": [0, "2.0"]}],
        ]
    )

    class _Response:
        def __init__(self, result):
            self._result = result

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"result": self._result}}

    def _get(*_args, **_kwargs):
        return _Response(next(responses))

    def _sleep(seconds):
        sleeps.append(seconds)
        current_time[0] += seconds

    monkeypatch.setattr("requests.get", _get)
    monkeypatch.setattr("time.sleep", _sleep)
    monkeypatch.setattr("time.time", lambda: current_time[0])

    collector = KeplerEnergyMetricCollector(
        {
            "prometheus_url": "http://prom",
            "series_wait_seconds": 5.0,
            "poll_interval_seconds": 1.5,
            "mode": "dynamic",
        }
    )
    target = {"predictor_pod_name": "pod-1"}
    spec = {"runtime": {"namespace": "kubeflow-user-example-com"}}

    start = collector.start(target=target, spec=spec)

    assert start["start_joules"] == 2.0
    assert sleeps == [1.5, 1.5]


def test_kepler_energy_metric_collector_falls_back_to_service_regex(monkeypatch):
    queries = []

    class _Response:
        def __init__(self, result):
            self._result = result

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"result": self._result}}

    def _get(*_args, **kwargs):
        query = kwargs["params"]["query"]
        queries.append(query)
        if 'pod_name="stale-pod"' in query:
            return _Response([])
        if 'pod_name=~"bench-svc-predictor-.*"' in query:
            return _Response(
                [
                    {
                        "metric": {"pod_name": "bench-svc-predictor-live"},
                        "value": [0, "5.5"],
                    }
                ]
            )
        raise AssertionError(f"unexpected query: {query}")

    monkeypatch.setattr("requests.get", _get)

    collector = KeplerEnergyMetricCollector({"prometheus_url": "http://prom", "mode": "dynamic"})
    target = {"predictor_pod_name": "stale-pod", "service_name": "bench-svc"}
    spec = {"runtime": {"namespace": "kubeflow-user-example-com"}}

    start = collector.start(target=target, spec=spec)

    assert start["start_joules"] == 5.5
    assert target["predictor_pod_name"] == "bench-svc-predictor-live"
    assert any('pod_name="stale-pod"' in query for query in queries)
    assert any('pod_name=~"bench-svc-predictor-.*"' in query for query in queries)


def test_execute_benchmark_persists_results(monkeypatch, tmp_path: Path):
    spec = {
        "metadata": {"name": "bench"},
        "runtime": {"namespace": "kubeflow-user-example-com"},
        "storage": {"results_mount_path": str(tmp_path / "results")},
        "scenario": {
            "dataset": {"type": "python-ref", "source_path": "x.py", "symbol": "dataset_factory", "source_code": """
from kfp_workflow.benchmark.interfaces import DatasetSource
class D(DatasetSource):
    def iter_sections(self):
        yield {\"payload\": [[1.0]], \"unit\": 1, \"start_index\": 0, \"end_index\": 0}
def dataset_factory():
    return D()
"""},
            "pipeline": {"type": "python-ref", "source_path": "y.py", "symbol": "pipeline_factory", "source_code": """
from kfp_workflow.benchmark.interfaces import ScenarioPipeline
class P(ScenarioPipeline):
    def run(self, dataset, *, target, results_dir, spec):
        rows = list(dataset.iter_sections())
        return {\"request_count\": len(rows), \"requests\": rows}
def pipeline_factory():
    return P()
"""},
        },
        "metrics": [
            {
                "type": "python-ref",
                "source_path": "m.py",
                "symbol": "metric_factory",
                "source_code": """
from kfp_workflow.benchmark.interfaces import MetricCollector
class M(MetricCollector):
    def start(self, *, target, spec):
        return {\"value\": 1.0}
    def finish(self, *, target, spec, start_state, scenario_result):
        return {\"delta\": scenario_result[\"request_count\"]}
def metric_factory():
    return M()
""",
            }
        ],
    }

    result = execute_benchmark(
        spec,
        target={
            "service_name": "bench-svc",
            "service_url": "http://svc",
            "predictor_pod_name": "pod-1",
        },
    )

    results_path = Path(result["results_path"])
    assert results_path.exists()
    saved = json.loads(results_path.read_text("utf-8"))
    assert saved["status"] == "succeeded"
    assert saved["scenario"]["request_count"] == 1
