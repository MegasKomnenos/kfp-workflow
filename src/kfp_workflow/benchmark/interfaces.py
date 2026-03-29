"""Standard runtime interfaces for benchmark definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List


class BenchmarkDefinition(ABC):
    """Interface for top-level Python benchmark definitions."""

    @abstractmethod
    def build_spec(self) -> Dict[str, Any]:
        """Return a JSON-serializable benchmark spec dict."""


class DatasetSource(ABC):
    """Interface for scenario datasets that yield replayable sections."""

    @abstractmethod
    def iter_sections(self) -> Iterable[Dict[str, Any]]:
        """Yield benchmark sections in replay order."""


class ScenarioPipeline(ABC):
    """Interface for pipelines that replay dataset sections."""

    @abstractmethod
    def run(
        self,
        dataset: DatasetSource,
        *,
        target: Dict[str, Any],
        results_dir: str,
        spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the scenario and return a JSON-serializable summary."""


class ScenarioDefinition(ABC):
    """Interface for scenario definitions."""

    @abstractmethod
    def dataset(self) -> DatasetSource:
        """Return the dataset source for this scenario."""

    @abstractmethod
    def pipeline(self) -> ScenarioPipeline:
        """Return the replay pipeline for this scenario."""


class MetricCollector(ABC):
    """Interface for benchmark metric collectors."""

    @abstractmethod
    def start(
        self,
        *,
        target: Dict[str, Any],
        spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Capture the initial metric state."""

    @abstractmethod
    def finish(
        self,
        *,
        target: Dict[str, Any],
        spec: Dict[str, Any],
        start_state: Dict[str, Any],
        scenario_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Capture the final state and return a JSON-serializable metric summary."""


class InlineScenarioDefinition(ScenarioDefinition):
    """Simple scenario built from an explicit dataset and pipeline."""

    def __init__(self, dataset_source: DatasetSource, scenario_pipeline: ScenarioPipeline) -> None:
        self._dataset_source = dataset_source
        self._scenario_pipeline = scenario_pipeline

    def dataset(self) -> DatasetSource:
        return self._dataset_source

    def pipeline(self) -> ScenarioPipeline:
        return self._scenario_pipeline


def ensure_metric_collectors(items: List[Any]) -> List[MetricCollector]:
    """Validate that *items* are metric collectors and return them."""
    collectors: List[MetricCollector] = []
    for item in items:
        if not isinstance(item, MetricCollector):
            raise TypeError(
                "Metric definitions must resolve to MetricCollector instances, "
                f"got {type(item)!r}."
            )
        collectors.append(item)
    return collectors
