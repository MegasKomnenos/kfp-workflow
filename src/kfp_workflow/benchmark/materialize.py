"""Benchmark spec loading and ref materialization."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
from typing import Any, Dict, Tuple

from kfp_workflow.config_override import apply_overrides
from kfp_workflow.benchmark.interfaces import BenchmarkDefinition
from kfp_workflow.specs import BenchmarkSpec
from kfp_workflow.utils import load_yaml


def _module_from_path(path: Path):
    module_name = f"kfp_workflow_benchmark_script_{next(_COUNTER)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Python spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_COUNTER = itertools.count()


def load_benchmark_input(path: str | Path) -> Dict[str, Any]:
    """Load a benchmark spec from YAML or a Python script."""
    source = Path(path)
    if source.suffix in {".yaml", ".yml"}:
        raw = load_yaml(source)
        if not isinstance(raw, dict):
            raise TypeError(f"Benchmark spec at {source} must be a mapping.")
        return raw
    if source.suffix != ".py":
        raise ValueError(f"Unsupported benchmark spec path: {source}")

    module = _module_from_path(source)
    if hasattr(module, "build_benchmark_spec"):
        raw = module.build_benchmark_spec()
    elif hasattr(module, "BENCHMARK"):
        benchmark = module.BENCHMARK
        if isinstance(benchmark, BenchmarkDefinition):
            raw = benchmark.build_spec()
        else:
            raw = benchmark
    else:
        raise AttributeError(
            f"Python benchmark spec {source} must export build_benchmark_spec() or BENCHMARK."
        )
    if not isinstance(raw, dict):
        raise TypeError(
            f"Python benchmark spec {source} must resolve to a dict, got {type(raw)!r}."
        )
    return raw


def load_materialized_benchmark_spec(
    path: str | Path,
    overrides: list[str] | None = None,
) -> Tuple[BenchmarkSpec, Dict[str, Any]]:
    """Load, validate, and materialize a benchmark spec."""
    source = Path(path)
    raw = load_benchmark_input(source)
    if overrides:
        raw = apply_overrides(raw, overrides)
    validated = BenchmarkSpec.model_validate(raw)
    materialized = validated.model_dump()
    materialized["scenario"] = _materialize_node(
        validated.scenario,
        base_dir=source.parent,
        interface="scenario",
    )
    materialized["metrics"] = [
        _materialize_node(item, base_dir=source.parent, interface="metric")
        for item in validated.metrics
    ]
    materialized["_spec_source"] = str(source)
    from kfp_workflow.benchmark.runtime import validate_materialized_benchmark

    validate_materialized_benchmark(materialized)
    return validated, materialized


def _materialize_node(node: Any, *, base_dir: Path, interface: str) -> Any:
    """Inline YAML refs and embed Python source refs into plain dicts."""
    if isinstance(node, list):
        return [
            _materialize_node(item, base_dir=base_dir, interface=interface)
            for item in node
        ]
    if not isinstance(node, dict):
        return node

    if "ref" in node:
        target, symbol = _parse_ref(node["ref"], base_dir)
        merged = {
            key: value
            for key, value in node.items()
            if key != "ref"
        }
        if target.suffix in {".yaml", ".yml"}:
            loaded = load_yaml(target)
            if not isinstance(loaded, dict):
                raise TypeError(f"Referenced YAML at {target} must be a mapping.")
            return _materialize_node(
                {**loaded, **merged},
                base_dir=target.parent,
                interface=interface,
            )
        embedded = {
            "type": "python-ref",
            "interface": interface,
            "source_path": str(target),
            "symbol": symbol,
            "source_code": target.read_text("utf-8"),
        }
        if merged:
            embedded["config"] = merged.get("config", {})
        return embedded

    out = dict(node)
    if interface == "scenario":
        if "dataset" in out:
            out["dataset"] = _materialize_node(
                out["dataset"],
                base_dir=base_dir,
                interface="dataset",
            )
        if "pipeline" in out:
            out["pipeline"] = _materialize_node(
                out["pipeline"],
                base_dir=base_dir,
                interface="pipeline",
            )
        return out
    if interface in {"dataset", "pipeline", "metric"} and out.get("kind") == "python":
        ref = out.get("ref") or out.get("entrypoint")
        if not ref:
            raise ValueError(
                f"Python {interface} definitions must provide 'ref' or 'entrypoint'."
            )
        target, symbol = _parse_ref(ref, base_dir)
        return {
            "type": "python-ref",
            "interface": interface,
            "source_path": str(target),
            "symbol": symbol,
            "source_code": target.read_text("utf-8"),
            "config": out.get("config", {}),
        }
    return out


def _parse_ref(raw_ref: str, base_dir: Path) -> Tuple[Path, str]:
    """Parse ``path.py:Symbol`` or a relative YAML path."""
    if ":" in raw_ref:
        path_str, symbol = raw_ref.split(":", 1)
        path = Path(path_str)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path, symbol

    path = Path(raw_ref)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path, ""
