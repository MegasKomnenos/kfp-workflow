from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..utils import dump_json
from .constants import LITERATURE


def compare_to_literature(fd_name: str, our_metrics: Dict[str, float]) -> Dict[str, Any]:
    out = {}
    for key, entry in LITERATURE.items():
        lit = entry["metrics"][fd_name]
        out[key] = {
            "source": entry["source"],
            "url": entry["url"],
            "literature_rmse": lit["rmse"],
            "literature_score": lit["score"],
            "delta_rmse": float(our_metrics["rmse"] - lit["rmse"]),
            "delta_score": float(our_metrics["score"] - lit["score"]),
        }
    return out


def aggregate_dataset_records(dataset_records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    leaderboard = []
    for fd_name, record in dataset_records.items():
        metrics = record["final"]["test_metrics"]
        leaderboard.append(
            {
                "dataset": fd_name,
                "rmse": metrics["rmse"],
                "score": metrics["score"],
                "mae": metrics["mae"],
                "best_epoch": record["final"]["best_epoch"],
            }
        )
    leaderboard.sort(key=lambda item: item["rmse"])
    return {"leaderboard": leaderboard, "datasets": dataset_records}


def render_summary_markdown(experiment_name: str, aggregate: Dict[str, Any]) -> str:
    lines = [f"# {experiment_name}", "", "| Dataset | RMSE | Score | MAE | Best Epoch |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in aggregate["leaderboard"]:
        lines.append(f"| {row['dataset']} | {row['rmse']:.4f} | {row['score']:.2f} | {row['mae']:.4f} | {row['best_epoch']} |")
    return "\n".join(lines) + "\n"


def write_aggregate(output_dir: Path, experiment_name: str, dataset_records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    aggregate = aggregate_dataset_records(dataset_records)
    dump_json(output_dir / "aggregate.json", aggregate)
    (output_dir / "summary.md").write_text(render_summary_markdown(experiment_name, aggregate))
    return aggregate


def collect_metric_files(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    out = []
    for path in paths:
        out.append({"path": str(path), "exists": path.exists()})
    return out
