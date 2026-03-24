"""Project-owned HPO engine using Optuna.

The engine creates the study, runs the trial loop, and calls
``plugin.hpo_objective()`` for each trial.  The plugin never sees the
Optuna ``trial`` object or manages the study.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, TYPE_CHECKING

import optuna

from kfp_workflow.specs import (
    HpoResult,
    HpoTrialResult,
    SearchParamSpec,
    TuneSpec,
)
from kfp_workflow.tune.exceptions import TrialPruned

if TYPE_CHECKING:
    from kfp_workflow.plugins.base import ModelPlugin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _suggest_value(trial: optuna.Trial, param: SearchParamSpec) -> Any:
    """Map a project-owned ``SearchParamSpec`` to an Optuna suggest call."""
    if param.type == "categorical":
        return trial.suggest_categorical(param.name, param.values)
    if param.type == "int":
        return trial.suggest_int(
            param.name,
            int(param.low),  # type: ignore[arg-type]
            int(param.high),  # type: ignore[arg-type]
            step=int(param.step or 1),
        )
    if param.type == "float":
        return trial.suggest_float(
            param.name,
            param.low,  # type: ignore[arg-type]
            param.high,  # type: ignore[arg-type]
            step=param.step,
        )
    if param.type == "log_float":
        return trial.suggest_float(
            param.name,
            param.low,  # type: ignore[arg-type]
            param.high,  # type: ignore[arg-type]
            log=True,
        )
    raise ValueError(f"Unknown search-param type: {param.type}")


def _build_sampler(
    algorithm: str,
    seed: int,
    search_space: List[SearchParamSpec],
) -> optuna.samplers.BaseSampler:
    """Instantiate the appropriate Optuna sampler."""
    if algorithm == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=min(10, max(1, len(search_space))),
        )
    if algorithm == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if algorithm == "grid":
        grid: Dict[str, list] = {}
        for p in search_space:
            if p.type == "categorical":
                grid[p.name] = list(p.values or [])
            else:
                raise ValueError(
                    f"Grid search requires categorical params; "
                    f"'{p.name}' is {p.type}"
                )
        return optuna.samplers.GridSampler(grid, seed=seed)
    raise ValueError(f"Unknown HPO algorithm: {algorithm}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_search_space(
    plugin: "ModelPlugin",
    spec_dict: Dict[str, Any],
) -> List[SearchParamSpec]:
    """Resolve the final search space.

    If the spec contains an explicit ``hpo.search_space`` list, validate and
    return it directly.  Otherwise delegate to the plugin's builtin profiles.
    """
    hpo = spec_dict.get("hpo", {})
    custom_raw = hpo.get("search_space", [])
    if custom_raw:
        return [SearchParamSpec.model_validate(p) for p in custom_raw]
    profile = hpo.get("builtin_profile", "default")
    return plugin.hpo_search_space(spec_dict, profile)


def run_hpo(
    plugin: "ModelPlugin",
    spec: TuneSpec,
    data_mount_path: str,
) -> HpoResult:
    """Run a full HPO study.  All orchestration is project-owned."""
    spec_dict = spec.model_dump()
    search_space = resolve_search_space(plugin, spec_dict)
    base_config = plugin.hpo_base_config(spec_dict)
    seed = spec.train.seed

    sampler = _build_sampler(spec.hpo.algorithm, seed, search_space)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    failed_count = 0

    def objective(trial: optuna.Trial) -> float:
        nonlocal failed_count
        suggested = {
            p.name: _suggest_value(trial, p) for p in search_space
        }
        merged = {**base_config, **suggested}
        try:
            value = plugin.hpo_objective(spec_dict, merged, data_mount_path)
        except TrialPruned:
            raise optuna.TrialPruned()
        except Exception:
            failed_count += 1
            logger.warning(
                "Trial %d failed (%d/%d)",
                trial.number,
                failed_count,
                spec.hpo.max_failed_trials,
            )
            if failed_count >= spec.hpo.max_failed_trials:
                study.stop()
            raise optuna.TrialPruned()
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    study.optimize(
        objective,
        n_trials=spec.hpo.max_trials,
        show_progress_bar=True,
    )
    wall_time = time.time() - t0

    # Collect trial results
    trials: List[HpoTrialResult] = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            status = "completed"
        elif t.state == optuna.trial.TrialState.PRUNED:
            status = "pruned"
        else:
            status = "failed"
        trials.append(
            HpoTrialResult(
                trial_number=t.number,
                params=t.params,
                objective_value=t.value if t.value is not None else float("inf"),
                status=status,
                user_attrs=dict(t.user_attrs),
            )
        )

    n_completed = sum(1 for t in trials if t.status == "completed")
    n_pruned = sum(1 for t in trials if t.status == "pruned")
    n_failed = sum(1 for t in trials if t.status == "failed")

    best_params = {**base_config, **study.best_params} if n_completed > 0 else base_config
    best_value = study.best_value if n_completed > 0 else float("inf")

    return HpoResult(
        best_params=best_params,
        best_value=best_value,
        n_trials=len(trials),
        n_completed=n_completed,
        n_pruned=n_pruned,
        n_failed=n_failed,
        trials=trials,
        wall_time_seconds=wall_time,
    )
