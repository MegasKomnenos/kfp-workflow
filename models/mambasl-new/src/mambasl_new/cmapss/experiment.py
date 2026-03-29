from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch

from ..specs import ExperimentSpec, expand_ablation_cases
from ..utils import dump_json
from .constants import FD_CONFIGS, LITERATURE
from .data import ensure_cmapss_downloaded, load_fd
from .preprocess import add_train_rul, choose_val_units, get_feature_cols, preprocess_frames
from .report import compare_to_literature
from .search_space import merge_params, resolve_search_space, suggest_value
from .train import fit_fixed_epochs, mae, nasa_score, predict_array, rmse, selection_value, train_model
from .windowing import make_last_windows, make_pseudo_test_windows, make_windows
from .model import configure_device


TARGETS_2026 = LITERATURE["INF_FUSION_2026"]["metrics"]


def prepare_fd_splits(spec: ExperimentSpec, fd_name: str):
    _, extract_dir = ensure_cmapss_downloaded(spec.data)
    train_df, test_df, rul_test = load_fd(extract_dir, fd_name)
    train_df = add_train_rul(train_df)
    train_units, val_units = choose_val_units(train_df["unit"].unique(), seed=spec.train_defaults.seed)
    tr_df = train_df[train_df["unit"].isin(train_units)].copy()
    va_df = train_df[train_df["unit"].isin(val_units)].copy()
    tr_df, va_df, te_df = preprocess_frames(
        tr_df,
        va_df,
        test_df.copy(),
        feature_mode=spec.train_defaults.feature_mode,
        norm_mode=spec.train_defaults.norm_mode,
        n_conditions=FD_CONFIGS[fd_name]["n_conditions"],
        seed=spec.train_defaults.seed,
    )
    test_targets = {uid: float(rul_test[i]) for i, uid in enumerate(sorted(te_df["unit"].unique().tolist()))}
    return tr_df, va_df, te_df, test_targets


def build_val_arrays(
    val_df,
    feature_cols,
    window_size: int,
    max_rul: float,
    spec: ExperimentSpec,
):
    if spec.train_defaults.val_mode == "all_windows":
        return make_windows(val_df, feature_cols, window_size, spec.train_defaults.hpo_val_stride, max_rul)
    return make_pseudo_test_windows(
        val_df,
        feature_cols,
        window_size=window_size,
        max_rul=max_rul,
        n_samples_per_unit=spec.train_defaults.val_pseudo_samples,
        min_history=spec.train_defaults.val_min_history,
    )


def fixed_base_config(spec: ExperimentSpec) -> Dict[str, Any]:
    defaults = {
        "d_model": 64,
        "d_state": 16,
        "d_conv": 3,
        "expand": 2,
        "num_kernels": 5,
        "tv_dt": True,
        "tv_B": True,
        "tv_C": True,
        "use_D": True,
        "projection": "last",
        "dropout": 0.2,
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "huber_delta": 2.0,
        "window_size": 50,
        "max_rul": 125.0,
    }
    return merge_params(defaults, spec.train_defaults.fixed_params)


def run_dataset_hpo(spec: ExperimentSpec, fd_name: str, device: torch.device) -> Tuple[Dict[str, Any], float]:
    tr_df, va_df, _, _ = prepare_fd_splits(spec, fd_name)
    feature_cols = get_feature_cols(spec.train_defaults.feature_mode)
    search_space = resolve_search_space(spec.hpo)

    def objective(trial):
        cfg = merge_params(fixed_base_config(spec), {param.name: suggest_value(trial, param) for param in search_space})
        x_train, y_train = make_windows(
            tr_df,
            feature_cols,
            int(cfg["window_size"]),
            spec.train_defaults.hpo_train_stride,
            float(cfg["max_rul"]),
        )
        x_val, y_val = build_val_arrays(va_df, feature_cols, int(cfg["window_size"]), float(cfg["max_rul"]), spec)
        if len(x_train) < int(cfg["batch_size"]) or len(x_val) == 0:
            raise optuna.TrialPruned()
        try:
            _, best_rmse, best_score, _, best_metric = train_model(
                cfg,
                x_train,
                y_train,
                x_val,
                y_val,
                max_epochs=spec.train_defaults.hpo_max_epochs,
                patience=spec.train_defaults.hpo_patience,
                selection_metric=spec.train_defaults.selection_metric,
                score_weight=spec.train_defaults.score_weight,
                device=device,
            )
        except RuntimeError as exc:
            raise optuna.TrialPruned() from exc
        trial.set_user_attr("best_val_rmse", best_rmse)
        trial.set_user_attr("best_val_score", best_score)
        return best_metric

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=spec.train_defaults.seed, n_startup_trials=min(10, spec.hpo.max_trial_count)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=min(10, spec.hpo.max_trial_count), n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=spec.hpo.max_trial_count, show_progress_bar=False)
    return merge_params(fixed_base_config(spec), study.best_params), float(study.best_value)


def evaluate_dataset(spec: ExperimentSpec, fd_name: str, cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    tr_df, va_df, te_df, test_targets = prepare_fd_splits(spec, fd_name)
    feature_cols = get_feature_cols(spec.train_defaults.feature_mode)

    x_train, y_train = make_windows(tr_df, feature_cols, int(cfg["window_size"]), 1, float(cfg["max_rul"]))
    x_val, y_val = build_val_arrays(va_df, feature_cols, int(cfg["window_size"]), float(cfg["max_rul"]), spec)
    x_test, y_test, units = make_last_windows(te_df, feature_cols, test_targets, int(cfg["window_size"]), float(cfg["max_rul"]))

    model, best_val_rmse, best_val_score, best_epoch, best_val_metric = train_model(
        cfg,
        x_train,
        y_train,
        x_val,
        y_val,
        max_epochs=spec.train_defaults.final_max_epochs,
        patience=spec.train_defaults.final_patience,
        selection_metric=spec.train_defaults.selection_metric,
        score_weight=spec.train_defaults.score_weight,
        device=device,
    )
    val_preds = predict_array(model, x_val, float(cfg["max_rul"]), device=device)
    val_metrics = {
        "rmse": rmse(val_preds, y_val),
        "mae": mae(val_preds, y_val),
        "score": nasa_score(val_preds, y_val),
        "n_units": int(len(y_val)),
    }
    if spec.train_defaults.refit_full_train:
        x_full = np.concatenate([x_train, x_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)
        model = fit_fixed_epochs(cfg, x_full, y_full, epochs=best_epoch, device=device)
    preds = predict_array(model, x_test, float(cfg["max_rul"]), device=device)
    metrics = {
        "rmse": rmse(preds, y_test),
        "mae": mae(preds, y_test),
        "score": nasa_score(preds, y_test),
        "n_units": int(len(y_test)),
    }
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return {
        "val_metrics": val_metrics,
        "val_rmse": float(best_val_rmse),
        "val_score": float(best_val_score),
        "val_selection_metric": float(best_val_metric),
        "best_epoch": int(best_epoch),
        "refit_full_train": bool(spec.train_defaults.refit_full_train),
        "val_mode": spec.train_defaults.val_mode,
        "val_pseudo_samples": int(spec.train_defaults.val_pseudo_samples),
        "val_min_history": int(spec.train_defaults.val_min_history),
        "test_metrics": metrics,
        "predictions": {str(int(uid)): {"pred": float(pred), "true": float(true)} for uid, pred, true in zip(units, preds, y_test)},
        "n_params": int(model.count_parameters()),
        "state_dict": state_dict,
    }


def run_ablation_suite(spec: ExperimentSpec, fd_name: str, best_cfg: Dict[str, Any], device: torch.device) -> List[Dict[str, Any]]:
    cases = expand_ablation_cases(spec)
    results = []
    for case in cases:
        case_spec = spec.model_copy(deep=True)
        if "feature_mode" in case["overrides"]:
            case_spec.train_defaults.feature_mode = case["overrides"]["feature_mode"]
        if "norm_mode" in case["overrides"]:
            case_spec.train_defaults.norm_mode = case["overrides"]["norm_mode"]
        t0 = time.time()
        eval_out = evaluate_dataset(case_spec, fd_name, merge_params(best_cfg, case["overrides"]), device=device)
        results.append(
            {
                "condition": case["name"],
                "overrides": case["overrides"],
                "train_minutes": round((time.time() - t0) / 60.0, 2),
                **{k: v for k, v in eval_out.items() if k not in {"state_dict", "predictions"}},
            }
        )
    return results


def run_ablation_only(spec: ExperimentSpec, fd_name: str, best_cfg: Dict[str, Any], output_dir: Path) -> List[Dict[str, Any]]:
    torch.set_num_threads(spec.runtime.torch_num_threads)
    device = configure_device(prefer_gpu=spec.runtime.use_gpu)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_ablation_suite(spec, fd_name, best_cfg, device=device)
    dump_json(output_dir / f"{fd_name.lower()}_ablations.json", results)
    return results


def emit_katib_metrics(metrics: Dict[str, float], selection_metric: str, score_weight: float) -> None:
    objective = selection_value(selection_metric, metrics["rmse"], metrics["score"], score_weight)
    print(f"objective={objective}", flush=True)
    print(f"rmse={metrics['rmse']}", flush=True)
    print(f"score={metrics['score']}", flush=True)
    print(f"mae={metrics['mae']}", flush=True)


def run_dataset_pipeline(
    spec: ExperimentSpec,
    fd_name: str,
    output_dir: Path,
    explicit_params: Optional[Dict[str, Any]] = None,
    run_hpo_stage: bool = False,
    run_ablation_stage: bool = False,
) -> Dict[str, Any]:
    torch.set_num_threads(spec.runtime.torch_num_threads)
    np.random.seed(spec.train_defaults.seed)
    torch.manual_seed(spec.train_defaults.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = configure_device(prefer_gpu=spec.runtime.use_gpu)
    output_dir.mkdir(parents=True, exist_ok=True)

    if explicit_params is not None:
        best_cfg = merge_params(fixed_base_config(spec), explicit_params)
        best_val = None
    elif run_hpo_stage and spec.hpo.enabled:
        best_cfg, best_val = run_dataset_hpo(spec, fd_name, device=device)
    else:
        best_cfg = fixed_base_config(spec)
        best_val = None

    final = evaluate_dataset(spec, fd_name, best_cfg, device=device)
    state_dict = final.pop("state_dict")
    result = {
        "experiment": "C-MAPSS FD001-FD004 RUL Benchmark — MambaSL New",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "dataset": fd_name,
        "settings": spec.model_dump(mode="json"),
        "hpo": {
            "objective_metric": spec.train_defaults.selection_metric,
            "best_val_objective": best_val,
            "best_params": best_cfg,
        },
        "final": final,
        "literature_comparison": compare_to_literature(fd_name, final["test_metrics"]),
    }
    if run_ablation_stage and spec.ablations.enabled:
        result["ablations"] = run_ablation_suite(spec, fd_name, best_cfg, device=device)

    dump_json(output_dir / f"{fd_name.lower()}_result.json", result)
    dump_json(output_dir / f"{fd_name.lower()}_metrics.json", final["test_metrics"])
    dump_json(output_dir / f"{fd_name.lower()}_config.json", best_cfg)
    dump_json(output_dir / f"{fd_name.lower()}_predictions.json", final["predictions"])
    if spec.outputs.retain_model_state:
        torch.save(state_dict, output_dir / f"{fd_name.lower()}_model.pt")
    return result


def load_best_params_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())
