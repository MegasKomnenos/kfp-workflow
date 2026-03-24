from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from .cmapss import (
    apply_scalers,
    ensure_cmapss_downloaded,
    extract_feature_groups,
    feature_columns,
    fit_scalers,
    group_by_unit,
    limited_units,
    load_rul_targets,
    load_split,
    make_test_windows,
    make_train_windows,
    make_val_windows,
)
from .config import ExperimentConfig, build_result_dir_name
from .model import MRHySPRegressor


@dataclass
class Metrics:
    rmse: float
    nasa_score: float
    mae: float
    n_samples: int


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_pred - y_true
    score = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return float(np.sum(score))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        rmse=rmse(y_true, y_pred),
        nasa_score=nasa_score(y_true, y_pred),
        mae=float(np.mean(np.abs(y_pred - y_true))),
        n_samples=int(len(y_true)),
    )


def batched_predict(model: MRHySPRegressor, x: np.ndarray, batch_size: int, max_rul: int) -> np.ndarray:
    parts = []
    for start in range(0, len(x), batch_size):
        part = model.predict(x[start:start + batch_size])
        parts.append(np.clip(part, 0.0, float(max_rul)))
    return np.concatenate(parts)


def per_unit_rows(unit_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> List[Dict[str, Union[float, int]]]:
    rows = []
    for uid, yt, yp in zip(unit_ids.tolist(), y_true.tolist(), y_pred.tolist()):
        rows.append(
            {
                "unit_id": int(uid),
                "y_true": float(yt),
                "y_pred": float(yp),
                "abs_error": float(abs(yp - yt)),
                "nasa_score": float(nasa_score(np.asarray([yt]), np.asarray([yp]))),
            }
        )
    return rows


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    np.random.seed(config.seed)
    data_dir = Path(config.data_dir)
    if config.download_if_missing:
        ensure_cmapss_downloaded(data_dir)

    run_mapping = config.to_mapping()
    out_dir = Path(config.output_dir) / build_result_dir_name(run_mapping)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not config.force:
        payload = json.loads(metrics_path.read_text())
        payload["result_dir"] = str(out_dir)
        print(json.dumps(payload, indent=2), flush=True)
        return payload
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_split(data_dir / f"train_{config.subset}.txt")
    test_raw = load_split(data_dir / f"test_{config.subset}.txt")
    test_rul = load_rul_targets(data_dir / f"RUL_{config.subset}.txt")
    feature_idx = feature_columns(config.feature_mode)

    train_units = sorted(np.unique(train_raw[:, 0].astype(int)).tolist())
    rng = np.random.RandomState(config.seed)
    rng.shuffle(train_units)
    n_val_units = max(1, int(len(train_units) * config.val_frac))
    val_units = train_units[:n_val_units]
    tr_units = train_units[n_val_units:]
    if config.limit_val_units > 0:
        val_units = val_units[:config.limit_val_units]
    if config.limit_train_units > 0:
        tr_units = tr_units[:config.limit_train_units]

    grouped_train = group_by_unit(train_raw)
    grouped_test = group_by_unit(test_raw)
    grouped_tr = {uid: grouped_train[uid] for uid in grouped_train if uid in set(tr_units)}
    grouped_val = {uid: grouped_train[uid] for uid in grouped_train if uid in set(val_units)}
    grouped_test = limited_units(grouped_test, config.limit_test_units)
    test_rul = test_rul[:len(grouped_test)]

    raw_feats_tr = extract_feature_groups(grouped_tr, feature_idx)
    raw_feats_val = extract_feature_groups(grouped_val, feature_idx)
    raw_feats_te = extract_feature_groups(grouped_test, feature_idx)
    global_scaler, cond_scalers = fit_scalers(raw_feats_tr, config.scaling_mode)
    feats_tr = apply_scalers(raw_feats_tr, global_scaler, cond_scalers, config.scaling_mode)
    feats_val = apply_scalers(raw_feats_val, global_scaler, cond_scalers, config.scaling_mode)
    feats_te = apply_scalers(raw_feats_te, global_scaler, cond_scalers, config.scaling_mode)

    x_tr, y_tr = make_train_windows(grouped_tr, feats_tr, config.seq_len, config.max_rul, config.train_stride)
    x_va, y_va = make_val_windows(
        grouped_val,
        feats_val,
        config.seq_len,
        config.max_rul,
        config.val_mode,
        config.val_samples_per_unit,
        config.seed,
    )
    x_te, y_te, uid_te = make_test_windows(grouped_test, feats_te, config.seq_len, test_rul, config.max_rul)

    val_model = MRHySPRegressor(
        mr_num_kernels=config.mr_num_kernels,
        n_kernels=config.n_kernels,
        n_groups=config.n_groups,
        n_kernels_sp=config.n_kernels_sp,
        n_jobs=config.n_jobs,
        random_state=config.seed,
    )
    val_fit_start = time.time()
    val_model.fit(x_tr, y_tr)
    val_fit_s = time.time() - val_fit_start
    y_va_pred = batched_predict(val_model, x_va, config.predict_batch_size, config.max_rul)
    val_metrics = compute_metrics(y_va, y_va_pred)

    full_grouped = {**grouped_tr, **grouped_val}
    full_feats_raw = extract_feature_groups(full_grouped, feature_idx)
    full_global_scaler, full_cond_scalers = fit_scalers(full_feats_raw, config.scaling_mode)
    full_feats = apply_scalers(full_feats_raw, full_global_scaler, full_cond_scalers, config.scaling_mode)
    full_test_feats = apply_scalers(raw_feats_te, full_global_scaler, full_cond_scalers, config.scaling_mode)
    x_full, y_full = make_train_windows(full_grouped, full_feats, config.seq_len, config.max_rul, config.train_stride)
    x_te_full, y_te_full, uid_te_full = make_test_windows(grouped_test, full_test_feats, config.seq_len, test_rul, config.max_rul)

    final_model = MRHySPRegressor(
        mr_num_kernels=config.mr_num_kernels,
        n_kernels=config.n_kernels,
        n_groups=config.n_groups,
        n_kernels_sp=config.n_kernels_sp,
        n_jobs=config.n_jobs,
        random_state=config.seed,
    )
    final_fit_start = time.time()
    final_model.fit(x_full, y_full)
    final_fit_s = time.time() - final_fit_start
    y_te_pred = batched_predict(final_model, x_te_full, config.predict_batch_size, config.max_rul)
    test_metrics = compute_metrics(y_te_full, y_te_pred)

    payload = {
        "subset": config.subset,
        "config": run_mapping,
        "val_metrics": asdict(val_metrics),
        "test_metrics": asdict(test_metrics),
        "val_fit_s": round(val_fit_s, 2),
        "final_fit_s": round(final_fit_s, 2),
        "train_windows": int(len(x_tr)),
        "full_train_windows": int(len(x_full)),
        "val_windows": int(len(x_va)),
        "test_windows": int(len(x_te_full)),
        "train_units": sorted(grouped_tr),
        "val_units": sorted(grouped_val),
        "result_dir": str(out_dir),
    }
    write_json(metrics_path, payload)
    write_json(
        out_dir / "test_predictions.json",
        {
            "subset": config.subset,
            "unit_ids": uid_te_full.tolist(),
            "y_true": y_te_full.tolist(),
            "y_pred": y_te_pred.tolist(),
        },
    )
    write_json(out_dir / "per_unit_metrics.json", per_unit_rows(uid_te_full, y_te_full, y_te_pred))
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at_epoch_s": int(time.time()),
            "config": run_mapping,
            "result_dir": str(out_dir),
        },
    )
    print(json.dumps(payload, indent=2), flush=True)
    print(f"val_rmse={val_metrics.rmse}", flush=True)
    print(f"val_nasa_score={val_metrics.nasa_score}", flush=True)
    print(f"test_rmse={test_metrics.rmse}", flush=True)
    print(f"test_nasa_score={test_metrics.nasa_score}", flush=True)
    return out_dir
