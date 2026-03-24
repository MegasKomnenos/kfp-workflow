from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from sklearn.preprocessing import StandardScaler


NASA_CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

OPS = [f"op{i}" for i in range(1, 4)]
SENSORS = [f"s{i}" for i in range(1, 22)]
ALL_FEATURES = OPS + SENSORS
SELECTED_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]
SELECTED_WITH_OPS = OPS + SELECTED_SENSORS


def ensure_cmapss_downloaded(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    expected = data_dir / "train_FD001.txt"
    if expected.exists():
        return
    zip_path = data_dir / "CMAPSSData.zip"
    if not zip_path.exists():
        urlretrieve(NASA_CMAPSS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)


def load_split(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def load_rul_targets(path: Path) -> np.ndarray:
    return np.loadtxt(path).reshape(-1)


def feature_columns(feature_mode: str) -> list[int]:
    if feature_mode == "all":
        names = ALL_FEATURES
    elif feature_mode == "selected":
        names = SELECTED_WITH_OPS
    else:
        names = SELECTED_SENSORS
    name_to_idx = {name: index for index, name in enumerate(ALL_FEATURES)}
    return [name_to_idx[name] for name in names]


def group_by_unit(arr: np.ndarray) -> dict[int, np.ndarray]:
    unit_ids = arr[:, 0].astype(int)
    return {int(uid): arr[unit_ids == uid] for uid in np.unique(unit_ids)}


def build_train_rul(unit_arr: np.ndarray, max_rul: int) -> np.ndarray:
    cycles = unit_arr[:, 1]
    rul = cycles.max() - cycles
    return np.clip(rul, 0, max_rul)


def extract_feature_groups(grouped: dict[int, np.ndarray], feature_idx: list[int]) -> dict[int, np.ndarray]:
    return {
        uid: unit[:, 2:][:, feature_idx].astype(np.float32, copy=False)
        for uid, unit in grouped.items()
    }


def round_condition(vec: np.ndarray) -> tuple[float, float, float]:
    if len(vec) < 3:
        return (0.0, 0.0, 0.0)
    return tuple(float(np.round(v, 3)) for v in vec[:3])


def fit_scalers(
    train_feats: dict[int, np.ndarray],
    scaling_mode: str,
) -> tuple[StandardScaler, dict[tuple[float, float, float], StandardScaler]]:
    all_rows = np.concatenate([train_feats[uid] for uid in sorted(train_feats)], axis=0)
    global_scaler = StandardScaler().fit(all_rows)
    if scaling_mode == "global":
        return global_scaler, {}
    buckets: dict[tuple[float, float, float], list[np.ndarray]] = {}
    for uid in sorted(train_feats):
        for row in train_feats[uid]:
            buckets.setdefault(round_condition(row), []).append(row)
    cond_scalers = {
        key: StandardScaler().fit(np.asarray(rows, dtype=np.float32))
        for key, rows in buckets.items()
        if len(rows) >= 8
    }
    return global_scaler, cond_scalers


def apply_scalers(
    feats_by_unit: dict[int, np.ndarray],
    global_scaler: StandardScaler,
    cond_scalers: dict[tuple[float, float, float], StandardScaler],
    scaling_mode: str,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if scaling_mode == "global":
        for uid, feats in feats_by_unit.items():
            out[uid] = global_scaler.transform(feats).astype(np.float32, copy=False)
        return out
    for uid, feats in feats_by_unit.items():
        scaled = np.empty_like(feats, dtype=np.float32)
        for index, row in enumerate(feats):
            scaler = cond_scalers.get(round_condition(row), global_scaler)
            scaled[index] = scaler.transform(row.reshape(1, -1))[0]
        out[uid] = scaled
    return out


def make_train_windows(
    raw_grouped: dict[int, np.ndarray],
    feat_grouped: dict[int, np.ndarray],
    seq_len: int,
    max_rul: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for uid in sorted(raw_grouped):
        unit_raw = raw_grouped[uid]
        feats = feat_grouped[uid]
        if len(unit_raw) < seq_len:
            continue
        rul = build_train_rul(unit_raw, max_rul)
        for end in range(seq_len, len(unit_raw) + 1, stride):
            start = end - seq_len
            xs.append(feats[start:end].T)
            ys.append(rul[end - 1])
    if not xs:
        raise ValueError("No training windows generated.")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def make_val_windows(
    raw_grouped: dict[int, np.ndarray],
    feat_grouped: dict[int, np.ndarray],
    seq_len: int,
    max_rul: int,
    mode: str,
    samples_per_unit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "all_windows":
        return make_train_windows(raw_grouped, feat_grouped, seq_len, max_rul, stride=1)
    rng = np.random.RandomState(seed)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for uid in sorted(raw_grouped):
        length = len(raw_grouped[uid])
        feats = feat_grouped[uid]
        if length < seq_len:
            continue
        if mode == "last":
            end_points = np.asarray([length], dtype=int)
        else:
            candidates = np.arange(seq_len, length + 1)
            if len(candidates) > samples_per_unit:
                end_points = np.sort(rng.choice(candidates, size=samples_per_unit, replace=False))
            else:
                end_points = candidates
        for end in end_points:
            start = end - seq_len
            xs.append(feats[start:end].T)
            target = 0.0 if mode == "last" else min(float(length - end), float(max_rul))
            ys.append(target)
    if not xs:
        raise ValueError("No validation windows generated.")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def make_test_windows(
    grouped: dict[int, np.ndarray],
    feat_grouped: dict[int, np.ndarray],
    seq_len: int,
    true_rul: np.ndarray,
    max_rul: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    uids: list[int] = []
    for index, uid in enumerate(sorted(grouped)):
        feats = feat_grouped[uid]
        if len(feats) >= seq_len:
            window = feats[-seq_len:]
        else:
            pad = np.repeat(feats[:1], seq_len - len(feats), axis=0)
            window = np.concatenate([pad, feats], axis=0)
        xs.append(window.T)
        ys.append(min(float(true_rul[index]), float(max_rul)))
        uids.append(uid)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(uids)


def limited_units(grouped: dict[int, np.ndarray], limit: int) -> dict[int, np.ndarray]:
    if limit <= 0:
        return grouped
    selected = sorted(grouped)[:limit]
    return {uid: grouped[uid] for uid in selected}

