from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def make_windows(df: pd.DataFrame, feature_cols: List[str], window_size: int, stride: int, max_rul: float) -> Tuple[np.ndarray, np.ndarray]:
    x_windows, y_windows = [], []
    for uid in sorted(df["unit"].unique().tolist()):
        sub = df[df["unit"] == uid]
        x = sub[feature_cols].to_numpy(np.float32)
        y = np.clip(sub["rul"].to_numpy(np.float32), 0.0, max_rul)
        for start in range(0, len(sub) - window_size + 1, stride):
            x_windows.append(x[start : start + window_size])
            y_windows.append(y[start + window_size - 1])
    return np.asarray(x_windows, np.float32), np.asarray(y_windows, np.float32)


def make_last_windows(df: pd.DataFrame, feature_cols: List[str], targets: Dict[int, float], window_size: int, max_rul: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_windows, y_windows, units = [], [], []
    for uid in sorted(df["unit"].unique().tolist()):
        sub = df[df["unit"] == uid]
        x = sub[feature_cols].to_numpy(np.float32)
        if len(x) >= window_size:
            win = x[-window_size:]
        else:
            pad = np.repeat(x[:1], window_size - len(x), axis=0)
            win = np.concatenate([pad, x], axis=0)
        x_windows.append(win)
        y_windows.append(min(float(targets[uid]), max_rul))
        units.append(uid)
    return np.asarray(x_windows, np.float32), np.asarray(y_windows, np.float32), np.asarray(units, np.int64)


def make_pseudo_test_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    max_rul: float,
    n_samples_per_unit: int,
    min_history: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_windows, y_windows = [], []
    for uid in sorted(df["unit"].unique().tolist()):
        sub = df[df["unit"] == uid]
        x = sub[feature_cols].to_numpy(np.float32)
        total = len(x)
        max_remaining = total - max(1, min_history)
        if max_remaining < 1:
            continue
        anchors = np.linspace(1, max_remaining, num=n_samples_per_unit + 2, dtype=np.float32)[1:-1]
        remaining_steps = sorted({int(round(v)) for v in anchors if int(round(v)) >= 1})
        for remaining in remaining_steps:
            cutoff = total - remaining
            prefix = x[:cutoff]
            if len(prefix) >= window_size:
                win = prefix[-window_size:]
            else:
                pad = np.repeat(prefix[:1], window_size - len(prefix), axis=0)
                win = np.concatenate([pad, prefix], axis=0)
            x_windows.append(win)
            y_windows.append(min(float(remaining), max_rul))
    return np.asarray(x_windows, np.float32), np.asarray(y_windows, np.float32)
