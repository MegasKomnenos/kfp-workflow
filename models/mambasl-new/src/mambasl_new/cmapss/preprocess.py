from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .constants import OP_COLS, SENSOR_14


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (max_cycle - out["cycle"]).astype(np.float32)
    return out


def choose_val_units(unit_ids: np.ndarray, seed: int, frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unit_ids = np.array(sorted(unit_ids.tolist()), dtype=int)
    n_val = max(1, int(round(len(unit_ids) * frac)))
    val_units = np.array(sorted(rng.choice(unit_ids, size=n_val, replace=False).tolist()), dtype=int)
    train_units = np.array([u for u in unit_ids if u not in set(val_units.tolist())], dtype=int)
    return train_units, val_units


def get_feature_cols(feature_mode: str) -> List[str]:
    if feature_mode == "sensors_only":
        return SENSOR_14[:]
    if feature_mode == "settings_plus_sensors":
        return OP_COLS + SENSOR_14
    raise ValueError(feature_mode)


@dataclass
class ConditionNormalizer:
    mode: str
    n_conditions: int
    seed: int

    def __post_init__(self) -> None:
        self.kmeans = None
        self.scalers: Dict[int, object] = {}

    def fit(self, ops: np.ndarray, feats: np.ndarray) -> None:
        if self.mode == "global_standard":
            scaler = StandardScaler()
            scaler.fit(feats)
            self.scalers[0] = scaler
            return
        if self.n_conditions == 1:
            labels = np.zeros(len(ops), dtype=int)
        else:
            self.kmeans = KMeans(n_clusters=self.n_conditions, random_state=self.seed, n_init=20)
            labels = self.kmeans.fit_predict(ops)
        for cid in np.unique(labels):
            idx = labels == cid
            scaler = MinMaxScaler(feature_range=(-1.0, 1.0)) if self.mode == "condition_minmax" else StandardScaler()
            scaler.fit(feats[idx])
            self.scalers[int(cid)] = scaler

    def _labels(self, ops: np.ndarray) -> np.ndarray:
        if self.mode == "global_standard" or self.kmeans is None:
            return np.zeros(len(ops), dtype=int)
        return self.kmeans.predict(ops)

    def transform(self, ops: np.ndarray, feats: np.ndarray) -> np.ndarray:
        labels = self._labels(ops)
        out = np.empty_like(feats, dtype=np.float32)
        for cid in np.unique(labels):
            idx = labels == cid
            out[idx] = self.scalers[int(cid)].transform(feats[idx]).astype(np.float32)
        return out


def preprocess_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_mode: str,
    norm_mode: str,
    n_conditions: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = get_feature_cols(feature_mode)
    norm = ConditionNormalizer(norm_mode, n_conditions=n_conditions, seed=seed)
    norm.fit(train_df[OP_COLS].to_numpy(np.float32), train_df[feature_cols].to_numpy(np.float32))

    def _tx(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.loc[:, feature_cols] = norm.transform(
            out[OP_COLS].to_numpy(np.float32),
            out[feature_cols].to_numpy(np.float32),
        )
        return out

    return _tx(train_df), _tx(val_df), _tx(test_df)
