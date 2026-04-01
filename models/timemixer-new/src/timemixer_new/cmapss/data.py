from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests

from ..specs import DataSpec
from .constants import COLS


def ensure_cmapss_downloaded(data_spec: DataSpec) -> Tuple[Path, Path]:
    data_root = Path(data_spec.data_root)
    zip_path = data_root / "CMAPSSData.zip"
    extract_dir = data_root / "CMAPSSData"
    data_root.mkdir(parents=True, exist_ok=True)

    should_download = data_spec.download_policy == "always"
    if data_spec.download_policy == "never":
        if not zip_path.exists() and not (extract_dir / "train_FD001.txt").exists():
            raise FileNotFoundError(f"C-MAPSS data missing under {data_root}")
    elif not (extract_dir / "train_FD001.txt").exists():
        should_download = True

    if should_download:
        with requests.get(data_spec.nasa_url, stream=True, timeout=120, headers={"User-Agent": "timemixer-new"}) as response:
            response.raise_for_status()
            with zip_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)

    if not zip_path.exists():
        return zip_path, extract_dir

    md5 = hashlib.md5(zip_path.read_bytes()).hexdigest()
    if md5 != data_spec.nasa_md5:
        raise RuntimeError(f"C-MAPSS zip MD5 mismatch: expected {data_spec.nasa_md5}, got {md5}")

    if should_download or not (extract_dir / "train_FD001.txt").exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)

    return zip_path, extract_dir


def load_fd(extract_dir: Path, fd_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    train = pd.read_csv(extract_dir / f"train_{fd_name}.txt", sep=r"\s+", header=None).iloc[:, :26]
    test = pd.read_csv(extract_dir / f"test_{fd_name}.txt", sep=r"\s+", header=None).iloc[:, :26]
    rul = pd.read_csv(extract_dir / f"RUL_{fd_name}.txt", sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=np.float32)
    train.columns = COLS
    test.columns = COLS
    return train, test, rul
