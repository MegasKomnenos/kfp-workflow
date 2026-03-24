from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_SUBSETS = ("FD001", "FD002", "FD003", "FD004")
VALID_FEATURE_MODES = ("all", "selected", "selected_sensors")
VALID_SCALING_MODES = ("global", "condition")
VALID_VAL_MODES = ("last", "sampled_last", "all_windows")


def read_mapping(path: Path) -> dict[str, Any]:
    raw = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(raw)
    return yaml.safe_load(raw) or {}


def write_yaml(path: Path, payload: Any) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _effective_mr_num_kernels(value: int) -> int:
    effective = (int(value) // 84) * 84
    if effective < 84:
        raise ValueError(f"mr_num_kernels={value} is too small; effective value would be < 84")
    return effective


@dataclass
class ExperimentConfig:
    subset: str
    data_dir: str = "data/cmapss"
    output_dir: str = "results/cmapss_mrhysp_runs"
    feature_mode: str = "selected"
    scaling_mode: str = "condition"
    seq_len: int = 50
    train_stride: int = 2
    max_rul: int = 125
    val_frac: float = 0.2
    val_mode: str = "sampled_last"
    val_samples_per_unit: int = 8
    mr_num_kernels: int = 84
    n_kernels: int = 1
    n_groups: int = 32
    n_kernels_sp: int = 128
    n_jobs: int = 4
    predict_batch_size: int = 512
    seed: int = 42
    run_name: str = ""
    force: bool = False
    download_if_missing: bool = False
    limit_train_units: int = 0
    limit_val_units: int = 0
    limit_test_units: int = 0

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "ExperimentConfig":
        cfg = cls(**mapping)
        return cfg.normalized()

    def normalized(self) -> "ExperimentConfig":
        if self.subset not in VALID_SUBSETS:
            raise ValueError(f"subset must be one of {VALID_SUBSETS}")
        if self.feature_mode not in VALID_FEATURE_MODES:
            raise ValueError(f"feature_mode must be one of {VALID_FEATURE_MODES}")
        if self.scaling_mode not in VALID_SCALING_MODES:
            raise ValueError(f"scaling_mode must be one of {VALID_SCALING_MODES}")
        if self.val_mode not in VALID_VAL_MODES:
            raise ValueError(f"val_mode must be one of {VALID_VAL_MODES}")
        if self.feature_mode == "selected_sensors":
            self.scaling_mode = "global"
        self.mr_num_kernels = _effective_mr_num_kernels(self.mr_num_kernels)
        return self

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_name"] = self.run_name or build_result_dir_name(payload)
        return payload


def build_result_dir_name(config: dict[str, Any]) -> str:
    run_name = str(config.get("run_name") or "")
    if run_name:
        return run_name
    return (
        f"{str(config['subset']).lower()}_mrhysp_{config['feature_mode']}_{config['scaling_mode']}"
        f"_sl{config['seq_len']}_ts{config['train_stride']}_mrk{config['mr_num_kernels']}"
        f"_hk{config['n_kernels']}_hg{config['n_groups']}_sp{config['n_kernels_sp']}"
        f"_mrul{config['max_rul']}_vf{int(round(float(config['val_frac']) * 100))}"
        f"_vm{config['val_mode']}_vp{config['val_samples_per_unit']}_seed{config['seed']}"
    )


def expand_ablation_spec(payload: dict[str, Any]) -> list[ExperimentConfig]:
    fixed = dict(payload.get("fixed") or {})
    axes = payload.get("axes") or {}
    include = payload.get("include") or []
    exclude_rules = payload.get("exclude_rules") or []

    axis_names = list(axes.keys())
    products = []
    if axis_names:
        for values in itertools.product(*(axes[name] for name in axis_names)):
            cfg = dict(fixed)
            cfg.update(dict(zip(axis_names, values)))
            products.append(cfg)
    else:
        products.append(dict(fixed))
    products.extend(include)

    configs: list[ExperimentConfig] = []
    seen: set[str] = set()
    for item in products:
        cfg = ExperimentConfig.from_mapping(dict(item))
        if _matches_exclude_rule(cfg.to_mapping(), exclude_rules):
            continue
        run_name = build_result_dir_name(cfg.to_mapping())
        if run_name in seen:
            continue
        seen.add(run_name)
        configs.append(cfg)
    return configs


def _matches_exclude_rule(config: dict[str, Any], rules: list[dict[str, Any]]) -> bool:
    for rule in rules:
        when = rule.get("when") or {}
        if all(config.get(key) == value for key, value in when.items()):
            return True
    return False

