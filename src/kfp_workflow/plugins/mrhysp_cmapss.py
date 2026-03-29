"""MR-HY-SP C-MAPSS model plugin.

Adapts the ``multirocket_new`` package (MRHySPRegressor: MultiRocket + HYDRA
+ SPRocket ensemble with RidgeCV) to the ``ModelPlugin`` ABC.

Unlike the PyTorch-based MambaSL plugin, MRHySP is entirely sklearn-based:
- Single ``model.fit(x, y)`` call (no epochs, learning rate, patience)
- Serialisation via ``joblib.dump / joblib.load``
- Data arrays are channels-first ``(N, C, T)`` float64
- CPU-only (no GPU support)

Only pure-ML modules are imported from the model package:
``multirocket_new.cmapss``, ``.model``, ``.runner``, ``.config``.
Orchestration modules (specs, experiment, search_space, kubeflow, cli) are
*not* imported — kfp-workflow owns that layer.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict

from kfp_workflow.plugins.base import (
    EvalResult,
    LoadDataResult,
    ModelPlugin,
    PreprocessResult,
    SaveResult,
    TrainResult,
)
from kfp_workflow.plugins.cmapss_utils import (
    cmapss_storage_root,
    resolve_cmapss_data_dir,
)


# ---------------------------------------------------------------------------
# Plugin config schemas — used for --set validation and documentation
# ---------------------------------------------------------------------------

class MRHySPModelConfig(BaseModel):
    """Schema for ``model.config`` accepted by the mrhysp-cmapss plugin."""

    mr_num_kernels: int = 84
    n_kernels: int = 1
    n_groups: int = 32
    n_kernels_sp: int = 128
    n_jobs: int = 4
    seq_len: int = 50
    train_stride: int = 2
    max_rul: int = 125
    val_mode: str = "sampled_last"
    val_samples_per_unit: int = 8
    predict_batch_size: int = 512

    model_config = ConfigDict(extra="allow")


class MRHySPDatasetConfig(BaseModel):
    """Schema for ``dataset.config`` accepted by the mrhysp-cmapss plugin."""

    fd_name: str = "FD001"
    feature_mode: str = "selected"
    scaling_mode: str = "condition"
    download_policy: Literal["never", "if_missing", "always"] = "never"
    download_if_missing: Optional[bool] = None

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effective_mr_num_kernels(value: int) -> int:
    """Round *value* down to the nearest multiple of 84.

    MultiRocket requires ``num_kernels`` to be a multiple of 84.
    Mirrors ``multirocket_new.config._effective_mr_num_kernels``.
    """
    effective = (int(value) // 84) * 84
    if effective < 84:
        raise ValueError(
            f"mr_num_kernels={value} is too small; effective value would be < 84"
        )
    return effective


def _build_cfg(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Merge model.config + train params into the flat dict expected by
    the MRHySP pipeline stages.

    Applies the ``mr_num_kernels`` multiple-of-84 constraint.
    """
    cfg: Dict[str, Any] = dict(spec["model"].get("config", {}))
    train = spec.get("train", {})

    # Map train-section fields
    cfg.setdefault("seed", train.get("seed", 42))
    cfg.setdefault("val_frac", train.get("val_split", 0.2))

    # Ensure all required keys have sensible defaults
    cfg.setdefault("mr_num_kernels", 84)
    cfg.setdefault("n_kernels", 1)
    cfg.setdefault("n_groups", 32)
    cfg.setdefault("n_kernels_sp", 128)
    cfg.setdefault("n_jobs", 4)
    cfg.setdefault("seq_len", 50)
    cfg.setdefault("train_stride", 2)
    cfg.setdefault("max_rul", 125)
    cfg.setdefault("val_mode", "sampled_last")
    cfg.setdefault("val_samples_per_unit", 8)
    cfg.setdefault("predict_batch_size", 512)

    # Enforce mr_num_kernels constraint (must be multiple of 84)
    cfg["mr_num_kernels"] = _effective_mr_num_kernels(int(cfg["mr_num_kernels"]))

    return cfg


def _download_policy(ds_cfg: Dict[str, Any]) -> str:
    """Normalize legacy and current download policy fields."""
    if "download_policy" in ds_cfg:
        return str(ds_cfg.get("download_policy") or "never")

    if "download_if_missing" in ds_cfg:
        warnings.warn(
            "dataset.config.download_if_missing is deprecated; "
            "use dataset.config.download_policy instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "if_missing" if ds_cfg.get("download_if_missing") else "never"

    return "never"


def _resolve_cmapss_data_dir(data_mount_path: str) -> Path:
    """Backward-compatible alias for the shared C-MAPSS resolver."""
    return resolve_cmapss_data_dir(data_mount_path)


class MRHySPCmapssPlugin(ModelPlugin):
    """MR-HY-SP ensemble trained/evaluated on C-MAPSS turbofan degradation data.

    MRHySPRegressor combines HYDRA transforms, MultiRocket features, and
    SPRocket prototype distances with RidgeCV regression.  Training is a
    single ``fit()`` call (no gradient descent).
    """

    @staticmethod
    def name() -> str:
        return "mrhysp-cmapss"

    @classmethod
    def model_config_schema(cls):
        return MRHySPModelConfig

    @classmethod
    def dataset_config_schema(cls):
        return MRHySPDatasetConfig

    @classmethod
    def serving_model_filenames(cls):
        return ["model.joblib", "model.pt"]

    # -- Stage 1: load_data ------------------------------------------------

    def load_data(
        self,
        spec: Dict[str, Any],
        data_mount_path: str,
    ) -> LoadDataResult:
        import numpy as np
        from multirocket_new.cmapss import (
            ensure_cmapss_downloaded,
            load_rul_targets,
            load_split,
        )

        ds_cfg = spec["dataset"].get("config", {})
        fd_name = ds_cfg.get("fd_name", "FD001")
        storage_root = cmapss_storage_root(data_mount_path)
        download_policy = _download_policy(ds_cfg)

        if download_policy in {"if_missing", "always"}:
            ensure_cmapss_downloaded(storage_root)

        data_dir = resolve_cmapss_data_dir(data_mount_path)

        train_raw = load_split(data_dir / f"train_{fd_name}.txt")
        test_raw = load_split(data_dir / f"test_{fd_name}.txt")
        test_rul = load_rul_targets(data_dir / f"RUL_{fd_name}.txt")

        n_train_units = len(np.unique(train_raw[:, 0].astype(int)))
        n_test_units = len(np.unique(test_raw[:, 0].astype(int)))

        return LoadDataResult(
            data_dir=str(data_dir),
            dataset_name=fd_name,
            num_train_samples=n_train_units,
            num_test_samples=n_test_units,
            metadata={
                "train_rows": int(len(train_raw)),
                "test_rows": int(len(test_raw)),
                "rul_test_count": int(len(test_rul)),
            },
        )

    # -- Stage 2: preprocess -----------------------------------------------

    def preprocess(
        self,
        spec: Dict[str, Any],
        load_result: LoadDataResult,
        artifacts_dir: str,
    ) -> PreprocessResult:
        import joblib
        import numpy as np
        from multirocket_new.cmapss import (
            apply_scalers,
            extract_feature_groups,
            feature_columns,
            fit_scalers,
            group_by_unit,
            load_rul_targets,
            load_split,
            make_test_windows,
            make_train_windows,
            make_val_windows,
        )

        ds_cfg = spec["dataset"].get("config", {})
        cfg = _build_cfg(spec)
        fd_name = load_result.dataset_name
        data_dir = Path(load_result.data_dir)

        # Reload raw arrays
        train_raw = load_split(data_dir / f"train_{fd_name}.txt")
        test_raw = load_split(data_dir / f"test_{fd_name}.txt")
        test_rul = load_rul_targets(data_dir / f"RUL_{fd_name}.txt")

        feature_idx = feature_columns(ds_cfg.get("feature_mode", "selected"))

        # Train/val unit split (same logic as runner.py)
        grouped_train = group_by_unit(train_raw)
        grouped_test = group_by_unit(test_raw)

        train_units = sorted(np.unique(train_raw[:, 0].astype(int)).tolist())
        rng = np.random.RandomState(int(cfg["seed"]))
        rng.shuffle(train_units)
        n_val_units = max(1, int(len(train_units) * float(cfg["val_frac"])))
        val_units = train_units[:n_val_units]
        tr_units = train_units[n_val_units:]

        grouped_tr = {uid: grouped_train[uid] for uid in grouped_train if uid in set(tr_units)}
        grouped_val = {uid: grouped_train[uid] for uid in grouped_train if uid in set(val_units)}

        # Feature extraction and scaling
        scaling_mode = ds_cfg.get("scaling_mode", "condition")
        raw_feats_tr = extract_feature_groups(grouped_tr, feature_idx)
        raw_feats_val = extract_feature_groups(grouped_val, feature_idx)
        raw_feats_te = extract_feature_groups(grouped_test, feature_idx)

        global_scaler, cond_scalers = fit_scalers(raw_feats_tr, scaling_mode)
        feats_tr = apply_scalers(raw_feats_tr, global_scaler, cond_scalers, scaling_mode)
        feats_val = apply_scalers(raw_feats_val, global_scaler, cond_scalers, scaling_mode)
        feats_te = apply_scalers(raw_feats_te, global_scaler, cond_scalers, scaling_mode)

        # Windowing
        seq_len = int(cfg["seq_len"])
        max_rul = int(cfg["max_rul"])
        train_stride = int(cfg["train_stride"])

        x_train, y_train = make_train_windows(
            grouped_tr, feats_tr, seq_len, max_rul, train_stride,
        )
        x_val, y_val = make_val_windows(
            grouped_val, feats_val, seq_len, max_rul,
            cfg["val_mode"], int(cfg["val_samples_per_unit"]), int(cfg["seed"]),
        )
        x_test, y_test, _ = make_test_windows(
            grouped_test, feats_te, seq_len, test_rul, max_rul,
        )

        # Determine feature_dim from channels dimension
        # Arrays are (N, C, T) — channels-first
        feature_dim = int(x_train.shape[1])

        # Save artifacts
        out = Path(artifacts_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths = {}
        for arr_name, arr in [
            ("x_train", x_train), ("y_train", y_train),
            ("x_val", x_val), ("y_val", y_val),
            ("x_test", x_test), ("y_test", y_test),
        ]:
            p = out / f"{arr_name}.npy"
            np.save(str(p), arr)
            paths[arr_name] = str(p)

        # Save scalers for serving
        joblib.dump(global_scaler, str(out / "global_scaler.joblib"))
        if cond_scalers:
            joblib.dump(cond_scalers, str(out / "cond_scalers.joblib"))

        return PreprocessResult(
            artifacts_dir=str(out),
            x_train_path=paths["x_train"],
            y_train_path=paths["y_train"],
            x_val_path=paths["x_val"],
            y_val_path=paths["y_val"],
            x_test_path=paths["x_test"],
            y_test_path=paths["y_test"],
            feature_dim=feature_dim,
            seq_len=seq_len,
            num_train=int(x_train.shape[0]),
            num_val=int(x_val.shape[0]),
            num_test=int(x_test.shape[0]),
        )

    # -- Stage 3: train ----------------------------------------------------

    def train(
        self,
        spec: Dict[str, Any],
        preprocess_result: PreprocessResult,
        model_dir: str,
    ) -> TrainResult:
        import joblib
        import numpy as np
        from multirocket_new.model import MRHySPRegressor
        from multirocket_new.runner import batched_predict, compute_metrics

        cfg = _build_cfg(spec)

        x_train = np.load(preprocess_result.x_train_path)
        y_train = np.load(preprocess_result.y_train_path)
        x_val = np.load(preprocess_result.x_val_path)
        y_val = np.load(preprocess_result.y_val_path)

        np.random.seed(int(cfg["seed"]))

        model = MRHySPRegressor(
            mr_num_kernels=int(cfg["mr_num_kernels"]),
            n_kernels=int(cfg["n_kernels"]),
            n_groups=int(cfg["n_groups"]),
            n_kernels_sp=int(cfg["n_kernels_sp"]),
            n_jobs=int(cfg["n_jobs"]),
            random_state=int(cfg["seed"]),
        )
        model.fit(x_train, y_train)

        # Evaluate on validation set
        y_val_pred = batched_predict(
            model, x_val, int(cfg["predict_batch_size"]), int(cfg["max_rul"]),
        )
        val_metrics = compute_metrics(y_val, y_val_pred)

        # Save model
        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)
        model_path = str(out / "model.joblib")
        joblib.dump(model, model_path)

        config_path = out / "model_config.json"
        config_path.write_text(json.dumps({
            "cfg": cfg,
            "feature_dim": preprocess_result.feature_dim,
            "seq_len": preprocess_result.seq_len,
        }, indent=2))

        return TrainResult(
            model_path=model_path,
            best_epoch=0,
            train_loss=float(val_metrics.rmse),
            val_loss=float(val_metrics.rmse),
            metadata={
                "val_rmse": float(val_metrics.rmse),
                "val_score": float(val_metrics.nasa_score),
                "val_mae": float(val_metrics.mae),
            },
        )

    # -- Stage 4: evaluate -------------------------------------------------

    def evaluate(
        self,
        spec: Dict[str, Any],
        train_result: TrainResult,
        preprocess_result: PreprocessResult,
    ) -> EvalResult:
        import joblib
        import numpy as np
        from multirocket_new.runner import batched_predict, compute_metrics

        cfg = _build_cfg(spec)

        model = joblib.load(train_result.model_path)

        x_test = np.load(preprocess_result.x_test_path)
        y_test = np.load(preprocess_result.y_test_path)

        y_pred = batched_predict(
            model, x_test, int(cfg["predict_batch_size"]), int(cfg["max_rul"]),
        )
        metrics = compute_metrics(y_test, y_pred)

        return EvalResult(
            metrics={
                "rmse": float(metrics.rmse),
                "score": float(metrics.nasa_score),
                "mae": float(metrics.mae),
                "n_test": int(metrics.n_samples),
            },
            model_path=train_result.model_path,
        )

    # -- Stage 5: save_model -----------------------------------------------

    def save_model(
        self,
        spec: Dict[str, Any],
        train_result: TrainResult,
        eval_result: EvalResult,
        final_model_dir: str,
    ) -> SaveResult:
        from kfp_workflow.registry.model_registry import FileModelRegistry

        model_name = spec["model"]["name"]
        model_version = spec["model"].get("version", "v1")

        final = Path(final_model_dir)
        final.mkdir(parents=True, exist_ok=True)

        # Copy model file
        src_model = Path(train_result.model_path)
        dst_model = final / "model.joblib"
        if src_model.resolve() != dst_model.resolve():
            shutil.copy2(str(src_model), str(dst_model))

        # Copy model config
        src_cfg = src_model.parent / "model_config.json"
        dst_cfg = final / "model_config.json"
        if src_cfg.exists() and src_cfg.resolve() != dst_cfg.resolve():
            shutil.copy2(str(src_cfg), str(dst_cfg))

        # Register in model registry
        registry_path = str(
            Path(spec["storage"]["model_mount_path"]) / ".model_registry.json"
        )
        try:
            registry = FileModelRegistry(registry_path=registry_path)
            registry.register_model(
                name=model_name,
                version=model_version,
                uri=str(dst_model),
                framework="sklearn",
                description=(
                    f"MR-HY-SP C-MAPSS "
                    f"{spec['dataset'].get('config', {}).get('fd_name', '')}"
                ),
                parameters={
                    "metrics": eval_result.metrics,
                    "config": _build_cfg(spec),
                },
            )
        except Exception:
            pass  # registry write may fail outside cluster; non-fatal

        return SaveResult(
            saved_path=str(dst_model),
            model_name=model_name,
            model_version=model_version,
        )

    # -- Inference ----------------------------------------------------------

    def load_serving_artifact(
        self,
        model_path: str,
        model_config: Dict[str, Any],
    ) -> Any:
        import joblib

        return joblib.load(model_path)

    def warmup_serving_artifact(
        self,
        artifact: Any,
        model_config: Dict[str, Any],
    ) -> None:
        import numpy as np

        feature_dim = int(model_config.get("feature_dim", 1))
        seq_len = int(model_config.get("seq_len", 1))
        sample = np.zeros((1, feature_dim, seq_len), dtype=np.float64)
        artifact.predict(sample)

    def predict_loaded(
        self,
        artifact: Any,
        input_data: Any,
        model_config: Dict[str, Any],
    ) -> Any:
        import numpy as np

        cfg = model_config.get("cfg", model_config)
        max_rul = float(cfg.get("max_rul", 125))
        preds = artifact.predict(np.asarray(input_data, dtype=np.float64))
        return np.clip(preds, 0.0, max_rul)

    def predict(
        self,
        model_path: str,
        input_data: Any,
        model_config: Dict[str, Any],
    ) -> Any:
        artifact = self.load_serving_artifact(model_path, model_config)
        return self.predict_loaded(artifact, input_data, model_config)

    # -- HPO hooks ----------------------------------------------------------

    def hpo_search_space(
        self,
        spec: Dict[str, Any],
        profile: str,
    ) -> list:
        from kfp_workflow.specs import SearchParamSpec

        if profile == "aggressive":
            return [
                SearchParamSpec(name="mr_num_kernels", type="int", low=84, high=840, step=84),
                SearchParamSpec(name="n_kernels", type="int", low=1, high=6, step=1),
                SearchParamSpec(name="n_groups", type="int", low=16, high=160, step=16),
                SearchParamSpec(name="n_kernels_sp", type="int", low=64, high=640, step=64),
                SearchParamSpec(name="seed", type="categorical", values=[7, 13, 21, 42, 84]),
            ]
        # "default" profile
        return [
            SearchParamSpec(name="mr_num_kernels", type="int", low=84, high=588, step=84),
            SearchParamSpec(name="n_kernels", type="int", low=1, high=4, step=1),
            SearchParamSpec(name="n_groups", type="int", low=16, high=128, step=16),
            SearchParamSpec(name="n_kernels_sp", type="int", low=64, high=512, step=64),
            SearchParamSpec(name="seed", type="categorical", values=[7, 13, 21, 42]),
        ]

    def hpo_base_config(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        return _build_cfg(spec)

    def hpo_objective(
        self,
        spec: Dict[str, Any],
        params: Dict[str, Any],
        data_mount_path: str,
    ) -> float:
        import numpy as np
        from multirocket_new.model import MRHySPRegressor
        from multirocket_new.runner import batched_predict, rmse

        from kfp_workflow.tune.exceptions import TrialPruned

        # --- Data loading & preprocessing (cached across trials) ----------
        ds_cfg = spec["dataset"].get("config", {})
        cfg = _build_cfg(spec)
        cache_key = (
            ds_cfg.get("fd_name", "FD001"),
            int(cfg["seq_len"]),
            int(cfg["max_rul"]),
            ds_cfg.get("feature_mode", "selected"),
            ds_cfg.get("scaling_mode", "condition"),
            cfg["val_mode"],
        )

        if not hasattr(self, "_hpo_cache") or self._hpo_cache_key != cache_key:
            load_result = self.load_data(spec, data_mount_path)
            import tempfile
            artifacts_dir = tempfile.mkdtemp(prefix="hpo_trial_")
            preprocess_result = self.preprocess(spec, load_result, artifacts_dir)
            self._hpo_cache = {
                "x_train": np.load(preprocess_result.x_train_path),
                "y_train": np.load(preprocess_result.y_train_path),
                "x_val": np.load(preprocess_result.x_val_path),
                "y_val": np.load(preprocess_result.y_val_path),
            }
            self._hpo_cache_key = cache_key

        x_train = self._hpo_cache["x_train"]
        y_train = self._hpo_cache["y_train"]
        x_val = self._hpo_cache["x_val"]
        y_val = self._hpo_cache["y_val"]

        if len(x_train) == 0 or len(x_val) == 0:
            raise TrialPruned()

        # --- Enforce mr_num_kernels constraint ----------------------------
        mr_num_kernels = _effective_mr_num_kernels(
            int(params.get("mr_num_kernels", cfg["mr_num_kernels"]))
        )

        seed = int(params.get("seed", cfg["seed"]))
        np.random.seed(seed)

        model = MRHySPRegressor(
            mr_num_kernels=mr_num_kernels,
            n_kernels=int(params.get("n_kernels", cfg["n_kernels"])),
            n_groups=int(params.get("n_groups", cfg["n_groups"])),
            n_kernels_sp=int(params.get("n_kernels_sp", cfg["n_kernels_sp"])),
            n_jobs=int(cfg["n_jobs"]),
            random_state=seed,
        )
        model.fit(x_train, y_train)

        y_val_pred = batched_predict(
            model, x_val, int(cfg["predict_batch_size"]), int(cfg["max_rul"]),
        )
        return float(rmse(y_val, y_val_pred))
