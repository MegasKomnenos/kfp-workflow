"""MambaSL C-MAPSS model plugin.

Adapts the ``mambasl_new.cmapss.*`` modules to the ``ModelPlugin`` ABC.
All ML logic lives in the ``mambasl-new`` package — this file is purely
an adapter that bridges the plugin interface to existing functions.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

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

class MambaSLModelConfig(BaseModel):
    """Schema for ``model.config`` accepted by the mambasl-cmapss plugin."""

    d_model: int = 64
    d_state: int = 16
    d_conv: int = 3
    expand: int = 2
    num_kernels: int = 5
    tv_dt: bool = True
    tv_B: bool = True
    tv_C: bool = True
    use_D: bool = True
    projection: str = "last"
    dropout: float = 0.2
    huber_delta: float = 2.0
    window_size: int = 50
    max_rul: float = 125.0

    model_config = ConfigDict(extra="allow")


class MambaSLDatasetConfig(BaseModel):
    """Schema for ``dataset.config`` accepted by the mambasl-cmapss plugin."""

    fd_name: str = "FD001"
    download_policy: str = "if_missing"
    feature_mode: str = "settings_plus_sensors"
    norm_mode: str = "condition_minmax"

    model_config = ConfigDict(extra="allow")


def _build_cfg(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Merge model.config + train params into the flat dict expected by
    ``mambasl_new.cmapss.train.train_model``.
    """
    cfg: Dict[str, Any] = dict(spec["model"].get("config", {}))
    train = spec.get("train", {})
    cfg.setdefault("batch_size", train.get("batch_size", 64))
    cfg.setdefault("lr", train.get("learning_rate", 1e-3))
    cfg.setdefault("weight_decay", train.get("weight_decay", 1e-4))
    # Ensure all required keys have sensible defaults
    cfg.setdefault("d_model", 64)
    cfg.setdefault("d_state", 16)
    cfg.setdefault("d_conv", 3)
    cfg.setdefault("expand", 2)
    cfg.setdefault("num_kernels", 5)
    cfg.setdefault("tv_dt", True)
    cfg.setdefault("tv_B", True)
    cfg.setdefault("tv_C", True)
    cfg.setdefault("use_D", True)
    cfg.setdefault("projection", "last")
    cfg.setdefault("dropout", 0.2)
    cfg.setdefault("huber_delta", 2.0)
    cfg.setdefault("window_size", 50)
    cfg.setdefault("max_rul", 125.0)
    return cfg


class MambaSLCmapssPlugin(ModelPlugin):
    """MambaSL model trained/evaluated on C-MAPSS turbofan degradation data."""

    @staticmethod
    def name() -> str:
        return "mambasl-cmapss"

    @classmethod
    def model_config_schema(cls):
        return MambaSLModelConfig

    @classmethod
    def dataset_config_schema(cls):
        return MambaSLDatasetConfig

    # -- Stage 1: load_data ------------------------------------------------

    def load_data(
        self,
        spec: Dict[str, Any],
        data_mount_path: str,
    ) -> LoadDataResult:
        from mambasl_new.cmapss.constants import FD_CONFIGS
        from mambasl_new.cmapss.data import ensure_cmapss_downloaded, load_fd
        from mambasl_new.specs import DataSpec

        ds_cfg = spec["dataset"].get("config", {})
        fd_name = ds_cfg.get("fd_name", "FD001")
        storage_root = cmapss_storage_root(data_mount_path)

        data_spec = DataSpec(
            data_root=str(storage_root),
            download_policy=ds_cfg.get("download_policy", "if_missing"),
            nasa_url=ds_cfg.get(
                "nasa_url",
                "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
            ),
            nasa_md5=ds_cfg.get("nasa_md5", "79a22f36e80606c69d0e9e4da5bb2b7a"),
        )

        _, extract_dir = ensure_cmapss_downloaded(data_spec)
        data_dir = resolve_cmapss_data_dir(str(extract_dir))
        train_df, test_df, rul_test = load_fd(data_dir, fd_name)

        return LoadDataResult(
            data_dir=str(data_dir),
            dataset_name=fd_name,
            num_train_samples=len(train_df),
            num_test_samples=len(test_df),
            metadata={
                "fd_config": FD_CONFIGS.get(fd_name, {}),
                "rul_test_count": int(len(rul_test)),
            },
        )

    # -- Stage 2: preprocess -----------------------------------------------

    def preprocess(
        self,
        spec: Dict[str, Any],
        load_result: LoadDataResult,
        artifacts_dir: str,
    ) -> PreprocessResult:
        import numpy as np
        from mambasl_new.cmapss.constants import FD_CONFIGS
        from mambasl_new.cmapss.data import load_fd
        from mambasl_new.cmapss.preprocess import (
            add_train_rul,
            choose_val_units,
            get_feature_cols,
            preprocess_frames,
        )
        from mambasl_new.cmapss.windowing import make_last_windows, make_windows

        ds_cfg = spec["dataset"].get("config", {})
        train_cfg = spec.get("train", {})
        model_cfg = _build_cfg(spec)
        fd_name = load_result.dataset_name

        extract_dir = Path(load_result.data_dir)
        train_df, test_df, rul_test = load_fd(extract_dir, fd_name)
        train_df = add_train_rul(train_df)

        seed = train_cfg.get("seed", 42)
        val_split = train_cfg.get("val_split", 0.2)
        train_units, val_units = choose_val_units(
            train_df["unit"].unique(), seed=seed, frac=val_split,
        )
        tr_df = train_df[train_df["unit"].isin(train_units)].copy()
        va_df = train_df[train_df["unit"].isin(val_units)].copy()

        feature_mode = ds_cfg.get("feature_mode", "settings_plus_sensors")
        norm_mode = ds_cfg.get("norm_mode", "condition_minmax")
        n_conditions = FD_CONFIGS[fd_name]["n_conditions"]

        tr_df, va_df, te_df = preprocess_frames(
            tr_df, va_df, test_df.copy(),
            feature_mode=feature_mode,
            norm_mode=norm_mode,
            n_conditions=n_conditions,
            seed=seed,
        )

        feature_cols = get_feature_cols(feature_mode)
        window_size = int(model_cfg["window_size"])
        max_rul = float(model_cfg["max_rul"])

        x_train, y_train = make_windows(tr_df, feature_cols, window_size, 1, max_rul)
        x_val, y_val = make_windows(va_df, feature_cols, window_size, 1, max_rul)

        test_targets = {
            uid: float(rul_test[i])
            for i, uid in enumerate(sorted(te_df["unit"].unique().tolist()))
        }
        x_test, y_test, _ = make_last_windows(
            te_df, feature_cols, test_targets, window_size, max_rul,
        )

        out = Path(artifacts_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths = {}
        for name, arr in [
            ("x_train", x_train), ("y_train", y_train),
            ("x_val", x_val), ("y_val", y_val),
            ("x_test", x_test), ("y_test", y_test),
        ]:
            p = out / f"{name}.npy"
            np.save(str(p), arr)
            paths[name] = str(p)

        return PreprocessResult(
            artifacts_dir=str(out),
            x_train_path=paths["x_train"],
            y_train_path=paths["y_train"],
            x_val_path=paths["x_val"],
            y_val_path=paths["y_val"],
            x_test_path=paths["x_test"],
            y_test_path=paths["y_test"],
            feature_dim=int(x_train.shape[2]),
            seq_len=int(x_train.shape[1]),
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
        import numpy as np
        import torch
        from mambasl_new.cmapss.model import configure_device
        from mambasl_new.cmapss.train import train_model

        cfg = _build_cfg(spec)
        train_cfg = spec.get("train", {})

        x_train = np.load(preprocess_result.x_train_path)
        y_train = np.load(preprocess_result.y_train_path)
        x_val = np.load(preprocess_result.x_val_path)
        y_val = np.load(preprocess_result.y_val_path)

        torch.set_num_threads(spec.get("runtime", {}).get("torch_num_threads", 4))
        np.random.seed(train_cfg.get("seed", 42))
        torch.manual_seed(train_cfg.get("seed", 42))
        device = configure_device(prefer_gpu=spec.get("runtime", {}).get("use_gpu", False))

        model, best_rmse, best_score, best_epoch, best_metric = train_model(
            cfg,
            x_train, y_train,
            x_val, y_val,
            max_epochs=train_cfg.get("max_epochs", 50),
            patience=train_cfg.get("patience", 8),
            selection_metric=train_cfg.get("selection_metric", "rmse"),
            score_weight=train_cfg.get("score_weight", 0.01),
            device=device,
        )

        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)
        model_path = str(out / "model.pt")
        torch.save(
            {k: v.detach().cpu() for k, v in model.state_dict().items()},
            model_path,
        )

        config_path = out / "model_config.json"
        config_path.write_text(json.dumps({
            "cfg": cfg,
            "feature_dim": preprocess_result.feature_dim,
            "seq_len": preprocess_result.seq_len,
        }, indent=2))

        return TrainResult(
            model_path=model_path,
            best_epoch=best_epoch,
            train_loss=float(best_metric),
            val_loss=float(best_rmse),
            metadata={
                "val_rmse": float(best_rmse),
                "val_score": float(best_score),
                "n_params": int(model.count_parameters()),
            },
        )

    # -- Stage 4: evaluate -------------------------------------------------

    def evaluate(
        self,
        spec: Dict[str, Any],
        train_result: TrainResult,
        preprocess_result: PreprocessResult,
    ) -> EvalResult:
        import numpy as np
        import torch
        from mambasl_new.cmapss.model import build_model, configure_device
        from mambasl_new.cmapss.train import mae, nasa_score, predict_array, rmse

        cfg = _build_cfg(spec)
        device = configure_device(prefer_gpu=spec.get("runtime", {}).get("use_gpu", False))

        model = build_model(
            cfg,
            c_in=preprocess_result.feature_dim,
            seq_len=preprocess_result.seq_len,
            device=device,
        )
        state_dict = torch.load(train_result.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        x_test = np.load(preprocess_result.x_test_path)
        y_test = np.load(preprocess_result.y_test_path)

        preds = predict_array(model, x_test, float(cfg["max_rul"]), device=device)
        metrics = {
            "rmse": rmse(preds, y_test),
            "score": nasa_score(preds, y_test),
            "mae": mae(preds, y_test),
            "n_test": int(len(y_test)),
        }

        return EvalResult(
            metrics=metrics,
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

        src_pt = Path(train_result.model_path)
        dst_pt = final / "model.pt"
        if src_pt.resolve() != dst_pt.resolve():
            shutil.copy2(str(src_pt), str(dst_pt))

        src_cfg = src_pt.parent / "model_config.json"
        dst_cfg = final / "model_config.json"
        if src_cfg.exists() and src_cfg.resolve() != dst_cfg.resolve():
            shutil.copy2(str(src_cfg), str(dst_cfg))

        registry_path = str(
            Path(spec["storage"]["model_mount_path"]) / ".model_registry.json"
        )
        try:
            registry = FileModelRegistry(registry_path=registry_path)
            registry.register_model(
                name=model_name,
                version=model_version,
                uri=str(dst_pt),
                framework="pytorch",
                description=f"MambaSL C-MAPSS {spec['dataset'].get('config', {}).get('fd_name', '')}",
                parameters={
                    "metrics": eval_result.metrics,
                    "config": _build_cfg(spec),
                },
            )
        except Exception:
            pass  # registry write may fail outside cluster; non-fatal

        return SaveResult(
            saved_path=str(dst_pt),
            model_name=model_name,
            model_version=model_version,
        )

    # -- Inference ----------------------------------------------------------

    def load_serving_artifact(
        self,
        model_path: str,
        model_config: Dict[str, Any],
    ) -> Any:
        import torch
        from mambasl_new.cmapss.model import build_model, configure_device

        cfg = model_config.get("cfg", model_config)
        device = configure_device(prefer_gpu=False)
        model = build_model(
            cfg,
            c_in=int(model_config.get("feature_dim", 1)),
            seq_len=int(model_config.get("seq_len", 1)),
            device=device,
        )
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return {"model": model, "device": device}

    def warmup_serving_artifact(
        self,
        artifact: Any,
        model_config: Dict[str, Any],
    ) -> None:
        import numpy as np
        import torch

        seq_len = int(model_config.get("seq_len", 1))
        feature_dim = int(model_config.get("feature_dim", 1))
        sample = torch.from_numpy(
            np.zeros((1, seq_len, feature_dim), dtype=np.float32)
        ).to(artifact["device"])
        with torch.no_grad():
            artifact["model"](sample)

    def predict_loaded(
        self,
        artifact: Any,
        input_data: Any,
        model_config: Dict[str, Any],
    ) -> Any:
        from mambasl_new.cmapss.train import predict_array

        cfg = model_config.get("cfg", model_config)
        return predict_array(
            artifact["model"],
            input_data,
            float(cfg.get("max_rul", 125.0)),
            device=artifact["device"],
        )

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
                SearchParamSpec(name="d_model", type="categorical", values=[32, 64, 128]),
                SearchParamSpec(name="d_state", type="categorical", values=[8, 16, 32]),
                SearchParamSpec(name="d_conv", type="categorical", values=[2, 3, 4]),
                SearchParamSpec(name="expand", type="categorical", values=[1, 2]),
                SearchParamSpec(name="num_kernels", type="categorical", values=[0, 3, 5, 7]),
                SearchParamSpec(name="tv_dt", type="categorical", values=[False, True]),
                SearchParamSpec(name="tv_B", type="categorical", values=[False, True]),
                SearchParamSpec(name="tv_C", type="categorical", values=[False, True]),
                SearchParamSpec(name="use_D", type="categorical", values=[False, True]),
                SearchParamSpec(name="projection", type="categorical", values=["last", "avg"]),
                SearchParamSpec(name="dropout", type="categorical", values=[0.0, 0.1, 0.2, 0.3, 0.4]),
                SearchParamSpec(name="batch_size", type="categorical", values=[64, 128, 256]),
                SearchParamSpec(name="lr", type="log_float", low=3e-4, high=3e-3),
                SearchParamSpec(name="weight_decay", type="log_float", low=1e-6, high=1e-3),
                SearchParamSpec(name="huber_delta", type="categorical", values=[1.0, 2.0, 5.0]),
                SearchParamSpec(name="window_size", type="categorical", values=[30, 40, 50, 60, 70]),
                SearchParamSpec(name="max_rul", type="categorical", values=[115.0, 120.0, 125.0, 130.0, 150.0]),
            ]
        # "default" profile
        return [
            SearchParamSpec(name="d_model", type="categorical", values=[32, 64, 128]),
            SearchParamSpec(name="d_state", type="categorical", values=[8, 16, 32]),
            SearchParamSpec(name="d_conv", type="categorical", values=[3, 4]),
            SearchParamSpec(name="expand", type="categorical", values=[1, 2]),
            SearchParamSpec(name="num_kernels", type="categorical", values=[0, 3, 5, 7]),
            SearchParamSpec(name="tv_dt", type="categorical", values=[False, True]),
            SearchParamSpec(name="tv_B", type="categorical", values=[False, True]),
            SearchParamSpec(name="tv_C", type="categorical", values=[False, True]),
            SearchParamSpec(name="use_D", type="categorical", values=[False, True]),
            SearchParamSpec(name="projection", type="categorical", values=["last", "avg"]),
            SearchParamSpec(name="dropout", type="categorical", values=[0.0, 0.1, 0.2, 0.3]),
            SearchParamSpec(name="batch_size", type="categorical", values=[64, 128, 256]),
            SearchParamSpec(name="lr", type="log_float", low=3e-4, high=3e-3),
            SearchParamSpec(name="weight_decay", type="log_float", low=1e-6, high=1e-3),
            SearchParamSpec(name="huber_delta", type="categorical", values=[1.0, 2.0, 5.0]),
            SearchParamSpec(name="window_size", type="categorical", values=[30, 40, 50]),
            SearchParamSpec(name="max_rul", type="categorical", values=[125.0, 130.0, 150.0]),
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
        import torch
        from mambasl_new.cmapss.model import configure_device
        from mambasl_new.cmapss.train import train_model

        from kfp_workflow.tune.exceptions import TrialPruned

        # --- Data loading & preprocessing (cached across trials) ----------
        cache_key = (
            spec["dataset"].get("config", {}).get("fd_name", "FD001"),
            int(params.get("window_size", 50)),
            float(params.get("max_rul", 125.0)),
        )
        if not hasattr(self, "_hpo_cache") or self._hpo_cache_key != cache_key:
            # Build a spec-dict that uses *this trial's* window_size/max_rul
            trial_spec = {
                **spec,
                "model": {
                    **spec.get("model", {}),
                    "config": {
                        **spec.get("model", {}).get("config", {}),
                        "window_size": int(params.get("window_size", 50)),
                        "max_rul": float(params.get("max_rul", 125.0)),
                    },
                },
            }
            load_result = self.load_data(trial_spec, data_mount_path)
            import tempfile
            artifacts_dir = tempfile.mkdtemp(prefix="hpo_trial_")
            preprocess_result = self.preprocess(
                trial_spec, load_result, artifacts_dir,
            )
            self._hpo_cache = {
                "x_train": np.load(preprocess_result.x_train_path),
                "y_train": np.load(preprocess_result.y_train_path),
                "x_val": np.load(preprocess_result.x_val_path),
                "y_val": np.load(preprocess_result.y_val_path),
                "feature_dim": preprocess_result.feature_dim,
            }
            self._hpo_cache_key = cache_key

        x_train = self._hpo_cache["x_train"]
        y_train = self._hpo_cache["y_train"]
        x_val = self._hpo_cache["x_val"]
        y_val = self._hpo_cache["y_val"]

        if len(x_train) < int(params.get("batch_size", 64)) or len(x_val) == 0:
            raise TrialPruned()

        # --- Device setup -------------------------------------------------
        train_cfg = spec.get("train", {})
        torch.set_num_threads(
            spec.get("runtime", {}).get("torch_num_threads", 4)
        )
        device = configure_device(
            prefer_gpu=spec.get("runtime", {}).get("use_gpu", False)
        )

        # --- Single-trial training ----------------------------------------
        # Inject feature_dim into params so build_model can read c_in
        trial_params = {**params, "c_in": self._hpo_cache["feature_dim"]}

        _, _, _, _, best_metric = train_model(
            trial_params,
            x_train, y_train,
            x_val, y_val,
            max_epochs=train_cfg.get("max_epochs", 25),
            patience=train_cfg.get("patience", 5),
            selection_metric=train_cfg.get("selection_metric", "rmse"),
            score_weight=train_cfg.get("score_weight", 0.01),
            device=device,
        )
        return float(best_metric)
