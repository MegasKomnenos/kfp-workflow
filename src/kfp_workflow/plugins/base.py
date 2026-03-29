"""ModelPlugin ABC and inter-stage result dataclasses.

Every model that plugs into the kfp-workflow pipeline must subclass
``ModelPlugin`` and implement all abstract methods.  Heavy data (numpy
arrays, DataFrames) is persisted as files on the PVC; only paths and
scalar metadata travel between pipeline stages as JSON.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Type


# ---------------------------------------------------------------------------
# Result dataclasses — one per pipeline stage
# ---------------------------------------------------------------------------

@dataclass
class LoadDataResult:
    """Output of ``ModelPlugin.load_data``."""

    data_dir: str
    """Directory containing raw data files (on PVC or local)."""

    dataset_name: str
    """Human-readable dataset identifier, e.g. ``"FD001"``."""

    num_train_samples: int
    num_test_samples: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessResult:
    """Output of ``ModelPlugin.preprocess``."""

    artifacts_dir: str
    """Directory containing saved ``.npy`` array files."""

    x_train_path: str
    y_train_path: str
    x_val_path: str
    y_val_path: str
    x_test_path: str
    y_test_path: str
    feature_dim: int
    """Number of input features (``c_in`` for the model)."""

    seq_len: int
    """Sequence / window length."""

    num_train: int
    num_val: int
    num_test: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    """Output of ``ModelPlugin.train``."""

    model_path: str
    """Path to the saved model artifact used for evaluation and serving."""

    best_epoch: int
    train_loss: float
    val_loss: float
    """Best validation metric value (e.g. RMSE)."""

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Output of ``ModelPlugin.evaluate``."""

    metrics: Dict[str, float]
    """Evaluation metrics, e.g. ``{"rmse": …, "score": …, "mae": …}``."""

    model_path: str
    """Pass-through of the model path for downstream stages."""

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SaveResult:
    """Output of ``ModelPlugin.save_model``."""

    saved_path: str
    model_name: str
    model_version: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def result_to_dict(result: object) -> Dict[str, Any]:
    """Serialise any result dataclass to a plain dict."""
    return asdict(result)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ModelPlugin(ABC):
    """Contract that every model plugin must satisfy.

    One method per KFP pipeline stage, plus ``predict`` for inference.
    """

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Unique identifier used as the lookup key in the plugin registry."""
        ...

    # -- Stage 1 -----------------------------------------------------------

    @abstractmethod
    def load_data(
        self,
        spec: Dict[str, Any],
        data_mount_path: str,
    ) -> LoadDataResult:
        """Load / download raw data and verify it exists.

        Parameters
        ----------
        spec:
            Full ``PipelineSpec`` as a plain dict (from ``json.loads``).
        data_mount_path:
            Root path where the data PVC is mounted.
        """
        ...

    # -- Stage 2 -----------------------------------------------------------

    @abstractmethod
    def preprocess(
        self,
        spec: Dict[str, Any],
        load_result: LoadDataResult,
        artifacts_dir: str,
    ) -> PreprocessResult:
        """Transform raw data into training-ready numpy arrays.

        Saves arrays as ``.npy`` files under *artifacts_dir*.
        """
        ...

    # -- Stage 3 -----------------------------------------------------------

    @abstractmethod
    def train(
        self,
        spec: Dict[str, Any],
        preprocess_result: PreprocessResult,
        model_dir: str,
    ) -> TrainResult:
        """Build model, train on preprocessed data, save best weights."""
        ...

    # -- Stage 4 -----------------------------------------------------------

    @abstractmethod
    def evaluate(
        self,
        spec: Dict[str, Any],
        train_result: TrainResult,
        preprocess_result: PreprocessResult,
    ) -> EvalResult:
        """Load trained model, predict on test set, compute metrics."""
        ...

    # -- Stage 5 -----------------------------------------------------------

    @abstractmethod
    def save_model(
        self,
        spec: Dict[str, Any],
        train_result: TrainResult,
        eval_result: EvalResult,
        final_model_dir: str,
    ) -> SaveResult:
        """Copy model to final PVC path and register in model registry."""
        ...

    # -- HPO hooks (optional) -----------------------------------------------

    def hpo_search_space(
        self,
        spec: Dict[str, Any],
        profile: str,
    ) -> list:
        """Return the search space for a builtin profile.

        Must return a list of ``kfp_workflow.specs.SearchParamSpec``
        instances.  Only called when the spec does not contain an
        explicit ``hpo.search_space``; in that case *profile* is
        ``"default"``, ``"aggressive"``, etc.
        """
        raise NotImplementedError(
            f"Plugin '{self.name()}' does not provide HPO search spaces."
        )

    def hpo_base_config(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Return the fixed base config that search params override.

        The returned dict contains all model / training parameters at
        their default values.  During HPO the engine merges suggested
        params on top of this base before calling ``hpo_objective``.
        """
        raise NotImplementedError(
            f"Plugin '{self.name()}' does not provide an HPO base config."
        )

    def hpo_objective(
        self,
        spec: Dict[str, Any],
        params: Dict[str, Any],
        data_mount_path: str,
    ) -> float:
        """Run **one** HPO trial and return the objective metric value.

        Parameters
        ----------
        spec:
            Full ``TuneSpec`` as a plain dict.
        params:
            Candidate hyperparameter values (base_config merged with the
            engine's suggested overrides).
        data_mount_path:
            Root path where data is accessible.

        Returns
        -------
        float
            Objective metric value (**lower is better**).

        Raises
        ------
        kfp_workflow.tune.exceptions.TrialPruned
            If the trial should be pruned.
        """
        raise NotImplementedError(
            f"Plugin '{self.name()}' does not support HPO trials."
        )

    # -- Inference ----------------------------------------------------------

    # -- Config schema hooks (optional) ------------------------------------

    @classmethod
    def model_config_schema(cls) -> Optional[Type]:
        """Return a Pydantic model for ``model.config`` validation.

        Override in subclasses to enable schema-based validation of
        model architecture parameters.  Return ``None`` (default) to
        accept any ``Dict[str, Any]`` without extra validation.
        """
        return None

    @classmethod
    def dataset_config_schema(cls) -> Optional[Type]:
        """Return a Pydantic model for ``dataset.config`` validation."""
        return None

    @classmethod
    def train_config_schema(cls) -> Optional[Type]:
        """Return a Pydantic model for ``train`` section validation."""
        return None

    @classmethod
    def serving_model_filenames(cls) -> List[str]:
        """Return preferred model artifact filenames for serving lookup."""
        return ["model.pt"]

    def load_serving_artifact(
        self,
        model_path: str,
        model_config: Dict[str, Any],
    ) -> Any:
        """Load and return a serving artifact for reuse across requests."""
        return model_path

    def warmup_serving_artifact(
        self,
        artifact: Any,
        model_config: Dict[str, Any],
    ) -> None:
        """Perform optional startup warmup before the predictor becomes ready."""
        return None

    def predict_loaded(
        self,
        artifact: Any,
        input_data: Any,
        model_config: Dict[str, Any],
    ) -> Any:
        """Run inference using a pre-loaded serving artifact."""
        return self.predict(
            model_path=str(artifact),
            input_data=input_data,
            model_config=model_config,
        )

    # -- Inference ----------------------------------------------------------

    @abstractmethod
    def predict(
        self,
        model_path: str,
        input_data: Any,
        model_config: Dict[str, Any],
    ) -> Any:
        """Run inference (used by the KServe custom predictor).

        Parameters
        ----------
        model_path:
            Path to saved model state-dict.
        input_data:
            Input tensor as numpy array ``[batch, seq_len, features]``.
        model_config:
            Model-specific config needed to rebuild the architecture.

        Returns
        -------
        numpy.ndarray
            Predictions.
        """
        ...
