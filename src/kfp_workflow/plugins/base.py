"""ModelPlugin ABC and inter-stage result dataclasses.

Every model that plugs into the kfp-workflow pipeline must subclass
``ModelPlugin`` and implement all abstract methods.  Heavy data (numpy
arrays, DataFrames) is persisted as files on the PVC; only paths and
scalar metadata travel between pipeline stages as JSON.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


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
    """Path to saved model state-dict (``.pt``) file."""

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
