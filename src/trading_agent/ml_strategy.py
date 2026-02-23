"""Online ML strategy for crypto directional prediction using River."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_DIR = Path("data/models")


@dataclass
class MLConfig:
    model_type: str = "adaptive_forest"
    n_trees: int = 10
    grace_period: int = 50
    confidence_threshold: float = 0.58
    kelly_fraction: float = 0.15
    signal_noise_url: str = "http://localhost:8000"


@dataclass
class Prediction:
    direction: int  # 1=up, 0=not up
    confidence: float
    calibrated_confidence: float
    probabilities: dict[int, float]


class CalibrationTracker:
    """Track predicted vs actual probabilities in buckets."""

    def __init__(self, n_buckets: int = 10):
        self._n_buckets = n_buckets
        self._predicted_sum: list[float] = [0.0] * n_buckets
        self._actual_sum: list[int] = [0] * n_buckets
        self._count: list[int] = [0] * n_buckets

    def update(self, predicted_prob: float, actual: int) -> None:
        bucket = min(int(predicted_prob * self._n_buckets), self._n_buckets - 1)
        self._predicted_sum[bucket] += predicted_prob
        self._actual_sum[bucket] += actual
        self._count[bucket] += 1

    def calibrated_probability(self, raw_prob: float) -> float:
        bucket = min(int(raw_prob * self._n_buckets), self._n_buckets - 1)
        if self._count[bucket] < 5:
            return raw_prob
        return self._actual_sum[bucket] / self._count[bucket]

    def calibration_error(self) -> float:
        errors = []
        for i in range(self._n_buckets):
            if self._count[i] >= 5:
                avg_pred = self._predicted_sum[i] / self._count[i]
                avg_actual = self._actual_sum[i] / self._count[i]
                errors.append(abs(avg_pred - avg_actual))
        return sum(errors) / len(errors) if errors else 0.0


class CryptoOnlineLearner:
    """Online ML model using River for crypto direction prediction."""

    def __init__(self, config: MLConfig | None = None):
        self._config = config or MLConfig()
        self._model = self._create_model()
        self._samples_seen: int = 0
        self._correct: int = 0
        self._total_predictions: int = 0
        self._last_features: dict[str, float] | None = None
        self._calibration = CalibrationTracker()

    def _create_model(self):
        from river import compose, preprocessing

        if self._config.model_type == "adaptive_forest":
            from river.forest import ARFClassifier
            model = ARFClassifier(
                n_models=self._config.n_trees,
                seed=42,
            )
        elif self._config.model_type == "logistic":
            from river.linear_model import LogisticRegression
            model = LogisticRegression()
        else:
            raise ValueError(f"Unknown model type: {self._config.model_type}")

        return compose.Pipeline(
            preprocessing.AdaptiveStandardScaler(),
            model,
        )

    def predict(self, features: dict[str, float]) -> Prediction | None:
        if not features:
            return None
        if self._samples_seen < self._config.grace_period:
            self._last_features = features
            return None

        proba = self._model.predict_proba_one(features)
        if not proba:
            self._last_features = features
            return None

        direction = max(proba, key=proba.get)
        confidence = proba[direction]
        calibrated = self._calibration.calibrated_probability(confidence)
        self._last_features = features
        self._total_predictions += 1

        return Prediction(
            direction=direction,
            confidence=confidence,
            calibrated_confidence=calibrated,
            probabilities=dict(proba),
        )

    def learn(self, features: dict[str, float], target: int) -> None:
        if not features:
            return
        self._model.learn_one(features, target)
        self._samples_seen += 1

    def learn_delayed(self, target: int) -> None:
        if self._last_features is not None:
            self.learn(self._last_features, target)

    def update_accuracy(self, predicted: int, actual: int) -> None:
        if predicted == actual:
            self._correct += 1

    def update_calibration(self, predicted_prob: float, actual: int) -> None:
        self._calibration.update(predicted_prob, actual)

    def kelly_size(self, prediction: Prediction, stop_loss_pct: float = 3.0, take_profit_pct: float = 8.0) -> float:
        """Calculate Kelly fraction for position sizing."""
        p = prediction.calibrated_confidence
        if p <= 0.5:
            return 0.0
        b = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 1.0
        kelly = (p * b - (1 - p)) / b
        return max(0.0, kelly * self._config.kelly_fraction)

    @property
    def accuracy(self) -> float:
        return self._correct / self._total_predictions if self._total_predictions else 0.0

    @property
    def calibration_error(self) -> float:
        return self._calibration.calibration_error()

    @property
    def is_warm(self) -> bool:
        return self._samples_seen >= self._config.grace_period

    @property
    def samples_seen(self) -> int:
        return self._samples_seen

    def should_trade(self, prediction: Prediction) -> bool:
        return (
            prediction.calibrated_confidence >= self._config.confidence_threshold
            and self.is_warm
        )

    def save(self, path: Path | None = None) -> None:
        path = path or (MODEL_DIR / "crypto_learner.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "samples_seen": self._samples_seen,
                "correct": self._correct,
                "total_predictions": self._total_predictions,
                "config": self._config,
                "last_features": self._last_features,
                "calibration": self._calibration,
            }, f)
        log.info("Model saved: %d samples, %.1f%% accuracy", self._samples_seen, self.accuracy * 100)

    @classmethod
    def load(cls, path: Path | None = None) -> CryptoOnlineLearner:
        path = path or (MODEL_DIR / "crypto_learner.pkl")
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        learner = cls(config=data["config"])
        learner._model = data["model"]
        learner._samples_seen = data["samples_seen"]
        learner._correct = data["correct"]
        learner._total_predictions = data["total_predictions"]
        learner._last_features = data.get("last_features")
        if "calibration" in data:
            learner._calibration = data["calibration"]
        log.info("Model loaded: %d samples, %.1f%% accuracy", learner._samples_seen, learner.accuracy * 100)
        return learner
