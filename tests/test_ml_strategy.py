"""Tests for online ML strategy and feature engine."""

import pytest

from trading_agent.feature_engine import CryptoFeatureEngine, FeatureConfig
from trading_agent.ml_strategy import (
    CalibrationTracker,
    CryptoOnlineLearner,
    MLConfig,
    Prediction,
)


class TestFeatureEngine:
    def test_raw_features(self):
        engine = CryptoFeatureEngine()
        features = engine.compute({"btc_close": 50000.0, "dxy": 104.5})
        assert "btc_close_raw" in features
        assert features["btc_close_raw"] == 50000.0
        assert "dxy_raw" in features

    def test_zscore_needs_history(self):
        engine = CryptoFeatureEngine(FeatureConfig(zscore_window=5))
        # Not enough history
        features = engine.compute({"btc_close": 50000.0})
        assert "btc_close_zscore" not in features
        # Build history
        for i in range(10):
            features = engine.compute({"btc_close": 50000.0 + i * 100})
        assert "btc_close_zscore" in features

    def test_momentum(self):
        engine = CryptoFeatureEngine(FeatureConfig(momentum_window=3))
        for price in [100, 105, 110, 115, 120]:
            features = engine.compute({"price": price})
        assert "price_momentum" in features
        assert features["price_momentum"] > 0  # uptrend

    def test_rate_of_change(self):
        engine = CryptoFeatureEngine(FeatureConfig(roc_periods=[1]))
        engine.compute({"x": 100})
        features = engine.compute({"x": 110})
        assert "x_roc1" in features
        assert features["x_roc1"] == pytest.approx(0.1, rel=1e-3)

    def test_compute_target(self):
        engine = CryptoFeatureEngine(FeatureConfig(up_threshold=0.3))
        assert engine.compute_target(100, 100.5) == 1  # +0.5% > 0.3%
        assert engine.compute_target(100, 100.1) == 0  # +0.1% < 0.3%
        assert engine.compute_target(100, 99) == 0     # negative

    def test_history_length(self):
        engine = CryptoFeatureEngine()
        assert engine.history_length == 0
        engine.compute({"a": 1, "b": 2})
        engine.compute({"a": 3, "b": 4})
        assert engine.history_length == 2

    def test_reset(self):
        engine = CryptoFeatureEngine()
        engine.compute({"a": 1})
        engine.reset()
        assert engine.history_length == 0


class TestCalibrationTracker:
    def test_uncalibrated_returns_raw(self):
        ct = CalibrationTracker()
        assert ct.calibrated_probability(0.7) == 0.7

    def test_calibrates_with_data(self):
        ct = CalibrationTracker(n_buckets=10)
        # Fill bucket 7 (0.7-0.8) with outcomes
        for _ in range(10):
            ct.update(0.75, 1)
        for _ in range(10):
            ct.update(0.75, 0)
        # Calibrated probability for 0.75 should be ~0.5
        assert ct.calibrated_probability(0.75) == pytest.approx(0.5, abs=0.05)


class TestCryptoOnlineLearner:
    def test_predict_before_warmup(self):
        learner = CryptoOnlineLearner(MLConfig(grace_period=5))
        pred = learner.predict({"f1": 1.0, "f2": 2.0})
        assert pred is None
        assert not learner.is_warm

    def test_warmup_and_predict(self):
        learner = CryptoOnlineLearner(MLConfig(grace_period=5, model_type="logistic"))
        for i in range(10):
            features = {"f1": float(i), "f2": float(i * 2)}
            target = 1 if i % 2 == 0 else 0
            learner.learn(features, target)
        assert learner.is_warm
        pred = learner.predict({"f1": 5.0, "f2": 10.0})
        assert pred is not None
        assert pred.direction in (0, 1)
        assert 0 <= pred.confidence <= 1

    def test_kelly_size(self):
        learner = CryptoOnlineLearner(MLConfig(kelly_fraction=0.25))
        pred = Prediction(direction=1, confidence=0.6, calibrated_confidence=0.6, probabilities={1: 0.6, 0: 0.4})
        kelly = learner.kelly_size(pred, stop_loss_pct=3.0, take_profit_pct=8.0)
        assert kelly > 0

    def test_kelly_zero_for_low_confidence(self):
        learner = CryptoOnlineLearner()
        pred = Prediction(direction=1, confidence=0.45, calibrated_confidence=0.45, probabilities={1: 0.45, 0: 0.55})
        kelly = learner.kelly_size(pred)
        assert kelly == 0.0

    def test_should_trade(self):
        learner = CryptoOnlineLearner(MLConfig(grace_period=3, confidence_threshold=0.6))
        for i in range(5):
            learner.learn({"f1": float(i)}, i % 2)
        pred_high = Prediction(direction=1, confidence=0.7, calibrated_confidence=0.7, probabilities={1: 0.7})
        pred_low = Prediction(direction=1, confidence=0.5, calibrated_confidence=0.5, probabilities={1: 0.5})
        assert learner.should_trade(pred_high) is True
        assert learner.should_trade(pred_low) is False

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "model.pkl"
        learner = CryptoOnlineLearner(MLConfig(grace_period=3, model_type="logistic"))
        for i in range(5):
            learner.learn({"f1": float(i)}, i % 2)
        learner.save(path)

        loaded = CryptoOnlineLearner.load(path)
        assert loaded.samples_seen == 5
        assert loaded.is_warm
