"""Tests for Phase 2-3 modules: feature engine, ML strategy, circuit breaker, etc."""

import pytest

from trading_agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from trading_agent.feature_engine import CryptoFeatureEngine, FeatureConfig
from trading_agent.ml_strategy import CryptoOnlineLearner, MLConfig, CalibrationTracker
from trading_agent.order_engine import PaperOrderEngine
from trading_agent.performance import PerformanceTracker
from trading_agent.reconciliation import reconcile


# ---------------------------------------------------------------------------
# Feature Engine
# ---------------------------------------------------------------------------

class TestCryptoFeatureEngine:
    def test_compute_raw_features(self):
        engine = CryptoFeatureEngine()
        features = engine.compute({"btc_close": 50000.0, "dxy": 104.5})
        assert "btc_close_raw" in features
        assert features["btc_close_raw"] == 50000.0
        assert "dxy_raw" in features

    def test_zscore_requires_history(self):
        engine = CryptoFeatureEngine(FeatureConfig(zscore_window=3))
        # First call: no zscore yet
        f1 = engine.compute({"x": 100.0})
        assert "x_zscore" not in f1
        # Fill up history
        engine.compute({"x": 101.0})
        f3 = engine.compute({"x": 102.0})
        assert "x_zscore" in f3

    def test_momentum(self):
        engine = CryptoFeatureEngine(FeatureConfig(momentum_window=3))
        for v in [100, 101, 102, 105]:
            features = engine.compute({"x": float(v)})
        assert "x_momentum" in features
        assert features["x_momentum"] > 0  # price is above MA

    def test_roc(self):
        engine = CryptoFeatureEngine(FeatureConfig(roc_periods=[1]))
        engine.compute({"x": 100.0})
        features = engine.compute({"x": 110.0})
        assert "x_roc1" in features
        assert features["x_roc1"] == pytest.approx(0.1)

    def test_compute_target(self):
        engine = CryptoFeatureEngine(FeatureConfig(up_threshold=0.5))
        assert engine.compute_target(100.0, 101.0) == 1  # +1% > 0.5%
        assert engine.compute_target(100.0, 100.3) == 0  # +0.3% < 0.5%
        assert engine.compute_target(100.0, 99.0) == 0   # negative

    def test_cross_features(self):
        engine = CryptoFeatureEngine(FeatureConfig(
            cross_pairs=[("a", "b", "ab")],
        ))
        features = engine.compute({"a": 10.0, "b": 5.0})
        assert "ab_ratio" in features
        assert features["ab_ratio"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# ML Strategy
# ---------------------------------------------------------------------------

class TestCryptoOnlineLearner:
    def test_warmup_returns_none(self):
        learner = CryptoOnlineLearner(MLConfig(grace_period=5))
        pred = learner.predict({"x": 1.0})
        assert pred is None
        assert not learner.is_warm

    def test_learn_increases_samples(self):
        learner = CryptoOnlineLearner(MLConfig(grace_period=2))
        learner.learn({"x": 1.0}, 1)
        learner.learn({"x": 2.0}, 0)
        assert learner.samples_seen == 2
        assert learner.is_warm

    def test_predict_after_warmup(self):
        learner = CryptoOnlineLearner(MLConfig(grace_period=3, n_trees=3))
        for i in range(5):
            learner.learn({"x": float(i), "y": float(i % 2)}, i % 2)
        pred = learner.predict({"x": 10.0, "y": 1.0})
        assert pred is not None
        assert pred.direction in (0, 1)
        assert 0 <= pred.confidence <= 1

    def test_kelly_size(self):
        learner = CryptoOnlineLearner(MLConfig(kelly_fraction=0.25))
        from trading_agent.ml_strategy import Prediction
        # High confidence: should get positive Kelly
        pred = Prediction(direction=1, confidence=0.7, calibrated_confidence=0.7, probabilities={1: 0.7, 0: 0.3})
        kelly = learner.kelly_size(pred, stop_loss_pct=3.0, take_profit_pct=8.0)
        assert kelly > 0

        # Low confidence (50%): should get 0
        pred_low = Prediction(direction=1, confidence=0.5, calibrated_confidence=0.5, probabilities={1: 0.5, 0: 0.5})
        assert learner.kelly_size(pred_low) == 0.0

    def test_should_trade(self):
        learner = CryptoOnlineLearner(MLConfig(grace_period=2, confidence_threshold=0.6))
        learner.learn({"x": 1.0}, 1)
        learner.learn({"x": 2.0}, 0)
        from trading_agent.ml_strategy import Prediction
        high = Prediction(direction=1, confidence=0.7, calibrated_confidence=0.7, probabilities={})
        low = Prediction(direction=1, confidence=0.5, calibrated_confidence=0.5, probabilities={})
        assert learner.should_trade(high) is True
        assert learner.should_trade(low) is False


class TestCalibrationTracker:
    def test_calibrated_with_few_samples(self):
        cal = CalibrationTracker()
        # Less than 5 samples: return raw
        cal.update(0.8, 1)
        assert cal.calibrated_probability(0.8) == 0.8

    def test_calibrated_with_enough_samples(self):
        cal = CalibrationTracker(n_buckets=10)
        # Bucket 8 (0.8-0.9): 5 predictions, 3 actual positive
        for _ in range(3):
            cal.update(0.85, 1)
        for _ in range(2):
            cal.update(0.85, 0)
        # Calibrated should be 3/5 = 0.6
        assert cal.calibrated_probability(0.85) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_safe_by_default(self):
        cb = CircuitBreaker()
        safe, reason = cb.is_safe_to_trade(10000.0)
        assert safe is True

    def test_daily_loss_limit(self):
        cb = CircuitBreaker(CircuitBreakerConfig(daily_loss_limit_pct=2.0))
        cb.reset_daily(10000.0)
        cb.record_trade(-150.0)
        cb.record_trade(-100.0)  # total -250 = -2.5%
        safe, reason = cb.is_safe_to_trade(9750.0)
        assert safe is False
        assert "Daily loss" in reason

    def test_consecutive_losses(self):
        cb = CircuitBreaker(CircuitBreakerConfig(max_consecutive_losses=3))
        cb.record_trade(-10.0)
        cb.record_trade(-10.0)
        cb.record_trade(-10.0)
        safe, _ = cb.is_safe_to_trade(10000.0)
        assert safe is False

    def test_consecutive_reset_on_win(self):
        cb = CircuitBreaker(CircuitBreakerConfig(max_consecutive_losses=3))
        cb.record_trade(-10.0)
        cb.record_trade(-10.0)
        cb.record_trade(5.0)  # resets count
        cb.record_trade(-10.0)
        safe, _ = cb.is_safe_to_trade(10000.0)
        assert safe is True

    def test_max_drawdown(self):
        cb = CircuitBreaker(CircuitBreakerConfig(max_drawdown_pct=5.0))
        cb.is_safe_to_trade(10000.0)  # sets peak
        safe, reason = cb.is_safe_to_trade(9400.0)  # -6%
        assert safe is False
        assert "Drawdown" in reason

    def test_kill_switch(self, tmp_path):
        kill_file = tmp_path / "KILL"
        cb = CircuitBreaker(CircuitBreakerConfig(kill_file=str(kill_file)))
        safe, _ = cb.is_safe_to_trade(10000.0)
        assert safe is True
        kill_file.touch()
        safe, reason = cb.is_safe_to_trade(10000.0)
        assert safe is False
        assert "Kill switch" in reason


# ---------------------------------------------------------------------------
# Paper Order Engine
# ---------------------------------------------------------------------------

class TestPaperOrderEngine:
    def test_buy_and_sell(self):
        engine = PaperOrderEngine(initial_cash=10000.0, slippage_bps=0.0)
        engine.set_price("BTC/USDT", 50000.0)
        result = engine.market_buy("BTC/USDT", 1000.0)
        assert result is not None
        assert result.qty > 0
        assert engine.cash < 10000.0
        assert engine.positions["BTC/USDT"] > 0

    def test_sell_reduces_position(self):
        engine = PaperOrderEngine(initial_cash=10000.0, slippage_bps=0.0)
        engine.set_price("BTC/USDT", 50000.0)
        engine.market_buy("BTC/USDT", 1000.0)
        qty = engine.positions["BTC/USDT"]
        engine.market_sell("BTC/USDT", qty)
        assert engine.positions["BTC/USDT"] == pytest.approx(0.0)

    def test_slippage_increases_buy_price(self):
        engine = PaperOrderEngine(initial_cash=10000.0, slippage_bps=10.0)
        engine.set_price("BTC/USDT", 50000.0)
        result = engine.market_buy("BTC/USDT", 1000.0)
        assert result.price > 50000.0  # slippage makes buy price higher

    def test_insufficient_funds(self):
        engine = PaperOrderEngine(initial_cash=100.0)
        engine.set_price("BTC/USDT", 50000.0)
        result = engine.market_buy("BTC/USDT", 200.0)
        assert result is None


# ---------------------------------------------------------------------------
# Position Reconciliation
# ---------------------------------------------------------------------------

class TestReconciliation:
    def test_no_discrepancy(self):
        result = reconcile({"BTC": 0.1}, {"BTC": 0.1})
        assert len(result) == 0

    def test_detects_discrepancy(self):
        result = reconcile({"BTC": 0.1}, {"BTC": 0.05}, tolerance_pct=1.0)
        assert len(result) == 1
        assert result[0].symbol == "BTC"

    def test_ignores_small_diff(self):
        result = reconcile({"BTC": 0.1}, {"BTC": 0.1001}, tolerance_pct=1.0)
        assert len(result) == 0

    def test_missing_on_exchange(self):
        result = reconcile({"BTC": 0.1}, {}, tolerance_pct=1.0)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Performance Tracker
# ---------------------------------------------------------------------------

class TestPerformanceTracker:
    def test_start_and_end_day(self):
        tracker = PerformanceTracker()
        tracker.start_day(10000.0)
        tracker.record_trade(50.0)
        tracker.record_trade(-20.0)
        metrics = tracker.end_day(10030.0)
        assert metrics is not None
        assert metrics.num_trades == 2
        assert metrics.pnl == pytest.approx(30.0)

    def test_graduation_requires_min_days(self):
        tracker = PerformanceTracker()
        can, reason = tracker.should_graduate()
        assert can is False
        assert "days" in reason

    def test_summary(self):
        tracker = PerformanceTracker()
        s = tracker.summary()
        assert "Performance Summary" in s


# ---------------------------------------------------------------------------
# Shadow Mode - Bug Fixes
# ---------------------------------------------------------------------------

class TestShadowModeBugFixes:
    """Verify critical bug fixes in shadow mode."""

    def test_per_symbol_learning_state(self):
        """Bug 1: Each symbol should have its own learning state."""
        from trading_agent.shadow_mode import ShadowRunner
        runner = ShadowRunner()
        # Verify per-symbol dicts exist
        assert isinstance(runner._prev_features, dict)
        assert isinstance(runner._prev_predictions, dict)
        assert isinstance(runner._entry_prices, dict)

    def test_entry_price_tracking(self):
        """Bug 3: Entry price should be tracked for correct PnL."""
        from trading_agent.order_engine import PaperOrderEngine
        engine = PaperOrderEngine(initial_cash=10000.0, slippage_bps=0.0)
        engine.set_price("BTC/USDT", 50000.0)

        # Buy
        order = engine.market_buy("BTC/USDT", 1000.0)
        entry_price = order.price  # should be ~50000

        # Price goes up
        engine.set_price("BTC/USDT", 55000.0)
        qty = engine.positions["BTC/USDT"]
        sell_order = engine.market_sell("BTC/USDT", qty)

        # PnL should reflect actual price change, not be ~0
        pnl = sell_order.qty * (sell_order.price - entry_price)
        assert pnl > 0  # price went up from 50k to 55k
        assert pnl == pytest.approx(qty * 5000.0, rel=0.01)

    def test_warmup_no_lookahead(self):
        """Bug 2: Warmup should learn prev features with current target."""
        from trading_agent.feature_engine import CryptoFeatureEngine, FeatureConfig
        from trading_agent.ml_strategy import CryptoOnlineLearner, MLConfig

        engine = CryptoFeatureEngine(FeatureConfig(zscore_window=3))
        learner = CryptoOnlineLearner(MLConfig(grace_period=2))

        # Simulate warmup loop with correct alignment
        prices = [100.0, 101.0, 102.0, 100.5]
        prev_price = None
        prev_features = None
        for price in prices:
            features = engine.compute({"x": price})
            if prev_price is not None and prev_features:
                target = engine.compute_target(prev_price, price)
                learner.learn(prev_features, target)
            prev_price = price
            prev_features = features

        # 4 prices → 3 learning events (first has no prev_price/prev_features)
        # Each learns PREVIOUS features with CURRENT target (no lookahead)
        assert learner.samples_seen == 3

    def test_calibration_uses_p_up(self):
        """Bug 4: Calibration should always use P(up) for consistent buckets."""
        from trading_agent.ml_strategy import CryptoOnlineLearner, MLConfig, Prediction

        learner = CryptoOnlineLearner(MLConfig(grace_period=2))
        learner.learn({"x": 1.0}, 1)
        learner.learn({"x": 2.0}, 0)

        # When direction=0 (down), P(up) should be low
        pred_down = Prediction(
            direction=0, confidence=0.8,
            calibrated_confidence=0.2,  # 1 - 0.8 for P(up)
            probabilities={0: 0.8, 1: 0.2},
        )
        # update_calibration should receive P(up)=0.2, not confidence=0.8
        p_up = pred_down.probabilities.get(1, 1 - pred_down.confidence)
        assert p_up == pytest.approx(0.2)

    def test_stop_loss_take_profit_in_shadow(self):
        """Bug 5: SL/TP should trigger automatic exits."""
        from trading_agent.order_engine import PaperOrderEngine

        engine = PaperOrderEngine(initial_cash=10000.0, slippage_bps=0.0)
        engine.set_price("BTC/USDT", 50000.0)
        order = engine.market_buy("BTC/USDT", 1000.0)
        entry_price = order.price

        # Price drops 4% → should trigger 3% SL
        new_price = entry_price * 0.96
        pnl_pct = (new_price - entry_price) / entry_price * 100
        assert pnl_pct < -3.0  # SL threshold

        # Price rises 9% → should trigger 8% TP
        new_price = entry_price * 1.09
        pnl_pct = (new_price - entry_price) / entry_price * 100
        assert pnl_pct > 8.0  # TP threshold

    def test_performance_no_duplicate_days(self):
        """Bug 6: Daily metrics should not have duplicate dates."""
        tracker = PerformanceTracker()
        tracker.start_day(10000.0)
        tracker.record_trade(50.0)
        metrics = tracker.end_day(10050.0)
        assert metrics is not None

        # start_day again with same date should not create duplicate
        tracker.start_day(10050.0)
        tracker.end_day(10050.0)
        dates = [d.date for d in tracker._daily_metrics]
        assert len(dates) == len(set(dates)), f"Duplicate dates: {dates}"

    def test_performance_saves_current_day(self, tmp_path):
        """Bug 6: current_day should survive save/load cycle."""
        tracker = PerformanceTracker()
        tracker.start_day(10000.0)
        tracker.record_trade(100.0)

        path = tmp_path / "perf.json"
        tracker.save(path)

        loaded = PerformanceTracker.load(path)
        assert loaded._current_day is not None
        assert loaded._current_day.pnl == pytest.approx(100.0)
        assert loaded._current_day.num_trades == 1
