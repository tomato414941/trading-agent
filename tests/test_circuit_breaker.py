"""Tests for circuit breaker module."""

from pathlib import Path

import pytest

from trading_agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


class TestCircuitBreaker:
    def test_safe_by_default(self):
        cb = CircuitBreaker()
        safe, reason = cb.is_safe_to_trade(10_000.0)
        assert safe is True
        assert reason == "ok"

    def test_kill_switch_file(self, tmp_path):
        kill_file = tmp_path / "KILL"
        kill_file.touch()
        cb = CircuitBreaker(config=CircuitBreakerConfig(kill_file=str(kill_file)))
        safe, reason = cb.is_safe_to_trade(10_000.0)
        assert safe is False
        assert "Kill switch" in reason

    def test_daily_loss_limit(self):
        cb = CircuitBreaker(config=CircuitBreakerConfig(daily_loss_limit_pct=3.0))
        cb.reset_daily(10_000.0)
        cb.record_trade(-350.0)  # -3.5%
        safe, reason = cb.is_safe_to_trade(9_650.0)
        assert safe is False
        assert "Daily loss" in reason

    def test_consecutive_losses(self):
        cb = CircuitBreaker(config=CircuitBreakerConfig(max_consecutive_losses=3))
        for _ in range(3):
            cb.record_trade(-10.0)
        safe, reason = cb.is_safe_to_trade(10_000.0)
        assert safe is False
        assert "Consecutive" in reason

    def test_consecutive_losses_reset_on_win(self):
        cb = CircuitBreaker(config=CircuitBreakerConfig(max_consecutive_losses=3))
        cb.record_trade(-10.0)
        cb.record_trade(-10.0)
        cb.record_trade(20.0)  # reset
        cb.record_trade(-10.0)
        safe, _ = cb.is_safe_to_trade(10_000.0)
        assert safe is True

    def test_max_drawdown(self):
        cb = CircuitBreaker(config=CircuitBreakerConfig(max_drawdown_pct=5.0))
        cb.is_safe_to_trade(10_000.0)  # sets peak
        safe, reason = cb.is_safe_to_trade(9_400.0)  # 6% down
        assert safe is False
        assert "Drawdown" in reason

    def test_halted_stays_halted(self):
        cb = CircuitBreaker(config=CircuitBreakerConfig(max_consecutive_losses=2))
        cb.record_trade(-10.0)
        cb.record_trade(-10.0)
        cb.is_safe_to_trade(10_000.0)  # triggers halt
        assert cb.halted is True
        # Still halted even with good equity
        safe, _ = cb.is_safe_to_trade(20_000.0)
        assert safe is False
