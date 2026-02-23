"""Tests for unified order engine."""

import pytest

from trading_agent.order_engine import PaperOrderEngine


class TestPaperOrderEngine:
    def test_buy_reduces_cash(self):
        engine = PaperOrderEngine(initial_cash=10_000.0)
        engine.set_price("BTC/USDT", 50_000.0)
        result = engine.market_buy("BTC/USDT", 1_000.0)
        assert result is not None
        assert result.side == "buy"
        assert result.qty > 0
        assert engine.cash < 10_000.0

    def test_sell_increases_cash(self):
        engine = PaperOrderEngine(initial_cash=10_000.0)
        engine.set_price("BTC/USDT", 50_000.0)
        engine.market_buy("BTC/USDT", 1_000.0)
        qty = engine.positions["BTC/USDT"]
        result = engine.market_sell("BTC/USDT", qty)
        assert result is not None
        assert result.side == "sell"
        # Cash should be close to original minus fees and slippage
        assert engine.cash > 9_900.0

    def test_buy_insufficient_funds(self):
        engine = PaperOrderEngine(initial_cash=100.0)
        engine.set_price("BTC/USDT", 50_000.0)
        result = engine.market_buy("BTC/USDT", 200.0)
        assert result is None

    def test_sell_insufficient_position(self):
        engine = PaperOrderEngine(initial_cash=10_000.0)
        engine.set_price("BTC/USDT", 50_000.0)
        result = engine.market_sell("BTC/USDT", 1.0)
        assert result is None

    def test_slippage_applied(self):
        engine = PaperOrderEngine(initial_cash=10_000.0, slippage_bps=10.0)
        engine.set_price("BTC/USDT", 50_000.0)
        result = engine.market_buy("BTC/USDT", 1_000.0)
        assert result is not None
        # Buy price should be higher than market due to slippage
        assert result.price > 50_000.0
        assert result.slippage_bps == 10.0

    def test_zero_price_rejected(self):
        engine = PaperOrderEngine(initial_cash=10_000.0)
        # No price set
        result = engine.market_buy("BTC/USDT", 1_000.0)
        assert result is None
