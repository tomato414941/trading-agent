"""Tests for live arbitrage executor (with mocked exchange)."""

import time
from unittest.mock import MagicMock, patch

import pytest

from trading_agent.arbitrage_executor import (
    LiveArbitrageState,
    check_and_act,
    close_position,
    open_position,
    _futures_symbol,
    _next_settlement_time,
)
from trading_agent.config import ArbitrageConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return ArbitrageConfig(
        entry_fr_threshold=0.0003,
        exit_fr_threshold=0.0001,
        exit_consecutive_periods=3,
        spot_fee_rate=0.001,
        futures_fee_rate=0.0005,
        position_fraction=0.3,
        futures_leverage=1.0,
        symbol="BTC/USDT",
    )


@pytest.fixture
def mock_exchanges():
    """Create mocked spot and futures exchanges."""
    ex_futures = MagicMock()
    ex_spot = MagicMock()

    # Default market info
    ex_futures.market.return_value = {"precision": {"amount": 6}}
    ex_futures.amount_to_precision.side_effect = lambda sym, qty: round(qty, 6)
    ex_futures.fetch_ticker.return_value = {"last": 50000.0}
    ex_futures.fetch_funding_rate.return_value = {"fundingRate": 0.0005}

    ex_spot.fetch_balance.return_value = {"USDT": {"free": 10000.0}}

    # Order responses
    ex_spot.create_market_buy_order.return_value = {"average": 50000.0, "cost": 3000.0}
    ex_spot.create_market_sell_order.return_value = {"average": 50000.0, "cost": 3000.0}
    ex_futures.create_market_sell_order.return_value = {"average": 50000.0, "cost": 3000.0}
    ex_futures.create_market_buy_order.return_value = {"average": 50000.0, "cost": 3000.0}

    return ex_futures, ex_spot


# ---------------------------------------------------------------------------
# Symbol conversion
# ---------------------------------------------------------------------------

class TestFuturesSymbol:
    def test_adds_suffix(self):
        assert _futures_symbol("BTC/USDT") == "BTC/USDT:USDT"

    def test_no_double_suffix(self):
        assert _futures_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "state.json"
        state = LiveArbitrageState(
            is_open=True, symbol="BTC/USDT", spot_qty=0.01,
            spot_entry_price=50000.0, total_fr_collected=5.0,
        )
        state.save(path)
        loaded = LiveArbitrageState.load(path)
        assert loaded.is_open is True
        assert loaded.symbol == "BTC/USDT"
        assert loaded.spot_qty == 0.01
        assert loaded.total_fr_collected == 5.0

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        state = LiveArbitrageState.load(path)
        assert state.is_open is False

    def test_summary_open(self):
        state = LiveArbitrageState(
            is_open=True, symbol="BTC/USDT", spot_qty=0.01,
            spot_entry_price=50000.0,
        )
        summary = state.summary()
        assert "OPEN" in summary
        assert "BTC/USDT" in summary

    def test_summary_closed(self):
        state = LiveArbitrageState()
        summary = state.summary()
        assert "CLOSED" in summary


# ---------------------------------------------------------------------------
# check_and_act
# ---------------------------------------------------------------------------

class TestCheckAndAct:
    def test_no_position_high_fr_opens(self, mock_exchanges, config):
        """FR above threshold -> opens position."""
        ex_futures, ex_spot = mock_exchanges
        ex_futures.fetch_funding_rate.return_value = {"fundingRate": 0.0005}
        state = LiveArbitrageState()

        action = check_and_act(ex_futures, ex_spot, state, config)

        assert action == "opened"
        assert state.is_open is True
        ex_spot.create_market_buy_order.assert_called_once()
        ex_futures.create_market_sell_order.assert_called_once()

    def test_no_position_low_fr_skips(self, mock_exchanges, config):
        """FR below threshold -> no trade."""
        ex_futures, ex_spot = mock_exchanges
        ex_futures.fetch_funding_rate.return_value = {"fundingRate": 0.0001}
        state = LiveArbitrageState()

        action = check_and_act(ex_futures, ex_spot, state, config)

        assert "waiting" in action
        assert state.is_open is False
        ex_spot.create_market_buy_order.assert_not_called()

    def test_position_no_fr_between_settlements(self, mock_exchanges, config):
        """FR should NOT accumulate between settlement times."""
        ex_futures, ex_spot = mock_exchanges
        ex_futures.fetch_funding_rate.return_value = {"fundingRate": 0.0005}
        # next_settlement is far in the future -> no FR accrual
        future_ms = int(time.time() * 1000) + 3600_000
        state = LiveArbitrageState(
            is_open=True, symbol="BTC/USDT",
            spot_qty=0.01, spot_entry_price=50000.0,
            futures_qty=0.01,
            next_settlement_ms=future_ms,
        )

        action = check_and_act(ex_futures, ex_spot, state, config)

        assert "holding" in action
        assert state.accumulated_fr == 0.0
        assert state.total_fr_collected == 0.0

    def test_position_collects_fr_at_settlement(self, mock_exchanges, config):
        """FR accrues only when settlement time has passed."""
        ex_futures, ex_spot = mock_exchanges
        ex_futures.fetch_funding_rate.return_value = {"fundingRate": 0.0005}
        # next_settlement is in the past -> should accrue FR
        past_ms = int(time.time() * 1000) - 1000
        state = LiveArbitrageState(
            is_open=True, symbol="BTC/USDT",
            spot_qty=0.01, spot_entry_price=50000.0,
            futures_qty=0.01,
            next_settlement_ms=past_ms,
        )

        action = check_and_act(ex_futures, ex_spot, state, config)

        assert "holding" in action
        # FR payment = 0.01 * 50000 * 0.0005 = 0.25
        assert state.accumulated_fr == pytest.approx(0.25)
        assert state.total_fr_collected == pytest.approx(0.25)
        assert state.next_settlement_ms > past_ms

    def test_exit_on_consecutive_low_fr(self, mock_exchanges, config):
        """N consecutive low FR at settlement -> closes position."""
        ex_futures, ex_spot = mock_exchanges
        ex_futures.fetch_funding_rate.return_value = {"fundingRate": 0.00005}
        # Settlement in the past so the settlement block fires
        past_ms = int(time.time() * 1000) - 1000
        state = LiveArbitrageState(
            is_open=True, symbol="BTC/USDT",
            spot_qty=0.01, spot_entry_price=50000.0,
            futures_qty=0.01,
            low_fr_count=2,  # already 2, this settlement makes 3
            next_settlement_ms=past_ms,
        )

        action = check_and_act(ex_futures, ex_spot, state, config)

        assert action == "closed"
        assert state.is_open is False
        assert state.num_round_trips == 1

    def test_low_fr_resets_on_high_fr(self, mock_exchanges, config):
        """Low FR count resets at settlement when FR goes above threshold."""
        ex_futures, ex_spot = mock_exchanges
        ex_futures.fetch_funding_rate.return_value = {"fundingRate": 0.0005}
        # Settlement in the past so the settlement block fires
        past_ms = int(time.time() * 1000) - 1000
        state = LiveArbitrageState(
            is_open=True, symbol="BTC/USDT",
            spot_qty=0.01, spot_entry_price=50000.0,
            futures_qty=0.01,
            low_fr_count=2,
            next_settlement_ms=past_ms,
        )

        check_and_act(ex_futures, ex_spot, state, config)

        assert state.low_fr_count == 0

    def test_low_balance_skips(self, mock_exchanges, config):
        """Insufficient balance -> skip."""
        ex_futures, ex_spot = mock_exchanges
        ex_futures.fetch_funding_rate.return_value = {"fundingRate": 0.0005}
        ex_spot.fetch_balance.return_value = {"USDT": {"free": 5.0}}
        state = LiveArbitrageState()

        action = check_and_act(ex_futures, ex_spot, state, config)

        assert "low balance" in action
        assert state.is_open is False


# ---------------------------------------------------------------------------
# open_position / close_position
# ---------------------------------------------------------------------------

class TestOpenClosePosition:
    def test_open_creates_both_orders(self, mock_exchanges, config):
        ex_futures, ex_spot = mock_exchanges
        state = LiveArbitrageState()

        open_position(ex_futures, ex_spot, state, "BTC/USDT", 3000.0, config)

        assert state.is_open is True
        assert state.spot_qty > 0
        ex_spot.create_market_buy_order.assert_called_once()
        ex_futures.create_market_sell_order.assert_called_once()

    def test_close_creates_both_orders(self, mock_exchanges, config):
        ex_futures, ex_spot = mock_exchanges
        state = LiveArbitrageState(
            is_open=True, symbol="BTC/USDT",
            spot_qty=0.06, spot_entry_price=50000.0,
            futures_qty=0.06, accumulated_fr=1.5,
        )

        close_position(ex_futures, ex_spot, state, config)

        assert state.is_open is False
        assert state.num_round_trips == 1
        assert len(state.trades) == 1
        ex_spot.create_market_sell_order.assert_called_once()
        ex_futures.create_market_buy_order.assert_called_once()

    def test_close_no_position_warns(self, mock_exchanges, config):
        ex_futures, ex_spot = mock_exchanges
        state = LiveArbitrageState()

        close_position(ex_futures, ex_spot, state, config)

        assert state.num_round_trips == 0
        ex_spot.create_market_sell_order.assert_not_called()

    def test_open_rollback_on_futures_failure(self, mock_exchanges, config):
        """Futures short fails -> spot buy is rolled back."""
        ex_futures, ex_spot = mock_exchanges
        ex_futures.create_market_sell_order.side_effect = Exception("network error")
        state = LiveArbitrageState()

        open_position(ex_futures, ex_spot, state, "BTC/USDT", 3000.0, config)

        assert state.is_open is False
        # Spot buy happened, then rollback sell
        ex_spot.create_market_buy_order.assert_called_once()
        ex_spot.create_market_sell_order.assert_called_once()

    def test_open_aborts_on_spot_failure(self, mock_exchanges, config):
        """Spot buy fails -> no futures order, no position."""
        ex_futures, ex_spot = mock_exchanges
        ex_spot.create_market_buy_order.side_effect = Exception("insufficient funds")
        state = LiveArbitrageState()

        open_position(ex_futures, ex_spot, state, "BTC/USDT", 3000.0, config)

        assert state.is_open is False
        ex_futures.create_market_sell_order.assert_not_called()


# ---------------------------------------------------------------------------
# Settlement time helper
# ---------------------------------------------------------------------------

class TestNextSettlementTime:
    def test_returns_future_timestamp(self):
        now_ms = int(time.time() * 1000)
        next_ms = _next_settlement_time(now_ms)
        assert next_ms > now_ms

    def test_returns_valid_hour(self):
        from datetime import datetime, timezone
        now_ms = int(time.time() * 1000)
        next_ms = _next_settlement_time(now_ms)
        dt = datetime.fromtimestamp(next_ms / 1000, tz=timezone.utc)
        assert dt.hour in (0, 8, 16)
        assert dt.minute == 0
        assert dt.second == 0
