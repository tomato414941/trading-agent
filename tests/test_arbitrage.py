"""Tests for funding rate arbitrage analysis and backtest."""

import pandas as pd
import pytest

from trading_agent.arbitrage import (
    FRAnalysisResult,
    _compute_negative_streaks,
    _merge_price_to_fr,
    analyze_funding_rate,
    run_arbitrage_backtest,
)
from trading_agent.config import ArbitrageConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_fr_df():
    """Factory: create FR DataFrame with specified rates."""
    def _make(rates: list[float], start: str = "2024-01-01") -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": pd.date_range(start, periods=len(rates), freq="8h"),
            "funding_rate": rates,
        })
    return _make


@pytest.fixture
def make_price_df():
    """Factory: create price DataFrame aligned to FR timestamps."""
    def _make(
        prices: list[float],
        start: str = "2024-01-01",
        freq: str = "8h",
    ) -> pd.DataFrame:
        n = len(prices)
        return pd.DataFrame({
            "timestamp": pd.date_range(start, periods=n, freq=freq),
            "open": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        })
    return _make


# ---------------------------------------------------------------------------
# Negative Streak Tests
# ---------------------------------------------------------------------------

class TestNegativeStreaks:
    def test_no_negatives(self):
        rates = pd.Series([0.001, 0.002, 0.0005])
        max_s, avg_s = _compute_negative_streaks(rates)
        assert max_s == 0
        assert avg_s == 0.0

    def test_all_negatives(self):
        rates = pd.Series([-0.001, -0.002, -0.0005])
        max_s, avg_s = _compute_negative_streaks(rates)
        assert max_s == 3
        assert avg_s == 3.0

    def test_mixed_streaks(self):
        # Streaks: [2, 1]
        rates = pd.Series([0.001, -0.001, -0.002, 0.001, -0.001, 0.001])
        max_s, avg_s = _compute_negative_streaks(rates)
        assert max_s == 2
        assert avg_s == 1.5

    def test_empty_series(self):
        rates = pd.Series([], dtype=float)
        max_s, avg_s = _compute_negative_streaks(rates)
        assert max_s == 0
        assert avg_s == 0.0


# ---------------------------------------------------------------------------
# FR Analysis Tests
# ---------------------------------------------------------------------------

class TestFRAnalysis:
    def test_constant_positive_fr(self, make_fr_df):
        fr_df = make_fr_df([0.0005] * 100)
        result = analyze_funding_rate("BTC/USDT", fr_df=fr_df)
        assert isinstance(result, FRAnalysisResult)
        assert result.mean_fr == pytest.approx(0.0005)
        # 0.0005 * 3 * 365 * 100 = 54.75%
        assert result.annualized_yield_pct == pytest.approx(54.75)
        assert result.positive_pct == pytest.approx(100.0)
        assert result.max_negative_streak == 0

    def test_all_zero_fr(self, make_fr_df):
        fr_df = make_fr_df([0.0] * 50)
        result = analyze_funding_rate("BTC/USDT", fr_df=fr_df)
        assert result.annualized_yield_pct == pytest.approx(0.0)
        assert result.positive_pct == pytest.approx(0.0)

    def test_mixed_fr_positive_pct(self, make_fr_df):
        # 7 positive, 3 negative = 70%
        rates = [0.001] * 7 + [-0.001] * 3
        fr_df = make_fr_df(rates)
        result = analyze_funding_rate("BTC/USDT", fr_df=fr_df)
        assert result.positive_pct == pytest.approx(70.0)
        assert result.max_negative_streak == 3

    def test_summary_format(self, make_fr_df):
        fr_df = make_fr_df([0.0003] * 20)
        result = analyze_funding_rate("BTC/USDT", fr_df=fr_df)
        summary = result.summary()
        assert "FR Analysis: BTC/USDT" in summary
        assert "Annualized Yield" in summary
        assert "Net Ann. Yield" in summary


# ---------------------------------------------------------------------------
# Merge Price to FR Tests
# ---------------------------------------------------------------------------

class TestMergePriceToFR:
    def test_basic_merge(self, make_fr_df, make_price_df):
        fr_df = make_fr_df([0.001, 0.002], start="2024-01-01")
        price_df = make_price_df([50000.0, 50100.0], start="2024-01-01")
        merged = _merge_price_to_fr(fr_df, price_df)
        assert "spot_price" in merged.columns
        assert len(merged) == 2
        assert merged.iloc[0]["spot_price"] == 50000.0


# ---------------------------------------------------------------------------
# Backtest Tests
# ---------------------------------------------------------------------------

class TestArbitrageBacktest:
    def test_no_entry_below_threshold(self, make_fr_df, make_price_df):
        """FR below threshold => no trades, cash unchanged."""
        fr_df = make_fr_df([0.0001] * 20)  # below default 0.0003
        price_df = make_price_df([50000.0] * 20)
        result = run_arbitrage_backtest(
            fr_df=fr_df, price_df=price_df, initial_cash=10000.0,
        )
        assert result.num_round_trips == 0
        assert result.total_fr_collected == pytest.approx(0.0)
        assert result.final_value == pytest.approx(10000.0)

    def test_entry_and_fr_collection(self, make_fr_df, make_price_df):
        """FR above threshold => opens position and collects FR."""
        # High FR for 20 periods, price flat
        fr_df = make_fr_df([0.001] * 20)
        price_df = make_price_df([50000.0] * 20)
        result = run_arbitrage_backtest(
            fr_df=fr_df, price_df=price_df, initial_cash=10000.0,
        )
        assert result.total_fr_collected > 0
        assert result.final_value > result.initial_cash

    def test_exit_on_consecutive_low_fr(self, make_fr_df, make_price_df):
        """Position closes after N consecutive low FR periods."""
        # 10 high FR, then 3 low FR (triggers exit), then 5 high FR
        rates = [0.001] * 10 + [0.00005] * 3 + [0.001] * 5
        fr_df = make_fr_df(rates)
        price_df = make_price_df([50000.0] * len(rates))
        config = ArbitrageConfig(
            exit_fr_threshold=0.0001,
            exit_consecutive_periods=3,
        )
        result = run_arbitrage_backtest(
            fr_df=fr_df, price_df=price_df, config=config, initial_cash=10000.0,
        )
        # Should have at least 1 completed round trip (closed by low FR)
        # plus possibly 1 more that's force-closed at end
        assert result.num_round_trips >= 1
        assert len(result.trades) >= 1

    def test_fees_reduce_return(self, make_fr_df, make_price_df):
        """Higher fees => lower return."""
        fr_df = make_fr_df([0.001] * 30)
        price_df = make_price_df([50000.0] * 30)

        low_fee_config = ArbitrageConfig(spot_fee_rate=0.0, futures_fee_rate=0.0)
        high_fee_config = ArbitrageConfig(spot_fee_rate=0.005, futures_fee_rate=0.005)

        result_low = run_arbitrage_backtest(
            fr_df=fr_df, price_df=price_df, config=low_fee_config,
        )
        result_high = run_arbitrage_backtest(
            fr_df=fr_df, price_df=price_df, config=high_fee_config,
        )
        assert result_low.final_value > result_high.final_value

    def test_flat_price_pure_fr_yield(self, make_fr_df, make_price_df):
        """With flat price, return should be pure FR minus fees."""
        fr_df = make_fr_df([0.001] * 50)
        price_df = make_price_df([50000.0] * 50)
        result = run_arbitrage_backtest(
            fr_df=fr_df, price_df=price_df, initial_cash=10000.0,
        )
        # FR collected should be positive
        assert result.total_fr_collected > 0
        # Net return = FR collected - fees
        net = result.total_fr_collected - result.total_fees_paid
        # final_value should be close to initial + net
        assert result.final_value == pytest.approx(
            result.initial_cash + net, rel=0.01,
        )

    def test_round_trip_count(self, make_fr_df, make_price_df):
        """Multiple entry/exit cycles counted correctly."""
        # Cycle 1: 5 high, 3 low (exit). Cycle 2: 5 high, 3 low (exit).
        rates = ([0.001] * 5 + [0.00005] * 3) * 2
        fr_df = make_fr_df(rates)
        price_df = make_price_df([50000.0] * len(rates))
        config = ArbitrageConfig(exit_consecutive_periods=3)
        result = run_arbitrage_backtest(
            fr_df=fr_df, price_df=price_df, config=config,
        )
        assert result.num_round_trips == 2

    def test_equity_curve_length(self, make_fr_df, make_price_df):
        """Equity curve has one entry per FR period."""
        n = 25
        fr_df = make_fr_df([0.001] * n)
        price_df = make_price_df([50000.0] * n)
        result = run_arbitrage_backtest(fr_df=fr_df, price_df=price_df)
        assert len(result.equity_curve) == n

    def test_max_drawdown_non_negative(self, make_fr_df, make_price_df):
        """Max drawdown is always >= 0."""
        fr_df = make_fr_df([0.001, -0.002, 0.001, -0.003, 0.001] * 4)
        price_df = make_price_df([50000.0] * 20)
        result = run_arbitrage_backtest(fr_df=fr_df, price_df=price_df)
        assert result.max_drawdown_pct >= 0

    def test_summary_format(self, make_fr_df, make_price_df):
        """Summary string contains key metrics."""
        fr_df = make_fr_df([0.001] * 10)
        price_df = make_price_df([50000.0] * 10)
        result = run_arbitrage_backtest(fr_df=fr_df, price_df=price_df)
        summary = result.summary()
        assert "FR Arbitrage Backtest" in summary
        assert "Annualized" in summary
        assert "FR Collected" in summary
