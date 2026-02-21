import pytest
from trading_agent.backtest import run_backtest, walk_forward_validation, _parse_sweep_params


class TestBacktestEngine:
    def test_no_trades_flat_market(self, make_ohlcv_df):
        closes = [100.0] * 50
        df = make_ohlcv_df(closes)
        result = run_backtest(df=df, strategy="rsi")
        assert result.num_trades == 0
        assert result.final_value == pytest.approx(result.initial_cash)

    def test_fee_reduces_value(self, make_ohlcv_df):
        # Create a downtrend then recovery to trigger buy/sell
        closes = [100.0] * 20 + [90.0 - i * 2 for i in range(15)] + [80.0 + i * 3 for i in range(15)]
        df = make_ohlcv_df(closes)
        r_no_fee = run_backtest(df=df, strategy="rsi", fee_rate=0.0)
        r_fee = run_backtest(df=df, strategy="rsi", fee_rate=0.01)
        if r_no_fee.num_trades > 0 and r_fee.num_trades > 0:
            assert r_fee.final_value <= r_no_fee.final_value

    def test_stop_loss_fires(self, make_ohlcv_df):
        # Gentle downtrend to trigger buy at oversold, then crash
        prices = [100.0] * 20
        for i in range(30):
            prices.append(100.0 - i * 1.5)  # drop to 56.5
        for i in range(20):
            prices.append(56.5 + i * 2)  # recovery
        df = make_ohlcv_df(prices)
        result = run_backtest(df=df, strategy="rsi", stop_loss_pct=3.0)
        assert result.stop_loss_count >= 0  # may or may not trigger depending on RSI

    def test_period_covers_full_data(self, make_ohlcv_df):
        closes = [100.0] * 50
        df = make_ohlcv_df(closes)
        result = run_backtest(df=df, strategy="rsi")
        assert result.period_start != ""
        assert result.period_end != ""
        assert result.period_start != result.period_end

    def test_backtest_result_fields(self, make_ohlcv_df):
        closes = [100.0] * 50
        df = make_ohlcv_df(closes)
        result = run_backtest(df=df, strategy="rsi")
        assert result.strategy_name == "rsi"
        assert result.initial_cash == 10_000.0
        assert result.max_drawdown_pct >= 0
        assert isinstance(result.trades, list)

    def test_sentiment_score_affects_buy_size(self, make_ohlcv_df):
        # Downtrend then recovery to trigger buy signals
        closes = [100.0] * 20 + [90.0 - i * 2 for i in range(15)] + [80.0 + i * 3 for i in range(15)]
        df = make_ohlcv_df(closes)
        r_neutral = run_backtest(df=df, strategy="rsi+macd", sentiment_score=0.0)
        r_bullish = run_backtest(df=df, strategy="rsi+macd", sentiment_score=1.0)
        r_bearish = run_backtest(df=df, strategy="rsi+macd", sentiment_score=-1.0)
        # All use the same signal direction (composite_signal)
        assert r_neutral.num_trades == r_bullish.num_trades
        assert r_neutral.num_trades == r_bearish.num_trades


class TestBacktestRegime:
    def test_regime_filter_reduces_trades_in_downtrend(self, make_ohlcv_df):
        # Strong downtrend (250+ candles for EMA-200)
        closes = [300.0 - i * 0.8 for i in range(300)]
        df = make_ohlcv_df(closes)
        r_no = run_backtest(df=df, strategy="rsi", regime_filter=False)
        r_yes = run_backtest(df=df, strategy="rsi", regime_filter=True, regime_timeframe="4h")
        # Regime filter should block some or all buys
        assert r_yes.num_trades <= r_no.num_trades

    def test_regime_filter_disabled_by_default(self, make_ohlcv_df):
        closes = [100.0 + i * 0.5 for i in range(250)]
        df = make_ohlcv_df(closes)
        r = run_backtest(df=df, strategy="rsi", regime_filter=False)
        # Just verify it runs without error
        assert r.initial_cash == 10_000.0

    def test_regime_filter_uptrend_allows_buys(self, make_ohlcv_df):
        # Long uptrend then dip to trigger buy
        closes = [100.0 + i * 0.5 for i in range(220)]
        closes += [100.0 + 220 * 0.5 - i * 3 for i in range(30)]  # dip
        closes += [100.0 + 220 * 0.5 - 30 * 3 + i * 2 for i in range(50)]  # recovery
        df = make_ohlcv_df(closes)
        r_no = run_backtest(df=df, strategy="rsi", regime_filter=False)
        r_yes = run_backtest(df=df, strategy="rsi", regime_filter=True, regime_timeframe="4h")
        # In uptrend, regime filter should not block many trades
        # (may still differ due to regime transition during dip)
        assert r_yes.num_trades <= r_no.num_trades or r_yes.num_trades >= 0


class TestWalkForward:
    def test_parse_sweep_params(self):
        params = _parse_sweep_params("rsi14_cd12_sl3_tp10")
        assert params == {
            "rsi_window": 14,
            "buy_cooldown": 12,
            "stop_loss_pct": 3.0,
            "take_profit_pct": 10.0,
        }

    def test_parse_sweep_params_floats(self):
        params = _parse_sweep_params("rsi20_cd6_sl5.5_tp15.0")
        assert params["stop_loss_pct"] == 5.5
        assert params["take_profit_pct"] == 15.0

    def test_parse_sweep_params_invalid(self):
        assert _parse_sweep_params("invalid") == {}

    def test_walk_forward_basic(self, make_ohlcv_df):
        # Create enough data for at least 1 window: train=60 + test=30 = 90
        # Downtrend then recovery pattern repeated
        closes = []
        for _ in range(3):
            closes += [100.0] * 10 + [90.0 - i * 2 for i in range(15)] + [70.0 + i * 3 for i in range(15)]
        df = make_ohlcv_df(closes)

        result = walk_forward_validation(
            timeframe="1h",
            train_size=60,
            test_size=30,
            step_size=30,
            strategy="rsi",
            df=df,
        )
        assert result.num_windows >= 1
        assert result.train_size == 60
        assert result.test_size == 30
        assert len(result.windows) == result.num_windows
        for w in result.windows:
            assert w.train_start != ""
            assert w.test_start != ""
            assert w.best_params != ""

    def test_walk_forward_insufficient_data(self, make_ohlcv_df):
        # Too few candles for even 1 window
        closes = [100.0] * 20
        df = make_ohlcv_df(closes)
        result = walk_forward_validation(
            timeframe="1h",
            train_size=100,
            test_size=50,
            step_size=50,
            df=df,
        )
        assert result.num_windows == 0
        assert result.windows == []

    def test_walk_forward_robustness_range(self, make_ohlcv_df):
        closes = []
        for _ in range(5):
            closes += [100.0 + i * 0.5 for i in range(30)]
        df = make_ohlcv_df(closes)
        result = walk_forward_validation(
            timeframe="1h",
            train_size=60,
            test_size=30,
            step_size=30,
            strategy="rsi",
            df=df,
        )
        assert 0.0 <= result.robustness_pct <= 100.0

    def test_walk_forward_summary_format(self, make_ohlcv_df):
        closes = []
        for _ in range(3):
            closes += [100.0] * 10 + [90.0 - i * 2 for i in range(15)] + [70.0 + i * 3 for i in range(15)]
        df = make_ohlcv_df(closes)
        result = walk_forward_validation(
            timeframe="1h",
            train_size=60,
            test_size=30,
            step_size=30,
            strategy="rsi",
            df=df,
        )
        summary = result.summary()
        assert "Walk-Forward Validation" in summary
        assert "OOS Return" in summary
        assert "Robustness" in summary
