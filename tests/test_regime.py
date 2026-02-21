from trading_agent.regime import compute_regime, regime_allows_buy, RegimeState


class TestComputeRegime:
    def test_uptrend(self, make_ohlcv_df):
        # Steady rise: EMA200 below price, ADX should be high
        closes = [100.0 + i * 1.0 for i in range(250)]
        df = make_ohlcv_df(closes)
        state = compute_regime(df)
        assert state.regime == "uptrend"
        assert state.ema200 > 0
        assert state.adx > 0

    def test_downtrend(self, make_ohlcv_df):
        closes = [350.0 - i * 1.0 for i in range(250)]
        df = make_ohlcv_df(closes)
        state = compute_regime(df)
        assert state.regime == "downtrend"

    def test_ranging(self, make_ohlcv_df):
        # Oscillating around a flat mean -> low ADX
        closes = [100.0 + (i % 3 - 1) * 0.01 for i in range(250)]
        df = make_ohlcv_df(closes)
        state = compute_regime(df)
        assert state.regime == "ranging"

    def test_insufficient_data_fallback(self, make_ohlcv_df):
        closes = [100.0] * 50
        df = make_ohlcv_df(closes)
        state = compute_regime(df)
        assert state.regime == "ranging"
        assert state.ema200 == 0.0
        assert state.adx == 0.0


class TestRegimeAllowsBuy:
    def test_none_allows(self):
        assert regime_allows_buy(None) is True

    def test_uptrend_allows(self):
        s = RegimeState(regime="uptrend", ema200=100, adx=30, updated_at=None)
        assert regime_allows_buy(s) is True

    def test_downtrend_blocks(self):
        s = RegimeState(regime="downtrend", ema200=100, adx=30, updated_at=None)
        assert regime_allows_buy(s) is False

    def test_ranging_allows(self):
        s = RegimeState(regime="ranging", ema200=100, adx=20, updated_at=None)
        assert regime_allows_buy(s) is True
