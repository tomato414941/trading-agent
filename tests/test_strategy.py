import pytest
from trading_agent.strategy import (
    rsi_signal,
    composite_signal,
    sentiment_weighted_signal,
    sentiment_multiplier,
    SignalFilter,
)


class TestRsiSignal:
    @pytest.mark.parametrize("rsi,expected", [
        (25, "buy"),
        (29.9, "buy"),
        (30, "hold"),
        (50, "hold"),
        (70, "hold"),
        (70.1, "sell"),
        (85, "sell"),
        (float("nan"), "hold"),
    ])
    def test_thresholds(self, rsi, expected):
        assert rsi_signal(rsi) == expected


class TestCompositeSignal:
    @pytest.mark.parametrize("rsi,macd_diff,prev_macd_diff,expected", [
        (25, 0.5, 0.3, "buy"),      # oversold + rising
        (25, 0.3, 0.5, "hold"),     # oversold + falling
        (75, 0.3, 0.5, "sell"),     # overbought + falling
        (75, 0.5, 0.3, "hold"),     # overbought + rising
        (50, 0.5, 0.3, "hold"),     # neutral RSI
        (float("nan"), 0.5, 0.3, "hold"),
    ])
    def test_composite(self, rsi, macd_diff, prev_macd_diff, expected):
        assert composite_signal(rsi, macd_diff, prev_macd_diff) == expected


class TestSentimentWeightedSignal:
    @pytest.mark.parametrize("rsi,macd_diff,prev,score,expected", [
        (35, 0.5, 0.3, 0.5, "buy"),     # RSI 35 < 40 (bullish relaxed)
        (35, 0.5, 0.3, 0.0, "hold"),    # RSI 35 > 30 (neutral, no buy)
        (65, 0.3, 0.5, -0.5, "sell"),   # RSI 65 > 60 (bearish relaxed)
        (65, 0.3, 0.5, 0.0, "hold"),    # RSI 65 < 70 (neutral, no sell)
        (25, 0.5, 0.3, 0.0, "buy"),     # RSI < 30, always buy
        (75, 0.3, 0.5, 0.0, "sell"),    # RSI > 70, always sell
    ])
    def test_sentiment_thresholds(self, rsi, macd_diff, prev, score, expected):
        assert sentiment_weighted_signal(rsi, macd_diff, prev, score) == expected


class TestSentimentMultiplier:
    @pytest.mark.parametrize("score,expected", [
        (0.0, 1.0),       # neutral → 1x
        (1.0, 1.25),      # max bullish → 1.25x
        (-1.0, 0.75),     # max bearish → 0.75x
        (0.5, 1.125),     # moderate bullish
        (-0.5, 0.875),    # moderate bearish
        (2.0, 1.25),      # clamped to +1
        (-3.0, 0.75),     # clamped to -1
    ])
    def test_multiplier_values(self, score, expected):
        assert sentiment_multiplier(score) == pytest.approx(expected)

    def test_custom_scale(self):
        assert sentiment_multiplier(1.0, scale=0.5) == pytest.approx(1.5)
        assert sentiment_multiplier(-1.0, scale=0.5) == pytest.approx(0.5)


class TestSignalFilter:
    def test_first_buy_allowed(self):
        sf = SignalFilter(buy_cooldown=12)
        assert sf.filter("buy") == "buy"

    def test_cooldown_blocks_buy(self):
        sf = SignalFilter(buy_cooldown=12)
        sf.filter("buy")
        assert sf.filter("buy") == "hold"

    def test_sell_passes_through(self):
        sf = SignalFilter(buy_cooldown=12)
        assert sf.filter("sell") == "sell"

    def test_hold_passes_through(self):
        sf = SignalFilter(buy_cooldown=12)
        assert sf.filter("hold") == "hold"

    def test_cooldown_expires(self):
        sf = SignalFilter(buy_cooldown=3)
        sf.filter("buy")  # tick 0
        for _ in range(3):
            sf.filter("hold")  # ticks 1-3
        assert sf.filter("buy") == "buy"  # tick 4

    def test_ticks_since_buy_property(self):
        sf = SignalFilter(buy_cooldown=12)
        sf.ticks_since_buy = 5
        assert sf.ticks_since_buy == 5
