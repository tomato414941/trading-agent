import pandas as pd
import pytest
from trading_agent.strategy import (
    rsi_signal,
    composite_signal,
    sentiment_weighted_signal,
    sentiment_multiplier,
    bb_volume_signal,
    bb_rsi_signal,
    funding_rate_signal,
    bb_volume_funding_signal,
    onchain_signal,
    bb_onchain_signal,
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


class TestBBVolumeSignal:
    def _make_row(self, close, bb_lower, bb_upper, vol_ratio):
        return pd.Series({
            "close": close,
            "bb_lower": bb_lower,
            "bb_upper": bb_upper,
            "vol_ratio": vol_ratio,
        })

    def test_buy_at_lower_bb_with_volume(self):
        row = self._make_row(close=95, bb_lower=96, bb_upper=104, vol_ratio=2.0)
        assert bb_volume_signal(row, None) == "buy"

    def test_sell_at_upper_bb_with_volume(self):
        row = self._make_row(close=105, bb_lower=96, bb_upper=104, vol_ratio=2.0)
        assert bb_volume_signal(row, None) == "sell"

    def test_hold_no_volume_spike(self):
        row = self._make_row(close=95, bb_lower=96, bb_upper=104, vol_ratio=1.0)
        assert bb_volume_signal(row, None) == "hold"

    def test_hold_in_middle(self):
        row = self._make_row(close=100, bb_lower=96, bb_upper=104, vol_ratio=2.0)
        assert bb_volume_signal(row, None) == "hold"

    def test_hold_on_nan(self):
        row = self._make_row(close=95, bb_lower=float("nan"), bb_upper=104, vol_ratio=2.0)
        assert bb_volume_signal(row, None) == "hold"


class TestBBRsiSignal:
    def _make_row(self, close, bb_lower, bb_upper, rsi, vol_ratio):
        return pd.Series({
            "close": close,
            "bb_lower": bb_lower,
            "bb_upper": bb_upper,
            "rsi": rsi,
            "vol_ratio": vol_ratio,
        })

    def test_buy_all_conditions(self):
        row = self._make_row(close=95, bb_lower=96, bb_upper=104, rsi=35, vol_ratio=2.0)
        assert bb_rsi_signal(row, None) == "buy"

    def test_sell_all_conditions(self):
        row = self._make_row(close=105, bb_lower=96, bb_upper=104, rsi=65, vol_ratio=2.0)
        assert bb_rsi_signal(row, None) == "sell"

    def test_hold_rsi_not_low_enough(self):
        row = self._make_row(close=95, bb_lower=96, bb_upper=104, rsi=45, vol_ratio=2.0)
        assert bb_rsi_signal(row, None) == "hold"

    def test_hold_no_volume(self):
        row = self._make_row(close=95, bb_lower=96, bb_upper=104, rsi=35, vol_ratio=1.0)
        assert bb_rsi_signal(row, None) == "hold"


class TestFundingRateSignal:
    def test_high_positive_fr_sell(self):
        assert funding_rate_signal(0.001) == "sell"

    def test_high_negative_fr_buy(self):
        assert funding_rate_signal(-0.001) == "buy"

    def test_neutral_fr_hold(self):
        assert funding_rate_signal(0.0001) == "hold"

    def test_nan_hold(self):
        assert funding_rate_signal(float("nan")) == "hold"

    def test_custom_threshold(self):
        assert funding_rate_signal(0.0003, threshold=0.0002) == "sell"
        assert funding_rate_signal(0.0003, threshold=0.0005) == "hold"


class TestBBVolumeFundingSignal:
    def _make_row(self, close, bb_lower, bb_upper, vol_ratio):
        return pd.Series({
            "close": close,
            "bb_lower": bb_lower,
            "bb_upper": bb_upper,
            "vol_ratio": vol_ratio,
        })

    def test_bb_and_fr_agree_buy(self):
        row = self._make_row(close=95, bb_lower=96, bb_upper=104, vol_ratio=2.0)
        assert bb_volume_funding_signal(row, None, funding_rate=-0.001) == "buy"

    def test_bb_signal_alone(self):
        row = self._make_row(close=95, bb_lower=96, bb_upper=104, vol_ratio=2.0)
        assert bb_volume_funding_signal(row, None, funding_rate=0.0) == "buy"

    def test_fr_signal_alone(self):
        row = self._make_row(close=100, bb_lower=96, bb_upper=104, vol_ratio=1.0)
        assert bb_volume_funding_signal(row, None, funding_rate=-0.001) == "buy"

    def test_no_signal(self):
        row = self._make_row(close=100, bb_lower=96, bb_upper=104, vol_ratio=1.0)
        assert bb_volume_funding_signal(row, None, funding_rate=0.0) == "hold"


class TestOnchainSignal:
    @staticmethod
    def _make_row(**kwargs):
        defaults = {"sth_sopr": 1.0, "sth_mvrv": 1.0, "exchange_netflow": 0.0}
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_buy_capitulation(self):
        row = self._make_row(sth_sopr=0.95, sth_mvrv=0.9, exchange_netflow=-100)
        assert onchain_signal(row, None) == "buy"

    def test_buy_two_of_three(self):
        row = self._make_row(sth_sopr=0.95, sth_mvrv=0.9, exchange_netflow=50)
        assert onchain_signal(row, None) == "buy"

    def test_sell_distribution(self):
        row = self._make_row(sth_sopr=1.1, sth_mvrv=1.4, exchange_netflow=100)
        assert onchain_signal(row, None) == "sell"

    def test_sell_two_of_three(self):
        row = self._make_row(sth_sopr=1.1, sth_mvrv=1.4, exchange_netflow=-50)
        assert onchain_signal(row, None) == "sell"

    def test_hold_mixed(self):
        row = self._make_row(sth_sopr=0.95, sth_mvrv=1.1, exchange_netflow=50)
        assert onchain_signal(row, None) == "hold"

    def test_hold_neutral(self):
        row = self._make_row(sth_sopr=1.02, sth_mvrv=1.1, exchange_netflow=0)
        assert onchain_signal(row, None) == "hold"

    def test_nan_handling(self):
        row = self._make_row(sth_sopr=float("nan"), sth_mvrv=0.9, exchange_netflow=-100)
        assert onchain_signal(row, None) == "buy"


class TestBBOnchainSignal:
    @staticmethod
    def _make_row(**kwargs):
        defaults = {
            "close": 100, "bb_lower": 96, "bb_upper": 104,
            "vol_ratio": 1.0, "vol_sma": 100, "volume": 100,
            "sth_sopr": 1.0, "sth_mvrv": 1.0, "exchange_netflow": 0.0,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_both_agree_buy(self):
        row = self._make_row(
            close=95, bb_lower=96, vol_ratio=2.0,
            sth_sopr=0.95, sth_mvrv=0.9, exchange_netflow=-100,
        )
        assert bb_onchain_signal(row, None) == "buy"

    def test_bb_only(self):
        row = self._make_row(close=95, bb_lower=96, vol_ratio=2.0)
        assert bb_onchain_signal(row, None) == "buy"

    def test_strong_onchain_override(self):
        row = self._make_row(
            close=100, bb_lower=96, vol_ratio=1.0,
            sth_sopr=0.9, sth_mvrv=0.8, exchange_netflow=-500,
        )
        assert bb_onchain_signal(row, None) == "buy"

    def test_hold_when_no_signals(self):
        row = self._make_row()
        assert bb_onchain_signal(row, None) == "hold"
