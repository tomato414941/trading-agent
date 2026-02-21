"""Signal generation with RSI, MACD, BB, Volume, and cooldown."""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# RSI thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Volume spike: current volume > X * rolling average
VOLUME_SPIKE_MULTIPLIER = 1.5

# Default cooldown: skip N candles after a buy before allowing another buy
DEFAULT_BUY_COOLDOWN = 12


def compute_indicators(df: pd.DataFrame, rsi_window: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = RSIIndicator(close=df["close"], window=rsi_window).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()  # histogram: macd - signal

    # Bollinger Bands (20-period, 2 std dev)
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = bb.bollinger_wband()

    # Volume metrics
    df["vol_sma"] = df["volume"].rolling(window=20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma"]

    return df


# Keep for backward compat (main.py, tests)
def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    return compute_indicators(df, rsi_window=window)


def rsi_signal(rsi: float) -> str:
    if pd.isna(rsi):
        return "hold"
    if rsi < RSI_OVERSOLD:
        return "buy"
    if rsi > RSI_OVERBOUGHT:
        return "sell"
    return "hold"


def composite_signal(rsi: float, macd_diff: float, prev_macd_diff: float) -> str:
    """RSI + MACD composite: require both indicators to agree.

    Buy:  RSI oversold AND MACD histogram rising (momentum recovering)
    Sell: RSI overbought AND MACD histogram falling (momentum fading)
    """
    if pd.isna(rsi) or pd.isna(macd_diff) or pd.isna(prev_macd_diff):
        return "hold"

    macd_rising = macd_diff > prev_macd_diff
    macd_falling = macd_diff < prev_macd_diff

    if rsi < RSI_OVERSOLD and macd_rising:
        return "buy"
    if rsi > RSI_OVERBOUGHT and macd_falling:
        return "sell"
    return "hold"


def sentiment_weighted_signal(
    rsi: float,
    macd_diff: float,
    prev_macd_diff: float,
    sentiment_score: float,
) -> str:
    """RSI + MACD + sentiment: LLM sentiment acts as a gate/boost.

    - sentiment_score > 0.3:  bullish news relaxes RSI buy threshold (< 40)
    - sentiment_score < -0.3: bearish news relaxes RSI sell threshold (> 60)
    - Otherwise: same as composite_signal (RSI < 30 + MACD rising)

    Sentiment NEVER overrides technicals alone — it only widens the trigger zone.
    """
    if pd.isna(rsi) or pd.isna(macd_diff) or pd.isna(prev_macd_diff):
        return "hold"

    macd_rising = macd_diff > prev_macd_diff
    macd_falling = macd_diff < prev_macd_diff

    # Adjust thresholds based on sentiment
    buy_threshold = RSI_OVERSOLD
    sell_threshold = RSI_OVERBOUGHT

    if sentiment_score > 0.3:
        buy_threshold = 40  # bullish news: buy on mild dips too
    elif sentiment_score < -0.3:
        sell_threshold = 60  # bearish news: sell earlier

    if rsi < buy_threshold and macd_rising:
        return "buy"
    if rsi > sell_threshold and macd_falling:
        return "sell"
    return "hold"


def sentiment_multiplier(score: float, scale: float = 0.25) -> float:
    """Smooth position size multiplier based on sentiment score.

    score [-1, +1] -> multiplier [1-scale, 1+scale]
    Default scale=0.25: bullish(+1) -> 1.25x, bearish(-1) -> 0.75x
    """
    clamped = max(-1.0, min(1.0, score))
    return 1.0 + clamped * scale


class SignalFilter:
    """Wraps raw signals with cooldown logic."""

    def __init__(self, buy_cooldown: int = DEFAULT_BUY_COOLDOWN):
        self.buy_cooldown = buy_cooldown
        self._ticks_since_buy = buy_cooldown  # allow first buy immediately

    @property
    def ticks_since_buy(self) -> int:
        return self._ticks_since_buy

    @ticks_since_buy.setter
    def ticks_since_buy(self, value: int) -> None:
        self._ticks_since_buy = value

    def filter(self, raw_signal: str) -> str:
        self._ticks_since_buy += 1

        if raw_signal == "buy":
            if self._ticks_since_buy > self.buy_cooldown:
                self._ticks_since_buy = 0
                return "buy"
            return "hold"

        if raw_signal == "sell":
            return "sell"

        return "hold"


def bb_volume_signal(
    row: pd.Series,
    prev_row: pd.Series | None,
) -> str:
    """Bollinger Band breakout + volume spike.

    Buy:  price touches/crosses lower BB AND volume spike (vol_ratio > 1.5)
    Sell: price touches/crosses upper BB AND volume spike
    """
    close = row.get("close")
    bb_lower = row.get("bb_lower")
    bb_upper = row.get("bb_upper")
    vol_ratio = row.get("vol_ratio")

    if any(pd.isna(v) for v in [close, bb_lower, bb_upper, vol_ratio]):
        return "hold"

    is_volume_spike = vol_ratio >= VOLUME_SPIKE_MULTIPLIER

    if close <= bb_lower and is_volume_spike:
        return "buy"
    if close >= bb_upper and is_volume_spike:
        return "sell"
    return "hold"


def bb_rsi_signal(
    row: pd.Series,
    prev_row: pd.Series | None,
) -> str:
    """Combined BB + RSI + Volume: require BB touch + RSI confirmation + volume.

    Buy:  price <= lower BB AND RSI < 40 AND volume spike
    Sell: price >= upper BB AND RSI > 60 AND volume spike
    """
    close = row.get("close")
    bb_lower = row.get("bb_lower")
    bb_upper = row.get("bb_upper")
    rsi = row.get("rsi")
    vol_ratio = row.get("vol_ratio")

    if any(pd.isna(v) for v in [close, bb_lower, bb_upper, rsi, vol_ratio]):
        return "hold"

    is_volume_spike = vol_ratio >= VOLUME_SPIKE_MULTIPLIER

    if close <= bb_lower and rsi < 40 and is_volume_spike:
        return "buy"
    if close >= bb_upper and rsi > 60 and is_volume_spike:
        return "sell"
    return "hold"


def generate_signal(df: pd.DataFrame) -> str:
    return rsi_signal(df.iloc[-1]["rsi"])
