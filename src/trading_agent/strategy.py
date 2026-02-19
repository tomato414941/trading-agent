"""Signal generation with RSI, MACD, and cooldown."""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

# RSI thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Default cooldown: skip N candles after a buy before allowing another buy
DEFAULT_BUY_COOLDOWN = 12


def compute_indicators(df: pd.DataFrame, rsi_window: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = RSIIndicator(close=df["close"], window=rsi_window).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()  # histogram: macd - signal
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


class SignalFilter:
    """Wraps raw signals with cooldown logic."""

    def __init__(self, buy_cooldown: int = DEFAULT_BUY_COOLDOWN):
        self.buy_cooldown = buy_cooldown
        self._ticks_since_buy = buy_cooldown  # allow first buy immediately

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


def generate_signal(df: pd.DataFrame) -> str:
    return rsi_signal(df.iloc[-1]["rsi"])
