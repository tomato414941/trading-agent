"""RSI-based signal generation with cooldown."""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator

# Thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Default cooldown: skip N candles after a buy before allowing another buy
DEFAULT_BUY_COOLDOWN = 12


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    rsi = RSIIndicator(close=df["close"], window=window)
    df = df.copy()
    df["rsi"] = rsi.rsi()
    return df


def rsi_signal(rsi: float) -> str:
    if pd.isna(rsi):
        return "hold"
    if rsi < RSI_OVERSOLD:
        return "buy"
    if rsi > RSI_OVERBOUGHT:
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
