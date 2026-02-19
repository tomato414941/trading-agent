"""RSI-based signal generation."""

import pandas as pd
from ta.momentum import RSIIndicator

# Thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


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


def generate_signal(df: pd.DataFrame) -> str:
    return rsi_signal(df.iloc[-1]["rsi"])
