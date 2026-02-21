"""Layer 3: Regime detection using EMA-200 and ADX on higher-timeframe data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from ta.trend import ADXIndicator, EMAIndicator

ADX_TREND_THRESHOLD = 25.0


@dataclass
class RegimeState:
    regime: str  # "uptrend" | "downtrend" | "ranging"
    ema200: float
    adx: float
    updated_at: datetime | None


def compute_regime(df: pd.DataFrame, adx_window: int = 14) -> RegimeState:
    """Classify market regime from OHLCV data (expects 200+ candles).

    ADX < 25 -> ranging (no strong trend)
    price > EMA200 and ADX >= 25 -> uptrend
    price < EMA200 and ADX >= 25 -> downtrend
    """
    df = df.copy()
    df["ema200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()
    adx_ind = ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=adx_window,
    )
    df["adx"] = adx_ind.adx()

    latest = df.iloc[-1]
    price = latest["close"]
    ema200 = latest["ema200"]
    adx = latest["adx"]

    if pd.isna(ema200) or pd.isna(adx):
        return RegimeState(
            regime="ranging", ema200=0.0, adx=0.0,
            updated_at=datetime.now(),
        )

    if adx < ADX_TREND_THRESHOLD:
        regime = "ranging"
    elif price > ema200:
        regime = "uptrend"
    else:
        regime = "downtrend"

    return RegimeState(
        regime=regime, ema200=float(ema200), adx=float(adx),
        updated_at=datetime.now(),
    )


def regime_allows_buy(state: RegimeState | None) -> bool:
    """Return True if regime does not block buying."""
    if state is None:
        return True
    return state.regime != "downtrend"
