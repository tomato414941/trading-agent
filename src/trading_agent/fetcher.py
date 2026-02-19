"""Fetch OHLCV data from Binance via ccxt (public API, no auth required)."""

import ccxt
import pandas as pd


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 100,
) -> pd.DataFrame:
    exchange = ccxt.binance()
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
