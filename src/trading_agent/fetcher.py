"""Fetch OHLCV data from Binance via ccxt (public API, no auth required)."""

import time

import ccxt
import pandas as pd

# Binance max per request
_PAGE_SIZE = 1000

# Timeframe → milliseconds
_TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


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


def fetch_ohlcv_paginated(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    total: int = 2000,
) -> pd.DataFrame:
    """Fetch more candles than the exchange limit by paginating backwards."""
    exchange = ccxt.binance()
    tf_ms = _TF_MS.get(timeframe, 3_600_000)

    all_data: list[list] = []
    # Start from now, go backwards
    end_ms = int(time.time() * 1000)
    remaining = total

    while remaining > 0:
        batch_size = min(remaining, _PAGE_SIZE)
        since_ms = end_ms - batch_size * tf_ms

        raw = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=batch_size,
        )
        if not raw:
            break

        all_data = raw + all_data
        end_ms = raw[0][0] - 1  # 1ms before oldest candle
        remaining -= len(raw)

        if len(raw) < batch_size:
            break
        time.sleep(0.2)  # rate limit courtesy

    # Deduplicate by timestamp, sort ascending
    seen = set()
    deduped = []
    for row in all_data:
        ts = row[0]
        if ts not in seen:
            seen.add(ts)
            deduped.append(row)
    deduped.sort(key=lambda r: r[0])

    df = pd.DataFrame(deduped, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
