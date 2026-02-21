"""Fetch OHLCV data from Binance via ccxt (public API, no auth required)."""

import json
import logging
import time
from pathlib import Path

import ccxt
import pandas as pd
import requests

log = logging.getLogger(__name__)

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


def fetch_ticker_price(symbol: str = "BTC/USDT") -> float:
    """Fetch current price via ticker API (faster than OHLCV)."""
    exchange = ccxt.binance()
    ticker = exchange.fetch_ticker(symbol)
    return float(ticker["last"])


# Funding rate interval: 8 hours
_FR_INTERVAL_MS = 8 * 3_600_000
_FR_PAGE_SIZE = 500

# Spot → futures symbol mapping
_FUTURES_SYMBOL = {
    "BTC/USDT": "BTC/USDT:USDT",
    "ETH/USDT": "ETH/USDT:USDT",
    "SOL/USDT": "SOL/USDT:USDT",
}


def _to_futures_symbol(symbol: str) -> str:
    return _FUTURES_SYMBOL.get(symbol, f"{symbol}:USDT")


def fetch_funding_rate(symbol: str = "BTC/USDT") -> dict:
    """Fetch current funding rate for a symbol."""
    exchange = ccxt.binance()
    return exchange.fetch_funding_rate(_to_futures_symbol(symbol))


def fetch_funding_rate_history(
    symbol: str = "BTC/USDT",
    total: int = 1000,
) -> pd.DataFrame:
    """Fetch funding rate history with pagination (8h intervals)."""
    exchange = ccxt.binance()
    futures_sym = _to_futures_symbol(symbol)

    all_data: list[dict] = []
    end_ms = int(time.time() * 1000)
    remaining = total

    while remaining > 0:
        batch_size = min(remaining, _FR_PAGE_SIZE)
        since_ms = end_ms - batch_size * _FR_INTERVAL_MS

        raw = exchange.fetch_funding_rate_history(
            futures_sym, since=since_ms, limit=batch_size,
        )
        if not raw:
            break

        all_data = raw + all_data
        end_ms = raw[0]["timestamp"] - 1
        remaining -= len(raw)

        if len(raw) < batch_size:
            break
        time.sleep(0.2)

    if not all_data:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.rename(columns={"fundingRate": "funding_rate"})
    df = df[["timestamp", "funding_rate"]].drop_duplicates("timestamp").sort_values("timestamp")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# On-chain data: BGeometrics (bitcoin-data.com)
# ---------------------------------------------------------------------------

_BG_BASE = "https://bitcoin-data.com/api/v1"
_BG_CACHE_DIR = Path("data/onchain_cache")

# Core 3 metrics used by onchain_signal() — fits BGeometrics Free (8 req/h, 15/day)
ONCHAIN_METRICS = {
    "sth_sopr": "sth-sopr",
    "sth_mvrv": "sth-mvrv",
    "exchange_netflow": "exchange-netflow",
}


def _bg_cache_path(metric_key: str) -> Path:
    return _BG_CACHE_DIR / f"{metric_key}.json"


def fetch_onchain_metric(
    metric_key: str,
    max_age_hours: float = 12,
) -> pd.DataFrame:
    """Fetch a single on-chain metric from BGeometrics with local cache.

    Returns DataFrame with columns: [date, value].
    Uses file cache to respect the 8 req/hour rate limit.
    """
    if metric_key not in ONCHAIN_METRICS:
        raise ValueError(f"Unknown metric: {metric_key}. Available: {list(ONCHAIN_METRICS)}")

    api_slug = ONCHAIN_METRICS[metric_key]
    cache_path = _bg_cache_path(metric_key)

    # Check cache freshness
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            log.debug("Using cached %s (%.1fh old)", metric_key, age_hours)
            data = json.loads(cache_path.read_text())
            return _bg_to_dataframe(data, metric_key)

    # Fetch from API
    url = f"{_BG_BASE}/{api_slug}"
    log.info("Fetching on-chain metric: %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Save cache
    _BG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    log.info("Cached %s: %d records", metric_key, len(data))

    return _bg_to_dataframe(data, metric_key)


def _bg_to_dataframe(data: list[dict], metric_key: str) -> pd.DataFrame:
    """Convert BGeometrics JSON to DataFrame with [date, value] columns."""
    if not data:
        return pd.DataFrame(columns=["date", "value"])

    # BGeometrics format: {"d": "2024-01-01", "unixTs": "...", "<metric>": "1.23"}
    # The value field name varies per metric; grab the non-standard field
    standard_keys = {"d", "unixTs"}
    sample = data[0]
    value_keys = [k for k in sample.keys() if k not in standard_keys]
    if not value_keys:
        raise ValueError(f"No value field found in {metric_key} response")
    value_field = value_keys[0]

    rows = []
    for rec in data:
        try:
            val = float(rec[value_field])
            rows.append({"date": rec["d"], "value": val})
        except (ValueError, KeyError, TypeError):
            continue

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_all_onchain(max_age_hours: float = 12) -> dict[str, pd.DataFrame]:
    """Fetch all configured on-chain metrics. Respects cache to stay within rate limit."""
    result = {}
    for key in ONCHAIN_METRICS:
        try:
            result[key] = fetch_onchain_metric(key, max_age_hours=max_age_hours)
        except Exception as e:
            log.warning("Failed to fetch %s: %s", key, e)
    return result


def get_latest_onchain(max_age_hours: float = 12) -> dict[str, float]:
    """Get latest value for each on-chain metric. Returns {metric_key: value}."""
    latest = {}
    for key in ONCHAIN_METRICS:
        try:
            df = fetch_onchain_metric(key, max_age_hours=max_age_hours)
            if not df.empty:
                latest[key] = df.iloc[-1]["value"]
        except Exception as e:
            log.warning("Failed to get latest %s: %s", key, e)
    return latest


def merge_onchain_daily(
    df: pd.DataFrame,
    onchain: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge on-chain daily data into an hourly OHLCV DataFrame.

    On-chain data is daily; we forward-fill to align with hourly candles.
    """
    df = df.copy()
    df["_date"] = df["timestamp"].dt.normalize()

    for key, oc_df in onchain.items():
        if oc_df.empty:
            continue
        oc = oc_df.rename(columns={"date": "_date", "value": key})
        oc["_date"] = oc["_date"].dt.normalize()
        df = pd.merge(df, oc[["_date", key]], on="_date", how="left")
        df[key] = df[key].ffill().fillna(0)

    df = df.drop(columns=["_date"])
    return df
