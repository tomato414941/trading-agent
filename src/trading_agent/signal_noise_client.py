"""Client for signal-noise REST API — fetches crypto-relevant signals."""

from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:8000"

# Crypto-relevant signals from signal-noise service
CRYPTO_SIGNALS = [
    # Sentiment / positioning
    "fear_greed",
    "binance_btc_oi",
    "deribit_btc_skew",
    "bitfinex_btc_ls_ratio",
    "cg_btc_dominance",
    # On-chain
    "mempool_fee",
    "bc_tx_fees_usd",
    "btc_active_addresses",
    # Macro
    "dxy",
    "gold",
    "sp500",
    "vix_close",
    "tsy_yield_10y",
    "tsy_yield_2y",
]


class SignalNoiseClient:
    """Fetch crypto-relevant signals from signal-noise service."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 10.0, cache_ttl: float = 3600.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._cache: dict[str, tuple[float, float]] = {}  # name -> (value, timestamp)
        self._cache_ttl = cache_ttl

    def health(self) -> bool:
        try:
            r = self._session.get(
                f"{self._base_url}/health", timeout=self._timeout
            )
            return r.ok and r.json().get("status") == "ok"
        except Exception:
            return False

    def get_latest_values(
        self, names: list[str] | None = None
    ) -> dict[str, float]:
        names = names or CRYPTO_SIGNALS
        values: dict[str, float] = {}

        for name in names:
            try:
                r = self._session.get(
                    f"{self._base_url}/signals/{name}/latest",
                    timeout=self._timeout,
                )
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                data = r.json()
                if data and data.get("value") is not None:
                    values[name] = float(data["value"])
                    self._cache[name] = (values[name], time.time())
            except requests.HTTPError:
                log.debug("Signal %s not available", name)
            except Exception as e:
                log.debug("Failed to fetch %s: %s", name, e)
                if name in self._cache:
                    val, ts = self._cache[name]
                    age = time.time() - ts
                    if age < self._cache_ttl:
                        values[name] = val
                        log.debug("Using cached %s (age=%.0fs)", name, age)
                    else:
                        log.warning("Cache expired for %s (age=%.0fs)", name, age)

        return values

    def get_data(self, name: str, since: str | None = None) -> list[dict]:
        params = {}
        if since:
            params["since"] = since
        try:
            r = self._session.get(
                f"{self._base_url}/signals/{name}/data",
                params=params,
                timeout=self._timeout,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("Failed to fetch data for %s: %s", name, e)
            return []
