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
    # On-chain
    "mempool_fee",
    "mempool_size",
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

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._cache: dict[str, float] = {}

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
                    self._cache[name] = values[name]
            except requests.HTTPError:
                log.debug("Signal %s not available", name)
            except Exception as e:
                log.debug("Failed to fetch %s: %s", name, e)
                # Use cached value if available
                if name in self._cache:
                    values[name] = self._cache[name]

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
