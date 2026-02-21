"""Authenticated ccxt exchange factory for Binance spot and futures."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import ccxt

log = logging.getLogger(__name__)

_SECRETS_FILE = "binance"


def load_secrets(name: str = _SECRETS_FILE) -> dict[str, str]:
    """Load key-value pairs from ~/.secrets/{name}.

    Supports formats:
      export KEY=value
      KEY=value
    """
    secret_path = Path.home() / ".secrets" / name
    if not secret_path.exists():
        return {}
    result: dict[str, str] = {}
    for line in secret_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            key, val = line.split("=", 1)
            result[key.strip()] = val.strip().strip("'\"")
    return result


def _get_credentials() -> tuple[str, str]:
    """Return (api_key, secret) from env vars or secrets file."""
    api_key = os.environ.get("BINANCE_API_KEY", "")
    secret = os.environ.get("BINANCE_SECRET_KEY", "")
    if api_key and secret:
        return api_key, secret

    secrets = load_secrets()
    api_key = api_key or secrets.get("BINANCE_API_KEY", "")
    secret = secret or secrets.get("BINANCE_SECRET_KEY", "")
    return api_key, secret


def create_futures_exchange(testnet: bool = True) -> ccxt.binance:
    """Create authenticated Binance futures exchange."""
    api_key, secret = _get_credentials()
    if not api_key or not secret:
        raise ValueError(
            "Binance API credentials not found. "
            "Set BINANCE_API_KEY/BINANCE_SECRET_KEY env vars "
            "or create ~/.secrets/binance"
        )

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    if testnet:
        exchange.set_sandbox_mode(True)

    log.info("Created futures exchange (testnet=%s)", testnet)
    return exchange


def create_spot_exchange(testnet: bool = True) -> ccxt.binance:
    """Create authenticated Binance spot exchange."""
    api_key, secret = _get_credentials()
    if not api_key or not secret:
        raise ValueError(
            "Binance API credentials not found. "
            "Set BINANCE_API_KEY/BINANCE_SECRET_KEY env vars "
            "or create ~/.secrets/binance"
        )

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    if testnet:
        exchange.set_sandbox_mode(True)

    log.info("Created spot exchange (testnet=%s)", testnet)
    return exchange
