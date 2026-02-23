"""Position reconciliation: compare internal state with exchange."""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class Discrepancy:
    symbol: str
    internal_qty: float
    exchange_qty: float
    diff_pct: float


def reconcile(
    internal: dict[str, float],
    exchange: dict[str, float],
    tolerance_pct: float = 1.0,
) -> list[Discrepancy]:
    """Compare internal positions with exchange positions.

    Returns list of symbols where quantities differ by more than tolerance_pct.
    """
    discrepancies: list[Discrepancy] = []
    all_symbols = set(internal) | set(exchange)

    for symbol in all_symbols:
        int_qty = internal.get(symbol, 0.0)
        ext_qty = exchange.get(symbol, 0.0)

        if abs(int_qty) < 1e-8 and abs(ext_qty) < 1e-8:
            continue

        ref = max(abs(int_qty), abs(ext_qty))
        diff_pct = abs(int_qty - ext_qty) / ref * 100 if ref > 0 else 0

        if diff_pct > tolerance_pct:
            discrepancies.append(Discrepancy(
                symbol=symbol,
                internal_qty=int_qty,
                exchange_qty=ext_qty,
                diff_pct=diff_pct,
            ))
            log.warning(
                "Position mismatch %s: internal=%.6f exchange=%.6f (%.1f%%)",
                symbol, int_qty, ext_qty, diff_pct,
            )

    return discrepancies


def fetch_exchange_positions(exchange) -> dict[str, float]:
    """Fetch current positions from exchange via CCXT."""
    positions: dict[str, float] = {}
    try:
        balance = exchange.fetch_balance()
        for currency, info in balance.items():
            if isinstance(info, dict) and info.get("total", 0) > 0:
                if currency not in ("USDT", "USD", "free", "used", "total", "info"):
                    positions[currency] = float(info["total"])
    except Exception as e:
        log.error("Failed to fetch exchange positions: %s", e)

    return positions
