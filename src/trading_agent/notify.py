"""Discord webhook notifications for trades and agent status."""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# Rate limit: max 5 messages per 2 seconds
_RATE_WINDOW = 2.0
_RATE_LIMIT = 5
_timestamps: deque[float] = deque(maxlen=_RATE_LIMIT)

COLORS = {
    "buy": 0x2ECC71,       # green
    "sell": 0xE74C3C,      # red
    "stop_loss": 0xE67E22,  # orange
    "take_profit": 0xF1C40F,  # yellow
    "info": 0x3498DB,      # blue
    "error": 0xE74C3C,     # red
}


def _get_webhook_url() -> str | None:
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if url:
        return url
    secret_path = Path.home() / ".secrets" / "discord"
    if secret_path.exists():
        text = secret_path.read_text().strip()
        # Handle "export VAR=value" or plain value
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if line.startswith("DISCORD_WEBHOOK_URL="):
                return line.split("=", 1)[1].strip("'\"")
        return text if text.startswith("https://") else None
    return None


def _wait_rate_limit() -> None:
    now = time.time()
    if len(_timestamps) >= _RATE_LIMIT:
        elapsed = now - _timestamps[0]
        if elapsed < _RATE_WINDOW:
            time.sleep(_RATE_WINDOW - elapsed)
    _timestamps.append(time.time())


def send(
    title: str,
    description: str,
    color_key: str = "info",
    fields: list[dict] | None = None,
) -> None:
    url = _get_webhook_url()
    if not url:
        return

    _wait_rate_limit()

    embed = {
        "title": title,
        "description": description,
        "color": COLORS.get(color_key, COLORS["info"]),
    }
    if fields:
        embed["fields"] = fields

    try:
        resp = requests.post(
            url,
            json={"embeds": [embed]},
            timeout=5,
        )
        resp.raise_for_status()
    except Exception:
        log.debug("Discord notification failed", exc_info=True)


def notify_trade(trade: dict, reason: str, rsi: float = 0.0) -> None:
    side = trade.get("side", "unknown")
    symbol = trade.get("symbol", "???")
    price = trade.get("price", 0)
    qty = trade.get("qty", 0)

    color_key = reason if reason in COLORS else side
    title = f"{'BUY' if side == 'buy' else 'SELL'} {symbol}"

    fields = [
        {"name": "Price", "value": f"${price:,.2f}", "inline": True},
        {"name": "Qty", "value": f"{qty:.6f}", "inline": True},
    ]

    if reason in ("stop_loss", "take_profit"):
        title = f"{reason.upper().replace('_', ' ')} {symbol}"
        pnl = trade.get("pnl", 0)
        if pnl:
            fields.append({"name": "PnL", "value": f"${pnl:+,.2f}", "inline": True})

    if rsi > 0:
        fields.append({"name": "RSI", "value": f"{rsi:.1f}", "inline": True})

    send(title, f"Reason: {reason}", color_key=color_key, fields=fields)


def notify_agent_status(status: str, details: str = "") -> None:
    color = "info" if status == "started" else "error"
    send(
        f"Agent {status.upper()}",
        details or f"Trading agent {status}",
        color_key=color,
    )
