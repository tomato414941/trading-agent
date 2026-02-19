"""Virtual portfolio tracker — no real money involved."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
STATE_FILE = DATA_DIR / "portfolio.json"
TRADE_LOG = DATA_DIR / "trades.csv"


@dataclass
class Portfolio:
    cash: float = 10_000.0
    position: float = 0.0  # BTC quantity
    entry_price: float = 0.0

    def buy(self, price: float, fraction: float = 0.1) -> dict | None:
        amount_usd = self.cash * fraction
        if amount_usd < 1.0:
            return None
        qty = amount_usd / price
        self.cash -= amount_usd
        self.position += qty
        self.entry_price = price
        return {"side": "buy", "price": price, "qty": qty, "cost": amount_usd}

    def sell(self, price: float) -> dict | None:
        if self.position <= 0:
            return None
        revenue = self.position * price
        pnl = revenue - (self.position * self.entry_price)
        trade = {
            "side": "sell",
            "price": price,
            "qty": self.position,
            "revenue": revenue,
            "pnl": pnl,
        }
        self.cash += revenue
        self.position = 0.0
        self.entry_price = 0.0
        return trade

    @property
    def total_value(self) -> float:
        return self.cash  # position value added at runtime with current price

    def save(self) -> None:
        DATA_DIR.mkdir(exist_ok=True)
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> Portfolio:
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            return cls(**data)
        return cls()


def log_trade(trade: dict, signal: str, rsi: float) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    file_exists = TRADE_LOG.exists()
    fieldnames = ["timestamp", "signal", "rsi", "side", "price", "qty", "pnl"]

    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "signal": signal,
            "rsi": round(rsi, 2),
            "side": trade.get("side", ""),
            "price": round(trade.get("price", 0), 2),
            "qty": round(trade.get("qty", 0), 6),
            "pnl": round(trade.get("pnl", 0), 2) if "pnl" in trade else "",
        })
