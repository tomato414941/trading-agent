"""Virtual portfolio tracker — no real money involved. Supports multiple symbols."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from trading_agent.config import RiskConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
STATE_FILE = DATA_DIR / "portfolio.json"
TRADE_LOG = DATA_DIR / "trades.csv"


@dataclass
class Position:
    qty: float = 0.0
    entry_price: float = 0.0
    ticks_since_buy: int = 999


@dataclass
class Portfolio:
    cash: float = 10_000.0
    positions: dict[str, dict] = field(default_factory=dict)

    def _pos(self, symbol: str) -> Position:
        raw = self.positions.get(symbol, {})
        return Position(**raw) if raw else Position()

    def _save_pos(self, symbol: str, pos: Position) -> None:
        self.positions[symbol] = asdict(pos)

    def get_position(self, symbol: str) -> Position:
        return self._pos(symbol)

    def buy(self, symbol: str, price: float, fraction: float = 0.1) -> dict | None:
        amount_usd = self.cash * fraction
        if amount_usd < 1.0:
            return None
        qty = amount_usd / price
        self.cash -= amount_usd
        pos = self._pos(symbol)
        # Weighted average cost basis
        total_cost = pos.qty * pos.entry_price + qty * price
        pos.qty += qty
        pos.entry_price = total_cost / pos.qty if pos.qty > 0 else price
        self._save_pos(symbol, pos)
        return {"side": "buy", "symbol": symbol, "price": price, "qty": qty, "cost": amount_usd}

    def sell(self, symbol: str, price: float) -> dict | None:
        pos = self._pos(symbol)
        if pos.qty <= 0:
            return None
        revenue = pos.qty * price
        pnl = revenue - (pos.qty * pos.entry_price)
        trade = {
            "side": "sell",
            "symbol": symbol,
            "price": price,
            "qty": pos.qty,
            "revenue": revenue,
            "pnl": pnl,
        }
        self.cash += revenue
        pos.qty = 0.0
        pos.entry_price = 0.0
        self._save_pos(symbol, pos)
        return trade

    def check_stop_loss(
        self, symbol: str, price: float, config: RiskConfig,
    ) -> str | None:
        """Return 'stop_loss' or 'take_profit' if threshold breached, else None."""
        pos = self._pos(symbol)
        if pos.qty <= 0 or pos.entry_price <= 0:
            return None
        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
        if pnl_pct <= -config.stop_loss_pct:
            return "stop_loss"
        if pnl_pct >= config.take_profit_pct:
            return "take_profit"
        return None

    def can_buy(self, prices: dict[str, float], config: RiskConfig) -> bool:
        """Return False if total position exposure exceeds max_exposure_pct."""
        total = self.total_value(prices)
        if total <= 0:
            return True
        position_value = sum(
            Position(**raw).qty * prices.get(sym, 0)
            for sym, raw in self.positions.items()
            if raw
        )
        exposure_pct = (position_value / total) * 100
        return exposure_pct < config.max_exposure_pct

    def total_value(self, prices: dict[str, float]) -> float:
        value = self.cash
        for symbol, raw in self.positions.items():
            pos = Position(**raw) if raw else Position()
            value += pos.qty * prices.get(symbol, 0)
        return value

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
    fieldnames = ["timestamp", "symbol", "signal", "rsi", "side", "price", "qty", "pnl"]

    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "symbol": trade.get("symbol", ""),
            "signal": signal,
            "rsi": round(rsi, 2),
            "side": trade.get("side", ""),
            "price": round(trade.get("price", 0), 2),
            "qty": round(trade.get("qty", 0), 6),
            "pnl": round(trade.get("pnl", 0), 2) if "pnl" in trade else "",
        })
