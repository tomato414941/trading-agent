"""Performance tracking with graduation criteria for paper -> live."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

METRICS_DIR = Path("data/metrics")
TRADES_LOG = Path("data/shadow_trades.csv")


@dataclass
class DailyMetrics:
    date: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    num_trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    equity: float = 0.0
    model_accuracy: float = 0.0
    model_samples: int = 0
    avg_confidence: float = 0.0


CAPITAL_TIERS = [
    {"min_days": 30, "max_capital": 0, "mode": "shadow"},
    {"min_days": 14, "max_capital": 100, "mode": "live"},
    {"min_days": 14, "max_capital": 500, "mode": "live"},
    {"min_days": 14, "max_capital": 2000, "mode": "live"},
    {"min_days": 30, "max_capital": 10000, "mode": "live"},
]


class PerformanceTracker:
    """Track daily performance and determine when to graduate tiers."""

    def __init__(self):
        self._daily_metrics: list[DailyMetrics] = []
        self._current_day: DailyMetrics | None = None
        self._peak_equity: float = 0.0
        self._current_tier: int = 0

    def start_day(self, equity: float) -> None:
        today = date.today().isoformat()
        if self._current_day and self._current_day.date == today:
            return
        if self._current_day:
            self._daily_metrics.append(self._current_day)
        self._current_day = DailyMetrics(date=today, equity=equity)
        if equity > self._peak_equity:
            self._peak_equity = equity

    def record_trade(self, pnl: float, confidence: float = 0.0) -> None:
        if not self._current_day:
            return
        self._current_day.num_trades += 1
        self._current_day.pnl += pnl
        if pnl > 0:
            self._current_day.wins += 1
        if self._current_day.num_trades > 0:
            self._current_day.win_rate = (
                self._current_day.wins / self._current_day.num_trades * 100
            )
        self._current_day.avg_confidence = (
            (self._current_day.avg_confidence * (self._current_day.num_trades - 1) + confidence)
            / self._current_day.num_trades
        )

    def update_model_stats(self, accuracy: float, samples: int) -> None:
        if self._current_day:
            self._current_day.model_accuracy = accuracy
            self._current_day.model_samples = samples

    def end_day(self, equity: float) -> DailyMetrics | None:
        if not self._current_day:
            return None
        if self._current_day.equity > 0:
            self._current_day.pnl_pct = (
                self._current_day.pnl / self._current_day.equity * 100
            )
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity * 100
            self._current_day.max_drawdown_pct = dd

        self._daily_metrics.append(self._current_day)
        result = self._current_day
        self._current_day = None
        return result

    def should_graduate(self) -> tuple[bool, str]:
        """Check if current tier's graduation criteria are met."""
        tier = CAPITAL_TIERS[min(self._current_tier, len(CAPITAL_TIERS) - 1)]
        n = len(self._daily_metrics)

        if n < tier["min_days"]:
            return False, f"Need {tier['min_days']} days, have {n}"

        window = self._daily_metrics[-tier["min_days"]:]

        total_pnl = sum(d.pnl for d in window)
        if total_pnl <= 0:
            return False, f"PnL negative: ${total_pnl:.2f}"

        total_trades = sum(d.num_trades for d in window)
        if total_trades < 20:
            return False, f"Only {total_trades} trades (need 20+)"

        total_wins = sum(d.wins for d in window)
        win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
        if win_rate < 50:
            return False, f"Win rate {win_rate:.1f}% < 50%"

        # Rolling Sharpe
        daily_returns = [d.pnl_pct for d in window if d.pnl_pct != 0]
        if len(daily_returns) >= 5:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) > 0 else 0
            if sharpe < 0.5:
                return False, f"Sharpe {sharpe:.2f} < 0.5"

        max_dd = max(d.max_drawdown_pct for d in window)
        if max_dd > 5:
            return False, f"Max DD {max_dd:.1f}% > 5%"

        return True, "All criteria met"

    def graduate(self) -> dict | None:
        can, reason = self.should_graduate()
        if not can:
            log.info("Cannot graduate: %s", reason)
            return None
        self._current_tier += 1
        if self._current_tier >= len(CAPITAL_TIERS):
            self._current_tier = len(CAPITAL_TIERS) - 1
        tier = CAPITAL_TIERS[self._current_tier]
        log.info("Graduated to tier %d: %s", self._current_tier, tier)
        return tier

    @property
    def current_tier(self) -> dict:
        return CAPITAL_TIERS[min(self._current_tier, len(CAPITAL_TIERS) - 1)]

    @property
    def days_tracked(self) -> int:
        return len(self._daily_metrics)

    def save(self, path: Path | None = None) -> None:
        path = path or (METRICS_DIR / "performance.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current_tier": self._current_tier,
            "peak_equity": self._peak_equity,
            "days": [asdict(d) for d in self._daily_metrics],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path | None = None) -> PerformanceTracker:
        path = path or (METRICS_DIR / "performance.json")
        tracker = cls()
        if not path.exists():
            return tracker
        data = json.loads(path.read_text())
        tracker._current_tier = data.get("current_tier", 0)
        tracker._peak_equity = data.get("peak_equity", 0)
        tracker._daily_metrics = [
            DailyMetrics(**d) for d in data.get("days", [])
        ]
        return tracker

    def summary(self) -> str:
        n = len(self._daily_metrics)
        tier = self.current_tier
        lines = [
            f"=== Performance Summary ===",
            f"Days tracked: {n}",
            f"Current tier: {tier['mode']} (max ${tier['max_capital']})",
        ]
        if n > 0:
            total_pnl = sum(d.pnl for d in self._daily_metrics)
            total_trades = sum(d.num_trades for d in self._daily_metrics)
            total_wins = sum(d.wins for d in self._daily_metrics)
            wr = total_wins / total_trades * 100 if total_trades > 0 else 0
            lines.append(f"Total PnL: ${total_pnl:.2f}")
            lines.append(f"Total trades: {total_trades} (win rate: {wr:.1f}%)")
            can, reason = self.should_graduate()
            lines.append(f"Graduate ready: {'YES' if can else 'NO'} ({reason})")
        return "\n".join(lines)
