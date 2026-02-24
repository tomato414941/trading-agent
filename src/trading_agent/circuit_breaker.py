"""Circuit breaker: kill switch, daily loss limit, drawdown halt."""

from __future__ import annotations

import json
import logging

from trading_agent._util import atomic_write_text
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

CB_STATE_PATH = Path("data/metrics/circuit_breaker.json")


@dataclass
class CircuitBreakerConfig:
    daily_loss_limit_pct: float = 3.0
    max_consecutive_losses: int = 5
    max_drawdown_pct: float = 10.0
    kill_file: str = "data/KILL_SWITCH"


@dataclass
class CircuitBreaker:
    """Centralized risk circuit breaker. Checked before every trade."""

    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    _daily_start_equity: float = 0.0
    _daily_pnl: float = 0.0
    _consecutive_losses: int = 0
    _peak_equity: float = 0.0
    _halted: bool = False
    _halt_reason: str = ""
    _current_date: str = ""

    def reset_daily(self, equity: float) -> None:
        today = date.today().isoformat()
        if today != self._current_date:
            self._current_date = today
            self._daily_start_equity = equity
            self._daily_pnl = 0.0
            self._halted = False
            self._halt_reason = ""
            log.info("Daily reset: equity=$%.2f", equity)
            self.save()

    def record_trade(self, pnl: float) -> None:
        self._daily_pnl += pnl
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self.save()

    def is_safe_to_trade(self, current_equity: float) -> tuple[bool, str]:
        # Kill switch file
        if Path(self.config.kill_file).exists():
            return False, "Kill switch file detected"

        if self._halted:
            return False, f"Halted: {self._halt_reason}"

        # Daily loss limit
        if self._daily_start_equity > 0:
            daily_loss_pct = -self._daily_pnl / self._daily_start_equity * 100
            if daily_loss_pct > self.config.daily_loss_limit_pct:
                self._halted = True
                self._halt_reason = (
                    f"Daily loss {daily_loss_pct:.1f}% > "
                    f"limit {self.config.daily_loss_limit_pct}%"
                )
                log.error("CIRCUIT BREAKER: %s", self._halt_reason)
                self.save()
                return False, self._halt_reason

        # Consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self._halted = True
            self._halt_reason = f"Consecutive losses: {self._consecutive_losses}"
            log.error("CIRCUIT BREAKER: %s", self._halt_reason)
            self.save()
            return False, self._halt_reason

        # Max drawdown from peak
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        if self._peak_equity > 0:
            dd_pct = (self._peak_equity - current_equity) / self._peak_equity * 100
            if dd_pct > self.config.max_drawdown_pct:
                self._halted = True
                self._halt_reason = (
                    f"Drawdown {dd_pct:.1f}% > limit {self.config.max_drawdown_pct}%"
                )
                log.error("CIRCUIT BREAKER: %s", self._halt_reason)
                self.save()
                return False, self._halt_reason

        return True, "ok"

    def save(self, path: Path | None = None) -> None:
        path = path or CB_STATE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "daily_start_equity": self._daily_start_equity,
            "daily_pnl": self._daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "peak_equity": self._peak_equity,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "current_date": self._current_date,
        }
        atomic_write_text(path, json.dumps(data, indent=2))

    @classmethod
    def load(
        cls,
        config: CircuitBreakerConfig | None = None,
        path: Path | None = None,
    ) -> CircuitBreaker:
        config = config or CircuitBreakerConfig()
        path = path or CB_STATE_PATH
        cb = cls(config=config)
        if not path.exists():
            return cb
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load circuit breaker state: %s", e)
            return cb
        cb._daily_start_equity = data.get("daily_start_equity", 0.0)
        cb._daily_pnl = data.get("daily_pnl", 0.0)
        cb._consecutive_losses = data.get("consecutive_losses", 0)
        cb._peak_equity = data.get("peak_equity", 0.0)
        cb._halted = data.get("halted", False)
        cb._halt_reason = data.get("halt_reason", "")
        cb._current_date = data.get("current_date", "")
        log.info(
            "Circuit breaker loaded: pnl=$%.2f, peak=$%.2f, halted=%s",
            cb._daily_pnl, cb._peak_equity, cb._halted,
        )
        return cb

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason
