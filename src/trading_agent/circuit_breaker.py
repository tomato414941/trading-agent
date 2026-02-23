"""Circuit breaker: kill switch, daily loss limit, drawdown halt."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)


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

    def record_trade(self, pnl: float) -> None:
        self._daily_pnl += pnl
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

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
                return False, self._halt_reason

        # Consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self._halted = True
            self._halt_reason = f"Consecutive losses: {self._consecutive_losses}"
            log.error("CIRCUIT BREAKER: %s", self._halt_reason)
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
                return False, self._halt_reason

        return True, "ok"

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason
