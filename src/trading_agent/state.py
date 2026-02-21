"""Thread-safe shared state between the three layers."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from trading_agent.portfolio import Portfolio
from trading_agent.regime import RegimeState


@dataclass
class SharedState:
    portfolio: Portfolio = field(default_factory=Portfolio.load)
    regimes: dict[str, RegimeState] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __enter__(self) -> SharedState:
        self._lock.acquire()
        return self

    def __exit__(self, *args) -> None:
        self._lock.release()
