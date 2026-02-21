"""Centralized configuration for risk management and trading parameters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskConfig:
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 15.0
    max_exposure_pct: float = 60.0
    buy_fraction: float = 0.1


@dataclass
class AgentConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    signal_timeframe: str = "1h"
    regime_timeframe: str = "4h"
    monitor_interval_sec: int = 120
    signal_interval_sec: int = 3600
    regime_interval_sec: int = 14400
    regime_candle_limit: int = 250
    signal_candle_limit: int = 100


DEFAULT_RISK = RiskConfig()
DEFAULT_AGENT = AgentConfig()
