"""Centralized configuration for risk management and trading parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskConfig:
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 15.0
    max_exposure_pct: float = 60.0
    buy_fraction: float = 0.1


DEFAULT_RISK = RiskConfig()
