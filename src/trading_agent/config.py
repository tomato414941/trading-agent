"""Centralized configuration for risk management and trading parameters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskConfig:
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 15.0
    trailing_stop_pct: float = 0.0  # 0 = disabled; X = sell when price drops X% from peak
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


@dataclass
class ArbitrageConfig:
    entry_fr_threshold: float = 0.0003  # 0.03%/8h: open when FR exceeds this
    exit_fr_threshold: float = 0.0001   # 0.01%/8h: consider closing below this
    exit_consecutive_periods: int = 3   # close after N consecutive low-FR periods
    spot_fee_rate: float = 0.001        # 0.1% per spot trade
    futures_fee_rate: float = 0.0005    # 0.05% per futures trade
    position_fraction: float = 0.3      # fraction of capital per position
    futures_leverage: float = 1.0       # 1x = fully collateralized (no liquidation risk)
    # Live execution settings
    testnet: bool = True               # use Binance testnet
    check_interval_sec: int = 300      # check every 5 minutes
    symbol: str = "BTC/USDT"           # target symbol


DEFAULT_RISK = RiskConfig()
DEFAULT_AGENT = AgentConfig()
DEFAULT_ARBITRAGE = ArbitrageConfig()
