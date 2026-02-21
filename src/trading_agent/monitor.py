"""Layer 1: Risk Monitor — lightweight, fast SL/TP checks every 1-5 minutes."""

from __future__ import annotations

import logging
import threading

from trading_agent.config import RiskConfig, AgentConfig
from trading_agent.fetcher import fetch_ticker_price
from trading_agent.portfolio import log_trade
from trading_agent.notify import notify_trade
from trading_agent.state import SharedState

log = logging.getLogger(__name__)


def check_risk_all_symbols(
    state: SharedState,
    symbols: list[str],
    risk: RiskConfig,
) -> None:
    """Fetch current prices and check SL/TP for all symbols."""
    for symbol in symbols:
        try:
            price = fetch_ticker_price(symbol)
        except Exception:
            log.debug("[MONITOR] [%s] Price fetch failed", symbol)
            continue

        trade = None
        sl_action = None

        with state:
            state.last_prices[symbol] = price
            portfolio = state.portfolio
            pos = portfolio.get_position(symbol)
            if pos.qty <= 0:
                continue

            sl_action = portfolio.check_stop_loss(symbol, price, risk)
            if sl_action:
                trade = portfolio.sell(symbol, price)
                if trade:
                    trade["reason"] = sl_action
                    pnl_pct = (
                        (price - pos.entry_price) / pos.entry_price * 100
                        if pos.entry_price > 0 else 0
                    )
                    log.warning(
                        "[MONITOR] [%s] %s triggered @ $%.2f (PnL: %+.1f%%)",
                        symbol, sl_action.upper().replace("_", " "),
                        price, pnl_pct,
                    )
                    portfolio.save()

        # Notify outside the lock
        if trade and sl_action:
            log_trade(trade, sl_action, 0.0)
            notify_trade(trade, sl_action, 0.0)


def run_monitor(
    state: SharedState,
    config: AgentConfig,
    risk: RiskConfig,
    stop_event: threading.Event,
) -> None:
    """Monitor loop — runs in a daemon thread."""
    log.info("[MONITOR] Started (interval=%ds)", config.monitor_interval_sec)
    while not stop_event.is_set():
        try:
            check_risk_all_symbols(state, config.symbols, risk)
        except Exception:
            log.exception("[MONITOR] Error in risk check")
        stop_event.wait(config.monitor_interval_sec)
    log.info("[MONITOR] Stopped")
