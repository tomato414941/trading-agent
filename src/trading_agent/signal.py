"""Layer 2: Signal generation and trade execution."""

from __future__ import annotations

import logging

from trading_agent.config import RiskConfig, AgentConfig
from trading_agent.fetcher import fetch_ohlcv
from trading_agent.strategy import (
    compute_indicators,
    composite_signal,
    sentiment_multiplier,
    SignalFilter,
    DEFAULT_BUY_COOLDOWN,
)
from trading_agent.sentiment import analyze_sentiment
from trading_agent.portfolio import log_trade
from trading_agent.notify import notify_trade
from trading_agent.regime import regime_allows_buy
from trading_agent.state import SharedState

log = logging.getLogger(__name__)


def tick_symbol(
    symbol: str,
    state: SharedState,
    sentiment_score: float,
    prices: dict[str, float],
    config: AgentConfig,
    risk: RiskConfig,
) -> None:
    log.info("[SIGNAL] [%s] Fetching %s candles...", symbol, config.signal_timeframe)
    df = fetch_ohlcv(symbol, config.signal_timeframe, limit=config.signal_candle_limit)
    df = compute_indicators(df)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest["close"]
    rsi = latest["rsi"]

    with state:
        portfolio = state.portfolio
        pos = portfolio.get_position(symbol)

        # SL/TP fallback (monitor should catch most, but check here too)
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
                    "[SIGNAL] [%s] %s triggered @ $%.2f (PnL: %+.1f%%)",
                    symbol, sl_action.upper().replace("_", " "), price, pnl_pct,
                )
                portfolio.save()
                log_trade(trade, sl_action, rsi)
                notify_trade(trade, sl_action, rsi)
            return

        # Generate signal
        raw_signal = composite_signal(rsi, latest["macd_diff"], prev["macd_diff"])

        multiplier = sentiment_multiplier(sentiment_score)
        adjusted_fraction = risk.buy_fraction * multiplier

        sig_filter = SignalFilter(buy_cooldown=DEFAULT_BUY_COOLDOWN)
        sig_filter.ticks_since_buy = pos.ticks_since_buy
        signal = sig_filter.filter(raw_signal)
        pos.ticks_since_buy = sig_filter.ticks_since_buy
        portfolio._save_pos(symbol, pos)

        # Regime filter: block buys in downtrend
        regime_state = state.regimes.get(symbol)
        regime_tag = regime_state.regime if regime_state else "unknown"

        if signal == "buy" and not regime_allows_buy(regime_state):
            log.info(
                "[SIGNAL] [%s] Buy blocked by regime filter (%s)",
                symbol, regime_tag,
            )
            signal = "hold"

        log.info(
            "[SIGNAL] [%s] $%.2f | RSI: %.1f | Sent: %+.2f (x%.2f) | "
            "Raw: %s | Signal: %s | Regime: %s | Pos: %.6f",
            symbol, price, rsi, sentiment_score, multiplier,
            raw_signal.upper(), signal.upper(), regime_tag, pos.qty,
        )

        trade = None
        if signal == "buy":
            if portfolio.can_buy(prices, risk):
                trade = portfolio.buy(symbol, price, fraction=adjusted_fraction)
            else:
                log.info("[SIGNAL] [%s] Max exposure reached, skipping buy", symbol)
        elif signal == "sell":
            trade = portfolio.sell(symbol, price)

        if trade:
            log.info(
                "[SIGNAL] [%s] TRADE: %s %.6f @ $%.2f",
                symbol, trade["side"].upper(), trade["qty"], price,
            )
            portfolio.save()
            log_trade(trade, signal, rsi)
            notify_trade(trade, signal, rsi)


def tick(
    state: SharedState,
    config: AgentConfig,
    risk: RiskConfig,
) -> None:
    """One full signal cycle across all symbols."""
    prices: dict[str, float] = {}
    for symbol in config.symbols:
        try:
            df = fetch_ohlcv(symbol, config.signal_timeframe, limit=2)
            prices[symbol] = df.iloc[-1]["close"]
        except Exception:
            pass

    with state:
        state.last_prices.update(prices)

    for symbol in config.symbols:
        try:
            log.info("[SIGNAL] [%s] Analyzing sentiment...", symbol)
            sent = analyze_sentiment(symbol=symbol)
            score = sent["score"]
            log.info("[SIGNAL] [%s] Sentiment: %+.2f (%s)",
                     symbol, score, sent["summary"][:60])
        except Exception:
            log.exception("[SIGNAL] [%s] Sentiment failed, using neutral", symbol)
            score = 0.0

        try:
            tick_symbol(symbol, state, score, prices, config, risk)
        except Exception:
            log.exception("[SIGNAL] [%s] Tick failed", symbol)

    with state:
        total = state.portfolio.total_value(prices)
        log.info("Portfolio: Cash $%.2f | Total $%.2f",
                 state.portfolio.cash, total)
        state.portfolio.save()
