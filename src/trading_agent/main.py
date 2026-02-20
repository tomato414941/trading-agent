"""Main loop — monitor multiple symbols, analyze sentiment, trade."""

import time
import logging

from trading_agent.config import DEFAULT_RISK
from trading_agent.fetcher import fetch_ohlcv
from trading_agent.strategy import (
    compute_indicators,
    composite_signal,
    sentiment_multiplier,
    SignalFilter,
    DEFAULT_BUY_COOLDOWN,
)
from trading_agent.sentiment import analyze_sentiment
from trading_agent.portfolio import Portfolio, log_trade
from trading_agent.notify import notify_trade, notify_agent_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "1h"
INTERVAL_SEC = 60 * 60


def tick_symbol(
    symbol: str,
    portfolio: Portfolio,
    sentiment_score: float,
    prices: dict[str, float],
) -> None:
    log.info("[%s] Fetching %s candles...", symbol, TIMEFRAME)
    df = fetch_ohlcv(symbol, TIMEFRAME)
    df = compute_indicators(df)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest["close"]
    rsi = latest["rsi"]

    pos = portfolio.get_position(symbol)

    # Stop-loss / take-profit check (before signal generation)
    sl_action = portfolio.check_stop_loss(symbol, price, DEFAULT_RISK)
    if sl_action:
        trade = portfolio.sell(symbol, price)
        if trade:
            trade["reason"] = sl_action
            pnl_pct = (price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
            log.warning(
                "[%s] %s triggered @ $%.2f (PnL: %+.1f%%)",
                symbol, sl_action.upper().replace("_", " "), price, pnl_pct,
            )
            log_trade(trade, sl_action, rsi)
            notify_trade(trade, sl_action, rsi)
        return

    # Generate raw signal (technicals only)
    raw_signal = composite_signal(rsi, latest["macd_diff"], prev["macd_diff"])

    # Sentiment adjusts position size, not direction
    multiplier = sentiment_multiplier(sentiment_score)
    adjusted_fraction = DEFAULT_RISK.buy_fraction * multiplier

    # Apply cooldown filter (per-symbol state)
    sig_filter = SignalFilter(buy_cooldown=DEFAULT_BUY_COOLDOWN)
    sig_filter.ticks_since_buy = pos.ticks_since_buy
    signal = sig_filter.filter(raw_signal)
    pos.ticks_since_buy = sig_filter.ticks_since_buy
    portfolio._save_pos(symbol, pos)

    log.info(
        "[%s] $%.2f | RSI: %.1f | Sent: %+.2f (x%.2f) | Raw: %s | Signal: %s | Pos: %.6f",
        symbol, price, rsi, sentiment_score, multiplier,
        raw_signal.upper(), signal.upper(), pos.qty,
    )

    trade = None
    if signal == "buy":
        if portfolio.can_buy(prices, DEFAULT_RISK):
            trade = portfolio.buy(symbol, price, fraction=adjusted_fraction)
        else:
            log.info("[%s] Max exposure reached, skipping buy", symbol)
    elif signal == "sell":
        trade = portfolio.sell(symbol, price)

    if trade:
        log.info(
            "[%s] TRADE: %s %.6f @ $%.2f",
            symbol, trade["side"].upper(), trade["qty"], price,
        )
        log_trade(trade, signal, rsi)
        notify_trade(trade, signal, rsi)


def tick() -> None:
    portfolio = Portfolio.load()

    # Fetch latest prices for all symbols (used for exposure check)
    prices: dict[str, float] = {}
    for symbol in SYMBOLS:
        try:
            df = fetch_ohlcv(symbol, TIMEFRAME, limit=2)
            prices[symbol] = df.iloc[-1]["close"]
        except Exception:
            pass

    # Process each symbol
    for symbol in SYMBOLS:
        try:
            log.info("[%s] Analyzing sentiment...", symbol)
            sent = analyze_sentiment(symbol=symbol)
            score = sent["score"]
            log.info("[%s] Sentiment: %+.2f (%s)", symbol, score, sent["summary"][:60])
        except Exception:
            log.exception("[%s] Sentiment failed, using neutral", symbol)
            score = 0.0

        try:
            tick_symbol(symbol, portfolio, score, prices)
        except Exception:
            log.exception("[%s] Tick failed", symbol)

    # Summary
    total = portfolio.total_value(prices)
    log.info("Portfolio: Cash $%.2f | Total $%.2f", portfolio.cash, total)
    portfolio.save()


def run(once: bool = False) -> None:
    log.info("=== Trading Agent started (%s, %s) ===", ", ".join(SYMBOLS), TIMEFRAME)
    notify_agent_status("started", f"Symbols: {', '.join(SYMBOLS)} | TF: {TIMEFRAME}")
    try:
        while True:
            try:
                tick()
            except Exception:
                log.exception("Error during tick")
            if once:
                break
            log.info("Next tick in %d seconds...", INTERVAL_SEC)
            time.sleep(INTERVAL_SEC)
    finally:
        notify_agent_status("stopped")


if __name__ == "__main__":
    import sys
    run(once="--once" in sys.argv)
