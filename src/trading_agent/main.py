"""Main loop — monitor multiple symbols, analyze sentiment, trade."""

import time
import logging

from trading_agent.fetcher import fetch_ohlcv
from trading_agent.strategy import (
    compute_indicators,
    sentiment_weighted_signal,
    SignalFilter,
    DEFAULT_BUY_COOLDOWN,
)
from trading_agent.sentiment import analyze_sentiment
from trading_agent.portfolio import Portfolio, log_trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "1h"
INTERVAL_SEC = 60 * 60


def tick_symbol(symbol: str, portfolio: Portfolio, sentiment_score: float) -> None:
    log.info("[%s] Fetching %s candles...", symbol, TIMEFRAME)
    df = fetch_ohlcv(symbol, TIMEFRAME)
    df = compute_indicators(df)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest["close"]
    rsi = latest["rsi"]

    pos = portfolio.get_position(symbol)

    # Generate raw signal
    raw_signal = sentiment_weighted_signal(
        rsi=rsi,
        macd_diff=latest["macd_diff"],
        prev_macd_diff=prev["macd_diff"],
        sentiment_score=sentiment_score,
    )

    # Apply cooldown filter (per-symbol state)
    sig_filter = SignalFilter(buy_cooldown=DEFAULT_BUY_COOLDOWN)
    sig_filter.ticks_since_buy = pos.ticks_since_buy
    signal = sig_filter.filter(raw_signal)
    pos.ticks_since_buy = sig_filter.ticks_since_buy
    portfolio._save_pos(symbol, pos)

    log.info(
        "[%s] $%.2f | RSI: %.1f | Sent: %+.2f | Raw: %s | Signal: %s | Pos: %.6f",
        symbol, price, rsi, sentiment_score,
        raw_signal.upper(), signal.upper(), pos.qty,
    )

    trade = None
    if signal == "buy":
        trade = portfolio.buy(symbol, price)
    elif signal == "sell":
        trade = portfolio.sell(symbol, price)

    if trade:
        log.info(
            "[%s] TRADE: %s %.6f @ $%.2f",
            symbol, trade["side"].upper(), trade["qty"], price,
        )
        log_trade(trade, signal, rsi)


def tick() -> None:
    portfolio = Portfolio.load()

    # Fetch sentiment once per symbol
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
            tick_symbol(symbol, portfolio, score)
        except Exception:
            log.exception("[%s] Tick failed", symbol)

    # Summary
    prices = {}
    for symbol in SYMBOLS:
        try:
            df = fetch_ohlcv(symbol, TIMEFRAME, limit=2)
            prices[symbol] = df.iloc[-1]["close"]
        except Exception:
            pass
    total = portfolio.total_value(prices)
    log.info("Portfolio: Cash $%.2f | Total $%.2f", portfolio.cash, total)
    portfolio.save()


def run(once: bool = False) -> None:
    log.info("=== Trading Agent started (%s, %s) ===", ", ".join(SYMBOLS), TIMEFRAME)
    while True:
        try:
            tick()
        except Exception:
            log.exception("Error during tick")
        if once:
            break
        log.info("Next tick in %d seconds...", INTERVAL_SEC)
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    import sys
    run(once="--once" in sys.argv)
