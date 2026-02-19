"""Main loop — fetch data, analyze sentiment, generate signal, execute virtual trade."""

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

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
INTERVAL_SEC = 60 * 60  # run every hour


def tick() -> None:
    portfolio = Portfolio.load()

    log.info("Fetching %s %s candles...", SYMBOL, TIMEFRAME)
    df = fetch_ohlcv(SYMBOL, TIMEFRAME)
    df = compute_indicators(df)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest["close"]
    rsi = latest["rsi"]

    # Sentiment analysis (falls back to 0.0 on failure)
    log.info("Analyzing news sentiment...")
    sent = analyze_sentiment()
    sentiment_score = sent["score"]
    log.info("Sentiment: %+.2f (%s)", sentiment_score, sent["summary"][:80])

    # Generate raw signal with all three indicators
    raw_signal = sentiment_weighted_signal(
        rsi=rsi,
        macd_diff=latest["macd_diff"],
        prev_macd_diff=prev["macd_diff"],
        sentiment_score=sentiment_score,
    )

    # Apply cooldown filter
    sig_filter = SignalFilter(buy_cooldown=DEFAULT_BUY_COOLDOWN)
    sig_filter._ticks_since_buy = portfolio.ticks_since_buy
    signal = sig_filter.filter(raw_signal)
    portfolio.ticks_since_buy = sig_filter._ticks_since_buy

    total = portfolio.cash + portfolio.position * price
    log.info(
        "Price: $%.2f | RSI: %.1f | Sentiment: %+.2f | Raw: %s | Signal: %s | Cash: $%.2f | Total: $%.2f",
        price, rsi, sentiment_score, raw_signal.upper(), signal.upper(),
        portfolio.cash, total,
    )

    trade = None
    if signal == "buy":
        trade = portfolio.buy(price)
    elif signal == "sell":
        trade = portfolio.sell(price)

    if trade:
        log.info("TRADE: %s %.6f BTC @ $%.2f", trade["side"].upper(), trade["qty"], price)
        log_trade(trade, signal, rsi)
    else:
        log.info("No trade executed.")

    portfolio.save()


def run(once: bool = False) -> None:
    log.info("=== Trading Agent started (%s, %s, RSI+MACD+Sentiment) ===", SYMBOL, TIMEFRAME)
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
