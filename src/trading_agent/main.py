"""Main loop — fetch data, compute RSI, generate signal, execute virtual trade."""

import time
import logging
from datetime import datetime

from trading_agent.fetcher import fetch_ohlcv
from trading_agent.strategy import compute_rsi, generate_signal
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
    df = compute_rsi(df)

    latest = df.iloc[-1]
    price = latest["close"]
    rsi = latest["rsi"]
    signal = generate_signal(df)

    total = portfolio.cash + portfolio.position * price
    log.info(
        "Price: $%.2f | RSI: %.1f | Signal: %s | Cash: $%.2f | BTC: %.6f | Total: $%.2f",
        price, rsi, signal.upper(), portfolio.cash, portfolio.position, total,
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
    log.info("=== Trading Agent started (%s, %s) ===", SYMBOL, TIMEFRAME)
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
