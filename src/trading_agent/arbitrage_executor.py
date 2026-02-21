"""Live FR arbitrage executor: spot long + futures short on Binance.

Usage:
    python -m trading_agent.arbitrage_executor --status
    python -m trading_agent.arbitrage_executor --run
    python -m trading_agent.arbitrage_executor --close
"""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import ccxt

from trading_agent.config import ArbitrageConfig, DEFAULT_ARBITRAGE
from trading_agent.exchange import create_futures_exchange, create_spot_exchange

log = logging.getLogger(__name__)

STATE_PATH = Path("data/arbitrage_state.json")
FUTURES_SYMBOL_SUFFIX = ":USDT"


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

@dataclass
class LiveArbitrageState:
    """Persistent state for live arbitrage positions."""
    is_open: bool = False
    symbol: str = ""
    spot_qty: float = 0.0
    spot_entry_price: float = 0.0
    futures_qty: float = 0.0
    futures_entry_price: float = 0.0
    accumulated_fr: float = 0.0
    entry_time: str = ""
    low_fr_count: int = 0
    # Lifetime stats
    total_fr_collected: float = 0.0
    total_fees_paid: float = 0.0
    num_round_trips: int = 0
    trades: list[dict] = field(default_factory=list)

    def save(self, path: Path = STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> LiveArbitrageState:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def summary(self) -> str:
        lines = ["=== Arbitrage State ==="]
        if self.is_open:
            lines.append(f"Position:  OPEN ({self.symbol})")
            lines.append(f"Spot:      {self.spot_qty:.6f} @ ${self.spot_entry_price:,.2f}")
            lines.append(f"Futures:   {self.futures_qty:.6f} short @ ${self.futures_entry_price:,.2f}")
            lines.append(f"FR Accum:  ${self.accumulated_fr:,.4f}")
            lines.append(f"Entry:     {self.entry_time}")
            lines.append(f"Low FR:    {self.low_fr_count} consecutive")
        else:
            lines.append("Position:  CLOSED (no active position)")
        lines.append(f"Lifetime FR: ${self.total_fr_collected:,.4f}")
        lines.append(f"Lifetime Fees: ${self.total_fees_paid:,.4f}")
        lines.append(f"Round Trips: {self.num_round_trips}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core execution functions
# ---------------------------------------------------------------------------

def _futures_symbol(symbol: str) -> str:
    """Convert spot symbol to futures: BTC/USDT -> BTC/USDT:USDT."""
    if FUTURES_SYMBOL_SUFFIX not in symbol:
        return symbol + FUTURES_SYMBOL_SUFFIX
    return symbol


def get_funding_rate(ex_futures: ccxt.binance, symbol: str) -> float:
    """Fetch current funding rate for symbol."""
    info = ex_futures.fetch_funding_rate(_futures_symbol(symbol))
    return float(info.get("fundingRate", 0))


def get_spot_price(ex_futures: ccxt.binance, symbol: str) -> float:
    """Fetch current spot price via ticker."""
    ticker = ex_futures.fetch_ticker(_futures_symbol(symbol))
    return float(ticker["last"])


def open_position(
    ex_futures: ccxt.binance,
    ex_spot: ccxt.binance,
    state: LiveArbitrageState,
    symbol: str,
    amount_usdt: float,
    config: ArbitrageConfig,
) -> None:
    """Open delta-neutral: spot buy + futures short."""
    price = get_spot_price(ex_futures, symbol)
    qty = amount_usdt / price

    # Round to exchange precision
    ex_futures.load_markets()
    qty = float(ex_futures.amount_to_precision(_futures_symbol(symbol), qty))

    if qty <= 0:
        log.warning("Quantity too small to open position: %s", qty)
        return

    # Set leverage to 1x
    try:
        ex_futures.set_leverage(int(config.futures_leverage), _futures_symbol(symbol))
    except Exception as e:
        log.warning("set_leverage failed (may already be set): %s", e)

    # Execute both legs
    log.info("Opening position: spot buy %.6f + futures short %.6f %s", qty, qty, symbol)

    spot_order = ex_spot.create_market_buy_order(symbol, qty)
    futures_order = ex_futures.create_market_sell_order(_futures_symbol(symbol), qty)

    spot_price = float(spot_order.get("average", price))
    futures_price = float(futures_order.get("average", price))
    spot_fee = float(spot_order.get("cost", amount_usdt)) * config.spot_fee_rate
    futures_fee = float(futures_order.get("cost", amount_usdt)) * config.futures_fee_rate

    # Update state
    state.is_open = True
    state.symbol = symbol
    state.spot_qty = qty
    state.spot_entry_price = spot_price
    state.futures_qty = qty
    state.futures_entry_price = futures_price
    state.accumulated_fr = 0.0
    state.entry_time = datetime.now(timezone.utc).isoformat()
    state.low_fr_count = 0
    state.total_fees_paid += spot_fee + futures_fee
    state.save()

    log.info(
        "Position opened: spot=%.2f futures=%.2f fees=%.4f",
        spot_price, futures_price, spot_fee + futures_fee,
    )


def close_position(
    ex_futures: ccxt.binance,
    ex_spot: ccxt.binance,
    state: LiveArbitrageState,
    config: ArbitrageConfig,
) -> None:
    """Close delta-neutral: spot sell + futures cover."""
    if not state.is_open:
        log.warning("No open position to close")
        return

    symbol = state.symbol
    qty = state.spot_qty

    log.info("Closing position: spot sell %.6f + futures cover %.6f %s", qty, qty, symbol)

    spot_order = ex_spot.create_market_sell_order(symbol, qty)
    futures_order = ex_futures.create_market_buy_order(
        _futures_symbol(symbol), qty, params={"reduceOnly": True}
    )

    spot_fee = float(spot_order.get("cost", 0)) * config.spot_fee_rate
    futures_fee = float(futures_order.get("cost", 0)) * config.futures_fee_rate

    pnl = state.accumulated_fr - (spot_fee + futures_fee)

    state.trades.append({
        "entry_time": state.entry_time,
        "exit_time": datetime.now(timezone.utc).isoformat(),
        "fr_collected": state.accumulated_fr,
        "pnl": pnl,
    })
    state.total_fees_paid += spot_fee + futures_fee
    state.num_round_trips += 1
    state.is_open = False
    state.spot_qty = 0.0
    state.futures_qty = 0.0
    state.accumulated_fr = 0.0
    state.low_fr_count = 0
    state.save()

    log.info("Position closed: PnL=$%.4f", pnl)


def check_and_act(
    ex_futures: ccxt.binance,
    ex_spot: ccxt.binance,
    state: LiveArbitrageState,
    config: ArbitrageConfig,
) -> str:
    """One tick of the arbitrage loop. Returns action taken."""
    symbol = config.symbol
    fr = get_funding_rate(ex_futures, symbol)
    price = get_spot_price(ex_futures, symbol)

    log.info("Tick: %s price=$%.2f FR=%.6f (%.4f%%)", symbol, price, fr, fr * 100)

    if state.is_open:
        # Record FR payment
        fr_payment = state.spot_qty * price * fr
        state.accumulated_fr += fr_payment
        state.total_fr_collected += fr_payment

        # Check exit
        if fr < config.exit_fr_threshold:
            state.low_fr_count += 1
        else:
            state.low_fr_count = 0

        if state.low_fr_count >= config.exit_consecutive_periods:
            close_position(ex_futures, ex_spot, state, config)
            return "closed"

        state.save()
        return f"holding (FR=${fr_payment:+.4f}, low_count={state.low_fr_count})"

    elif fr > config.entry_fr_threshold:
        # Calculate position size
        balance = ex_spot.fetch_balance()
        free_usdt = float(balance.get("USDT", {}).get("free", 0))
        invest = free_usdt * config.position_fraction

        if invest < 10:
            log.warning("Insufficient balance: $%.2f", free_usdt)
            return "skip (low balance)"

        open_position(ex_futures, ex_spot, state, symbol, invest, config)
        return "opened"

    else:
        return f"waiting (FR={fr:.6f} < threshold={config.entry_fr_threshold})"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(config: ArbitrageConfig = DEFAULT_ARBITRAGE) -> None:
    """Main execution loop. Ctrl+C to stop."""
    log.info("Starting arbitrage loop (testnet=%s, interval=%ds)", config.testnet, config.check_interval_sec)

    ex_futures = create_futures_exchange(testnet=config.testnet)
    ex_spot = create_spot_exchange(testnet=config.testnet)
    state = LiveArbitrageState.load()

    stop = False

    def _handle_signal(signum, frame):
        nonlocal stop
        stop = True
        log.info("Received signal %d, stopping...", signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while not stop:
        try:
            action = check_and_act(ex_futures, ex_spot, state, config)
            log.info("Action: %s", action)
            print(f"[{datetime.now(timezone.utc).isoformat()}] {action}", flush=True)
        except Exception:
            log.exception("Error in arbitrage loop")

        # Wait in small increments so we can respond to signals
        for _ in range(config.check_interval_sec):
            if stop:
                break
            time.sleep(1)

    log.info("Arbitrage loop stopped")
    print(state.summary())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = sys.argv[1:]

    if "--status" in args:
        state = LiveArbitrageState.load()
        print(state.summary())

    elif "--close" in args:
        config = DEFAULT_ARBITRAGE
        state = LiveArbitrageState.load()
        if not state.is_open:
            print("No open position")
        else:
            ex_futures = create_futures_exchange(testnet=config.testnet)
            ex_spot = create_spot_exchange(testnet=config.testnet)
            close_position(ex_futures, ex_spot, state, config)
            print("Position closed")
            print(state.summary())

    elif "--run" in args:
        config = DEFAULT_ARBITRAGE
        # Parse optional overrides
        for i, a in enumerate(args):
            if a == "--threshold" and i + 1 < len(args):
                config.entry_fr_threshold = float(args[i + 1])
            if a == "--interval" and i + 1 < len(args):
                config.check_interval_sec = int(args[i + 1])
            if a == "--symbol" and i + 1 < len(args):
                config.symbol = args[i + 1]
            if a == "--live":
                config.testnet = False
        run_loop(config)

    else:
        print("Usage:")
        print("  --status              Show current arbitrage state")
        print("  --run                 Start arbitrage loop (testnet)")
        print("    --threshold 0.0003  FR entry threshold")
        print("    --interval 300      Check interval in seconds")
        print("    --symbol BTC/USDT   Target symbol")
        print("    --live              Use mainnet (CAUTION!)")
        print("  --close               Close open position")
