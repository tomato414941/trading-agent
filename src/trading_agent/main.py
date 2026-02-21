"""Orchestrator: run three layers on independent schedules via threading."""

from __future__ import annotations

import argparse
import logging
import threading
import time

from trading_agent.config import DEFAULT_RISK, AgentConfig
from trading_agent.fetcher import fetch_ohlcv
from trading_agent.monitor import run_monitor
from trading_agent.notify import notify_agent_status
from trading_agent.regime import compute_regime
from trading_agent.signal import tick
from trading_agent.state import SharedState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run_regime_loop(
    state: SharedState,
    config: AgentConfig,
    stop_event: threading.Event,
) -> None:
    """Layer 3: Regime detection loop — runs in a daemon thread."""
    log.info("[REGIME] Started (interval=%ds, tf=%s)",
             config.regime_interval_sec, config.regime_timeframe)
    while not stop_event.is_set():
        for symbol in config.symbols:
            try:
                df = fetch_ohlcv(
                    symbol, config.regime_timeframe,
                    limit=config.regime_candle_limit,
                )
                regime_state = compute_regime(df)
                with state:
                    state.regimes[symbol] = regime_state
                log.info(
                    "[REGIME] [%s] %s (EMA200=%.2f, ADX=%.1f)",
                    symbol, regime_state.regime.upper(),
                    regime_state.ema200, regime_state.adx,
                )
            except Exception:
                log.exception("[REGIME] [%s] Failed", symbol)
        stop_event.wait(config.regime_interval_sec)
    log.info("[REGIME] Stopped")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crypto trading agent")
    p.add_argument("--once", action="store_true",
                   help="Run one signal cycle and exit")
    p.add_argument("--signal-tf", default="1h",
                   help="Signal timeframe (default: 1h)")
    p.add_argument("--regime-tf", default="4h",
                   help="Regime timeframe (default: 4h)")
    p.add_argument("--monitor-interval", type=int, default=120,
                   help="Risk monitor interval in seconds (default: 120)")
    p.add_argument("--signal-interval", type=int, default=3600,
                   help="Signal interval in seconds (default: 3600)")
    p.add_argument("--regime-interval", type=int, default=14400,
                   help="Regime interval in seconds (default: 14400)")
    return p.parse_args()


def run() -> None:
    args = parse_args()
    config = AgentConfig(
        signal_timeframe=args.signal_tf,
        regime_timeframe=args.regime_tf,
        monitor_interval_sec=args.monitor_interval,
        signal_interval_sec=args.signal_interval,
        regime_interval_sec=args.regime_interval,
    )
    risk = DEFAULT_RISK
    state = SharedState()
    stop_event = threading.Event()

    log.info("=== Trading Agent started ===")
    log.info("  Symbols: %s", ", ".join(config.symbols))
    log.info("  Signal TF: %s (every %ds)",
             config.signal_timeframe, config.signal_interval_sec)
    log.info("  Regime TF: %s (every %ds)",
             config.regime_timeframe, config.regime_interval_sec)
    log.info("  Monitor: every %ds", config.monitor_interval_sec)
    notify_agent_status(
        "started",
        f"Symbols: {', '.join(config.symbols)} | "
        f"Signal: {config.signal_timeframe} | "
        f"Monitor: {config.monitor_interval_sec}s",
    )

    if args.once:
        # Run one signal cycle synchronously (no threads)
        tick(state, config, risk)
        return

    # Layer 1: Risk Monitor (daemon thread)
    monitor_thread = threading.Thread(
        target=run_monitor,
        args=(state, config, risk, stop_event),
        daemon=True, name="risk-monitor",
    )
    monitor_thread.start()

    # Layer 3: Regime Detector (daemon thread)
    regime_thread = threading.Thread(
        target=run_regime_loop,
        args=(state, config, stop_event),
        daemon=True, name="regime-detector",
    )
    regime_thread.start()

    # Layer 2: Signal Generator (main thread)
    try:
        while True:
            try:
                tick(state, config, risk)
            except Exception:
                log.exception("Error during signal tick")
            log.info("Next signal tick in %ds...", config.signal_interval_sec)
            time.sleep(config.signal_interval_sec)
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        stop_event.set()
        monitor_thread.join(timeout=10)
        regime_thread.join(timeout=10)
        notify_agent_status("stopped")


if __name__ == "__main__":
    run()
