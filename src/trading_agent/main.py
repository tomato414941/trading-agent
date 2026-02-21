"""Main loop — orchestrate signal generation across multiple symbols."""

import time
import logging

from trading_agent.config import DEFAULT_RISK, DEFAULT_AGENT
from trading_agent.state import SharedState
from trading_agent.signal import tick
from trading_agent.notify import notify_agent_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run(once: bool = False) -> None:
    config = DEFAULT_AGENT
    risk = DEFAULT_RISK
    state = SharedState()

    log.info("=== Trading Agent started ===")
    log.info("  Symbols: %s", ", ".join(config.symbols))
    log.info("  Signal TF: %s (every %ds)", config.signal_timeframe,
             config.signal_interval_sec)
    notify_agent_status(
        "started",
        f"Symbols: {', '.join(config.symbols)} | TF: {config.signal_timeframe}",
    )
    try:
        while True:
            try:
                tick(state, config, risk)
            except Exception:
                log.exception("Error during tick")
            if once:
                break
            log.info("Next tick in %ds...", config.signal_interval_sec)
            time.sleep(config.signal_interval_sec)
    finally:
        notify_agent_status("stopped")


if __name__ == "__main__":
    import sys
    run(once="--once" in sys.argv)
