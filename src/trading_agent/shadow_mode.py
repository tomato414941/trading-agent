"""Shadow mode: paper trading with real data for strategy validation."""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timezone

from trading_agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from trading_agent.config import AgentConfig, RiskConfig
from trading_agent.feature_engine import CryptoFeatureEngine, FeatureConfig
from trading_agent.fetcher import fetch_ohlcv, fetch_ohlcv_paginated
from trading_agent.ml_strategy import CryptoOnlineLearner, MLConfig
from trading_agent.order_engine import PaperOrderEngine
from trading_agent.performance import PerformanceTracker
from trading_agent.signal_noise_client import SignalNoiseClient
from trading_agent.strategy import compute_indicators

log = logging.getLogger(__name__)


class ShadowRunner:
    """Run full ML strategy against real data with paper execution."""

    def __init__(
        self,
        agent_config: AgentConfig | None = None,
        risk_config: RiskConfig | None = None,
        ml_config: MLConfig | None = None,
        initial_cash: float = 10_000.0,
    ):
        self._agent_config = agent_config or AgentConfig()
        self._risk_config = risk_config or RiskConfig()
        self._ml_config = ml_config or MLConfig()

        self._engine = PaperOrderEngine(initial_cash=initial_cash)
        self._breaker = CircuitBreaker(CircuitBreakerConfig())
        self._learner = CryptoOnlineLearner.load()
        self._feature_engine = CryptoFeatureEngine(FeatureConfig())
        self._signal_client = SignalNoiseClient(self._ml_config.signal_noise_url)
        self._performance = PerformanceTracker.load()

        self._prev_prices: dict[str, float] = {}
        self._stop = False

    def tick(self) -> dict:
        """One cycle: fetch data, predict, (paper) trade, learn."""
        result = {"timestamp": datetime.now(timezone.utc).isoformat(), "actions": []}

        for symbol in self._agent_config.symbols:
            try:
                action = self._tick_symbol(symbol)
                result["actions"].append({"symbol": symbol, **action})
            except Exception as e:
                log.exception("Shadow tick failed for %s: %s", symbol, e)
                result["actions"].append({"symbol": symbol, "error": str(e)})

        return result

    def _tick_symbol(self, symbol: str) -> dict:
        # Fetch OHLCV
        df = fetch_ohlcv(symbol, self._agent_config.signal_timeframe,
                        self._agent_config.signal_candle_limit)
        df = compute_indicators(df)
        current_price = float(df.iloc[-1]["close"])
        self._engine.set_price(symbol, current_price)

        # Build features from OHLCV indicators + signal-noise
        ohlcv_signals = {
            f"{symbol.split('/')[0].lower()}_close": current_price,
            f"{symbol.split('/')[0].lower()}_rsi": float(df.iloc[-1].get("rsi", 50)),
            f"{symbol.split('/')[0].lower()}_macd_diff": float(df.iloc[-1].get("macd_diff", 0)),
            f"{symbol.split('/')[0].lower()}_vol_ratio": float(df.iloc[-1].get("vol_ratio", 1)),
        }

        # Fetch signal-noise data (graceful degradation)
        sn_values = {}
        if self._signal_client.health():
            sn_values = self._signal_client.get_latest_values()

        all_signals = {**ohlcv_signals, **sn_values}
        features = self._feature_engine.compute(all_signals)

        # Delayed learning: teach model about the outcome of previous prediction
        prev_price = self._prev_prices.get(symbol)
        if prev_price is not None:
            target = self._feature_engine.compute_target(prev_price, current_price)
            self._learner.learn_delayed(target)

            # Update accuracy tracking
            last_pred = getattr(self, "_last_prediction", None)
            if last_pred is not None:
                self._learner.update_accuracy(last_pred.direction, target)
                self._learner.update_calibration(last_pred.confidence, target)

        self._prev_prices[symbol] = current_price

        # Predict
        prediction = self._learner.predict(features)
        self._last_prediction = prediction

        action = {"price": current_price, "signal": "hold", "features_count": len(features)}

        if prediction is None:
            action["signal"] = f"warmup ({self._learner.samples_seen}/{self._ml_config.grace_period})"
            return action

        action["confidence"] = prediction.calibrated_confidence
        action["direction"] = prediction.direction

        # Circuit breaker check
        equity = self._engine.cash + sum(
            self._engine.positions.get(s, 0) * self._engine.get_price(s)
            for s in self._agent_config.symbols
        )
        self._breaker.reset_daily(equity)
        safe, reason = self._breaker.is_safe_to_trade(equity)
        if not safe:
            action["signal"] = f"blocked ({reason})"
            return action

        # Trade decision
        if not self._learner.should_trade(prediction):
            action["signal"] = f"low confidence ({prediction.calibrated_confidence:.3f})"
            return action

        kelly = self._learner.kelly_size(
            prediction,
            stop_loss_pct=self._risk_config.stop_loss_pct,
            take_profit_pct=self._risk_config.take_profit_pct,
        )

        if prediction.direction == 1 and kelly > 0:
            amount = self._engine.cash * kelly
            order = self._engine.market_buy(symbol, amount)
            if order:
                action["signal"] = "buy"
                action["qty"] = order.qty
                action["kelly"] = kelly
                self._performance.record_trade(0, prediction.calibrated_confidence)
        elif prediction.direction == 0 and self._engine.positions.get(symbol, 0) > 0:
            qty = self._engine.positions[symbol]
            entry_price = current_price  # simplified
            order = self._engine.market_sell(symbol, qty)
            if order:
                pnl = order.qty * (order.price - entry_price)
                action["signal"] = "sell"
                action["qty"] = order.qty
                self._breaker.record_trade(pnl)
                self._performance.record_trade(pnl, prediction.calibrated_confidence)

        return action

    def run(self, interval_sec: int | None = None) -> None:
        """Main loop. Ctrl+C to stop."""
        interval = interval_sec or self._agent_config.signal_interval_sec
        log.info("Starting shadow mode (interval=%ds)", interval)

        def _handle_signal(signum, frame):
            self._stop = True
            log.info("Received signal %d, stopping shadow mode...", signum)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        equity = self._engine.cash
        self._performance.start_day(equity)

        while not self._stop:
            try:
                result = self.tick()
                for a in result["actions"]:
                    log.info("Shadow: %s %s (confidence=%.3f)",
                            a.get("symbol", "?"), a.get("signal", "?"),
                            a.get("confidence", 0))
            except Exception:
                log.exception("Shadow tick error")

            # Save state periodically
            self._learner.save()
            self._performance.update_model_stats(
                self._learner.accuracy, self._learner.samples_seen
            )
            self._performance.save()

            for _ in range(interval):
                if self._stop:
                    break
                time.sleep(1)

        # End of day
        equity = self._engine.cash + sum(
            self._engine.positions.get(s, 0) * self._engine.get_price(s)
            for s in self._agent_config.symbols
        )
        metrics = self._performance.end_day(equity)
        if metrics:
            log.info("Day summary: PnL=$%.2f trades=%d", metrics.pnl, metrics.num_trades)

        self._learner.save()
        self._performance.save()

        # Check graduation
        can, reason = self._performance.should_graduate()
        log.info("Graduation check: %s (%s)", "READY" if can else "NOT READY", reason)
        print(self._performance.summary())

    def warmup(self, candles: int = 200) -> None:
        """Pre-warm model using historical OHLCV + signal-noise data."""
        from trading_agent.signal_noise_client import CRYPTO_SIGNALS

        log.info("Warmup: fetching %d candles per symbol...", candles)

        # Fetch historical signal-noise data (daily resolution)
        sn_history: dict[str, list[tuple[str, float]]] = {}
        for name in CRYPTO_SIGNALS:
            rows = self._signal_client.get_data(name)
            if rows:
                sn_history[name] = [
                    (r["timestamp"], float(r["value"])) for r in rows if r.get("value") is not None
                ]

        for symbol in self._agent_config.symbols:
            prefix = symbol.split("/")[0].lower()
            log.info("Warmup %s: fetching OHLCV...", symbol)
            df = fetch_ohlcv_paginated(symbol, self._agent_config.signal_timeframe, total=candles)
            df = compute_indicators(df)
            log.info("Warmup %s: got %d candles, training...", symbol, len(df))

            prev_price = None
            for i in range(len(df)):
                row = df.iloc[i]
                price = float(row["close"])
                ts = str(row["timestamp"])

                ohlcv_signals = {
                    f"{prefix}_close": price,
                    f"{prefix}_rsi": float(row.get("rsi", 50)),
                    f"{prefix}_macd_diff": float(row.get("macd_diff", 0)),
                    f"{prefix}_vol_ratio": float(row.get("vol_ratio", 1)),
                }

                # Find closest signal-noise values for this timestamp
                sn_values = {}
                for name, hist in sn_history.items():
                    for j in range(len(hist) - 1, -1, -1):
                        if hist[j][0] <= ts:
                            sn_values[name] = hist[j][1]
                            break

                all_signals = {**ohlcv_signals, **sn_values}
                features = self._feature_engine.compute(all_signals)

                if prev_price is not None and features:
                    target = self._feature_engine.compute_target(prev_price, price)
                    self._learner.learn(features, target)

                prev_price = price

            log.info(
                "Warmup %s done: %d samples, warm=%s",
                symbol, self._learner.samples_seen, self._learner.is_warm,
            )

        self._learner.save()
        log.info(
            "Warmup complete: %d total samples, accuracy tracking starts on live ticks",
            self._learner.samples_seen,
        )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    interval = 3600
    for i, a in enumerate(sys.argv):
        if a == "--interval" and i + 1 < len(sys.argv):
            interval = int(sys.argv[i + 1])

    if "--warmup" in sys.argv:
        candles = 200
        for i, a in enumerate(sys.argv):
            if a == "--candles" and i + 1 < len(sys.argv):
                candles = int(sys.argv[i + 1])
        runner = ShadowRunner()
        runner.warmup(candles=candles)
        print(f"Model warmed up: {runner._learner.samples_seen} samples")
    elif "--once" in sys.argv:
        runner = ShadowRunner()
        result = runner.tick()
        for a in result["actions"]:
            print(f"{a.get('symbol')}: {a.get('signal')} (price={a.get('price', 0):.2f})")
    else:
        runner = ShadowRunner()
        runner.run(interval_sec=interval)
