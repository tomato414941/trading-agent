"""Backtest engine — simulate trading strategies on historical data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from trading_agent.fetcher import fetch_ohlcv
from trading_agent.strategy import (
    compute_indicators,
    rsi_signal,
    composite_signal,
    SignalFilter,
    DEFAULT_BUY_COOLDOWN,
)


@dataclass
class BacktestResult:
    strategy_name: str
    initial_cash: float
    final_value: float
    total_return_pct: float
    buy_and_hold_pct: float
    num_trades: int
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== {self.strategy_name} ===",
            f"Period:          {self.trades[0]['timestamp']} → {self.trades[-1]['timestamp']}" if self.trades else "No trades",
            f"Initial:         ${self.initial_cash:,.2f}",
            f"Final:           ${self.final_value:,.2f}",
            f"Return:          {self.total_return_pct:+.2f}%",
            f"Buy & Hold:      {self.buy_and_hold_pct:+.2f}%",
            f"Trades:          {self.num_trades}",
            f"Win Rate:        {self.win_rate:.1f}%",
            f"Max Drawdown:    {self.max_drawdown_pct:.2f}%",
            f"Sharpe Ratio:    {self.sharpe_ratio:.2f}",
        ]
        return "\n".join(lines)


# --- Signal functions: (row, prev_row) -> str ---

def signal_rsi_only(row: pd.Series, prev_row: pd.Series | None) -> str:
    return rsi_signal(row["rsi"])


def signal_rsi_macd(row: pd.Series, prev_row: pd.Series | None) -> str:
    if prev_row is None:
        return "hold"
    return composite_signal(row["rsi"], row["macd_diff"], prev_row["macd_diff"])


STRATEGIES: dict[str, Callable] = {
    "rsi": signal_rsi_only,
    "rsi+macd": signal_rsi_macd,
}


def run_backtest(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 500,
    initial_cash: float = 10_000.0,
    buy_fraction: float = 0.1,
    buy_cooldown: int = DEFAULT_BUY_COOLDOWN,
    strategy: str = "rsi",
    df: pd.DataFrame | None = None,
) -> BacktestResult:
    if df is None:
        df = fetch_ohlcv(symbol, timeframe, limit)
    df = compute_indicators(df)

    signal_fn = STRATEGIES[strategy]

    cash = initial_cash
    position = 0.0
    entry_price = 0.0
    trades: list[dict] = []
    equity_curve: list[float] = []
    sig_filter = SignalFilter(buy_cooldown=buy_cooldown)
    prev_row = None

    for i, row in df.iterrows():
        price = row["close"]
        raw = signal_fn(row, prev_row)
        signal = sig_filter.filter(raw)
        prev_row = row

        equity = cash + position * price
        equity_curve.append(equity)

        if signal == "buy" and cash > 1.0:
            cost = cash * buy_fraction
            qty = cost / price
            cash -= cost
            position += qty
            entry_price = price
            trades.append({
                "timestamp": row["timestamp"],
                "side": "buy",
                "price": price,
                "qty": qty,
                "pnl": None,
            })

        elif signal == "sell" and position > 0:
            revenue = position * price
            pnl = revenue - position * entry_price
            trades.append({
                "timestamp": row["timestamp"],
                "side": "sell",
                "price": price,
                "qty": position,
                "pnl": pnl,
            })
            cash += revenue
            position = 0.0
            entry_price = 0.0

    # Final valuation
    final_price = df.iloc[-1]["close"]
    final_value = cash + position * final_price
    first_price = df.iloc[0]["close"]

    # Metrics
    total_return_pct = (final_value / initial_cash - 1) * 100
    buy_and_hold_pct = (final_price / first_price - 1) * 100

    sell_trades = [t for t in trades if t["side"] == "sell"]
    num_trades = len(sell_trades)
    wins = sum(1 for t in sell_trades if t["pnl"] and t["pnl"] > 0)
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0.0

    # Max drawdown
    eq = pd.Series(equity_curve)
    peak = eq.cummax()
    drawdown = (eq - peak) / peak * 100
    max_drawdown_pct = abs(drawdown.min())

    # Sharpe ratio (annualized)
    returns = eq.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        periods_per_year = {"1h": 8760, "4h": 2190, "1d": 365}.get(timeframe, 8760)
        sharpe_ratio = (returns.mean() / returns.std()) * (periods_per_year ** 0.5)
    else:
        sharpe_ratio = 0.0

    return BacktestResult(
        strategy_name=strategy,
        initial_cash=initial_cash,
        final_value=final_value,
        total_return_pct=total_return_pct,
        buy_and_hold_pct=buy_and_hold_pct,
        num_trades=num_trades,
        win_rate=win_rate,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        trades=trades,
    )


def compare(results: list[BacktestResult]) -> str:
    header = f"{'Strategy':<12} {'Return':>8} {'B&H':>8} {'Trades':>7} {'Win%':>6} {'MaxDD':>7} {'Sharpe':>7}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        lines.append(
            f"{r.strategy_name:<12} {r.total_return_pct:>+7.2f}% {r.buy_and_hold_pct:>+7.2f}% {r.num_trades:>7} {r.win_rate:>5.1f}% {r.max_drawdown_pct:>6.2f}% {r.sharpe_ratio:>7.2f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "1h"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    # Fetch data once, run both strategies on the same data
    df = fetch_ohlcv(symbol, timeframe, limit)

    print(f"Backtest: {symbol} {timeframe} ({limit} candles)\n")

    results = []
    for strat in STRATEGIES:
        result = run_backtest(symbol, timeframe, limit, strategy=strat, df=df)
        results.append(result)
        print(result.summary())
        if result.trades:
            print(f"  Trades: {len(result.trades)} entries")
        print()

    print("=== Comparison ===")
    print(compare(results))
