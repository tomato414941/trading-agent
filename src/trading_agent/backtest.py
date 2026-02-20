"""Backtest engine — simulate trading strategies on historical data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from itertools import product
from pathlib import Path

from trading_agent.fetcher import fetch_ohlcv, fetch_ohlcv_paginated
from trading_agent.strategy import (
    compute_indicators,
    rsi_signal,
    composite_signal,
    sentiment_multiplier,
    SignalFilter,
    DEFAULT_BUY_COOLDOWN,
)

# Binance taker fee
DEFAULT_FEE_RATE = 0.001


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
    stop_loss_count: int = 0
    take_profit_count: int = 0
    period_start: str = ""
    period_end: str = ""
    trades: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        low_sample = " (low sample)" if self.num_trades < 3 else ""
        lines = [
            f"=== {self.strategy_name} ===",
            f"Period:          {self.period_start} → {self.period_end}",
            f"Initial:         ${self.initial_cash:,.2f}",
            f"Final:           ${self.final_value:,.2f}",
            f"Return:          {self.total_return_pct:+.2f}%",
            f"Buy & Hold:      {self.buy_and_hold_pct:+.2f}%",
            f"Trades:          {self.num_trades}{low_sample}",
            f"Win Rate:        {self.win_rate:.1f}%{low_sample}",
            f"Max Drawdown:    {self.max_drawdown_pct:.2f}%",
            f"Sharpe Ratio:    {self.sharpe_ratio:.2f}{low_sample}",
        ]
        if self.stop_loss_count or self.take_profit_count:
            lines.append(f"Stop Loss:       {self.stop_loss_count}")
            lines.append(f"Take Profit:     {self.take_profit_count}")
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
    fee_rate: float = DEFAULT_FEE_RATE,
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
    max_exposure_pct: float = 100.0,
    strategy: str = "rsi",
    sentiment_score: float = 0.0,
    df: pd.DataFrame | None = None,
) -> BacktestResult:
    if df is None:
        df = fetch_ohlcv(symbol, timeframe, limit)
    df = compute_indicators(df)
    df = df.reset_index(drop=True)

    signal_fn = STRATEGIES[strategy]

    # Sentiment adjusts buy size, not direction
    sent_mult = sentiment_multiplier(sentiment_score)
    effective_buy_fraction = buy_fraction * sent_mult

    cash = initial_cash
    position = 0.0
    avg_entry = 0.0
    trades: list[dict] = []
    equity_curve: list[float] = []
    sig_filter = SignalFilter(buy_cooldown=buy_cooldown)
    prev_row = None
    pending_signal: str | None = None
    sl_count = 0
    tp_count = 0

    for i, row in df.iterrows():
        price = row["close"]

        # Stop-loss / take-profit check
        if position > 0 and avg_entry > 0:
            pnl_pct = (price - avg_entry) / avg_entry * 100
            if stop_loss_pct > 0 and pnl_pct <= -stop_loss_pct:
                fee = position * price * fee_rate
                revenue = position * price - fee
                pnl = revenue - (position * avg_entry)
                trades.append({
                    "timestamp": row["timestamp"],
                    "side": "sell",
                    "price": price,
                    "qty": position,
                    "fee": fee,
                    "pnl": pnl,
                    "reason": "stop_loss",
                })
                cash += revenue
                position = 0.0
                avg_entry = 0.0
                sl_count += 1
                pending_signal = None
            elif take_profit_pct > 0 and pnl_pct >= take_profit_pct:
                fee = position * price * fee_rate
                revenue = position * price - fee
                pnl = revenue - (position * avg_entry)
                trades.append({
                    "timestamp": row["timestamp"],
                    "side": "sell",
                    "price": price,
                    "qty": position,
                    "fee": fee,
                    "pnl": pnl,
                    "reason": "take_profit",
                })
                cash += revenue
                position = 0.0
                avg_entry = 0.0
                tp_count += 1
                pending_signal = None

        # Execute pending signal from previous candle at this candle's open
        if pending_signal is not None and i > 0:
            exec_price = row["open"]

            if pending_signal == "buy" and cash > 1.0:
                # Max exposure check
                equity = cash + position * exec_price
                if equity > 0 and max_exposure_pct < 100:
                    exposure = (position * exec_price / equity) * 100
                    if exposure >= max_exposure_pct:
                        pending_signal = None
            if pending_signal == "buy" and cash > 1.0:
                cost_before_fee = cash * effective_buy_fraction
                fee = cost_before_fee * fee_rate
                cost_total = cost_before_fee + fee
                if cost_total > cash:
                    cost_before_fee = cash / (1 + fee_rate)
                    fee = cost_before_fee * fee_rate
                    cost_total = cost_before_fee + fee
                qty = cost_before_fee / exec_price
                # Weighted average cost basis
                total_cost = position * avg_entry + qty * exec_price
                position += qty
                avg_entry = total_cost / position if position > 0 else exec_price
                cash -= cost_total
                trades.append({
                    "timestamp": row["timestamp"],
                    "side": "buy",
                    "price": exec_price,
                    "qty": qty,
                    "fee": fee,
                    "pnl": None,
                })

            elif pending_signal == "sell" and position > 0:
                revenue_before_fee = position * exec_price
                fee = revenue_before_fee * fee_rate
                revenue = revenue_before_fee - fee
                pnl = revenue - (position * avg_entry)
                trades.append({
                    "timestamp": row["timestamp"],
                    "side": "sell",
                    "price": exec_price,
                    "qty": position,
                    "fee": fee,
                    "pnl": pnl,
                })
                cash += revenue
                position = 0.0
                avg_entry = 0.0

        pending_signal = None

        # Compute signal on this candle (will execute on next candle's open)
        equity = cash + position * price
        equity_curve.append(equity)

        raw = signal_fn(row, prev_row)
        signal = sig_filter.filter(raw)
        prev_row = row

        if signal in ("buy", "sell"):
            pending_signal = signal

    # Final valuation
    final_price = df.iloc[-1]["close"]
    final_value = cash + position * final_price
    first_price = df.iloc[0]["close"]

    period_start = str(df.iloc[0]["timestamp"])
    period_end = str(df.iloc[-1]["timestamp"])

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
        stop_loss_count=sl_count,
        take_profit_count=tp_count,
        period_start=period_start,
        period_end=period_end,
        trades=trades,
    )


def compare(results: list[BacktestResult]) -> str:
    header = f"{'Strategy':<14} {'Return':>8} {'B&H':>8} {'Trades':>7} {'Win%':>6} {'MaxDD':>7} {'Sharpe':>7}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        lines.append(
            f"{r.strategy_name:<14} {r.total_return_pct:>+7.2f}% {r.buy_and_hold_pct:>+7.2f}% {r.num_trades:>7} {r.win_rate:>5.1f}% {r.max_drawdown_pct:>6.2f}% {r.sharpe_ratio:>7.2f}"
        )
    return "\n".join(lines)


SENTIMENT_SCENARIOS: dict[str, float] = {
    "sent+0.5": 0.5,
    "sent-0.5": -0.5,
    "sent=0.0": 0.0,
}


def parameter_sweep(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1000,
    strategy: str = "rsi+macd",
    rsi_windows: list[int] | None = None,
    cooldowns: list[int] | None = None,
    stop_losses: list[float] | None = None,
    take_profits: list[float] | None = None,
    df: pd.DataFrame | None = None,
) -> list[BacktestResult]:
    if rsi_windows is None:
        rsi_windows = [10, 14, 20]
    if cooldowns is None:
        cooldowns = [6, 12, 24]
    if stop_losses is None:
        stop_losses = [0, 3, 5, 8]
    if take_profits is None:
        take_profits = [0, 10, 15, 20]

    if df is None:
        df = fetch_ohlcv(symbol, timeframe, limit)

    results: list[BacktestResult] = []
    combos = list(product(rsi_windows, cooldowns, stop_losses, take_profits))
    total = len(combos)

    for idx, (rsi_w, cd, sl, tp) in enumerate(combos, 1):
        df_ind = compute_indicators(df.copy(), rsi_window=rsi_w)
        r = run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            buy_cooldown=cd,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            strategy=strategy,
            df=df_ind,
        )
        r.strategy_name = f"rsi{rsi_w}_cd{cd}_sl{sl}_tp{tp}"
        results.append(r)

        if idx % 20 == 0 or idx == total:
            print(f"  sweep: {idx}/{total} done")

    results.sort(key=lambda r: r.total_return_pct, reverse=True)
    return results


def save_sweep_results(results: list[BacktestResult], path: str | Path) -> None:
    path = Path(path)
    rows = []
    for r in results:
        rows.append({
            "strategy": r.strategy_name,
            "return_pct": round(r.total_return_pct, 4),
            "buy_hold_pct": round(r.buy_and_hold_pct, 4),
            "trades": r.num_trades,
            "win_rate": round(r.win_rate, 2),
            "max_dd_pct": round(r.max_drawdown_pct, 4),
            "sharpe": round(r.sharpe_ratio, 4),
            "sl_count": r.stop_loss_count,
            "tp_count": r.take_profit_count,
            "period_start": r.period_start,
            "period_end": r.period_end,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {len(rows)} results to {path}")


if __name__ == "__main__":
    import sys

    DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    if "--sweep" in sys.argv:
        # python -m trading_agent.backtest --sweep BTC/USDT --limit 1000
        args = [a for a in sys.argv[1:] if a != "--sweep"]
        symbol = args[0] if args else "BTC/USDT"
        limit = 1000
        timeframe = "1h"
        for i, a in enumerate(args):
            if a == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
            if a == "--tf" and i + 1 < len(args):
                timeframe = args[i + 1]

        print(f"Fetching {limit} candles for {symbol} ({timeframe})...")
        if limit > 1000:
            df = fetch_ohlcv_paginated(symbol, timeframe, total=limit)
        else:
            df = fetch_ohlcv(symbol, timeframe, limit)
        print(f"Got {len(df)} candles: {df.iloc[0]['timestamp']} → {df.iloc[-1]['timestamp']}")

        results = parameter_sweep(symbol, timeframe, limit, df=df)
        print("\nTop 10 results:")
        print(compare(results[:10]))

        csv_path = f"sweep_{symbol.replace('/', '_')}_{timeframe}_{limit}.csv"
        save_sweep_results(results, csv_path)
    else:
        timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        symbols = sys.argv[3:] if len(sys.argv) > 3 else DEFAULT_SYMBOLS

        for symbol in symbols:
            df = fetch_ohlcv(symbol, timeframe, limit)
            print(f"\n{'='*60}")
            print(f"  {symbol}  ({timeframe}, {limit} candles)")
            print(f"{'='*60}\n")

            results = []
            for strat in STRATEGIES:
                result = run_backtest(symbol, timeframe, limit, strategy=strat, df=df)
                results.append(result)
                print(result.summary())
                print()

            for name, score in SENTIMENT_SCENARIOS.items():
                result = run_backtest(
                    symbol, timeframe, limit,
                    strategy="rsi+macd", sentiment_score=score, df=df,
                )
                result = BacktestResult(
                    strategy_name=name,
                    initial_cash=result.initial_cash,
                    final_value=result.final_value,
                    total_return_pct=result.total_return_pct,
                    buy_and_hold_pct=result.buy_and_hold_pct,
                    num_trades=result.num_trades,
                    win_rate=result.win_rate,
                    max_drawdown_pct=result.max_drawdown_pct,
                    sharpe_ratio=result.sharpe_ratio,
                    stop_loss_count=result.stop_loss_count,
                    take_profit_count=result.take_profit_count,
                    period_start=result.period_start,
                    period_end=result.period_end,
                    trades=result.trades,
                )
                results.append(result)
                print(result.summary())
                print()

            print(compare(results))
            print()
