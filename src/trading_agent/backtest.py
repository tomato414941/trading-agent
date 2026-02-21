"""Backtest engine — simulate trading strategies on historical data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from itertools import product
from pathlib import Path

from trading_agent.fetcher import fetch_ohlcv, fetch_ohlcv_paginated, _TF_MS
from trading_agent.regime import ADX_TREND_THRESHOLD
from trading_agent.strategy import (
    compute_indicators,
    rsi_signal,
    composite_signal,
    bb_volume_signal,
    bb_rsi_signal,
    bb_volume_funding_signal,
    funding_rate_signal,
    sentiment_multiplier,
    SignalFilter,
    DEFAULT_BUY_COOLDOWN,
)

# Binance taker fee
DEFAULT_FEE_RATE = 0.001


@dataclass
class WalkForwardWindow:
    window_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: str
    train_return_pct: float
    test_return_pct: float
    test_trades: int
    test_win_rate: float
    test_max_dd_pct: float
    test_sharpe: float


@dataclass
class WalkForwardResult:
    symbol: str
    timeframe: str
    num_windows: int
    train_size: int
    test_size: int
    windows: list[WalkForwardWindow] = field(default_factory=list)
    aggregate_return_pct: float = 0.0
    aggregate_trades: int = 0
    aggregate_win_rate: float = 0.0
    avg_max_dd_pct: float = 0.0
    avg_sharpe: float = 0.0
    robustness_pct: float = 0.0

    def summary(self) -> str:
        lines = [
            "=== Walk-Forward Validation ===",
            f"Symbol:          {self.symbol}",
            f"Timeframe:       {self.timeframe}",
            f"Windows:         {self.num_windows}  (train={self.train_size}, test={self.test_size})",
            f"OOS Return:      {self.aggregate_return_pct:+.2f}%  (compounded)",
            f"OOS Trades:      {self.aggregate_trades}",
            f"OOS Win Rate:    {self.aggregate_win_rate:.1f}%",
            f"Avg Max DD:      {self.avg_max_dd_pct:.2f}%",
            f"Avg Sharpe:      {self.avg_sharpe:.2f}",
            f"Robustness:      {self.robustness_pct:.0f}%  ({sum(1 for w in self.windows if w.test_return_pct > 0)}/{self.num_windows} profitable)",
            "",
            "--- Per-Window ---",
        ]
        for w in self.windows:
            lines.append(
                f"  W{w.window_num}: train {w.train_return_pct:+.2f}% → test {w.test_return_pct:+.2f}%  "
                f"({w.test_trades} trades, WR {w.test_win_rate:.0f}%, DD {w.test_max_dd_pct:.1f}%)  "
                f"[{w.best_params}]"
            )
        return "\n".join(lines)


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
    trailing_stop_count: int = 0
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
        if self.stop_loss_count or self.take_profit_count or self.trailing_stop_count:
            lines.append(f"Stop Loss:       {self.stop_loss_count}")
            lines.append(f"Take Profit:     {self.take_profit_count}")
            lines.append(f"Trailing Stop:   {self.trailing_stop_count}")
        return "\n".join(lines)


# --- Signal functions: (row, prev_row) -> str ---

def signal_rsi_only(row: pd.Series, prev_row: pd.Series | None) -> str:
    return rsi_signal(row["rsi"])


def signal_rsi_macd(row: pd.Series, prev_row: pd.Series | None) -> str:
    if prev_row is None:
        return "hold"
    return composite_signal(row["rsi"], row["macd_diff"], prev_row["macd_diff"])


def signal_bb_volume(row: pd.Series, prev_row: pd.Series | None) -> str:
    return bb_volume_signal(row, prev_row)


def signal_bb_rsi(row: pd.Series, prev_row: pd.Series | None) -> str:
    return bb_rsi_signal(row, prev_row)


def signal_funding(row: pd.Series, prev_row: pd.Series | None) -> str:
    fr = row.get("funding_rate", 0.0)
    return funding_rate_signal(fr)


def signal_bb_vol_funding(row: pd.Series, prev_row: pd.Series | None) -> str:
    fr = row.get("funding_rate", 0.0)
    return bb_volume_funding_signal(row, prev_row, funding_rate=fr)


STRATEGIES: dict[str, Callable] = {
    "rsi": signal_rsi_only,
    "rsi+macd": signal_rsi_macd,
    "bb+vol": signal_bb_volume,
    "bb+rsi+vol": signal_bb_rsi,
    "funding": signal_funding,
    "bb+vol+fr": signal_bb_vol_funding,
}


def _resample_to_regime(
    df: pd.DataFrame,
    signal_tf: str,
    regime_tf: str,
) -> pd.DataFrame:
    """Resample signal-timeframe OHLCV to regime-timeframe for EMA/ADX."""
    tf_ratio = _TF_MS.get(regime_tf, 14_400_000) // _TF_MS.get(signal_tf, 3_600_000)
    if tf_ratio <= 1:
        return df.copy()

    df = df.copy().reset_index(drop=True)
    df["group"] = df.index // tf_ratio
    resampled = df.groupby("group").agg({
        "timestamp": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).reset_index(drop=True)
    return resampled


def _precompute_regime_column(
    df: pd.DataFrame,
    signal_tf: str,
    regime_tf: str,
) -> list[str]:
    """Pre-compute regime classification for each signal candle.

    Computes EMA-200 and ADX once on the resampled dataframe, then
    classifies each row. O(n) instead of O(n^2).
    """
    from ta.trend import ADXIndicator, EMAIndicator

    regime_df = _resample_to_regime(df, signal_tf, regime_tf)
    tf_ratio = _TF_MS.get(regime_tf, 14_400_000) // _TF_MS.get(signal_tf, 3_600_000)
    tf_ratio = max(tf_ratio, 1)

    # Compute indicators once on the full resampled df
    rdf = regime_df.copy()
    rdf["ema200"] = EMAIndicator(close=rdf["close"], window=200).ema_indicator()
    adx_ind = ADXIndicator(high=rdf["high"], low=rdf["low"], close=rdf["close"], window=14)
    rdf["adx"] = adx_ind.adx()

    # Classify each regime candle
    regime_per_candle: list[str] = []
    for _, row in rdf.iterrows():
        ema200 = row["ema200"]
        adx = row["adx"]
        price = row["close"]
        if pd.isna(ema200) or pd.isna(adx):
            regime_per_candle.append("ranging")
        elif adx < ADX_TREND_THRESHOLD:
            regime_per_candle.append("ranging")
        elif price > ema200:
            regime_per_candle.append("uptrend")
        else:
            regime_per_candle.append("downtrend")

    # Map back to signal candles
    regimes: list[str] = []
    for i in range(len(df)):
        regime_idx = min(i // tf_ratio, len(regime_per_candle) - 1)
        regimes.append(regime_per_candle[regime_idx])

    return regimes


def _merge_funding_rates(df: pd.DataFrame, fr_df: pd.DataFrame) -> pd.DataFrame:
    """Merge funding rate data into OHLCV dataframe by nearest timestamp.

    Funding rates are 8-hourly; forward-fill to align with 1h candles.
    """
    if fr_df is None or fr_df.empty:
        df["funding_rate"] = 0.0
        return df
    fr = fr_df[["timestamp", "funding_rate"]].copy()
    fr = fr.sort_values("timestamp")
    df = df.sort_values("timestamp")
    df = pd.merge_asof(df, fr, on="timestamp", direction="backward")
    df["funding_rate"] = df["funding_rate"].fillna(0.0)
    return df


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
    trailing_stop_pct: float = 0.0,
    max_exposure_pct: float = 100.0,
    strategy: str = "rsi",
    sentiment_score: float = 0.0,
    regime_filter: bool = False,
    regime_timeframe: str = "4h",
    df: pd.DataFrame | None = None,
    funding_df: pd.DataFrame | None = None,
) -> BacktestResult:
    if df is None:
        df = fetch_ohlcv(symbol, timeframe, limit)
    df = compute_indicators(df)
    if funding_df is not None:
        df = _merge_funding_rates(df, funding_df)
    elif "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0
    df = df.reset_index(drop=True)

    signal_fn = STRATEGIES[strategy]

    # Pre-compute regime column if enabled
    regime_col: list[str] | None = None
    if regime_filter:
        regime_col = _precompute_regime_column(df, timeframe, regime_timeframe)

    # Sentiment adjusts buy size, not direction
    sent_mult = sentiment_multiplier(sentiment_score)
    effective_buy_fraction = buy_fraction * sent_mult

    cash = initial_cash
    position = 0.0
    avg_entry = 0.0
    high_watermark = 0.0
    trades: list[dict] = []
    equity_curve: list[float] = []
    sig_filter = SignalFilter(buy_cooldown=buy_cooldown)
    prev_row = None
    pending_signal: str | None = None
    sl_count = 0
    tp_count = 0
    ts_count = 0

    for i, row in df.iterrows():
        price = row["close"]

        # Update high watermark
        if position > 0:
            high_watermark = max(high_watermark, price)

        # Stop-loss / take-profit / trailing-stop check
        exit_reason = None
        if position > 0 and avg_entry > 0:
            pnl_pct = (price - avg_entry) / avg_entry * 100
            if stop_loss_pct > 0 and pnl_pct <= -stop_loss_pct:
                exit_reason = "stop_loss"
                sl_count += 1
            elif take_profit_pct > 0 and pnl_pct >= take_profit_pct:
                exit_reason = "take_profit"
                tp_count += 1
            elif trailing_stop_pct > 0 and high_watermark > 0:
                drop_from_peak = (high_watermark - price) / high_watermark * 100
                if drop_from_peak >= trailing_stop_pct:
                    exit_reason = "trailing_stop"
                    ts_count += 1

        if exit_reason:
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
                "reason": exit_reason,
            })
            cash += revenue
            position = 0.0
            avg_entry = 0.0
            high_watermark = 0.0
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
                high_watermark = max(high_watermark, exec_price)
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
                high_watermark = 0.0

        pending_signal = None

        # Compute signal on this candle (will execute on next candle's open)
        equity = cash + position * price
        equity_curve.append(equity)

        raw = signal_fn(row, prev_row)
        signal = sig_filter.filter(raw)
        prev_row = row

        # Regime filter: block buys in downtrend
        if regime_col is not None and signal == "buy":
            if regime_col[i] == "downtrend":
                signal = "hold"

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
        trailing_stop_count=ts_count,
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
    trailing_stops: list[float] | None = None,
    regime_filter: bool = False,
    regime_timeframe: str = "4h",
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
    if trailing_stops is None:
        trailing_stops = [0]

    if df is None:
        df = fetch_ohlcv(symbol, timeframe, limit)

    results: list[BacktestResult] = []
    combos = list(product(rsi_windows, cooldowns, stop_losses, take_profits, trailing_stops))
    total = len(combos)

    for idx, (rsi_w, cd, sl, tp, ts) in enumerate(combos, 1):
        df_ind = compute_indicators(df.copy(), rsi_window=rsi_w)
        r = run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            buy_cooldown=cd,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            trailing_stop_pct=ts,
            strategy=strategy,
            regime_filter=regime_filter,
            regime_timeframe=regime_timeframe,
            df=df_ind,
        )
        r.strategy_name = f"rsi{rsi_w}_cd{cd}_sl{sl}_tp{tp}_ts{ts}"
        results.append(r)

        if idx % 20 == 0 or idx == total:
            print(f"  sweep: {idx}/{total} done")

    results.sort(key=lambda r: r.total_return_pct, reverse=True)
    return results


def _parse_sweep_params(name: str) -> dict:
    """Parse sweep strategy name like 'rsi14_cd12_sl3_tp10_ts5' back to params."""
    import re
    m = re.match(r"rsi(\d+)_cd(\d+)_sl([\d.]+)_tp([\d.]+)(?:_ts([\d.]+))?", name)
    if not m:
        return {}
    params = {
        "rsi_window": int(m.group(1)),
        "buy_cooldown": int(m.group(2)),
        "stop_loss_pct": float(m.group(3)),
        "take_profit_pct": float(m.group(4)),
    }
    if m.group(5) is not None:
        params["trailing_stop_pct"] = float(m.group(5))
    return params


def walk_forward_validation(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    train_size: int = 1000,
    test_size: int = 500,
    step_size: int = 500,
    strategy: str = "rsi+macd",
    regime_filter: bool = False,
    regime_timeframe: str = "4h",
    df: pd.DataFrame | None = None,
) -> WalkForwardResult:
    """Walk-forward validation: optimize on train, evaluate on test, roll forward.

    Splits data into rolling windows of (train_size + test_size), stepping
    forward by step_size each iteration. On each window, runs parameter_sweep
    on the training portion, then evaluates the best parameters on the
    unseen test portion.
    """
    if df is None:
        total_needed = train_size + test_size + step_size * 5
        df = fetch_ohlcv_paginated(symbol, timeframe, total=total_needed)

    df = df.reset_index(drop=True)
    n = len(df)

    windows: list[WalkForwardWindow] = []
    start = 0
    window_num = 0

    while start + train_size + test_size <= n:
        window_num += 1
        train_df = df.iloc[start : start + train_size].reset_index(drop=True)
        test_df = df.iloc[start + train_size : start + train_size + test_size].reset_index(drop=True)

        # Optimize on training data
        sweep_results = parameter_sweep(
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            regime_filter=regime_filter,
            regime_timeframe=regime_timeframe,
            df=train_df,
        )

        best = sweep_results[0]
        params = _parse_sweep_params(best.strategy_name)

        # Evaluate on test data with optimized params
        test_df_ind = compute_indicators(
            test_df.copy(),
            rsi_window=params.get("rsi_window", 14),
        )
        test_result = run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            buy_cooldown=params.get("buy_cooldown", DEFAULT_BUY_COOLDOWN),
            stop_loss_pct=params.get("stop_loss_pct", 0.0),
            take_profit_pct=params.get("take_profit_pct", 0.0),
            trailing_stop_pct=params.get("trailing_stop_pct", 0.0),
            strategy=strategy,
            regime_filter=regime_filter,
            regime_timeframe=regime_timeframe,
            df=test_df_ind,
        )

        wf_window = WalkForwardWindow(
            window_num=window_num,
            train_start=str(train_df.iloc[0]["timestamp"]),
            train_end=str(train_df.iloc[-1]["timestamp"]),
            test_start=str(test_df.iloc[0]["timestamp"]),
            test_end=str(test_df.iloc[-1]["timestamp"]),
            best_params=best.strategy_name,
            train_return_pct=best.total_return_pct,
            test_return_pct=test_result.total_return_pct,
            test_trades=test_result.num_trades,
            test_win_rate=test_result.win_rate,
            test_max_dd_pct=test_result.max_drawdown_pct,
            test_sharpe=test_result.sharpe_ratio,
        )
        windows.append(wf_window)

        if window_num % 2 == 0 or start + train_size + test_size + step_size > n:
            print(f"  walk-forward: window {window_num} done "
                  f"(train {best.total_return_pct:+.2f}% → test {test_result.total_return_pct:+.2f}%)")

        start += step_size

    # Aggregate out-of-sample metrics
    if not windows:
        return WalkForwardResult(
            symbol=symbol,
            timeframe=timeframe,
            num_windows=0,
            train_size=train_size,
            test_size=test_size,
        )

    # Compounded OOS return
    compounded = 1.0
    for w in windows:
        compounded *= (1 + w.test_return_pct / 100)
    aggregate_return_pct = (compounded - 1) * 100

    total_trades = sum(w.test_trades for w in windows)

    # Weighted average win rate by trade count
    if total_trades > 0:
        agg_win_rate = sum(w.test_win_rate * w.test_trades for w in windows) / total_trades
    else:
        agg_win_rate = 0.0

    avg_dd = sum(w.test_max_dd_pct for w in windows) / len(windows)
    avg_sharpe = sum(w.test_sharpe for w in windows) / len(windows)
    profitable_windows = sum(1 for w in windows if w.test_return_pct > 0)
    robustness = (profitable_windows / len(windows)) * 100

    return WalkForwardResult(
        symbol=symbol,
        timeframe=timeframe,
        num_windows=len(windows),
        train_size=train_size,
        test_size=test_size,
        windows=windows,
        aggregate_return_pct=aggregate_return_pct,
        aggregate_trades=total_trades,
        aggregate_win_rate=agg_win_rate,
        avg_max_dd_pct=avg_dd,
        avg_sharpe=avg_sharpe,
        robustness_pct=robustness,
    )


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

    def _parse_cli_args(argv: list[str], exclude_flags: list[str] | None = None) -> tuple[str, int, str, bool]:
        """Parse common CLI args: symbol, limit, timeframe, regime flag."""
        exclude = set(exclude_flags or [])
        args = [a for a in argv[1:] if a not in exclude]
        symbol = "BTC/USDT"
        limit = 1000
        timeframe = "1h"
        use_regime = "--regime" in argv
        for i, a in enumerate(args):
            if a not in ("--limit", "--tf") and not a.startswith("--") and "/" in a:
                symbol = a
            if a == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
            if a == "--tf" and i + 1 < len(args):
                timeframe = args[i + 1]
        return symbol, limit, timeframe, use_regime

    if "--walk-forward" in sys.argv:
        # python -m trading_agent.backtest --walk-forward BTC/USDT --limit 5000 --regime
        symbol, limit, timeframe, use_regime = _parse_cli_args(
            sys.argv, ["--walk-forward", "--regime"])
        train_size = 1000
        test_size = 500
        step_size = 500
        for i, a in enumerate(sys.argv):
            if a == "--train" and i + 1 < len(sys.argv):
                train_size = int(sys.argv[i + 1])
            if a == "--test" and i + 1 < len(sys.argv):
                test_size = int(sys.argv[i + 1])
            if a == "--step" and i + 1 < len(sys.argv):
                step_size = int(sys.argv[i + 1])

        regime_label = " +regime" if use_regime else ""
        print(f"Walk-Forward: {symbol} ({timeframe}{regime_label})")
        print(f"  train={train_size}, test={test_size}, step={step_size}")
        print(f"Fetching {limit} candles...")

        if limit > 1000:
            df = fetch_ohlcv_paginated(symbol, timeframe, total=limit)
        else:
            df = fetch_ohlcv(symbol, timeframe, limit)
        print(f"Got {len(df)} candles: {df.iloc[0]['timestamp']} → {df.iloc[-1]['timestamp']}\n")

        wf_result = walk_forward_validation(
            symbol=symbol,
            timeframe=timeframe,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            regime_filter=use_regime,
            df=df,
        )
        print()
        print(wf_result.summary())

    elif "--sweep" in sys.argv:
        # python -m trading_agent.backtest --sweep BTC/USDT --limit 1000 --regime
        symbol, limit, timeframe, use_regime = _parse_cli_args(
            sys.argv, ["--sweep", "--regime"])

        regime_label = " +regime" if use_regime else ""
        print(f"Fetching {limit} candles for {symbol} ({timeframe}{regime_label})...")
        if limit > 1000:
            df = fetch_ohlcv_paginated(symbol, timeframe, total=limit)
        else:
            df = fetch_ohlcv(symbol, timeframe, limit)
        print(f"Got {len(df)} candles: {df.iloc[0]['timestamp']} → {df.iloc[-1]['timestamp']}")

        results = parameter_sweep(
            symbol, timeframe, limit,
            regime_filter=use_regime, df=df,
        )
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
