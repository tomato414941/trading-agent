"""Funding rate arbitrage: analysis and backtest engine.

Strategy: Spot long + Futures short = delta-neutral.
Collect funding rate payments every 8 hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_agent.config import ArbitrageConfig, DEFAULT_ARBITRAGE
from trading_agent.fetcher import fetch_funding_rate_history, fetch_ohlcv_paginated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_negative_streaks(rates: pd.Series) -> tuple[int, float]:
    """Return (max_streak, avg_streak) of consecutive negative FR periods."""
    streaks: list[int] = []
    current = 0
    for r in rates:
        if r < 0:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    if not streaks:
        return 0, 0.0
    return max(streaks), sum(streaks) / len(streaks)


# ---------------------------------------------------------------------------
# Phase 1: FR Analysis
# ---------------------------------------------------------------------------

@dataclass
class FRAnalysisResult:
    symbol: str
    period_start: str
    period_end: str
    total_periods: int
    mean_fr: float
    median_fr: float
    positive_pct: float
    std_fr: float
    max_fr: float
    min_fr: float
    annualized_yield_pct: float
    round_trip_fee_pct: float
    net_annualized_yield_pct: float
    max_negative_streak: int
    negative_streak_avg: float

    def summary(self) -> str:
        lines = [
            f"=== FR Analysis: {self.symbol} ===",
            f"Period:            {self.period_start} -> {self.period_end}",
            f"Data Points:       {self.total_periods} (8h periods)",
            f"Mean FR:           {self.mean_fr:.6f} ({self.mean_fr * 100:.4f}%)",
            f"Median FR:         {self.median_fr:.6f}",
            f"Positive Rate:     {self.positive_pct:.1f}%",
            f"Annualized Yield:  {self.annualized_yield_pct:+.2f}%",
            f"Round-Trip Fee:    {self.round_trip_fee_pct:.2f}%",
            f"Net Ann. Yield:    {self.net_annualized_yield_pct:+.2f}%",
            f"Max Neg Streak:    {self.max_negative_streak} periods",
            f"Avg Neg Streak:    {self.negative_streak_avg:.1f} periods",
            f"FR Std Dev:        {self.std_fr:.6f}",
        ]
        return "\n".join(lines)


def analyze_funding_rate(
    symbol: str = "BTC/USDT",
    periods: int = 1000,
    config: ArbitrageConfig = DEFAULT_ARBITRAGE,
    fr_df: pd.DataFrame | None = None,
) -> FRAnalysisResult:
    """Fetch FR history and compute yield/risk statistics."""
    if fr_df is None:
        fr_df = fetch_funding_rate_history(symbol, total=periods)

    rates = fr_df["funding_rate"]
    mean_fr = float(rates.mean())
    median_fr = float(rates.median())
    positive_pct = float((rates > 0).mean() * 100)
    std_fr = float(rates.std())
    max_fr = float(rates.max())
    min_fr = float(rates.min())

    # 3 settlements per day, 365 days
    annualized_yield_pct = mean_fr * 3 * 365 * 100

    round_trip_fee_pct = (config.spot_fee_rate + config.futures_fee_rate) * 2 * 100

    # Assume ~12 round trips/year (monthly rebalance) for fee amortization
    net_annualized_yield_pct = annualized_yield_pct - round_trip_fee_pct * 12

    max_neg, avg_neg = _compute_negative_streaks(rates)

    ts = fr_df["timestamp"]
    return FRAnalysisResult(
        symbol=symbol,
        period_start=str(ts.iloc[0]),
        period_end=str(ts.iloc[-1]),
        total_periods=len(fr_df),
        mean_fr=mean_fr,
        median_fr=median_fr,
        positive_pct=positive_pct,
        std_fr=std_fr,
        max_fr=max_fr,
        min_fr=min_fr,
        annualized_yield_pct=annualized_yield_pct,
        round_trip_fee_pct=round_trip_fee_pct,
        net_annualized_yield_pct=net_annualized_yield_pct,
        max_negative_streak=max_neg,
        negative_streak_avg=avg_neg,
    )


def compare_fr_yields(
    symbols: list[str] | None = None,
    periods: int = 1000,
    config: ArbitrageConfig = DEFAULT_ARBITRAGE,
) -> str:
    """Analyze multiple symbols and return formatted comparison table."""
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    results = [analyze_funding_rate(s, periods, config) for s in symbols]
    header = (
        f"{'Symbol':<12} {'Ann.Yield':>10} {'Net Yield':>10} "
        f"{'Pos%':>6} {'MeanFR':>10} {'MaxNeg':>7}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        lines.append(
            f"{r.symbol:<12} {r.annualized_yield_pct:>+9.2f}% "
            f"{r.net_annualized_yield_pct:>+9.2f}% {r.positive_pct:>5.1f}% "
            f"{r.mean_fr:>10.6f} {r.max_negative_streak:>7}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 2: Backtest Engine
# ---------------------------------------------------------------------------

@dataclass
class ArbitragePosition:
    """A single spot-long + futures-short pair."""
    symbol: str
    entry_time: pd.Timestamp
    spot_qty: float
    spot_entry_price: float
    futures_entry_price: float
    accumulated_fr: float = 0.0
    entry_fees: float = 0.0
    hold_periods: int = 0


@dataclass
class ArbitrageBacktestResult:
    symbol: str
    initial_cash: float
    final_value: float
    total_return_pct: float
    annualized_return_pct: float
    total_fr_collected: float
    total_fees_paid: float
    num_round_trips: int
    avg_hold_periods: float
    max_drawdown_pct: float
    period_start: str
    period_end: str
    equity_curve: list[float] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== FR Arbitrage Backtest: {self.symbol} ===",
            f"Period:            {self.period_start} -> {self.period_end}",
            f"Initial:           ${self.initial_cash:,.2f}",
            f"Final:             ${self.final_value:,.2f}",
            f"Return:            {self.total_return_pct:+.2f}%",
            f"Annualized:        {self.annualized_return_pct:+.2f}%",
            f"FR Collected:      ${self.total_fr_collected:,.2f}",
            f"Fees Paid:         ${self.total_fees_paid:,.2f}",
            f"Round Trips:       {self.num_round_trips}",
            f"Avg Hold:          {self.avg_hold_periods:.1f} periods (8h)",
            f"Max Drawdown:      {self.max_drawdown_pct:.2f}%",
        ]
        return "\n".join(lines)


def _merge_price_to_fr(
    fr_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge spot prices onto FR timestamps via merge_asof."""
    fr = fr_df.copy()
    pr = price_df[["timestamp", "close"]].copy()
    pr = pr.rename(columns={"close": "spot_price"})
    fr = fr.sort_values("timestamp")
    pr = pr.sort_values("timestamp")
    merged = pd.merge_asof(fr, pr, on="timestamp", direction="nearest")
    return merged


def run_arbitrage_backtest(
    symbol: str = "BTC/USDT",
    periods: int = 1000,
    initial_cash: float = 10_000.0,
    config: ArbitrageConfig = DEFAULT_ARBITRAGE,
    fr_df: pd.DataFrame | None = None,
    price_df: pd.DataFrame | None = None,
) -> ArbitrageBacktestResult:
    """Simulate FR arbitrage: spot long + futures short, collect FR every 8h."""
    if fr_df is None:
        fr_df = fetch_funding_rate_history(symbol, total=periods)
    if price_df is None:
        price_df = fetch_ohlcv_paginated(symbol, "1h", total=periods * 8)

    data = _merge_price_to_fr(fr_df, price_df)

    cash = initial_cash
    position: ArbitragePosition | None = None
    total_fr = 0.0
    total_fees = 0.0
    trades: list[dict] = []
    equity_curve: list[float] = []
    low_fr_count = 0
    peak_equity = initial_cash

    max_dd = 0.0
    hold_periods_list: list[int] = []

    for _, row in data.iterrows():
        fr = row["funding_rate"]
        price = row["spot_price"]
        ts = row["timestamp"]

        if pd.isna(price) or pd.isna(fr):
            equity_curve.append(cash if position is None else cash + position.spot_qty * price)
            continue

        if position is not None:
            # Collect FR payment
            fr_payment = position.spot_qty * price * fr
            cash += fr_payment
            total_fr += fr_payment
            position.accumulated_fr += fr_payment
            position.hold_periods += 1

            # Check exit condition
            if fr < config.exit_fr_threshold:
                low_fr_count += 1
            else:
                low_fr_count = 0

            if low_fr_count >= config.exit_consecutive_periods:
                # Close: sell spot + cover futures short
                spot_revenue = position.spot_qty * price
                exit_fees = (
                    spot_revenue * config.spot_fee_rate
                    + spot_revenue * config.futures_fee_rate
                )
                cash += spot_revenue - exit_fees
                total_fees += exit_fees

                pnl = (
                    spot_revenue
                    - position.spot_qty * position.spot_entry_price
                    + position.accumulated_fr
                    - position.entry_fees
                    - exit_fees
                )

                trades.append({
                    "entry_time": str(position.entry_time),
                    "exit_time": str(ts),
                    "hold_periods": position.hold_periods,
                    "fr_collected": position.accumulated_fr,
                    "pnl": pnl,
                })

                hold_periods_list.append(position.hold_periods)
                position = None
                low_fr_count = 0

        elif fr > config.entry_fr_threshold:
            # Open delta-neutral position
            invest = cash * config.position_fraction
            entry_fees = (
                invest * config.spot_fee_rate
                + invest * config.futures_fee_rate
            )
            spot_cost = invest - entry_fees
            spot_qty = spot_cost / price

            position = ArbitragePosition(
                symbol=symbol,
                entry_time=ts,
                spot_qty=spot_qty,
                spot_entry_price=price,
                futures_entry_price=price,
                entry_fees=entry_fees,
            )
            cash -= invest
            total_fees += entry_fees

        # Track equity
        equity = cash
        if position is not None:
            equity += position.spot_qty * price
        equity_curve.append(equity)

        # Max drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Force-close any open position at end
    if position is not None and len(data) > 0:
        last_price = data.iloc[-1]["spot_price"]
        if not pd.isna(last_price):
            spot_revenue = position.spot_qty * last_price
            exit_fees = (
                spot_revenue * config.spot_fee_rate
                + spot_revenue * config.futures_fee_rate
            )
            cash += spot_revenue - exit_fees
            total_fees += exit_fees

            pnl = (
                spot_revenue
                - position.spot_qty * position.spot_entry_price
                + position.accumulated_fr
                - position.entry_fees
                - exit_fees
            )
            trades.append({
                "entry_time": str(position.entry_time),
                "exit_time": str(data.iloc[-1]["timestamp"]),
                "hold_periods": position.hold_periods,
                "fr_collected": position.accumulated_fr,
                "pnl": pnl,
            })
            hold_periods_list.append(position.hold_periods)
            position = None

    final_value = cash
    total_return_pct = (final_value - initial_cash) / initial_cash * 100

    # Annualize: periods are 8h each
    total_days = len(data) * 8 / 24 if len(data) > 0 else 1
    total_years = total_days / 365
    if total_years > 0 and final_value > 0:
        annualized = ((final_value / initial_cash) ** (1 / total_years) - 1) * 100
    else:
        annualized = 0.0

    avg_hold = sum(hold_periods_list) / len(hold_periods_list) if hold_periods_list else 0.0

    ts_col = data["timestamp"]
    return ArbitrageBacktestResult(
        symbol=symbol,
        initial_cash=initial_cash,
        final_value=final_value,
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized,
        total_fr_collected=total_fr,
        total_fees_paid=total_fees,
        num_round_trips=len(trades),
        avg_hold_periods=avg_hold,
        max_drawdown_pct=max_dd,
        period_start=str(ts_col.iloc[0]) if len(ts_col) > 0 else "",
        period_end=str(ts_col.iloc[-1]) if len(ts_col) > 0 else "",
        equity_curve=equity_curve,
        trades=trades,
    )


# ---------------------------------------------------------------------------
# Parameter Sweep
# ---------------------------------------------------------------------------

def arbitrage_parameter_sweep(
    symbol: str = "BTC/USDT",
    periods: int = 1000,
    entry_thresholds: list[float] | None = None,
    exit_thresholds: list[float] | None = None,
    exit_periods: list[int] | None = None,
    fr_df: pd.DataFrame | None = None,
    price_df: pd.DataFrame | None = None,
) -> list[ArbitrageBacktestResult]:
    """Sweep entry/exit parameters to find optimal configuration."""
    if entry_thresholds is None:
        entry_thresholds = [0.0001, 0.0003, 0.0005, 0.001]
    if exit_thresholds is None:
        exit_thresholds = [0.0, 0.0001, 0.0003]
    if exit_periods is None:
        exit_periods = [1, 3, 5]

    if fr_df is None:
        fr_df = fetch_funding_rate_history(symbol, total=periods)
    if price_df is None:
        price_df = fetch_ohlcv_paginated(symbol, "1h", total=periods * 8)

    results: list[ArbitrageBacktestResult] = []
    combos = [
        (et, xt, ep)
        for et in entry_thresholds
        for xt in exit_thresholds
        for ep in exit_periods
        if et > xt  # entry must be stricter than exit
    ]

    total = len(combos)
    for i, (et, xt, ep) in enumerate(combos, 1):
        cfg = ArbitrageConfig(
            entry_fr_threshold=et,
            exit_fr_threshold=xt,
            exit_consecutive_periods=ep,
        )
        result = run_arbitrage_backtest(
            symbol=symbol,
            periods=periods,
            config=cfg,
            fr_df=fr_df,
            price_df=price_df,
        )
        results.append(result)
        if i % 10 == 0 or i == total:
            print(f"  sweep: {i}/{total} done", flush=True)

    results.sort(key=lambda r: r.annualized_return_pct, reverse=True)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    periods = 1000

    # Parse --periods
    for i, a in enumerate(args):
        if a == "--periods" and i + 1 < len(args):
            periods = int(args[i + 1])

    # Parse symbol (first non-flag arg)
    symbol = "BTC/USDT"
    for a in args:
        if not a.startswith("--") and "/" in a:
            symbol = a
            break

    if "--analyze" in args:
        result = analyze_funding_rate(symbol, periods)
        print(result.summary())

    elif "--compare" in args:
        print(compare_fr_yields(periods=periods))

    elif "--sweep" in args:
        print(f"Sweeping {symbol} with {periods} periods...")
        results = arbitrage_parameter_sweep(symbol, periods=periods)
        print("\nTop 5 configurations:")
        for r in results[:5]:
            print(
                f"  Return: {r.annualized_return_pct:+.2f}% "
                f"| FR: ${r.total_fr_collected:.2f} "
                f"| Fees: ${r.total_fees_paid:.2f} "
                f"| Trips: {r.num_round_trips} "
                f"| AvgHold: {r.avg_hold_periods:.0f}"
            )

    else:
        print(f"Backtesting {symbol} with {periods} periods...")
        result = run_arbitrage_backtest(symbol, periods=periods)
        print(result.summary())
        if result.trades:
            print(f"\nTrades ({len(result.trades)}):")
            for t in result.trades:
                print(
                    f"  {t['entry_time']} -> {t['exit_time']} "
                    f"| {t['hold_periods']} periods "
                    f"| FR: ${t['fr_collected']:.2f} "
                    f"| PnL: ${t['pnl']:+.2f}"
                )
