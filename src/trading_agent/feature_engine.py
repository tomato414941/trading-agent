"""Feature engine: compute ML features from raw signals + OHLCV data."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    zscore_window: int = 30
    momentum_window: int = 7
    roc_periods: list[int] = field(default_factory=lambda: [1, 3, 7])
    volatility_window: int = 20
    up_threshold: float = 0.3  # % price change to classify as "up"
    cross_pairs: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            ("btc_close", "dxy", "btc_dxy"),
            ("btc_close", "gold", "btc_gold"),
            ("btc_close", "sp500", "btc_sp500"),
            ("tsy_yield_10y", "tsy_yield_2y", "yield_curve"),
            ("binance_btc_oi", "btc_close", "oi_price"),
        ]
    )
    momentum_pairs: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            ("btc_close", "sp500", "btc_vs_sp500"),
            ("btc_close", "dxy", "btc_vs_dxy"),
            ("binance_btc_oi", "btc_close", "oi_vs_price"),
        ]
    )


class CryptoFeatureEngine:
    """Transforms raw signal values into ML features."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self._history: dict[str, list[float]] = {}

    def compute(self, latest_values: dict[str, float]) -> dict[str, float]:
        features: dict[str, float] = {}

        # Filter NaN/inf inputs
        latest_values = {
            k: v for k, v in latest_values.items()
            if isinstance(v, (int, float)) and math.isfinite(v)
        }
        if not latest_values:
            return features

        for name, value in latest_values.items():
            if name not in self._history:
                self._history[name] = []
            self._history[name].append(value)
            history = self._history[name]

            features[f"{name}_raw"] = value

            # Z-score
            w = self.config.zscore_window
            if len(history) >= max(w, 3):
                recent = history[-w:]
                mean = np.mean(recent)
                std = np.std(recent)
                features[f"{name}_zscore"] = (
                    (value - mean) / std if std > 1e-10 else 0.0
                )

            # Momentum
            m = self.config.momentum_window
            if len(history) >= m:
                ma = np.mean(history[-m:])
                features[f"{name}_momentum"] = (
                    (value - ma) / abs(ma) if abs(ma) > 1e-10 else 0.0
                )

            # Rate of change
            for period in self.config.roc_periods:
                if len(history) > period:
                    prev = history[-(period + 1)]
                    features[f"{name}_roc{period}"] = (
                        (value - prev) / abs(prev) if abs(prev) > 1e-10 else 0.0
                    )

            # Volatility
            vw = self.config.volatility_window
            if len(history) >= vw:
                vol = np.std(history[-vw:])
                features[f"{name}_vol"] = vol
                if len(history) >= vw * 2:
                    prev_vol = np.std(history[-(vw * 2) : -vw])
                    features[f"{name}_vol_change"] = (
                        (vol - prev_vol) / prev_vol if prev_vol > 1e-10 else 0.0
                    )

        # Cross-signal features
        for a, b, name in self.config.cross_pairs:
            va = latest_values.get(a)
            vb = latest_values.get(b)
            if va is not None and vb is not None and abs(vb) > 1e-10:
                features[f"{name}_ratio"] = va / vb

        # Momentum divergence
        m = self.config.momentum_window
        for a, b, name in self.config.momentum_pairs:
            ha = self._history.get(a)
            hb = self._history.get(b)
            if not ha or not hb or len(ha) < m or len(hb) < m:
                continue
            ma_a = np.mean(ha[-m:])
            ma_b = np.mean(hb[-m:])
            mom_a = (ha[-1] - ma_a) / abs(ma_a) if abs(ma_a) > 1e-10 else 0.0
            mom_b = (hb[-1] - ma_b) / abs(ma_b) if abs(ma_b) > 1e-10 else 0.0
            features[f"{name}_div"] = mom_a - mom_b

        # Sanitize outputs
        features = {k: v for k, v in features.items() if math.isfinite(v)}
        return features

    def compute_target(self, prev_price: float, curr_price: float) -> int:
        if abs(prev_price) < 1e-10:
            return 0
        pct = (curr_price - prev_price) / prev_price * 100
        return 1 if pct > self.config.up_threshold else 0

    def reset(self) -> None:
        self._history.clear()

    @property
    def history_length(self) -> int:
        if not self._history:
            return 0
        return max(len(v) for v in self._history.values())
