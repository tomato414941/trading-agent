import pandas as pd
import pytest


@pytest.fixture
def make_ohlcv_df():
    """Factory: create OHLCV DataFrame with specified close prices."""
    def _make(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
        n = len(closes)
        if opens is None:
            opens = closes
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": opens,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        })
    return _make
