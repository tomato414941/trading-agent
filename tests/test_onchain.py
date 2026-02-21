"""Tests for on-chain data fetching and merging."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from trading_agent.fetcher import (
    fetch_onchain_metric,
    _bg_to_dataframe,
    merge_onchain_daily,
    ONCHAIN_METRICS,
)


class TestBGToDataframe:
    def test_basic_conversion(self):
        data = [
            {"d": "2024-01-01", "unixTs": "1704067200", "sthSopr": "0.98"},
            {"d": "2024-01-02", "unixTs": "1704153600", "sthSopr": "1.02"},
        ]
        df = _bg_to_dataframe(data, "sth_sopr")
        assert len(df) == 2
        assert list(df.columns) == ["date", "value"]
        assert df.iloc[0]["value"] == pytest.approx(0.98)
        assert df.iloc[1]["value"] == pytest.approx(1.02)

    def test_empty_data(self):
        df = _bg_to_dataframe([], "sth_sopr")
        assert len(df) == 0
        assert "date" in df.columns
        assert "value" in df.columns

    def test_skips_invalid_values(self):
        data = [
            {"d": "2024-01-01", "unixTs": "1704067200", "val": "0.98"},
            {"d": "2024-01-02", "unixTs": "1704153600", "val": "N/A"},
            {"d": "2024-01-03", "unixTs": "1704240000", "val": "1.05"},
        ]
        df = _bg_to_dataframe(data, "test")
        assert len(df) == 2

    def test_sorted_by_date(self):
        data = [
            {"d": "2024-01-03", "unixTs": "1704240000", "val": "1.0"},
            {"d": "2024-01-01", "unixTs": "1704067200", "val": "0.9"},
        ]
        df = _bg_to_dataframe(data, "test")
        assert df.iloc[0]["value"] == pytest.approx(0.9)
        assert df.iloc[1]["value"] == pytest.approx(1.0)


class TestFetchOnchainMetric:
    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            fetch_onchain_metric("nonexistent_metric")

    @patch("trading_agent.fetcher.requests.get")
    def test_fetches_and_caches(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"d": "2024-01-01", "unixTs": "1704067200", "sthSopr": "0.98"},
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch("trading_agent.fetcher._BG_CACHE_DIR", tmp_path):
            df = fetch_onchain_metric("sth_sopr", max_age_hours=12)
            assert len(df) == 1
            assert mock_get.call_count == 1

            # Second call should use cache
            df2 = fetch_onchain_metric("sth_sopr", max_age_hours=12)
            assert len(df2) == 1
            assert mock_get.call_count == 1  # No new API call

    @patch("trading_agent.fetcher.requests.get")
    def test_cache_expired(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"d": "2024-01-01", "unixTs": "1704067200", "sthSopr": "0.98"},
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch("trading_agent.fetcher._BG_CACHE_DIR", tmp_path):
            df = fetch_onchain_metric("sth_sopr", max_age_hours=0)
            assert mock_get.call_count == 1

            # max_age_hours=0 forces refetch
            df2 = fetch_onchain_metric("sth_sopr", max_age_hours=0)
            assert mock_get.call_count == 2


class TestMergeOnchainDaily:
    def test_merge_into_hourly(self):
        # Hourly OHLCV
        timestamps = pd.date_range("2024-01-01", periods=48, freq="h")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "close": range(48),
        })

        # Daily on-chain
        oc = {
            "sth_sopr": pd.DataFrame({
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "value": [0.95, 1.02],
            }),
        }

        merged = merge_onchain_daily(df, oc)
        assert "sth_sopr" in merged.columns
        # First 24 rows should have day 1 value
        assert merged.iloc[0]["sth_sopr"] == pytest.approx(0.95)
        assert merged.iloc[23]["sth_sopr"] == pytest.approx(0.95)
        # Next 24 rows should have day 2 value
        assert merged.iloc[24]["sth_sopr"] == pytest.approx(1.02)

    def test_empty_onchain(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
            "close": range(24),
        })
        merged = merge_onchain_daily(df, {})
        assert len(merged) == 24

    def test_forward_fill(self):
        timestamps = pd.date_range("2024-01-01", periods=72, freq="h")
        df = pd.DataFrame({"timestamp": timestamps, "close": range(72)})

        # Only 1 day of on-chain data for 3 days of OHLCV
        oc = {
            "sth_sopr": pd.DataFrame({
                "date": pd.to_datetime(["2024-01-01"]),
                "value": [0.95],
            }),
        }
        merged = merge_onchain_daily(df, oc)
        # Day 2 and 3 should forward-fill from day 1
        assert merged.iloc[47]["sth_sopr"] == pytest.approx(0.95)
