"""Tests for exchange factory module."""

from unittest.mock import patch

import pytest

from trading_agent.exchange import load_secrets, create_futures_exchange, create_spot_exchange


class TestLoadSecrets:
    def test_missing_file(self, tmp_path):
        with patch("trading_agent.exchange.Path.home", return_value=tmp_path):
            result = load_secrets()
        assert result == {}

    def test_parses_export_format(self, tmp_path):
        secrets_dir = tmp_path / ".secrets"
        secrets_dir.mkdir()
        (secrets_dir / "binance").write_text(
            "export BINANCE_API_KEY=test_key\n"
            "export BINANCE_SECRET_KEY=test_secret\n"
        )
        with patch("trading_agent.exchange.Path.home", return_value=tmp_path):
            result = load_secrets()
        assert result["BINANCE_API_KEY"] == "test_key"
        assert result["BINANCE_SECRET_KEY"] == "test_secret"

    def test_parses_plain_format(self, tmp_path):
        secrets_dir = tmp_path / ".secrets"
        secrets_dir.mkdir()
        (secrets_dir / "binance").write_text(
            "BINANCE_API_KEY=test_key\n"
            "BINANCE_SECRET_KEY=test_secret\n"
        )
        with patch("trading_agent.exchange.Path.home", return_value=tmp_path):
            result = load_secrets()
        assert result["BINANCE_API_KEY"] == "test_key"
        assert result["BINANCE_SECRET_KEY"] == "test_secret"

    def test_skips_comments_and_empty_lines(self, tmp_path):
        secrets_dir = tmp_path / ".secrets"
        secrets_dir.mkdir()
        (secrets_dir / "binance").write_text(
            "# comment\n\nBINANCE_API_KEY=key\n"
        )
        with patch("trading_agent.exchange.Path.home", return_value=tmp_path):
            result = load_secrets()
        assert result == {"BINANCE_API_KEY": "key"}

    def test_strips_quotes(self, tmp_path):
        secrets_dir = tmp_path / ".secrets"
        secrets_dir.mkdir()
        (secrets_dir / "binance").write_text(
            "BINANCE_API_KEY='quoted_key'\n"
            "BINANCE_SECRET_KEY=\"double_quoted\"\n"
        )
        with patch("trading_agent.exchange.Path.home", return_value=tmp_path):
            result = load_secrets()
        assert result["BINANCE_API_KEY"] == "quoted_key"
        assert result["BINANCE_SECRET_KEY"] == "double_quoted"


class TestCreateExchange:
    def test_raises_without_credentials(self, tmp_path):
        with (
            patch("trading_agent.exchange.Path.home", return_value=tmp_path),
            patch.dict("os.environ", {}, clear=True),
        ):
            with pytest.raises(ValueError, match="credentials not found"):
                create_futures_exchange()

    def test_futures_exchange_type(self, tmp_path):
        secrets_dir = tmp_path / ".secrets"
        secrets_dir.mkdir()
        (secrets_dir / "binance").write_text(
            "BINANCE_API_KEY=test\nBINANCE_SECRET_KEY=test\n"
        )
        with patch("trading_agent.exchange.Path.home", return_value=tmp_path):
            ex = create_futures_exchange(testnet=True)
        assert ex.options.get("defaultType") == "future"

    def test_spot_exchange_type(self, tmp_path):
        secrets_dir = tmp_path / ".secrets"
        secrets_dir.mkdir()
        (secrets_dir / "binance").write_text(
            "BINANCE_API_KEY=test\nBINANCE_SECRET_KEY=test\n"
        )
        with patch("trading_agent.exchange.Path.home", return_value=tmp_path):
            ex = create_spot_exchange(testnet=True)
        assert ex.options.get("defaultType") == "spot"
