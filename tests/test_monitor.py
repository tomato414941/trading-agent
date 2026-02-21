from unittest.mock import patch

from trading_agent.config import RiskConfig
from trading_agent.monitor import check_risk_all_symbols
from trading_agent.state import SharedState
from trading_agent.portfolio import Portfolio


class TestCheckRisk:
    def test_stop_loss_triggers_sell(self):
        portfolio = Portfolio(cash=9000.0)
        portfolio.buy("BTC/USDT", 50000.0)

        state = SharedState(portfolio=portfolio)
        risk = RiskConfig(stop_loss_pct=5.0)

        with patch("trading_agent.monitor.fetch_ticker_price", return_value=47000.0):
            with patch("trading_agent.monitor.log_trade"):
                with patch("trading_agent.monitor.notify_trade"):
                    check_risk_all_symbols(state, ["BTC/USDT"], risk)

        assert state.portfolio.get_position("BTC/USDT").qty == 0.0
        # Sold at loss (47k vs 50k entry), but got cash back from position
        assert state.portfolio.cash > 8000.0

    def test_no_trigger_when_price_stable(self):
        portfolio = Portfolio(cash=9000.0)
        portfolio.buy("BTC/USDT", 50000.0)
        original_qty = portfolio.get_position("BTC/USDT").qty

        state = SharedState(portfolio=portfolio)
        risk = RiskConfig(stop_loss_pct=5.0)

        with patch("trading_agent.monitor.fetch_ticker_price", return_value=49000.0):
            check_risk_all_symbols(state, ["BTC/USDT"], risk)

        assert state.portfolio.get_position("BTC/USDT").qty == original_qty

    def test_take_profit_triggers_sell(self):
        portfolio = Portfolio(cash=9000.0)
        portfolio.buy("BTC/USDT", 50000.0)

        state = SharedState(portfolio=portfolio)
        risk = RiskConfig(take_profit_pct=10.0)

        with patch("trading_agent.monitor.fetch_ticker_price", return_value=56000.0):
            with patch("trading_agent.monitor.log_trade"):
                with patch("trading_agent.monitor.notify_trade"):
                    check_risk_all_symbols(state, ["BTC/USDT"], risk)

        assert state.portfolio.get_position("BTC/USDT").qty == 0.0

    def test_no_position_skips(self):
        state = SharedState(portfolio=Portfolio(cash=10000.0))
        risk = RiskConfig()

        with patch("trading_agent.monitor.fetch_ticker_price", return_value=50000.0):
            check_risk_all_symbols(state, ["BTC/USDT"], risk)

        assert state.last_prices["BTC/USDT"] == 50000.0

    def test_price_fetch_failure_skips(self):
        portfolio = Portfolio(cash=9000.0)
        portfolio.buy("BTC/USDT", 50000.0)

        state = SharedState(portfolio=portfolio)
        risk = RiskConfig(stop_loss_pct=5.0)

        with patch("trading_agent.monitor.fetch_ticker_price", side_effect=Exception("network")):
            check_risk_all_symbols(state, ["BTC/USDT"], risk)

        assert state.portfolio.get_position("BTC/USDT").qty > 0
