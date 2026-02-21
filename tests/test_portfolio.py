import pytest
from trading_agent.config import RiskConfig
from trading_agent.portfolio import Portfolio


class TestPortfolioBuy:
    def test_basic_buy(self):
        p = Portfolio(cash=10000.0)
        trade = p.buy("BTC/USDT", 50000.0)
        assert trade is not None
        assert trade["side"] == "buy"
        assert trade["cost"] == pytest.approx(1000.0)
        assert p.cash == pytest.approx(9000.0)

    def test_buy_updates_position(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)
        pos = p.get_position("BTC/USDT")
        assert pos.qty == pytest.approx(0.02)
        assert pos.entry_price == pytest.approx(50000.0)

    def test_buy_insufficient_cash(self):
        p = Portfolio(cash=0.5)
        assert p.buy("BTC/USDT", 50000.0) is None

    def test_weighted_average_cost_basis(self):
        p = Portfolio(cash=20000.0)
        p.buy("BTC/USDT", 50000.0, fraction=0.25)  # 5000 USD -> 0.1 BTC @ 50k
        p.buy("BTC/USDT", 40000.0, fraction=1/3)    # 5000 USD -> 0.125 BTC @ 40k
        pos = p.get_position("BTC/USDT")
        expected_qty = 0.1 + 0.125
        expected_avg = (0.1 * 50000 + 0.125 * 40000) / expected_qty
        assert pos.qty == pytest.approx(expected_qty, rel=0.01)
        assert pos.entry_price == pytest.approx(expected_avg, rel=0.01)


class TestPortfolioSell:
    def test_sell_no_position(self):
        p = Portfolio(cash=10000.0)
        assert p.sell("BTC/USDT", 50000.0) is None

    def test_sell_calculates_pnl(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)  # 0.02 BTC
        trade = p.sell("BTC/USDT", 55000.0)
        assert trade is not None
        assert trade["pnl"] == pytest.approx(0.02 * 5000.0)  # +100
        assert trade["side"] == "sell"

    def test_sell_clears_position(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)
        p.sell("BTC/USDT", 55000.0)
        pos = p.get_position("BTC/USDT")
        assert pos.qty == 0.0


class TestTotalValue:
    def test_cash_only(self):
        p = Portfolio(cash=10000.0)
        assert p.total_value({}) == pytest.approx(10000.0)

    def test_cash_plus_positions(self):
        p = Portfolio(cash=9000.0)
        p.buy("BTC/USDT", 50000.0)  # spend 900 (9000 * 0.1)
        total = p.total_value({"BTC/USDT": 50000.0})
        assert total == pytest.approx(9000.0)  # 8100 cash + 900 position


class TestStopLoss:
    def test_no_position_returns_none(self):
        p = Portfolio(cash=10000.0)
        cfg = RiskConfig(stop_loss_pct=5.0, take_profit_pct=15.0)
        assert p.check_stop_loss("BTC/USDT", 50000.0, cfg) is None

    def test_stop_loss_triggers(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)
        cfg = RiskConfig(stop_loss_pct=5.0)
        # Price dropped 6%
        assert p.check_stop_loss("BTC/USDT", 47000.0, cfg) == "stop_loss"

    def test_stop_loss_not_triggered(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)
        cfg = RiskConfig(stop_loss_pct=5.0)
        # Price dropped only 3%
        assert p.check_stop_loss("BTC/USDT", 48500.0, cfg) is None

    def test_take_profit_triggers(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)
        cfg = RiskConfig(take_profit_pct=15.0)
        # Price rose 16%
        assert p.check_stop_loss("BTC/USDT", 58000.0, cfg) == "take_profit"

    def test_take_profit_not_triggered(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)
        cfg = RiskConfig(take_profit_pct=15.0)
        # Price rose 10%
        assert p.check_stop_loss("BTC/USDT", 55000.0, cfg) is None


class TestTrailingStop:
    def test_trailing_stop_triggers(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)
        # Price rises to 60k, then drops
        p.update_high_watermark("BTC/USDT", 60000.0)
        pos = p.get_position("BTC/USDT")
        assert pos.high_watermark == 60000.0
        # Drop from 60k to 54k = 10% from peak
        cfg = RiskConfig(stop_loss_pct=20.0, trailing_stop_pct=10.0)
        assert p.check_stop_loss("BTC/USDT", 54000.0, cfg) == "trailing_stop"

    def test_trailing_stop_not_triggered(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)
        p.update_high_watermark("BTC/USDT", 55000.0)
        # Drop from 55k to 52k = 5.5% from peak (< 10% threshold)
        cfg = RiskConfig(stop_loss_pct=20.0, trailing_stop_pct=10.0)
        assert p.check_stop_loss("BTC/USDT", 52000.0, cfg) is None

    def test_trailing_stop_disabled_when_zero(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)
        p.update_high_watermark("BTC/USDT", 60000.0)
        cfg = RiskConfig(stop_loss_pct=90.0, trailing_stop_pct=0.0)
        # 50% drop from peak but trailing stop disabled, SL at 90% so won't trigger
        assert p.check_stop_loss("BTC/USDT", 30000.0, cfg) is None

    def test_stop_loss_takes_priority_over_trailing(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)
        p.update_high_watermark("BTC/USDT", 52000.0)
        # Price at 45k: -10% from entry (SL triggers), also -13.5% from peak (TS triggers)
        cfg = RiskConfig(stop_loss_pct=10.0, trailing_stop_pct=10.0)
        assert p.check_stop_loss("BTC/USDT", 45000.0, cfg) == "stop_loss"

    def test_high_watermark_updates_on_buy(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)
        pos = p.get_position("BTC/USDT")
        assert pos.high_watermark == 50000.0

    def test_high_watermark_only_increases(self):
        p = Portfolio(cash=10000.0)
        p.buy("BTC/USDT", 50000.0)
        p.update_high_watermark("BTC/USDT", 55000.0)
        p.update_high_watermark("BTC/USDT", 53000.0)  # lower, should not update
        pos = p.get_position("BTC/USDT")
        assert pos.high_watermark == 55000.0


class TestCanBuy:
    def test_empty_portfolio_can_buy(self):
        p = Portfolio(cash=10000.0)
        cfg = RiskConfig(max_exposure_pct=60.0)
        assert p.can_buy({"BTC/USDT": 50000.0}, cfg) is True

    def test_max_exposure_blocks_buy(self):
        p = Portfolio(cash=1000.0)
        # Simulate heavy position: manual setup
        p.positions["BTC/USDT"] = {"qty": 0.1, "entry_price": 50000.0, "ticks_since_buy": 0}
        # Position value = 0.1 * 50000 = 5000, total = 6000
        # Exposure = 5000/6000 = 83% > 60%
        cfg = RiskConfig(max_exposure_pct=60.0)
        assert p.can_buy({"BTC/USDT": 50000.0}, cfg) is False
