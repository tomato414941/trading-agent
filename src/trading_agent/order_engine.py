"""Unified order engine: Paper, Backtest, and Live execution share one interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import ccxt

log = logging.getLogger(__name__)


@dataclass
class OrderResult:
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    price: float
    fee: float
    slippage_bps: float
    timestamp: str = ""


class OrderEngine(ABC):
    @abstractmethod
    def market_buy(self, symbol: str, amount_usdt: float) -> OrderResult | None:
        ...

    @abstractmethod
    def market_sell(self, symbol: str, qty: float) -> OrderResult | None:
        ...

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def get_balance(self) -> float:
        ...


class PaperOrderEngine(OrderEngine):
    """Paper trading: track orders in memory with configurable slippage."""

    def __init__(
        self,
        initial_cash: float = 10_000.0,
        fee_rate: float = 0.001,
        slippage_bps: float = 5.0,
    ):
        self.cash = initial_cash
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.positions: dict[str, float] = {}  # symbol -> qty
        self._prices: dict[str, float] = {}

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = price

    def get_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    def get_balance(self) -> float:
        return self.cash

    def market_buy(self, symbol: str, amount_usdt: float) -> OrderResult | None:
        price = self.get_price(symbol)
        if price <= 0 or amount_usdt <= 0 or amount_usdt > self.cash:
            return None

        slip = price * self.slippage_bps / 10_000
        exec_price = price + slip
        fee = amount_usdt * self.fee_rate
        net_amount = amount_usdt - fee
        qty = net_amount / exec_price

        self.cash -= amount_usdt
        self.positions[symbol] = self.positions.get(symbol, 0.0) + qty

        return OrderResult(
            symbol=symbol, side="buy", qty=qty,
            price=exec_price, fee=fee,
            slippage_bps=self.slippage_bps,
        )

    def market_sell(self, symbol: str, qty: float) -> OrderResult | None:
        price = self.get_price(symbol)
        held = self.positions.get(symbol, 0.0)
        if price <= 0 or qty <= 0 or held < qty:
            return None

        slip = price * self.slippage_bps / 10_000
        exec_price = price - slip
        revenue = qty * exec_price
        fee = revenue * self.fee_rate
        net_revenue = revenue - fee

        self.positions[symbol] = held - qty
        self.cash += net_revenue

        return OrderResult(
            symbol=symbol, side="sell", qty=qty,
            price=exec_price, fee=fee,
            slippage_bps=self.slippage_bps,
        )


class LiveOrderEngine(OrderEngine):
    """Real execution via CCXT with slippage protection."""

    def __init__(
        self,
        exchange: ccxt.Exchange,
        fee_rate: float = 0.001,
        max_slippage_bps: float = 10.0,
        max_book_fraction: float = 0.1,
    ):
        self._exchange = exchange
        self.fee_rate = fee_rate
        self.max_slippage_bps = max_slippage_bps
        self.max_book_fraction = max_book_fraction

    def get_price(self, symbol: str) -> float:
        ticker = self._exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    def get_balance(self) -> float:
        balance = self._exchange.fetch_balance()
        return float(balance.get("USDT", {}).get("free", 0))

    def market_buy(self, symbol: str, amount_usdt: float) -> OrderResult | None:
        orderbook = self._exchange.fetch_order_book(symbol, limit=5)
        best_ask = orderbook["asks"][0][0] if orderbook["asks"] else None
        if best_ask is None:
            log.error("No asks in orderbook for %s", symbol)
            return None

        book_depth = sum(p * q for p, q in orderbook["asks"][:5])
        if amount_usdt > book_depth * self.max_book_fraction:
            log.warning(
                "Order $%.2f > %.0f%% of book depth $%.2f, skipping",
                amount_usdt, self.max_book_fraction * 100, book_depth,
            )
            return None

        qty = amount_usdt / best_ask
        qty = float(self._exchange.amount_to_precision(symbol, qty))
        if qty <= 0:
            return None

        order = self._exchange.create_market_buy_order(symbol, qty)
        filled_price = float(order.get("average", best_ask))
        filled_qty = float(order.get("filled", qty))
        fee = float(order.get("cost", amount_usdt)) * self.fee_rate

        slippage = abs(filled_price - best_ask) / best_ask * 10_000
        if slippage > self.max_slippage_bps:
            log.warning("High slippage: %.1f bps on %s buy", slippage, symbol)

        return OrderResult(
            symbol=symbol, side="buy", qty=filled_qty,
            price=filled_price, fee=fee, slippage_bps=slippage,
        )

    def market_sell(self, symbol: str, qty: float) -> OrderResult | None:
        orderbook = self._exchange.fetch_order_book(symbol, limit=5)
        best_bid = orderbook["bids"][0][0] if orderbook["bids"] else None
        if best_bid is None:
            log.error("No bids in orderbook for %s", symbol)
            return None

        amount_usdt = qty * best_bid
        book_depth = sum(p * q for p, q in orderbook["bids"][:5])
        if amount_usdt > book_depth * self.max_book_fraction:
            log.warning(
                "Sell $%.2f > %.0f%% of book depth $%.2f, skipping",
                amount_usdt, self.max_book_fraction * 100, book_depth,
            )
            return None

        qty = float(self._exchange.amount_to_precision(symbol, qty))
        if qty <= 0:
            return None

        order = self._exchange.create_market_sell_order(symbol, qty)
        filled_price = float(order.get("average", best_bid))
        filled_qty = float(order.get("filled", qty))
        fee = float(order.get("cost", amount_usdt)) * self.fee_rate

        slippage = abs(filled_price - best_bid) / best_bid * 10_000
        if slippage > self.max_slippage_bps:
            log.warning("High slippage: %.1f bps on %s sell", slippage, symbol)

        return OrderResult(
            symbol=symbol, side="sell", qty=filled_qty,
            price=filled_price, fee=fee, slippage_bps=slippage,
        )
