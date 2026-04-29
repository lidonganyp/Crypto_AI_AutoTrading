"""Exchange adapter and slippage utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ccxt


@dataclass
class SlippageEstimate:
    symbol: str
    reference_price: float
    expected_price: float
    slippage_pct: float


@dataclass
class OrderBookDepthSnapshot:
    symbol: str
    bid_notional_top5: float
    ask_notional_top5: float
    bid_ask_spread_pct: float
    depth_imbalance: float
    large_bid_notional: float
    large_ask_notional: float
    large_order_net_notional: float


class SlippageGuard:
    """Estimate slippage before allowing an order."""

    def __init__(self, max_slippage_pct: float):
        self.max_slippage_pct = max_slippage_pct

    def check(self, estimate: SlippageEstimate) -> tuple[bool, str]:
        if estimate.slippage_pct > self.max_slippage_pct:
            return (
                False,
                f"slippage {estimate.slippage_pct:.4%} exceeds limit {self.max_slippage_pct:.4%}",
            )
        return True, ""


class OKXExchangeAdapter:
    """Thin adapter for OKX exchange interactions."""

    def __init__(
        self,
        proxy_url: str = "",
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        exchange: Any | None = None,
    ):
        if exchange is not None:
            self.exchange = exchange
            return

        params = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
                "adjustForTimeDifference": True,
                "recvWindow": 20000,
            },
        }
        if proxy_url:
            params["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }
        if api_key:
            params["apiKey"] = api_key
        if api_secret:
            params["secret"] = api_secret
        if api_passphrase:
            params["password"] = api_passphrase
        self.exchange = ccxt.okx(params)

    def fetch_last_price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    def fetch_free_balance(self, asset: str) -> float | None:
        balance = self.exchange.fetch_balance()
        free_balances = balance.get("free") or {}
        value = free_balances.get(asset)
        return float(value) if value is not None else None

    def fetch_total_balance(self, asset: str) -> float | None:
        balance = self.exchange.fetch_balance()
        total_balances = balance.get("total") or {}
        free_balances = balance.get("free") or {}
        value = total_balances.get(asset)
        if value is None:
            value = free_balances.get(asset)
        return float(value) if value is not None else None

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        if symbol:
            orders = self.exchange.fetch_open_orders(symbol)
        else:
            orders = self.exchange.fetch_open_orders()
        return [
            self._normalize_order(order, float(order.get("amount") or 0.0))
            for order in orders
        ]

    def estimate_slippage(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reference_price: float,
    ) -> SlippageEstimate:
        order_book = self.exchange.fetch_order_book(symbol, limit=20)
        levels = order_book["asks"] if side.upper() == "BUY" else order_book["bids"]
        remaining = quantity
        notional = 0.0
        for price, size in levels:
            take = min(remaining, float(size))
            notional += take * float(price)
            remaining -= take
            if remaining <= 1e-12:
                break

        expected_price = (
            notional / quantity if quantity > 0 and notional > 0 else reference_price
        )
        slippage_pct = abs(expected_price / reference_price - 1.0) if reference_price > 0 else 0.0
        return SlippageEstimate(
            symbol=symbol,
            reference_price=reference_price,
            expected_price=expected_price,
            slippage_pct=slippage_pct,
        )

    def summarize_order_book_depth(self, symbol: str, depth: int = 5) -> OrderBookDepthSnapshot:
        order_book = self.exchange.fetch_order_book(symbol, limit=max(20, depth))
        bids = order_book.get("bids", [])[:depth]
        asks = order_book.get("asks", [])[:depth]
        bid_notional = sum(float(price) * float(size) for price, size in bids)
        ask_notional = sum(float(price) * float(size) for price, size in asks)
        total = bid_notional + ask_notional
        imbalance = ((bid_notional - ask_notional) / total) if total > 0 else 0.0
        large_bid_notional = max((float(price) * float(size) for price, size in bids), default=0.0)
        large_ask_notional = max((float(price) * float(size) for price, size in asks), default=0.0)
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0
        mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
        spread_pct = ((best_ask - best_bid) / mid) if mid > 0 else 0.0
        return OrderBookDepthSnapshot(
            symbol=symbol,
            bid_notional_top5=bid_notional,
            ask_notional_top5=ask_notional,
            bid_ask_spread_pct=spread_pct,
            depth_imbalance=imbalance,
            large_bid_notional=large_bid_notional,
            large_ask_notional=large_ask_notional,
            large_order_net_notional=large_bid_notional - large_ask_notional,
        )

    def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        order = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side.lower(),
            amount=quantity,
        )
        return self._normalize_order(order, quantity)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> dict:
        order = self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side.lower(),
            amount=quantity,
            price=price,
        )
        return self._normalize_order(order, quantity)

    def fetch_order(self, exchange_order_id: str, symbol: str) -> dict:
        order = self.exchange.fetch_order(exchange_order_id, symbol)
        return self._normalize_order(order, float(order.get("amount") or 0.0))

    def cancel_order(self, exchange_order_id: str, symbol: str) -> dict:
        order = self.exchange.cancel_order(exchange_order_id, symbol)
        return self._normalize_order(order, float(order.get("amount") or 0.0))

    @staticmethod
    def _normalize_order(order: dict, fallback_qty: float) -> dict:
        status = str(order.get("status") or "")
        side = str(order.get("side") or "").upper()
        raw_amount = order.get("amount")
        raw_filled = order.get("filled")
        raw_remaining = order.get("remaining")
        requested_qty = float(raw_amount) if raw_amount is not None else float(fallback_qty or 0.0)
        if raw_filled is not None:
            filled_qty = float(raw_filled)
        elif status.lower() in {"closed", "filled"} and raw_amount is not None:
            filled_qty = float(raw_amount)
        else:
            filled_qty = 0.0
        if raw_remaining is not None:
            remaining_qty = max(float(raw_remaining), 0.0)
        else:
            remaining_qty = max(requested_qty - filled_qty, 0.0)
        return {
            "exchange_order_id": order.get("id", ""),
            "symbol": str(order.get("symbol") or ""),
            "side": side,
            "status": status,
            "price": float(order.get("price") or 0.0),
            "average_price": float(order.get("average") or order.get("price") or 0.0),
            "requested_qty": requested_qty,
            "filled_qty": filled_qty,
            "remaining_qty": remaining_qty,
            "raw": order,
        }


class BinanceExchangeAdapter(OKXExchangeAdapter):
    """Thin adapter for Binance exchange interactions."""

    def __init__(
        self,
        proxy_url: str = "",
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        exchange: Any | None = None,
    ):
        if exchange is not None:
            self.exchange = exchange
            return

        params = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        if proxy_url:
            params["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }
        if api_key:
            params["apiKey"] = api_key
        if api_secret:
            params["secret"] = api_secret
        self.exchange = ccxt.binance(params)
        try:
            self.exchange.load_time_difference()
        except Exception:
            pass
