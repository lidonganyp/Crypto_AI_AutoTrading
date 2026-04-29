"""Binance market data collector for CryptoAI v3."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

import ccxt
from loguru import logger

from core.storage import Storage
from execution.exchange_adapter import OrderBookDepthSnapshot


class BinanceMarketDataCollector:
    """Collect Binance market data via ccxt."""

    def __init__(
        self,
        storage: Storage,
        proxy: str = "",
        api_key: str = "",
        api_secret: str = "",
        exchange: Any | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self.storage = storage
        if exchange is not None:
            self.exchange = exchange
        else:
            params = {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                    "adjustForTimeDifference": True,
                    "recvWindow": 20000,
                },
            }
            if proxy:
                params["proxies"] = {
                    "http": proxy,
                    "https": proxy,
                }
            if api_key:
                params["apiKey"] = api_key
            if api_secret:
                params["secret"] = api_secret
            self.exchange = ccxt.binance(params)
            try:
                self.exchange.load_time_difference()
            except Exception as exc:
                logger.warning(f"Binance time difference sync failed: {exc}")
        self._sleep = sleep_fn or time.sleep
        logger.info(f"BinanceMarketDataCollector initialized (proxy={proxy})")

    @staticmethod
    def _spot_symbol(symbol: str) -> str:
        return str(symbol or "").replace(":USDT", "")

    def _mark_operation(self, operation: str, failed: bool) -> None:
        self.last_operation_name = operation
        self.last_operation_failed = bool(failed)

    def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 300,
    ) -> list[dict]:
        all_candles = []
        try:
            market_symbol = self._spot_symbol(symbol)
            while True:
                raw = self.exchange.fetch_ohlcv(
                    market_symbol,
                    timeframe,
                    since=since,
                    limit=limit,
                )
                if not raw:
                    break
                candles = [
                    {
                        "timestamp": int(c[0]),
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                    }
                    for c in raw
                ]
                all_candles.extend(candles)
                if len(raw) < limit:
                    break
                since = raw[-1][0] + 1
                self._sleep(0.2)
            if all_candles:
                self.storage.insert_ohlcv(symbol, timeframe, all_candles)
            self._mark_operation("fetch_historical_ohlcv", False)
            return all_candles
        except Exception as exc:
            logger.error(f"Binance fetch OHLCV failed for {symbol}: {exc}")
            self._mark_operation("fetch_historical_ohlcv", True)
            return all_candles

    def fetch_latest_price(self, symbol: str) -> float | None:
        try:
            ticker = self.exchange.fetch_ticker(self._spot_symbol(symbol))
            self._mark_operation("fetch_latest_price", False)
            return float(ticker["last"])
        except Exception as exc:
            logger.error(f"Binance price fetch failed for {symbol}: {exc}")
            self._mark_operation("fetch_latest_price", True)
            return None

    def measure_latency(self, symbol: str) -> dict:
        received_at_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        try:
            start = time.perf_counter()
            ticker = self.exchange.fetch_ticker(self._spot_symbol(symbol))
            latency_seconds = max(0.0, time.perf_counter() - start)
            exchange_ts = ticker.get("timestamp")
            self._mark_operation("measure_latency", False)
            return {
                "symbol": symbol,
                "exchange_timestamp_ms": int(exchange_ts) if exchange_ts is not None else None,
                "received_at_ms": received_at_ms,
                "latency_seconds": latency_seconds,
                "status": "ok",
            }
        except Exception as exc:
            logger.error(f"Binance latency measurement failed for {symbol}: {exc}")
            self._mark_operation("measure_latency", True)
            return {
                "symbol": symbol,
                "exchange_timestamp_ms": None,
                "received_at_ms": received_at_ms,
                "latency_seconds": None,
                "status": "failed",
                "error": str(exc),
            }

    def fetch_available_instruments(self) -> list[str]:
        try:
            self.exchange.load_markets()
            self._mark_operation("fetch_available_instruments", False)
            return [sym for sym in self.exchange.symbols if "/USDT" in sym]
        except Exception as exc:
            logger.error(f"Failed to fetch Binance instruments: {exc}")
            self._mark_operation("fetch_available_instruments", True)
            return []

    def fetch_funding_rate(self, symbol: str) -> float | None:
        try:
            future_symbol = symbol if ":USDT" in symbol else symbol.replace("/USDT", "/USDT:USDT")
            rate = self.exchange.fetch_funding_rate(future_symbol)
            value = rate.get("fundingRate")
            self._mark_operation("fetch_funding_rate", value is None)
            return float(value) if value is not None else None
        except Exception as exc:
            logger.debug(f"Funding rate not available for {symbol}: {exc}")
            self._mark_operation("fetch_funding_rate", True)
            return None

    def summarize_order_book_depth(
        self,
        symbol: str,
        depth: int = 5,
    ) -> OrderBookDepthSnapshot:
        order_book = self.exchange.fetch_order_book(
            self._spot_symbol(symbol),
            limit=max(20, depth),
        )
        bids = order_book.get("bids", [])[:depth]
        asks = order_book.get("asks", [])[:depth]
        bid_notional = sum(float(price) * float(size) for price, size in bids)
        ask_notional = sum(float(price) * float(size) for price, size in asks)
        total = bid_notional + ask_notional
        imbalance = ((bid_notional - ask_notional) / total) if total > 0 else 0.0
        large_bid_notional = max(
            (float(price) * float(size) for price, size in bids),
            default=0.0,
        )
        large_ask_notional = max(
            (float(price) * float(size) for price, size in asks),
            default=0.0,
        )
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0
        mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
        spread_pct = ((best_ask - best_bid) / mid) if mid > 0 else 0.0
        self._mark_operation("summarize_order_book_depth", False)
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
