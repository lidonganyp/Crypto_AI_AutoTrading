"""OKX Market Data Collector — via ccxt with SOCKS5 proxy"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from loguru import logger
import ccxt
from typing import Any, Callable

from core.storage import Storage
from execution.exchange_adapter import OrderBookDepthSnapshot


class OKXMarketDataCollector:
    """通过 ccxt 统一接口采集 OKX 市场数据（通过代理）"""

    def __init__(
        self,
        storage: Storage,
        proxy: str = "",
        exchange: Any | None = None,
        spot_exchange: Any | None = None,
        swap_exchange: Any | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self.storage = storage
        if exchange is not None:
            self.spot_exchange = exchange
            self.swap_exchange = exchange
        else:
            base_params = {"enableRateLimit": True}
            if proxy:
                base_params["proxies"] = {
                    "http": proxy,
                    "https": proxy,
                }
            self.spot_exchange = spot_exchange or ccxt.okx(
                {
                    **base_params,
                    "options": {"defaultType": "spot"},
                }
            )
            self.swap_exchange = swap_exchange or ccxt.okx(
                {
                    **base_params,
                    "options": {"defaultType": "swap"},
                }
            )
        self.exchange = self.spot_exchange
        self._sleep = sleep_fn or time.sleep
        logger.info(f"OKXMarketDataCollector initialized (proxy={proxy})")

    def _mark_operation(self, operation: str, failed: bool) -> None:
        self.last_operation_name = operation
        self.last_operation_failed = bool(failed)

    @staticmethod
    def _is_swap_symbol(symbol: str) -> bool:
        return ":USDT" in symbol

    @classmethod
    def _spot_symbol(cls, symbol: str) -> str:
        return symbol.replace(":USDT", "") if cls._is_swap_symbol(symbol) else symbol

    @classmethod
    def _swap_symbol(cls, symbol: str) -> str:
        return symbol if cls._is_swap_symbol(symbol) else symbol.replace("/USDT", "/USDT:USDT")

    def _fetch_ohlcv_once(
        self,
        exchange: Any,
        symbol: str,
        timeframe: str,
        since: int | None,
        limit: int,
    ):
        return exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    @staticmethod
    def _okx_inst_id(symbol: str) -> str:
        spot_symbol = symbol.replace(":USDT", "")
        base, quote = spot_symbol.split("/", 1)
        if ":USDT" in symbol:
            return f"{base}-{quote}-SWAP"
        return f"{base}-{quote}"

    @staticmethod
    def _okx_bar(timeframe: str) -> str:
        mapping = {
            "5m": "5m",
            "15m": "15m",
            "1h": "1H",
            "4h": "4H",
            "1d": "1Dutc",
        }
        return mapping.get(timeframe.lower(), timeframe)

    def _raw_fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> list[dict]:
        exchange = self.spot_exchange
        if not hasattr(exchange, "publicGetMarketCandles"):
            raw = self._fetch_ohlcv_once(
                exchange,
                symbol,
                self._to_ccxt_timeframe(timeframe),
                None,
                limit,
            )
            return [
                {
                    "timestamp": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
                for row in raw
                if len(row) >= 6
            ]
        payload = exchange.publicGetMarketCandles(
            {
                "instId": self._okx_inst_id(symbol),
                "bar": self._okx_bar(timeframe),
                "limit": min(limit, 300),
            }
        )
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        candles = [
            {
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
            for row in rows
            if len(row) >= 6
        ]
        candles.sort(key=lambda candle: candle["timestamp"])
        return candles

    def _raw_fetch_ticker(self, symbol: str) -> dict:
        exchange = self.spot_exchange
        if not hasattr(exchange, "publicGetMarketTicker"):
            return exchange.fetch_ticker(symbol)
        payload = exchange.publicGetMarketTicker(
            {"instId": self._okx_inst_id(symbol)}
        )
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not rows:
            raise ccxt.ExchangeError(f"empty ticker for {symbol}")
        return rows[0]

    def _cached_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[dict]:
        cached = self.storage.get_ohlcv(symbol, timeframe, limit=limit)
        if cached:
            cached.sort(key=lambda candle: candle["timestamp"])
        return cached

    def _cached_latest_price(self, symbol: str) -> float | None:
        for timeframe in ("1h", "4h", "1d"):
            cached = self._cached_ohlcv(symbol, timeframe, limit=1)
            if cached:
                return float(cached[-1]["close"])
        return None

    def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 300,
    ) -> list[dict]:
        """获取历史 K 线数据并存入数据库"""
        all_candles = []
        normalized_timeframe = self._to_ccxt_timeframe(timeframe)
        primary_exchange = self.spot_exchange
        primary_symbol = self._spot_symbol(symbol)

        def fetch_with_exchange(target_exchange: Any, target_symbol: str) -> list[dict]:
            candles_accumulated: list[dict] = []
            next_since = since
            page_limit = min(int(limit or 300), 300)
            while True:
                logger.debug(
                    f"Fetching {target_symbol} {normalized_timeframe} since={next_since} limit={page_limit}"
                )
                raw = self._fetch_ohlcv_once(
                    target_exchange,
                    target_symbol,
                    normalized_timeframe,
                    next_since,
                    page_limit,
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
                candles_accumulated.extend(candles)

                if len(raw) < page_limit:
                    break

                next_since = raw[-1][0] + 1
                self._sleep(0.3)
            return candles_accumulated

        try:
            if since is None:
                all_candles = self._raw_fetch_ohlcv(primary_symbol, timeframe, limit)
            else:
                all_candles = fetch_with_exchange(primary_exchange, primary_symbol)

            if all_candles:
                count = self.storage.insert_ohlcv(
                    symbol, timeframe, all_candles
                )
                logger.info(
                    f"Saved {count} candles for {symbol} {timeframe}"
                )
            self._mark_operation("fetch_historical_ohlcv", False)
            return all_candles

        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            cached = self._cached_ohlcv(symbol, timeframe, limit)
            if cached:
                logger.warning(f"Using cached OHLCV for {symbol} {timeframe} after network failure")
                self._mark_operation("fetch_historical_ohlcv", True)
                return cached
            self._mark_operation("fetch_historical_ohlcv", True)
            return all_candles
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {e}")
            cached = self._cached_ohlcv(symbol, timeframe, limit)
            if cached:
                logger.warning(f"Using cached OHLCV for {symbol} {timeframe} after exchange failure")
                self._mark_operation("fetch_historical_ohlcv", True)
                return cached
            self._mark_operation("fetch_historical_ohlcv", True)
            return all_candles

    def fetch_latest_price(self, symbol: str) -> float | None:
        """获取最新价格"""
        try:
            ticker = self._raw_fetch_ticker(self._spot_symbol(symbol))
            self._mark_operation("fetch_latest_price", False)
            return float(ticker["last"])
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            cached_price = self._cached_latest_price(symbol)
            if cached_price is not None:
                logger.warning(f"Using cached latest price for {symbol}")
                self._mark_operation("fetch_latest_price", True)
                return cached_price
            self._mark_operation("fetch_latest_price", True)
            return None

    def measure_latency(self, symbol: str) -> dict:
        """Measure exchange API latency using local round-trip time."""
        received_at_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        try:
            start = time.perf_counter()
            ticker = self._raw_fetch_ticker(self._spot_symbol(symbol))
            latency_seconds = max(0.0, time.perf_counter() - start)
            exchange_ts = ticker.get("ts") or ticker.get("timestamp")
            self._mark_operation("measure_latency", False)
            return {
                "symbol": symbol,
                "exchange_timestamp_ms": int(exchange_ts) if exchange_ts is not None else None,
                "received_at_ms": received_at_ms,
                "latency_seconds": latency_seconds,
                "status": "ok",
            }
        except Exception as e:
            logger.error(f"Error measuring latency for {symbol}: {e}")
            self._mark_operation("measure_latency", True)
            return {
                "symbol": symbol,
                "exchange_timestamp_ms": None,
                "received_at_ms": received_at_ms,
                "latency_seconds": None,
                "status": "failed",
                "error": str(e),
            }

    def fetch_funding_rate(self, symbol: str) -> float | None:
        """Fetch funding rate for swap symbol when available."""
        try:
            rate = self.swap_exchange.fetch_funding_rate(self._swap_symbol(symbol))
            value = rate.get("fundingRate")
            self._mark_operation("fetch_funding_rate", value is None)
            return float(value) if value is not None else None
        except Exception as e:
            logger.debug(f"Funding rate not available for {symbol}: {e}")
            self._mark_operation("fetch_funding_rate", True)
            return None

    def fetch_available_instruments(self) -> list[str]:
        """获取交易所所有可交易交易对（spot USDT）"""
        try:
            self.spot_exchange.load_markets()
            instruments = [
                sym for sym in self.spot_exchange.symbols
                if sym.endswith("/USDT")
            ]
            logger.info(f"交易所可用 USDT 交易对: {len(instruments)} 个")
            self._mark_operation("fetch_available_instruments", False)
            return instruments
        except Exception as e:
            logger.error(f"Failed to fetch instruments: {e}")
            self._mark_operation("fetch_available_instruments", True)
            return []

    def summarize_order_book_depth(
        self,
        symbol: str,
        depth: int = 5,
    ) -> OrderBookDepthSnapshot:
        order_book = self.spot_exchange.fetch_order_book(
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

    def backfill_days(
        self,
        symbol: str,
        timeframe: str = "1H",
        days: int = 30,
    ) -> int:
        """回填指定天数的历史数据"""
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        since_ms = now_ms - (days * 24 * 60 * 60 * 1000)

        logger.info(
            f"Backfilling {symbol} {timeframe} for {days} days..."
        )
        candles = self.fetch_historical_ohlcv(
            symbol, timeframe, since=since_ms
        )
        logger.info(
            f"Backfill complete: {len(candles)} candles for "
            f"{symbol} {timeframe}"
        )
        return len(candles)

    @staticmethod
    def _to_ccxt_timeframe(tf: str) -> str:
        """OKX timeframes: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 1D, 1W, 1M"""
        mapping = {
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }
        return mapping.get(tf.lower(), tf)
