"""Market data collector — Binance API via ccxt"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from loguru import logger
import ccxt

from core.storage import Storage


class MarketDataCollector:
    """通过 ccxt 统一接口采集 Binance 市场数据"""

    def __init__(self, storage: Storage, api_key: str = "",
                 api_secret: str = "", testnet: bool = False):
        self.storage = storage

        exchange_class = ccxt.binance
        if testnet:
            exchange_class = ccxt.binance

        self.exchange = exchange_class({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)

        logger.info(f"MarketDataCollector initialized (testnet={testnet})")

    def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """获取历史 K 线数据并存入数据库

        Args:
            symbol: 交易对，如 "BTC/USDT"
            timeframe: K线周期，如 "15m", "1h", "4h", "1d"
            since: 起始时间戳（毫秒）
            limit: 单次请求最大条数（Binance 默认 1000）
        """
        all_candles = []

        try:
            while True:
                logger.debug(
                    f"Fetching {symbol} {timeframe} since={since} limit={limit}"
                )
                raw = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit
                )

                if not raw:
                    break

                candles = [
                    {
                        "timestamp": int(c[0]),
                        "open": c[1],
                        "high": c[2],
                        "low": c[3],
                        "close": c[4],
                        "volume": c[5],
                    }
                    for c in raw
                ]
                all_candles.extend(candles)

                # Binance 最多返回 1000 条
                if len(raw) < limit:
                    break

                # 下一批从最后一条的时间戳开始
                since = raw[-1][0] + 1
                time.sleep(0.2)  # rate limit

            if all_candles:
                count = self.storage.insert_ohlcv(symbol, timeframe, all_candles)
                logger.info(
                    f"Saved {count} candles for {symbol} {timeframe}"
                )

            return all_candles

        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            return all_candles
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {e}")
            return all_candles

    def fetch_latest_price(self, symbol: str) -> float | None:
        """获取最新价格"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker["last"]
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def fetch_funding_rate(self, symbol: str) -> float | None:
        """获取资金费率（合约）"""
        try:
            # 需要转换为合约符号
            future_symbol = symbol.replace("/USDT", "/USDT:USDT")
            rate = self.exchange.fetch_funding_rate(future_symbol)
            return rate.get("fundingRate")
        except Exception as e:
            logger.debug(f"Funding rate not available for {symbol}: {e}")
            return None

    def backfill_days(
        self,
        symbol: str,
        timeframe: str = "1h",
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
