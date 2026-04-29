"""Data collector — orchestrates all data sources"""
from __future__ import annotations

from loguru import logger

from core.storage import Storage
from core.market_data import MarketDataCollector
from core.sentiment import SentimentCollector


class DataCollector:
    """数据采集总调度器"""

    def __init__(
        self,
        storage: Storage,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
    ):
        self.storage = storage
        self.market = MarketDataCollector(
            storage, api_key, api_secret, testnet
        )
        self.sentiment = SentimentCollector(storage)

    def backfill_all(self, symbols: list[str], days: int = 30):
        """回填所有交易对的历史数据"""
        timeframes = ["15m", "1h", "4h", "1d"]
        total = 0

        for symbol in symbols:
            for tf in timeframes:
                count = self.market.backfill_days(symbol, tf, days)
                total += count

        logger.info(f"Backfill complete. Total candles: {total}")
        return total

    def collect_latest(self, symbols: list[str]):
        """采集最新数据（每次分析周期调用）"""
        # 1. 更新最新 K 线
        for symbol in symbols:
            for tf in ["1h", "4h"]:
                self.market.fetch_historical_ohlcv(
                    symbol, tf, limit=100
                )

        # 2. 更新情绪数据
        self.sentiment.get_latest_sentiment()

        logger.info(f"Latest data collected for {symbols}")
