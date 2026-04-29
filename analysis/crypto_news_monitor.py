"""Crypto News Monitor — 加密新闻与热度监控

核心理念:
- "新闻驱动短期波动，但不要让新闻驱动你的策略" — Mark Douglas
- "关注聪明钱在做什么，而不是 Twitter 在讨论什么" — Raoul Pal
- "趋势一旦形成，新闻会解释为什么" — 行业共识

功能:
1. 监控加密市场热点币种（CoinGecko 趋势）
2. 监控赛道轮动信号（哪些赛道在涨/跌）
3. 监控重大事件（减半、ETF、监管等）
4. 为 WatchlistManager 提供热度数据

数据来源（全部免费）:
- CoinGecko API: 趋势币、市值排名
- Alternative.me: 恐惧贪婪指数
- 如 API 不可用，使用缓存 + 规则推断
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from loguru import logger

import requests


@dataclass
class TrendingCoin:
    """趋势币种"""
    symbol: str
    name: str
    market_cap_rank: int = 0
    price_btc: float = 0
    score: float = 0       # 热度评分
    source: str = ""       # 数据来源


@dataclass
class SectorMomentum:
    """赛道动量"""
    sector: str
    avg_change_24h: float
    avg_change_7d: float
    top_performer: str
    momentum_score: float   # -1.0 (弱) 到 1.0 (强)
    rotation_signal: str    # "rotate_in" / "rotate_out" / "hold"


@dataclass
class MarketEvent:
    """重大市场事件"""
    event_type: str          # "halving" / "etf" / "regulation" / "upgrade" / "macro"
    description: str
    impact: str              # "bullish" / "bearish" / "neutral"
    date: str
    days_away: int = 0


@dataclass
class NewsDigest:
    """新闻摘要"""
    trending: list[TrendingCoin] = field(default_factory=list)
    sector_momentum: list[SectorMomentum] = field(default_factory=list)
    upcoming_events: list[MarketEvent] = field(default_factory=list)
    fear_greed: float | None = None
    summary: str = ""
    timestamp: str = ""


class CryptoNewsMonitor:
    """
    加密新闻与热度监控器

    用法:
        monitor = CryptoNewsMonitor()
        digest = monitor.get_digest()
        # 获取趋势币
        for coin in digest.trending:
            logger.info(f"🔥 {coin.name} ({coin.symbol})")
        # 获取赛道轮动信号
        for sector in digest.sector_momentum:
            if sector.rotation_signal != "hold":
                logger.info(f"🔄 {sector.sector}: {sector.rotation_signal}")
    """

    # API 端点
    COINGECKO_BASE = "https://api.coingecko.com/api/v3"
    FEAR_GREED_URL = "https://api.alternative.me/fng/"

    # 缓存
    CACHE_TTL_SECONDS = 1800  # 30 分钟缓存

    def __init__(self):
        self._cache: dict = {}
        self._cache_time: dict = {}
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "CryptoAI/1.0",
        })

    def get_digest(self) -> NewsDigest:
        """获取完整的新闻摘要"""
        logger.info("📰 获取加密市场新闻摘要...")

        trending = self._get_trending()
        sector_momentum = self._get_sector_momentum()
        events = self._get_upcoming_events()
        fear_greed = self._get_fear_greed()

        # 生成摘要
        summary_lines = []

        if trending:
            top3 = trending[:3]
            summary_lines.append(
                "🔥 热度 Top: " +
                ", ".join(f"{c.name}({c.symbol})" for c in top3)
            )

        hot_sectors = [s for s in sector_momentum if s.rotation_signal == "rotate_in"]
        if hot_sectors:
            summary_lines.append(
                "📈 强势赛道: " + ", ".join(s.sector for s in hot_sectors)
            )

        weak_sectors = [s for s in sector_momentum if s.rotation_signal == "rotate_out"]
        if weak_sectors:
            summary_lines.append(
                "📉 弱势赛道: " + ", ".join(s.sector for s in weak_sectors)
            )

        if events:
            for e in events[:3]:
                summary_lines.append(
                    f"📅 {e.description} ({e.days_away} 天后)"
                )

        digest = NewsDigest(
            trending=trending,
            sector_momentum=sector_momentum,
            upcoming_events=events,
            fear_greed=fear_greed,
            summary="\n".join(summary_lines),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(f"📰 新闻摘要: {len(trending)} 趋势币, "
                     f"{len(sector_momentum)} 赛道, "
                     f"{len(events)} 事件")
        return digest

    def _get_trending(self) -> list[TrendingCoin]:
        """获取 CoinGecko 趋势币"""
        cache_key = "trending"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        trending = []

        try:
            # CoinGecko 搜索趋势（免费 API）
            url = f"{self.COINGECKO_BASE}/search/trending"
            resp = self._session.get(url, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                coins = data.get("coins", [])

                for i, item in enumerate(coins[:15]):
                    coin_data = item.get("item", {})
                    symbol = coin_data.get("symbol", "").upper()
                    name = coin_data.get("name", "")
                    rank = coin_data.get("market_cap_rank", 0) or 0
                    price_btc = coin_data.get("price_btc", 0) or 0

                    # 只保留 /USDT 格式
                    trading_symbol = f"{symbol}/USDT"

                    trending.append(TrendingCoin(
                        symbol=trading_symbol,
                        name=name,
                        market_cap_rank=rank,
                        price_btc=price_btc,
                        score=round(1.0 - i * 0.05, 2),
                        source="coingecko_trending",
                    ))

                self._set_cache(cache_key, trending)
                logger.info(f"🔥 趋势币: {len(trending)} 个")

        except requests.RequestException as e:
            logger.warning(f"趋势币获取失败: {e}")
        except Exception as e:
            logger.warning(f"趋势币解析失败: {e}")

        return trending

    def _get_sector_momentum(self) -> list[SectorMomentum]:
        """获取赛道动量数据"""
        cache_key = "sector_momentum"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # 代表币种（每个赛道用 1-2 个代表来估算赛道动量）
        sector_proxies = {
            "L1_公链": "solana",
            "L2_扩容": "arbitrum",
            "DeFi": "uniswap",
            "AI_人工智能": "render-token",
            "Meme": "dogecoin",
            "RWA": "ondo-finance",
        }

        momentum_list = []

        try:
            # 批量获取代表币种数据
            ids = ",".join(sector_proxies.values())
            url = f"{self.COINGECKO_BASE}/coins/markets"
            params = {
                "vs_currency": "usd",
                "ids": ids,
                "order": "market_cap_desc",
                "sparkline": "false",
                "price_change_percentage": "24h,7d",
            }

            resp = self._session.get(url, params=params, timeout=15)

            if resp.status_code == 200:
                data = resp.json()

                # 建立 id → 数据 的映射
                coin_data = {}
                for coin in data:
                    coin_data[coin.get("id", "")] = coin

                for sector, coin_id in sector_proxies.items():
                    d = coin_data.get(coin_id, {})
                    change_24h = d.get("price_change_percentage_24h_in_currency", 0) or 0
                    change_7d = 0
                    pct_data = d.get("price_change_percentage_7d_in_currency", 0)
                    if pct_data is not None:
                        change_7d = pct_data
                    elif isinstance(d.get("price_change_percentage_7d_in_currency"), str):
                        try:
                            change_7d = float(d["price_change_percentage_7d_in_currency"])
                        except:
                            change_7d = 0

                    name = d.get("name", coin_id)

                    # 动量评分
                    momentum = 0
                    if change_24h > 5:
                        momentum = min(1.0, change_24h / 20)
                    elif change_24h < -5:
                        momentum = max(-1.0, change_24h / 20)
                    else:
                        momentum = change_24h / 20

                    # 轮动信号
                    if momentum > 0.3 and change_7d > 0:
                        signal = "rotate_in"
                    elif momentum < -0.3 and change_7d < 0:
                        signal = "rotate_out"
                    else:
                        signal = "hold"

                    momentum_list.append(SectorMomentum(
                        sector=sector,
                        avg_change_24h=round(change_24h, 2),
                        avg_change_7d=round(change_7d, 2),
                        top_performer=name,
                        momentum_score=round(momentum, 3),
                        rotation_signal=signal,
                    ))

                self._set_cache(cache_key, momentum_list)

        except requests.RequestException as e:
            logger.warning(f"赛道动量获取失败: {e}")
        except Exception as e:
            logger.warning(f"赛道动量解析失败: {e}")

        return momentum_list

    def _get_upcoming_events(self) -> list[MarketEvent]:
        """获取即将到来的重大事件"""
        now = datetime.now(timezone.utc)
        events = []

        # ── 已知的周期性事件 ──
        # BTC 减半
        next_halving = datetime(2028, 4, 1, tzinfo=timezone.utc)
        days_to_halving = (next_halving - now).days
        if days_to_halving > 0 and days_to_halving < 365:
            events.append(MarketEvent(
                event_type="halving",
                description="BTC 减半",
                impact="bullish",
                date=next_halving.strftime("%Y-%m-%d"),
                days_away=days_to_halving,
            ))

        # ── 通过监控新增的热点 ──
        # 这里可以扩展接入 CoinGlass 的日历 API
        # 目前使用内置数据

        return events

    def _get_fear_greed(self) -> float | None:
        """获取恐惧贪婪指数"""
        cache_key = "fear_greed"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            resp = self._session.get(self.FEAR_GREED_URL, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                value = data.get("data", [{}])[0].get("value")
                if value is not None:
                    fgi = float(value)
                    self._set_cache(cache_key, fgi)
                    return fgi
        except Exception as e:
            logger.warning(f"恐惧贪婪指数获取失败: {e}")

        return None

    def get_trending_symbols(self) -> list[str]:
        """快速获取趋势币种列表（供 WatchlistManager 使用）"""
        digest = self.get_digest()
        return [f"{c.symbol}/USDT" if "/USDT" not in c.symbol else c.symbol
                for c in digest.trending]

    def _get_cached(self, key: str):
        """获取缓存"""
        now = datetime.now(timezone.utc).timestamp()
        if key in self._cache:
            cached_at = self._cache_time.get(key, 0)
            if now - cached_at < self.CACHE_TTL_SECONDS:
                return self._cache[key]
        return None

    def _set_cache(self, key: str, value):
        """设置缓存"""
        self._cache[key] = value
        self._cache_time[key] = datetime.now(timezone.utc).timestamp()
