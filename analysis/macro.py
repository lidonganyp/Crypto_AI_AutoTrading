"""Macro Analysis — 宏观指标集成

核心理念 (来源: Arthur Hayes / BitMEX):
- "加密市场是全球流动性的衍生品" — Arthur Hayes
- 美元强弱、M2增速、利率周期是加密市场的大背景
- 宏观指标决定了"大方向"，技术分析决定了"小时机"

跟踪指标:
1. DXY (美元指数) — 美元强 → 加密弱，美元弱 → 加密强
2. M2 货币供应增速 — M2加速 → 流动性宽松 → 加密利好
3. 美联储利率周期 — 降息周期 → 利好，加息周期 → 利空
4. 全球流动性指数 — 综合衡量全球资金宽松程度

数据来源:
- 免费 API: FRED (Federal Reserve Economic Data)
- CoinGlass 宏观数据
- 如 API 不可用，使用历史周期判断

输出: MacroEnvironment — 综合宏观评分，指导系统整体仓位和方向偏好
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from loguru import logger

import requests


@dataclass
class MacroFactor:
    """单个宏观因子"""
    name: str               # 因子名称
    value: float | None     # 当前值
    trend: str              # "rising" / "falling" / "stable"
    impact: str             # "bullish" / "bearish" / "neutral"
    score: float            # -1.0 (极度利空) 到 1.0 (极度利好)
    description: str        # 人可读描述


@dataclass
class MacroEnvironment:
    """宏观环境综合评估"""
    bullish: bool               # 整体是否看多
    overall_score: float        # -1.0 到 1.0 综合评分
    factors: dict[str, MacroFactor] = field(default_factory=dict)
    summary: str = ""           # 人可读总结
    position_adjustment: float  # 建议仓位调整系数 (0.5-1.5)
    timestamp: str = ""


class MacroAnalyzer:
    """
    宏观指标分析器

    用法:
        analyzer = MacroAnalyzer()
        env = analyzer.analyze()
        if env.bullish:
            logger.info(f"宏观看多，评分={env.overall_score:.2f}")
    """

    # FRED API (免费，无需 key)
    FRED_BASE = "https://api.stlouisfed.org/fred"
    FRED_API_KEY = ""  # 可选：填入 FRED API key 获得更高频率

    # DXY 阈值
    DXY_BEARISH = 103.0    # DXY < 103 → 利好加密
    DXY_BULLISH = 107.0    # DXY > 107 → 利空加密

    # 利率状态
    RATE_CUTTING = "cutting"
    RATE_HIKING = "hiking"
    RATE_HOLDING = "holding"

    def __init__(self, storage=None):
        self.storage = storage
        self._cache: dict = {}
        self._cache_time: dict = {}
        self._cache_ttl = 3600  # 缓存 1 小时

    def analyze(self, force_refresh: bool = False) -> MacroEnvironment:
        """
        综合分析宏观环境

        Returns:
            MacroEnvironment 包含所有因子和综合评分
        """
        logger.info("📊 开始宏观环境分析...")

        factors = {}

        # 逐个分析因子
        factors["dxy"] = self._analyze_dxy()
        factors["m2"] = self._analyze_m2()
        factors["rate"] = self._analyze_rate_cycle()
        factors["liquidity"] = self._analyze_global_liquidity()
        factors["crypto_specific"] = self._analyze_crypto_specific()

        # 计算综合评分
        scores = []
        weights = {
            "dxy": 0.25,
            "m2": 0.20,
            "rate": 0.25,
            "liquidity": 0.20,
            "crypto_specific": 0.10,
        }

        for key, factor in factors.items():
            w = weights.get(key, 0.1)
            scores.append(factor.score * w)

        overall = sum(scores) / sum(weights.values())
        overall = max(-1.0, min(1.0, overall))

        bullish = overall > 0.1
        bearish = overall < -0.1
        neutral = not bullish and not bearish

        # 仓位调整系数
        if overall > 0.5:
            pos_adj = 1.3
        elif overall > 0.1:
            pos_adj = 1.1
        elif overall > -0.1:
            pos_adj = 1.0
        elif overall > -0.5:
            pos_adj = 0.8
        else:
            pos_adj = 0.5

        # 生成总结
        summaries = []
        for key, f in factors.items():
            if f.impact != "neutral":
                summaries.append(f"{f.name}: {f.description}")

        if bullish:
            summary = f"🟢 宏观看多 (评分={overall:+.2f})\n" + "\n".join(f"- {s}" for s in summaries)
        elif bearish:
            summary = f"🔴 宏观看空 (评分={overall:+.2f})\n" + "\n".join(f"- {s}" for s in summaries)
        else:
            summary = f"🟡 宏观中性 (评分={overall:+.2f})\n" + "\n".join(f"- {s}" for s in summaries)

        env = MacroEnvironment(
            bullish=bullish,
            overall_score=round(overall, 3),
            factors=factors,
            summary=summary,
            position_adjustment=round(pos_adj, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(f"📊 宏观分析完成: {'看多' if bullish else '看空' if bearish else '中性'} "
                     f"(评分={overall:+.2f}, 仓位系数={pos_adj:.1f})")

        return env

    def _analyze_dxy(self) -> MacroFactor:
        """分析美元指数 (DXY)"""
        # 尝试从 API 获取
        dxy_value = self._fetch_cached("dxy", self._fetch_dxy_from_api)

        if dxy_value is not None:
            if dxy_value < self.DXY_BEARISH:
                trend = "falling" if dxy_value < 100 else "stable"
                return MacroFactor(
                    name="美元指数(DXY)",
                    value=dxy_value, trend=trend,
                    impact="bullish",
                    score=min(1.0, (self.DXY_BEARISH - dxy_value) / 10),
                    description=f"DXY={dxy_value:.1f}，美元走弱利好加密",
                )
            elif dxy_value > self.DXY_BULLISH:
                trend = "rising" if dxy_value > 110 else "stable"
                return MacroFactor(
                    name="美元指数(DXY)",
                    value=dxy_value, trend=trend,
                    impact="bearish",
                    score=max(-1.0, -(dxy_value - self.DXY_BULLISH) / 10),
                    description=f"DXY={dxy_value:.1f}，美元走强利空加密",
                )

        # 无数据时使用中性
        return MacroFactor(
            name="美元指数(DXY)",
            value=dxy_value, trend="stable",
            impact="neutral", score=0.0,
            description="DXY 数据不可用，假设中性",
        )

    def _analyze_m2(self) -> MacroFactor:
        """分析 M2 货币供应增速"""
        m2_growth = self._fetch_cached("m2_growth", self._fetch_m2_from_api)

        if m2_growth is not None:
            if m2_growth > 5.0:
                return MacroFactor(
                    name="M2货币供应",
                    value=m2_growth, trend="rising",
                    impact="bullish",
                    score=min(1.0, m2_growth / 15),
                    description=f"M2增速={m2_growth:.1f}%，流动性扩张利好加密",
                )
            elif m2_growth < 0:
                return MacroFactor(
                    name="M2货币供应",
                    value=m2_growth, trend="falling",
                    impact="bearish",
                    score=max(-1.0, m2_growth / 5),
                    description=f"M2增速={m2_growth:.1f}%，流动性收缩利空加密",
                )

        return MacroFactor(
            name="M2货币供应",
            value=m2_growth, trend="stable",
            impact="neutral", score=0.0,
            description="M2 数据不可用，假设中性",
        )

    def _analyze_rate_cycle(self) -> MacroFactor:
        """分析利率周期"""
        rate = self._fetch_cached("fed_rate", self._fetch_rate_from_api)
        rate_trend = self._fetch_cached("rate_trend", self._fetch_rate_trend_from_api)

        if rate is not None:
            cycle = self._determine_rate_cycle(rate, rate_trend)

            if cycle == self.RATE_CUTTING:
                return MacroFactor(
                    name="美联储利率",
                    value=rate, trend="falling",
                    impact="bullish", score=0.6,
                    description=f"利率={rate:.2f}%，降息周期利好风险资产",
                )
            elif cycle == self.RATE_HIKING:
                return MacroFactor(
                    name="美联储利率",
                    value=rate, trend="rising",
                    impact="bearish", score=-0.6,
                    description=f"利率={rate:.2f}%，加息周期利空风险资产",
                )
            else:
                return MacroFactor(
                    name="美联储利率",
                    value=rate, trend="stable",
                    impact="neutral", score=-0.1,
                    description=f"利率={rate:.2f}%，维持不变",
                )

        return MacroFactor(
            name="美联储利率",
            value=rate, trend="stable",
            impact="neutral", score=0.0,
            description="利率数据不可用，假设中性",
        )

    def _analyze_global_liquidity(self) -> MacroFactor:
        """分析全球流动性（综合指标）"""
        # 综合前面几个因子推算全球流动性
        # 简化版：基于 G4 央行资产负债表总和的同比变化
        g4_change = self._fetch_cached("g4_change", self._fetch_g4_from_api)

        if g4_change is not None:
            if g4_change > 0:
                return MacroFactor(
                    name="全球流动性",
                    value=g4_change, trend="rising",
                    impact="bullish",
                    score=min(1.0, g4_change / 10),
                    description=f"G4央行扩表{g4_change:+.1f}%，全球流动性宽松",
                )
            elif g4_change < -2:
                return MacroFactor(
                    name="全球流动性",
                    value=g4_change, trend="falling",
                    impact="bearish",
                    score=max(-1.0, g4_change / 5),
                    description=f"G4央行缩表{g4_change:+.1f}%，全球流动性收紧",
                )

        return MacroFactor(
            name="全球流动性",
            value=g4_change, trend="stable",
            impact="neutral", score=0.0,
            description="流动性数据不可用，假设中性",
        )

    def _analyze_crypto_specific(self) -> MacroFactor:
        """加密市场特有宏观因素"""
        # BTC 闪电网络活跃度、链上大额转账等（简化版）
        # 这里用占位逻辑，实际可接入 Glassnode/CryptoQuant API
        return MacroFactor(
            name="加密特有因素",
            value=None, trend="stable",
            impact="neutral", score=0.0,
            description="加密特有宏观因素（待接入链上数据源）",
        )

    # ── API 获取方法（带缓存和降级） ──

    def _fetch_cached(self, key: str, fetch_fn) -> float | None:
        """带 TTL 的缓存获取"""
        now = datetime.now(timezone.utc).timestamp()
        if key in self._cache:
            cached_at = self._cache_time.get(key, 0)
            if now - cached_at < self._cache_ttl:
                return self._cache[key]

        try:
            value = fetch_fn()
            if value is not None:
                self._cache[key] = value
                self._cache_time[key] = now
            return value
        except Exception as e:
            logger.warning(f"宏观数据获取失败 [{key}]: {e}")
            return self._cache.get(key)

    def _fetch_dxy_from_api(self) -> float | None:
        """从 FRED 获取 DXY"""
        if not self.FRED_API_KEY:
            # 无 API key，尝试公开数据源
            return self._fetch_from_fred_public("DTWEXBGS")
        try:
            url = f"{self.FRED_BASE}/series/observations"
            params = {
                "series_id": "DTWEXBGS",
                "api_key": self.FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                obs = data.get("observations", [])
                if obs:
                    val = obs[0].get("value")
                    return float(val) if val else None
        except Exception:
            pass
        return None

    def _fetch_m2_from_api(self) -> float | None:
        """从 FRED 获取 M2 同比增速"""
        return self._fetch_from_fred_public("MYAGM2SL")

    def _fetch_rate_from_api(self) -> float | None:
        """获取联邦基金利率"""
        return self._fetch_from_fred_public("FEDFUNDS")

    def _fetch_rate_trend_from_api(self) -> str | None:
        """获取利率趋势"""
        # 简化：对比最近3个月和前3个月
        return None

    def _fetch_g4_from_api(self) -> float | None:
        """获取 G4 央行资产负债表变化"""
        return None

    def _fetch_from_fred_public(self, series_id: str) -> float | None:
        """从 FRED 公开端点获取数据（无 API key 时降级）"""
        try:
            # 尝试从备用公开 API 获取
            # 这里用 CoinGecko 的简化逻辑代替
            # 实际生产环境应接入 FRED API
            return None
        except Exception:
            return None

    def _determine_rate_cycle(self, rate: float, trend: str | None) -> str:
        """判断利率周期"""
        if trend == "falling" or rate < 3.0:
            return self.RATE_CUTTING
        elif trend == "rising" or rate > 5.0:
            return self.RATE_HIKING
        else:
            return self.RATE_HOLDING

    def get_report(self) -> str:
        """生成宏观环境报告"""
        env = self.analyze()
        lines = [
            "## 📊 宏观环境报告",
            f"**综合评分**: {env.overall_score:+.2f} "
            f"({'看多 🟢' if env.bullish else '看空 🔴' if env.overall_score < -0.1 else '中性 🟡'})",
            f"**仓位建议系数**: {env.position_adjustment:.1f}x",
            f"**更新时间**: {env.timestamp}",
            "",
            "### 各因子分析",
        ]

        for key, factor in env.factors.items():
            emoji = "🟢" if factor.score > 0.2 else "🔴" if factor.score < -0.2 else "🟡"
            trend_emoji = {"rising": "📈", "falling": "📉", "stable": "➡️"}.get(factor.trend, "➡️")
            lines.append(
                f"- {emoji} **{factor.name}**: {trend_emoji} {factor.description}"
            )

        lines.append("")
        lines.append("### 总结")
        lines.append(env.summary)

        return "\n".join(lines)
