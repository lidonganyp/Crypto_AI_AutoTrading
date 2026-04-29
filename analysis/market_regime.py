"""Market Regime Detector — 市场状态识别引擎

这是整个系统的地基。不同市场状态用完全不同的策略。
判断错误市场状态 = 用牛市策略在熊市里做多 = 爆仓。

6 种市场状态：
1. BULL_TREND    — 牛市上涨，趋势跟踪
2. BULL_CONSOL   — 牛市震荡，区间交易
3. BEAR_TREND    — 熊市下跌，空仓或对冲
4. BEAR_RALLY    — 熊市反弹，短线博反弹
5. EXTREME_FEAR  — 极端恐慌，逆向抄底（极小仓位）
6. EXTREME_GREED — 极端贪婪，减仓锁定利润
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
from loguru import logger

import pandas as pd
import numpy as np


@dataclass
class MarketRegime:
    """市场状态"""
    state: str          # BULL_TREND, BEAR_TREND, etc.
    confidence: float   # 0.0-1.0
    trend_strength: float  # -1.0 to 1.0 (负=熊, 正=牛)
    volatility: float   # 当前波动率水平 (低/中/高)
    key_factors: list[str]
    suggested_action: str  # 这个状态下系统应该做什么
    max_position_pct: float  # 这个状态下最大允许仓位
    time_horizon: str  # 建议持仓周期: "scalp" / "swing" / "position" / "none"


class MarketRegimeDetector:
    """市场状态检测器 — 综合多维信号判断当前市场状态"""

    # 均线参数
    EMA_FAST = 9
    EMA_MID = 21
    EMA_SLOW = 55
    EMA_MACRO = 200  # 牛熊分界线

    def detect(self, daily_candles: list[dict],
               fear_greed: float | None = None,
               funding_rate: float | None = None,
               exchange_flow: str | None = None) -> MarketRegime:
        """
        综合判断市场状态

        Args:
            daily_candles: 日线 K 线数据 (至少 200 根)
            fear_greed: 恐惧贪婪指数 0-100
            funding_rate: 资金费率 (正=多头付费)
            exchange_flow: "inflow" / "outflow" / None
        """
        if len(daily_candles) < 60:
            logger.warning(f"Only {len(daily_candles)} daily candles, need 60+")
            return MarketRegime(
                state="UNKNOWN", confidence=0.3,
                trend_strength=0, volatility=0.5,
                key_factors=["数据不足"],
                suggested_action="等待数据积累",
                max_position_pct=0.10,
                time_horizon="none",
            )

        df = pd.DataFrame(daily_candles)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        close = df["close"]
        factors = []
        scores = {"bull": 0.0, "bear": 0.0, "extreme_fear": 0.0,
                  "extreme_greed": 0.0, "volatile": 0.0}

        # ── 1. 均线系统（权重最高）──
        ema9 = close.ewm(span=self.EMA_FAST, adjust=False).mean()
        ema21 = close.ewm(span=self.EMA_MID, adjust=False).mean()
        ema55 = close.ewm(span=self.EMA_SLOW, adjust=False).mean()
        ema200 = close.ewm(span=self.EMA_MACRO, adjust=False).mean()

        current_price = close.iloc[-1]

        # 价格 vs 200 日均线（牛熊核心指标）
        if not pd.isna(ema200.iloc[-1]):
            price_vs_200 = (current_price - ema200.iloc[-1]) / ema200.iloc[-1] * 100
            if price_vs_200 > 20:
                scores["bull"] += 3.0
                factors.append(f"价格高于200MA {price_vs_200:+.1f}% (强势牛市)")
            elif price_vs_200 > 5:
                scores["bull"] += 2.0
                factors.append(f"价格高于200MA {price_vs_200:+.1f}% (温和牛市)")
            elif price_vs_200 > -5:
                scores["bear"] += 1.0
                factors.append(f"价格接近200MA {price_vs_200:+.1f}% (震荡)")
            elif price_vs_200 > -20:
                scores["bear"] += 2.5
                factors.append(f"价格低于200MA {price_vs_200:+.1f}% (熊市)")
            else:
                scores["bear"] += 4.0
                factors.append(f"价格远低于200MA {price_vs_200:+.1f}% (深度熊市)")

        # 均线排列
        aligned_bull = ema9.iloc[-1] > ema21.iloc[-1] > ema55.iloc[-1]
        aligned_bear = ema9.iloc[-1] < ema21.iloc[-1] < ema55.iloc[-1]
        if aligned_bull:
            scores["bull"] += 2.0
            factors.append("均线多头排列 (EMA9>21>55)")
        elif aligned_bear:
            scores["bear"] += 2.0
            factors.append("均线空头排列 (EMA9<21<55)")

        # ── 2. 趋势强度 (ADX 逻辑简化版) ──
        high_low_range = df["high"].iloc[-20:] - df["low"].iloc[-20:]
        avg_range = high_low_range.mean()
        price_range_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100

        if abs(price_range_20) > 10 and avg_range < high_low_range.iloc[-1] * 1.5:
            if price_range_20 > 0:
                scores["bull"] += 1.5
                factors.append(f"20日涨幅 {price_range_20:+.1f}% (趋势上涨)")
            else:
                scores["bear"] += 1.5
                factors.append(f"20日跌幅 {price_range_20:+.1f}% (趋势下跌)")

        # ── 3. 波动率 ──
        returns = close.pct_change().dropna()
        volatility_20 = returns.iloc[-20:].std() * np.sqrt(365) * 100
        volatility_60 = returns.iloc[-60:].std() * np.sqrt(365) * 100
        vol_ratio = volatility_20 / volatility_60 if volatility_60 > 0 else 1.0

        if vol_ratio > 2.0:
            scores["volatile"] += 3.0
            factors.append(f"波动率急剧升高 (当前{volatility_20:.0f}% vs 60日均{volatility_60:.0f}%)")
        elif vol_ratio > 1.5:
            scores["volatile"] += 1.5
            factors.append(f"波动率偏高 (比均值高{vol_ratio:.1f}倍)")

        # ── 4. 成交量确认 ──
        vol_avg_20 = df["volume"].iloc[-20:].mean()
        vol_avg_60 = df["volume"].iloc[-60:].mean()
        vol_change = (vol_avg_20 / vol_avg_60 - 1) * 100 if vol_avg_60 > 0 else 0

        if vol_change > 50:
            factors.append(f"成交量放大 {(vol_change):+.0f}% (资金活跃)")
        elif vol_change < -30:
            factors.append(f"成交量萎缩 {vol_change:+.0f}% (资金观望)")

        # 量价背离检测
        recent_high = close.iloc[-5:].max()
        prev_high = close.iloc[-20:-5].max() if len(close) > 20 else recent_high
        recent_vol = df["volume"].iloc[-5:].mean()
        prev_vol = df["volume"].iloc[-20:-5].mean() if len(close) > 20 else recent_vol
        if close.iloc[-1] > prev_high and recent_vol < prev_vol * 0.8:
            factors.append("⚠️ 量价背离: 价格新高但成交量萎缩")

        # ── 5. 恐惧贪婪指数 ──
        if fear_greed is not None:
            if fear_greed <= 10:
                scores["extreme_fear"] += 3.0
                factors.append(f"恐惧贪婪指数 {fear_greed} (极度恐慌)")
            elif fear_greed <= 25:
                scores["extreme_fear"] += 1.5
                factors.append(f"恐惧贪婪指数 {fear_greed} (恐慌)")
            elif fear_greed >= 90:
                scores["extreme_greed"] += 3.0
                factors.append(f"恐惧贪婪指数 {fear_greed} (极度贪婪)")
            elif fear_greed >= 75:
                scores["extreme_greed"] += 1.5
                factors.append(f"恐惧贪婪指数 {fear_greed} (贪婪)")

        # ── 6. 资金费率 ──
        if funding_rate is not None:
            if funding_rate > 0.05:
                scores["extreme_greed"] += 2.0
                factors.append(f"资金费率 {funding_rate:.4f} (多头极度拥挤)")
            elif funding_rate < -0.01:
                scores["extreme_fear"] += 1.5
                factors.append(f"资金费率 {funding_rate:.4f} (空头付费)")

        # ── 7. 交易所出入金 ──
        if exchange_flow == "outflow":
            scores["bull"] += 1.5
            factors.append("BTC 从交易所大量流出 (大户囤币)")
        elif exchange_flow == "inflow":
            scores["bear"] += 1.5
            factors.append("BTC 流入交易所 (大户抛售)")

        # ════════════════════════════════════════
        # 综合决策
        # ════════════════════════════════════════

        max_score = max(scores.values())
        total_score = sum(scores.values())

        # 极端状态优先判断
        if scores["extreme_fear"] >= 3.0 and scores["extreme_greed"] < 1.0:
            return MarketRegime(
                state="EXTREME_FEAR",
                confidence=min(0.9, 0.5 + scores["extreme_fear"] / 10),
                trend_strength=-0.5,
                volatility=min(1.0, vol_ratio / 2),
                key_factors=factors,
                suggested_action="分批抄底，极小仓位（每批≤5%），严格止损",
                max_position_pct=0.15,
                time_horizon="position",
            )

        if scores["extreme_greed"] >= 3.0 and scores["extreme_fear"] < 1.0:
            return MarketRegime(
                state="EXTREME_GREED",
                confidence=min(0.9, 0.5 + scores["extreme_greed"] / 10),
                trend_strength=0.5,
                volatility=min(1.0, vol_ratio / 2),
                key_factors=factors,
                suggested_action="逐步减仓锁定利润，不开新仓",
                max_position_pct=0.0,
                time_horizon="none",
            )

        # 牛熊判断
        if scores["bull"] > scores["bear"] * 1.5:
            # 牛市
            trend_strength = min(1.0, scores["bull"] / 8)
            volatility_level = min(1.0, vol_ratio / 2)

            if scores["volatile"] >= 2.0:
                state = "BULL_CONSOL"
                action = "牛市震荡期，高抛低吸或观望等待突破"
                horizon = "swing"
                max_pos = 0.20
            else:
                state = "BULL_TREND"
                action = "趋势跟踪，逢回调做多，让利润奔跑"
                horizon = "position"
                max_pos = 0.30

            return MarketRegime(
                state=state,
                confidence=min(0.9, 0.4 + trend_strength * 0.3),
                trend_strength=trend_strength,
                volatility=volatility_level,
                key_factors=factors,
                suggested_action=action,
                max_position_pct=max_pos,
                time_horizon=horizon,
            )

        elif scores["bear"] > scores["bull"] * 1.5:
            # 熊市
            trend_strength = -min(1.0, scores["bear"] / 8)

            if scores["volatile"] >= 2.0:
                state = "BEAR_RALLY"
                action = "熊市超跌反弹，短线博反弹，严格止损"
                horizon = "scalp"
                max_pos = 0.10
            else:
                state = "BEAR_TREND"
                action = "熊市下跌趋势，空仓观望或对冲"
                horizon = "none"
                max_pos = 0.0

            return MarketRegime(
                state=state,
                confidence=min(0.9, 0.4 + abs(trend_strength) * 0.3),
                trend_strength=trend_strength,
                volatility=min(1.0, vol_ratio / 2),
                key_factors=factors,
                suggested_action=action,
                max_position_pct=max_pos,
                time_horizon=horizon,
            )

        else:
            # 震荡/不明
            return MarketRegime(
                state="UNKNOWN",
                confidence=0.3,
                trend_strength=0,
                volatility=min(1.0, vol_ratio / 2),
                key_factors=factors + ["多空信号不明确"],
                suggested_action="信号不明确，降低仓位或观望",
                max_position_pct=0.15,
                time_horizon="swing",
            )
