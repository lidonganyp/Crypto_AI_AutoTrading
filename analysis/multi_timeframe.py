"""Multi-Timeframe Confluence — 多周期共振检测器

核心原则：只在多个时间维度指向同一个方向时才交易。
1个时间框架的信号 = 噪音
3-4个时间框架对齐 = 高概率机会

时间层级：
  日线 (1D)    → 定方向（大趋势，看山是山）
  4小时 (4H)   → 定节奏（波段结构，山在哪里）
  1小时 (1H)   → 找入场（精确时机，从哪里上）
  15分钟 (15m) → 精确入场点（最后一击）
"""

from __future__ import annotations

from dataclasses import dataclass
from loguru import logger

import pandas as pd
import numpy as np


@dataclass
class TimeframeSignal:
    """单个时间框架的信号"""
    timeframe: str        # "1d", "4h", "1h", "15m"
    direction: str        # "bullish", "bearish", "neutral"
    strength: float       # -1.0 to 1.0
    key_signals: list[str]
    support_level: float | None = None
    resistance_level: float | None = None


@dataclass
class ConfluenceResult:
    """多周期共振结果"""
    overall_direction: str     # "LONG", "SHORT", "FLAT"
    confidence: float          # 0.0-1.0
    aligned_timeframes: list[str]  # 方向一致的时间框架
    conflicting_timeframes: list[str]
    timeframe_signals: list[TimeframeSignal]
    entry_suggestion: str     # 具体入场建议
    is_high_probability: bool


class MultiTimeframeConfluence:
    """多周期共振检测器"""

    def analyze(
        self,
        candles_by_tf: dict[str, list[dict]],
    ) -> ConfluenceResult:
        """
        分析多个时间周期的共振情况

        Args:
            candles_by_tf: {"1d": [...], "4h": [...], "1h": [...], "15m": [...]}
                           每个时间框架的 K 线数据
        """
        signals = {}
        priority_order = ["1d", "4h", "1h", "15m"]

        for tf in priority_order:
            candles = candles_by_tf.get(tf)
            if candles and len(candles) >= 30:
                signals[tf] = self._analyze_single_tf(candles, tf)
                logger.debug(
                    f"{tf}: {signals[tf].direction} "
                    f"(strength={signals[tf].strength:.2f})"
                )

        if not signals:
            return ConfluenceResult(
                overall_direction="FLAT",
                confidence=0.0,
                aligned_timeframes=[],
                conflicting_timeframes=[],
                timeframe_signals=[],
                entry_suggestion="数据不足",
                is_high_probability=False,
            )

        # 统计多空
        bull_tfs = [tf for tf, s in signals.items() if s.direction == "bullish"]
        bear_tfs = [tf for tf, s in signals.items() if s.direction == "bearish"]
        neutral_tfs = [tf for tf, s in signals.items() if s.direction == "neutral"]

        # 加权评分（大周期权重更高）
        weights = {"1d": 3.0, "4h": 2.5, "1h": 2.0, "15m": 1.0}
        total_weight = 0
        weighted_score = 0

        for tf, sig in signals.items():
            w = weights.get(tf, 1.0)
            total_weight += w
            weighted_score += sig.strength * w

        normalized_score = weighted_score / total_weight if total_weight > 0 else 0

        # 决策逻辑
        # 日线方向是最高优先级
        daily_signal = signals.get("1d")
        daily_direction = daily_signal.direction if daily_signal else "neutral"

        # 如果日线看空，几乎不应该做多（除非极端恐慌抄底）
        if daily_direction == "bearish":
            return ConfluenceResult(
                overall_direction="FLAT",
                confidence=0.2,
                aligned_timeframes=bear_tfs,
                conflicting_timeframes=bull_tfs,
                timeframe_signals=list(signals.values()),
                entry_suggestion="日线看空，不建议开多仓。等待日线反转信号。",
                is_high_probability=False,
            )

        # 如果日线看多
        if daily_direction == "bullish":
            # 检查小周期是否有回调到位的信号
            four_h = signals.get("4h")
            one_h = signals.get("1h")

            # 理想入场：日线多 + 4H 回调到位 + 1H 开始反转
            if (
                four_h and four_h.direction == "neutral"
                and one_h and one_h.direction == "bullish"
            ):
                entry = self._build_entry_suggestion(signals)
                return ConfluenceResult(
                    overall_direction="LONG",
                    confidence=min(0.95, 0.6 + normalized_score * 0.2),
                    aligned_timeframes=[tf for tf in ["1d", "1h"] if tf in signals],
                    conflicting_timeframes=[],
                    timeframe_signals=list(signals.values()),
                    entry_suggestion=entry,
                    is_high_probability=True,
                )

            # 全部看多
            if len(bull_tfs) >= 3 and not bear_tfs:
                return ConfluenceResult(
                    overall_direction="LONG",
                    confidence=min(0.9, 0.5 + normalized_score * 0.25),
                    aligned_timeframes=bull_tfs,
                    conflicting_timeframes=[],
                    timeframe_signals=list(signals.values()),
                    entry_suggestion=f"多周期看多共振 ({', '.join(bull_tfs)})，可以逢低做多",
                    is_high_probability=True,
                )

            # 日线多但小周期不确定
            if len(bull_tfs) >= 2:
                return ConfluenceResult(
                    overall_direction="LONG",
                    confidence=0.5,
                    aligned_timeframes=bull_tfs,
                    conflicting_timeframes=neutral_tfs,
                    timeframe_signals=list(signals.values()),
                    entry_suggestion="大方向看多，但小周期信号不够明确，可小仓位试探",
                    is_high_probability=False,
                )

        # 中性/不明
        return ConfluenceResult(
            overall_direction="FLAT",
            confidence=0.2 + abs(normalized_score) * 0.1,
            aligned_timeframes=[],
            conflicting_timeframes=bull_tfs + bear_tfs,
            timeframe_signals=list(signals.values()),
            entry_suggestion="多周期信号不一致，观望等待",
            is_high_probability=False,
        )

    def _analyze_single_tf(
        self, candles: list[dict], timeframe: str
    ) -> TimeframeSignal:
        """分析单个时间框架"""
        df = pd.DataFrame(candles)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        close = df["close"]
        key_signals = []

        # EMA 系统
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema55 = close.ewm(span=55, adjust=False).mean()

        current = close.iloc[-1]
        prev = close.iloc[-2]

        # 趋势判断
        bull_score = 0.0

        # 价格 vs 均线
        if not pd.isna(ema21.iloc[-1]):
            if current > ema21.iloc[-1]:
                bull_score += 1.0
                key_signals.append(f"价格在 EMA21 上方")
            else:
                bull_score -= 1.0
                key_signals.append(f"价格在 EMA21 下方")

        if not pd.isna(ema55.iloc[-1]):
            if current > ema55.iloc[-1]:
                bull_score += 0.5
            else:
                bull_score -= 0.5

        # EMA 交叉
        if not pd.isna(ema9.iloc[-2]) and not pd.isna(ema21.iloc[-2]):
            # 金叉
            if (ema9.iloc[-2] <= ema21.iloc[-2]) and (ema9.iloc[-1] > ema21.iloc[-1]):
                bull_score += 2.0
                key_signals.append("🟢 EMA9 上穿 EMA21 (金叉)")
            # 死叉
            elif (ema9.iloc[-2] >= ema21.iloc[-2]) and (ema9.iloc[-1] < ema21.iloc[-1]):
                bull_score -= 2.0
                key_signals.append("🔴 EMA9 下穿 EMA21 (死叉)")

        # EMA 排列
        if (not pd.isna(ema9.iloc[-1]) and not pd.isna(ema21.iloc[-1])
                and not pd.isna(ema55.iloc[-1])):
            if ema9.iloc[-1] > ema21.iloc[-1] > ema55.iloc[-1]:
                bull_score += 1.0
                key_signals.append("均线多头排列")
            elif ema9.iloc[-1] < ema21.iloc[-1] < ema55.iloc[-1]:
                bull_score -= 1.0
                key_signals.append("均线空头排列")

        # K线形态
        body = abs(current - prev)
        upper_wick = df["high"].iloc[-1] - max(current, prev)
        lower_wick = min(current, prev) - df["low"].iloc[-1]

        # 看涨吞没
        if len(df) >= 2:
            prev_body_top = max(df["open"].iloc[-2], df["close"].iloc[-2])
            prev_body_bot = min(df["open"].iloc[-2], df["close"].iloc[-2])
            curr_body_top = max(df["open"].iloc[-1], df["close"].iloc[-1])
            curr_body_bot = min(df["open"].iloc[-1], df["close"].iloc[-1])

            if (df["close"].iloc[-2] < df["open"].iloc[-2]  # 前一根阴线
                    and df["close"].iloc[-1] > df["open"].iloc[-1]  # 当前阳线
                    and curr_body_bot <= prev_body_bot
                    and curr_body_top >= prev_body_top):
                bull_score += 1.5
                key_signals.append("🟢 看涨吞没形态")

        # 长下影线（锤子线）
        if lower_wick > body * 2:
            bull_score += 1.0
            key_signals.append("🟢 长下影线 (潜在支撑)")

        # 长上影线（射击之星）
        if upper_wick > body * 2:
            bull_score -= 1.0
            key_signals.append("🔴 长上影线 (潜在阻力)")

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]

        if not pd.isna(rsi_val):
            if rsi_val < 30:
                bull_score += 1.5
                key_signals.append(f"RSI 超卖 ({rsi_val:.0f})")
            elif rsi_val < 40:
                bull_score += 0.5
            elif rsi_val > 70:
                bull_score -= 1.5
                key_signals.append(f"RSI 超买 ({rsi_val:.0f})")
            elif rsi_val > 60:
                bull_score -= 0.5

        # 量化方向
        normalized = max(-1.0, min(1.0, bull_score / 4.0))

        if normalized > 0.3:
            direction = "bullish"
        elif normalized < -0.3:
            direction = "bearish"
        else:
            direction = "neutral"

        # 支撑/阻力位（基于近期高低点）
        support = df["low"].iloc[-50:].min() if len(df) >= 50 else df["low"].min()
        resistance = df["high"].iloc[-50:].max() if len(df) >= 50 else df["high"].max()

        return TimeframeSignal(
            timeframe=timeframe,
            direction=direction,
            strength=round(normalized, 3),
            key_signals=key_signals,
            support_level=support,
            resistance_level=resistance,
        )

    def _build_entry_suggestion(
        self, signals: dict[str, TimeframeSignal]
    ) -> str:
        """构建具体入场建议"""
        parts = []

        daily = signals.get("1d")
        four_h = signals.get("4h")
        one_h = signals.get("1h")
        fifteen = signals.get("15m")

        if daily:
            parts.append(f"日线看多")

        if four_h and four_h.direction == "neutral":
            parts.append("4H 回调整理中")
            if four_h.support_level:
                parts.append(f"4H 支撑位 ${four_h.support_level:,.0f}")

        if one_h and one_h.direction == "bullish":
            if one_h.key_signals:
                parts.append(f"1H 出现 {one_h.key_signals[0]}")
            if one_h.support_level:
                parts.append(f"1H 支撑 ${one_h.support_level:,.0f}")

        if fifteen and fifteen.direction == "bullish":
            parts.append("15m 多头确认")

        return "，".join(parts) + " → 可在支撑位附近做多"
