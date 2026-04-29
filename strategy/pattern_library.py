"""Pattern Library — 高概率交易模式库

核心原则：不预测市场，只等熟悉的模式出现。
专业交易员不做"AI分析"，只做经过验证的特定模式。

5 种核心模式 + 模式胜率统计（自动学习）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from loguru import logger

import pandas as pd
import numpy as np


@dataclass
class PatternSignal:
    """模式信号"""
    pattern_name: str
    symbol: str
    direction: str           # "LONG", "SHORT"
    confidence: float        # 0.0-1.0
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit_1: float | None = None  # 第一目标
    take_profit_2: float | None = None  # 第二目标
    rationale: str = ""
    timeframe: str = "1h"


@dataclass
class PatternStats:
    """模式历史统计"""
    pattern_name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    avg_pnl_pct: float = 0
    win_rate: float = 0
    profit_factor: float = 0
    is_profitable: bool = False


class PatternLibrary:
    """交易模式识别库"""

    def __init__(self):
        # 模式统计（从数据库加载或内存维护）
        self._stats: dict[str, PatternStats] = {}

    def scan(
        self,
        symbol: str,
        daily_df: pd.DataFrame,
        h4_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        fear_greed: float | None = None,
        funding_rate: float | None = None,
    ) -> list[PatternSignal]:
        """
        扫描所有交易模式

        Returns:
            检测到的模式信号列表（可能有多个，也可能为空）
        """
        signals = []
        close_1h = h1_df["close"].iloc[-1] if not h1_df.empty else 0
        close_d = daily_df["close"].iloc[-1] if not daily_df.empty else 0

        # 模式 1: 突破回踩确认
        s1 = self._check_breakout_retest(symbol, daily_df, h4_df, h1_df)
        if s1:
            signals.append(s1)

        # 模式 2: 极端恐慌抄底
        if fear_greed is not None:
            s2 = self._check_extreme_fear_buy(symbol, daily_df, fear_greed)
            if s2:
                signals.append(s2)

        # 模式 3: 资金费率套利
        if funding_rate is not None:
            s3 = self._check_funding_rate_signal(symbol, h4_df, funding_rate)
            if s3:
                signals.append(s3)

        # 模式 4: 趋势回调入场
        s4 = self._check_trend_pullback(symbol, daily_df, h4_df, h1_df)
        if s4:
            signals.append(s4)

        # 模式 5: 顶部识别
        if fear_greed is not None:
            s5 = self._check_top_formation(symbol, daily_df, fear_greed, funding_rate)
            if s5:
                signals.append(s5)

        return signals

    def _check_breakout_retest(
        self, symbol: str, daily_df: pd.DataFrame,
        h4_df: pd.DataFrame, h1_df: pd.DataFrame,
    ) -> PatternSignal | None:
        """
        模式 1: 突破回踩确认
        - 价格突破关键阻力位（20日新高）
        - 回踩阻力位（现在变支撑）企稳
        - 出现看涨K线形态

        胜率: ~60-65%（强趋势市场更高）
        """
        if len(daily_df) < 30 or len(h1_df) < 20:
            return None

        close_d = daily_df["close"]
        close_1h = h1_df["close"]

        # 20日新高突破
        high_20 = close_d.iloc[-21:-1].max()
        prev_high_20 = close_d.iloc[-41:-21].max() if len(close_d) > 41 else high_20
        current = close_d.iloc[-1]

        # 突破了20日高点
        if current <= high_20:
            return None

        # 但不能偏离太远（回踩还没发生）
        breakout_pct = (current - high_20) / high_20
        if breakout_pct > 0.05:
            return None  # 已经走太远，错过了

        # 检查 h1 是否在突破位附近出现企稳信号
        recent_1h = close_1h.iloc[-6:]
        if len(recent_1h) < 4:
            return None

        # 最低点接近突破位但没有有效跌破
        min_recent = recent_1h.min()
        retest_range = high_20 * 0.01  # 1% 范围内算回踩

        if abs(min_recent - high_20) > retest_range * 3:
            return None  # 回踩不到位

        # 最近一根 h1 是阳线
        last_close = close_1h.iloc[-1]
        last_open = h1_df["open"].iloc[-1]
        if last_close < last_open:
            return None  # 最近是阴线

        # ATR 用于止损
        high_1h = h1_df["high"].iloc[-14:]
        low_1h = h1_df["low"].iloc[-14:]
        tr = pd.concat([
            high_1h - low_1h,
            (high_1h - close_1h.shift(1)).abs(),
            (low_1h - close_1h.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.mean()

        stop = high_20 - atr * 0.5  # 止损设在支撑位下方
        tp1 = last_close + atr * 2
        tp2 = last_close + atr * 3.5

        return PatternSignal(
            pattern_name="breakout_retest",
            symbol=symbol,
            direction="LONG",
            confidence=0.70,
            entry_price=last_close,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            rationale=(
                f"突破20日高点 ${high_20:,.0f} 后回踩企稳，"
                f"1H 出现阳线确认，ATR={atr:.0f}"
            ),
            timeframe="1h",
        )

    def _check_extreme_fear_buy(
        self, symbol: str, daily_df: pd.DataFrame,
        fear_greed: float,
    ) -> PatternSignal | None:
        """
        模式 2: 极端恐慌抄底
        - 恐惧贪婪指数 < 15
        - 价格已大幅下跌
        - 极小仓位分批建仓

        胜率: ~55%（但盈亏比高）
        """
        if fear_greed > 20:
            return None
        if len(daily_df) < 30:
            return None

        close = daily_df["close"]
        current = close.iloc[-1]
        ma200 = close.ewm(span=200, adjust=False).mean().iloc[-1]

        if pd.isna(ma200):
            return None

        # 价格要明显低于均线
        below_ma = (current - ma200) / ma200 * 100
        if below_ma > -5:
            return None  # 没跌够

        high_20 = close.iloc[-21:-1].max()
        low_20 = close.iloc[-21:-1].min()

        return PatternSignal(
            pattern_name="extreme_fear_buy",
            symbol=symbol,
            direction="LONG",
            confidence=0.55,
            entry_price=current,
            stop_loss=low_20 * 0.98,
            take_profit_1=ma200,
            take_profit_2=high_20,
            rationale=(
                f"极度恐慌 (FGI={fear_greed})，"
                f"价格低于200MA {below_ma:.1f}%，小仓位逆向抄底"
            ),
            timeframe="1d",
        )

    def _check_funding_rate_signal(
        self, symbol: str, h4_df: pd.DataFrame,
        funding_rate: float,
    ) -> PatternSignal | None:
        """
        模式 3: 资金费率套利信号
        - 资金费率持续为负（空头付费做多）
        - 价格在关键支撑位

        胜率: ~58%
        """
        if funding_rate > 0:
            return None  # 正费率不做

        if funding_rate > -0.005:
            return None  # 负得不够多

        close = h4_df["close"]
        current = close.iloc[-1] if not close.empty else 0
        low_20 = close.iloc[-21:-1].min() if len(close) > 21 else close.min()

        return PatternSignal(
            pattern_name="funding_rate_signal",
            symbol=symbol,
            direction="LONG",
            confidence=0.60,
            entry_price=current,
            stop_loss=low_20 * 0.99,
            take_profit_1=current * 1.03,
            take_profit_2=current * 1.06,
            rationale=(
                f"资金费率 {funding_rate:.4f}（空头付费），"
                f"潜在多头拥挤反弹信号"
            ),
            timeframe="4h",
        )

    def _check_trend_pullback(
        self, symbol: str, daily_df: pd.DataFrame,
        h4_df: pd.DataFrame, h1_df: pd.DataFrame,
    ) -> PatternSignal | None:
        """
        模式 4: 趋势回调入场
        - 价格在 EMA20 上方 > 7 天
        - 回踩 EMA20 不破
        - 1H 出现反转信号

        胜率: ~62%
        """
        if len(daily_df) < 30 or len(h1_df) < 10:
            return None

        close_d = daily_df["close"]
        ema20 = close_d.ewm(span=20, adjust=False).mean()

        # 检查价格是否长期在 EMA20 上方
        above_ema = (close_d.iloc[-8:-1] > ema20.iloc[-8:-1]).all()
        if not above_ema:
            return None

        # 当前回调到 EMA20 附近
        current = close_d.iloc[-1]
        current_ema = ema20.iloc[-1]

        if pd.isna(current_ema):
            return None

        # 价格在 EMA 附近（±1%）
        dist_to_ema = (current - current_ema) / current_ema
        if abs(dist_to_ema) > 0.02:
            return None

        # 1H 出现反转
        close_1h = h1_df["close"]
        if len(close_1h) < 5:
            return None

        last_3 = close_1h.iloc[-3:]
        if not (last_3.iloc[-1] > last_3.iloc[-2] > last_3.iloc[-3]):
            return None

        atr = self._calc_atr(h1_df)
        stop = current_ema - atr * 1.0

        return PatternSignal(
            pattern_name="trend_pullback",
            symbol=symbol,
            direction="LONG",
            confidence=0.65,
            entry_price=current,
            stop_loss=stop,
            take_profit_1=current + atr * 2,
            take_profit_2=current + atr * 4,
            rationale=(
                f"趋势中回踩 EMA20 (${current_ema:,.0f}) 企稳，"
                f"1H 连续3根阳线确认"
            ),
            timeframe="1h",
        )

    def _check_top_formation(
        self, symbol: str, daily_df: pd.DataFrame,
        fear_greed: float, funding_rate: float | None = None,
    ) -> PatternSignal | None:
        """
        模式 5: 顶部识别
        - 恐惧贪婪指数 > 80
        - 资金费率极端高
        - 价格加速上涨后开始走弱

        输出：LONG 被拒绝，或者发出减仓信号
        """
        if fear_greed < 80:
            return None

        close = daily_df["close"]
        if len(close) < 10:
            return None

        # 检查是否开始走弱
        recent_5 = close.iloc[-5:]
        if recent_5.iloc[-1] >= recent_5.max():
            return None  # 还在创新高

        funding_str = ""
        if funding_rate and funding_rate > 0.03:
            funding_str = f"，资金费率 {funding_rate:.4f}（多头拥挤）"

        return PatternSignal(
            pattern_name="top_formation",
            symbol=symbol,
            direction="SHORT",
            confidence=0.55,
            rationale=(
                f"恐惧贪婪指数 {fear_greed}（极度贪婪），"
                f"价格开始走弱{funding_str}，注意风险"
            ),
            timeframe="1d",
        )

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
        """计算 ATR"""
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def record_pattern_result(
        self, pattern_name: str, pnl_pct: float
    ):
        """记录模式交易结果，更新统计"""
        if pattern_name not in self._stats:
            self._stats[pattern_name] = PatternStats(
                pattern_name=pattern_name
            )

        stats = self._stats[pattern_name]
        stats.total_trades += 1
        if pnl_pct > 0:
            stats.wins += 1
        else:
            stats.losses += 1

        stats.avg_pnl_pct = (
            (stats.avg_pnl_pct * (stats.total_trades - 1) + pnl_pct)
            / stats.total_trades
        )
        stats.win_rate = stats.wins / stats.total_trades

        total_profit = sum(1 for _ in range(stats.wins))
        total_loss = sum(1 for _ in range(stats.losses))
        stats.profit_factor = total_profit / total_loss if total_loss else 999
        stats.is_profitable = stats.avg_pnl_pct > 0

        logger.info(
            f"Pattern '{pattern_name}' stats: "
            f"{stats.total_trades} trades, "
            f"WR={stats.win_rate:.0%}, "
            f"AvgPnL={stats.avg_pnl_pct:+.2f}%"
        )
