"""Technical indicator analysis"""
from __future__ import annotations

import numpy as np
from loguru import logger

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger.warning("pandas-ta not installed, using fallback indicators")

import pandas as pd


class TechnicalAnalyzer:
    """技术指标计算"""

    @staticmethod
    def calculate_all(candles: list[dict]) -> dict:
        """计算所有技术指标

        Args:
            candles: OHLCV 数据列表，每个元素含 timestamp, open, high,
                     low, close, volume

        Returns:
            指标字典
        """
        if not candles or len(candles) < 20:
            return {}

        df = pd.DataFrame(candles)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 确保 float 类型
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        result = {}

        try:
            if HAS_PANDAS_TA:
                result.update(TechnicalAnalyzer._pandas_ta_indicators(df))
            else:
                result.update(TechnicalAnalyzer._manual_indicators(df))
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        # 额外的自定义指标
        result.update(TechnicalAnalyzer._custom_indicators(df))

        return result

    @staticmethod
    def _pandas_ta_indicators(df: pd.DataFrame) -> dict:
        """使用 pandas-ta 计算指标"""
        result = {}

        # RSI
        rsi = ta.rsi(df["close"], length=14)
        if rsi is not None and not rsi.isna().all():
            result["RSI_14"] = round(rsi.iloc[-1], 2)
            result["RSI_status"] = TechnicalAnalyzer._rsi_status(
                result["RSI_14"]
            )

        # MACD
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            result["MACD"] = round(macd["MACD_12_26_9"].iloc[-1], 4)
            result["MACD_signal"] = round(
                macd["MACDs_12_26_9"].iloc[-1], 4
            )
            result["MACD_hist"] = round(
                macd["MACDh_12_26_9"].iloc[-1], 4
            )
            result["MACD_cross"] = (
                "bullish" if result["MACD"] > result["MACD_signal"]
                else "bearish"
            )

        # EMA
        for period in [9, 21, 55]:
            ema = ta.ema(df["close"], length=period)
            if ema is not None and not ema.isna().all():
                result[f"EMA_{period}"] = round(ema.iloc[-1], 2)

        # Bollinger Bands
        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is not None:
            # pandas-ta 新版列名为三段式: BBU_20_2.0_2.0
            upper_col = [c for c in bb.columns if c.startswith("BBU")][0]
            middle_col = [c for c in bb.columns if c.startswith("BBM")][0]
            lower_col = [c for c in bb.columns if c.startswith("BBL")][0]
            bb_upper = bb[upper_col].iloc[-1]
            bb_middle = bb[middle_col].iloc[-1]
            bb_lower = bb[lower_col].iloc[-1]
            if not (pd.isna(bb_upper) or pd.isna(bb_lower)):
                result["BB_upper"] = round(bb_upper, 2)
                result["BB_middle"] = round(bb_middle, 2)
                result["BB_lower"] = round(bb_lower, 2)
                close = df["close"].iloc[-1]
                result["BB_position"] = TechnicalAnalyzer._bb_position(
                    close, result["BB_upper"], result["BB_lower"]
                )

        # ATR
        atr = ta.atr(df["high"], df["low"], df["close"], length=14)
        if atr is not None and not atr.isna().all():
            result["ATR_14"] = round(atr.iloc[-1], 2)

        return result

    @staticmethod
    def _manual_indicators(df: pd.DataFrame) -> dict:
        """手动计算核心指标（pandas-ta 不可用时的 fallback）"""
        result = {}
        close = df["close"]

        # RSI (14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]
        if not pd.isna(rsi_val):
            result["RSI_14"] = round(rsi_val, 2)
            result["RSI_status"] = TechnicalAnalyzer._rsi_status(
                result["RSI_14"]
            )

        # EMA 9/21/55
        for period in [9, 21, 55]:
            ema = close.ewm(span=period, adjust=False).mean()
            result[f"EMA_{period}"] = round(ema.iloc[-1], 2)

        # ATR (14)
        high, low = df["high"], df["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_val = atr.iloc[-1]
        if not pd.isna(atr_val):
            result["ATR_14"] = round(atr_val, 2)

        return result

    @staticmethod
    def _custom_indicators(df: pd.DataFrame) -> dict:
        """自定义指标"""
        result = {}
        close = df["close"]
        volume = df["volume"]

        # 价格变化
        result["price_change_1h"] = round(
            close.pct_change(1).iloc[-1] * 100, 2
        )
        result["price_change_24h"] = round(
            close.pct_change(24).iloc[-1] * 100, 2
        )

        # 成交量变化
        result["volume_change"] = round(
            volume.pct_change(1).iloc[-1] * 100, 2
        )
        result["volume_vs_avg_20"] = round(
            (volume.iloc[-1] / volume.iloc[-20:].mean() - 1) * 100, 2
        )

        # EMA 交叉
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        result["EMA_cross_9_21"] = (
            "bullish" if ema9.iloc[-1] > ema21.iloc[-1] else "bearish"
        )

        # 当前价格
        result["current_price"] = round(close.iloc[-1], 2)

        return result

    @staticmethod
    def _rsi_status(rsi: float) -> str:
        if rsi > 70:
            return "overbought"
        elif rsi > 55:
            return "bullish"
        elif rsi < 30:
            return "oversold"
        elif rsi < 45:
            return "bearish"
        return "neutral"

    @staticmethod
    def _bb_position(price: float, upper: float,
                     lower: float) -> str:
        if upper == lower:
            return "middle"
        pct = (price - lower) / (upper - lower)
        if pct > 0.8:
            return "near_upper"
        elif pct < 0.2:
            return "near_lower"
        return "middle"

    @staticmethod
    def generate_signal(technical: dict) -> dict:
        """根据技术指标生成简单信号

        Returns:
            {"direction": "LONG"/"FLAT", "confidence": float, "reasons": [...]}
        """
        if not technical:
            return {"direction": "FLAT", "confidence": 0, "reasons": []}

        reasons = []
        score = 0  # -100 to 100

        rsi = technical.get("RSI_14")
        if rsi is not None:
            if rsi < 30:
                score += 30
                reasons.append(f"RSI 超卖 ({rsi:.1f})")
            elif rsi < 45:
                score += 10
                reasons.append(f"RSI 偏低 ({rsi:.1f})")
            elif rsi > 70:
                score -= 30
                reasons.append(f"RSI 超买 ({rsi:.1f})")
            elif rsi > 55:
                score -= 10

        macd_cross = technical.get("MACD_cross")
        if macd_cross == "bullish":
            score += 20
            reasons.append("MACD 金叉")
        elif macd_cross == "bearish":
            score -= 20
            reasons.append("MACD 死叉")

        ema_cross = technical.get("EMA_cross_9_21")
        if ema_cross == "bullish":
            score += 15
            reasons.append("EMA9 > EMA21")
        elif ema_cross == "bearish":
            score -= 15
            reasons.append("EMA9 < EMA21")

        bb_pos = technical.get("BB_position")
        if bb_pos == "near_lower":
            score += 15
            reasons.append("价格触及布林下轨")
        elif bb_pos == "near_upper":
            score -= 15
            reasons.append("价格触及布林上轨")

        vol_change = technical.get("volume_vs_avg_20")
        if vol_change is not None and vol_change > 50:
            score += 10
            reasons.append(f"成交量放大 ({vol_change:+.1f}%)")

        # 转换
        if score > 30:
            direction = "LONG"
            confidence = min(0.7, 0.4 + score / 200)
        elif score > 0:
            direction = "LONG"
            confidence = min(0.5, 0.3 + score / 200)
        else:
            direction = "FLAT"
            confidence = 0.3

        return {
            "direction": direction,
            "confidence": round(confidence, 2),
            "reasons": reasons,
        }
