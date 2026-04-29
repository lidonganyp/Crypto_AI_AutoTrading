"""Feature engineering for CryptoAI v3."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone

import pandas as pd

from core.models import FeatureSnapshot


@dataclass
class FeatureInput:
    symbol: str
    candles_1h: list[dict]
    candles_4h: list[dict]
    candles_1d: list[dict]
    funding_rate: float | None = None
    bid_ask_spread_pct: float | None = None
    bid_notional_top5: float | None = None
    ask_notional_top5: float | None = None
    depth_imbalance: float | None = None
    large_order_net_notional: float | None = None
    sentiment_value: float | None = None
    llm_sentiment_score: float | None = None
    lunarcrush_sentiment: float | None = None
    market_regime_score: float | None = None
    onchain_netflow_score: float | None = None
    onchain_whale_score: float | None = None


class FeaturePipeline:
    """Build a richer, model-friendly feature set."""

    def build(self, payload: FeatureInput) -> FeatureSnapshot:
        one_h = self._frame(payload.candles_1h)
        four_h = self._frame(payload.candles_4h)
        one_d = self._frame(payload.candles_1d)

        valid = not one_h.empty and not four_h.empty and not one_d.empty
        if not valid:
            return FeatureSnapshot(
                symbol=payload.symbol,
                timeframe="4h",
                values={},
                valid=False,
            )

        adx_4h, di_plus_4h, di_minus_4h = self._adx_dmi(four_h)
        atr_4h = self._atr(four_h)
        atr_ratio_4h = atr_4h / max(float(four_h["close"].iloc[-1]), 1e-9)
        atr_percentile_4h = self._atr_percentile(four_h)
        support_1d, resistance_1d = self._support_resistance(one_d)
        fib_position_1d, fib_382_1d, fib_618_1d = self._fibonacci_metrics(one_d)
        ma5_1h = self._rolling_last(one_h["close"], 5)
        ma10_1h = self._rolling_last(one_h["close"], 10)
        ma20_1h = self._rolling_last(one_h["close"], 20)
        ma20_4h = self._rolling_last(four_h["close"], 20)
        ma50_4h = self._rolling_last(four_h["close"], 50)
        ma100_4h = self._rolling_last(four_h["close"], 100)
        ma200_1d = self._rolling_last(one_d["close"], 200)
        close_1h = float(one_h["close"].iloc[-1])
        close_4h = float(four_h["close"].iloc[-1])
        close_1d = float(one_d["close"].iloc[-1])
        obv_1h = self._obv(one_h)
        obv_4h = self._obv(four_h)
        volatility_20d = self._volatility(one_d["close"])
        llm_sentiment = float(payload.llm_sentiment_score or 0.0)
        fear_greed_sentiment = float(payload.sentiment_value or 0.0)
        dmi_spread = di_plus_4h - di_minus_4h
        trend_alignment_score = self._clamp(
            (
                int(close_4h > ma50_4h)
                + int(close_1d > ma200_1d)
                + int(di_plus_4h > di_minus_4h)
            )
            / 3.0,
            0.0,
            1.0,
        )
        price_structure_score = self._clamp(
            (1.0 - min(max(self._distance_to_level(close_1d, support_1d), 0.0), 1.0))
            * 0.4
            + (1.0 - min(max(self._distance_to_level(resistance_1d, close_1d), 0.0), 1.0))
            * 0.2
            + fib_position_1d * 0.4,
            0.0,
            1.0,
        )
        volatility_regime_score = self._clamp(1.0 - abs(atr_percentile_4h - 0.5) * 2, 0.0, 1.0)
        directional_conviction = self._clamp((adx_4h / 100.0) * (dmi_spread / 100.0), -1.0, 1.0)
        black_swan_index = self._clamp(
            max(
                abs(self._pct_change(one_h["close"], 1)) * 8.0,
                abs(self._pct_change(four_h["close"], 1)) * 4.0,
                atr_ratio_4h * 10.0,
                volatility_20d * 25.0,
            ),
            0.0,
            1.0,
        )

        values = {
            "close_1h": close_1h,
            "ma5_1h": ma5_1h,
            "ma10_1h": ma10_1h,
            "ma20_1h": ma20_1h,
            "ma5_gap_1h": self._gap_ratio(close_1h, ma5_1h),
            "ma10_gap_1h": self._gap_ratio(close_1h, ma10_1h),
            "ma20_gap_1h": self._gap_ratio(close_1h, ma20_1h),
            "rsi_1h": self._rsi(one_h["close"]),
            "macd_1h": self._macd(one_h["close"]),
            "obv_1h": obv_1h,
            "obv_slope_1h": self._slope_ratio(one_h["close"], one_h["volume"], 10),
            "volume_ratio_1h": self._volume_ratio(one_h),
            "volume_zscore_1h": self._volume_zscore(one_h),
            "volume_trend_1h": self._volume_trend(one_h),
            "high_low_range_1h": self._range_ratio(one_h),
            "return_1h": self._pct_change(one_h["close"], 1),
            "funding_rate": float(payload.funding_rate or 0.0),
            "bid_ask_spread_pct": float(payload.bid_ask_spread_pct or 0.0),
            "bid_notional_top5": float(payload.bid_notional_top5 or 0.0),
            "ask_notional_top5": float(payload.ask_notional_top5 or 0.0),
            "top5_depth_notional": float(payload.bid_notional_top5 or 0.0)
            + float(payload.ask_notional_top5 or 0.0),
            "depth_imbalance": float(payload.depth_imbalance or 0.0),
            "large_order_net_notional": float(payload.large_order_net_notional or 0.0),
            "close_4h": close_4h,
            "ma20_4h": ma20_4h,
            "ma50_4h": ma50_4h,
            "ma100_4h": ma100_4h,
            "ma20_gap_4h": self._gap_ratio(close_4h, ma20_4h),
            "ma50_gap_4h": self._gap_ratio(close_4h, ma50_4h),
            "ma100_gap_4h": self._gap_ratio(close_4h, ma100_4h),
            "rsi_4h": self._rsi(four_h["close"]),
            "macd_4h": self._macd(four_h["close"]),
            "obv_4h": obv_4h,
            "obv_slope_4h": self._slope_ratio(four_h["close"], four_h["volume"], 10),
            "atr_4h": atr_4h,
            "atr_ratio_4h": atr_ratio_4h,
            "atr_percentile_4h": atr_percentile_4h,
            "adx_4h": adx_4h,
            "di_plus_4h": di_plus_4h,
            "di_minus_4h": di_minus_4h,
            "dmi_spread_4h": dmi_spread,
            "trend_strength_4h": adx_4h / 100.0,
            "high_low_range_4h": self._range_ratio(four_h),
            "return_4h": self._pct_change(four_h["close"], 1),
            "close_1d": close_1d,
            "ma200_1d": ma200_1d,
            "ma200_gap_1d": self._gap_ratio(close_1d, ma200_1d),
            "support_distance_1d": self._distance_to_level(close_1d, support_1d),
            "resistance_distance_1d": self._distance_to_level(resistance_1d, close_1d),
            "fibonacci_position_1d": fib_position_1d,
            "fib_382_distance_1d": fib_382_1d,
            "fib_618_distance_1d": fib_618_1d,
            "return_24h": self._pct_change(one_h["close"], 24),
            "return_7d": self._pct_change(one_d["close"], 7),
            "volatility_20d": volatility_20d,
            "sentiment_value": fear_greed_sentiment,
            "llm_sentiment_score": llm_sentiment,
            "lunarcrush_sentiment": float(payload.lunarcrush_sentiment or 0.0),
            "market_regime_score": float(payload.market_regime_score or 0.0),
            "onchain_netflow_score": float(payload.onchain_netflow_score or 0.0),
            "onchain_whale_score": float(payload.onchain_whale_score or 0.0),
            "black_swan_index": black_swan_index,
            "trend_alignment_score": trend_alignment_score,
            "price_structure_score": price_structure_score,
            "volatility_regime_score": volatility_regime_score,
            "directional_conviction": directional_conviction,
        }
        clean_values = self._sanitize(values)

        timestamp = pd.to_datetime(
            int(four_h["timestamp"].iloc[-1]),
            unit="ms",
            utc=True,
        ).to_pydatetime()
        return FeatureSnapshot(
            symbol=payload.symbol,
            timeframe="4h",
            timestamp=timestamp.astimezone(timezone.utc),
            values=clean_values,
            valid=valid,
        )

    @staticmethod
    def _frame(candles: list[dict]) -> pd.DataFrame:
        frame = pd.DataFrame(candles or [])
        if frame.empty:
            return frame
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.ffill().fillna(0.0)

    @staticmethod
    def _sanitize(values: dict[str, float]) -> dict[str, float]:
        clean: dict[str, float] = {}
        for key, value in values.items():
            if pd.isna(value) or value in (float("inf"), float("-inf")):
                clean[key] = 0.0
            else:
                clean[key] = float(value)
        return clean

    @staticmethod
    def _rolling_last(series: pd.Series, window: int) -> float:
        if series.empty:
            return 0.0
        value = series.rolling(window).mean().iloc[-1]
        return float(value) if pd.notna(value) else float(series.iloc[-1])

    @staticmethod
    def _pct_change(series: pd.Series, periods: int) -> float:
        if len(series) <= periods:
            return 0.0
        value = series.pct_change(periods).iloc[-1]
        return float(value) if pd.notna(value) else 0.0

    @staticmethod
    def _gap_ratio(price: float, moving_average: float) -> float:
        if moving_average == 0:
            return 0.0
        return (price / moving_average) - 1.0

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> float:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty and pd.notna(rsi.iloc[-1]) else 50.0

    @staticmethod
    def _macd(series: pd.Series) -> float:
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        value = (macd - signal).iloc[-1] if not macd.empty else 0.0
        return float(value) if pd.notna(value) else 0.0

    @staticmethod
    def _volume_ratio(frame: pd.DataFrame) -> float:
        current = float(frame["volume"].iloc[-1])
        average = float(frame["volume"].tail(20).mean())
        if average <= 0:
            return 0.0
        return current / average

    @staticmethod
    def _volume_zscore(frame: pd.DataFrame, window: int = 20) -> float:
        windowed = frame["volume"].tail(window)
        std = float(windowed.std()) if len(windowed) > 1 else 0.0
        if std <= 0:
            return 0.0
        mean = float(windowed.mean())
        return (float(frame["volume"].iloc[-1]) - mean) / std

    @staticmethod
    def _volume_trend(frame: pd.DataFrame, periods: int = 5) -> float:
        if len(frame) <= periods:
            return 0.0
        recent = float(frame["volume"].tail(periods).mean())
        baseline = float(frame["volume"].tail(periods * 3).head(periods).mean())
        if baseline <= 0:
            return 0.0
        return recent / baseline - 1.0

    @staticmethod
    def _obv(frame: pd.DataFrame) -> float:
        close = frame["close"]
        direction = close.diff().fillna(0.0).apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
        obv = (direction * frame["volume"]).cumsum()
        value = obv.iloc[-1] if not obv.empty else 0.0
        return float(value)

    @staticmethod
    def _slope_ratio(close: pd.Series, volume: pd.Series, periods: int) -> float:
        if len(close) <= periods or len(volume) <= periods:
            return 0.0
        obv = (close.diff().fillna(0.0).apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0)) * volume).cumsum()
        base = abs(float(obv.iloc[-periods])) if abs(float(obv.iloc[-periods])) > 1e-9 else 1.0
        return (float(obv.iloc[-1]) - float(obv.iloc[-periods])) / base

    @staticmethod
    def _atr(frame: pd.DataFrame, period: int = 14) -> float:
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period).mean()
        value = atr.iloc[-1] if not atr.empty else 0.0
        return float(value) if pd.notna(value) else 0.0

    @staticmethod
    def _atr_percentile(frame: pd.DataFrame, period: int = 14, lookback: int = 90) -> float:
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period).mean().dropna().tail(lookback)
        if atr.empty:
            return 0.5
        latest = float(atr.iloc[-1])
        return float((atr <= latest).mean())

    @staticmethod
    def _adx_dmi(frame: pd.DataFrame, period: int = 14) -> tuple[float, float, float]:
        high = frame["high"]
        low = frame["low"]
        close = frame["close"]

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period).mean().replace(0, 1e-10)
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)) * 100
        adx = dx.rolling(period).mean()
        adx_value = float(adx.iloc[-1]) if not adx.empty and pd.notna(adx.iloc[-1]) else 0.0
        plus_value = float(plus_di.iloc[-1]) if not plus_di.empty and pd.notna(plus_di.iloc[-1]) else 0.0
        minus_value = float(minus_di.iloc[-1]) if not minus_di.empty and pd.notna(minus_di.iloc[-1]) else 0.0
        return adx_value, plus_value, minus_value

    @staticmethod
    def _support_resistance(frame: pd.DataFrame, window: int = 60) -> tuple[float, float]:
        recent = frame.tail(window)
        if recent.empty:
            return 0.0, 0.0
        return float(recent["low"].min()), float(recent["high"].max())

    @staticmethod
    def _fibonacci_metrics(frame: pd.DataFrame, window: int = 60) -> tuple[float, float, float]:
        recent = frame.tail(window)
        if recent.empty:
            return 0.5, 0.0, 0.0
        low = float(recent["low"].min())
        high = float(recent["high"].max())
        close = float(recent["close"].iloc[-1])
        span = max(high - low, 1e-9)
        fib_382 = low + span * 0.382
        fib_618 = low + span * 0.618
        position = (close - low) / span
        return (
            float(position),
            (close / fib_382 - 1.0) if fib_382 else 0.0,
            (close / fib_618 - 1.0) if fib_618 else 0.0,
        )

    @staticmethod
    def _distance_to_level(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator - 1.0

    @staticmethod
    def _range_ratio(frame: pd.DataFrame, periods: int = 5) -> float:
        recent = frame.tail(periods)
        if recent.empty:
            return 0.0
        close = max(float(recent["close"].iloc[-1]), 1e-9)
        return float((recent["high"].max() - recent["low"].min()) / close)

    @staticmethod
    def _volatility(series: pd.Series, period: int = 20) -> float:
        returns = series.pct_change().dropna().tail(period)
        if returns.empty:
            return 0.0
        value = returns.std()
        return float(value) if pd.notna(value) else 0.0

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))
