"""Intraday feature mining for next-generation candidate research."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(slots=True)
class FeaturePoint:
    """Derived intraday feature snapshot at a single candle index."""

    index: int
    timestamp: int
    close: float
    bar_return: float
    momentum_slope: float
    realized_volatility: float
    wick_imbalance: float
    breakout_pressure: float
    reclaim_score: float
    mean_reversion_zscore: float
    volume_impulse: float
    range_expansion: float
    regime_consistency: float


@dataclass(slots=True)
class FeatureMiningResult:
    """Feature sequence plus compact summary suitable for persistence."""

    points: list[FeaturePoint] = field(default_factory=list)
    summary: dict[str, float | int] = field(default_factory=dict)


class FeatureMiner:
    """Mine lightweight intraday features from raw OHLCV candles."""

    def __init__(self, warmup_bars: int = 24):
        self.warmup_bars = max(12, warmup_bars)

    def mine(self, candles: list[dict]) -> FeatureMiningResult:
        if len(candles) <= self.warmup_bars:
            return FeatureMiningResult(
                points=[],
                summary={"feature_points": 0},
            )

        opens = [float(item["open"]) for item in candles]
        highs = [float(item["high"]) for item in candles]
        lows = [float(item["low"]) for item in candles]
        closes = [float(item["close"]) for item in candles]
        volumes = [float(item.get("volume", 0.0)) for item in candles]

        true_ranges = [self._true_range_ratio(opens[0], highs[0], lows[0], closes[0])]
        for idx in range(1, len(candles)):
            true_ranges.append(
                self._true_range_ratio(closes[idx - 1], highs[idx], lows[idx], closes[idx])
            )

        points: list[FeaturePoint] = []
        for idx in range(self.warmup_bars, len(candles)):
            lookback_start = idx - self.warmup_bars
            close_window = closes[lookback_start:idx]
            return_window = self._window_returns(closes, lookback_start, idx)
            range_window = true_ranges[lookback_start:idx]
            volume_window = volumes[lookback_start:idx]
            range_high = max(highs[lookback_start:idx])
            range_low = min(lows[lookback_start:idx])
            close_value = closes[idx]
            current_range = max(highs[idx] - lows[idx], 1e-9)
            avg_range = max(self._average(range_window), 1e-9)
            avg_volume = max(self._average(volume_window), 1e-9)
            momentum_fast = self._ratio_change(closes, idx, 4)
            momentum_slow = self._ratio_change(closes, idx, 12)
            mean_price = self._average(close_window)
            std_price = max(self._stddev(close_window), 1e-9)
            upper_wick = highs[idx] - max(opens[idx], close_value)
            lower_wick = min(opens[idx], close_value) - lows[idx]
            close_location = self._range_location(close_value, range_low, range_high)
            bar_return = self._ratio_change(closes, idx, 1)
            downside_pressure = min(bar_return, 0.0)
            reclaim_score = (
                close_location - 0.5
                + (lower_wick - upper_wick) / current_range
                + abs(downside_pressure) * 2.0
            )
            pos_count = sum(1 for value in return_window[-8:] if value > 0)
            neg_count = sum(1 for value in return_window[-8:] if value < 0)
            regime_consistency = self._consistency_ratio(pos_count, neg_count)

            points.append(
                FeaturePoint(
                    index=idx,
                    timestamp=int(candles[idx]["timestamp"]),
                    close=close_value,
                    bar_return=bar_return,
                    momentum_slope=(momentum_fast * 0.6) + (momentum_slow * 0.4),
                    realized_volatility=self._stddev(return_window),
                    wick_imbalance=(lower_wick - upper_wick) / current_range,
                    breakout_pressure=((close_value - range_high) / max(close_value, 1e-9))
                    + max(momentum_fast, 0.0),
                    reclaim_score=reclaim_score,
                    mean_reversion_zscore=(close_value - mean_price) / std_price,
                    volume_impulse=(volumes[idx] / avg_volume) - 1.0,
                    range_expansion=(true_ranges[idx] / avg_range) - 1.0,
                    regime_consistency=regime_consistency,
                )
            )

        return FeatureMiningResult(points=points, summary=self._summarize(points))

    @staticmethod
    def _window_returns(closes: list[float], start: int, end: int) -> list[float]:
        returns: list[float] = []
        for idx in range(max(start + 1, 1), end):
            prev_close = closes[idx - 1]
            if prev_close <= 0:
                continue
            returns.append((closes[idx] / prev_close) - 1.0)
        return returns or [0.0]

    @staticmethod
    def _true_range_ratio(prev_close: float, high: float, low: float, close: float) -> float:
        anchor = max(abs(prev_close), abs(close), 1e-9)
        return max(high - low, abs(high - prev_close), abs(low - prev_close)) / anchor

    @staticmethod
    def _average(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _stddev(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(max(variance, 0.0))

    @staticmethod
    def _ratio_change(values: list[float], idx: int, periods: int) -> float:
        if idx < periods or values[idx - periods] <= 0:
            return 0.0
        return (values[idx] / values[idx - periods]) - 1.0

    @staticmethod
    def _range_location(value: float, range_low: float, range_high: float) -> float:
        width = range_high - range_low
        if width <= 0:
            return 0.5
        return max(0.0, min(1.0, (value - range_low) / width))

    @staticmethod
    def _consistency_ratio(pos_count: int, neg_count: int) -> float:
        total = pos_count + neg_count
        if total <= 0:
            return 0.0
        return max(pos_count, neg_count) / total

    def _summarize(self, points: list[FeaturePoint]) -> dict[str, float | int]:
        if not points:
            return {"feature_points": 0}

        breakout_ready = sum(
            1
            for point in points
            if point.breakout_pressure > 0 and point.volume_impulse > 0
        )
        reclaim_ready = sum(
            1
            for point in points
            if point.reclaim_score > 0.25 and point.wick_imbalance > 0
        )
        return {
            "feature_points": len(points),
            "avg_realized_volatility": round(
                self._average([point.realized_volatility for point in points]), 6
            ),
            "avg_range_expansion": round(
                self._average([point.range_expansion for point in points]), 6
            ),
            "avg_volume_impulse": round(
                self._average([point.volume_impulse for point in points]), 6
            ),
            "avg_regime_consistency": round(
                self._average([point.regime_consistency for point in points]), 6
            ),
            "breakout_ready_points": breakout_ready,
            "reclaim_ready_points": reclaim_ready,
        }
