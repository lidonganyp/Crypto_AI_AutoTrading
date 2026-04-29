"""Helpers for aligning model targets with realized trade outcomes."""
from __future__ import annotations

from math import ceil


def timeframe_hours(timeframe: str) -> float:
    value = str(timeframe or "").strip().lower()
    if not value:
        return 4.0
    try:
        amount = float(value[:-1])
    except ValueError:
        return 4.0
    unit = value[-1]
    if unit == "m":
        return amount / 60.0
    if unit == "h":
        return amount
    if unit == "d":
        return amount * 24.0
    if unit == "w":
        return amount * 24.0 * 7.0
    return 4.0


def effective_trade_horizon_hours(settings) -> int:
    max_hold_hours = int(getattr(settings.strategy, "max_hold_hours", 0) or 0)
    if max_hold_hours > 0:
        return max_hold_hours
    configured = int(getattr(settings.training, "prediction_horizon_hours", 0) or 0)
    return max(4, configured)


def horizon_bars(max_hold_hours: int, timeframe: str) -> int:
    candle_hours = max(timeframe_hours(timeframe), 1e-9)
    return max(1, int(ceil(max_hold_hours / candle_hours)))


def simulate_long_trade(
    *,
    future_candles: list[dict],
    entry_price: float,
    timeframe: str,
    max_hold_hours: int,
    stop_loss_pct: float,
    take_profit_levels: list[float] | None,
    round_trip_cost_pct: float = 0.15,
) -> dict[str, float | int | str] | None:
    if entry_price <= 0:
        return None
    limit = horizon_bars(max_hold_hours, timeframe)
    window = list(future_candles[:limit])
    if not window:
        return None

    stop_price = entry_price * (1.0 - max(float(stop_loss_pct or 0.0), 0.0))
    targets = [
        entry_price * (1.0 + level)
        for level in sorted(float(level) for level in (take_profit_levels or []) if level > 0)
    ]
    candle_hours = timeframe_hours(timeframe)
    favorable_excursion_pct = 0.0
    adverse_excursion_pct = 0.0

    for index, candle in enumerate(window, start=1):
        high = float(candle.get("high") or candle.get("close") or entry_price)
        low = float(candle.get("low") or candle.get("close") or entry_price)
        close = float(candle.get("close") or entry_price)
        favorable_excursion_pct = max(
            favorable_excursion_pct,
            max(0.0, (high / entry_price - 1.0) * 100.0),
        )
        adverse_excursion_pct = max(
            adverse_excursion_pct,
            max(0.0, (1.0 - low / entry_price) * 100.0),
        )

        # Conservative assumption for same-candle overlap: stop triggers before target.
        if stop_price > 0 and low <= stop_price:
            exit_price = stop_price
            exit_reason = "fixed_stop_loss"
        else:
            exit_price = 0.0
            exit_reason = ""
            for target_index, target_price in enumerate(targets, start=1):
                if high >= target_price:
                    exit_price = target_price
                    exit_reason = f"take_profit_{target_index}"
                    break
            if not exit_reason:
                if index >= limit:
                    exit_price = close
                    exit_reason = "time_stop"
                else:
                    continue

        gross_return_pct = (exit_price / entry_price - 1.0) * 100.0
        net_return_pct = gross_return_pct - float(round_trip_cost_pct or 0.0)
        return {
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "gross_return_pct": gross_return_pct,
            "net_return_pct": net_return_pct,
            "bars_held": index,
            "holding_hours": index * candle_hours,
            "favorable_excursion_pct": favorable_excursion_pct,
            "adverse_excursion_pct": adverse_excursion_pct,
            "exit_timestamp": float(candle.get("timestamp") or 0.0),
        }
    return None
