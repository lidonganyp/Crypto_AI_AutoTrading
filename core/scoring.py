"""Shared scoring helpers for expectancy-led model and symbol ranking."""
from __future__ import annotations


def objective_score_sample_factor(metrics: dict[str, float | int]) -> float:
    executed_count = int(metrics.get("executed_count", 0) or 0)
    if executed_count > 0:
        return min(executed_count, 8) / 8
    sample_count = int(metrics.get("sample_count", 0) or 0)
    return min(sample_count, 8) / 16 if sample_count > 0 else 0.0


def objective_score_from_metrics(metrics: dict[str, float | int]) -> float:
    sample_factor = objective_score_sample_factor(metrics)
    if sample_factor <= 0:
        return 0.0
    expectancy_pct = float(
        metrics.get(
            "expectancy_pct",
            metrics.get("avg_trade_return_pct", 0.0),
        )
        or 0.0
    )
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    max_drawdown_pct = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
    trade_win_rate = float(
        metrics.get(
            "trade_win_rate",
            metrics.get("executed_precision", 0.0),
        )
        or 0.0
    )
    avg_cost_pct = float(metrics.get("avg_cost_pct", 0.0) or 0.0)
    avg_trade_return_pct = float(metrics.get("avg_trade_return_pct", 0.0) or 0.0)
    accuracy = float(metrics.get("accuracy", 0.0) or 0.0)
    quality_score = (
        expectancy_pct * 18.0
        + min(max(profit_factor, 0.0), 4.0) * 3.5
        + avg_trade_return_pct * 6.0
        + (trade_win_rate - 0.5) * 4.0
        + (accuracy - 0.5) * 0.75
        - max_drawdown_pct * 1.6
        - avg_cost_pct * 3.0
    )
    return quality_score * sample_factor


def objective_score_quality(metrics: dict[str, float | int]) -> float:
    sample_factor = objective_score_sample_factor(metrics)
    if sample_factor <= 0:
        return 0.0
    score = (
        float(metrics.get("objective_score", 0.0) or 0.0)
        if "objective_score" in metrics
        else objective_score_from_metrics(metrics)
    )
    return score / sample_factor
