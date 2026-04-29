"""Validation ladder for candidate strategies."""

from __future__ import annotations

from .config import EvolutionConfig
from .models import PromotionStage, ScoreCard, StrategyGenome, ValidationMetrics


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ValidationPipeline:
    """Score candidates across backtest, walk-forward, shadow and live evidence."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def score(self, genome: StrategyGenome, metrics: ValidationMetrics) -> ScoreCard:
        reasons: list[str] = []
        sample_confidence = self._sample_confidence(metrics.trade_count)

        edge_score = _clamp(
            (
                metrics.backtest_expectancy * 0.20
                + metrics.walkforward_expectancy * 0.30
                + metrics.shadow_expectancy * 0.30
                + metrics.live_expectancy * 0.20
            ),
            -1.0,
            1.0,
        )
        if metrics.trade_count < self.config.min_trade_count:
            reasons.append("insufficient_trade_count")
            edge_score *= 0.45 + sample_confidence * 0.55

        drawdown_penalty = metrics.max_drawdown_pct / max(self.config.max_drawdown_pct, 1.0)
        cost_penalty = metrics.cost_drag_pct / max(self.config.max_cost_drag_pct, 0.01)
        latency_penalty = metrics.latency_ms / max(self.config.max_latency_ms, 1.0)
        robustness_score = _clamp(
            1.0 - (drawdown_penalty * 0.45 + cost_penalty * 0.25 + latency_penalty * 0.10),
            0.0,
            1.0,
        )
        if metrics.regime_consistency < self.config.min_regime_consistency:
            reasons.append("weak_regime_consistency")
            robustness_score *= 0.75

        deployment_score = _clamp(
            (metrics.shadow_expectancy * 0.4 + metrics.live_expectancy * 0.4)
            + (metrics.regime_consistency * 0.2),
            -1.0,
            1.0,
        )
        throughput_reward = self._high_frequency_reward(metrics)
        if throughput_reward > 1e-6:
            deployment_score = _clamp(deployment_score + throughput_reward, -1.0, 1.0)
            reasons.append("high_frequency_deployment_bonus")
        if metrics.trade_count < self.config.min_trade_count:
            deployment_score *= 0.60 + sample_confidence * 0.40
        total_score = round(
            edge_score * 0.45 + robustness_score * 0.30 + deployment_score * 0.25,
            4,
        )
        stage = self._decide_stage(total_score, metrics, reasons)

        return ScoreCard(
            genome=genome,
            stage=stage,
            edge_score=round(edge_score, 4),
            robustness_score=round(robustness_score, 4),
            deployment_score=round(deployment_score, 4),
            total_score=total_score,
            reasons=reasons,
        )

    def _sample_confidence(self, trade_count: int) -> float:
        if self.config.min_trade_count <= 0:
            return 1.0
        return _clamp(trade_count / float(self.config.min_trade_count), 0.0, 1.0)

    def _high_frequency_reward(self, metrics: ValidationMetrics) -> float:
        min_trades = max(1, int(self.config.min_trade_count))
        target_trades = max(min_trades, int(self.config.high_frequency_trade_count_target))
        if int(metrics.trade_count) < min_trades:
            return 0.0
        if metrics.shadow_expectancy <= 0.0 or metrics.live_expectancy <= 0.0:
            return 0.0
        if metrics.cost_drag_pct > self.config.max_cost_drag_pct * 0.75:
            return 0.0
        if metrics.latency_ms > self.config.max_latency_ms * 0.75:
            return 0.0
        if metrics.regime_consistency < self.config.min_regime_consistency:
            return 0.0
        if target_trades <= min_trades:
            throughput_progress = 1.0
        else:
            throughput_progress = _clamp(
                (float(metrics.trade_count) - float(min_trades))
                / float(target_trades - min_trades),
                0.0,
                1.0,
            )
        expectancy_support = _clamp(
            ((metrics.shadow_expectancy + metrics.live_expectancy) * 0.5) / 0.20,
            0.0,
            1.0,
        )
        reward_cap = max(0.0, float(self.config.high_frequency_deployment_reward_cap))
        return round(reward_cap * throughput_progress * expectancy_support, 4)

    def _decide_stage(
        self,
        total_score: float,
        metrics: ValidationMetrics,
        reasons: list[str],
    ) -> PromotionStage:
        if metrics.max_drawdown_pct > self.config.max_drawdown_pct:
            reasons.append("drawdown_limit_breached")
            return PromotionStage.REJECT
        if metrics.cost_drag_pct > self.config.max_cost_drag_pct:
            reasons.append("cost_drag_too_high")
            return PromotionStage.REJECT
        if metrics.latency_ms > self.config.max_latency_ms:
            reasons.append("latency_too_high")
            return PromotionStage.REJECT
        if total_score >= self.config.live_threshold and metrics.live_expectancy > 0:
            reasons.append("promote_live")
            return PromotionStage.LIVE
        if total_score >= self.config.paper_threshold and metrics.shadow_expectancy > 0:
            reasons.append("promote_paper")
            return PromotionStage.PAPER
        if total_score >= self.config.shadow_threshold:
            reasons.append("promote_shadow")
            return PromotionStage.SHADOW
        reasons.append("reject_low_edge")
        return PromotionStage.REJECT
