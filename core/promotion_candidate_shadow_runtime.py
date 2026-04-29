"""Shadow-stage candidate observation helpers."""
from __future__ import annotations

from datetime import datetime, timedelta


class PromotionCandidateShadowRuntime:
    """Evaluate shadow-stage candidates before live canary promotion."""

    def __init__(self, candidate_runtime):
        self.candidate_runtime = candidate_runtime
        self.runtime = candidate_runtime.runtime

    def observe_shadow_candidate(
        self,
        *,
        symbol: str,
        candidate: dict,
        now: datetime,
        registered_at: str,
        challenger_model_id: str,
        active_model_id: str,
        baseline_accuracy: float,
        adaptive_requirements: dict,
    ) -> tuple[str, dict | None]:
        min_evaluations = int(
            candidate.get(
                "min_shadow_evaluations",
                adaptive_requirements.get("shadow_min_evaluations", 0),
            )
            or adaptive_requirements.get("shadow_min_evaluations", 0)
        )
        max_age_hours = int(
            candidate.get(
                "max_shadow_age_hours",
                adaptive_requirements.get("shadow_max_age_hours", 72),
            )
            or adaptive_requirements.get("shadow_max_age_hours", 72)
        )
        challenger_metrics = self.runtime.build_model_scorecard(
            symbol=symbol,
            model_id=challenger_model_id,
            evaluation_type="challenger_shadow",
            started_at=registered_at,
        )
        champion_metrics = self.runtime.build_model_scorecard(
            symbol=symbol,
            model_id=active_model_id,
            evaluation_type="execution",
            started_at=registered_at,
        )
        registered_dt = self.runtime._parse_iso_datetime(registered_at)
        if (
            int(challenger_metrics.get("sample_count", 0) or 0) < min_evaluations
            or int(challenger_metrics.get("executed_count", 0) or 0) <= 0
        ):
            if (
                registered_dt is not None
                and now - registered_dt < timedelta(hours=max_age_hours)
            ):
                return "continue", None
            challenger_scorecard = self.runtime.record_model_scorecard(
                symbol=symbol,
                model_id=challenger_model_id,
                model_path=str(candidate.get("challenger_model_path") or ""),
                stage="canary_shadow",
                evaluation_type="challenger_shadow",
                started_at=registered_at,
                extra_metadata={"status": "rejected_timeout"},
            )
            champion_scorecard = self.runtime.record_model_scorecard(
                symbol=symbol,
                model_id=active_model_id,
                model_path=str(candidate.get("active_model_path") or ""),
                stage="canary_shadow_baseline",
                evaluation_type="execution",
                started_at=registered_at,
                extra_metadata={"status": "baseline"},
            )
            self.candidate_runtime.reject_promotion_candidate(
                symbol,
                candidate,
                now,
                (
                    "no_shadow_executions"
                    if int(challenger_metrics.get("executed_count", 0) or 0) <= 0
                    else "insufficient_shadow_samples"
                ),
                {
                    **challenger_scorecard,
                    "champion_accuracy": float(
                        champion_scorecard.get("accuracy", 0.0) or 0.0
                    ),
                    "champion_objective_score": float(
                        champion_scorecard.get("objective_score", 0.0) or 0.0
                    ),
                },
            )
            return "rejected", None

        challenger_scorecard = self.runtime.record_model_scorecard(
            symbol=symbol,
            model_id=challenger_model_id,
            model_path=str(candidate.get("challenger_model_path") or ""),
            stage="canary_shadow",
            evaluation_type="challenger_shadow",
            started_at=registered_at,
            extra_metadata={"status": "decision"},
        )
        champion_scorecard = self.runtime.record_model_scorecard(
            symbol=symbol,
            model_id=active_model_id,
            model_path=str(candidate.get("active_model_path") or ""),
            stage="canary_shadow_baseline",
            evaluation_type="execution",
            started_at=registered_at,
            extra_metadata={"status": "baseline"},
        )
        shadow_accuracy = float(challenger_scorecard.get("accuracy", 0.0) or 0.0)
        shadow_objective_quality = self.runtime.objective_score_quality(
            challenger_scorecard
        )
        champion_objective_quality = self.runtime.objective_score_quality(
            champion_scorecard
        )
        shadow_expectancy_pct = float(
            challenger_scorecard.get("expectancy_pct", 0.0) or 0.0
        )
        shadow_profit_factor = float(
            challenger_scorecard.get("profit_factor", 0.0) or 0.0
        )
        shadow_max_drawdown_pct = float(
            challenger_scorecard.get("max_drawdown_pct", 0.0) or 0.0
        )
        minimum_shadow_accuracy = self.runtime.accuracy_safety_floor(
            baseline_accuracy
        )
        minimum_shadow_objective_quality = float(
            candidate.get(
                "baseline_objective_quality",
                candidate.get("baseline_objective_score", 0.0),
            )
            or 0.0
        ) - 0.35
        minimum_shadow_expectancy_pct = max(
            -0.10,
            float(candidate.get("baseline_expectancy_pct", 0.0) or 0.0) - 0.15,
        )
        minimum_shadow_profit_factor = max(
            0.80,
            float(candidate.get("baseline_profit_factor", 0.0) or 0.0) - 0.25,
        )
        maximum_shadow_drawdown_pct = max(
            1.50,
            float(candidate.get("baseline_max_drawdown_pct", 0.0) or 0.0) + 1.50,
        )
        minimum_shadow_trade_return = float(
            candidate.get("baseline_avg_trade_return_pct", 0.0) or 0.0
        ) - 0.20
        if int(champion_scorecard.get("sample_count", 0) or 0) >= min_evaluations:
            minimum_shadow_objective_quality = max(
                minimum_shadow_objective_quality,
                champion_objective_quality - 0.20,
            )
            minimum_shadow_expectancy_pct = max(
                minimum_shadow_expectancy_pct,
                float(champion_scorecard.get("expectancy_pct", 0.0) or 0.0) - 0.10,
            )
            minimum_shadow_profit_factor = max(
                minimum_shadow_profit_factor,
                max(
                    0.80,
                    float(champion_scorecard.get("profit_factor", 0.0) or 0.0)
                    - 0.20,
                ),
            )
            maximum_shadow_drawdown_pct = min(
                maximum_shadow_drawdown_pct,
                max(
                    1.50,
                    float(champion_scorecard.get("max_drawdown_pct", 0.0) or 0.0)
                    + 1.00,
                ),
            )
            minimum_shadow_trade_return = max(
                minimum_shadow_trade_return,
                float(champion_scorecard.get("avg_trade_return_pct", 0.0) or 0.0)
                - 0.15,
            )

        if (
            float(challenger_scorecard.get("objective_score", 0.0) or 0.0) < -0.10
            or shadow_objective_quality + 1e-12 < minimum_shadow_objective_quality
            or shadow_expectancy_pct + 1e-12 < minimum_shadow_expectancy_pct
            or shadow_profit_factor + 1e-12 < minimum_shadow_profit_factor
            or shadow_max_drawdown_pct - 1e-12 > maximum_shadow_drawdown_pct
            or float(challenger_scorecard.get("avg_trade_return_pct", 0.0) or 0.0)
            + 1e-12
            < minimum_shadow_trade_return
            or shadow_accuracy + 1e-12 < minimum_shadow_accuracy
        ):
            self.candidate_runtime.reject_promotion_candidate(
                symbol,
                candidate,
                now,
                self._shadow_rejection_reason(
                    challenger_scorecard=challenger_scorecard,
                    shadow_objective_quality=shadow_objective_quality,
                    minimum_shadow_objective_quality=minimum_shadow_objective_quality,
                    shadow_expectancy_pct=shadow_expectancy_pct,
                    minimum_shadow_expectancy_pct=minimum_shadow_expectancy_pct,
                    shadow_profit_factor=shadow_profit_factor,
                    minimum_shadow_profit_factor=minimum_shadow_profit_factor,
                    shadow_max_drawdown_pct=shadow_max_drawdown_pct,
                    maximum_shadow_drawdown_pct=maximum_shadow_drawdown_pct,
                    minimum_shadow_trade_return=minimum_shadow_trade_return,
                    shadow_accuracy=shadow_accuracy,
                    minimum_shadow_accuracy=minimum_shadow_accuracy,
                ),
                {
                    **challenger_scorecard,
                    "champion_accuracy": float(
                        champion_scorecard.get("accuracy", 0.0) or 0.0
                    ),
                    "champion_objective_score": float(
                        champion_scorecard.get("objective_score", 0.0) or 0.0
                    ),
                },
            )
            return "rejected", None

        advanced = self.candidate_runtime.advance_promotion_candidate_to_live(
            symbol,
            candidate,
            now,
            challenger_scorecard,
            champion_scorecard,
        )
        advanced["shadow_objective_score"] = float(
            challenger_scorecard.get("objective_score", 0.0) or 0.0
        )
        advanced["shadow_objective_quality"] = self.runtime.objective_score_quality(
            challenger_scorecard
        )
        advanced["shadow_expectancy_pct"] = float(
            challenger_scorecard.get("expectancy_pct", 0.0) or 0.0
        )
        advanced["shadow_profit_factor"] = float(
            challenger_scorecard.get("profit_factor", 0.0) or 0.0
        )
        advanced["shadow_max_drawdown_pct"] = float(
            challenger_scorecard.get("max_drawdown_pct", 0.0) or 0.0
        )
        advanced["shadow_avg_trade_return_pct"] = float(
            challenger_scorecard.get("avg_trade_return_pct", 0.0) or 0.0
        )
        return "shadow_to_live", advanced

    @staticmethod
    def _shadow_rejection_reason(
        *,
        challenger_scorecard: dict[str, float | int],
        shadow_objective_quality: float,
        minimum_shadow_objective_quality: float,
        shadow_expectancy_pct: float,
        minimum_shadow_expectancy_pct: float,
        shadow_profit_factor: float,
        minimum_shadow_profit_factor: float,
        shadow_max_drawdown_pct: float,
        maximum_shadow_drawdown_pct: float,
        minimum_shadow_trade_return: float,
        shadow_accuracy: float,
        minimum_shadow_accuracy: float,
    ) -> str:
        if float(challenger_scorecard.get("objective_score", 0.0) or 0.0) < -0.10:
            return (
                "shadow_negative_objective_"
                f"{float(challenger_scorecard.get('objective_score', 0.0) or 0.0):.2f}"
            )
        if shadow_objective_quality + 1e-12 < minimum_shadow_objective_quality:
            return (
                "shadow_objective_quality_"
                f"{shadow_objective_quality:.2f}"
                f"_below_{minimum_shadow_objective_quality:.2f}"
            )
        if shadow_expectancy_pct + 1e-12 < minimum_shadow_expectancy_pct:
            return (
                "shadow_expectancy_"
                f"{shadow_expectancy_pct:.2f}"
                f"_below_{minimum_shadow_expectancy_pct:.2f}"
            )
        if shadow_profit_factor + 1e-12 < minimum_shadow_profit_factor:
            return (
                "shadow_profit_factor_"
                f"{shadow_profit_factor:.2f}"
                f"_below_{minimum_shadow_profit_factor:.2f}"
            )
        if shadow_max_drawdown_pct - 1e-12 > maximum_shadow_drawdown_pct:
            return (
                "shadow_drawdown_"
                f"{shadow_max_drawdown_pct:.2f}"
                f"_above_{maximum_shadow_drawdown_pct:.2f}"
            )
        if (
            float(challenger_scorecard.get("avg_trade_return_pct", 0.0) or 0.0)
            + 1e-12
            < minimum_shadow_trade_return
        ):
            return (
                "shadow_trade_return_"
                f"{float(challenger_scorecard.get('avg_trade_return_pct', 0.0) or 0.0):.2f}"
                f"_below_{minimum_shadow_trade_return:.2f}"
            )
        return (
            f"shadow_accuracy_{shadow_accuracy:.2f}"
            f"_below_{minimum_shadow_accuracy:.2f}"
        )
