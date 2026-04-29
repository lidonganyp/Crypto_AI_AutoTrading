"""Live-canary candidate observation helpers."""
from __future__ import annotations

from datetime import datetime, timedelta


class PromotionCandidateLiveRuntime:
    """Evaluate live-canary candidates before final promotion."""

    def __init__(self, candidate_runtime):
        self.candidate_runtime = candidate_runtime
        self.runtime = candidate_runtime.runtime

    def observe_live_candidate(
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
    ) -> tuple[str, None]:
        stage_started_at = str(candidate.get("live_started_at") or registered_at)
        min_evaluations = int(
            candidate.get(
                "min_live_evaluations",
                adaptive_requirements.get("live_min_evaluations", 0),
            )
            or adaptive_requirements.get("live_min_evaluations", 0)
        )
        max_age_hours = int(
            candidate.get(
                "max_live_age_hours",
                adaptive_requirements.get("live_max_age_hours", 72),
            )
            or adaptive_requirements.get("live_max_age_hours", 72)
        )
        challenger_metrics = self.runtime.build_model_scorecard(
            symbol=symbol,
            model_id=challenger_model_id,
            evaluation_type="challenger_live",
            started_at=stage_started_at,
        )
        live_trade_count = self.runtime._ab_variant_count(
            symbol,
            stage_started_at,
            "challenger_live",
        )
        stage_started_dt = self.runtime._parse_iso_datetime(stage_started_at)
        if (
            int(challenger_metrics.get("sample_count", 0) or 0) < min_evaluations
            or live_trade_count <= 0
            or int(challenger_metrics.get("executed_count", 0) or 0) <= 0
        ):
            if (
                stage_started_dt is not None
                and now - stage_started_dt < timedelta(hours=max_age_hours)
            ):
                return "continue", None
            challenger_scorecard = self.runtime.record_model_scorecard(
                symbol=symbol,
                model_id=challenger_model_id,
                model_path=str(candidate.get("challenger_model_path") or ""),
                stage="canary_live",
                evaluation_type="challenger_live",
                started_at=stage_started_at,
                extra_metadata={"status": "rejected_timeout"},
            )
            champion_scorecard = self.runtime.record_model_scorecard(
                symbol=symbol,
                model_id=active_model_id,
                model_path=str(candidate.get("active_model_path") or ""),
                stage="canary_live_baseline",
                evaluation_type="execution",
                started_at=stage_started_at,
                extra_metadata={"status": "baseline"},
            )
            self.candidate_runtime.reject_promotion_candidate(
                symbol,
                candidate,
                now,
                (
                    "no_live_canary_execution"
                    if live_trade_count <= 0
                    else "insufficient_live_samples"
                ),
                {
                    **challenger_scorecard,
                    "champion_accuracy": float(
                        champion_scorecard.get("accuracy", 0.0) or 0.0
                    ),
                    "champion_objective_score": float(
                        champion_scorecard.get("objective_score", 0.0) or 0.0
                    ),
                    "live_trade_count": live_trade_count,
                },
            )
            return "rejected", None

        challenger_scorecard = self.runtime.record_model_scorecard(
            symbol=symbol,
            model_id=challenger_model_id,
            model_path=str(candidate.get("challenger_model_path") or ""),
            stage="canary_live",
            evaluation_type="challenger_live",
            started_at=stage_started_at,
            extra_metadata={"status": "decision"},
        )
        champion_scorecard = self.runtime.record_model_scorecard(
            symbol=symbol,
            model_id=active_model_id,
            model_path=str(candidate.get("active_model_path") or ""),
            stage="canary_live_baseline",
            evaluation_type="execution",
            started_at=stage_started_at,
            extra_metadata={"status": "baseline"},
        )
        live_pnl_summary = self.runtime.build_model_live_pnl_summary(
            symbol=symbol,
            model_id=challenger_model_id,
            started_at=stage_started_at,
        )
        live_pnl_objective_score = self.runtime.objective_score_from_metrics(
            live_pnl_summary
        )
        live_pnl_metrics = dict(live_pnl_summary)
        live_pnl_metrics["objective_score"] = live_pnl_objective_score
        live_pnl_objective_quality = self.runtime.objective_score_quality(
            live_pnl_metrics
        )
        live_pnl_closed_trade_count = int(
            live_pnl_summary.get("closed_trade_count", 0) or 0
        )
        live_accuracy = float(challenger_scorecard.get("accuracy", 0.0) or 0.0)
        challenger_objective_quality = self.runtime.objective_score_quality(
            challenger_scorecard
        )
        champion_objective_quality = self.runtime.objective_score_quality(
            champion_scorecard
        )
        challenger_expectancy_pct = float(
            challenger_scorecard.get("expectancy_pct", 0.0) or 0.0
        )
        challenger_profit_factor = float(
            challenger_scorecard.get("profit_factor", 0.0) or 0.0
        )
        challenger_max_drawdown_pct = float(
            challenger_scorecard.get("max_drawdown_pct", 0.0) or 0.0
        )
        champion_expectancy_pct = float(
            champion_scorecard.get("expectancy_pct", 0.0) or 0.0
        )
        champion_profit_factor = float(
            champion_scorecard.get("profit_factor", 0.0) or 0.0
        )
        champion_max_drawdown_pct = float(
            champion_scorecard.get("max_drawdown_pct", 0.0) or 0.0
        )
        shadow_objective_quality = float(
            candidate.get("shadow_objective_quality", 0.0)
            or self.runtime.objective_score_quality(
                {
                    "objective_score": float(
                        candidate.get("shadow_objective_score", 0.0) or 0.0
                    ),
                    "sample_count": int(candidate.get("shadow_eval_count", 0) or 0),
                    "executed_count": int(
                        candidate.get(
                            "shadow_executed_count",
                            candidate.get("shadow_eval_count", 0),
                        )
                        or 0
                    ),
                }
            )
        )
        minimum_live_accuracy = self.runtime.accuracy_safety_floor(
            baseline_accuracy
        )
        minimum_live_objective_quality = max(
            float(
                candidate.get(
                    "baseline_objective_quality",
                    candidate.get("baseline_objective_score", 0.0),
                )
                or 0.0
            )
            - 0.40,
            shadow_objective_quality - 0.25,
        )
        minimum_live_expectancy_pct = max(
            -0.10,
            float(candidate.get("baseline_expectancy_pct", 0.0) or 0.0) - 0.15,
            float(candidate.get("shadow_expectancy_pct", 0.0) or 0.0) - 0.10,
        )
        minimum_live_profit_factor = max(
            0.80,
            float(candidate.get("baseline_profit_factor", 0.0) or 0.0) - 0.25,
            float(candidate.get("shadow_profit_factor", 0.0) or 0.0) - 0.15,
        )
        maximum_live_drawdown_pct = min(
            max(
                1.50,
                float(candidate.get("baseline_max_drawdown_pct", 0.0) or 0.0)
                + 1.25,
            ),
            float(candidate.get("shadow_max_drawdown_pct", 0.0) or 0.0) + 1.0,
        )
        minimum_live_trade_return = max(
            float(candidate.get("baseline_avg_trade_return_pct", 0.0) or 0.0) - 0.15,
            float(candidate.get("shadow_avg_trade_return_pct", 0.0) or 0.0) - 0.10,
        )
        minimum_live_realized_trade_count = max(
            2,
            int(
                candidate.get(
                    "min_live_realized_trade_count",
                    adaptive_requirements.get("live_min_realized_trade_count", 3),
                )
                or adaptive_requirements.get("live_min_realized_trade_count", 3)
            ),
        )
        minimum_live_realized_objective_quality = max(
            minimum_live_objective_quality,
            shadow_objective_quality - 0.20,
        )
        minimum_live_realized_expectancy_pct = max(
            minimum_live_expectancy_pct,
            float(candidate.get("shadow_expectancy_pct", 0.0) or 0.0) - 0.05,
        )
        minimum_live_realized_profit_factor = max(
            minimum_live_profit_factor,
            float(candidate.get("shadow_profit_factor", 0.0) or 0.0) - 0.10,
        )
        maximum_live_realized_drawdown_pct = min(
            maximum_live_drawdown_pct,
            float(candidate.get("shadow_max_drawdown_pct", 0.0) or 0.0) + 0.75,
        )
        minimum_live_realized_trade_return = max(
            minimum_live_trade_return,
            float(candidate.get("shadow_avg_trade_return_pct", 0.0) or 0.0) - 0.05,
        )
        if int(champion_scorecard.get("sample_count", 0) or 0) >= min_evaluations:
            minimum_live_objective_quality = max(
                minimum_live_objective_quality,
                champion_objective_quality - 0.20,
            )
            minimum_live_expectancy_pct = max(
                minimum_live_expectancy_pct,
                champion_expectancy_pct - 0.05,
            )
            minimum_live_profit_factor = max(
                minimum_live_profit_factor,
                max(0.80, champion_profit_factor - 0.15),
            )
            maximum_live_drawdown_pct = min(
                maximum_live_drawdown_pct,
                max(1.50, champion_max_drawdown_pct + 0.75),
            )
            minimum_live_trade_return = max(
                minimum_live_trade_return,
                float(champion_scorecard.get("avg_trade_return_pct", 0.0) or 0.0)
                - 0.05,
            )
            minimum_live_realized_objective_quality = max(
                minimum_live_realized_objective_quality,
                champion_objective_quality - 0.25,
            )
            minimum_live_realized_expectancy_pct = max(
                minimum_live_realized_expectancy_pct,
                champion_expectancy_pct - 0.10,
            )
            minimum_live_realized_profit_factor = max(
                minimum_live_realized_profit_factor,
                max(0.80, champion_profit_factor - 0.20),
            )
            maximum_live_realized_drawdown_pct = min(
                maximum_live_realized_drawdown_pct,
                max(1.75, champion_max_drawdown_pct + 1.00),
            )
            minimum_live_realized_trade_return = max(
                minimum_live_realized_trade_return,
                float(champion_scorecard.get("avg_trade_return_pct", 0.0) or 0.0)
                - 0.10,
            )

        live_pnl_expectancy_pct = float(
            live_pnl_summary.get("expectancy_pct", 0.0) or 0.0
        )
        live_pnl_profit_factor = float(
            live_pnl_summary.get("profit_factor", 0.0) or 0.0
        )
        live_pnl_max_drawdown_pct = float(
            live_pnl_summary.get("max_drawdown_pct", 0.0) or 0.0
        )
        live_pnl_avg_trade_return_pct = float(
            live_pnl_summary.get("avg_trade_return_pct", 0.0) or 0.0
        )
        live_realized_negative = (
            live_pnl_closed_trade_count > 0
            and (
                live_pnl_objective_score < -0.10
                or live_pnl_expectancy_pct < -0.05
            )
        )
        live_realized_under_floor = (
            live_pnl_closed_trade_count >= minimum_live_realized_trade_count
            and (
                live_pnl_objective_quality + 1e-12
                < minimum_live_realized_objective_quality
                or live_pnl_expectancy_pct + 1e-12
                < minimum_live_realized_expectancy_pct
                or live_pnl_profit_factor + 1e-12
                < minimum_live_realized_profit_factor
                or live_pnl_max_drawdown_pct - 1e-12
                > maximum_live_realized_drawdown_pct
                or live_pnl_avg_trade_return_pct + 1e-12
                < minimum_live_realized_trade_return
            )
        )
        if (
            float(challenger_scorecard.get("objective_score", 0.0) or 0.0) < -0.10
            or live_realized_negative
            or live_realized_under_floor
            or challenger_objective_quality + 1e-12 < minimum_live_objective_quality
            or challenger_expectancy_pct + 1e-12 < minimum_live_expectancy_pct
            or challenger_profit_factor + 1e-12 < minimum_live_profit_factor
            or challenger_max_drawdown_pct - 1e-12 > maximum_live_drawdown_pct
            or float(challenger_scorecard.get("avg_trade_return_pct", 0.0) or 0.0)
            + 1e-12
            < minimum_live_trade_return
            or live_accuracy + 1e-12 < minimum_live_accuracy
        ):
            self.candidate_runtime.reject_promotion_candidate(
                symbol,
                candidate,
                now,
                self._live_rejection_reason(
                    challenger_scorecard=challenger_scorecard,
                    live_pnl_closed_trade_count=live_pnl_closed_trade_count,
                    minimum_live_realized_trade_count=minimum_live_realized_trade_count,
                    live_pnl_objective_score=live_pnl_objective_score,
                    live_realized_negative=live_realized_negative,
                    live_pnl_objective_quality=live_pnl_objective_quality,
                    minimum_live_realized_objective_quality=minimum_live_realized_objective_quality,
                    live_pnl_expectancy_pct=live_pnl_expectancy_pct,
                    minimum_live_realized_expectancy_pct=minimum_live_realized_expectancy_pct,
                    live_pnl_profit_factor=live_pnl_profit_factor,
                    minimum_live_realized_profit_factor=minimum_live_realized_profit_factor,
                    live_pnl_max_drawdown_pct=live_pnl_max_drawdown_pct,
                    maximum_live_realized_drawdown_pct=maximum_live_realized_drawdown_pct,
                    live_pnl_avg_trade_return_pct=live_pnl_avg_trade_return_pct,
                    minimum_live_realized_trade_return=minimum_live_realized_trade_return,
                    challenger_objective_quality=challenger_objective_quality,
                    minimum_live_objective_quality=minimum_live_objective_quality,
                    challenger_expectancy_pct=challenger_expectancy_pct,
                    minimum_live_expectancy_pct=minimum_live_expectancy_pct,
                    challenger_profit_factor=challenger_profit_factor,
                    minimum_live_profit_factor=minimum_live_profit_factor,
                    challenger_max_drawdown_pct=challenger_max_drawdown_pct,
                    maximum_live_drawdown_pct=maximum_live_drawdown_pct,
                    live_accuracy=live_accuracy,
                    minimum_live_accuracy=minimum_live_accuracy,
                ),
                {
                    **challenger_scorecard,
                    "champion_accuracy": float(
                        champion_scorecard.get("accuracy", 0.0) or 0.0
                    ),
                    "champion_objective_score": float(
                        champion_scorecard.get("objective_score", 0.0) or 0.0
                    ),
                    "live_trade_count": live_trade_count,
                    "live_pnl_closed_trade_count": live_pnl_closed_trade_count,
                    "live_pnl_realized_net_pnl": float(
                        live_pnl_summary.get("realized_net_pnl", 0.0) or 0.0
                    ),
                    "live_pnl_expectancy_pct": live_pnl_expectancy_pct,
                    "live_pnl_profit_factor": live_pnl_profit_factor,
                    "live_pnl_max_drawdown_pct": live_pnl_max_drawdown_pct,
                    "live_pnl_avg_trade_return_pct": live_pnl_avg_trade_return_pct,
                    "live_pnl_avg_holding_hours": float(
                        live_pnl_summary.get("avg_holding_hours", 0.0) or 0.0
                    ),
                    "live_pnl_objective_score": live_pnl_objective_score,
                    "live_pnl_objective_quality": live_pnl_objective_quality,
                },
            )
            return "rejected", None

        self.candidate_runtime.promote_candidate_to_active(
            symbol,
            candidate,
            now,
            {
                **challenger_scorecard,
                "live_trade_count": live_trade_count,
                "live_pnl_closed_trade_count": live_pnl_closed_trade_count,
                "live_pnl_realized_net_pnl": float(
                    live_pnl_summary.get("realized_net_pnl", 0.0) or 0.0
                ),
                "live_pnl_expectancy_pct": live_pnl_expectancy_pct,
                "live_pnl_profit_factor": live_pnl_profit_factor,
                "live_pnl_max_drawdown_pct": live_pnl_max_drawdown_pct,
                "live_pnl_avg_trade_return_pct": live_pnl_avg_trade_return_pct,
                "live_pnl_avg_holding_hours": float(
                    live_pnl_summary.get("avg_holding_hours", 0.0) or 0.0
                ),
                "live_pnl_objective_score": live_pnl_objective_score,
                "live_pnl_objective_quality": live_pnl_objective_quality,
            },
        )
        return "promoted", None

    @staticmethod
    def _live_rejection_reason(
        *,
        challenger_scorecard: dict[str, float | int],
        live_pnl_closed_trade_count: int,
        minimum_live_realized_trade_count: int,
        live_pnl_objective_score: float,
        live_realized_negative: bool,
        live_pnl_objective_quality: float,
        minimum_live_realized_objective_quality: float,
        live_pnl_expectancy_pct: float,
        minimum_live_realized_expectancy_pct: float,
        live_pnl_profit_factor: float,
        minimum_live_realized_profit_factor: float,
        live_pnl_max_drawdown_pct: float,
        maximum_live_realized_drawdown_pct: float,
        live_pnl_avg_trade_return_pct: float,
        minimum_live_realized_trade_return: float,
        challenger_objective_quality: float,
        minimum_live_objective_quality: float,
        challenger_expectancy_pct: float,
        minimum_live_expectancy_pct: float,
        challenger_profit_factor: float,
        minimum_live_profit_factor: float,
        challenger_max_drawdown_pct: float,
        maximum_live_drawdown_pct: float,
        live_accuracy: float,
        minimum_live_accuracy: float,
    ) -> str:
        if float(challenger_scorecard.get("objective_score", 0.0) or 0.0) < -0.10:
            return (
                "live_negative_objective_"
                f"{float(challenger_scorecard.get('objective_score', 0.0) or 0.0):.2f}"
            )
        if live_realized_negative:
            return (
                "live_realized_negative_objective_"
                f"{live_pnl_objective_score:.2f}"
            )
        if (
            live_pnl_closed_trade_count >= minimum_live_realized_trade_count
            and live_pnl_objective_quality + 1e-12
            < minimum_live_realized_objective_quality
        ):
            return (
                "live_realized_objective_quality_"
                f"{live_pnl_objective_quality:.2f}"
                f"_below_{minimum_live_realized_objective_quality:.2f}"
            )
        if (
            live_pnl_closed_trade_count >= minimum_live_realized_trade_count
            and live_pnl_expectancy_pct + 1e-12
            < minimum_live_realized_expectancy_pct
        ):
            return (
                "live_realized_expectancy_"
                f"{live_pnl_expectancy_pct:.2f}"
                f"_below_{minimum_live_realized_expectancy_pct:.2f}"
            )
        if (
            live_pnl_closed_trade_count >= minimum_live_realized_trade_count
            and live_pnl_profit_factor + 1e-12
            < minimum_live_realized_profit_factor
        ):
            return (
                "live_realized_profit_factor_"
                f"{live_pnl_profit_factor:.2f}"
                f"_below_{minimum_live_realized_profit_factor:.2f}"
            )
        if (
            live_pnl_closed_trade_count >= minimum_live_realized_trade_count
            and live_pnl_max_drawdown_pct - 1e-12
            > maximum_live_realized_drawdown_pct
        ):
            return (
                "live_realized_drawdown_"
                f"{live_pnl_max_drawdown_pct:.2f}"
                f"_above_{maximum_live_realized_drawdown_pct:.2f}"
            )
        if (
            live_pnl_closed_trade_count >= minimum_live_realized_trade_count
            and live_pnl_avg_trade_return_pct + 1e-12
            < minimum_live_realized_trade_return
        ):
            return (
                "live_realized_trade_return_"
                f"{live_pnl_avg_trade_return_pct:.2f}"
                f"_below_{minimum_live_realized_trade_return:.2f}"
            )
        if challenger_objective_quality + 1e-12 < minimum_live_objective_quality:
            return (
                "live_objective_quality_"
                f"{challenger_objective_quality:.2f}"
                f"_below_{minimum_live_objective_quality:.2f}"
            )
        if challenger_expectancy_pct + 1e-12 < minimum_live_expectancy_pct:
            return (
                "live_expectancy_"
                f"{challenger_expectancy_pct:.2f}"
                f"_below_{minimum_live_expectancy_pct:.2f}"
            )
        if challenger_profit_factor + 1e-12 < minimum_live_profit_factor:
            return (
                "live_profit_factor_"
                f"{challenger_profit_factor:.2f}"
                f"_below_{minimum_live_profit_factor:.2f}"
            )
        if challenger_max_drawdown_pct - 1e-12 > maximum_live_drawdown_pct:
            return (
                "live_drawdown_"
                f"{challenger_max_drawdown_pct:.2f}"
                f"_above_{maximum_live_drawdown_pct:.2f}"
            )
        if (
            float(challenger_scorecard.get("avg_trade_return_pct", 0.0) or 0.0)
            + 1e-12
            < minimum_live_trade_return
        ):
            return (
                "live_trade_return_"
                f"{float(challenger_scorecard.get('avg_trade_return_pct', 0.0) or 0.0):.2f}"
                f"_below_{minimum_live_trade_return:.2f}"
            )
        return (
            f"live_accuracy_{live_accuracy:.2f}"
            f"_below_{minimum_live_accuracy:.2f}"
        )
