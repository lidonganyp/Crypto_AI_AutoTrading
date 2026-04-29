"""Post-promotion observation decision helpers."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


class PromotionObservationDecisionRuntime:
    """Evaluate observed promoted models and route accept/rollback decisions."""

    def __init__(self, observation_runtime):
        self.observation_runtime = observation_runtime
        self.runtime = observation_runtime.runtime

    def observe_promoted_models(self, now: datetime) -> dict[str, int]:
        candidate_updates = self.runtime.observe_promotion_candidates(now)
        observations = self.runtime.get_model_promotion_observations()
        if not observations:
            return {
                "accepted": 0,
                "rolled_back": 0,
                **candidate_updates,
            }

        accepted = 0
        rolled_back = 0
        changed = False

        for symbol, observation in list(observations.items()):
            promoted_at = str(observation.get("promoted_at") or "")
            if not promoted_at:
                observations.pop(symbol, None)
                changed = True
                continue

            active_model_path = Path(str(observation.get("active_model_path") or ""))
            current_signature = list(self.runtime.model_file_signature(active_model_path))
            if current_signature != list(observation.get("active_model_signature") or []):
                observations.pop(symbol, None)
                changed = True
                continue

            metrics = self._observation_metrics(
                symbol=symbol,
                observation=observation,
                promoted_at=promoted_at,
            )
            adaptive_requirements = self.runtime.promotion_adaptive_requirements(
                symbol,
                training_metadata=dict(observation.get("training_metadata", {}) or {}),
            )
            eval_count = int(metrics.get("eval_count", 0) or 0)
            min_evaluations = int(
                observation.get(
                    "min_evaluations",
                    adaptive_requirements.get("observation_min_evaluations", 8),
                )
                or adaptive_requirements.get("observation_min_evaluations", 8)
            )
            promoted_at_dt = self.runtime._parse_iso_datetime(promoted_at)
            max_age_hours = int(
                observation.get(
                    "max_observation_age_hours",
                    adaptive_requirements.get("observation_max_age_hours", 72),
                )
                or adaptive_requirements.get("observation_max_age_hours", 72)
            )
            if eval_count < min_evaluations:
                if (
                    promoted_at_dt is not None
                    and now - promoted_at_dt < timedelta(hours=max_age_hours)
                ):
                    continue
                observations.pop(symbol, None)
                changed = True
                continue

            should_rollback, reason = self._rollback_decision(
                observation=observation,
                metrics=metrics,
            )
            if should_rollback:
                self.observation_runtime.rollback_promoted_model(
                    symbol,
                    observation,
                    now,
                    reason,
                    metrics,
                )
                observations.pop(symbol, None)
                changed = True
                rolled_back += 1
                continue

            self.observation_runtime.accept_promoted_model(
                symbol,
                observation,
                now,
                metrics,
            )
            observations.pop(symbol, None)
            changed = True
            accepted += 1

        if changed:
            self.runtime.set_model_promotion_observations(observations)
        return {
            "accepted": accepted,
            "rolled_back": rolled_back,
            **candidate_updates,
        }

    def _observation_metrics(
        self,
        *,
        symbol: str,
        observation: dict,
        promoted_at: str,
    ) -> dict[str, float | int]:
        active_model_id = self.runtime.observation_active_model_id(observation)
        scorecard = self.runtime.build_model_scorecard(
            symbol=symbol,
            model_id=active_model_id,
            evaluation_type="execution",
            started_at=promoted_at,
        )
        return {
            "eval_count": int(scorecard.get("sample_count", 0) or 0),
            "accuracy": float(scorecard.get("accuracy", 0.0) or 0.0),
            "executed_count": int(scorecard.get("executed_count", 0) or 0),
            "expectancy_pct": float(scorecard.get("expectancy_pct", 0.0) or 0.0),
            "profit_factor": float(scorecard.get("profit_factor", 0.0) or 0.0),
            "max_drawdown_pct": float(scorecard.get("max_drawdown_pct", 0.0) or 0.0),
            "trade_win_rate": float(scorecard.get("trade_win_rate", 0.0) or 0.0),
            "avg_cost_pct": float(scorecard.get("avg_cost_pct", 0.0) or 0.0),
            "avg_trade_return_pct": float(
                scorecard.get("avg_trade_return_pct", 0.0) or 0.0
            ),
            "objective_score": float(scorecard.get("objective_score", 0.0) or 0.0),
            "objective_quality": self.runtime.objective_score_quality(scorecard),
        }

    def _rollback_decision(
        self,
        *,
        observation: dict,
        metrics: dict[str, float | int],
    ) -> tuple[bool, str]:
        accuracy = float(metrics.get("accuracy", 0.0) or 0.0)
        expectancy_pct = float(metrics.get("expectancy_pct", 0.0) or 0.0)
        profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
        max_drawdown_pct = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
        baseline_holdout_accuracy = float(
            observation.get("baseline_holdout_accuracy", 0.0) or 0.0
        )
        minimum_accuracy = self.runtime.accuracy_safety_floor(
            baseline_holdout_accuracy
        )
        minimum_expectancy_pct = max(
            -0.10,
            float(observation.get("baseline_expectancy_pct", 0.0) or 0.0) - 0.20,
        )
        minimum_profit_factor = max(
            0.80,
            float(observation.get("baseline_profit_factor", 0.0) or 0.0) - 0.25,
        )
        maximum_drawdown_pct = max(
            1.50,
            float(observation.get("baseline_max_drawdown_pct", 0.0) or 0.0) + 1.25,
        )
        minimum_objective_quality = float(
            observation.get(
                "baseline_objective_quality",
                observation.get("baseline_objective_score", 0.0),
            )
            or 0.0
        ) - 0.35
        minimum_trade_return = float(
            observation.get("baseline_avg_trade_return_pct", 0.0) or 0.0
        ) - 0.25
        recent_wf = observation.get("recent_walkforward_baseline_summary", {}) or {}
        recent_count = int(recent_wf.get("history_count", 0) or 0)
        if recent_count >= 2:
            recent_avg_expectancy = float(
                recent_wf.get(
                    "avg_expectancy_pct",
                    recent_wf.get("avg_trade_return_pct", 0.0),
                )
                or 0.0
            )
            recent_avg_profit_factor = float(
                recent_wf.get("avg_profit_factor", 0.0) or 0.0
            )
            recent_avg_drawdown = float(
                recent_wf.get("avg_max_drawdown_pct", 0.0) or 0.0
            )
            if recent_avg_expectancy > 0:
                minimum_expectancy_pct = max(
                    minimum_expectancy_pct,
                    recent_avg_expectancy
                    - max(0.15, abs(recent_avg_expectancy) * 0.35),
                )
            if recent_avg_profit_factor > 1.0:
                minimum_profit_factor = max(
                    minimum_profit_factor,
                    recent_avg_profit_factor - 0.15,
                )
            if recent_avg_drawdown > 0:
                maximum_drawdown_pct = min(
                    maximum_drawdown_pct,
                    max(
                        recent_avg_drawdown + 1.0,
                        recent_avg_drawdown * 1.35,
                    ),
                )

        objective_score = float(metrics.get("objective_score", 0.0) or 0.0)
        objective_quality = float(metrics.get("objective_quality", 0.0) or 0.0)
        avg_trade_return_pct = float(metrics.get("avg_trade_return_pct", 0.0) or 0.0)
        if objective_score < -0.10:
            return True, f"post_promotion_negative_objective_{objective_score:.2f}"
        if objective_quality + 1e-12 < minimum_objective_quality:
            return True, (
                "post_promotion_objective_quality_"
                f"{objective_quality:.2f}"
                f"_below_{minimum_objective_quality:.2f}"
            )
        if expectancy_pct + 1e-12 < minimum_expectancy_pct:
            return True, (
                "post_promotion_expectancy_"
                f"{expectancy_pct:.2f}"
                f"_below_{minimum_expectancy_pct:.2f}"
            )
        if profit_factor + 1e-12 < minimum_profit_factor:
            return True, (
                "post_promotion_profit_factor_"
                f"{profit_factor:.2f}"
                f"_below_{minimum_profit_factor:.2f}"
            )
        if max_drawdown_pct - 1e-12 > maximum_drawdown_pct:
            return True, (
                "post_promotion_drawdown_"
                f"{max_drawdown_pct:.2f}"
                f"_above_{maximum_drawdown_pct:.2f}"
            )
        if avg_trade_return_pct + 1e-12 < minimum_trade_return:
            return True, (
                "post_promotion_trade_return_"
                f"{avg_trade_return_pct:.2f}"
                f"_below_{minimum_trade_return:.2f}"
            )
        if accuracy + 1e-12 < minimum_accuracy:
            return True, (
                f"post_promotion_accuracy_{accuracy:.2f}"
                f"_below_{minimum_accuracy:.2f}"
            )
        return False, ""
