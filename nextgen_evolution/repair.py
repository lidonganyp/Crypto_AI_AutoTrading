"""Strategy repair engine for the autonomous next-generation runtime."""

from __future__ import annotations

from dataclasses import replace

from .config import EvolutionConfig
from .repair_feedback import RepairFeedbackSummary
from .models import PromotionStage, RepairActionType, RepairPlan, StrategyRuntimeSnapshot
from .runtime_override_policy import lifecycle_policy_runtime_override_state


class StrategyRepairEngine:
    """Turn degraded runtime evidence into candidate repair plans."""

    def __init__(self, config: EvolutionConfig | None = None):
        self.config = config or EvolutionConfig()

    def propose(
        self,
        snapshot: StrategyRuntimeSnapshot,
        feedback: RepairFeedbackSummary | None = None,
    ) -> RepairPlan:
        genome = snapshot.scorecard.genome
        params = dict(genome.params)
        reasons: list[str] = []
        runtime_overrides: dict[str, float] = {}
        priority = 1
        capital_multiplier = 1.0
        action = RepairActionType.MUTATE_AND_REVALIDATE
        validation_stage = PromotionStage.SHADOW
        runtime_recovery_mode = self._runtime_override_recovery_mode(snapshot)

        if feedback is not None and feedback.retire_recommended:
            return RepairPlan(
                strategy_id=snapshot.directive_id,
                action=RepairActionType.RETIRE,
                priority=max(5, feedback.consecutive_failures + 2),
                candidate_genome=None,
                validation_stage=PromotionStage.REJECT,
                capital_multiplier=0.0,
                runtime_overrides={
                    "retire_recommended": 1.0,
                    "repair_attempts": float(feedback.attempts),
                    "repair_consecutive_failures": float(feedback.consecutive_failures),
                },
                reasons=self._dedupe_reasons(
                    [
                        "repair_lineage_exhausted",
                        (
                            f"repair_runtime_override_recovery_mode:{runtime_recovery_mode}"
                            if runtime_recovery_mode
                            else ""
                        ),
                        f"repair_failures:{feedback.failures}",
                        f"repair_consecutive_failures:{feedback.consecutive_failures}",
                    ]
                ),
            )

        if snapshot.current_drawdown_pct >= self.config.autonomy_repair_drawdown_pct:
            params = self._tighten_risk(params)
            runtime_overrides["max_weight_multiplier"] = round(
                self.config.autonomy_live_scale_down_factor,
                4,
            )
            capital_multiplier = min(
                capital_multiplier,
                self.config.autonomy_live_scale_down_factor,
            )
            priority += 3
            reasons.append("drawdown_breach")
            if snapshot.stage == PromotionStage.LIVE:
                action = RepairActionType.QUARANTINE
            else:
                action = RepairActionType.TIGHTEN_RISK

        if snapshot.metrics.live_expectancy < self.config.autonomy_repair_expectancy_floor:
            params = self._mutate_for_expectancy(params)
            priority += 2
            reasons.append("live_expectancy_below_floor")

        if snapshot.metrics.walkforward_expectancy < 0:
            params = self._mutate_for_expectancy(params)
            priority += 1
            reasons.append("walkforward_expectancy_negative")

        if snapshot.metrics.cost_drag_pct >= self.config.max_cost_drag_pct * 0.70:
            params = self._raise_selectivity(params)
            priority += 1
            reasons.append("cost_drag_high")
            if action == RepairActionType.MUTATE_AND_REVALIDATE:
                action = RepairActionType.RAISE_SELECTIVITY

        if snapshot.metrics.regime_consistency < self.config.min_regime_consistency:
            params = self._specialize_regime(params)
            priority += 1
            reasons.append("regime_consistency_low")

        if snapshot.consecutive_losses >= 3:
            params = self._tighten_risk(params)
            priority += 1
            reasons.append("loss_streak")

        if feedback is not None:
            params, action, feedback_reasons, feedback_overrides, feedback_capital_multiplier = self._apply_feedback_bias(
                params=params,
                action=action,
                feedback=feedback,
            )
            reasons.extend(feedback_reasons)
            runtime_overrides.update(feedback_overrides)
            capital_multiplier = min(capital_multiplier, feedback_capital_multiplier)
            if feedback.probation_required:
                runtime_overrides.setdefault(
                    "max_weight_multiplier",
                    round(self.config.autonomy_live_scale_down_factor, 4),
                )
                capital_multiplier = min(
                    capital_multiplier,
                    self.config.autonomy_live_scale_down_factor,
                )
                priority += min(2, max(0, feedback.consecutive_failures))
                validation_stage = PromotionStage.SHADOW
                reasons.append("repair_probation")
            elif (
                feedback.suggested_validation_stage == PromotionStage.PAPER
                and snapshot.stage != PromotionStage.LIVE
            ):
                validation_stage = PromotionStage.PAPER
                reasons.append("repair_lineage_recovered")

        (
            params,
            action,
            priority,
            validation_stage,
            recovery_reasons,
            recovery_overrides,
            recovery_capital_multiplier,
        ) = self._apply_runtime_recovery_bias(
            snapshot=snapshot,
            params=params,
            action=action,
            priority=priority,
            validation_stage=validation_stage,
        )
        reasons.extend(recovery_reasons)
        runtime_overrides.update(recovery_overrides)
        capital_multiplier = min(capital_multiplier, recovery_capital_multiplier)

        if not reasons:
            reasons.append("continuous_improvement")

        candidate_genome = replace(
            genome,
            strategy_id=self._build_repair_strategy_id(
                genome.strategy_id,
                snapshot.directive_id,
            ),
            params=params,
            mutation_of=genome.strategy_id,
            tags=self._merge_tags(genome.tags, ("repair", action.value)),
        )
        return RepairPlan(
            strategy_id=snapshot.directive_id,
            action=action,
            priority=priority,
            candidate_genome=candidate_genome,
            validation_stage=validation_stage,
            capital_multiplier=capital_multiplier,
            runtime_overrides=runtime_overrides,
            reasons=self._dedupe_reasons(reasons),
        )

    @staticmethod
    def _merge_tags(existing: tuple[str, ...], extra: tuple[str, ...]) -> tuple[str, ...]:
        tags: list[str] = []
        for value in existing + extra:
            if value not in tags:
                tags.append(value)
        return tuple(tags)

    @staticmethod
    def _dedupe_reasons(reasons: list[str]) -> list[str]:
        merged: list[str] = []
        for reason in reasons:
            value = str(reason).strip()
            if value and value not in merged:
                merged.append(value)
        return merged

    def _apply_feedback_bias(
        self,
        *,
        params: dict[str, float],
        action: RepairActionType,
        feedback: RepairFeedbackSummary,
    ) -> tuple[dict[str, float], RepairActionType, list[str], dict[str, float], float]:
        updated = dict(params)
        reasons: list[str] = []
        runtime_overrides: dict[str, float] = {}
        capital_multiplier = 1.0
        autonomy_outcomes = dict(feedback.notes.get("autonomy_outcomes") or {})
        if feedback.preferred_action == RepairActionType.RAISE_SELECTIVITY:
            updated = self._raise_selectivity(updated)
            updated = self._shorten_holding_window(updated, factor=0.90)
            if action == RepairActionType.MUTATE_AND_REVALIDATE:
                action = RepairActionType.RAISE_SELECTIVITY
            reasons.append("repair_feedback_selectivity_bias")
            runtime_overrides["max_weight_multiplier"] = 0.85
            runtime_overrides["entry_cooldown_bars_multiplier"] = 1.10
            capital_multiplier = min(capital_multiplier, 0.85)
        elif feedback.preferred_action == RepairActionType.TIGHTEN_RISK:
            updated = self._tighten_risk(updated)
            updated = self._shorten_holding_window(updated, factor=0.80)
            if action == RepairActionType.MUTATE_AND_REVALIDATE:
                action = RepairActionType.TIGHTEN_RISK
            reasons.append("repair_feedback_risk_bias")
            runtime_overrides["max_weight_multiplier"] = round(
                self.config.autonomy_live_scale_down_factor,
                4,
            )
            runtime_overrides["entry_cooldown_bars_multiplier"] = 1.25
            capital_multiplier = min(
                capital_multiplier,
                self.config.autonomy_live_scale_down_factor,
            )
        if int(autonomy_outcomes.get("profit_lock_harvest_count") or 0) >= 2:
            updated = self._shorten_holding_window(updated, factor=0.92)
            runtime_overrides["take_profit_bias"] = 1.10
            reasons.append("repair_feedback_profit_lock_harvest_bias")
        if int(autonomy_outcomes.get("profit_lock_exit_count") or 0) > 0:
            updated = self._tighten_risk(updated)
            runtime_overrides["take_profit_bias"] = 0.95
            runtime_overrides["max_weight_multiplier"] = min(
                float(runtime_overrides.get("max_weight_multiplier", 1.0)),
                0.45,
            )
            capital_multiplier = min(capital_multiplier, 0.45)
            reasons.append("repair_feedback_profit_lock_exit_bias")
        if int(autonomy_outcomes.get("forced_exit_count") or 0) >= 2:
            updated = self._tighten_risk(updated)
            runtime_overrides["entry_cooldown_bars_multiplier"] = max(
                float(runtime_overrides.get("entry_cooldown_bars_multiplier", 1.0)),
                1.35,
            )
            runtime_overrides["max_weight_multiplier"] = min(
                float(runtime_overrides.get("max_weight_multiplier", 1.0)),
                0.40,
            )
            capital_multiplier = min(capital_multiplier, 0.40)
            reasons.append("repair_feedback_forced_exit_bias")
        return updated, action, reasons, runtime_overrides, capital_multiplier

    def _apply_runtime_recovery_bias(
        self,
        *,
        snapshot: StrategyRuntimeSnapshot,
        params: dict[str, float],
        action: RepairActionType,
        priority: int,
        validation_stage: PromotionStage,
    ) -> tuple[dict[str, float], RepairActionType, int, PromotionStage, list[str], dict[str, float], float]:
        recovery_mode = self._runtime_override_recovery_mode(snapshot)
        if not recovery_mode:
            return (
                dict(params),
                action,
                priority,
                validation_stage,
                [],
                {},
                1.0,
            )
        updated = dict(params)
        reasons = [f"repair_runtime_override_recovery_mode:{recovery_mode}"]
        runtime_overrides: dict[str, float] = {}
        capital_multiplier = 1.0
        if recovery_mode == "hold":
            updated = self._tighten_risk(updated)
            updated = self._shorten_holding_window(updated, factor=0.85)
            if action in {
                RepairActionType.MUTATE_AND_REVALIDATE,
                RepairActionType.RAISE_SELECTIVITY,
            }:
                action = RepairActionType.TIGHTEN_RISK
            runtime_overrides["max_weight_multiplier"] = round(
                self.config.autonomy_live_scale_down_factor,
                4,
            )
            runtime_overrides["entry_cooldown_bars_multiplier"] = 1.25
            capital_multiplier = min(
                capital_multiplier,
                self.config.autonomy_live_scale_down_factor,
            )
            priority += 2
            validation_stage = PromotionStage.SHADOW
            reasons.append("repair_runtime_override_hold_bias")
        elif recovery_mode == "accelerate":
            priority = max(1, int(priority) - 1)
            reasons.append("repair_runtime_override_accelerate_relief")
        elif recovery_mode == "release":
            priority = max(1, int(priority) - 2)
            reasons.append("repair_runtime_override_release_relief")
        return (
            updated,
            action,
            priority,
            validation_stage,
            reasons,
            runtime_overrides,
            capital_multiplier,
        )

    @staticmethod
    def _runtime_override_recovery_mode(snapshot: StrategyRuntimeSnapshot) -> str:
        state = lifecycle_policy_runtime_override_state(snapshot.notes)
        return str(state.get("recovery_mode") or "").strip().lower()

    @staticmethod
    def _build_repair_strategy_id(strategy_id: str, runtime_id: str | None = None) -> str:
        if not runtime_id:
            return f"{strategy_id}:repair"
        parts = str(runtime_id).split("|", 2)
        if len(parts) < 2:
            return f"{strategy_id}:repair"
        symbol = "".join(char if char.isalnum() else "_" for char in parts[0].upper())
        timeframe = "".join(char if char.isalnum() else "_" for char in parts[1].lower())
        return f"{strategy_id}@{symbol}_{timeframe}:repair"

    @staticmethod
    def _tighten_risk(params: dict[str, float]) -> dict[str, float]:
        updated = dict(params)
        for key, value in list(updated.items()):
            lowered = str(key).lower()
            if "hold" in lowered or "cooldown" in lowered:
                updated[key] = round(max(1.0, float(value) * 0.75), 6)
            elif lowered == "stop" or lowered.endswith("_stop"):
                updated[key] = round(max(0.0005, float(value) * 0.85), 6)
            elif "lookback" in lowered:
                updated[key] = round(max(2.0, float(value) * 1.10), 6)
        return updated

    @staticmethod
    def _raise_selectivity(params: dict[str, float]) -> dict[str, float]:
        updated = dict(params)
        for key, value in list(updated.items()):
            lowered = str(key).lower()
            if any(token in lowered for token in ("buffer", "floor", "zscore", "shock", "ceiling")):
                updated[key] = round(max(0.0, float(value) * 1.10), 6)
        return updated

    @staticmethod
    def _mutate_for_expectancy(params: dict[str, float]) -> dict[str, float]:
        updated = dict(params)
        for key, value in list(updated.items()):
            lowered = str(key).lower()
            if "hold" in lowered:
                updated[key] = round(max(1.0, float(value) * 0.80), 6)
            elif any(token in lowered for token in ("buffer", "floor", "zscore", "shock")):
                updated[key] = round(max(0.0, float(value) * 1.08), 6)
        return updated

    @staticmethod
    def _specialize_regime(params: dict[str, float]) -> dict[str, float]:
        updated = dict(params)
        for key, value in list(updated.items()):
            lowered = str(key).lower()
            if "lookback" in lowered:
                updated[key] = round(max(2.0, float(value) * 1.15), 6)
            elif "cooldown" in lowered:
                updated[key] = round(max(1.0, float(value) * 1.20), 6)
        return updated

    @staticmethod
    def _shorten_holding_window(params: dict[str, float], *, factor: float) -> dict[str, float]:
        updated = dict(params)
        for key, value in list(updated.items()):
            lowered = str(key).lower()
            if "hold" in lowered:
                updated[key] = round(max(1.0, float(value) * float(factor)), 6)
        return updated
