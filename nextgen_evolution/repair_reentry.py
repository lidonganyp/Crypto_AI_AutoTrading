"""Plan safe runtime re-entry for successfully revalidated repair candidates."""

from __future__ import annotations

from dataclasses import replace

from .experiment_lab import ExperimentResult
from .models import ExecutionAction, ExecutionDirective, PromotionStage, ScoreCard
from .repair_cycle import RepairExecutionResult
from .runtime_override_policy import (
    build_repair_reentry_notes,
    compose_runtime_policy_notes,
    lifecycle_policy_reentry_state,
    lifecycle_policy_runtime_override_state,
    lifecycle_policy_runtime_overrides,
    lifecycle_policy_staged_exit_state,
)


class RepairReentryPlanner:
    """Convert successful repair validations into capped rollout directives."""

    _STAGE_ORDER = {
        PromotionStage.REJECT: 0,
        PromotionStage.SHADOW: 1,
        PromotionStage.PAPER: 2,
        PromotionStage.LIVE: 3,
    }
    _AUTO_STAGE_CAP = PromotionStage.PAPER

    def plan(
        self,
        repair_results: list[RepairExecutionResult],
    ) -> tuple[list[ExperimentResult], list[ExecutionDirective]]:
        accepted_results: list[ExperimentResult] = []
        directives: list[ExecutionDirective] = []
        for item in repair_results:
            normalized = self._normalized_result(item)
            if normalized is None:
                continue
            result, card = normalized
            accepted_results.append(result)
            directives.append(
                ExecutionDirective(
                    strategy_id=self.runtime_id(
                        result.symbol,
                        result.timeframe,
                        card.genome.strategy_id,
                    ),
                    action=self._action_for_stage(card.stage),
                    from_stage=PromotionStage.REJECT,
                    target_stage=card.stage,
                    capital_multiplier=float(item.plan.capital_multiplier),
                    reasons=self._directive_reasons(item, card.stage),
                )
            )
        return accepted_results, directives

    @classmethod
    def _normalized_result(
        cls,
        item: RepairExecutionResult,
    ) -> tuple[ExperimentResult, ScoreCard] | None:
        candidate = item.plan.candidate_genome
        if candidate is None:
            return None
        card = next(
            (
                score
                for score in item.experiment.scorecards
                if score.genome.strategy_id == candidate.strategy_id
            ),
            None,
        )
        if card is None or card.stage == PromotionStage.REJECT:
            return None
        capped_stage = cls._min_stage(
            card.stage,
            item.plan.validation_stage,
            cls._AUTO_STAGE_CAP,
        )
        if capped_stage == PromotionStage.REJECT:
            return None
        capped_card = replace(
            card,
            stage=capped_stage,
            reasons=cls._dedupe_reasons(
                [*card.reasons, "repair_reentry_candidate"]
            ),
        )
        notes = dict(item.experiment.notes or {})
        repair_reentry_notes = build_repair_reentry_notes(
            source_runtime_id=item.source_runtime_id,
            source_strategy_id=item.source_strategy_id,
            raw_stage=card.stage.value,
            effective_target_stage=capped_stage.value,
            requested_validation_stage=item.plan.validation_stage.value,
            runtime_overrides=dict(item.plan.runtime_overrides or {}),
        )
        notes = compose_runtime_policy_notes(
            base_notes=notes,
            repair_reentry_notes=repair_reentry_notes,
            runtime_overrides={
                **lifecycle_policy_runtime_overrides(notes),
                **dict(item.plan.runtime_overrides or {}),
            },
            runtime_override_state=lifecycle_policy_runtime_override_state(notes),
            staged_exit_state=lifecycle_policy_staged_exit_state(notes),
            reentry_state=lifecycle_policy_reentry_state(notes),
        )
        normalized = replace(
            item.experiment,
            scorecards=[capped_card],
            promoted=[capped_card],
            allocations=[],
            notes=notes,
        )
        return normalized, capped_card

    @classmethod
    def _min_stage(cls, *stages: PromotionStage) -> PromotionStage:
        return min(stages, key=lambda item: cls._STAGE_ORDER[item])

    @classmethod
    def _action_for_stage(cls, stage: PromotionStage) -> ExecutionAction:
        if stage == PromotionStage.PAPER:
            return ExecutionAction.PROMOTE_TO_PAPER
        if stage == PromotionStage.LIVE:
            return ExecutionAction.PROMOTE_TO_LIVE
        return ExecutionAction.PROMOTE_TO_SHADOW

    @staticmethod
    def _dedupe_reasons(reasons: list[str]) -> list[str]:
        merged: list[str] = []
        for reason in reasons:
            text = str(reason).strip()
            if text and text not in merged:
                merged.append(text)
        return merged

    def _directive_reasons(
        self,
        item: RepairExecutionResult,
        target_stage: PromotionStage,
    ) -> list[str]:
        return self._dedupe_reasons(
            [
                "repair_revalidation_passed",
                f"repair_action:{item.plan.action.value}",
                f"repair_target_stage:{target_stage.value}",
                *list(item.plan.reasons),
            ]
        )

    @staticmethod
    def runtime_id(symbol: str, timeframe: str, strategy_id: str) -> str:
        return f"{symbol}|{timeframe}|{strategy_id}"
