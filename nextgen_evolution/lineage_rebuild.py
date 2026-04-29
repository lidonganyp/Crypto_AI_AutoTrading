"""Plan alternative-family rebuilds after a lineage is retired."""

from __future__ import annotations

from dataclasses import replace

from .alpha_factory import StrategyPrimitive
from .config import EvolutionConfig
from .experiment_lab import ExperimentLab, ExperimentResult
from .models import AutonomyDirective, PromotionStage, RepairActionType, RepairPlan, StrategyGenome


class LineageRebuildPlanner:
    """Expand retired lineages into conservative alternative-family rebuild attempts."""

    DEFAULT_REBUILDS_PER_LINEAGE = 2
    DEFAULT_MAX_REBUILDS_PER_CYCLE = 4

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        *,
        primitives: list[StrategyPrimitive] | None = None,
    ):
        self.config = config or EvolutionConfig()
        self.primitives = list(primitives or ExperimentLab.default_primitives())

    def expand(
        self,
        directive: AutonomyDirective,
        *,
        results: list[ExperimentResult],
    ) -> AutonomyDirective:
        rebuilds = self.plan(
            directive,
            results=results,
        )
        if not rebuilds:
            return directive
        return replace(
            directive,
            repairs=self._sorted_repairs([*directive.repairs, *rebuilds]),
            notes={
                **dict(directive.notes or {}),
                "lineage_rebuild_count": len(rebuilds),
                "lineage_rebuild_runtime_ids": [item.strategy_id for item in rebuilds],
            },
        )

    def plan(
        self,
        directive: AutonomyDirective,
        *,
        results: list[ExperimentResult],
    ) -> list[RepairPlan]:
        runtime_index = self._runtime_index(results)
        existing_strategy_ids = {
            card.genome.strategy_id
            for result in results
            for card in result.scorecards
        }
        generated_strategy_ids: set[str] = set()
        rebuilds: list[RepairPlan] = []
        max_rebuilds = self.DEFAULT_MAX_REBUILDS_PER_CYCLE
        rebuilds_per_lineage = self.DEFAULT_REBUILDS_PER_LINEAGE

        for plan in self._retire_repairs(directive):
            if len(rebuilds) >= max_rebuilds:
                break
            source_context = runtime_index.get(plan.strategy_id)
            if source_context is None:
                continue
            source_result, source_card = source_context
            ordered = self._ordered_primitives(
                source_family=source_card.genome.family,
                reasons=plan.reasons,
            )
            local_count = 0
            for primitive in ordered:
                if local_count >= rebuilds_per_lineage or len(rebuilds) >= max_rebuilds:
                    break
                candidate = self._build_candidate(
                    primitive=primitive,
                    source_runtime_id=plan.strategy_id,
                    source_result=source_result,
                    source_family=source_card.genome.family,
                )
                if (
                    candidate.strategy_id in existing_strategy_ids
                    or candidate.strategy_id in generated_strategy_ids
                ):
                    continue
                rebuilds.append(
                    RepairPlan(
                        strategy_id=plan.strategy_id,
                        action=RepairActionType.REBUILD_LINEAGE,
                        priority=max(1, int(plan.priority) - local_count),
                        candidate_genome=candidate,
                        validation_stage=PromotionStage.SHADOW,
                        capital_multiplier=self._rebuild_capital_multiplier(),
                        runtime_overrides={"lineage_reset": 1.0},
                        reasons=self._dedupe_reasons(
                            [
                                "lineage_rebuild_requested",
                                "repair_lineage_exhausted",
                                f"rebuild_source_family:{source_card.genome.family}",
                                f"rebuild_target_family:{candidate.family}",
                                f"rebuild_source_runtime:{plan.strategy_id}",
                                *list(plan.reasons),
                            ]
                        ),
                    )
                )
                generated_strategy_ids.add(candidate.strategy_id)
                local_count += 1
        return self._sorted_repairs(rebuilds)

    @staticmethod
    def _runtime_index(
        results: list[ExperimentResult],
    ) -> dict[str, tuple[ExperimentResult, object]]:
        indexed: dict[str, tuple[ExperimentResult, object]] = {}
        for result in results:
            for card in result.scorecards:
                indexed[
                    f"{result.symbol}|{result.timeframe}|{card.genome.strategy_id}"
                ] = (result, card)
        return indexed

    @staticmethod
    def _retire_repairs(directive: AutonomyDirective) -> list[RepairPlan]:
        return sorted(
            [
                item
                for item in directive.repairs
                if item.action == RepairActionType.RETIRE
            ],
            key=lambda item: item.priority,
            reverse=True,
        )

    def _ordered_primitives(
        self,
        *,
        source_family: str,
        reasons: list[str],
    ) -> list[StrategyPrimitive]:
        scored: list[tuple[tuple[int, int, int], StrategyPrimitive]] = []
        source_tags = self._family_tags(source_family)
        source_bucket = self._family_bucket(source_tags, source_family)
        reason_set = {str(item).strip() for item in reasons if str(item).strip()}
        regime_switch_bias = "regime_consistency_low" in reason_set

        for index, primitive in enumerate(self.primitives):
            if primitive.family == source_family:
                continue
            target_tags = set(primitive.tags)
            target_bucket = self._family_bucket(target_tags, primitive.family)
            overlap = len(source_tags & target_tags)
            score = 0
            if target_bucket != source_bucket:
                score += 3
            if regime_switch_bias and target_bucket != source_bucket:
                score += 2
            if overlap == 0:
                score += 1
            scored.append(((score, -overlap, -index), primitive))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored]

    def _build_candidate(
        self,
        *,
        primitive: StrategyPrimitive,
        source_runtime_id: str,
        source_result: ExperimentResult,
        source_family: str,
    ) -> StrategyGenome:
        symbol_slug = "".join(
            char if char.isalnum() else "_"
            for char in source_result.symbol.upper()
        )
        timeframe_slug = "".join(
            char if char.isalnum() else "_"
            for char in source_result.timeframe.lower()
        )
        source_family_slug = "".join(
            char if char.isalnum() else "_"
            for char in str(source_family).lower()
        )
        return StrategyGenome(
            strategy_id=(
                f"{primitive.family}@{symbol_slug}_{timeframe_slug}:"
                f"rebuild_from_{source_family_slug}"
            ),
            family=primitive.family,
            params=dict(primitive.base_params),
            mutation_of=None,
            tags=self._merge_tags(
                primitive.tags,
                (
                    "repair",
                    "rebuild_lineage",
                    f"source_runtime:{source_runtime_id}",
                    f"source_family:{source_family}",
                ),
            ),
        )

    def _family_tags(self, family: str) -> set[str]:
        for primitive in self.primitives:
            if primitive.family == family:
                return set(primitive.tags)
        return set()

    @staticmethod
    def _family_bucket(tags: set[str], family: str) -> str:
        family_text = str(family).lower()
        if {"trend", "momentum", "breakout", "pullback"} & tags:
            return "trend"
        if {"reversion", "intraday"} & tags:
            return "reversion"
        if {"volatility", "reversal"} & tags:
            return "volatility"
        if "trend" in family_text or "breakout" in family_text:
            return "trend"
        if "revert" in family_text:
            return "reversion"
        if "volatility" in family_text or "reclaim" in family_text:
            return "volatility"
        return "general"

    def _rebuild_capital_multiplier(self) -> float:
        return round(
            max(0.1, min(1.0, float(self.config.autonomy_live_scale_down_factor))),
            4,
        )

    @staticmethod
    def _merge_tags(existing: tuple[str, ...], extra: tuple[str, ...]) -> tuple[str, ...]:
        merged: list[str] = []
        for value in existing + extra:
            text = str(value).strip()
            if text and text not in merged:
                merged.append(text)
        return tuple(merged)

    @staticmethod
    def _dedupe_reasons(reasons: list[str]) -> list[str]:
        merged: list[str] = []
        for reason in reasons:
            text = str(reason).strip()
            if text and text not in merged:
                merged.append(text)
        return merged

    @staticmethod
    def _sorted_repairs(repairs: list[RepairPlan]) -> list[RepairPlan]:
        return sorted(repairs, key=lambda item: item.priority, reverse=True)
