"""Portfolio-level allocation across symbols and strategy families."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime

from .config import EvolutionConfig
from .experiment_lab import ExperimentResult
from .models import (
    AutonomyDirective,
    PortfolioAllocation,
    PromotionStage,
    RuntimeEvidenceSnapshot,
    RuntimeState,
    ScoreCard,
)
from .runtime_override_policy import (
    apply_staged_exit_to_capital,
    lifecycle_policy_repair_reentry_notes,
    merged_reentry_state,
    merged_repair_reentry_notes,
    merged_staged_exit_state,
    merged_runtime_overrides,
    reentry_state_active,
    staged_exit_active,
)


class PortfolioAllocator:
    """Allocate capital across batch experiment results with symbol and family caps."""

    RUNTIME_OVERRIDE_WEIGHT_CAP_REASON = "portfolio_runtime_override_weight_cap_applied"
    RUNTIME_OVERRIDE_HOLD_CAP_REASON = "portfolio_runtime_override_hold_cap_applied"
    STAGED_EXIT_CAP_REASON = "portfolio_staged_exit_cap_applied"

    def __init__(self, config: EvolutionConfig | None = None):
        self.config = config or EvolutionConfig()

    def allocate(
        self,
        results: list[ExperimentResult],
        total_capital: float,
        previous_states: list[RuntimeState] | None = None,
        runtime_evidence: dict[str, RuntimeEvidenceSnapshot] | None = None,
        latest_close_time_by_runtime: dict[str, datetime] | None = None,
        directive: AutonomyDirective | None = None,
    ) -> list[PortfolioAllocation]:
        if total_capital <= 0:
            return []

        candidates = self._portfolio_candidates(results)
        if not candidates:
            return []

        candidates = candidates[: self.config.max_portfolio_positions]
        timeframe_by_symbol = {
            result.symbol: result.timeframe
            for result in results
        }
        symbol_remaining = defaultdict(
            lambda: total_capital * self.config.max_portfolio_symbol_weight
        )
        family_remaining = defaultdict(
            lambda: total_capital * self.config.max_portfolio_family_weight
        )
        strategy_remaining = {
            self._candidate_key(symbol, card): total_capital * self._strategy_cap(card)
            for symbol, card in candidates
        }
        assigned: dict[tuple[str, str], float] = defaultdict(float)
        remaining_capital = total_capital
        active = list(candidates)

        for _ in range(max(2, len(active) * 3)):
            if remaining_capital <= 0 or not active:
                break
            raw_total = sum(
                self._score_weight(card)
                for symbol, card in active
                if self._available_capacity(
                    symbol=symbol,
                    card=card,
                    strategy_remaining=strategy_remaining,
                    symbol_remaining=symbol_remaining,
                    family_remaining=family_remaining,
                )
                > 0
            )
            if raw_total <= 0:
                break

            round_remaining = remaining_capital
            round_symbol_remaining = dict(symbol_remaining)
            round_family_remaining = dict(family_remaining)
            round_strategy_remaining = dict(strategy_remaining)
            increments: list[tuple[str, ScoreCard, float]] = []
            for symbol, card in active:
                available = self._available_capacity(
                    symbol=symbol,
                    card=card,
                    strategy_remaining=round_strategy_remaining,
                    symbol_remaining=round_symbol_remaining,
                    family_remaining=round_family_remaining,
                )
                if available <= 0:
                    continue
                target = round_remaining * self._score_weight(card) / raw_total
                increment = min(target, available)
                if increment > 0:
                    increments.append((symbol, card, increment))
                    key = self._candidate_key(symbol, card)
                    round_strategy_remaining[key] -= increment
                    round_symbol_remaining[symbol] -= increment
                    round_family_remaining[card.genome.family] -= increment

            if not increments:
                break

            next_active: list[tuple[str, ScoreCard]] = []
            progress = 0.0
            for symbol, card, increment in increments:
                key = self._candidate_key(symbol, card)
                assigned[key] += increment
                strategy_remaining[key] -= increment
                symbol_remaining[symbol] -= increment
                family_remaining[card.genome.family] -= increment
                progress += increment
                if self._available_capacity(
                    symbol=symbol,
                    card=card,
                    strategy_remaining=strategy_remaining,
                    symbol_remaining=symbol_remaining,
                    family_remaining=family_remaining,
                ) > 1e-6:
                    next_active.append((symbol, card))

            remaining_capital -= progress
            if progress <= 1e-6:
                break
            active = next_active

        allocations: list[PortfolioAllocation] = []
        for symbol, card in candidates:
            key = self._candidate_key(symbol, card)
            capital = assigned.get(key, 0.0)
            if capital <= 0:
                continue
            allocations.append(
                PortfolioAllocation(
                    symbol=symbol,
                    strategy_id=card.genome.strategy_id,
                    family=card.genome.family,
                    stage=card.stage,
                    allocated_capital=round(capital, 2),
                    weight=round(capital / total_capital, 4),
                    score=card.total_score,
                    timeframe=timeframe_by_symbol.get(symbol, ""),
                    reasons=list(card.reasons)
                    + ["portfolio_symbol_capped", "portfolio_family_capped"],
                )
            )
        allocations.sort(
            key=lambda item: (item.weight, item.score, item.allocated_capital),
            reverse=True,
        )
        return self._apply_runtime_override_constraints(
            allocations=allocations,
            results=results,
            total_capital=total_capital,
            previous_states=previous_states,
            runtime_evidence=runtime_evidence,
            latest_close_time_by_runtime=latest_close_time_by_runtime,
            directive=directive,
        )

    def _portfolio_candidates(
        self,
        results: list[ExperimentResult],
    ) -> list[tuple[str, ScoreCard]]:
        stage_rank = {
            PromotionStage.LIVE: 2,
            PromotionStage.PAPER: 1,
        }
        ordered: list[tuple[str, ScoreCard]] = []
        for result in results:
            for card in result.promoted:
                if card.stage not in {PromotionStage.PAPER, PromotionStage.LIVE}:
                    continue
                ordered.append((result.symbol, card))
        ordered.sort(
            key=lambda item: (
                stage_rank.get(item[1].stage, 0),
                item[1].total_score,
                item[1].deployment_score,
            ),
            reverse=True,
        )

        lineage_counts: dict[tuple[str, str], int] = defaultdict(int)
        diversified: list[tuple[str, ScoreCard]] = []
        for symbol, card in ordered:
            lineage = self._lineage_key(card)
            lineage_key = (symbol, lineage)
            if lineage_counts[lineage_key] >= self.config.max_allocations_per_lineage:
                continue
            lineage_counts[lineage_key] += 1
            diversified.append((symbol, card))
        return diversified

    @staticmethod
    def _lineage_key(card: ScoreCard) -> str:
        return card.genome.mutation_of or f"{card.genome.family}:seed"

    @staticmethod
    def _candidate_key(symbol: str, card: ScoreCard) -> tuple[str, str]:
        return symbol, card.genome.strategy_id

    @staticmethod
    def _score_weight(card: ScoreCard) -> float:
        stage_boost = 1.15 if card.stage == PromotionStage.LIVE else 1.0
        return max(0.0, card.total_score) * stage_boost

    def _strategy_cap(self, card: ScoreCard) -> float:
        return (
            self.config.max_live_weight
            if card.stage == PromotionStage.LIVE
            else self.config.max_paper_weight
        )

    @staticmethod
    def _available_capacity(
        *,
        symbol: str,
        card: ScoreCard,
        strategy_remaining: dict[tuple[str, str], float],
        symbol_remaining: dict[str, float],
        family_remaining: dict[str, float],
    ) -> float:
        return min(
            strategy_remaining.get((symbol, card.genome.strategy_id), 0.0),
            symbol_remaining[symbol],
            family_remaining[card.genome.family],
        )

    def _apply_runtime_override_constraints(
        self,
        *,
        allocations: list[PortfolioAllocation],
        results: list[ExperimentResult],
        total_capital: float,
        previous_states: list[RuntimeState] | None,
        runtime_evidence: dict[str, RuntimeEvidenceSnapshot] | None,
        latest_close_time_by_runtime: dict[str, datetime] | None,
        directive: AutonomyDirective | None,
    ) -> list[PortfolioAllocation]:
        if not allocations:
            return []
        override_index = self._runtime_override_policy_index(
            results=results,
            previous_states=previous_states,
            runtime_evidence=runtime_evidence,
        )
        reentry_state_index = self._reentry_state_index(
            results=results,
            previous_states=previous_states,
            runtime_evidence=runtime_evidence,
            latest_close_time_by_runtime=latest_close_time_by_runtime,
        )
        staged_exit_index = self._staged_exit_index(
            results=results,
            previous_states=previous_states,
            runtime_evidence=runtime_evidence,
            directive=directive,
        )
        if not override_index and not staged_exit_index:
            return allocations
        previous_index = {
            item.runtime_id: item
            for item in (previous_states or [])
        }
        adjusted: list[PortfolioAllocation] = []
        for item in allocations:
            runtime_id = self.runtime_id(
                item.symbol,
                item.timeframe or "5m",
                item.strategy_id,
            )
            runtime_policy = dict(override_index.get(runtime_id) or {})
            runtime_overrides = dict(runtime_policy.get("runtime_overrides") or {})
            runtime_override_state = dict(runtime_policy.get("runtime_override_state") or {})
            staged_exit_state = dict(staged_exit_index.get(runtime_id) or {})
            reentry_state = dict(reentry_state_index.get(runtime_id) or {})
            if reentry_state_active(reentry_state) and str(reentry_state.get("phase") or "") == "cooldown":
                continue
            adjusted_capital = float(item.allocated_capital)
            reasons = list(item.reasons)
            max_weight_multiplier = float(
                runtime_overrides.get("max_weight_multiplier") or 1.0
            )
            if 0.0 < max_weight_multiplier < 1.0:
                adjusted_capital = round(adjusted_capital * max_weight_multiplier, 2)
                if adjusted_capital <= 0:
                    continue
                if self.RUNTIME_OVERRIDE_WEIGHT_CAP_REASON not in reasons:
                    reasons.append(self.RUNTIME_OVERRIDE_WEIGHT_CAP_REASON)
            if str(runtime_override_state.get("recovery_mode") or "").strip().lower() == "hold":
                hold_capital = self._runtime_override_hold_capital_limit(
                    previous=previous_index.get(runtime_id),
                    runtime_evidence=(runtime_evidence or {}).get(runtime_id),
                )
                if hold_capital > 0 and adjusted_capital > hold_capital + 1e-6:
                    adjusted_capital = round(hold_capital, 2)
                    if self.RUNTIME_OVERRIDE_HOLD_CAP_REASON not in reasons:
                        reasons.append(self.RUNTIME_OVERRIDE_HOLD_CAP_REASON)
            if staged_exit_active(staged_exit_state):
                staged_capital = apply_staged_exit_to_capital(
                    desired_capital=adjusted_capital,
                    allocated_capital=adjusted_capital,
                    staged_exit_state=staged_exit_state,
                )
                if staged_capital <= 0:
                    continue
                if staged_capital < adjusted_capital - 1e-6:
                    adjusted_capital = staged_capital
                    if self.STAGED_EXIT_CAP_REASON not in reasons:
                        reasons.append(self.STAGED_EXIT_CAP_REASON)
            adjusted.append(
                replace(
                    item,
                    allocated_capital=round(adjusted_capital, 2),
                    weight=round(float(adjusted_capital) / total_capital, 4),
                    reasons=reasons,
                )
            )
        adjusted.sort(
            key=lambda item: (item.weight, item.score, item.allocated_capital),
            reverse=True,
        )
        return adjusted

    def _runtime_override_policy_index(
        self,
        *,
        results: list[ExperimentResult],
        previous_states: list[RuntimeState] | None,
        runtime_evidence: dict[str, RuntimeEvidenceSnapshot] | None,
    ) -> dict[str, dict]:
        previous_index = {
            item.runtime_id: item
            for item in (previous_states or [])
        }
        override_index: dict[str, dict] = {}
        for result in results:
            current_notes = lifecycle_policy_repair_reentry_notes(result.notes)
            for card in result.promoted:
                runtime_id = self.runtime_id(
                    result.symbol,
                    result.timeframe,
                    card.genome.strategy_id,
                )
                merged, override_state = merged_runtime_overrides(
                    config=self.config,
                    previous=previous_index.get(runtime_id),
                    repair_reentry_notes=current_notes,
                    runtime_evidence=(runtime_evidence or {}).get(runtime_id),
                )
                if merged or override_state:
                    override_index[runtime_id] = {
                        "runtime_overrides": merged,
                        "runtime_override_state": override_state,
                    }
        return override_index

    @staticmethod
    def _runtime_override_hold_capital_limit(
        *,
        previous: RuntimeState | None,
        runtime_evidence: RuntimeEvidenceSnapshot | None,
    ) -> float:
        limits: list[float] = []
        if previous is not None:
            if float(previous.desired_capital or 0.0) > 0:
                limits.append(float(previous.desired_capital))
            if float(previous.current_capital or 0.0) > 0:
                limits.append(float(previous.current_capital))
        if runtime_evidence is not None and float(runtime_evidence.current_capital or 0.0) > 0:
            limits.append(float(runtime_evidence.current_capital))
        if not limits and previous is not None and float(previous.allocated_capital or 0.0) > 0:
            limits.append(float(previous.allocated_capital))
        if not limits:
            return 0.0
        return round(min(limits), 2)

    def _staged_exit_index(
        self,
        *,
        results: list[ExperimentResult],
        previous_states: list[RuntimeState] | None,
        runtime_evidence: dict[str, RuntimeEvidenceSnapshot] | None,
        directive: AutonomyDirective | None,
    ) -> dict[str, dict]:
        previous_index = {
            item.runtime_id: item
            for item in (previous_states or [])
        }
        directive_index: dict[str, object] = {}
        for item in list((directive.execution if directive is not None else []) or []):
            directive_index[str(item.strategy_id)] = item
        staged: dict[str, dict] = {}
        for result in results:
            for card in result.promoted:
                runtime_id = self.runtime_id(
                    result.symbol,
                    result.timeframe,
                    card.genome.strategy_id,
                )
                merged = merged_staged_exit_state(
                    config=self.config,
                    previous=previous_index.get(runtime_id),
                    directive=directive_index.get(runtime_id),
                    runtime_evidence=(runtime_evidence or {}).get(runtime_id),
                )
                if merged:
                    staged[runtime_id] = merged
        return staged

    def _reentry_state_index(
        self,
        *,
        results: list[ExperimentResult],
        previous_states: list[RuntimeState] | None,
        runtime_evidence: dict[str, RuntimeEvidenceSnapshot] | None,
        latest_close_time_by_runtime: dict[str, datetime] | None,
    ) -> dict[str, dict]:
        previous_index = {
            item.runtime_id: item
            for item in (previous_states or [])
        }
        merged: dict[str, dict] = {}
        for result in results:
            current_notes = lifecycle_policy_repair_reentry_notes(result.notes)
            for card in result.promoted:
                runtime_id = self.runtime_id(
                    result.symbol,
                    result.timeframe,
                    card.genome.strategy_id,
                )
                repair_notes = merged_repair_reentry_notes(
                    previous=previous_index.get(runtime_id),
                    current_notes=current_notes,
                )
                runtime_overrides, runtime_override_state = merged_runtime_overrides(
                    config=self.config,
                    previous=previous_index.get(runtime_id),
                    repair_reentry_notes=current_notes,
                    runtime_evidence=(runtime_evidence or {}).get(runtime_id),
                )
                merged[runtime_id] = merged_reentry_state(
                    config=self.config,
                    previous=previous_index.get(runtime_id),
                    repair_reentry_notes=repair_notes,
                    runtime_overrides=runtime_overrides,
                    runtime_override_state=runtime_override_state,
                    latest_close_time_by_runtime=latest_close_time_by_runtime,
                    runtime_id=runtime_id,
                    timeframe=result.timeframe,
                )
        return merged

    @staticmethod
    def _repair_runtime_overrides_from_result(
        result: ExperimentResult,
    ) -> dict[str, float]:
        raw = dict(lifecycle_policy_repair_reentry_notes(result.notes)).get("runtime_overrides") or {}
        overrides: dict[str, float] = {}
        for key, value in dict(raw).items():
            name = str(key).strip()
            if not name:
                continue
            try:
                overrides[name] = float(value)
            except (TypeError, ValueError):
                continue
        return overrides

    @staticmethod
    def runtime_id(symbol: str, timeframe: str, strategy_id: str) -> str:
        return f"{symbol}|{timeframe}|{strategy_id}"
