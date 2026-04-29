"""High-level orchestration wrapper for the autonomous next-generation stack."""

from __future__ import annotations

from dataclasses import replace

from .config import EvolutionConfig
from .experiment_lab import ExperimentResult
from .repair_feedback import RepairFeedbackSummary
from .runtime_override_policy import (
    compose_runtime_policy_notes,
    lifecycle_policy_repair_reentry_notes,
    lifecycle_policy_reentry_state,
    lifecycle_policy_staged_exit_state,
    latest_managed_close_time_index,
    merged_reentry_state,
    merged_repair_reentry_notes,
    merged_runtime_overrides,
)
from .runtime_evidence import RuntimeEvidenceCollector
from .models import (
    AutonomyDirective,
    PromotionStage,
    RuntimeEvidenceSnapshot,
    StrategyRuntimeSnapshot,
)
from .planner import AutonomyPlanner


class AutonomousDirector:
    """Build runtime snapshots from experiment results and produce autonomy directives."""

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        planner: AutonomyPlanner | None = None,
        evidence_collector: RuntimeEvidenceCollector | None = None,
    ):
        self.config = config or EvolutionConfig()
        self.planner = planner or AutonomyPlanner(self.config)
        self.evidence_collector = evidence_collector

    def build_runtime_snapshots(
        self,
        results: list[ExperimentResult],
        *,
        stage_overrides: dict[str, PromotionStage] | None = None,
        runtime_overrides: dict[str, dict] | None = None,
        previous_states: list | None = None,
        latest_close_time_by_runtime: dict | None = None,
    ) -> list[StrategyRuntimeSnapshot]:
        snapshots: list[StrategyRuntimeSnapshot] = []
        stage_overrides = stage_overrides or {}
        runtime_overrides = runtime_overrides or {}
        previous_index = {
            item.runtime_id: item
            for item in (previous_states or [])
        }
        for result in results:
            allocation_map = {allocation.strategy_id: allocation for allocation in result.allocations}
            for scorecard in result.scorecards:
                metrics = result.metrics_by_strategy.get(scorecard.genome.strategy_id)
                if metrics is None:
                    continue
                runtime_id = (
                    f"{result.symbol}|{result.timeframe}|"
                    f"{scorecard.genome.strategy_id}"
                )
                stage = (
                    stage_overrides.get(runtime_id)
                    or stage_overrides.get(scorecard.genome.strategy_id)
                )
                effective_scorecard = (
                    replace(scorecard, stage=stage)
                    if stage is not None and stage != scorecard.stage
                    else scorecard
                )
                previous = previous_index.get(runtime_id)
                allocation = allocation_map.get(scorecard.genome.strategy_id)
                runtime = (
                    runtime_overrides.get(runtime_id)
                    or runtime_overrides.get(scorecard.genome.strategy_id, {})
                    or {}
                )
                notes = dict(runtime.get("notes", {}) or {})
                current_repair_reentry_notes = lifecycle_policy_repair_reentry_notes(result.notes)
                repair_reentry_notes = merged_repair_reentry_notes(
                    previous=previous,
                    current_notes=current_repair_reentry_notes,
                )
                evidence = self._runtime_evidence_snapshot(
                    runtime_id=runtime_id,
                    symbol=result.symbol,
                    timeframe=result.timeframe,
                    strategy_id=effective_scorecard.genome.strategy_id,
                    family=effective_scorecard.genome.family,
                    runtime=runtime,
                )
                merged_overrides, override_state = merged_runtime_overrides(
                    config=self.config,
                    previous=previous,
                    repair_reentry_notes=current_repair_reentry_notes,
                    runtime_evidence=evidence,
                )
                reentry_state = merged_reentry_state(
                    config=self.config,
                    previous=previous,
                    repair_reentry_notes=repair_reentry_notes,
                    runtime_overrides=merged_overrides,
                    runtime_override_state=override_state,
                    latest_close_time_by_runtime=latest_close_time_by_runtime,
                    runtime_id=runtime_id,
                    timeframe=result.timeframe,
                )
                staged_exit_state = {}
                if previous is not None:
                    staged_exit_state = lifecycle_policy_staged_exit_state(previous.notes)
                effective_reentry_state = reentry_state
                if not effective_reentry_state and previous is not None:
                    effective_reentry_state = lifecycle_policy_reentry_state(previous.notes)
                notes = compose_runtime_policy_notes(
                    base_notes=notes,
                    repair_reentry_notes=repair_reentry_notes,
                    runtime_overrides=merged_overrides,
                    runtime_override_state=override_state,
                    staged_exit_state=staged_exit_state,
                    reentry_state=effective_reentry_state,
                )
                notes.setdefault("symbol", result.symbol)
                notes.setdefault("timeframe", result.timeframe)
                snapshots.append(
                    StrategyRuntimeSnapshot(
                        scorecard=effective_scorecard,
                        metrics=metrics,
                        runtime_id=(
                            f"{result.symbol}|{result.timeframe}|"
                            f"{effective_scorecard.genome.strategy_id}"
                        ),
                        allocated_capital=float(
                            runtime.get(
                                "allocated_capital",
                                allocation.allocated_capital if allocation else 0.0,
                            )
                        ),
                        current_weight=float(
                            runtime.get(
                                "current_weight",
                                allocation.weight if allocation else 0.0,
                            )
                        ),
                        realized_pnl=float(runtime.get("realized_pnl", 0.0)),
                        unrealized_pnl=float(runtime.get("unrealized_pnl", 0.0)),
                        current_drawdown_pct=float(
                            runtime.get("current_drawdown_pct", 0.0)
                        ),
                        consecutive_losses=int(runtime.get("consecutive_losses", 0)),
                        health_status=str(runtime.get("health_status", "active")),
                        notes=notes,
                    )
                )
        return snapshots

    def plan_from_experiments(
        self,
        results: list[ExperimentResult],
        *,
        stage_overrides: dict[str, PromotionStage] | None = None,
        runtime_overrides: dict[str, dict] | None = None,
        repair_feedback: dict[str, RepairFeedbackSummary] | None = None,
        previous_states: list | None = None,
    ) -> AutonomyDirective:
        runtime_overrides = dict(runtime_overrides or {})
        latest_close_time_by_runtime = {}
        if self.evidence_collector is not None:
            evidence_overrides = self.evidence_collector.collect_for_results(
                results,
                previous_states=previous_states,
            )
            latest_close_time_by_runtime = latest_managed_close_time_index(
                self.evidence_collector.storage,
                managed_source=self.evidence_collector.MANAGED_SOURCE,
            )
            for key, value in evidence_overrides.items():
                merged = dict(runtime_overrides.get(key, {}) or {})
                merged.update({k: v for k, v in value.items() if k != "notes"})
                merged["notes"] = {
                    **dict((runtime_overrides.get(key, {}) or {}).get("notes", {}) or {}),
                    **dict(value.get("notes", {}) or {}),
                }
                runtime_overrides[key] = merged
        snapshots = self.build_runtime_snapshots(
            results,
            stage_overrides=stage_overrides,
            runtime_overrides=runtime_overrides,
            previous_states=previous_states,
            latest_close_time_by_runtime=latest_close_time_by_runtime,
        )
        return self.planner.plan(
            snapshots,
            repair_feedback=repair_feedback,
        )

    @staticmethod
    def _runtime_evidence_snapshot(
        *,
        runtime_id: str,
        symbol: str,
        timeframe: str,
        strategy_id: str,
        family: str,
        runtime: dict,
    ) -> RuntimeEvidenceSnapshot | None:
        if not runtime:
            return None
        notes = dict(runtime.get("notes", {}) or {})
        has_any = any(
            key in runtime
            for key in (
                "current_capital",
                "realized_pnl",
                "unrealized_pnl",
                "current_drawdown_pct",
                "consecutive_losses",
                "health_status",
            )
        ) or bool(notes)
        if not has_any:
            return None
        return RuntimeEvidenceSnapshot(
            runtime_id=runtime_id,
            symbol=symbol,
            timeframe=timeframe,
            strategy_id=strategy_id,
            family=family,
            open_position=bool(runtime.get("current_capital", 0.0)),
            current_capital=float(runtime.get("current_capital", 0.0) or 0.0),
            realized_pnl=float(runtime.get("realized_pnl", 0.0) or 0.0),
            unrealized_pnl=float(runtime.get("unrealized_pnl", 0.0) or 0.0),
            total_net_pnl=float(notes.get("total_net_pnl", 0.0) or 0.0),
            current_drawdown_pct=float(runtime.get("current_drawdown_pct", 0.0) or 0.0),
            max_drawdown_pct=float(notes.get("max_drawdown_pct", 0.0) or 0.0),
            closed_trade_count=int(notes.get("closed_trade_count", 0) or 0),
            win_rate=float(notes.get("win_rate", 0.0) or 0.0),
            consecutive_losses=int(runtime.get("consecutive_losses", 0) or 0),
            health_status=str(runtime.get("health_status", "unproven") or "unproven"),
            notes=notes,
        )
