"""Execute repair plans into revalidated experiment runs."""

from __future__ import annotations

from dataclasses import dataclass

from .experiment_lab import ExperimentLab, ExperimentResult
from .models import RepairPlan
from .promotion_registry import PromotionRegistry


@dataclass(slots=True)
class RepairExecutionResult:
    """One executed repair validation with lineage back to the source runtime."""

    plan: RepairPlan
    source_runtime_id: str
    source_strategy_id: str
    symbol: str
    timeframe: str
    experiment: ExperimentResult
    repair_execution_id: int | None = None


class RepairCycleRunner:
    """Turn repair plans into direct candidate revalidation runs."""

    def __init__(
        self,
        lab: ExperimentLab,
        *,
        registry: PromotionRegistry | None = None,
    ):
        self.lab = lab
        self.registry = registry

    def run(
        self,
        repairs: list[RepairPlan],
        *,
        autonomy_cycle_id: int | None = None,
        total_capital: float = 10000.0,
        candle_limit: int = 1200,
        notes: dict | None = None,
    ) -> list[RepairExecutionResult]:
        executions: list[RepairExecutionResult] = []
        for plan in repairs:
            executed = self.run_plan(
                plan,
                autonomy_cycle_id=autonomy_cycle_id,
                total_capital=total_capital,
                candle_limit=candle_limit,
                notes=notes,
            )
            if executed is not None:
                executions.append(executed)
        return executions

    def run_plan(
        self,
        plan: RepairPlan,
        *,
        autonomy_cycle_id: int | None = None,
        total_capital: float = 10000.0,
        candle_limit: int = 1200,
        notes: dict | None = None,
    ) -> RepairExecutionResult | None:
        candidate = plan.candidate_genome
        if candidate is None:
            return None
        source_runtime_id = str(plan.strategy_id or "")
        symbol, timeframe, source_strategy_id = self.parse_runtime_id(source_runtime_id)
        if not symbol or not timeframe:
            return None

        repair_context = {
            "autonomy_cycle_id": autonomy_cycle_id,
            "source_runtime_id": source_runtime_id,
            "source_strategy_id": source_strategy_id or (candidate.mutation_of or source_runtime_id),
            "source_symbol": symbol,
            "source_timeframe": timeframe,
            "candidate_strategy_id": candidate.strategy_id,
            "candidate_family": candidate.family,
            "candidate_mutation_of": candidate.mutation_of,
            "repair_action": plan.action.value,
            "validation_stage": plan.validation_stage.value,
            "priority": int(plan.priority),
            "capital_multiplier": float(plan.capital_multiplier),
            "runtime_overrides": dict(plan.runtime_overrides or {}),
            "reasons": list(plan.reasons),
        }
        merged_notes = {
            "source": "repair_cycle",
            "repair_validation": repair_context,
        }
        merged_notes.update(notes or {})

        result = self.lab.run_candidates_for_symbol(
            symbol=symbol,
            timeframe=timeframe,
            genomes=[candidate],
            total_capital=total_capital,
            candle_limit=candle_limit,
            notes=merged_notes,
        )
        persisted = (
            self.registry.persist_experiment(result)
            if self.registry is not None
            else result
        )
        repair_execution_id = (
            self.registry.persist_repair_execution(
                plan,
                persisted,
                source_runtime_id=source_runtime_id,
                autonomy_cycle_id=autonomy_cycle_id,
            )
            if self.registry is not None
            else None
        )
        return RepairExecutionResult(
            plan=plan,
            source_runtime_id=source_runtime_id,
            source_strategy_id=repair_context["source_strategy_id"],
            symbol=symbol,
            timeframe=timeframe,
            experiment=persisted,
            repair_execution_id=repair_execution_id,
        )

    @staticmethod
    def parse_runtime_id(runtime_id: str) -> tuple[str, str, str]:
        parts = str(runtime_id).split("|", 2)
        if len(parts) != 3:
            return "", "", str(runtime_id)
        return parts[0], parts[1], parts[2]
