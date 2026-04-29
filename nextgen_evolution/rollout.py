"""Runtime rollout state machine for staged deployment."""

from __future__ import annotations

from .config import EvolutionConfig
from .models import (
    ExecutionAction,
    ExecutionDirective,
    PromotionStage,
    RuntimeLifecycleState,
    RuntimeState,
)


class RolloutStateMachine:
    """Resolve staged deployment state without mutating the scoring layer."""

    def __init__(self, config: EvolutionConfig | None = None):
        self.config = config or EvolutionConfig()

    def resolve(
        self,
        previous: RuntimeState | None,
        directive: ExecutionDirective,
    ) -> tuple[RuntimeLifecycleState, int]:
        previous_state = (
            previous.lifecycle_state
            if previous is not None
            else RuntimeLifecycleState.IDLE
        )
        previous_limited_live_cycles = (
            int(previous.limited_live_cycles)
            if previous is not None
            else 0
        )

        if directive.action in {ExecutionAction.EXIT, ExecutionAction.PAUSE_NEW}:
            return RuntimeLifecycleState.ROLLBACK, 0
        if directive.target_stage == PromotionStage.SHADOW:
            return RuntimeLifecycleState.SHADOW, 0
        if directive.target_stage == PromotionStage.PAPER:
            return RuntimeLifecycleState.PAPER, 0
        if directive.target_stage == PromotionStage.LIVE:
            if previous_state == RuntimeLifecycleState.LIVE:
                return RuntimeLifecycleState.LIVE, previous_limited_live_cycles
            if previous_state == RuntimeLifecycleState.LIMITED_LIVE:
                cycles = previous_limited_live_cycles + 1
                if cycles >= self.config.autonomy_limited_live_cycles:
                    return RuntimeLifecycleState.LIVE, cycles
                return RuntimeLifecycleState.LIMITED_LIVE, cycles
            return RuntimeLifecycleState.LIMITED_LIVE, 1
        if directive.action == ExecutionAction.OBSERVE:
            return RuntimeLifecycleState.IDLE, 0
        return RuntimeLifecycleState.IDLE, 0

    def bounded_capital(
        self,
        *,
        lifecycle_state: RuntimeLifecycleState,
        allocated_capital: float,
        total_capital: float,
        capital_multiplier: float,
    ) -> float:
        base = max(0.0, float(allocated_capital or 0.0))
        desired = max(0.0, base * max(0.0, float(capital_multiplier or 0.0)))
        if lifecycle_state == RuntimeLifecycleState.PAPER:
            return round(
                min(desired, total_capital * self.config.max_paper_weight),
                2,
            )
        if lifecycle_state == RuntimeLifecycleState.LIMITED_LIVE:
            return round(
                min(
                    desired,
                    total_capital * self.config.autonomy_limited_live_max_weight,
                    total_capital * self.config.autonomy_live_blast_radius_capital_pct,
                ),
                2,
            )
        if lifecycle_state == RuntimeLifecycleState.LIVE:
            return round(
                min(
                    desired,
                    total_capital * self.config.max_live_weight,
                    total_capital * self.config.autonomy_live_blast_radius_capital_pct,
                ),
                2,
            )
        return 0.0
