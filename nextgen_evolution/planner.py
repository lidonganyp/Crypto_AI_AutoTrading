"""Autonomy planner that decides deployment, rollback, and repair actions."""

from __future__ import annotations

from .config import EvolutionConfig
from .repair_feedback import RepairFeedbackSummary
from .models import (
    AutonomyDirective,
    ExecutionAction,
    ExecutionDirective,
    PromotionStage,
    StrategyRuntimeSnapshot,
)
from .repair import StrategyRepairEngine
from .runtime_override_policy import (
    lifecycle_policy_reentry_state,
    lifecycle_policy_staged_exit_state,
    lifecycle_policy_runtime_override_state,
    reentry_state_active,
    staged_exit_active,
)


class AutonomyPlanner:
    """Convert runtime evidence into concrete execution and repair directives."""

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        repair_engine: StrategyRepairEngine | None = None,
    ):
        self.config = config or EvolutionConfig()
        self.repair_engine = repair_engine or StrategyRepairEngine(self.config)

    def plan(
        self,
        snapshots: list[StrategyRuntimeSnapshot],
        *,
        repair_feedback: dict[str, RepairFeedbackSummary] | None = None,
    ) -> AutonomyDirective:
        execution: list[ExecutionDirective] = []
        repairs = []
        quarantined: list[str] = []
        retired: list[str] = []
        repair_feedback = repair_feedback or {}

        ordered = sorted(
            snapshots,
            key=lambda item: (
                item.stage == PromotionStage.LIVE,
                item.scorecard.total_score,
                item.metrics.trade_count,
            ),
            reverse=True,
        )
        for snapshot in ordered:
            parent_runtime_id = self._parent_runtime_id(snapshot)
            feedback = (
                repair_feedback.get(snapshot.directive_id)
                or repair_feedback.get(snapshot.strategy_id)
                or repair_feedback.get(parent_runtime_id)
                or repair_feedback.get(snapshot.scorecard.genome.mutation_of or "")
            )
            directive, repair_required = self._directive_for_snapshot(
                snapshot,
                feedback=feedback,
            )
            execution.append(directive)
            if repair_required:
                repairs.append(
                    self.repair_engine.propose(
                        snapshot,
                        feedback=feedback,
                    )
                )
            if directive.action == ExecutionAction.EXIT:
                quarantined.append(snapshot.directive_id)
            if (
                directive.action == ExecutionAction.OBSERVE
                and directive.target_stage == PromotionStage.REJECT
            ) or (
                feedback is not None and feedback.retire_recommended
            ):
                retired.append(snapshot.directive_id)

        requested_repairs = sorted(repairs, key=lambda item: item.priority, reverse=True)
        repairs = requested_repairs[: self.config.autonomy_max_repair_queue]
        dropped_repairs = requested_repairs[len(repairs):]
        quarantined = list(dict.fromkeys(quarantined))
        retired = list(dict.fromkeys(retired))
        return AutonomyDirective(
            execution=execution,
            repairs=repairs,
            quarantined=quarantined,
            retired=retired,
            notes={
                "strategy_count": len(snapshots),
                "live_count": sum(1 for item in snapshots if item.stage == PromotionStage.LIVE),
                "repair_queue_size": len(repairs),
                "repair_queue_requested_size": len(requested_repairs),
                "repair_queue_dropped_count": len(dropped_repairs),
                "repair_queue_dropped_runtime_ids": [
                    item.strategy_id for item in dropped_repairs
                ],
                "quarantine_count": len(quarantined),
                "retire_count": len(retired),
                "feedback_runtime_count": len({id(item) for item in repair_feedback.values()}),
            },
        )

    def _directive_for_snapshot(
        self,
        snapshot: StrategyRuntimeSnapshot,
        *,
        feedback: RepairFeedbackSummary | None = None,
    ) -> tuple[ExecutionDirective, bool]:
        if feedback is not None and feedback.retire_recommended:
            return self._retire_directive(snapshot), True
        severe_reasons = self._severe_degrade_reasons(snapshot)
        mild_reasons = self._mild_degrade_reasons(snapshot)
        severe_degrade = bool(severe_reasons)
        mild_degrade = bool(mild_reasons)
        ready_for_paper = (
            snapshot.scorecard.total_score >= self.config.paper_threshold
            and snapshot.metrics.trade_count >= self.config.autonomy_min_runtime_trades
        )
        ready_for_live = (
            snapshot.scorecard.total_score >= self.config.live_threshold
            and snapshot.metrics.trade_count >= self.config.autonomy_min_runtime_trades
            and snapshot.metrics.live_expectancy >= self.config.autonomy_repair_expectancy_floor
        )
        aggressive_live = (
            snapshot.scorecard.total_score
            >= self.config.live_threshold + self.config.autonomy_scale_up_score_bonus
            and snapshot.realized_pnl >= 0.0
            and snapshot.current_drawdown_pct
            <= self.config.autonomy_repair_drawdown_pct * 0.50
        )
        staged_exit_state = self._staged_exit_state(snapshot)
        staged_exit_phase = str(staged_exit_state.get("phase") or "")
        staged_exit_target = float(staged_exit_state.get("target_multiplier") or 1.0)
        reentry_state = self._reentry_state(snapshot)
        reentry_phase = str(reentry_state.get("phase") or "")
        reentry_active = reentry_state_active(reentry_state)
        runtime_override_recovery_mode = self._runtime_override_recovery_mode(snapshot)
        runtime_override_reasons = self._runtime_override_reasons(snapshot)
        runtime_override_hold_active = runtime_override_recovery_mode == "hold"
        runtime_override_accelerate_active = runtime_override_recovery_mode == "accelerate"
        runtime_override_release_active = self._runtime_override_release_active(
            snapshot,
            reentry_state=reentry_state,
        )
        effective_reentry_active = reentry_active and not runtime_override_release_active

        if snapshot.stage == PromotionStage.LIVE:
            if severe_degrade:
                return (
                    ExecutionDirective(
                        strategy_id=snapshot.directive_id,
                        action=ExecutionAction.EXIT,
                        from_stage=snapshot.stage,
                        target_stage=PromotionStage.SHADOW,
                        capital_multiplier=0.0,
                        reasons=self._merge_reasons(
                            severe_reasons,
                            runtime_override_reasons,
                        ),
                    ),
                    True,
                )
            if mild_degrade:
                capital_multiplier = self.config.autonomy_live_scale_down_factor
                if "profit_lock_harvest" in mild_reasons:
                    capital_multiplier = self._profit_lock_scale_down_multiplier(snapshot)
                if staged_exit_active(staged_exit_state):
                    capital_multiplier = min(capital_multiplier, staged_exit_target)
                return (
                    ExecutionDirective(
                        strategy_id=snapshot.directive_id,
                        action=ExecutionAction.SCALE_DOWN,
                        from_stage=snapshot.stage,
                        target_stage=PromotionStage.LIVE,
                        capital_multiplier=capital_multiplier,
                        reasons=self._merge_reasons(
                            mild_reasons,
                            runtime_override_reasons,
                        ),
                    ),
                    False,
                )
            if staged_exit_active(staged_exit_state):
                if staged_exit_phase in {"harvest", "deep_harvest"}:
                    return (
                        ExecutionDirective(
                            strategy_id=snapshot.directive_id,
                            action=ExecutionAction.SCALE_DOWN,
                            from_stage=snapshot.stage,
                            target_stage=PromotionStage.LIVE,
                            capital_multiplier=staged_exit_target,
                            reasons=self._merge_reasons(
                                [
                                    "profit_lock_harvest",
                                    "profit_lock_staged_exit_active",
                                    f"profit_lock_phase:{staged_exit_phase}",
                                ],
                                runtime_override_reasons,
                            ),
                        ),
                        False,
                    )
                if staged_exit_phase == "reentry":
                    return (
                        ExecutionDirective(
                            strategy_id=snapshot.directive_id,
                            action=ExecutionAction.KEEP,
                            from_stage=snapshot.stage,
                            target_stage=PromotionStage.LIVE,
                            reasons=self._merge_reasons(
                                [
                                    "profit_lock_reentry_active",
                                    "profit_lock_staged_exit_active",
                                    f"profit_lock_phase:{staged_exit_phase}",
                                ],
                                runtime_override_reasons,
                            ),
                        ),
                    False,
                )
            if (
                effective_reentry_active
                and reentry_phase == "recovery"
                and runtime_override_accelerate_active
                and aggressive_live
            ):
                return (
                    ExecutionDirective(
                        strategy_id=snapshot.directive_id,
                        action=ExecutionAction.SCALE_UP,
                        from_stage=snapshot.stage,
                        target_stage=PromotionStage.LIVE,
                        capital_multiplier=self.config.autonomy_live_scale_up_factor,
                        reasons=self._merge_reasons(
                            [
                                "repair_reentry_recovery_accelerated",
                                f"repair_reentry_phase:{reentry_phase}",
                                "runtime_override_accelerate_allows_scale_up",
                            ],
                            runtime_override_reasons,
                        ),
                    ),
                    False,
                )
            if effective_reentry_active:
                reason_key = (
                    "repair_reentry_recovery_active"
                    if reentry_phase == "recovery"
                    else "repair_reentry_probation_active"
                )
                return (
                    ExecutionDirective(
                        strategy_id=snapshot.directive_id,
                        action=ExecutionAction.KEEP,
                        from_stage=snapshot.stage,
                        target_stage=PromotionStage.LIVE,
                        reasons=self._merge_reasons(
                            [
                                reason_key,
                                f"repair_reentry_phase:{reentry_phase or 'probation'}",
                            ],
                            runtime_override_reasons,
                        ),
                        ),
                    False,
                )
            if aggressive_live:
                if runtime_override_hold_active:
                    return (
                        ExecutionDirective(
                            strategy_id=snapshot.directive_id,
                            action=ExecutionAction.KEEP,
                            from_stage=snapshot.stage,
                            target_stage=PromotionStage.LIVE,
                            reasons=self._merge_reasons(
                                [
                                    "runtime_override_hold_blocks_expansion",
                                    "live_strategy_recovery_guarded",
                                ],
                                runtime_override_reasons,
                            ),
                        ),
                        False,
                    )
                return (
                    ExecutionDirective(
                        strategy_id=snapshot.directive_id,
                        action=ExecutionAction.SCALE_UP,
                        from_stage=snapshot.stage,
                        target_stage=PromotionStage.LIVE,
                        capital_multiplier=self.config.autonomy_live_scale_up_factor,
                        reasons=self._merge_reasons(
                            ["live_strategy_outperforming"],
                            runtime_override_reasons,
                        ),
                    ),
                    False,
                )
            return (
                ExecutionDirective(
                    strategy_id=snapshot.directive_id,
                    action=ExecutionAction.KEEP,
                    from_stage=snapshot.stage,
                    target_stage=PromotionStage.LIVE,
                    reasons=self._merge_reasons(
                        ["live_strategy_stable"],
                        runtime_override_reasons,
                    ),
                ),
                False,
            )

        if snapshot.stage == PromotionStage.PAPER:
            if severe_degrade:
                return (
                    ExecutionDirective(
                        strategy_id=snapshot.directive_id,
                        action=ExecutionAction.PAUSE_NEW,
                        from_stage=snapshot.stage,
                        target_stage=PromotionStage.SHADOW,
                        capital_multiplier=0.0,
                        reasons=self._merge_reasons(
                            severe_reasons,
                            runtime_override_reasons,
                        ),
                    ),
                    True,
                )
            if effective_reentry_active:
                if (
                    reentry_phase == "recovery"
                    and ready_for_live
                    and self._repair_reentry_ready_for_live(
                        snapshot,
                        accelerate=runtime_override_accelerate_active,
                    )
                ):
                    if runtime_override_hold_active:
                        return (
                            ExecutionDirective(
                                strategy_id=snapshot.directive_id,
                                action=ExecutionAction.KEEP,
                                from_stage=snapshot.stage,
                                target_stage=PromotionStage.PAPER,
                                reasons=self._merge_reasons(
                                    [
                                        "repair_reentry_recovery_active",
                                        f"repair_reentry_phase:{reentry_phase}",
                                        "runtime_override_hold_blocks_expansion",
                                    ],
                                    runtime_override_reasons,
                                ),
                            ),
                            False,
                        )
                    return (
                        ExecutionDirective(
                            strategy_id=snapshot.directive_id,
                            action=ExecutionAction.PROMOTE_TO_LIVE,
                            from_stage=snapshot.stage,
                            target_stage=PromotionStage.LIVE,
                            reasons=self._merge_reasons(
                                [
                                    (
                                        "repair_reentry_recovery_accelerated"
                                        if runtime_override_accelerate_active
                                        else "repair_reentry_recovery_complete"
                                    ),
                                    f"repair_reentry_phase:{reentry_phase}",
                                ],
                                runtime_override_reasons,
                            ),
                        ),
                        False,
                    )
                return (
                    ExecutionDirective(
                        strategy_id=snapshot.directive_id,
                        action=ExecutionAction.KEEP,
                        from_stage=snapshot.stage,
                        target_stage=PromotionStage.PAPER,
                        reasons=self._merge_reasons(
                            [
                                (
                                    "repair_reentry_recovery_active"
                                    if reentry_phase == "recovery"
                                    else "repair_reentry_probation_active"
                                ),
                                f"repair_reentry_phase:{reentry_phase or 'probation'}",
                            ],
                            runtime_override_reasons,
                        ),
                    ),
                    False,
                )
            if ready_for_live:
                if runtime_override_hold_active:
                    return (
                        ExecutionDirective(
                            strategy_id=snapshot.directive_id,
                            action=ExecutionAction.KEEP,
                            from_stage=snapshot.stage,
                            target_stage=PromotionStage.PAPER,
                            reasons=self._merge_reasons(
                                [
                                    "paper_validation_continues",
                                    "runtime_override_hold_blocks_expansion",
                                ],
                                runtime_override_reasons,
                            ),
                        ),
                        False,
                    )
                return (
                    ExecutionDirective(
                        strategy_id=snapshot.directive_id,
                        action=ExecutionAction.PROMOTE_TO_LIVE,
                        from_stage=snapshot.stage,
                        target_stage=PromotionStage.LIVE,
                        reasons=self._merge_reasons(
                            ["promotion_requirements_met"],
                            runtime_override_reasons,
                        ),
                    ),
                    False,
                )
            return (
                ExecutionDirective(
                    strategy_id=snapshot.directive_id,
                    action=ExecutionAction.KEEP,
                    from_stage=snapshot.stage,
                    target_stage=PromotionStage.PAPER,
                    reasons=self._merge_reasons(
                        ["paper_validation_continues"],
                        runtime_override_reasons,
                    ),
                ),
                False,
            )

        if snapshot.stage == PromotionStage.SHADOW:
            if severe_degrade:
                return (
                    ExecutionDirective(
                        strategy_id=snapshot.directive_id,
                        action=ExecutionAction.PAUSE_NEW,
                        from_stage=snapshot.stage,
                        target_stage=PromotionStage.SHADOW,
                        capital_multiplier=0.0,
                        reasons=self._merge_reasons(
                            severe_reasons,
                            runtime_override_reasons,
                        ),
                    ),
                    True,
                )
            if effective_reentry_active:
                if (
                    reentry_phase == "recovery"
                    and ready_for_paper
                    and self._repair_reentry_ready_for_paper(snapshot)
                ):
                    if runtime_override_hold_active:
                        return (
                            ExecutionDirective(
                                strategy_id=snapshot.directive_id,
                                action=ExecutionAction.OBSERVE,
                                from_stage=snapshot.stage,
                                target_stage=PromotionStage.SHADOW,
                                reasons=self._merge_reasons(
                                    [
                                        "repair_reentry_recovery_active",
                                        f"repair_reentry_phase:{reentry_phase}",
                                        "runtime_override_hold_blocks_expansion",
                                    ],
                                    runtime_override_reasons,
                                ),
                            ),
                            False,
                        )
                    return (
                        ExecutionDirective(
                            strategy_id=snapshot.directive_id,
                            action=ExecutionAction.PROMOTE_TO_PAPER,
                            from_stage=snapshot.stage,
                            target_stage=PromotionStage.PAPER,
                            reasons=self._merge_reasons(
                                [
                                    "repair_reentry_recovery_complete",
                                    f"repair_reentry_phase:{reentry_phase}",
                                ],
                                runtime_override_reasons,
                            ),
                        ),
                        False,
                    )
                return (
                    ExecutionDirective(
                        strategy_id=snapshot.directive_id,
                        action=ExecutionAction.OBSERVE,
                        from_stage=snapshot.stage,
                        target_stage=PromotionStage.SHADOW,
                        reasons=self._merge_reasons(
                            [
                                (
                                    "repair_reentry_recovery_active"
                                    if reentry_phase == "recovery"
                                    else "repair_reentry_probation_active"
                                ),
                                f"repair_reentry_phase:{reentry_phase or 'probation'}",
                            ],
                            runtime_override_reasons,
                        ),
                    ),
                    False,
                )
            if ready_for_paper:
                if runtime_override_hold_active:
                    return (
                        ExecutionDirective(
                            strategy_id=snapshot.directive_id,
                            action=ExecutionAction.OBSERVE,
                            from_stage=snapshot.stage,
                            target_stage=PromotionStage.SHADOW,
                            reasons=self._merge_reasons(
                                [
                                    "collect_more_shadow_evidence",
                                    "runtime_override_hold_blocks_expansion",
                                ],
                                runtime_override_reasons,
                            ),
                        ),
                        False,
                    )
                return (
                    ExecutionDirective(
                        strategy_id=snapshot.directive_id,
                        action=ExecutionAction.PROMOTE_TO_PAPER,
                        from_stage=snapshot.stage,
                        target_stage=PromotionStage.PAPER,
                        reasons=self._merge_reasons(
                            ["shadow_requirements_met"],
                            runtime_override_reasons,
                        ),
                    ),
                    False,
                )
            return (
                ExecutionDirective(
                    strategy_id=snapshot.directive_id,
                    action=ExecutionAction.OBSERVE,
                    from_stage=snapshot.stage,
                    target_stage=PromotionStage.SHADOW,
                    reasons=self._merge_reasons(
                        ["collect_more_shadow_evidence"],
                        runtime_override_reasons,
                    ),
                ),
                False,
            )

        if snapshot.scorecard.total_score >= self.config.shadow_threshold:
            return (
                ExecutionDirective(
                    strategy_id=snapshot.directive_id,
                    action=ExecutionAction.PROMOTE_TO_SHADOW,
                    from_stage=snapshot.stage,
                    target_stage=PromotionStage.SHADOW,
                    reasons=self._merge_reasons(
                        ["candidate_enters_shadow"],
                        runtime_override_reasons,
                    ),
                ),
                False,
            )
        return (
            ExecutionDirective(
                strategy_id=snapshot.directive_id,
                action=ExecutionAction.OBSERVE,
                from_stage=snapshot.stage,
                target_stage=PromotionStage.REJECT,
                reasons=self._merge_reasons(
                    ["candidate_not_ready"],
                    runtime_override_reasons,
                ),
            ),
            False,
        )

    @staticmethod
    def _retire_directive(snapshot: StrategyRuntimeSnapshot) -> ExecutionDirective:
        if snapshot.stage == PromotionStage.LIVE:
            return ExecutionDirective(
                strategy_id=snapshot.directive_id,
                action=ExecutionAction.EXIT,
                from_stage=snapshot.stage,
                target_stage=PromotionStage.SHADOW,
                capital_multiplier=0.0,
                reasons=["repair_lineage_exhausted"],
            )
        return ExecutionDirective(
            strategy_id=snapshot.directive_id,
            action=ExecutionAction.OBSERVE,
            from_stage=snapshot.stage,
            target_stage=PromotionStage.REJECT,
            capital_multiplier=0.0,
            reasons=["repair_lineage_exhausted"],
        )

    def _severe_degrade_reasons(self, snapshot: StrategyRuntimeSnapshot) -> list[str]:
        reasons: list[str] = []
        if snapshot.current_drawdown_pct >= self.config.autonomy_repair_drawdown_pct:
            reasons.append(self._stage_reason(snapshot, "drawdown_limit_breached"))
        if snapshot.metrics.live_expectancy < self.config.autonomy_repair_expectancy_floor:
            reasons.append(self._stage_reason(snapshot, "expectancy_floor_breached"))
        if self._severe_overstay(snapshot):
            reasons.append(self._stage_reason(snapshot, "overstay_hard_limit"))
        if self._severe_profit_lock(snapshot):
            reasons.append("profit_lock_exit")
        if snapshot.health_status in {"degraded", "failing"}:
            reasons.append(self._stage_reason(snapshot, f"health_{snapshot.health_status}"))
        return reasons

    def _mild_degrade_reasons(self, snapshot: StrategyRuntimeSnapshot) -> list[str]:
        reasons: list[str] = []
        if (
            snapshot.current_drawdown_pct
            >= self.config.autonomy_repair_drawdown_pct * 0.60
        ):
            reasons.append(self._stage_reason(snapshot, "drawdown_warning"))
        if (
            snapshot.metrics.live_expectancy
            < self.config.autonomy_repair_expectancy_floor + 0.03
        ):
            reasons.append(self._stage_reason(snapshot, "expectancy_soft_warning"))
        if self._mild_overstay(snapshot):
            reasons.append(self._stage_reason(snapshot, "overstay_soft_limit"))
        if self._mild_profit_lock(snapshot):
            reasons.append("profit_lock_harvest")
        if snapshot.consecutive_losses >= 2:
            reasons.append(self._stage_reason(snapshot, "loss_streak_warning"))
        return reasons

    @staticmethod
    def _stage_reason(snapshot: StrategyRuntimeSnapshot, suffix: str) -> str:
        if snapshot.stage == PromotionStage.LIVE:
            prefix = "live_strategy"
        elif snapshot.stage == PromotionStage.PAPER:
            prefix = "paper_strategy"
        elif snapshot.stage == PromotionStage.SHADOW:
            prefix = "shadow_candidate"
        else:
            prefix = "candidate"
        return f"{prefix}_{suffix}"

    @staticmethod
    def _parent_runtime_id(snapshot: StrategyRuntimeSnapshot) -> str:
        mutation_of = snapshot.scorecard.genome.mutation_of
        if not mutation_of:
            return ""
        symbol = str(snapshot.notes.get("symbol") or "").strip()
        timeframe = str(snapshot.notes.get("timeframe") or "").strip()
        if not symbol or not timeframe:
            return ""
        return f"{symbol}|{timeframe}|{mutation_of}"

    @staticmethod
    def _merge_reasons(*groups: list[str]) -> list[str]:
        merged: list[str] = []
        for group in groups:
            for reason in list(group or []):
                text = str(reason).strip()
                if text and text not in merged:
                    merged.append(text)
        return merged

    @staticmethod
    def _runtime_override_recovery_mode(snapshot: StrategyRuntimeSnapshot) -> str:
        state = lifecycle_policy_runtime_override_state(snapshot.notes)
        return str(state.get("recovery_mode") or "").strip().lower()

    def _runtime_override_reasons(self, snapshot: StrategyRuntimeSnapshot) -> list[str]:
        mode = self._runtime_override_recovery_mode(snapshot)
        if not mode:
            return []
        return [f"runtime_override_recovery_mode:{mode}"]

    def _runtime_override_release_active(
        self,
        snapshot: StrategyRuntimeSnapshot,
        *,
        reentry_state: dict | None = None,
    ) -> bool:
        if self._runtime_override_recovery_mode(snapshot) == "release":
            return True
        reentry = dict(reentry_state or self._reentry_state(snapshot))
        return bool(reentry.get("release_ready"))

    def _mild_overstay(self, snapshot: StrategyRuntimeSnapshot) -> bool:
        threshold = self._overstay_threshold_minutes(
            snapshot,
            multiplier=self.config.autonomy_overstay_soft_multiplier,
        )
        if threshold <= 0:
            return False
        holding_minutes = float(snapshot.notes.get("holding_minutes") or 0.0)
        if holding_minutes < threshold:
            return False
        return snapshot.unrealized_pnl <= 0.0 or snapshot.realized_pnl < 0.0

    def _severe_overstay(self, snapshot: StrategyRuntimeSnapshot) -> bool:
        threshold = self._overstay_threshold_minutes(
            snapshot,
            multiplier=self.config.autonomy_overstay_hard_multiplier,
        )
        if threshold <= 0:
            return False
        holding_minutes = float(snapshot.notes.get("holding_minutes") or 0.0)
        if holding_minutes < threshold:
            return False
        return snapshot.unrealized_pnl < 0.0 or snapshot.realized_pnl < 0.0

    def _mild_profit_lock(self, snapshot: StrategyRuntimeSnapshot) -> bool:
        context = self._profit_lock_context(snapshot)
        if context is None:
            return False
        return (
            context["peak_return_pct"] >= self.config.autonomy_profit_lock_min_return_pct
            and context["retrace_pct"] >= self.config.autonomy_profit_lock_soft_retrace_pct
            and context["current_profit"] > 0.0
        )

    def _severe_profit_lock(self, snapshot: StrategyRuntimeSnapshot) -> bool:
        context = self._profit_lock_context(snapshot)
        if context is None:
            return False
        return (
            context["peak_return_pct"] >= self.config.autonomy_profit_lock_min_return_pct
            and context["retrace_pct"] >= self.config.autonomy_profit_lock_hard_retrace_pct
            and context["current_profit"] > 0.0
        )

    @staticmethod
    def _profit_lock_context(snapshot: StrategyRuntimeSnapshot) -> dict[str, float] | None:
        peak_return_pct = float(snapshot.notes.get("peak_unrealized_return_pct") or 0.0)
        retrace_pct = float(snapshot.notes.get("profit_retrace_pct") or 0.0)
        current_profit = float(snapshot.unrealized_pnl or 0.0)
        if peak_return_pct <= 0 or retrace_pct <= 0:
            return None
        return {
            "peak_return_pct": peak_return_pct,
            "retrace_pct": retrace_pct,
            "current_profit": current_profit,
        }

    def _profit_lock_scale_down_multiplier(
        self,
        snapshot: StrategyRuntimeSnapshot,
    ) -> float:
        context = self._profit_lock_context(snapshot)
        if context is None:
            return self.config.autonomy_profit_lock_soft_scale_down_factor
        retrace_pct = float(context["retrace_pct"])
        midpoint = (
            self.config.autonomy_profit_lock_soft_retrace_pct
            + self.config.autonomy_profit_lock_hard_retrace_pct
        ) / 2.0
        if retrace_pct >= midpoint:
            return min(
                self.config.autonomy_profit_lock_soft_scale_down_factor,
                self.config.autonomy_profit_lock_deep_scale_down_factor,
            )
        return min(1.0, self.config.autonomy_profit_lock_soft_scale_down_factor)

    @staticmethod
    def _overstay_threshold_minutes(
        snapshot: StrategyRuntimeSnapshot,
        *,
        multiplier: float,
    ) -> float:
        try:
            hold_bars = float(snapshot.scorecard.genome.params.get("hold_bars", 0.0) or 0.0)
        except Exception:
            hold_bars = 0.0
        timeframe_minutes = float(snapshot.notes.get("timeframe_minutes") or 0.0)
        if hold_bars <= 0 or timeframe_minutes <= 0:
            return 0.0
        return hold_bars * timeframe_minutes * max(0.0, float(multiplier))

    @staticmethod
    def _staged_exit_state(snapshot: StrategyRuntimeSnapshot) -> dict:
        return lifecycle_policy_staged_exit_state(snapshot.notes)

    @staticmethod
    def _reentry_state(snapshot: StrategyRuntimeSnapshot) -> dict:
        return lifecycle_policy_reentry_state(snapshot.notes)

    def _repair_reentry_ready_for_live(
        self,
        snapshot: StrategyRuntimeSnapshot,
        *,
        accelerate: bool = False,
    ) -> bool:
        closed_trade_count = int(snapshot.notes.get("closed_trade_count") or 0)
        win_rate = float(snapshot.notes.get("win_rate") or 0.0)
        total_net_pnl = float(snapshot.notes.get("total_net_pnl") or 0.0)
        required_successes = max(1, int(self.config.autonomy_repair_promote_after_successes))
        if accelerate:
            required_successes = max(1, required_successes - 1)
        return (
            closed_trade_count >= required_successes
            and win_rate >= 0.50
            and total_net_pnl >= 0.0
            and snapshot.current_drawdown_pct
            <= self.config.autonomy_repair_drawdown_pct * 0.50
            and snapshot.consecutive_losses == 0
            and snapshot.health_status in {"active", "unproven"}
        )

    def _repair_reentry_ready_for_paper(
        self,
        snapshot: StrategyRuntimeSnapshot,
    ) -> bool:
        total_net_pnl = float(snapshot.notes.get("total_net_pnl") or 0.0)
        return (
            total_net_pnl >= 0.0
            and snapshot.current_drawdown_pct
            <= self.config.autonomy_repair_drawdown_pct * 0.60
            and snapshot.consecutive_losses == 0
            and snapshot.health_status in {"active", "unproven"}
        )
