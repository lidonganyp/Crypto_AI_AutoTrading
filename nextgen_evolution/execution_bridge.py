"""Bridge autonomy directives into paper execution intents."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone

from core.models import SignalDirection
from execution.paper_trader import PaperTrader

from .config import EvolutionConfig
from .experiment_lab import ExperimentResult
from .models import (
    AutonomyDirective,
    ExecutionAction,
    ExecutionIntent,
    ExecutionIntentAction,
    PortfolioAllocation,
    PromotionStage,
    RuntimeEvidenceSnapshot,
    RuntimeLifecycleState,
    RuntimeState,
)
from .portfolio_allocator import PortfolioAllocator
from .promotion_registry import PromotionRegistry
from .rollout import RolloutStateMachine
from .runtime_override_policy import (
    apply_staged_exit_to_capital,
    compose_runtime_policy_notes,
    lifecycle_policy_repair_reentry_notes,
    lifecycle_policy_runtime_override_state,
    decay_runtime_overrides,
    lifecycle_policy_runtime_overrides,
    lifecycle_policy_reentry_state,
    lifecycle_policy_staged_exit_state,
    latest_managed_close_time_index,
    merged_repair_reentry_notes,
    merged_reentry_state,
    merged_staged_exit_state,
    merged_runtime_overrides,
    repair_runtime_overrides_from_notes,
    reentry_state_active,
    staged_exit_active,
    runtime_override_performance_snapshot,
    runtime_override_recovery_policy,
)
from .runtime_evidence import RuntimeEvidenceCollector


class AutonomyPaperBridge:
    """Translate autonomy output into safe symbol-level paper execution."""

    MANAGED_SOURCE = "nextgen_autonomy"
    BRIDGE_MODE = "paper"

    def __init__(
        self,
        feed,
        *,
        registry: PromotionRegistry | None = None,
        paper_trader: PaperTrader | None = None,
        config: EvolutionConfig | None = None,
        rollout: RolloutStateMachine | None = None,
    ):
        self.feed = feed
        self.storage = feed.storage
        self.config = config or EvolutionConfig()
        self.registry = registry
        self.paper_trader = paper_trader or PaperTrader(self.storage)
        self.rollout = rollout or RolloutStateMachine(self.config)

    @staticmethod
    def runtime_id(symbol: str, timeframe: str, strategy_id: str) -> str:
        return f"{symbol}|{timeframe}|{strategy_id}"

    @staticmethod
    def parse_runtime_id(runtime_id: str) -> tuple[str, str, str]:
        parts = str(runtime_id).split("|", 2)
        if len(parts) != 3:
            return "", "", str(runtime_id)
        return parts[0], parts[1], parts[2]

    def build_runtime_states(
        self,
        *,
        results: list[ExperimentResult],
        directive: AutonomyDirective,
        portfolio_allocations: list[PortfolioAllocation],
        total_capital: float,
        previous_states: list[RuntimeState] | None = None,
    ) -> list[RuntimeState]:
        previous_index = {
            item.runtime_id: item
            for item in (previous_states or [])
        }
        allocation_index = {
            self.runtime_id(item.symbol, item.timeframe or "5m", item.strategy_id): item
            for item in portfolio_allocations
        }
        score_index: dict[str, tuple] = {}
        for result in results:
            for card in result.scorecards:
                score_index[
                    self.runtime_id(result.symbol, result.timeframe, card.genome.strategy_id)
                ] = (result, card)
        runtime_evidence_index = self._runtime_evidence_index(
            runtime_ids=[item.strategy_id for item in directive.execution],
            previous_states=previous_states,
        )
        latest_close_time_by_runtime = latest_managed_close_time_index(
            self.storage,
            managed_source=self.MANAGED_SOURCE,
        )

        states: list[RuntimeState] = []
        for item in directive.execution:
            symbol, timeframe, strategy_id = self.parse_runtime_id(item.strategy_id)
            result_card = score_index.get(item.strategy_id)
            if not result_card:
                continue
            result, card = result_card
            previous = previous_index.get(item.strategy_id)
            lifecycle_state, limited_live_cycles = self.rollout.resolve(previous, item)
            allocation = allocation_index.get(item.strategy_id)
            current_repair_reentry_notes = lifecycle_policy_repair_reentry_notes(result.notes)
            repair_reentry_notes = self._merged_repair_reentry_notes(
                result=result,
                previous=previous,
            )
            repair_runtime_overrides, override_state = self._merged_runtime_overrides(
                previous=previous,
                repair_reentry_notes=current_repair_reentry_notes,
                runtime_evidence=runtime_evidence_index.get(item.strategy_id),
            )
            reentry_state = self._merged_reentry_state(
                previous=previous,
                runtime_id=item.strategy_id,
                timeframe=timeframe or result.timeframe,
                repair_reentry_notes=repair_reentry_notes,
                runtime_overrides=repair_runtime_overrides,
                runtime_override_state=override_state,
                latest_close_time_by_runtime=latest_close_time_by_runtime,
            )
            staged_exit_state = self._merged_staged_exit_state(
                previous=previous,
                directive=item,
                runtime_evidence=runtime_evidence_index.get(item.strategy_id),
            )
            allocated_capital = (
                float(allocation.allocated_capital)
                if allocation is not None
                else 0.0
            )
            current_capital = (
                float(previous.current_capital)
                if previous is not None
                else 0.0
            )
            desired_capital = self.rollout.bounded_capital(
                lifecycle_state=lifecycle_state,
                allocated_capital=allocated_capital,
                total_capital=float(total_capital),
                capital_multiplier=float(item.capital_multiplier),
            )
            desired_capital = self._apply_runtime_overrides_to_capital(
                desired_capital=desired_capital,
                runtime_overrides=repair_runtime_overrides,
                apply_weight_multiplier=not self._portfolio_consumed_weight_cap(allocation),
            )
            desired_capital = self._apply_staged_exit_to_capital(
                desired_capital=desired_capital,
                allocated_capital=allocated_capital,
                staged_exit_state=staged_exit_state,
                apply_staged_exit=not self._portfolio_consumed_staged_exit_cap(allocation),
            )
            states.append(
                RuntimeState(
                    runtime_id=item.strategy_id,
                    symbol=symbol or result.symbol,
                    timeframe=timeframe or result.timeframe,
                    strategy_id=strategy_id,
                    family=card.genome.family,
                    lifecycle_state=lifecycle_state,
                    promotion_stage=card.stage,
                    target_stage=item.target_stage,
                    last_directive_action=item.action,
                    score=float(card.total_score),
                    allocated_capital=allocated_capital,
                    desired_capital=desired_capital,
                    current_capital=current_capital,
                    current_weight=(
                        float(allocation.weight)
                        if allocation is not None
                        else float(previous.current_weight if previous is not None else 0.0)
                    ),
                    capital_multiplier=float(item.capital_multiplier),
                    limited_live_cycles=limited_live_cycles,
                    notes=compose_runtime_policy_notes(
                        base_notes={
                            "reasons": list(item.reasons),
                            "result_symbol": result.symbol,
                            "result_timeframe": result.timeframe,
                            "allocation_reasons": list(allocation.reasons) if allocation is not None else [],
                        },
                        repair_reentry_notes=repair_reentry_notes,
                        runtime_overrides=repair_runtime_overrides,
                        runtime_override_state=override_state,
                        staged_exit_state=staged_exit_state,
                        reentry_state=reentry_state,
                    ),
                )
            )
        return states

    def apply(
        self,
        *,
        results: list[ExperimentResult],
        directive: AutonomyDirective,
        portfolio_allocations: list[PortfolioAllocation],
        total_capital: float,
        autonomy_cycle_id: int | None = None,
        previous_states: list[RuntimeState] | None = None,
    ) -> tuple[list[RuntimeState], list[ExecutionIntent]]:
        if previous_states is None and self.registry is not None:
            previous_states = self.registry.load_runtime_states(hydrate_legacy=False)
        runtime_states = self.build_runtime_states(
            results=results,
            directive=directive,
            portfolio_allocations=portfolio_allocations,
            total_capital=total_capital,
            previous_states=previous_states,
        )
        if self.registry is not None:
            self.registry.persist_runtime_states(runtime_states)
        intents = self.build_execution_intents(runtime_states)
        executed = self.execute_paper(
            intents,
            autonomy_cycle_id=autonomy_cycle_id,
        )
        return runtime_states, executed

    def build_execution_intents(
        self,
        runtime_states: list[RuntimeState],
    ) -> list[ExecutionIntent]:
        latest_prices = {
            (item.symbol, item.timeframe): self._latest_price(item.symbol, item.timeframe)
            for item in runtime_states
        }
        state_by_symbol: dict[str, list[RuntimeState]] = defaultdict(list)
        for item in runtime_states:
            state_by_symbol[item.symbol].append(item)

        positions_by_symbol = {
            str(item["symbol"]): item
            for item in self.storage.get_positions()
        }
        open_trade_by_symbol = {
            str(item["symbol"]): item
            for item in self.storage.get_open_trades()
        }

        intents: list[ExecutionIntent] = []
        all_symbols = sorted(
            set(state_by_symbol) | set(positions_by_symbol) | set(open_trade_by_symbol)
        )
        for symbol in all_symbols:
            primary = self._primary_state(state_by_symbol.get(symbol, []))
            current_position = positions_by_symbol.get(symbol)
            current_trade = open_trade_by_symbol.get(symbol)
            current_metadata = self._trade_metadata(current_trade)
            managed_trade = current_metadata.get("source") == self.MANAGED_SOURCE
            current_runtime_id = str(current_metadata.get("runtime_id") or "")
            if primary is None:
                if managed_trade and current_position and current_trade:
                    price = latest_prices.get((symbol, str(current_metadata.get("timeframe") or "5m"))) or float(
                        current_position.get("entry_price") or 0.0
                    )
                    intents.append(
                        self._intent_from_state(
                            state=self._state_from_trade(current_trade, current_metadata),
                            action=ExecutionIntentAction.CLOSE,
                            current_capital=float(current_position["quantity"]) * price,
                            price=price,
                            reasons=["runtime_not_selected"],
                            notes={"mode": "managed_close"},
                        )
                    )
                continue

            price = latest_prices.get((primary.symbol, primary.timeframe)) or 0.0
            if current_position and current_trade and not managed_trade:
                intents.append(
                    self._intent_from_state(
                        state=primary,
                        action=ExecutionIntentAction.SKIP,
                        price=price,
                        reasons=["symbol_managed_elsewhere"],
                    )
                )
                continue

            if primary.lifecycle_state in {
                RuntimeLifecycleState.IDLE,
                RuntimeLifecycleState.SHADOW,
                RuntimeLifecycleState.ROLLBACK,
            }:
                if managed_trade and current_position and current_trade:
                    intents.append(
                        self._intent_from_state(
                            state=primary,
                            action=ExecutionIntentAction.CLOSE,
                            current_capital=float(current_position["quantity"]) * price,
                            price=price,
                            reasons=["rollout_requires_exit"],
                        )
                    )
                else:
                    intents.append(
                        self._intent_from_state(
                            state=primary,
                            action=ExecutionIntentAction.HOLD,
                            price=price,
                            reasons=["non_deployable_state"],
                        )
                    )
                continue

            if current_position and current_trade and managed_trade:
                current_capital = float(current_position["quantity"]) * max(price, 0.0)
                if current_runtime_id and current_runtime_id != primary.runtime_id:
                    intents.append(
                        self._intent_from_state(
                            state=self._state_from_trade(current_trade, current_metadata),
                            action=ExecutionIntentAction.CLOSE,
                            current_capital=current_capital,
                            price=price,
                            reasons=["runtime_rotation_out"],
                        )
                    )
                    intents.append(
                        self._intent_from_state(
                            state=primary,
                            action=ExecutionIntentAction.OPEN,
                            desired_capital=primary.desired_capital,
                            price=price,
                            reasons=["runtime_rotation_in"],
                        )
                    )
                    continue
                scale_down_reasons = self._scale_down_reasons(primary)
                if (
                    primary.desired_capital > 0
                    and current_capital < primary.desired_capital * 0.95
                ):
                    intents.append(
                        self._intent_from_state(
                            state=primary,
                            action=ExecutionIntentAction.OPEN,
                            desired_capital=primary.desired_capital,
                            execution_capital=max(0.0, primary.desired_capital - current_capital),
                            current_capital=current_capital,
                            price=price,
                            reasons=["rebalance_up"],
                            notes={
                                "position_adjustment": "rebalance_up",
                                "allow_position_add": True,
                            },
                        )
                    )
                    continue
                if (
                    bool(scale_down_reasons)
                    or (
                        primary.desired_capital > 0
                        and current_capital > primary.desired_capital * 1.05
                    )
                ):
                    ratio = (
                        max(0.0, current_capital - primary.desired_capital)
                        / max(current_capital, 1e-9)
                    )
                    close_quantity = float(current_position["quantity"]) * ratio
                    intents.append(
                        self._intent_from_state(
                            state=primary,
                            action=ExecutionIntentAction.REDUCE,
                            desired_capital=primary.desired_capital,
                            current_capital=current_capital,
                            price=price,
                            close_quantity=close_quantity,
                            reasons=scale_down_reasons or ["rebalance_down"],
                            notes=self._scale_down_notes(primary),
                        )
                    )
                    continue
                intents.append(
                    self._intent_from_state(
                        state=primary,
                        action=ExecutionIntentAction.HOLD,
                        desired_capital=primary.desired_capital,
                        current_capital=current_capital,
                        price=price,
                        reasons=["position_aligned"],
                    )
                )
                continue

            cooldown_reasons = self._reentry_cooldown_reasons(primary)
            if cooldown_reasons:
                intents.append(
                    self._intent_from_state(
                        state=primary,
                        action=ExecutionIntentAction.SKIP,
                        desired_capital=primary.desired_capital,
                        price=price,
                        reasons=cooldown_reasons,
                    )
                )
                continue
            if primary.desired_capital <= 0 or price <= 0:
                intents.append(
                    self._intent_from_state(
                        state=primary,
                        action=ExecutionIntentAction.SKIP,
                        desired_capital=primary.desired_capital,
                        price=price,
                        reasons=["missing_price_or_capital"],
                    )
                )
                continue
            intents.append(
                self._intent_from_state(
                    state=primary,
                    action=ExecutionIntentAction.OPEN,
                    desired_capital=primary.desired_capital,
                    price=price,
                    reasons=["deploy_primary_runtime"],
                )
            )

        action_rank = {
            ExecutionIntentAction.CLOSE: 0,
            ExecutionIntentAction.REDUCE: 1,
            ExecutionIntentAction.OPEN: 2,
            ExecutionIntentAction.HOLD: 3,
            ExecutionIntentAction.SKIP: 4,
            ExecutionIntentAction.ROTATE: 5,
        }
        intents.sort(
            key=lambda item: (
                action_rank.get(item.action, 99),
                item.symbol,
                item.runtime_id,
            )
        )
        return intents

    def execute_paper(
        self,
        intents: list[ExecutionIntent],
        *,
        autonomy_cycle_id: int | None = None,
    ) -> list[ExecutionIntent]:
        executed: list[ExecutionIntent] = []
        for item in intents:
            updated = item
            if item.action == ExecutionIntentAction.OPEN:
                updated = self._execute_open(item)
            elif item.action == ExecutionIntentAction.CLOSE:
                updated = self._execute_close(item)
            elif item.action == ExecutionIntentAction.REDUCE:
                updated = self._execute_reduce(item)
            elif item.action == ExecutionIntentAction.HOLD:
                updated = replace(item, status="no_action")
            else:
                updated = replace(item, status="skipped")
            self.storage.insert_execution_event(
                "nextgen_autonomy_intent",
                updated.symbol,
                {
                    "autonomy_cycle_id": autonomy_cycle_id,
                    "runtime_id": updated.runtime_id,
                    "strategy_id": updated.strategy_id,
                    "action": updated.action.value,
                    "lifecycle_state": updated.lifecycle_state.value,
                    "status": updated.status,
                    "desired_capital": updated.desired_capital,
                    "current_capital": updated.current_capital,
                    "quantity": updated.quantity,
                    "close_quantity": updated.close_quantity,
                    "price": updated.price,
                    "reasons": updated.reasons,
                    "notes": updated.notes,
                },
            )
            executed.append(updated)
        if self.registry is not None:
            self.registry.persist_execution_intents(
                executed,
                autonomy_cycle_id=autonomy_cycle_id,
            )
        return executed

    def _execute_open(self, item: ExecutionIntent) -> ExecutionIntent:
        position_value = float(item.notes.get("execution_capital") or item.desired_capital)
        result = self.paper_trader.execute_open(
            item.symbol,
            SignalDirection.LONG,
            item.price,
            confidence=min(0.99, max(0.0, 0.50 + max(0.0, item.notes.get("score", 0.0)))),
            rationale=";".join(item.reasons) or "nextgen_autonomy_open",
            position_value=position_value,
            take_profit_pct=self._take_profit_pct(item),
            metadata={
                "source": self.MANAGED_SOURCE,
                "bridge_mode": self.BRIDGE_MODE,
                "runtime_id": item.runtime_id,
                "strategy_id": item.strategy_id,
                "family": item.family,
                "timeframe": item.timeframe,
                "lifecycle_state": item.lifecycle_state.value,
                "execution_action": item.action.value,
                "allow_position_add": bool(item.notes.get("allow_position_add")),
                "position_adjustment": str(item.notes.get("position_adjustment") or ""),
                "runtime_lifecycle_policy": dict(item.notes.get("runtime_lifecycle_policy") or {}),
                "target_desired_capital": float(item.desired_capital),
                "execution_capital": position_value,
            },
        )
        if not result:
            return replace(item, status="rejected")
        return replace(
            item,
            status="executed",
            quantity=float(result.get("quantity") or 0.0),
            notes={
                **item.notes,
                "order_id": result.get("order_id"),
                "trade_id": result.get("trade_id"),
            },
        )

    def _execute_close(self, item: ExecutionIntent) -> ExecutionIntent:
        result = self.paper_trader.execute_close(
            item.symbol,
            item.price,
            reason="nextgen_autonomy_close",
        )
        if not result:
            return replace(item, status="skipped")
        return replace(
            item,
            status="executed",
            close_quantity=float(result.get("closed_qty") or 0.0),
            notes={
                **item.notes,
                "order_id": result.get("order_id"),
                "trade_id": result.get("trade_id"),
            },
        )

    def _execute_reduce(self, item: ExecutionIntent) -> ExecutionIntent:
        if item.close_quantity <= 1e-8:
            return replace(item, status="skipped")
        close_reason = (
            "nextgen_autonomy_profit_lock_reduce"
            if str(item.notes.get("position_adjustment") or "") == "profit_lock_harvest"
            else "nextgen_autonomy_reduce"
        )
        result = self.paper_trader.execute_close(
            item.symbol,
            item.price,
            reason=close_reason,
            close_qty=item.close_quantity,
        )
        if not result:
            return replace(item, status="skipped")
        return replace(
            item,
            status="executed",
            close_quantity=float(result.get("closed_qty") or 0.0),
            notes={
                **item.notes,
                "order_id": result.get("order_id"),
                "trade_id": result.get("trade_id"),
            },
        )

    def _primary_state(self, states: list[RuntimeState]) -> RuntimeState | None:
        deployable = [
            item
            for item in states
            if item.lifecycle_state
            in {
                RuntimeLifecycleState.PAPER,
                RuntimeLifecycleState.LIMITED_LIVE,
                RuntimeLifecycleState.LIVE,
            }
            and item.desired_capital > 0
        ]
        if not deployable:
            return max(states, key=lambda item: item.score, default=None)
        lifecycle_rank = {
            RuntimeLifecycleState.LIVE: 3,
            RuntimeLifecycleState.LIMITED_LIVE: 2,
            RuntimeLifecycleState.PAPER: 1,
        }
        return max(
            deployable,
            key=lambda item: (
                lifecycle_rank.get(item.lifecycle_state, 0),
                item.desired_capital,
                item.score,
            ),
        )

    @staticmethod
    def _trade_metadata(trade: dict | None) -> dict:
        if not trade:
            return {}
        raw = trade.get("metadata_json")
        if isinstance(raw, dict):
            return dict(raw)
        import json

        try:
            payload = json.loads(raw or "{}")
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def _state_from_trade(self, trade: dict, metadata: dict) -> RuntimeState:
        runtime_id = str(metadata.get("runtime_id") or "")
        symbol, timeframe, strategy_id = self.parse_runtime_id(runtime_id)
        return RuntimeState(
            runtime_id=runtime_id,
            symbol=symbol or str(trade.get("symbol") or ""),
            timeframe=timeframe or str(metadata.get("timeframe") or "5m"),
            strategy_id=strategy_id or str(metadata.get("strategy_id") or ""),
            family=str(metadata.get("family") or ""),
            lifecycle_state=RuntimeLifecycleState(
                str(metadata.get("lifecycle_state") or RuntimeLifecycleState.PAPER.value)
            ),
            promotion_stage=PromotionStage.PAPER,
            target_stage=PromotionStage.PAPER,
            last_directive_action=self._default_execution_action(metadata),
            score=float(metadata.get("score") or 0.0),
            desired_capital=float(trade.get("entry_price") or 0.0)
            * float(trade.get("quantity") or 0.0),
            current_capital=float(trade.get("entry_price") or 0.0)
            * float(trade.get("quantity") or 0.0),
            notes=compose_runtime_policy_notes(
                base_notes={},
                repair_reentry_notes=lifecycle_policy_repair_reentry_notes(metadata),
                runtime_overrides=lifecycle_policy_runtime_overrides(metadata),
                runtime_override_state=lifecycle_policy_runtime_override_state(metadata),
                staged_exit_state=lifecycle_policy_staged_exit_state(metadata),
                reentry_state=lifecycle_policy_reentry_state(metadata),
            ),
        )

    @staticmethod
    def _default_execution_action(metadata: dict):
        from .models import ExecutionAction

        raw = str(metadata.get("execution_action") or ExecutionAction.KEEP.value)
        return ExecutionAction(raw)

    def _intent_from_state(
        self,
        *,
        state: RuntimeState,
        action: ExecutionIntentAction,
        desired_capital: float | None = None,
        execution_capital: float | None = None,
        current_capital: float = 0.0,
        price: float = 0.0,
        close_quantity: float = 0.0,
        reasons: list[str] | None = None,
        notes: dict | None = None,
    ) -> ExecutionIntent:
        quantity = 0.0
        effective_capital = (
            float(execution_capital)
            if execution_capital is not None
            else (
                state.desired_capital
                if desired_capital is None
                else float(desired_capital)
            )
        )
        if price > 0 and effective_capital > 0:
            quantity = effective_capital / price
        return ExecutionIntent(
            runtime_id=state.runtime_id,
            symbol=state.symbol,
            timeframe=state.timeframe,
            strategy_id=state.strategy_id,
            family=state.family,
            lifecycle_state=state.lifecycle_state,
            action=action,
            desired_capital=(
                state.desired_capital if desired_capital is None else float(desired_capital)
            ),
            current_capital=float(current_capital),
            price=float(price),
            quantity=float(quantity),
            close_quantity=float(close_quantity),
            reasons=list(reasons or []),
            notes={
                **dict(state.notes or {}),
                **dict(notes or {}),
                "execution_capital": effective_capital,
                "score": float(state.score),
                "last_directive_action": state.last_directive_action.value,
            },
        )

    @staticmethod
    def _scale_down_reasons(state: RuntimeState) -> list[str]:
        reasons = [str(item) for item in (state.notes or {}).get("reasons", []) if str(item).strip()]
        staged_exit_state = lifecycle_policy_staged_exit_state(state.notes)
        if (
            staged_exit_active(staged_exit_state)
            and str(staged_exit_state.get("phase") or "") in {"harvest", "deep_harvest"}
        ):
            return ["profit_lock_harvest"]
        if "profit_lock_harvest" in reasons:
            return ["profit_lock_harvest"]
        if state.last_directive_action == ExecutionAction.SCALE_DOWN:
            return ["rebalance_down"]
        return []

    @staticmethod
    def _scale_down_notes(state: RuntimeState) -> dict:
        reasons = [str(item) for item in (state.notes or {}).get("reasons", []) if str(item).strip()]
        staged_exit_state = lifecycle_policy_staged_exit_state(state.notes)
        if "profit_lock_harvest" in reasons:
            return {
                "position_adjustment": "profit_lock_harvest",
                "autonomy_scale_down_reason": "profit_lock_harvest",
                "planner_reasons": reasons,
                "staged_exit_phase": str(staged_exit_state.get("phase") or ""),
            }
        if (
            staged_exit_active(staged_exit_state)
            and str(staged_exit_state.get("phase") or "") in {"harvest", "deep_harvest"}
        ):
            return {
                "position_adjustment": "profit_lock_harvest",
                "autonomy_scale_down_reason": "profit_lock_harvest",
                "planner_reasons": reasons,
                "staged_exit_phase": str(staged_exit_state.get("phase") or ""),
            }
        return {
            "position_adjustment": "rebalance_down",
            "autonomy_scale_down_reason": "rebalance_down",
            "planner_reasons": reasons,
        }

    @classmethod
    def _repair_runtime_overrides_from_notes(cls, payload: dict) -> dict[str, float]:
        return repair_runtime_overrides_from_notes(payload)

    def _merged_runtime_overrides(
        self,
        *,
        previous: RuntimeState | None,
        repair_reentry_notes: dict,
        runtime_evidence: RuntimeEvidenceSnapshot | None = None,
    ) -> tuple[dict[str, float], dict]:
        return merged_runtime_overrides(
            config=self.config,
            previous=previous,
            repair_reentry_notes=repair_reentry_notes,
            runtime_evidence=runtime_evidence,
        )

    def _merged_staged_exit_state(
        self,
        *,
        previous: RuntimeState | None,
        directive,
        runtime_evidence: RuntimeEvidenceSnapshot | None = None,
    ) -> dict:
        return merged_staged_exit_state(
            config=self.config,
            previous=previous,
            directive=directive,
            runtime_evidence=runtime_evidence,
        )

    def _merged_reentry_state(
        self,
        *,
        previous: RuntimeState | None,
        runtime_id: str,
        timeframe: str,
        repair_reentry_notes: dict,
        runtime_overrides: dict[str, float],
        runtime_override_state: dict,
        latest_close_time_by_runtime: dict[str, datetime] | None,
    ) -> dict:
        return merged_reentry_state(
            config=self.config,
            previous=previous,
            repair_reentry_notes=repair_reentry_notes,
            runtime_overrides=runtime_overrides,
            runtime_override_state=runtime_override_state,
            latest_close_time_by_runtime=latest_close_time_by_runtime,
            runtime_id=runtime_id,
            timeframe=timeframe,
        )

    def _decay_runtime_overrides(
        self,
        runtime_overrides: dict[str, float],
    ) -> dict[str, float]:
        return decay_runtime_overrides(runtime_overrides, config=self.config)

    def _runtime_evidence_index(
        self,
        *,
        runtime_ids: list[str],
        previous_states: list[RuntimeState] | None,
    ) -> dict[str, RuntimeEvidenceSnapshot]:
        unique_ids = [
            str(item).strip()
            for item in runtime_ids
            if str(item).strip()
        ]
        if not unique_ids:
            return {}
        return RuntimeEvidenceCollector(self.feed, self.config).collect(
            unique_ids,
            previous_states=previous_states,
        )

    def _runtime_override_recovery_policy(
        self,
        runtime_evidence: RuntimeEvidenceSnapshot | None,
    ) -> tuple[str, int]:
        return runtime_override_recovery_policy(
            config=self.config,
            runtime_evidence=runtime_evidence,
        )

    @staticmethod
    def _runtime_override_performance_snapshot(
        runtime_evidence: RuntimeEvidenceSnapshot | None,
    ) -> dict:
        return runtime_override_performance_snapshot(runtime_evidence)

    @staticmethod
    def _merged_repair_reentry_notes(
        *,
        result: ExperimentResult,
        previous: RuntimeState | None,
    ) -> dict:
        return merged_repair_reentry_notes(
            previous=previous,
            current_notes=lifecycle_policy_repair_reentry_notes(result.notes),
        )

    @staticmethod
    def _apply_runtime_overrides_to_capital(
        *,
        desired_capital: float,
        runtime_overrides: dict[str, float],
        apply_weight_multiplier: bool = True,
    ) -> float:
        adjusted = float(desired_capital)
        max_weight_multiplier = float(runtime_overrides.get("max_weight_multiplier") or 1.0)
        if apply_weight_multiplier and max_weight_multiplier > 0:
            adjusted *= min(1.0, max_weight_multiplier)
        return round(max(0.0, adjusted), 2)

    @staticmethod
    def _apply_staged_exit_to_capital(
        *,
        desired_capital: float,
        allocated_capital: float,
        staged_exit_state: dict,
        apply_staged_exit: bool = True,
    ) -> float:
        if not apply_staged_exit:
            return round(max(0.0, desired_capital), 2)
        return apply_staged_exit_to_capital(
            desired_capital=desired_capital,
            allocated_capital=allocated_capital,
            staged_exit_state=staged_exit_state,
        )

    @staticmethod
    def _portfolio_consumed_weight_cap(
        allocation: PortfolioAllocation | None,
    ) -> bool:
        if allocation is None:
            return False
        return PortfolioAllocator.RUNTIME_OVERRIDE_WEIGHT_CAP_REASON in list(
            allocation.reasons or []
        )

    @staticmethod
    def _portfolio_consumed_staged_exit_cap(
        allocation: PortfolioAllocation | None,
    ) -> bool:
        if allocation is None:
            return False
        return PortfolioAllocator.STAGED_EXIT_CAP_REASON in list(
            allocation.reasons or []
        )

    def _latest_price(self, symbol: str, timeframe: str) -> float | None:
        candles = self.feed.load_candles(symbol, timeframe, limit=1)
        if not candles:
            return None
        return float(candles[-1]["close"])

    @staticmethod
    def _take_profit_pct(item: ExecutionIntent) -> float:
        runtime_overrides = lifecycle_policy_runtime_overrides(item.notes)
        bias = float(runtime_overrides.get("take_profit_bias") or 1.0)
        return max(0.001, 0.10 * max(0.1, bias))

    def _reentry_cooldown_reasons(self, state: RuntimeState) -> list[str]:
        reentry_state = lifecycle_policy_reentry_state(state.notes)
        if reentry_state_active(reentry_state) and str(reentry_state.get("phase") or "") == "cooldown":
            return ["repair_reentry_cooldown_active"]
        runtime_overrides = lifecycle_policy_runtime_overrides(state.notes)
        multiplier = float(runtime_overrides.get("entry_cooldown_bars_multiplier") or 0.0)
        if multiplier <= 0:
            return []
        timeframe_minutes = self._timeframe_minutes(state.timeframe)
        if timeframe_minutes <= 0:
            return []
        repair_reentry = lifecycle_policy_repair_reentry_notes(state.notes)
        source_runtime_id = str(repair_reentry.get("source_runtime_id") or "")
        latest_close = self._latest_managed_close_time(
            symbol=state.symbol,
            runtime_id=source_runtime_id or state.runtime_id,
            strategy_id=state.strategy_id,
        )
        if latest_close is None:
            return []
        cooldown_minutes = timeframe_minutes * multiplier
        elapsed_minutes = (datetime.now(timezone.utc) - latest_close).total_seconds() / 60.0
        if elapsed_minutes >= cooldown_minutes:
            return []
        return ["repair_reentry_cooldown_active"]

    def _latest_managed_close_time(
        self,
        *,
        symbol: str,
        runtime_id: str,
        strategy_id: str,
    ) -> datetime | None:
        latest: datetime | None = None
        for row in self.storage.get_pnl_ledger(limit=5000, event_type="close"):
            metadata = self._trade_metadata(row)
            if metadata.get("source") != self.MANAGED_SOURCE:
                continue
            row_runtime_id = str(metadata.get("runtime_id") or "")
            row_strategy_id = str(metadata.get("strategy_id") or "")
            same_runtime = bool(runtime_id) and row_runtime_id == runtime_id
            same_strategy = bool(strategy_id) and row_strategy_id == strategy_id
            same_symbol = str(row.get("symbol") or "") == str(symbol or "")
            if not same_runtime and not same_strategy and not same_symbol:
                continue
            event_time = str(row.get("event_time") or "").strip()
            if not event_time:
                continue
            try:
                parsed = datetime.fromisoformat(event_time)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if latest is None or parsed > latest:
                latest = parsed
        return latest

    @staticmethod
    def _timeframe_minutes(timeframe: str) -> float:
        text = str(timeframe or "").strip().lower()
        if not text:
            return 0.0
        try:
            if text.endswith("m"):
                return float(text[:-1] or 0.0)
            if text.endswith("h"):
                return float(text[:-1] or 0.0) * 60.0
            if text.endswith("d"):
                return float(text[:-1] or 0.0) * 1440.0
        except ValueError:
            return 0.0
        return 0.0
