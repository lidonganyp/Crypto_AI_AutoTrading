"""Bridge autonomy directives into guarded limited-live execution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from core.models import SignalDirection
from execution.live_trader import LiveTrader

from .config import EvolutionConfig
from .execution_bridge import AutonomyPaperBridge
from .experiment_lab import ExperimentResult
from .live_runtime import AutonomyLiveStatus
from .models import (
    AutonomyDirective,
    ExecutionIntent,
    ExecutionIntentAction,
    PortfolioAllocation,
    RuntimeLifecycleState,
    RuntimeState,
)
from .promotion_registry import PromotionRegistry
from .rollout import RolloutStateMachine


class AutonomyLiveBridge(AutonomyPaperBridge):
    """Translate autonomy output into tightly-bounded limited-live execution."""

    BRIDGE_MODE = "live"

    def __init__(
        self,
        feed,
        *,
        registry: PromotionRegistry | None = None,
        live_trader: LiveTrader | None = None,
        config: EvolutionConfig | None = None,
        rollout: RolloutStateMachine | None = None,
        operator_status: AutonomyLiveStatus | None = None,
    ):
        self.feed = feed
        self.storage = feed.storage
        self.config = config or EvolutionConfig()
        self.registry = registry
        self.live_trader = live_trader or LiveTrader(self.storage, enabled=False)
        self.rollout = rollout or RolloutStateMachine(self.config)
        self.operator_status = operator_status

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
        executed = self.execute_live(
            intents,
            autonomy_cycle_id=autonomy_cycle_id,
        )
        return runtime_states, executed

    def build_execution_intents(
        self,
        runtime_states: list[RuntimeState],
    ) -> list[ExecutionIntent]:
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
        reserved_slots = {
            symbol
            for symbol, trade in open_trade_by_symbol.items()
            if self._is_live_managed_trade(self._trade_metadata(trade))
        }

        intents: list[ExecutionIntent] = []
        all_symbols = sorted(
            set(state_by_symbol)
            | {
                symbol
                for symbol, trade in open_trade_by_symbol.items()
                if self._is_live_managed_trade(self._trade_metadata(trade))
            }
        )
        for symbol in all_symbols:
            primary = self._primary_live_state(state_by_symbol.get(symbol, []))
            current_position = positions_by_symbol.get(symbol)
            current_trade = open_trade_by_symbol.get(symbol)
            current_metadata = self._trade_metadata(current_trade)
            live_managed_trade = self._is_live_managed_trade(current_metadata)
            current_runtime_id = str(current_metadata.get("runtime_id") or "")
            price = self._intent_price(
                symbol=symbol,
                primary=primary,
                current_metadata=current_metadata,
                current_position=current_position,
            )
            if self._force_flatten() and live_managed_trade and current_position and current_trade:
                flatten_state = (
                    primary
                    if primary is not None
                    else self._state_from_trade(current_trade, current_metadata)
                )
                intents.append(
                    self._intent_from_state(
                        state=flatten_state,
                        action=ExecutionIntentAction.CLOSE,
                        current_capital=float(current_position["quantity"]) * max(price, 0.0),
                        price=price,
                        reasons=["operator_forced_flatten", *self._operator_gate_reasons()],
                        notes={"bridge_mode": self.BRIDGE_MODE},
                    )
                )
                continue

            if current_position and current_trade and not live_managed_trade:
                if primary is None:
                    continue
                intents.append(
                    self._intent_from_state(
                        state=primary,
                        action=ExecutionIntentAction.SKIP,
                        desired_capital=primary.desired_capital,
                        current_capital=float(current_position["quantity"]) * max(price, 0.0),
                        price=price,
                        reasons=["symbol_managed_elsewhere"],
                        notes={"bridge_mode": self.BRIDGE_MODE},
                    )
                )
                continue

            if primary is None:
                if live_managed_trade and current_position and current_trade:
                    intents.append(
                        self._intent_from_state(
                            state=self._state_from_trade(current_trade, current_metadata),
                            action=ExecutionIntentAction.CLOSE,
                            current_capital=float(current_position["quantity"]) * max(price, 0.0),
                            price=price,
                            reasons=["runtime_not_selected"],
                            notes={"bridge_mode": self.BRIDGE_MODE},
                        )
                    )
                continue

            if not self._is_whitelisted(symbol):
                if live_managed_trade and current_position and current_trade:
                    intents.append(
                        self._intent_from_state(
                            state=primary,
                            action=ExecutionIntentAction.CLOSE,
                            current_capital=float(current_position["quantity"]) * max(price, 0.0),
                            price=price,
                            reasons=["symbol_not_whitelisted"],
                            notes={"bridge_mode": self.BRIDGE_MODE},
                        )
                    )
                else:
                    intents.append(
                        self._intent_from_state(
                            state=primary,
                            action=ExecutionIntentAction.SKIP,
                            desired_capital=primary.desired_capital,
                            price=price,
                            reasons=["symbol_not_whitelisted"],
                            notes={"bridge_mode": self.BRIDGE_MODE},
                        )
                    )
                continue

            if primary.lifecycle_state not in {
                RuntimeLifecycleState.LIMITED_LIVE,
                RuntimeLifecycleState.LIVE,
            }:
                if live_managed_trade and current_position and current_trade:
                    intents.append(
                        self._intent_from_state(
                            state=primary,
                            action=ExecutionIntentAction.CLOSE,
                            current_capital=float(current_position["quantity"]) * max(price, 0.0),
                            price=price,
                            reasons=["rollout_requires_exit"],
                            notes={"bridge_mode": self.BRIDGE_MODE},
                        )
                    )
                else:
                    intents.append(
                        self._intent_from_state(
                            state=primary,
                            action=ExecutionIntentAction.HOLD,
                            desired_capital=primary.desired_capital,
                            price=price,
                            reasons=["non_live_state"],
                            notes={"bridge_mode": self.BRIDGE_MODE},
                        )
                    )
                continue

            if current_trade and live_managed_trade and current_position is None:
                intents.append(
                    self._intent_from_state(
                        state=primary,
                        action=ExecutionIntentAction.SKIP,
                        desired_capital=primary.desired_capital,
                        price=price,
                        reasons=["live_trade_without_position"],
                        notes={"bridge_mode": self.BRIDGE_MODE},
                    )
                )
                continue

            if current_position and current_trade and live_managed_trade:
                current_capital = float(current_position["quantity"]) * max(price, 0.0)
                if current_runtime_id and current_runtime_id != primary.runtime_id:
                    intents.append(
                        self._intent_from_state(
                            state=self._state_from_trade(current_trade, current_metadata),
                            action=ExecutionIntentAction.CLOSE,
                            current_capital=current_capital,
                            price=price,
                            reasons=["runtime_rotation_out"],
                            notes={"bridge_mode": self.BRIDGE_MODE},
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
                                "bridge_mode": self.BRIDGE_MODE,
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
                            notes={
                                "bridge_mode": self.BRIDGE_MODE,
                                **self._scale_down_notes(primary),
                            },
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
                        notes={"bridge_mode": self.BRIDGE_MODE},
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
                        notes={"bridge_mode": self.BRIDGE_MODE},
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
                        notes={"bridge_mode": self.BRIDGE_MODE},
                    )
                )
                continue

            if not self._allow_live_entries():
                intents.append(
                    self._intent_from_state(
                        state=primary,
                        action=ExecutionIntentAction.SKIP,
                        desired_capital=primary.desired_capital,
                        price=price,
                        reasons=["operator_gate_blocked", *self._operator_gate_reasons()],
                        notes={"bridge_mode": self.BRIDGE_MODE},
                    )
                )
                continue

            if (
                symbol not in reserved_slots
                and len(reserved_slots) >= max(1, int(self.config.autonomy_live_max_active_runtimes))
            ):
                intents.append(
                    self._intent_from_state(
                        state=primary,
                        action=ExecutionIntentAction.SKIP,
                        desired_capital=primary.desired_capital,
                        price=price,
                        reasons=["live_runtime_cap_reached"],
                        notes={"bridge_mode": self.BRIDGE_MODE},
                    )
                )
                continue

            intents.append(
                self._intent_from_state(
                    state=primary,
                    action=ExecutionIntentAction.OPEN,
                    desired_capital=primary.desired_capital,
                    price=price,
                    reasons=["deploy_live_runtime"],
                    notes={"bridge_mode": self.BRIDGE_MODE},
                )
            )
            reserved_slots.add(symbol)

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

    def execute_live(
        self,
        intents: list[ExecutionIntent],
        *,
        autonomy_cycle_id: int | None = None,
    ) -> list[ExecutionIntent]:
        executed: list[ExecutionIntent] = []
        for item in intents:
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
                "nextgen_autonomy_live_intent",
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
                    "bridge_mode": self.BRIDGE_MODE,
                    "explicit_live_allowed": self._explicit_live_allowed(),
                    "live_trader_enabled": bool(self.live_trader.enabled),
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
        result = self._with_live_entry_gate(
            lambda: self.live_trader.execute_open(
                item.symbol,
                SignalDirection.LONG,
                item.price,
                confidence=min(
                    0.99,
                    max(0.0, 0.50 + max(0.0, item.notes.get("score", 0.0))),
                ),
                rationale=";".join(item.reasons) or "nextgen_autonomy_live_open",
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
        )
        if not result:
            return replace(item, status="rejected")
        return replace(
            item,
            status="dry_run" if result.get("dry_run") else "executed",
            quantity=float(result.get("quantity") or 0.0),
            notes={
                **item.notes,
                "order_id": result.get("order_id"),
                "trade_id": result.get("trade_id"),
                "execution_type": result.get(
                    "execution_type",
                    "DRY_RUN" if result.get("dry_run") else "",
                ),
                "dry_run": bool(result.get("dry_run")),
            },
        )

    def _execute_close(self, item: ExecutionIntent) -> ExecutionIntent:
        result = self._with_live_close_gate(
            lambda: self.live_trader.execute_close(
                item.symbol,
                item.price,
                reason="nextgen_autonomy_live_close",
            )
        )
        if not result:
            return replace(item, status="skipped")
        return replace(
            item,
            status="dry_run" if result.get("dry_run") else "executed",
            close_quantity=float(result.get("closed_qty") or 0.0),
            notes={
                **item.notes,
                "order_id": result.get("order_id"),
                "trade_id": result.get("trade_id"),
                "execution_type": result.get(
                    "execution_type",
                    "DRY_RUN" if result.get("dry_run") else "",
                ),
                "dry_run": bool(result.get("dry_run")),
            },
        )

    def _execute_reduce(self, item: ExecutionIntent) -> ExecutionIntent:
        if item.close_quantity <= 1e-8:
            return replace(item, status="skipped")
        close_reason = (
            "nextgen_autonomy_live_profit_lock_reduce"
            if str(item.notes.get("position_adjustment") or "") == "profit_lock_harvest"
            else "nextgen_autonomy_live_reduce"
        )
        result = self._with_live_close_gate(
            lambda: self.live_trader.execute_close(
                item.symbol,
                item.price,
                reason=close_reason,
                close_qty=item.close_quantity,
            )
        )
        if not result:
            return replace(item, status="skipped")
        return replace(
            item,
            status="dry_run" if result.get("dry_run") else "executed",
            close_quantity=float(result.get("closed_qty") or 0.0),
            notes={
                **item.notes,
                "order_id": result.get("order_id"),
                "trade_id": result.get("trade_id"),
                "execution_type": result.get(
                    "execution_type",
                    "DRY_RUN" if result.get("dry_run") else "",
                ),
                "dry_run": bool(result.get("dry_run")),
            },
        )

    def _primary_live_state(self, states: list[RuntimeState]) -> RuntimeState | None:
        deployable = [
            item
            for item in states
            if item.lifecycle_state
            in {
                RuntimeLifecycleState.LIMITED_LIVE,
                RuntimeLifecycleState.LIVE,
            }
            and item.desired_capital > 0
        ]
        if not deployable:
            return None
        lifecycle_rank = {
            RuntimeLifecycleState.LIVE: 2,
            RuntimeLifecycleState.LIMITED_LIVE: 1,
        }
        return max(
            deployable,
            key=lambda item: (
                lifecycle_rank.get(item.lifecycle_state, 0),
                item.desired_capital,
                item.score,
            ),
        )

    def _is_whitelisted(self, symbol: str) -> bool:
        whitelist = {str(item) for item in self.config.autonomy_live_whitelist}
        return symbol in whitelist

    def _is_live_managed_trade(self, metadata: dict) -> bool:
        return (
            metadata.get("source") == self.MANAGED_SOURCE
            and str(metadata.get("bridge_mode") or "") == self.BRIDGE_MODE
        )

    def _intent_price(
        self,
        *,
        symbol: str,
        primary: RuntimeState | None,
        current_metadata: dict,
        current_position: dict | None,
    ) -> float:
        timeframe = (
            primary.timeframe
            if primary is not None
            else str(current_metadata.get("timeframe") or "5m")
        )
        latest = self._latest_price(symbol, timeframe)
        if latest is not None:
            return float(latest)
        if current_position is not None:
            return float(current_position.get("entry_price") or 0.0)
        return 0.0

    def _explicit_live_allowed(self) -> bool:
        if not self.config.autonomy_live_require_explicit_enable:
            return True
        return bool(self.config.autonomy_live_enabled)

    def _allow_live_entries(self) -> bool:
        if self.operator_status is not None:
            return bool(self.operator_status.allow_entries)
        return self._explicit_live_allowed()

    def _allow_live_closes(self) -> bool:
        if self.operator_status is not None:
            return bool(self.operator_status.allow_managed_closes)
        return self._explicit_live_allowed()

    def _force_flatten(self) -> bool:
        return bool(self.operator_status.force_flatten) if self.operator_status is not None else False

    def _operator_gate_reasons(self) -> list[str]:
        if self.operator_status is None:
            return []
        return [str(item) for item in self.operator_status.reasons if str(item).strip()]

    def _with_live_entry_gate(self, action):
        original_enabled = self.live_trader.enabled
        if not self._allow_live_entries():
            self.live_trader.enabled = False
        try:
            return action()
        finally:
            self.live_trader.enabled = original_enabled

    def _with_live_close_gate(self, action):
        original_enabled = self.live_trader.enabled
        if not self._allow_live_closes():
            self.live_trader.enabled = False
        try:
            return action()
        finally:
            self.live_trader.enabled = original_enabled
