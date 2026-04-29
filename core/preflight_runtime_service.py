"""Run-once preflight and symbol-preparation helpers."""
from __future__ import annotations

from datetime import datetime

from config import Settings, get_settings
from core.storage import Storage


class PreflightRuntimeService:
    """Prepare cycle symbols and execute aborting/non-aborting preflight checks."""

    def __init__(
        self,
        storage: Storage,
        settings: Settings | None = None,
        *,
        cycle_runtime,
        notifier,
        reconciler,
        runtime_coordination=None,
        observe_promoted_models,
        refresh_learning_runtime_overrides,
        apply_runtime_overrides,
        get_active_symbols,
        get_shadow_observation_symbols,
        maybe_rebuild_execution_pool,
        check_market_latency,
        get_positions,
        account_state,
        apply_model_degradation,
        persist_runtime_settings_effective,
        enforce_accuracy_guard,
        manual_recovery_blocked,
        trigger_manual_recovery,
        requires_manual_recovery,
        get_circuit_breaker_active,
        get_circuit_breaker_reason,
        get_cooldown_until,
        set_circuit_breaker_state,
    ):
        self.storage = storage
        self.settings = settings or get_settings()
        self.cycle_runtime = cycle_runtime
        self.notifier = notifier
        self.reconciler = reconciler
        self.runtime_coordination = runtime_coordination
        self.observe_promoted_models = observe_promoted_models
        self.refresh_learning_runtime_overrides = refresh_learning_runtime_overrides
        self.apply_runtime_overrides = apply_runtime_overrides
        self.get_active_symbols = get_active_symbols
        self.get_shadow_observation_symbols = get_shadow_observation_symbols
        self.maybe_rebuild_execution_pool = maybe_rebuild_execution_pool
        self.check_market_latency = check_market_latency
        self.get_positions = get_positions
        self.account_state = account_state
        self.apply_model_degradation = apply_model_degradation
        self.persist_runtime_settings_effective = persist_runtime_settings_effective
        self.enforce_accuracy_guard = enforce_accuracy_guard
        self.manual_recovery_blocked = manual_recovery_blocked
        self.trigger_manual_recovery = trigger_manual_recovery
        self.requires_manual_recovery = requires_manual_recovery
        self.get_circuit_breaker_active = get_circuit_breaker_active
        self.get_circuit_breaker_reason = get_circuit_breaker_reason
        self.get_cooldown_until = get_cooldown_until
        self.set_circuit_breaker_state = set_circuit_breaker_state

    def prepare_cycle_symbols(self, now: datetime) -> dict:
        if self.runtime_coordination is not None:
            self.runtime_coordination.prepare_cycle_runtime(now)
        else:
            self.observe_promoted_models(now)
            self.refresh_learning_runtime_overrides(now)
            self.apply_runtime_overrides()
        active_symbols = self.get_active_symbols(force_refresh=False, now=now)
        shadow_symbols = self.get_shadow_observation_symbols(force_refresh=False, now=now)
        rebuild_result = self.maybe_rebuild_execution_pool(now, active_symbols)
        if rebuild_result and rebuild_result.get("changed"):
            active_symbols = self.get_active_symbols(force_refresh=False, now=now)
            shadow_symbols = self.get_shadow_observation_symbols(force_refresh=False, now=now)
        return {
            "active_symbols": active_symbols,
            "shadow_symbols": shadow_symbols,
        }

    def run_preflight(
        self,
        *,
        now: datetime,
        cycle_id: int,
        active_symbols: list[str],
        opened_positions: int,
        closed_positions: int,
    ) -> dict:
        if self.check_market_latency(now, active_symbols):
            self.cycle_runtime.fail_cycle(
                cycle_id,
                reconciliation_status="market_latency_block",
                notes=self.get_circuit_breaker_reason(),
                opened_positions=opened_positions,
                closed_positions=closed_positions,
                circuit_breaker_active=self.get_circuit_breaker_active(),
            )
            return {"abort": True}

        positions = self.get_positions()
        account = self.account_state(now, positions)
        self.apply_model_degradation(now)
        if self.runtime_coordination is not None:
            self.runtime_coordination.persist_effective_runtime_state()
        else:
            self.persist_runtime_settings_effective()
        self.enforce_accuracy_guard(now)
        account = self.account_state(now, positions)
        if self.manual_recovery_blocked():
            self.cycle_runtime.fail_cycle(
                cycle_id,
                reconciliation_status="manual_recovery_required",
                notes="manual_recovery_required",
                opened_positions=opened_positions,
                closed_positions=closed_positions,
                circuit_breaker_active=self.get_circuit_breaker_active(),
            )
            return {"abort": True}

        reconciliation = self.reconciler.run()
        if reconciliation.mismatch_count > 0:
            self.set_circuit_breaker_state(True, "reconciliation_mismatch")
            self.storage.set_state("last_cycle_status", "failed")
            self.storage.insert_execution_event(
                "reconciliation",
                "SYSTEM",
                {
                    "status": reconciliation.status,
                    "mismatch_count": reconciliation.mismatch_count,
                    "mismatch_ratio_pct": reconciliation.mismatch_ratio_pct,
                    "details": reconciliation.details,
                },
            )
            level = (
                "critical"
                if reconciliation.mismatch_ratio_pct
                >= self.settings.risk.reconciliation_alert_threshold_pct
                else "error"
            )
            self.notifier.notify(
                "reconciliation",
                "对账异常",
                str(reconciliation.details),
                level=level,
            )
            if level == "critical":
                self.trigger_manual_recovery(
                    "reconciliation_mismatch",
                    f"ratio_pct={reconciliation.mismatch_ratio_pct:.4%}",
                )
            self.cycle_runtime.fail_cycle(
                cycle_id,
                reconciliation_status=reconciliation.status,
                notes=str(reconciliation.details),
                opened_positions=opened_positions,
                closed_positions=closed_positions,
                circuit_breaker_active=True,
            )
            return {"abort": True}

        if account.circuit_breaker_active:
            self.storage.set_state("last_cycle_status", "failed")
            circuit_reason = self.get_circuit_breaker_reason() or "risk threshold breached"
            if self.requires_manual_recovery(self.get_circuit_breaker_reason()):
                self.trigger_manual_recovery(
                    self.get_circuit_breaker_reason(),
                    circuit_reason,
                )
            self.storage.insert_execution_event(
                "circuit_breaker",
                "SYSTEM",
                {
                    "reason": circuit_reason,
                    "cooldown_until": (
                        self.get_cooldown_until().isoformat()
                        if self.get_cooldown_until()
                        else None
                    ),
                },
            )
            self.notifier.notify(
                "circuit_breaker",
                "风控熔断已激活",
                circuit_reason,
                level="critical"
                if self.requires_manual_recovery(self.get_circuit_breaker_reason())
                else "error",
            )
        return {
            "abort": False,
            "positions": positions,
            "account": account,
            "reconciliation": reconciliation,
        }
