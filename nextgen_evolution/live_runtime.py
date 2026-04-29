"""Runtime/operator controls for guarded nextgen live rollout."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field, replace

from config import Settings, get_settings
from execution.exchange_adapter import BinanceExchangeAdapter, OKXExchangeAdapter, SlippageGuard
from execution.live_trader import LiveTrader

from .config import EvolutionConfig


@dataclass(slots=True)
class AutonomyLiveStatus:
    """Effective operator state for one autonomy-live run."""

    requested_live: bool
    effective_live: bool
    dry_run: bool
    allow_entries: bool
    allow_managed_closes: bool
    force_flatten: bool
    runtime_mode: str
    allow_live_orders: bool
    provider: str
    whitelist: tuple[str, ...] = ()
    max_active_runtimes: int = 1
    kill_switch_active: bool = False
    kill_switch_reason: str = ""
    manual_recovery_required: bool = False
    manual_recovery_approved: bool = False
    model_degradation_status: str = ""
    reasons: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


class AutonomyLiveRuntime:
    """Resolve live operator state and build a guarded LiveTrader."""

    OPERATOR_REQUEST_STATE_KEY = "nextgen_autonomy_live_operator_request"
    REQUESTED_ENABLED_STATE_KEY = "nextgen_autonomy_live_requested_enabled"
    REQUESTED_REASON_STATE_KEY = "nextgen_autonomy_live_requested_reason"
    KILL_SWITCH_STATE_KEY = "nextgen_autonomy_live_kill_switch"
    KILL_SWITCH_REASON_STATE_KEY = "nextgen_autonomy_live_kill_switch_reason"
    STATUS_STATE_KEY = "nextgen_autonomy_live_status"
    EFFECTIVE_ENABLED_STATE_KEY = "nextgen_autonomy_live_effective_enabled"
    MODE_STATE_KEY = "nextgen_autonomy_live_mode"
    REASON_STATE_KEY = "nextgen_autonomy_live_reason"

    def __init__(
        self,
        storage,
        *,
        settings: Settings | None = None,
        okx_exchange_adapter_cls=OKXExchangeAdapter,
        binance_exchange_adapter_cls=BinanceExchangeAdapter,
        live_trader_cls=LiveTrader,
        slippage_guard_cls=SlippageGuard,
    ):
        self.storage = storage
        self.settings = settings or get_settings()
        self.okx_exchange_adapter_cls = okx_exchange_adapter_cls
        self.binance_exchange_adapter_cls = binance_exchange_adapter_cls
        self.live_trader_cls = live_trader_cls
        self.slippage_guard_cls = slippage_guard_cls

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        value = str(symbol or "").strip()
        if not value or "/" not in value or ":" in value:
            return value
        quote = value.split("/", 1)[1]
        return f"{value}:{quote}"

    def build_config(
        self,
        *,
        base_config: EvolutionConfig | None = None,
        requested_live: bool = False,
        whitelist: tuple[str, ...] | list[str] | None = None,
        max_active_runtimes: int | None = None,
    ) -> EvolutionConfig:
        config = replace(base_config or EvolutionConfig())
        configured_symbols = (
            list(whitelist)
            if whitelist is not None
            else list(self.settings.exchange.symbols or self.settings.exchange.core_symbols)
        )
        normalized_whitelist = tuple(
            item
            for item in (
                self.normalize_symbol(symbol)
                for symbol in configured_symbols
            )
            if item
        )
        derived_max_active = min(
            max(1, int(config.autonomy_live_max_active_runtimes)),
            max(1, int(self.settings.exchange.max_active_symbols)),
            max(1, int(self.settings.risk.max_positions)),
        )
        if max_active_runtimes is not None:
            derived_max_active = max(1, int(max_active_runtimes))
        config.autonomy_live_enabled = bool(requested_live)
        config.autonomy_live_require_explicit_enable = True
        config.autonomy_live_whitelist = normalized_whitelist
        config.autonomy_live_max_active_runtimes = derived_max_active
        return config

    def set_kill_switch(self, enabled: bool, *, reason: str = "") -> None:
        self.storage.set_state(self.KILL_SWITCH_STATE_KEY, "true" if enabled else "false")
        self.storage.set_state(self.KILL_SWITCH_REASON_STATE_KEY, reason if enabled else "")
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_kill_switch",
            "SYSTEM",
            {
                "enabled": bool(enabled),
                "reason": reason,
            },
        )

    def load_operator_request(self) -> dict:
        raw = self.storage.get_json_state(self.OPERATOR_REQUEST_STATE_KEY, {}) or {}
        if not isinstance(raw, dict):
            raw = {}
        whitelist_raw = raw.get("whitelist")
        whitelist = (
            None
            if whitelist_raw is None
            else tuple(
                item
                for item in (
                    self.normalize_symbol(symbol)
                    for symbol in whitelist_raw
                )
                if item
            )
        )
        max_active_raw = raw.get("max_active_runtimes")
        try:
            max_active_runtimes = (
                max(1, int(max_active_raw))
                if max_active_raw is not None
                else None
            )
        except (TypeError, ValueError):
            max_active_runtimes = None
        return {
            "requested_live": bool(raw.get("requested_live", False)),
            "whitelist": whitelist,
            "max_active_runtimes": max_active_runtimes,
            "reason": str(raw.get("reason") or ""),
            "updated_at": str(raw.get("updated_at") or ""),
        }

    def set_operator_request(
        self,
        *,
        requested_live: bool | None = None,
        whitelist: tuple[str, ...] | list[str] | None = None,
        max_active_runtimes: int | None = None,
        reason: str = "",
    ) -> dict:
        current = self.load_operator_request()
        payload = {
            "requested_live": (
                current["requested_live"]
                if requested_live is None
                else bool(requested_live)
            ),
            "whitelist": (
                None
                if whitelist is None and current["whitelist"] is None
                else list(
                    current["whitelist"]
                    if whitelist is None
                    else (
                        item
                        for item in (
                            self.normalize_symbol(symbol)
                            for symbol in whitelist
                        )
                        if item
                    )
                )
            ),
            "max_active_runtimes": (
                current["max_active_runtimes"]
                if max_active_runtimes is None
                else max(1, int(max_active_runtimes))
            ),
            "reason": str(reason or current["reason"] or ""),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.storage.set_json_state(self.OPERATOR_REQUEST_STATE_KEY, payload)
        self.storage.set_state(
            self.REQUESTED_ENABLED_STATE_KEY,
            "true" if payload["requested_live"] else "false",
        )
        self.storage.set_state(self.REQUESTED_REASON_STATE_KEY, payload["reason"])
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_operator_request",
            "SYSTEM",
            payload,
        )
        return self.load_operator_request()

    def resolve_operator_request(
        self,
        *,
        requested_live: bool | None = None,
        whitelist: tuple[str, ...] | list[str] | None = None,
        max_active_runtimes: int | None = None,
    ) -> dict:
        stored = self.load_operator_request()
        resolved_whitelist = stored["whitelist"]
        if whitelist is not None:
            resolved_whitelist = tuple(
                item
                for item in (
                    self.normalize_symbol(symbol)
                    for symbol in whitelist
                )
                if item
            )
        resolved_max_active_runtimes = stored["max_active_runtimes"]
        if max_active_runtimes is not None:
            resolved_max_active_runtimes = max(1, int(max_active_runtimes))
        return {
            "requested_live": (
                stored["requested_live"]
                if requested_live is None
                else bool(requested_live)
            ),
            "whitelist": resolved_whitelist,
            "max_active_runtimes": resolved_max_active_runtimes,
            "reason": stored["reason"],
            "updated_at": stored["updated_at"],
        }

    def evaluate(
        self,
        *,
        requested_live: bool,
        config: EvolutionConfig,
    ) -> AutonomyLiveStatus:
        runtime_mode = str(self.settings.app.runtime_mode or "").strip().lower()
        provider = str(self.settings.exchange.provider or "").strip().lower()
        allow_live_orders = bool(self.settings.app.allow_live_orders)
        kill_switch_active = self.storage.get_state(self.KILL_SWITCH_STATE_KEY, "false") == "true"
        kill_switch_reason = self.storage.get_state(self.KILL_SWITCH_REASON_STATE_KEY, "") or ""
        manual_recovery_required = (
            self.storage.get_state("manual_recovery_required", "false") == "true"
        )
        manual_recovery_approved = (
            self.storage.get_state("manual_recovery_approved", "false") == "true"
        )
        model_degradation_status = str(
            self.storage.get_state("model_degradation_status", "active") or "active"
        ).strip().lower()
        supported_provider = provider in {"okx", "binance"}
        missing_credentials = self._missing_credentials(provider)
        recovery_blocked = manual_recovery_required and not manual_recovery_approved
        force_flatten = bool(
            kill_switch_active
            or recovery_blocked
            or model_degradation_status == "disabled"
        )
        close_capable = bool(
            runtime_mode == "live"
            and supported_provider
            and not missing_credentials
        )
        allow_entries = bool(
            requested_live
            and allow_live_orders
            and close_capable
            and not force_flatten
            and bool(config.autonomy_live_whitelist)
        )
        allow_managed_closes = bool(close_capable and (allow_entries or force_flatten))

        reasons: list[str] = []
        if requested_live or force_flatten:
            if runtime_mode != "live":
                reasons.append("runtime_mode_not_live")
            if requested_live and not allow_live_orders:
                reasons.append("live_orders_not_allowed")
            if kill_switch_active:
                reasons.append("live_kill_switch_active")
            if recovery_blocked:
                reasons.append("manual_recovery_required")
            if model_degradation_status == "disabled":
                reasons.append("model_degradation_disabled")
            if requested_live and not config.autonomy_live_whitelist:
                reasons.append("live_whitelist_empty")
            if not supported_provider:
                reasons.append("unsupported_exchange_provider")
            if missing_credentials:
                reasons.append("missing_exchange_credentials")
        effective_live = allow_entries
        status = AutonomyLiveStatus(
            requested_live=bool(requested_live),
            effective_live=effective_live,
            dry_run=not effective_live,
            allow_entries=allow_entries,
            allow_managed_closes=allow_managed_closes,
            force_flatten=force_flatten,
            runtime_mode=runtime_mode,
            allow_live_orders=allow_live_orders,
            provider=provider,
            whitelist=tuple(config.autonomy_live_whitelist),
            max_active_runtimes=max(1, int(config.autonomy_live_max_active_runtimes)),
            kill_switch_active=kill_switch_active,
            kill_switch_reason=kill_switch_reason,
            manual_recovery_required=manual_recovery_required,
            manual_recovery_approved=manual_recovery_approved,
            model_degradation_status=model_degradation_status,
            reasons=reasons,
            notes={
                "missing_credentials": missing_credentials,
                "close_capable": close_capable,
            },
        )
        self.storage.set_json_state(self.STATUS_STATE_KEY, asdict(status))
        self.storage.set_state(
            self.EFFECTIVE_ENABLED_STATE_KEY,
            "true" if status.effective_live else "false",
        )
        self.storage.set_state(
            self.MODE_STATE_KEY,
            "live" if status.effective_live else "dry_run",
        )
        self.storage.set_state(self.REASON_STATE_KEY, ",".join(status.reasons))
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_gate",
            "SYSTEM",
            asdict(status),
        )
        return status

    def build_live_trader(
        self,
        *,
        status: AutonomyLiveStatus,
    ):
        exchange = (
            self._build_exchange_adapter(status.provider)
            if (status.effective_live or status.allow_managed_closes)
            else None
        )
        return self.live_trader_cls(
            self.storage,
            exchange=exchange,
            enabled=(status.effective_live or status.allow_managed_closes),
            slippage_guard=self.slippage_guard_cls(self.settings.risk.max_slippage_pct),
            order_timeout_seconds=self.settings.risk.order_timeout_seconds,
            limit_order_timeout_seconds=self.settings.risk.limit_order_timeout_seconds,
            limit_order_retry_count=self.settings.risk.limit_order_retry_count,
            order_poll_interval_seconds=self.settings.risk.order_poll_interval_seconds,
        )

    def _missing_credentials(self, provider: str) -> list[str]:
        provider = str(provider or "").strip().lower()
        missing: list[str] = []
        if provider == "binance":
            if not self.settings.exchange.api_key.get_secret_value():
                missing.append("api_key")
            if not self.settings.exchange.api_secret.get_secret_value():
                missing.append("api_secret")
            return missing
        if provider == "okx":
            if not self.settings.exchange.api_key.get_secret_value():
                missing.append("api_key")
            if not self.settings.exchange.api_secret.get_secret_value():
                missing.append("api_secret")
            if not self.settings.exchange.api_passphrase.get_secret_value():
                missing.append("api_passphrase")
            return missing
        return ["provider"]

    def _build_exchange_adapter(self, provider: str):
        provider = str(provider or "").strip().lower()
        if provider == "binance":
            return self.binance_exchange_adapter_cls(
                proxy_url=self.settings.exchange.proxy_url,
                api_key=self.settings.exchange.api_key.get_secret_value(),
                api_secret=self.settings.exchange.api_secret.get_secret_value(),
            )
        return self.okx_exchange_adapter_cls(
            proxy_url=self.settings.exchange.proxy_url,
            api_key=self.settings.exchange.api_key.get_secret_value(),
            api_secret=self.settings.exchange.api_secret.get_secret_value(),
            api_passphrase=self.settings.exchange.api_passphrase.get_secret_value(),
        )
