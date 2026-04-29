"""Failover wrapper for read-only market data providers."""
from __future__ import annotations

from datetime import datetime, timezone
import inspect
from typing import Any, Callable

from core.storage import Storage


class FailoverMarketDataCollector:
    """Route read-side market data calls through a primary collector with fallback."""

    ROUTE_STATE_KEY = "market_data_last_route"
    STATS_STATE_KEY = "market_data_failover_stats"

    def __init__(
        self,
        storage: Storage,
        *,
        primary_provider: str,
        primary_collector: object,
        secondary_provider: str,
        secondary_factory: Callable[[], object] | None = None,
    ):
        self.storage = storage
        self.primary_provider = str(primary_provider or "").strip().lower() or "primary"
        self.secondary_provider = str(secondary_provider or "").strip().lower() or "secondary"
        self.primary = primary_collector
        self._secondary_factory = secondary_factory
        self._secondary: object | None = None

    def __getattr__(self, name: str):
        return getattr(self.primary, name)

    @property
    def secondary(self) -> object | None:
        if self._secondary is None and self._secondary_factory is not None:
            self._secondary = self._secondary_factory()
        return self._secondary

    def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 300,
    ) -> list[dict]:
        return self._execute(
            operation="fetch_historical_ohlcv",
            symbol=symbol,
            primary_call=lambda collector: self._invoke(
                collector,
                "fetch_historical_ohlcv",
                symbol,
                timeframe,
                since=since,
                limit=limit,
            ),
            secondary_call=lambda collector: self._invoke(
                collector,
                "fetch_historical_ohlcv",
                symbol,
                timeframe,
                since=since,
                limit=limit,
            ),
            is_usable=lambda result: bool(result),
            default=[],
        )

    def fetch_latest_price(self, symbol: str) -> float | None:
        if not hasattr(self.primary, "fetch_latest_price"):
            return None
        return self._execute(
            operation="fetch_latest_price",
            symbol=symbol,
            primary_call=lambda collector: self._invoke(
                collector,
                "fetch_latest_price",
                symbol,
            ),
            secondary_call=lambda collector: self._invoke(
                collector,
                "fetch_latest_price",
                symbol,
            ),
            is_usable=lambda result: result is not None,
            default=None,
        )

    def measure_latency(self, symbol: str) -> dict:
        if not hasattr(self.primary, "measure_latency"):
            return {
                "symbol": symbol,
                "exchange_timestamp_ms": None,
                "received_at_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
                "latency_seconds": None,
                "status": "failed",
                "error": "unsupported",
            }
        return self._execute(
            operation="measure_latency",
            symbol=symbol,
            primary_call=lambda collector: self._invoke(
                collector,
                "measure_latency",
                symbol,
            ),
            secondary_call=lambda collector: self._invoke(
                collector,
                "measure_latency",
                symbol,
            ),
            is_usable=lambda result: isinstance(result, dict)
            and str(result.get("status", "")).lower() == "ok",
            default={
                "symbol": symbol,
                "exchange_timestamp_ms": None,
                "received_at_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
                "latency_seconds": None,
                "status": "failed",
                "error": "no_market_data_provider",
            },
        )

    def fetch_available_instruments(self) -> list[str]:
        if not hasattr(self.primary, "fetch_available_instruments"):
            return []
        return self._execute(
            operation="fetch_available_instruments",
            symbol="*",
            primary_call=lambda collector: self._invoke(
                collector,
                "fetch_available_instruments",
            ),
            secondary_call=lambda collector: self._invoke(
                collector,
                "fetch_available_instruments",
            ),
            is_usable=lambda result: bool(result),
            default=[],
        )

    def fetch_funding_rate(self, symbol: str) -> float | None:
        if not hasattr(self.primary, "fetch_funding_rate"):
            return None
        return self._execute(
            operation="fetch_funding_rate",
            symbol=symbol,
            primary_call=lambda collector: self._invoke(
                collector,
                "fetch_funding_rate",
                symbol,
            ),
            secondary_call=lambda collector: self._invoke(
                collector,
                "fetch_funding_rate",
                symbol,
            ),
            is_usable=lambda result: result is not None,
            default=None,
        )

    def summarize_order_book_depth(
        self,
        symbol: str,
        depth: int = 5,
    ):
        if not hasattr(self.primary, "summarize_order_book_depth"):
            return None
        return self._execute(
            operation="summarize_order_book_depth",
            symbol=symbol,
            primary_call=lambda collector: self._invoke(
                collector,
                "summarize_order_book_depth",
                symbol,
                depth=depth,
            ),
            secondary_call=lambda collector: self._invoke(
                collector,
                "summarize_order_book_depth",
                symbol,
                depth=depth,
            ),
            is_usable=lambda result: result is not None,
            default=None,
        )

    @staticmethod
    def _invoke(collector: object, method_name: str, *args, **kwargs):
        method = getattr(collector, method_name)
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(*args, **kwargs)
        parameters = list(signature.parameters.values())
        accepts_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if accepts_var_keyword:
            return method(*args, **kwargs)
        accepted_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
        return method(*args, **accepted_kwargs)

    def _execute(
        self,
        *,
        operation: str,
        symbol: str,
        primary_call: Callable[[object], Any],
        secondary_call: Callable[[object], Any],
        is_usable: Callable[[Any], bool],
        default: Any,
    ):
        primary_result = default
        primary_error = ""
        primary_failed = False
        try:
            primary_result = primary_call(self.primary)
            primary_failed = bool(getattr(self.primary, "last_operation_failed", False))
        except Exception as exc:
            primary_error = str(exc)
            primary_failed = True

        selected_provider = self.primary_provider
        selected_result = primary_result
        fallback_used = False
        secondary_failed = False
        secondary_error = ""
        secondary_result = default

        secondary = self.secondary
        if primary_failed and secondary is not None:
            try:
                secondary_result = secondary_call(secondary)
                secondary_failed = bool(getattr(secondary, "last_operation_failed", False))
            except Exception as exc:
                secondary_error = str(exc)
                secondary_failed = True

            if not secondary_failed and is_usable(secondary_result):
                selected_provider = self.secondary_provider
                selected_result = secondary_result
                fallback_used = True
            elif is_usable(primary_result):
                selected_provider = self.primary_provider
                selected_result = primary_result
            else:
                selected_provider = self.secondary_provider
                selected_result = secondary_result

        route_payload = {
            "operation": operation,
            "symbol": symbol,
            "primary_provider": self.primary_provider,
            "selected_provider": selected_provider,
            "fallback_used": fallback_used,
            "primary_failed": primary_failed,
            "secondary_attempted": bool(primary_failed and secondary is not None),
            "secondary_failed": secondary_failed,
            "primary_error": primary_error,
            "secondary_error": secondary_error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._persist_route(route_payload)
        if primary_failed:
            self.storage.insert_execution_event(
                "market_data_failover",
                symbol,
                route_payload,
            )
        return selected_result

    def _persist_route(self, route_payload: dict) -> None:
        self.storage.set_json_state(self.ROUTE_STATE_KEY, route_payload)
        stats = self.storage.get_json_state(self.STATS_STATE_KEY, {}) or {}
        operation = str(route_payload["operation"])
        bucket = dict(stats.get(operation, {}) or {})
        bucket["calls"] = int(bucket.get("calls", 0) or 0) + 1
        selected_provider = str(route_payload["selected_provider"])
        bucket[f"{selected_provider}_selected"] = int(
            bucket.get(f"{selected_provider}_selected", 0) or 0
        ) + 1
        if route_payload.get("fallback_used"):
            bucket["fallback_used"] = int(bucket.get("fallback_used", 0) or 0) + 1
        if route_payload.get("primary_failed"):
            bucket["primary_failures"] = int(bucket.get("primary_failures", 0) or 0) + 1
        if route_payload.get("secondary_failed"):
            bucket["secondary_failures"] = int(bucket.get("secondary_failures", 0) or 0) + 1
        bucket["last_selected_provider"] = selected_provider
        bucket["last_updated_at"] = route_payload["updated_at"]
        stats[operation] = bucket
        self.storage.set_json_state(self.STATS_STATE_KEY, stats)
