"""Live trader for CryptoAI v3."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from loguru import logger

from core.models import OrderStatus, SignalDirection
from core.storage import Storage
from execution.exchange_adapter import SlippageEstimate, SlippageGuard
from execution.order_manager import OrderManager


class LiveTrader:
    """Guarded live execution path with dry-run fallback."""

    ESTIMATED_FEE_PCT = 0.00075

    def __init__(
        self,
        storage: Storage,
        exchange: Any | None = None,
        enabled: bool = False,
        initial_balance: float = 10000.0,
        slippage_guard: SlippageGuard | None = None,
        order_timeout_seconds: int = 30,
        limit_order_timeout_seconds: int = 300,
        limit_order_retry_count: int = 1,
        order_poll_interval_seconds: int = 2,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self.storage = storage
        self.exchange = exchange
        self.enabled = enabled
        self.initial_balance = initial_balance
        self.order_manager = OrderManager(storage)
        self.slippage_guard = slippage_guard or SlippageGuard(0.001)
        self.order_timeout_seconds = max(0, order_timeout_seconds)
        self.limit_order_timeout_seconds = max(0, limit_order_timeout_seconds)
        self.limit_order_retry_count = max(0, limit_order_retry_count)
        self.order_poll_interval_seconds = max(0, order_poll_interval_seconds)
        self._sleep = sleep_fn or time.sleep
        mode = "LIVE" if enabled else "DRY RUN"
        logger.warning(f"LiveTrader initialized in {mode} mode")

    @staticmethod
    def _trade_metadata(trade: dict | None) -> dict:
        if not trade:
            return {}
        raw = trade.get("metadata_json")
        if isinstance(raw, dict):
            return dict(raw)
        try:
            payload = json.loads(raw or "{}")
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _slippage_cost(
        *,
        side: str,
        reference_price: float,
        fill_price: float,
        quantity: float,
    ) -> float:
        if reference_price <= 0 or quantity <= 0:
            return 0.0
        if side.lower() == "buy":
            return max(fill_price - reference_price, 0.0) * quantity
        return max(reference_price - fill_price, 0.0) * quantity

    def _record_open_ledger_entry(
        self,
        *,
        trade_id: str,
        symbol: str,
        direction: str,
        reference_price: float,
        fill_price: float,
        quantity: float,
        event_time: str,
        metadata: dict | None = None,
    ) -> None:
        notional_value = max(fill_price, 0.0) * max(quantity, 0.0)
        fee_cost = notional_value * self.ESTIMATED_FEE_PCT
        net_pnl = -fee_cost
        self.storage.insert_pnl_ledger_entry(
            {
                "trade_id": trade_id,
                "symbol": symbol,
                "direction": direction,
                "event_type": "open",
                "event_time": event_time,
                "quantity": quantity,
                "notional_value": notional_value,
                "reference_price": reference_price,
                "fill_price": fill_price,
                "gross_pnl": 0.0,
                "fee_cost": fee_cost,
                "slippage_cost": self._slippage_cost(
                    side="buy",
                    reference_price=reference_price,
                    fill_price=fill_price,
                    quantity=quantity,
                ),
                "net_pnl": net_pnl,
                "net_return_pct": (net_pnl / notional_value * 100) if notional_value > 0 else 0.0,
                "holding_hours": 0.0,
                "model_id": str((metadata or {}).get("model_id") or ""),
                "metadata": dict(metadata or {}),
            }
        )

    def _record_close_ledger_entry(
        self,
        *,
        trade: dict,
        symbol: str,
        reference_price: float,
        exit_price: float,
        closed_qty: float,
        incremental_pnl: float,
        exit_time: str,
        reason: str,
        execution_type: str,
        remaining_qty: float,
        exchange_order: dict | None,
    ) -> None:
        trade_metadata = self._trade_metadata(trade)
        entry_price = float(trade.get("entry_price") or 0.0)
        entry_time_raw = str(trade.get("entry_time") or "")
        holding_hours = 0.0
        if entry_time_raw:
            try:
                holding_hours = (
                    datetime.fromisoformat(exit_time) - datetime.fromisoformat(entry_time_raw)
                ).total_seconds() / 3600
            except Exception:
                holding_hours = 0.0
        fee_cost = max(exit_price, 0.0) * max(closed_qty, 0.0) * self.ESTIMATED_FEE_PCT
        entry_notional = max(entry_price, 0.0) * max(closed_qty, 0.0)
        net_pnl = float(incremental_pnl) - fee_cost
        self.storage.insert_pnl_ledger_entry(
            {
                "trade_id": str(trade.get("id") or ""),
                "symbol": symbol,
                "direction": str(trade.get("direction") or "LONG"),
                "event_type": "close",
                "event_time": exit_time,
                "quantity": closed_qty,
                "notional_value": entry_notional,
                "reference_price": reference_price,
                "fill_price": exit_price,
                "gross_pnl": incremental_pnl,
                "fee_cost": fee_cost,
                "slippage_cost": self._slippage_cost(
                    side="sell",
                    reference_price=reference_price,
                    fill_price=exit_price,
                    quantity=closed_qty,
                ),
                "net_pnl": net_pnl,
                "net_return_pct": (net_pnl / entry_notional * 100) if entry_notional > 0 else 0.0,
                "holding_hours": holding_hours,
                "model_id": str(trade_metadata.get("model_id") or ""),
                "metadata": {
                    **trade_metadata,
                    "reason": reason,
                    "execution_type": execution_type,
                    "remaining_qty": remaining_qty,
                    "exchange_order_id": (
                        exchange_order.get("exchange_order_id", "")
                        if isinstance(exchange_order, dict)
                        else ""
                    ),
                },
            }
        )

    def is_ready(self) -> bool:
        return self.enabled and self.exchange is not None

    @staticmethod
    def _quote_asset(symbol: str) -> str:
        if "/" not in symbol:
            return "USDT"
        quote = symbol.split("/", 1)[1]
        return quote.split(":", 1)[0]

    @staticmethod
    def _is_filled_status(status: str) -> bool:
        return status.lower() in {"closed", "filled"}

    @staticmethod
    def _is_open_status(status: str) -> bool:
        return status.lower() in {"open", "new", "partially_filled", "pending"}

    @staticmethod
    def _filled_quantity(exchange_order: dict | None) -> float:
        if not exchange_order:
            return 0.0
        try:
            return max(float(exchange_order.get("filled_qty") or 0.0), 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _mark_partially_filled(exchange_order: dict) -> dict:
        return {
            **exchange_order,
            "status": "partially_filled",
        }

    @staticmethod
    def _final_order_status(requested_qty: float, filled_qty: float) -> OrderStatus:
        if max(requested_qty - filled_qty, 0.0) <= 1e-8:
            return OrderStatus.FILLED
        return OrderStatus.PARTIALLY_FILLED

    def _available_quote_balance(self, symbol: str) -> float | None:
        if self.exchange is None or not hasattr(self.exchange, "fetch_free_balance"):
            return None
        try:
            return self.exchange.fetch_free_balance(self._quote_asset(symbol))
        except Exception as exc:
            logger.warning(f"Unable to fetch free balance for {symbol}: {exc}")
            return None

    def _create_trade_id(self, symbol: str) -> str:
        now = datetime.now(timezone.utc)
        return (
            f"T-{now.strftime('%Y%m%d%H%M')}-"
            f"{symbol.replace('/', '').replace(':', '')}-{uuid.uuid4().hex[:4]}"
        )

    @classmethod
    def _position_add_allowed(
        cls,
        *,
        existing_trade: dict | None,
        metadata: dict | None,
    ) -> bool:
        requested = dict(metadata or {})
        if not bool(requested.get("allow_position_add")) or existing_trade is None:
            return False
        existing_metadata = cls._trade_metadata(existing_trade)
        for key in ("source", "bridge_mode", "runtime_id", "strategy_id", "family", "timeframe"):
            current = str(existing_metadata.get(key) or "").strip()
            incoming = str(requested.get(key) or "").strip()
            if current and incoming and current != incoming:
                return False
        return True

    @staticmethod
    def _merge_rationale(existing: str, incoming: str) -> str:
        values: list[str] = []
        for value in (existing, incoming):
            text = str(value or "").strip()
            if text and text not in values:
                values.append(text)
        return ";".join(values)

    @classmethod
    def _merge_trade_metadata(
        cls,
        existing_trade: dict | None,
        metadata: dict | None,
    ) -> dict:
        merged = {
            **cls._trade_metadata(existing_trade),
            **dict(metadata or {}),
        }
        merged.pop("allow_position_add", None)
        return merged

    def _create_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        price: float,
        quantity: float,
        reason: str,
    ) -> str:
        return self.order_manager.create(
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            reason=reason,
        )

    def _create_and_reject_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        price: float,
        quantity: float,
        reason: str,
        event_type: str,
        payload: dict,
        reject_reason: str,
    ) -> None:
        order_id = self._create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            reason=reason,
        )
        self._reject_order(
            order_id,
            event_type,
            symbol,
            payload,
            reject_reason,
        )

    def _submit_live_order(
        self,
        *,
        symbol: str,
        manager_side: str,
        estimate_side: str,
        submit_side: str,
        price: float,
        quantity: float,
        reason: str,
        payload: dict,
        reject_event_type: str,
    ) -> tuple[str, dict, str, SlippageEstimate] | None:
        order_id: str | None = None
        try:
            estimate = self.exchange.estimate_slippage(
                symbol=symbol,
                side=estimate_side,
                quantity=quantity,
                reference_price=price,
            )
            allowed, _ = self.slippage_guard.check(estimate)
            order_id = self._create_order(
                symbol=symbol,
                side=manager_side,
                order_type="MARKET" if allowed else "LIMIT",
                price=price,
                quantity=quantity,
                reason=reason,
            )
            exchange_order, execution_type = self._submit_market_or_limit(
                order_id=order_id,
                symbol=symbol,
                side=submit_side,
                quantity=quantity,
                reference_price=price,
                estimate=estimate,
                payload=payload,
            )
        except Exception as exc:
            logger.exception(
                f"Live {'open' if reject_event_type == 'live_open_rejected' else 'close'} failed for {symbol}"
            )
            self._record_live_failure(
                order_id=order_id,
                event_type=reject_event_type,
                symbol=symbol,
                payload=payload,
                reason=f"exchange error: {exc}",
            )
            return None
        if not exchange_order:
            return None
        return order_id, exchange_order, execution_type, estimate

    def _dry_run_open_result(
        self,
        *,
        symbol: str,
        direction: SignalDirection,
        price: float,
        rationale: str,
        quantity: float,
        position_value: float,
        payload: dict,
    ) -> dict:
        order_id = self._create_order(
            symbol=symbol,
            side=direction.value,
            order_type="MARKET",
            price=price,
            quantity=quantity,
            reason=rationale,
        )
        logger.warning(f"[DRY RUN] Would open {symbol} {direction.value} @ {price}")
        self.order_manager.transition(order_id, OrderStatus.SUBMITTED, "live dry-run submit")
        self.order_manager.transition(order_id, OrderStatus.FILLED, "live dry-run filled")
        self.storage.insert_execution_event("live_open_dry_run", symbol, payload)
        return {
            "order_id": order_id,
            "symbol": symbol,
            "direction": direction.value,
            "price": price,
            "quantity": quantity,
            "position_value": position_value,
            "dry_run": True,
        }

    def _dry_run_close_result(
        self,
        *,
        symbol: str,
        current_price: float,
        reason: str,
        close_qty: float | None,
        pos: dict | None,
        payload: dict,
    ) -> dict:
        requested_close_qty = 0.0
        remaining_qty = 0.0
        pnl = 0.0
        pnl_pct = 0.0
        is_full_close = False
        if pos is not None:
            total_qty = pos["quantity"]
            requested_close_qty = total_qty if close_qty is None else min(close_qty, total_qty)
            remaining_qty = max(total_qty - requested_close_qty, 0.0)
            pnl = (current_price - pos["entry_price"]) * requested_close_qty
            pnl_pct = (
                (current_price / pos["entry_price"] - 1) * 100
                if pos["entry_price"]
                else 0.0
            )
            is_full_close = remaining_qty <= 1e-10
        order_id = self._create_order(
            symbol=symbol,
            side="CLOSE",
            order_type="MARKET",
            price=current_price,
            quantity=requested_close_qty,
            reason=reason,
        )
        logger.warning(f"[DRY RUN] Would close {symbol} ({reason})")
        self.order_manager.transition(order_id, OrderStatus.SUBMITTED, "live dry-run close submit")
        self.order_manager.transition(order_id, OrderStatus.FILLED, "live dry-run close filled")
        self.storage.insert_execution_event("live_close_dry_run", symbol, payload)
        return {
            "order_id": order_id,
            "symbol": symbol,
            "exit_price": current_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "closed_qty": requested_close_qty,
            "remaining_qty": remaining_qty,
            "is_full_close": is_full_close,
            "dry_run": True,
        }

    def _reject_order(
        self,
        order_id: str,
        event_type: str,
        symbol: str,
        payload: dict,
        reason: str,
    ) -> None:
        self.order_manager.transition(order_id, OrderStatus.REJECTED, reason)
        self.storage.insert_execution_event(
            event_type,
            symbol,
            {
                **payload,
                "order_id": order_id,
                "reason": reason,
            },
        )
        logger.warning(f"Live order rejected: {reason}")

    def _record_live_failure(
        self,
        order_id: str | None,
        event_type: str,
        symbol: str,
        payload: dict,
        reason: str,
    ) -> None:
        if order_id:
            self.order_manager.transition(order_id, OrderStatus.REJECTED, reason)
        self.storage.insert_execution_event(
            event_type,
            symbol,
            {
                **payload,
                **({"order_id": order_id} if order_id else {}),
                "reason": reason,
            },
        )
        logger.warning(f"Live order failed: {reason}")

    def _cancel_order(
        self,
        order_id: str,
        event_type: str,
        symbol: str,
        payload: dict,
        reason: str,
        exchange_order_id: str,
    ) -> None:
        if hasattr(self.exchange, "cancel_order") and exchange_order_id:
            try:
                self.exchange.cancel_order(exchange_order_id, symbol)
            except Exception as exc:
                logger.warning(f"Cancel order failed for {exchange_order_id}: {exc}")
        self.order_manager.transition(order_id, OrderStatus.CANCELLED, reason)
        self.storage.insert_execution_event(
            event_type,
            symbol,
            {
                **payload,
                "order_id": order_id,
                "exchange_order_id": exchange_order_id,
                "reason": reason,
            },
        )

    def _poll_order_until_complete(
        self,
        symbol: str,
        exchange_order: dict,
        timeout_seconds: int,
    ) -> dict:
        exchange_order_id = exchange_order.get("exchange_order_id", "")
        status = str(exchange_order.get("status") or "")
        if self._is_filled_status(status):
            return exchange_order
        if not exchange_order_id or not hasattr(self.exchange, "fetch_order"):
            return exchange_order
        if timeout_seconds <= 0:
            return exchange_order

        deadline = time.monotonic() + timeout_seconds
        latest = exchange_order
        while time.monotonic() < deadline:
            self._sleep(self.order_poll_interval_seconds)
            latest = self.exchange.fetch_order(exchange_order_id, symbol)
            latest_status = str(latest.get("status") or "")
            if self._is_filled_status(latest_status):
                return latest
            if not self._is_open_status(latest_status):
                return latest
        return latest

    def _limit_retry_price(
        self,
        side: str,
        reference_price: float,
        estimate: SlippageEstimate,
        attempt_index: int,
    ) -> float:
        buy_cap = reference_price * (1 + self.slippage_guard.max_slippage_pct)
        sell_cap = reference_price * (1 - self.slippage_guard.max_slippage_pct)
        if side.lower() == "buy":
            candidate = min(max(reference_price, estimate.expected_price), buy_cap)
            if attempt_index > 0:
                candidate = min(buy_cap, candidate * (1 + 0.0005 * attempt_index))
            return candidate

        candidate = max(min(reference_price, estimate.expected_price), sell_cap)
        if attempt_index > 0:
            candidate = max(sell_cap, candidate * (1 - 0.0005 * attempt_index))
        return candidate

    def _place_limit_with_retry(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        reference_price: float,
        estimate: SlippageEstimate,
        payload: dict,
        timeout_event_type: str,
    ) -> dict | None:
        if not hasattr(self.exchange, "place_limit_order"):
            self._reject_order(
                order_id,
                "live_open_rejected" if side.lower() == "buy" else "live_close_rejected",
                symbol,
                {**payload, "slippage_pct": estimate.slippage_pct},
                "slippage exceeded limit and no limit-order fallback is available",
            )
            return None

        for attempt in range(self.limit_order_retry_count + 1):
            limit_price = self._limit_retry_price(side, reference_price, estimate, attempt)
            self.order_manager.transition(
                order_id,
                OrderStatus.SUBMITTED,
                f"limit attempt {attempt + 1} @ {limit_price:.8f}",
            )
            exchange_order = self.exchange.place_limit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=limit_price,
            )
            final_order = self._poll_order_until_complete(
                symbol=symbol,
                exchange_order=exchange_order,
                timeout_seconds=self.limit_order_timeout_seconds,
            )
            final_status = str(final_order.get("status") or "")
            if self._is_filled_status(final_status):
                return final_order
            filled_qty = self._filled_quantity(final_order)
            if filled_qty > 0:
                if self._is_open_status(final_status):
                    self._cancel_order(
                        order_id=order_id,
                        event_type=timeout_event_type,
                        symbol=symbol,
                        payload={
                            **payload,
                            "attempt": attempt + 1,
                            "limit_price": limit_price,
                            "filled_qty": filled_qty,
                        },
                        reason=(
                            f"limit order partially filled after "
                            f"{self.limit_order_timeout_seconds}s"
                        ),
                        exchange_order_id=final_order.get("exchange_order_id", ""),
                    )
                return self._mark_partially_filled(final_order)

            self._cancel_order(
                order_id=order_id,
                event_type=timeout_event_type,
                symbol=symbol,
                payload={
                    **payload,
                    "attempt": attempt + 1,
                    "limit_price": limit_price,
                },
                reason=f"limit order timeout after {self.limit_order_timeout_seconds}s",
                exchange_order_id=final_order.get("exchange_order_id", ""),
            )
        return None

    def _submit_market_or_limit(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        reference_price: float,
        estimate: SlippageEstimate,
        payload: dict,
    ) -> tuple[dict | None, str]:
        allowed, _ = self.slippage_guard.check(estimate)
        if allowed:
            self.order_manager.transition(order_id, OrderStatus.SUBMITTED, "live submit")
            exchange_order = self.exchange.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
            )
            final_order = self._poll_order_until_complete(
                symbol=symbol,
                exchange_order=exchange_order,
                timeout_seconds=self.order_timeout_seconds,
            )
            final_status = str(final_order.get("status") or "")
            if self._is_filled_status(final_status):
                return final_order, "MARKET"
            filled_qty = self._filled_quantity(final_order)
            if filled_qty > 0:
                if self._is_open_status(final_status):
                    self._cancel_order(
                        order_id=order_id,
                        event_type="live_order_timeout",
                        symbol=symbol,
                        payload={**payload, "filled_qty": filled_qty},
                        reason=f"market order partially filled after {self.order_timeout_seconds}s",
                        exchange_order_id=final_order.get("exchange_order_id", ""),
                    )
                return self._mark_partially_filled(final_order), "MARKET"
            self._cancel_order(
                order_id=order_id,
                event_type="live_order_timeout",
                symbol=symbol,
                payload=payload,
                reason=f"market order timeout after {self.order_timeout_seconds}s",
                exchange_order_id=final_order.get("exchange_order_id", ""),
            )
            return None, "MARKET"

        final_order = self._place_limit_with_retry(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            reference_price=reference_price,
            estimate=estimate,
            payload=payload,
            timeout_event_type=(
                "live_open_limit_timeout" if side.lower() == "buy" else "live_close_limit_timeout"
            ),
        )
        return final_order, "LIMIT"

    def execute_open(
        self,
        symbol: str,
        direction: SignalDirection,
        price: float,
        confidence: float,
        rationale: str,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        quantity: float | None = None,
        position_value: float | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        payload = {
            "symbol": symbol,
            "direction": direction.value,
            "price": price,
            "confidence": confidence,
            "rationale": rationale,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "quantity": quantity,
            "position_value": position_value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": str((metadata or {}).get("model_id") or ""),
        }

        if direction != SignalDirection.LONG:
            self._create_and_reject_order(
                symbol=symbol,
                side=direction.value,
                order_type="MARKET",
                price=price,
                quantity=quantity or 0.0,
                reason=rationale,
                event_type="live_open_rejected",
                payload=payload,
                reject_reason="spot live trader only supports LONG entries",
            )
            return None

        if price <= 0:
            self._create_and_reject_order(
                symbol=symbol,
                side=direction.value,
                order_type="MARKET",
                price=price,
                quantity=quantity or 0.0,
                reason=rationale,
                event_type="live_open_rejected",
                payload=payload,
                reject_reason="invalid price",
            )
            return None

        positions = self.storage.get_positions()
        existing_position = next((position for position in positions if position["symbol"] == symbol), None)
        open_trades = self.storage.get_open_trades()
        existing_trade = next((item for item in open_trades if item["symbol"] == symbol), None)
        allow_position_add = self._position_add_allowed(
            existing_trade=existing_trade,
            metadata=metadata,
        )
        if existing_position is not None and not allow_position_add:
            self._create_and_reject_order(
                symbol=symbol,
                side=direction.value,
                order_type="MARKET",
                price=price,
                quantity=quantity or 0.0,
                reason=rationale,
                event_type="live_open_rejected",
                payload=payload,
                reject_reason="position already open",
            )
            return None

        quantity_for_check = quantity or (
            (position_value / price) if position_value and price > 0 else 0.0
        )
        if quantity_for_check <= 0:
            self._create_and_reject_order(
                symbol=symbol,
                side=direction.value,
                order_type="MARKET",
                price=price,
                quantity=quantity or 0.0,
                reason=rationale,
                event_type="live_open_rejected",
                payload=payload,
                reject_reason="invalid quantity",
            )
            return None

        if not self.is_ready():
            return self._dry_run_open_result(
                symbol=symbol,
                direction=direction,
                price=price,
                rationale=rationale,
                quantity=quantity_for_check,
                position_value=position_value or (quantity_for_check * price),
                payload=payload,
            )

        available_balance = self._available_quote_balance(symbol)
        estimated_notional = quantity_for_check * price
        if available_balance is not None and estimated_notional > available_balance:
            self._create_and_reject_order(
                symbol=symbol,
                side=direction.value,
                order_type="MARKET",
                price=price,
                quantity=quantity_for_check,
                reason=rationale,
                event_type="live_open_rejected",
                payload=payload,
                reject_reason=(
                    f"insufficient balance: need {estimated_notional:.4f}, "
                    f"free {available_balance:.4f}"
                ),
            )
            return None

        submit_result = self._submit_live_order(
            symbol=symbol,
            manager_side=direction.value,
            estimate_side="BUY",
            submit_side="buy",
            price=price,
            quantity=quantity_for_check,
            reason=rationale,
            payload=payload,
            reject_event_type="live_open_rejected",
        )
        if not submit_result:
            return None
        order_id, exchange_order, execution_type, estimate = submit_result

        filled_qty = float(exchange_order.get("filled_qty") or quantity_for_check)
        fill_price = float(
            exchange_order.get("average_price")
            or exchange_order.get("price")
            or estimate.expected_price
            or price
        )
        if filled_qty <= 0:
            self._reject_order(
                order_id,
                "live_open_rejected",
                symbol,
                {**payload, "exchange_order": exchange_order},
                "exchange returned zero filled quantity",
            )
            return None

        stop_loss = round(fill_price * (1 - stop_loss_pct), 8)
        take_profit = round(fill_price * (1 + take_profit_pct), 8)
        now = datetime.now(timezone.utc)
        trade_metadata = self._merge_trade_metadata(existing_trade, metadata)
        if allow_position_add and existing_position is not None and existing_trade is not None:
            existing_qty = float(existing_position.get("quantity") or 0.0)
            existing_entry = float(existing_position.get("entry_price") or 0.0)
            total_qty = existing_qty + filled_qty
            blended_entry = (
                ((existing_entry * existing_qty) + (fill_price * filled_qty)) / total_qty
                if total_qty > 0
                else fill_price
            )
            stop_loss = round(blended_entry * (1 - stop_loss_pct), 8)
            take_profit = round(blended_entry * (1 + take_profit_pct), 8)
            existing_initial_qty = float(
                existing_trade.get("initial_quantity")
                or existing_trade.get("quantity")
                or existing_qty
            )
            existing_confidence = float(existing_trade.get("confidence") or 0.0)
            blended_confidence = (
                ((existing_confidence * existing_qty) + (float(confidence) * filled_qty)) / total_qty
                if total_qty > 0
                else float(confidence)
            )
            merged_rationale = self._merge_rationale(
                str(existing_trade.get("rationale") or ""),
                rationale,
            )
            self.storage.update_open_trade_position(
                str(existing_trade["id"]),
                entry_price=blended_entry,
                quantity=total_qty,
                initial_quantity=existing_initial_qty + filled_qty,
                rationale=merged_rationale,
                confidence=blended_confidence,
                metadata=trade_metadata,
            )
            trade_id = str(existing_trade["id"])
            self._record_open_ledger_entry(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction.value,
                reference_price=price,
                fill_price=fill_price,
                quantity=filled_qty,
                event_time=now.isoformat(),
                metadata=trade_metadata,
            )
            self.storage.upsert_position(
                {
                    **existing_position,
                    "symbol": symbol,
                    "direction": direction.value,
                    "entry_price": blended_entry,
                    "quantity": total_qty,
                    "entry_time": now.isoformat(),
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }
            )
        else:
            trade_id = self._create_trade_id(symbol)
            self.storage.insert_trade(
                {
                    "id": trade_id,
                    "symbol": symbol,
                    "direction": direction.value,
                    "entry_price": fill_price,
                    "quantity": filled_qty,
                    "entry_time": now.isoformat(),
                    "rationale": rationale,
                    "confidence": confidence,
                    "metadata": trade_metadata,
                }
            )
            self._record_open_ledger_entry(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction.value,
                reference_price=price,
                fill_price=fill_price,
                quantity=filled_qty,
                event_time=now.isoformat(),
                metadata=trade_metadata,
            )
            self.storage.upsert_position(
                {
                    "symbol": symbol,
                    "direction": direction.value,
                    "entry_price": fill_price,
                    "quantity": filled_qty,
                    "entry_time": now.isoformat(),
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }
            )
        self.storage.insert_execution_event(
            "live_open",
            symbol,
            {
                **payload,
                "order_id": order_id,
                "trade_id": trade_id,
                "filled_qty": filled_qty,
                "fill_price": fill_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "execution_type": execution_type,
                "exchange_order": exchange_order,
                "model_id": str(trade_metadata.get("model_id") or ""),
                "position_adjustment": str(trade_metadata.get("position_adjustment") or ""),
                "allow_position_add": allow_position_add,
            },
        )
        self.order_manager.transition(
            order_id,
            self._final_order_status(quantity_for_check, filled_qty),
            exchange_order.get("exchange_order_id", "live filled"),
        )
        return {
            "order_id": order_id,
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction.value,
            "price": fill_price,
            "quantity": filled_qty,
            "position_value": filled_qty * fill_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "execution_type": execution_type,
            "is_full_fill": max(quantity_for_check - filled_qty, 0.0) <= 1e-8,
            "dry_run": False,
        }

    def execute_close(
        self,
        symbol: str,
        current_price: float,
        reason: str = "manual",
        close_qty: float | None = None,
    ) -> dict | None:
        payload = {
            "symbol": symbol,
            "price": current_price,
            "reason": reason,
            "close_qty": close_qty,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        positions = self.storage.get_positions()
        pos = next((item for item in positions if item["symbol"] == symbol), None)

        if not self.is_ready():
            return self._dry_run_close_result(
                symbol=symbol,
                current_price=current_price,
                reason=reason,
                close_qty=close_qty,
                pos=pos,
                payload=payload,
            )

        if not pos:
            self._create_and_reject_order(
                symbol=symbol,
                side="CLOSE",
                order_type="MARKET",
                price=current_price,
                quantity=close_qty or 0.0,
                reason=reason,
                event_type="live_close_rejected",
                payload=payload,
                reject_reason="no position to close",
            )
            return None

        total_qty = pos["quantity"]
        qty_to_close = total_qty if close_qty is None else min(close_qty, total_qty)
        if qty_to_close <= 0:
            self._create_and_reject_order(
                symbol=symbol,
                side="CLOSE",
                order_type="MARKET",
                price=current_price,
                quantity=close_qty or 0.0,
                reason=reason,
                event_type="live_close_rejected",
                payload=payload,
                reject_reason="invalid close quantity",
            )
            return None

        submit_result = self._submit_live_order(
            symbol=symbol,
            manager_side="CLOSE",
            estimate_side="SELL",
            submit_side="sell",
            price=current_price,
            quantity=qty_to_close,
            reason=reason,
            payload=payload,
            reject_event_type="live_close_rejected",
        )
        if not submit_result:
            return None
        order_id, exchange_order, execution_type, estimate = submit_result

        filled_qty = float(exchange_order.get("filled_qty") or qty_to_close)
        exit_price = float(
            exchange_order.get("average_price")
            or exchange_order.get("price")
            or estimate.expected_price
            or current_price
        )
        if filled_qty <= 0:
            self._reject_order(
                order_id,
                "live_close_rejected",
                symbol,
                {**payload, "exchange_order": exchange_order},
                "exchange returned zero filled quantity",
            )
            return None

        remaining_qty = max(total_qty - filled_qty, 0.0)
        pnl = (exit_price - pos["entry_price"]) * filled_qty
        pnl_pct = (
            (exit_price / pos["entry_price"] - 1) * 100 if pos["entry_price"] else 0.0
        )
        open_trades = self.storage.get_open_trades()
        trade = next((item for item in open_trades if item["symbol"] == symbol), None)
        recorded_pnl = pnl
        recorded_pnl_pct = pnl_pct
        now = datetime.now(timezone.utc).isoformat()
        if trade:
            if remaining_qty <= 1e-10:
                prev_pnl = trade.get("pnl") or 0
                total_pnl = prev_pnl + pnl
                original_qty = trade.get("initial_quantity") or trade["quantity"]
                total_pnl_pct = (
                    (total_pnl / (original_qty * trade["entry_price"])) * 100
                    if original_qty > 0 and trade["entry_price"] > 0
                    else pnl_pct
                )
                recorded_pnl = total_pnl
                recorded_pnl_pct = total_pnl_pct
                self.storage.update_trade_exit(
                    trade["id"],
                    exit_price,
                    now,
                    total_pnl,
                    total_pnl_pct,
                )
            else:
                self.storage.upsert_trade_partial_close(
                    trade["id"],
                    closed_qty=filled_qty,
                    exit_price=exit_price,
                    exit_time=now,
                    realized_pnl=pnl,
                    realized_pnl_pct=pnl_pct,
                    remaining_qty=remaining_qty,
                )
            self._record_close_ledger_entry(
                trade=trade,
                symbol=symbol,
                reference_price=current_price,
                exit_price=exit_price,
                closed_qty=filled_qty,
                incremental_pnl=pnl,
                exit_time=now,
                reason=reason,
                execution_type=execution_type,
                remaining_qty=remaining_qty,
                exchange_order=exchange_order,
            )

        if remaining_qty <= 1e-10:
            self.storage.delete_position(symbol)
        else:
            self.storage.upsert_position({**pos, "quantity": remaining_qty})

        self.storage.insert_execution_event(
            "live_close",
            symbol,
            {
                **payload,
                "order_id": order_id,
                "filled_qty": filled_qty,
                "exit_price": exit_price,
                "remaining_qty": remaining_qty,
                "pnl": recorded_pnl,
                "incremental_pnl": pnl,
                "pnl_pct": recorded_pnl_pct,
                "fee_cost": max(exit_price, 0.0) * max(filled_qty, 0.0) * self.ESTIMATED_FEE_PCT,
                "execution_type": execution_type,
                "exchange_order": exchange_order,
            },
        )
        self.order_manager.transition(
            order_id,
            self._final_order_status(qty_to_close, filled_qty),
            reason,
        )
        return {
            "order_id": order_id,
            "symbol": symbol,
            "exit_price": exit_price,
            "pnl": recorded_pnl,
            "pnl_pct": recorded_pnl_pct,
            "reason": reason,
            "closed_qty": filled_qty,
            "remaining_qty": remaining_qty,
            "is_full_close": remaining_qty <= 1e-10,
            "execution_type": execution_type,
            "dry_run": False,
            "trade_id": trade["id"] if trade else "",
            "entry_time": trade["entry_time"] if trade else pos["entry_time"],
            "entry_price": trade["entry_price"] if trade else pos["entry_price"],
            "confidence": trade.get("confidence", 0.0) if trade else 0.0,
            "rationale": trade.get("rationale", "") if trade else "",
        }
