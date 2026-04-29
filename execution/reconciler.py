"""Position and trade reconciliation helpers."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from core.storage import Storage


@dataclass
class ReconciliationResult:
    status: str
    mismatch_count: int
    mismatch_ratio_pct: float
    details: dict = field(default_factory=dict)


class Reconciler:
    """Compare local state with exchange-facing state when available."""

    def __init__(self, storage: Storage, exchange: Any | None = None):
        self.storage = storage
        self.exchange = exchange

    @staticmethod
    def _base_asset(symbol: str) -> str:
        return str(symbol).split("/", 1)[0].split(":", 1)[0]

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        text = str(symbol or "").strip().upper().replace(" ", "")
        if ":USDT" in text:
            text = text.replace(":USDT", "")
        if "-" in text and "/" not in text:
            parts = text.split("-")
            if len(parts) >= 2:
                text = f"{parts[0]}/{parts[1]}"
        return text

    @staticmethod
    def _normalize_order_side(side: str) -> str:
        raw = str(side or "").strip().upper()
        if raw in {"LONG", "BUY"}:
            return "BUY"
        if raw in {"CLOSE", "SELL", "SHORT"}:
            return "SELL"
        return raw or "UNKNOWN"

    def run(self) -> ReconciliationResult:
        positions = self.storage.get_positions()
        open_trades = self.storage.get_open_trades()
        open_orders = [
            order
            for order in self.storage.get_orders(limit=500)
            if str(order.get("status") or "").upper()
            in {"CREATED", "SUBMITTED", "PARTIALLY_FILLED"}
        ]

        positions_by_symbol = {position["symbol"]: position for position in positions}
        trades_by_symbol = {trade["symbol"]: trade for trade in open_trades}

        missing_positions = sorted(
            symbol for symbol in trades_by_symbol if symbol not in positions_by_symbol
        )
        missing_trades = sorted(
            symbol for symbol in positions_by_symbol if symbol not in trades_by_symbol
        )
        quantity_mismatches = []
        mismatch_ratio_pct = 0.0
        for symbol in sorted(set(positions_by_symbol) & set(trades_by_symbol)):
            pos_qty = float(positions_by_symbol[symbol]["quantity"])
            trade_qty = float(trades_by_symbol[symbol]["quantity"])
            if abs(pos_qty - trade_qty) > 1e-8:
                base_qty = max(abs(trade_qty), 1e-8)
                ratio_pct = abs(pos_qty - trade_qty) / base_qty
                mismatch_ratio_pct = max(mismatch_ratio_pct, ratio_pct)
                quantity_mismatches.append(
                    {
                        "symbol": symbol,
                        "position_quantity": pos_qty,
                        "trade_quantity": trade_qty,
                        "ratio_pct": ratio_pct,
                    }
                )

        exchange_balance_mismatches = []
        exchange_checked = False
        exchange_errors: list[str] = []
        exchange_order_state_checked = False
        missing_exchange_orders = []
        unexpected_exchange_orders = []
        exchange_order_quantity_mismatches = []
        if self.exchange is not None and hasattr(self.exchange, "fetch_total_balance"):
            tracked_symbols = sorted(set(positions_by_symbol) | set(trades_by_symbol))
            for symbol in tracked_symbols:
                try:
                    exchange_qty = self.exchange.fetch_total_balance(
                        self._base_asset(symbol)
                    )
                    exchange_checked = True
                except Exception as exc:
                    exchange_errors.append(f"{symbol}: {exc}")
                    continue
                if exchange_qty is None:
                    continue
                local_qty = float(
                    positions_by_symbol.get(symbol, {}).get(
                        "quantity",
                        trades_by_symbol.get(symbol, {}).get("quantity", 0.0),
                    )
                )
                # External/manual holdings above the locally tracked quantity are tolerated.
                # The dangerous case is when the exchange has less than the bot thinks it has.
                if self._is_exchange_deficit(exchange_qty, local_qty):
                    base_qty = max(abs(local_qty), 1e-8)
                    ratio_pct = abs(local_qty - exchange_qty) / base_qty
                    mismatch_ratio_pct = max(mismatch_ratio_pct, ratio_pct)
                    exchange_balance_mismatches.append(
                        {
                            "symbol": symbol,
                            "exchange_quantity": exchange_qty,
                            "local_quantity": local_qty,
                            "ratio_pct": ratio_pct,
                        }
                    )
        if self.exchange is not None and hasattr(self.exchange, "fetch_open_orders"):
            try:
                exchange_orders = self.exchange.fetch_open_orders()
                exchange_order_state_checked = True
                local_order_counts = Counter()
                local_order_qty: dict[tuple[str, str], float] = {}
                for order in open_orders:
                    key = (
                        self._normalize_symbol(order.get("symbol", "")),
                        self._normalize_order_side(order.get("side", "")),
                    )
                    local_order_counts[key] += 1
                    local_order_qty[key] = local_order_qty.get(key, 0.0) + float(order.get("quantity") or 0.0)

                exchange_order_counts = Counter()
                exchange_order_qty: dict[tuple[str, str], float] = {}
                for order in exchange_orders:
                    key = (
                        self._normalize_symbol(order.get("symbol", "")),
                        self._normalize_order_side(order.get("side", "")),
                    )
                    if not key[0]:
                        continue
                    exchange_order_counts[key] += 1
                    exchange_order_qty[key] = exchange_order_qty.get(key, 0.0) + float(order.get("requested_qty") or 0.0)

                for key in sorted(local_order_counts):
                    symbol, side = key
                    local_count = int(local_order_counts.get(key, 0))
                    exchange_count = int(exchange_order_counts.get(key, 0))
                    if local_count > exchange_count:
                        missing_exchange_orders.append(
                            {
                                "symbol": symbol,
                                "side": side,
                                "local_open_orders": local_count,
                                "exchange_open_orders": exchange_count,
                            }
                        )
                    local_qty = float(local_order_qty.get(key, 0.0))
                    exchange_qty = float(exchange_order_qty.get(key, 0.0))
                    if self._is_exchange_deficit(exchange_qty, local_qty):
                        exchange_order_quantity_mismatches.append(
                            {
                                "symbol": symbol,
                                "side": side,
                                "local_open_qty": local_qty,
                                "exchange_open_qty": exchange_qty,
                            }
                        )
            except Exception as exc:
                exchange_errors.append(f"open_orders: {exc}")

        mismatch_count = (
            len(missing_positions)
            + len(missing_trades)
            + len(quantity_mismatches)
            + len(exchange_balance_mismatches)
            + len(missing_exchange_orders)
            + len(unexpected_exchange_orders)
            + len(exchange_order_quantity_mismatches)
            + len(exchange_errors)
        )
        if missing_positions or missing_trades or exchange_errors:
            mismatch_ratio_pct = max(mismatch_ratio_pct, 1.0)
        status = "ok" if mismatch_count == 0 else "mismatch"
        details = {
            "missing_positions": missing_positions,
            "missing_trades": missing_trades,
            "quantity_mismatches": quantity_mismatches,
            "exchange_balance_mismatches": exchange_balance_mismatches,
            "exchange_state_checked": exchange_checked,
            "missing_exchange_orders": missing_exchange_orders,
            "unexpected_exchange_orders": unexpected_exchange_orders,
            "exchange_order_quantity_mismatches": exchange_order_quantity_mismatches,
            "exchange_order_state_checked": exchange_order_state_checked,
            "exchange_errors": exchange_errors,
            "mismatch_ratio_pct": mismatch_ratio_pct,
        }
        self.storage.insert_reconciliation_run(status, mismatch_count, details)
        return ReconciliationResult(
            status=status,
            mismatch_count=mismatch_count,
            mismatch_ratio_pct=mismatch_ratio_pct,
            details=details,
        )

    @staticmethod
    def _is_exchange_deficit(exchange_value: float, local_value: float) -> bool:
        local = float(local_value or 0.0)
        exchange = float(exchange_value or 0.0)
        if local <= 1e-8:
            return False
        return exchange + 1e-8 < local
