"""Paper Trader — 模拟盘执行引擎"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from loguru import logger

from core.models import OrderStatus, SignalDirection, Trade
from core.storage import Storage
from execution.order_manager import OrderManager
from typing import Callable, Any


class PaperTrader:
    """模拟盘交易执行"""

    ESTIMATED_FEE_PCT = 0.00075

    def __init__(self, storage: Storage, initial_balance: float = 10000.0):
        self.storage = storage
        self.initial_balance = initial_balance
        self.order_manager = OrderManager(storage)
        # 平仓回调列表：(callback_fn, ...) 平仓成功后依次调用
        self._on_close_callbacks: list[Callable[[dict], Any]] = []

    def register_on_close(self, callback: Callable[[dict], Any]):
        """注册平仓成功后的回调（用于更新心理/风控计数器等）"""
        self._on_close_callbacks.append(callback)

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
        direction: str,
        reference_price: float,
        fill_price: float,
        quantity: float,
    ) -> float:
        if reference_price <= 0 or quantity <= 0:
            return 0.0
        if str(direction).upper() == "LONG":
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
                    direction=direction,
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
        exit_price: float,
        reference_price: float,
        closed_qty: float,
        incremental_pnl: float,
        exit_time: str,
        reason: str,
        remaining_qty: float,
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
                    direction="SHORT",
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
                    "remaining_qty": remaining_qty,
                },
            }
        )

    def get_balance(self) -> float:
        """计算当前可用余额"""
        positions = self.storage.get_positions()
        used = sum(
            p["entry_price"] * p["quantity"] for p in positions
        )
        ledger_rows = self.storage.get_pnl_ledger(limit=5000)
        if ledger_rows:
            realized_total = sum(float(row.get("net_pnl") or 0.0) for row in ledger_rows)
        else:
            realized_total = sum(
                (trade.get("pnl") or 0.0)
                for trade in (
                    self.storage.get_closed_trades() + self.storage.get_open_trades()
                )
            )
        return self.initial_balance + realized_total - used

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
        """开仓

        Args:
            quantity: 指定下单数量（优先级最高）
            position_value: 指定下单金额（如未指定 quantity）
            若两者都未指定，则使用默认的 balance * 0.30
        """
        if direction != SignalDirection.LONG:
            logger.warning(f"Paper trader only supports LONG entries for {symbol}")
            return None

        if price <= 0:
            logger.error(f"Invalid price: {price}")
            return None

        positions = self.storage.get_positions()
        existing_position = next((p for p in positions if p["symbol"] == symbol), None)
        open_trades = self.storage.get_open_trades()
        existing_trade = next((t for t in open_trades if t["symbol"] == symbol), None)
        allow_position_add = self._position_add_allowed(
            existing_trade=existing_trade,
            metadata=metadata,
        )
        if existing_position is not None and not allow_position_add:
            logger.warning(f"Already have position for {symbol}")
            return None

        # 计算仓位
        balance = self.get_balance()
        if balance <= 0:
            logger.error("No available balance")
            return None

        # 确定下单数量
        if quantity is not None:
            if quantity <= 0:
                logger.error(f"Invalid quantity: {quantity}")
                return None
            position_value = quantity * price
            # 余额校验：即使指定了 quantity 也不能超过可用余额
            if position_value > balance:
                logger.warning(
                    f"Quantity {quantity} (${position_value:.2f}) exceeds balance ${balance:.2f}, capping"
                )
                position_value = balance
                quantity = position_value / price
        elif position_value is not None:
            if position_value <= 0:
                logger.error(f"Invalid position_value: {position_value}")
                return None
            if position_value > balance:
                logger.warning(
                    f"Position value ${position_value:.2f} exceeds balance ${balance:.2f}, capping"
                )
                position_value = balance
            quantity = position_value / price
        else:
            # 默认: 可用资金的 30%
            position_value = balance * 0.30
            quantity = position_value / price

        stop_loss = round(price * (1 - stop_loss_pct), 8)
        take_profit = round(price * (1 + take_profit_pct), 8)

        now = datetime.now(timezone.utc)
        order_id = self.order_manager.create(
            symbol=symbol,
            side=direction.value,
            order_type="LIMIT",
            price=price,
            quantity=quantity,
            reason=rationale,
        )
        self.order_manager.transition(order_id, OrderStatus.SUBMITTED, "paper submit")
        trade_metadata = self._merge_trade_metadata(existing_trade, metadata)
        if allow_position_add and existing_position is not None and existing_trade is not None:
            existing_qty = float(existing_position.get("quantity") or 0.0)
            existing_entry = float(existing_position.get("entry_price") or 0.0)
            total_qty = existing_qty + float(quantity)
            blended_entry = (
                ((existing_entry * existing_qty) + (float(price) * float(quantity))) / total_qty
                if total_qty > 0
                else float(price)
            )
            blended_stop = round(blended_entry * (1 - stop_loss_pct), 8)
            blended_target = round(blended_entry * (1 + take_profit_pct), 8)
            existing_initial_qty = float(
                existing_trade.get("initial_quantity")
                or existing_trade.get("quantity")
                or existing_qty
            )
            existing_confidence = float(existing_trade.get("confidence") or 0.0)
            blended_confidence = (
                ((existing_confidence * existing_qty) + (float(confidence) * float(quantity))) / total_qty
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
                initial_quantity=existing_initial_qty + float(quantity),
                rationale=merged_rationale,
                confidence=blended_confidence,
                metadata=trade_metadata,
            )
            self._record_open_ledger_entry(
                trade_id=str(existing_trade["id"]),
                symbol=symbol,
                direction=direction.value,
                reference_price=price,
                fill_price=price,
                quantity=quantity,
                event_time=now.isoformat(),
                metadata=trade_metadata,
            )
            self.storage.upsert_position({
                **existing_position,
                "symbol": symbol,
                "direction": direction.value,
                "entry_price": blended_entry,
                "quantity": total_qty,
                "entry_time": now.isoformat(),
                "stop_loss": blended_stop,
                "take_profit": blended_target,
            })
            trade_id = str(existing_trade["id"])
            stop_loss = blended_stop
            take_profit = blended_target
            logger.info(
                f"📈 ADD {symbol} {direction.value} @ {price:.2f} "
                f"qty={quantity:.6f} total_qty={total_qty:.6f}"
            )
        else:
            trade_id = (
                f"T-{now.strftime('%Y%m%d%H%M')}-"
                f"{symbol.replace('/', '')}-{uuid.uuid4().hex[:4]}"
            )
            self.storage.insert_trade({
                "id": trade_id,
                "symbol": symbol,
                "direction": direction.value,
                "entry_price": price,
                "quantity": quantity,
                "entry_time": now.isoformat(),
                "rationale": rationale,
                "confidence": confidence,
                "metadata": trade_metadata,
            })
            self._record_open_ledger_entry(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction.value,
                reference_price=price,
                fill_price=price,
                quantity=quantity,
                event_time=now.isoformat(),
                metadata=trade_metadata,
            )
            self.storage.upsert_position({
                "symbol": symbol,
                "direction": direction.value,
                "entry_price": price,
                "quantity": quantity,
                "entry_time": now.isoformat(),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            })
            logger.info(
                f"📈 OPEN {symbol} {direction.value} @ {price:.2f} "
                f"qty={quantity:.6f} stop={stop_loss:.2f} target={take_profit:.2f}"
            )

        self.storage.insert_execution_event(
            "open",
            symbol,
            {
                "order_id": order_id,
                "trade_id": trade_id,
                "direction": direction.value,
                "price": price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence,
                "model_id": str(trade_metadata.get("model_id") or ""),
                "position_adjustment": str(trade_metadata.get("position_adjustment") or ""),
                "allow_position_add": allow_position_add,
            },
        )
        self.order_manager.transition(order_id, OrderStatus.FILLED, "paper filled")

        return {
            "order_id": order_id,
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction.value,
            "price": price,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def execute_close(
        self,
        symbol: str,
        current_price: float,
        reason: str = "manual",
        close_qty: float | None = None,
    ) -> dict | None:
        """平仓（支持部分平仓）

        Args:
            close_qty: 平仓数量。None 表示全部平仓。
        """
        positions = self.storage.get_positions()
        pos = next((p for p in positions if p["symbol"] == symbol), None)

        if not pos:
            logger.warning(f"No position for {symbol}")
            return None

        entry_price = pos["entry_price"]
        total_qty = pos["quantity"]

        # 确定平仓数量
        if close_qty is None:
            qty_to_close = total_qty
        else:
            qty_to_close = min(close_qty, total_qty)
            if qty_to_close <= 0:
                logger.warning(f"Invalid close_qty={close_qty} for {symbol}")
                return None

        remaining_qty = total_qty - qty_to_close

        pnl = (current_price - entry_price) * qty_to_close
        pnl_pct = (current_price / entry_price - 1) * 100
        exit_time = datetime.now(timezone.utc).isoformat()

        # 查找对应的 open trade
        open_trades = self.storage.get_open_trades()
        trade = next(
            (t for t in open_trades if t["symbol"] == symbol), None
        )

        recorded_pnl = pnl
        recorded_pnl_pct = pnl_pct
        if trade:
            if remaining_qty <= 1e-10:
                # 全部平仓：将本次 PnL 与之前部分平仓累计的 PnL 合并
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
                    trade["id"], current_price,
                    exit_time,
                    total_pnl, total_pnl_pct,
                )
            else:
                # 部分平仓：记录部分成交，更新 trade 的 quantity 为剩余
                # (PnL 按本次平仓部分计算)
                self.storage.upsert_trade_partial_close(
                    trade["id"],
                    closed_qty=qty_to_close,
                    exit_price=current_price,
                    exit_time=exit_time,
                    realized_pnl=pnl,
                    realized_pnl_pct=pnl_pct,
                    remaining_qty=remaining_qty,
                )
            self._record_close_ledger_entry(
                trade=trade,
                symbol=symbol,
                exit_price=current_price,
                reference_price=current_price,
                closed_qty=qty_to_close,
                incremental_pnl=pnl,
                exit_time=exit_time,
                reason=reason,
                remaining_qty=remaining_qty,
            )

        # 更新或删除持仓
        if remaining_qty <= 1e-10:
            self.storage.delete_position(symbol)
            action = "FULL"
        else:
            self.storage.upsert_position({
                **pos,
                "quantity": remaining_qty,
            })
            action = f"PARTIAL ({qty_to_close:.6f}/{total_qty:.6f})"

        emoji = "✅" if pnl >= 0 else "❌"
        logger.info(
            f"{emoji} CLOSE {action} {symbol} @ {current_price:.2f} "
            f"PnL={recorded_pnl:+.2f} ({recorded_pnl_pct:+.2f}%) reason={reason}"
        )
        self.storage.insert_execution_event(
            "close",
            symbol,
            {
                "price": current_price,
                "pnl": recorded_pnl,
                "incremental_pnl": pnl,
                "pnl_pct": recorded_pnl_pct,
                "reason": reason,
                "closed_qty": qty_to_close,
                "remaining_qty": remaining_qty,
                "is_full_close": remaining_qty <= 1e-10,
                "fee_cost": max(current_price, 0.0) * max(qty_to_close, 0.0) * self.ESTIMATED_FEE_PCT,
            },
        )

        order_id = self.order_manager.create(
            symbol=symbol,
            side="CLOSE",
            order_type="MARKET",
            price=current_price,
            quantity=qty_to_close,
            reason=reason,
        )
        self.order_manager.transition(order_id, OrderStatus.SUBMITTED, "paper close submit")

        result = {
            "symbol": symbol,
            "exit_price": current_price,
            "pnl": recorded_pnl,
            "pnl_pct": recorded_pnl_pct,
            "reason": reason,
            "closed_qty": qty_to_close,
            "remaining_qty": remaining_qty,
            "is_full_close": remaining_qty <= 1e-10,
            "order_id": order_id,
        }

        # 触发所有平仓回调（仅全部平仓时触发，避免部分平仓时重复计次）
        if result["is_full_close"]:
            for cb in self._on_close_callbacks:
                try:
                    cb(result)
                except Exception as e:
                    logger.error(f"Close callback error: {e}")
            self.order_manager.transition(order_id, OrderStatus.FILLED, reason)
        else:
            self.order_manager.transition(order_id, OrderStatus.PARTIALLY_FILLED, reason)

        if trade:
            trade_metadata = self._trade_metadata(trade)
            result.update(
                {
                    "trade_id": trade["id"],
                    "entry_time": trade["entry_time"],
                    "entry_price": trade["entry_price"],
                    "confidence": trade.get("confidence", 0.0),
                    "rationale": trade.get("rationale", ""),
                    "metadata": trade_metadata,
                }
            )

        return result

    def check_stop_loss_take_profit(
        self, current_prices: dict[str, float]
    ) -> list[dict]:
        """检查所有持仓的止损止盈"""
        results = []
        positions = self.storage.get_positions()

        for pos in positions:
            symbol = pos["symbol"]
            price = current_prices.get(symbol)
            if not price:
                continue

            stop = pos.get("stop_loss")
            target = pos.get("take_profit")

            if stop and price <= stop:
                result = self.execute_close(symbol, price, "stop_loss")
                if result:
                    results.append(result)
            elif target and price >= target:
                # 分批止盈: 先平一半，保留一半
                half_qty = pos["quantity"] / 2
                result = self.execute_close(
                    symbol, price, "take_profit", close_qty=half_qty
                )
                if result:
                    results.append(result)
                    # 提高剩余持仓的止盈价（给更多空间）
                    if not result["is_full_close"]:
                        new_target = price * 1.10
                        self.storage.upsert_position({
                            **pos,
                            "quantity": result["remaining_qty"],
                            "take_profit": new_target,
                        })
                        logger.info(
                            f"  📈 Raised target for {symbol} "
                            f"to {new_target:.2f} (remaining half)"
                        )

        return results
