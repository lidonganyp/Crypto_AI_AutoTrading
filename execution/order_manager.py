"""Order state management for CryptoAI v3."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from core.models import OrderStatus
from core.storage import Storage


class OrderManager:
    """Create and update order records in a consistent way."""

    def __init__(self, storage: Storage):
        self.storage = storage

    def create(
        self,
        symbol: str,
        side: str,
        order_type: str,
        price: float,
        quantity: float,
        reason: str = "",
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        order_id = f"O-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.storage.insert_order(
            {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "status": OrderStatus.CREATED.value,
                "price": price,
                "quantity": quantity,
                "reason": reason,
                "created_at": now,
                "updated_at": now,
            }
        )
        return order_id

    def transition(self, order_id: str, status: OrderStatus, reason: str = ""):
        self.storage.update_order_status(order_id, status.value, reason=reason)
