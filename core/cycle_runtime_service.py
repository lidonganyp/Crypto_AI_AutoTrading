"""Cycle bookkeeping helpers for the runtime loop."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from config import Settings, get_settings
from core.storage import Storage


class CycleRuntimeService:
    """Persist and finalize runtime loop cycle state consistently."""

    def __init__(self, storage: Storage, settings: Settings | None = None):
        self.storage = storage
        self.settings = settings or get_settings()

    def start_cycle(
        self,
        now: datetime,
        active_symbols: list[str],
        shadow_symbols: list[str],
        circuit_breaker_active: bool,
    ) -> int:
        self.storage.set_state("last_cycle_started", now.isoformat())
        self.storage.set_state("last_cycle_status", "running")
        return self.storage.insert_cycle_run(
            {
                "started_at": now.isoformat(),
                "status": "running",
                "symbols": active_symbols,
                "opened_positions": 0,
                "closed_positions": 0,
                "circuit_breaker_active": circuit_breaker_active,
                "reconciliation_status": "pending",
                "notes": json.dumps({"shadow_symbols": shadow_symbols}, default=str),
            }
        )

    def fail_cycle(
        self,
        cycle_id: int,
        *,
        reconciliation_status: str,
        notes: str,
        opened_positions: int,
        closed_positions: int,
        circuit_breaker_active: bool,
    ) -> None:
        self.storage.set_state("last_cycle_status", "failed")
        self.storage.update_cycle_run(
            cycle_id,
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "opened_positions": opened_positions,
                "closed_positions": closed_positions,
                "circuit_breaker_active": circuit_breaker_active,
                "reconciliation_status": reconciliation_status,
                "notes": notes,
            },
        )

    def complete_cycle(
        self,
        cycle_id: int,
        *,
        final_cycle_status: str,
        reconciliation_status: str,
        notes: str,
        opened_positions: int,
        closed_positions: int,
        circuit_breaker_active: bool,
    ) -> None:
        self.storage.set_state(
            "last_cycle_completed",
            datetime.now(timezone.utc).isoformat(),
        )
        self.storage.set_state("last_cycle_status", final_cycle_status)
        self.storage.update_cycle_run(
            cycle_id,
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": final_cycle_status,
                "opened_positions": opened_positions,
                "closed_positions": closed_positions,
                "circuit_breaker_active": circuit_breaker_active,
                "reconciliation_status": reconciliation_status,
                "notes": notes,
            },
        )
