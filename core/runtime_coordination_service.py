"""Coordinate runtime refresh flows across learning, overrides, and pool rebuilds."""
from __future__ import annotations

from datetime import datetime


class RuntimeCoordinationService:
    """Own repeated runtime coordination sequences without changing strategy logic."""

    def __init__(
        self,
        *,
        observe_promoted_models,
        refresh_learning_runtime_overrides,
        apply_runtime_overrides,
        persist_runtime_settings_effective,
        refresh_runtime_learning_feedback,
        rebuild_execution_symbols,
    ):
        self.observe_promoted_models = observe_promoted_models
        self.refresh_learning_runtime_overrides = refresh_learning_runtime_overrides
        self.apply_runtime_overrides = apply_runtime_overrides
        self.persist_runtime_settings_effective = persist_runtime_settings_effective
        self.refresh_runtime_learning_feedback = refresh_runtime_learning_feedback
        self.rebuild_execution_symbols = rebuild_execution_symbols

    def prepare_cycle_runtime(self, now: datetime) -> None:
        self.observe_promoted_models(now)
        self.refresh_learning_runtime_overrides(now)
        self.apply_runtime_overrides()

    def persist_effective_runtime_state(self) -> None:
        self.persist_runtime_settings_effective()

    def refresh_after_paper_feedback(self, now: datetime, *, reason: str) -> None:
        self.refresh_runtime_learning_feedback(now, reason=reason)
        self.rebuild_execution_symbols(
            force=True,
            now=now,
            reason=reason,
        )
