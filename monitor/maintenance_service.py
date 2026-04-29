"""Maintenance and retention helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config import MaintenanceSettings
from core.storage import Storage


class MaintenanceService:
    """Delete expired operational data according to retention policy."""

    def __init__(self, storage: Storage, settings: MaintenanceSettings):
        self.storage = storage
        self.settings = settings

    def run(self) -> dict[str, int]:
        now = datetime.now(timezone.utc)
        summary = {
            "feature_snapshots": self.storage.delete_older_than(
                "feature_snapshots",
                "created_at",
                (now - timedelta(days=self.settings.retain_feature_days)).isoformat(),
            ),
            "prediction_runs": self.storage.delete_older_than(
                "prediction_runs",
                "created_at",
                (now - timedelta(days=self.settings.retain_prediction_days)).isoformat(),
            ),
            "prediction_evaluations": self.storage.delete_older_than(
                "prediction_evaluations",
                "created_at",
                (now - timedelta(days=self.settings.retain_prediction_days)).isoformat(),
            ),
            "shadow_trade_runs": self.storage.delete_older_than(
                "shadow_trade_runs",
                "created_at",
                (now - timedelta(days=self.settings.retain_prediction_days)).isoformat(),
            ),
            "account_snapshots": self.storage.delete_older_than(
                "account_snapshots",
                "created_at",
                (now - timedelta(days=self.settings.retain_account_days)).isoformat(),
            ),
            "execution_events": self.storage.delete_older_than(
                "execution_events",
                "created_at",
                (now - timedelta(days=self.settings.retain_execution_days)).isoformat(),
            ),
            "scheduler_runs": self.storage.delete_older_than(
                "scheduler_runs",
                "created_at",
                (now - timedelta(days=self.settings.retain_scheduler_days)).isoformat(),
            ),
            "report_artifacts": self.storage.delete_older_than(
                "report_artifacts",
                "created_at",
                (now - timedelta(days=self.settings.retain_report_days)).isoformat(),
            ),
        }
        return summary
