"""Dashboard database and state access helpers."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import get_settings
from core.report_metrics import parse_markdown_metrics as parse_report_metrics
from core.runtime_state_view import RuntimeStateSnapshot, RuntimeStateView


def resolve_dashboard_db_path(settings=None) -> str:
    settings = settings or get_settings()
    path = Path(settings.app.db_path)
    if path.is_absolute():
        return str(path)
    return str(Path(settings.app.project_root) / path)


def query_df(sql: str, params: tuple = (), db_path: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path or resolve_dashboard_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return pd.DataFrame([dict(row) for row in rows]) if rows else pd.DataFrame()
    finally:
        conn.close()


def query_one(sql: str, params: tuple = (), db_path: str | None = None) -> dict | None:
    conn = sqlite3.connect(db_path or resolve_dashboard_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def execute_sql(sql: str, params: tuple = (), db_path: str | None = None) -> None:
    conn = sqlite3.connect(db_path or resolve_dashboard_db_path(), check_same_thread=False)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def load_json(value: str | None, default=None):
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def get_state_row(key: str, db_path: str | None = None) -> dict | None:
    return query_one(
        "SELECT key, value, updated_at FROM system_state WHERE key=?",
        (key,),
        db_path=db_path,
    )


def get_state_json(key: str, default=None, db_path: str | None = None):
    row = get_state_row(key, db_path=db_path)
    if not row:
        return default
    return load_json(row.get("value"), default)


def set_state_json(key: str, value, db_path: str | None = None) -> None:
    execute_sql(
        """INSERT OR REPLACE INTO system_state (key, value, updated_at)
           VALUES (?, ?, ?)""",
        (key, json.dumps(value, default=str), datetime.now(timezone.utc).isoformat()),
        db_path=db_path,
    )


class DashboardRuntimeStateView:
    """Dashboard-friendly runtime state access with one-shot normalization."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or resolve_dashboard_db_path()

    def _storage(self):
        from core.storage import Storage

        return Storage(self.db_path)

    def snapshot(self) -> RuntimeStateSnapshot:
        return RuntimeStateView(self._storage()).snapshot()


def parse_report_history(
    report_type: str,
    limit: int = 30,
    db_path: str | None = None,
) -> pd.DataFrame:
    reports = query_df(
        "SELECT created_at, content FROM report_artifacts WHERE report_type=? ORDER BY created_at DESC LIMIT ?",
        (report_type, limit),
        db_path=db_path,
    )
    if reports.empty:
        return pd.DataFrame()
    rows = []
    for _, row in reports.iterrows():
        metrics = parse_report_metrics(row["content"])
        metrics["created_at"] = row["created_at"]
        rows.append(metrics)
    history = pd.DataFrame(rows)
    if history.empty:
        return history
    history["created_at"] = pd.to_datetime(history["created_at"], errors="coerce")
    return history.sort_values("created_at")
