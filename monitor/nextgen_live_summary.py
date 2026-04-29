"""Shared summary helpers for nextgen live operator state."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from core.storage import Storage

_ISSUE_EVENT_TYPES = (
    "nextgen_autonomy_live_run_failed",
    "nextgen_autonomy_live_guard_callback_failed",
)


def build_nextgen_live_summary(storage: Storage) -> dict:
    operator_request = storage.get_json_state(
        "nextgen_autonomy_live_operator_request",
        {},
    ) or {}
    status = storage.get_json_state("nextgen_autonomy_live_status", {}) or {}
    if not isinstance(operator_request, dict):
        operator_request = {}
    if not isinstance(status, dict):
        status = {}

    with storage._conn() as conn:
        latest_run = conn.execute(
            """
            SELECT payload_json, created_at
            FROM execution_events
            WHERE event_type='nextgen_autonomy_live_run'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        recent_runs = conn.execute(
            """
            SELECT payload_json, created_at
            FROM execution_events
            WHERE event_type='nextgen_autonomy_live_run'
            ORDER BY id DESC
            LIMIT 5
            """
        ).fetchall()
        recent_issues = conn.execute(
            """
            SELECT event_type, payload_json, created_at
            FROM execution_events
            WHERE event_type IN (
                'nextgen_autonomy_live_run_failed',
                'nextgen_autonomy_live_guard_callback_failed'
            )
            ORDER BY id DESC
            LIMIT 20
            """
        ).fetchall()
        latest_issue = conn.execute(
            """
            SELECT event_type, payload_json, created_at
            FROM execution_events
            WHERE event_type IN (
                'nextgen_autonomy_live_run_failed',
                'nextgen_autonomy_live_guard_callback_failed'
            )
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    run_payload = _payload(latest_run["payload_json"]) if latest_run else {}
    recent_queue = build_recent_nextgen_live_queue_runs(
        recent_runs or [],
        recent_issues or [],
    )
    issue_payload = _payload(latest_issue["payload_json"]) if latest_issue else {}
    whitelist = operator_request.get("whitelist")
    if not isinstance(whitelist, list):
        whitelist = []
    reasons = status.get("reasons")
    if not isinstance(reasons, list):
        reasons = []
    return {
        "requested_live": bool(operator_request.get("requested_live", False)),
        "requested_reason": str(operator_request.get("reason") or ""),
        "requested_updated_at": str(operator_request.get("updated_at") or ""),
        "whitelist": [str(item) for item in whitelist if str(item).strip()],
        "max_active_runtimes": _as_int(operator_request.get("max_active_runtimes")),
        "effective_live": bool(status.get("effective_live", False)),
        "mode": "live" if bool(status.get("effective_live", False)) else "dry_run",
        "allow_entries": bool(status.get("allow_entries", False)),
        "allow_managed_closes": bool(status.get("allow_managed_closes", False)),
        "force_flatten": bool(status.get("force_flatten", False)),
        "kill_switch_active": bool(status.get("kill_switch_active", False)),
        "kill_switch_reason": str(status.get("kill_switch_reason") or ""),
        "reasons": [str(item) for item in reasons if str(item).strip()],
        "last_run_status": str(run_payload.get("status") or ""),
        "last_run_reason": str(run_payload.get("reason") or ""),
        "last_run_trigger": str(run_payload.get("trigger") or ""),
        "last_run_at": str(latest_run["created_at"] or "") if latest_run else "",
        "repair_queue_requested_size": _as_int(
            run_payload.get("repair_queue_requested_size")
        ) or 0,
        "repair_queue_dropped_count": _as_int(
            run_payload.get("repair_queue_dropped_count")
        ) or 0,
        "repair_queue_dropped_runtime_ids": [
            str(item).strip()
            for item in list(run_payload.get("repair_queue_dropped_runtime_ids") or [])
            if str(item).strip()
        ],
        "repair_queue_hold_priority_count": _as_int(
            run_payload.get("repair_queue_hold_priority_count")
        ) or 0,
        "repair_queue_postponed_rebuild_count": _as_int(
            run_payload.get("repair_queue_postponed_rebuild_count")
        ) or 0,
        "repair_queue_reprioritized_count": _as_int(
            run_payload.get("repair_queue_reprioritized_count")
        ) or 0,
        "repair_queue_hold_priority_active": bool(
            run_payload.get("repair_queue_hold_priority_active", False)
        ),
        "repair_queue_postponed_rebuild_active": bool(
            run_payload.get("repair_queue_postponed_rebuild_active", False)
        ),
        "repair_queue_dropped_active": bool(
            run_payload.get("repair_queue_dropped_active", False)
        ),
        "repair_queue_reprioritized_active": bool(
            run_payload.get("repair_queue_reprioritized_active", False)
        ),
        "recent_repair_queue_runs": recent_queue,
        "recent_repair_queue_summary": _compact_recent_queue_summary(recent_queue[:3]),
        "latest_issue_event_type": (
            str(latest_issue["event_type"] or "") if latest_issue else ""
        ),
        "latest_issue_reason": str(
            issue_payload.get("error")
            or issue_payload.get("reason")
            or ""
        ),
        "latest_issue_at": str(latest_issue["created_at"] or "") if latest_issue else "",
    }


def _payload(raw: str | None) -> dict:
    try:
        payload = json.loads(raw or "{}")
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _as_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_recent_nextgen_live_queue_runs(run_rows, issue_rows) -> list[dict]:
    parsed_issues = [
        _issue_row(row)
        for row in list(issue_rows or [])
    ]
    summary: list[dict] = []
    parsed_runs = [
        _run_row(row)
        for row in list(run_rows or [])
    ]
    for index, row in enumerate(parsed_runs):
        next_newer_run = parsed_runs[index - 1] if index > 0 else None
        latest_issue = _latest_issue_for_run(
            run=row,
            next_newer_run=next_newer_run,
            issues=parsed_issues,
        )
        payload = _payload(row["payload_json"])
        summary.append(
            {
                "created_at": row["created_at"],
                "age_minutes": _age_minutes(row["created_at"]),
                "autonomy_cycle_id": _as_int(payload.get("autonomy_cycle_id")),
                "status": str(payload.get("status") or ""),
                "trigger": str(payload.get("trigger") or ""),
                "reason": str(payload.get("reason") or ""),
                "requested_size": _as_int(payload.get("repair_queue_requested_size")) or 0,
                "dropped_count": _as_int(payload.get("repair_queue_dropped_count")) or 0,
                "hold_priority_count": _as_int(
                    payload.get("repair_queue_hold_priority_count")
                )
                or 0,
                "postponed_rebuild_count": _as_int(
                    payload.get("repair_queue_postponed_rebuild_count")
                )
                or 0,
                "reprioritized_count": _as_int(
                    payload.get("repair_queue_reprioritized_count")
                )
                or 0,
                "latest_issue_event_type": (
                    str(latest_issue.get("event_type") or "")
                    if latest_issue is not None
                    else ""
                ),
                "latest_issue_reason": (
                    str(latest_issue.get("reason") or "")
                    if latest_issue is not None
                    else ""
                ),
            }
        )
    return summary


def _compact_recent_queue_summary(items: list[dict]) -> str:
    parts: list[str] = []
    for item in items:
        age_minutes = int(item.get("age_minutes") or 0)
        parts.append(
            f"{age_minutes}m:h{int(item.get('hold_priority_count') or 0)}"
            f"/r{int(item.get('postponed_rebuild_count') or 0)}"
            f"/q{int(item.get('reprioritized_count') or 0)}"
        )
    return " | ".join(parts)


def _age_minutes(created_at: str) -> int:
    parsed = _parse_time(created_at)
    if parsed is None:
        return 0
    return max(
        0,
        int((datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() // 60),
    )


def _parse_time(created_at: str) -> datetime | None:
    text = str(created_at or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _run_row(row) -> dict:
    return {
        "payload_json": row["payload_json"],
        "created_at": str(row["created_at"] or ""),
        "_created_at": _parse_time(row["created_at"]),
    }


def _issue_row(row) -> dict:
    payload = _payload(row["payload_json"])
    return {
        "event_type": str(row["event_type"] or ""),
        "created_at": str(row["created_at"] or ""),
        "_created_at": _parse_time(row["created_at"]),
        "reason": str(payload.get("error") or payload.get("reason") or ""),
    }


def _latest_issue_for_run(
    *,
    run: dict,
    next_newer_run: dict | None,
    issues: list[dict],
) -> dict | None:
    run_created_at = run.get("_created_at")
    newer_created_at = (
        next_newer_run.get("_created_at")
        if next_newer_run is not None
        else None
    )
    if run_created_at is None:
        return None
    for issue in issues:
        issue_created_at = issue.get("_created_at")
        if issue_created_at is None:
            continue
        if issue_created_at < run_created_at:
            continue
        if newer_created_at is not None and issue_created_at >= newer_created_at:
            continue
        if str(issue.get("event_type") or "") not in _ISSUE_EVENT_TYPES:
            continue
        return issue
    return None
