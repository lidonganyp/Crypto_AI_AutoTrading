"""Guard and alert aggregation for CryptoAI v3."""
from __future__ import annotations

import json
from collections import Counter

from core.i18n import get_default_language, normalize_language, text_for
from core.storage import Storage
from monitor.nextgen_live_summary import build_nextgen_live_summary


class GuardReporter:
    """Aggregate recent guardrails and alert-triggering execution events."""

    GUARDED_EVENT_TYPES = {
        "model_accuracy_guard",
        "model_degradation",
        "api_failure",
        "abnormal_move",
        "circuit_breaker",
        "manual_recovery_required",
        "reconciliation",
        "data_quality_failure",
        "market_latency_circuit_breaker",
        "market_latency_warning",
        "funding_rate_block",
        "bearish_news_block",
        "live_order_timeout",
        "live_open_limit_timeout",
        "live_close_limit_timeout",
        "nextgen_autonomy_live_kill_switch",
        "nextgen_autonomy_live_run_failed",
        "nextgen_autonomy_live_guard_callback_failed",
    }

    def __init__(self, storage: Storage):
        self.storage = storage

    def build(self, limit: int = 200) -> dict:
        with self.storage._conn() as conn:
            events = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM execution_events ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]
            latest_account = conn.execute(
                "SELECT * FROM account_snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

        guarded_events = [
            event for event in events if event["event_type"] in self.GUARDED_EVENT_TYPES
        ]
        event_counts = Counter(event["event_type"] for event in guarded_events)
        reason_counts = Counter()
        for event in guarded_events:
            payload = json.loads(event["payload_json"])
            reason = payload.get("error") or payload.get("reason") or event["event_type"]
            reason_counts[reason] += 1

        latest_guard_event = guarded_events[0] if guarded_events else None
        latest_guard_payload = (
            json.loads(latest_guard_event["payload_json"]) if latest_guard_event else {}
        )
        nextgen_live = build_nextgen_live_summary(self.storage)
        return {
            "guard_event_count": len(guarded_events),
            "event_counts": event_counts.most_common(10),
            "top_reasons": reason_counts.most_common(10),
            "latest_guard_event_type": latest_guard_event["event_type"] if latest_guard_event else "",
            "latest_guard_payload": latest_guard_payload,
            "latest_account": dict(latest_account) if latest_account else None,
            "nextgen_live": nextgen_live,
        }

    @staticmethod
    def render(report: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        latest_account = report.get("latest_account") or {}
        latest_payload = report.get("latest_guard_payload") or {}
        nextgen_live = report.get("nextgen_live") or {}
        lines = [
            text_for(lang, "# 风控告警报告", "# Guard Report"),
            text_for(lang, f"- 风控事件数: {report['guard_event_count']}", f"- Guard Events: {report['guard_event_count']}"),
            text_for(lang, f"- 最近风控事件: {report.get('latest_guard_event_type') or 'none'}", f"- Latest Guard Event: {report.get('latest_guard_event_type') or 'none'}"),
            text_for(lang, f"- 熔断状态: {bool(latest_account.get('circuit_breaker_active'))}", f"- Circuit Breaker Active: {bool(latest_account.get('circuit_breaker_active'))}"),
            text_for(lang, f"- 冷却截止时间: {latest_account.get('cooldown_until') or 'none'}", f"- Cooldown Until: {latest_account.get('cooldown_until') or 'none'}"),
            text_for(lang, f"- 最近告警原因: {latest_payload.get('reason') or 'none'}", f"- Latest Guard Reason: {latest_payload.get('reason') or 'none'}"),
            text_for(lang, f"- Nextgen Live 请求启用: {nextgen_live.get('requested_live', False)}", f"- Nextgen Live Requested: {nextgen_live.get('requested_live', False)}"),
            text_for(lang, f"- Nextgen Live 实际生效: {nextgen_live.get('effective_live', False)}", f"- Nextgen Live Effective: {nextgen_live.get('effective_live', False)}"),
            text_for(lang, f"- Nextgen Live 强制平仓: {nextgen_live.get('force_flatten', False)}", f"- Nextgen Live Force Flatten: {nextgen_live.get('force_flatten', False)}"),
            text_for(lang, f"- Nextgen Live Kill Switch: {nextgen_live.get('kill_switch_active', False)}", f"- Nextgen Live Kill Switch: {nextgen_live.get('kill_switch_active', False)}"),
            text_for(lang, f"- Nextgen Live 最近运行: {nextgen_live.get('last_run_status') or 'none'}", f"- Nextgen Live Latest Run: {nextgen_live.get('last_run_status') or 'none'}"),
            text_for(lang, f"- Nextgen Live 最近问题: {nextgen_live.get('latest_issue_event_type') or 'none'}", f"- Nextgen Live Latest Issue: {nextgen_live.get('latest_issue_event_type') or 'none'}"),
            text_for(lang, f"- Nextgen Live 被丢弃 Repair 数: {nextgen_live.get('repair_queue_dropped_count', 0)}", f"- Nextgen Live Dropped Repair Count: {nextgen_live.get('repair_queue_dropped_count', 0)}"),
            text_for(lang, f"- Nextgen Live Hold Repair 数: {nextgen_live.get('repair_queue_hold_priority_count', 0)}", f"- Nextgen Live Hold Repair Count: {nextgen_live.get('repair_queue_hold_priority_count', 0)}"),
            text_for(lang, f"- Nextgen Live 延后 Rebuild 数: {nextgen_live.get('repair_queue_postponed_rebuild_count', 0)}", f"- Nextgen Live Postponed Rebuild Count: {nextgen_live.get('repair_queue_postponed_rebuild_count', 0)}"),
            text_for(lang, f"- Nextgen Live 最近队列趋势: {nextgen_live.get('recent_repair_queue_summary') or '无'}", f"- Nextgen Live Recent Queue Trend: {nextgen_live.get('recent_repair_queue_summary') or 'none'}"),
            "",
            text_for(lang, "## 风控事件统计", "## Guard Event Counts"),
        ]
        if report["event_counts"]:
            for event_type, count in report["event_counts"]:
                lines.append(f"- {event_type}: {count}")
        else:
            lines.append(text_for(lang, "- 无", "- none"))

        lines.append("")
        lines.append(text_for(lang, "## 主要告警原因", "## Top Guard Reasons"))
        if report["top_reasons"]:
            for reason, count in report["top_reasons"]:
                lines.append(f"- {reason}: {count}")
        else:
            lines.append(text_for(lang, "- 无", "- none"))

        return "\n".join(lines)
