"""Cycle failure aggregation and reporting."""
from __future__ import annotations

import json
from collections import Counter

from core.i18n import get_default_language, normalize_language, text_for
from core.storage import Storage


class FailureReporter:
    """Aggregate recent failure reasons from cycle and execution history."""

    FAILURE_EVENT_TYPES = {
        "api_failure",
        "data_quality_failure",
        "market_latency_circuit_breaker",
        "live_order_timeout",
        "live_open_limit_timeout",
        "live_close_limit_timeout",
        "live_open_rejected",
        "live_close_rejected",
        "reconciliation",
        "circuit_breaker",
        "manual_recovery_required",
    }

    def __init__(self, storage: Storage):
        self.storage = storage

    def build(self, limit: int = 50) -> dict:
        with self.storage._conn() as conn:
            cycles = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM cycle_runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]
            events = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM execution_events ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]

        failed_cycles = [cycle for cycle in cycles if cycle["status"] != "ok"]
        cycle_reasons = Counter(
            cycle.get("notes") or cycle.get("reconciliation_status") or "unknown"
            for cycle in failed_cycles
        )
        event_reasons = Counter()
        for event in events:
            if event["event_type"] not in self.FAILURE_EVENT_TYPES:
                continue
            payload = json.loads(event["payload_json"])
            reason = payload.get("reason") or event["event_type"]
            event_reasons[reason] += 1

        return {
            "failed_cycle_count": len(failed_cycles),
            "top_cycle_failures": cycle_reasons.most_common(10),
            "top_execution_failures": event_reasons.most_common(10),
        }

    @staticmethod
    def render(report: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        lines = [
            text_for(lang, "# 故障报告", "# Failure Report"),
            text_for(lang, f"- 失败周期数: {report['failed_cycle_count']}", f"- Failed Cycles: {report['failed_cycle_count']}"),
            "",
            text_for(lang, "## 主要周期失败原因", "## Top Cycle Failures"),
        ]
        if report["top_cycle_failures"]:
            for reason, count in report["top_cycle_failures"]:
                lines.append(f"- {reason}: {count}")
        else:
            lines.append(text_for(lang, "- 无", "- none"))

        lines.append("")
        lines.append(text_for(lang, "## 主要执行失败原因", "## Top Execution Failures"))
        if report["top_execution_failures"]:
            for reason, count in report["top_execution_failures"]:
                lines.append(f"- {reason}: {count}")
        else:
            lines.append(text_for(lang, "- 无", "- none"))
        return "\n".join(lines)
