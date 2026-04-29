"""Incident aggregation for CryptoAI v3."""
from __future__ import annotations

from collections import Counter

from core.i18n import get_default_language, normalize_language, text_for
from core.storage import Storage


class IncidentReporter:
    """Aggregate recent operational incidents from stored artifacts."""

    def __init__(self, storage: Storage):
        self.storage = storage

    def build(self, limit: int = 100) -> dict:
        with self.storage._conn() as conn:
            cycles = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM cycle_runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]
            schedulers = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM scheduler_runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]
            reconciliations = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM reconciliation_runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]

        failed_cycles = [cycle for cycle in cycles if cycle["status"] != "ok"]
        failed_scheduler = [run for run in schedulers if run["status"] != "ok"]
        mismatch_reconciliations = [
            item for item in reconciliations if item["status"] != "ok"
        ]

        cycle_causes = Counter(
            (cycle.get("notes") or cycle.get("reconciliation_status") or "unknown")
            for cycle in failed_cycles
        )
        scheduler_causes = Counter(
            (run.get("output") or run.get("status") or "unknown")
            for run in failed_scheduler
        )

        return {
            "failed_cycles": len(failed_cycles),
            "failed_scheduler_runs": len(failed_scheduler),
            "reconciliation_mismatches": len(mismatch_reconciliations),
            "top_cycle_causes": cycle_causes.most_common(10),
            "top_scheduler_causes": scheduler_causes.most_common(10),
        }

    @staticmethod
    def render(report: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        lines = [
            text_for(lang, "# 事故报告", "# Incident Report"),
            text_for(lang, f"- 失败周期数: {report['failed_cycles']}", f"- Failed Cycles: {report['failed_cycles']}"),
            text_for(lang, f"- 失败调度次数: {report['failed_scheduler_runs']}", f"- Failed Scheduler Runs: {report['failed_scheduler_runs']}"),
            text_for(lang, f"- 对账不一致次数: {report['reconciliation_mismatches']}", f"- Reconciliation Mismatches: {report['reconciliation_mismatches']}"),
            "",
            text_for(lang, "## 主要周期故障原因", "## Top Cycle Causes"),
        ]
        if report["top_cycle_causes"]:
            for cause, count in report["top_cycle_causes"]:
                lines.append(f"- {cause}: {count}")
        else:
            lines.append(text_for(lang, "- 无", "- none"))

        lines.append("")
        lines.append(text_for(lang, "## 主要调度故障原因", "## Top Scheduler Causes"))
        if report["top_scheduler_causes"]:
            for cause, count in report["top_scheduler_causes"]:
                lines.append(f"- {cause}: {count}")
        else:
            lines.append(text_for(lang, "- 无", "- none"))

        return "\n".join(lines)
