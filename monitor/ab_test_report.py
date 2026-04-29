"""A/B test reporting for CryptoAI v3."""
from __future__ import annotations

from core.i18n import get_default_language, normalize_language, text_for
from core.storage import Storage


class ABTestReporter:
    """Aggregate challenger/champion comparison runs."""

    def __init__(self, storage: Storage):
        self.storage = storage

    def build(self, limit: int = 200) -> dict:
        with self.storage._conn() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM ab_test_runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]

        total = len(rows)
        challenger_shadow = [row for row in rows if row["selected_variant"] == "challenger_shadow"]
        challenger_live = [row for row in rows if row["selected_variant"] == "challenger_live"]
        agreement = [row for row in rows if row["selected_variant"] == "agreement"]
        avg_gap = (
            sum(abs(row["challenger_probability"] - row["champion_probability"]) for row in rows) / total
            if total
            else 0.0
        )
        return {
            "total_runs": total,
            "agreement_count": len(agreement),
            "challenger_shadow_count": len(challenger_shadow),
            "challenger_live_count": len(challenger_live),
            "avg_probability_gap": avg_gap,
            "latest_run": rows[0] if rows else None,
        }

    @staticmethod
    def render(report: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        latest = report.get("latest_run") or {}
        return "\n".join(
            [
                text_for(lang, "# A/B 测试报告", "# AB Test Report"),
                text_for(lang, f"- 总运行次数: {report['total_runs']}", f"- Total Runs: {report['total_runs']}"),
                text_for(lang, f"- 一致信号次数: {report['agreement_count']}", f"- Agreement Count: {report['agreement_count']}"),
                text_for(lang, f"- 挑战者影子次数: {report['challenger_shadow_count']}", f"- Challenger Shadow Count: {report['challenger_shadow_count']}"),
                text_for(lang, f"- 挑战者实盘次数: {report['challenger_live_count']}", f"- Challenger Live Count: {report['challenger_live_count']}"),
                text_for(lang, f"- 平均概率差: {report['avg_probability_gap']:.4f}", f"- Average Probability Gap: {report['avg_probability_gap']:.4f}"),
                text_for(lang, f"- 最近选中版本: {latest.get('selected_variant', 'none')}", f"- Latest Selected Variant: {latest.get('selected_variant', 'none')}"),
                text_for(lang, f"- 最近冠军模型: {latest.get('champion_model_version', 'none')}", f"- Latest Champion Model: {latest.get('champion_model_version', 'none')}"),
                text_for(lang, f"- 最近挑战者模型: {latest.get('challenger_model_version', 'none')}", f"- Latest Challenger Model: {latest.get('challenger_model_version', 'none')}"),
            ]
        )
