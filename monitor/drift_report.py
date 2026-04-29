"""Backtest / walk-forward / runtime drift comparison report."""
from __future__ import annotations

import json
from core.i18n import get_default_language, normalize_language, text_for
from core.report_metrics import parse_markdown_metrics
from core.storage import Storage


class DriftReporter:
    """Compare historical expectation with recent live/runtime behavior."""

    def __init__(self, storage: Storage):
        self.storage = storage

    def build(self) -> dict:
        with self.storage._conn() as conn:
            latest_backtest = conn.execute(
                "SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            latest_walkforward = conn.execute(
                "SELECT * FROM walkforward_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            latest_performance = conn.execute(
                "SELECT * FROM report_artifacts WHERE report_type='performance' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            first_account = conn.execute(
                "SELECT equity FROM account_snapshots ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            latest_account = conn.execute(
                "SELECT equity FROM account_snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

        backtest_summary = (
            json.loads(latest_backtest["summary_json"]) if latest_backtest else {}
        )
        walkforward_summary = (
            json.loads(latest_walkforward["summary_json"]) if latest_walkforward else {}
        )
        performance_metrics = (
            parse_markdown_metrics(latest_performance["content"])
            if latest_performance
            else {}
        )

        expected_return = float(
            walkforward_summary.get(
                "total_return_pct",
                backtest_summary.get("total_return_pct", 0.0),
            )
        )
        runtime_win_rate = self._parse_pct(performance_metrics.get("Win Rate", "0"))
        runtime_accuracy = self._parse_pct(
            performance_metrics.get("XGBoost Direction Accuracy", "0")
        )
        walkforward_win_rate = float(walkforward_summary.get("avg_win_rate", 0.0))
        backtest_win_rate = float(backtest_summary.get("win_rate", 0.0))
        runtime_return_pct = 0.0
        if first_account and latest_account:
            first_equity = float(first_account["equity"] or 0.0)
            latest_equity = float(latest_account["equity"] or 0.0)
            if first_equity > 0:
                runtime_return_pct = (latest_equity / first_equity - 1.0) * 100
        drift_return_gap = expected_return - runtime_return_pct
        drift_win_rate_gap = max(walkforward_win_rate, backtest_win_rate) - runtime_win_rate
        drift_accuracy_gap = float(performance_metrics.get("Latest Holdout Accuracy", "0").replace("%", "") or 0.0) - runtime_accuracy

        severity = "low"
        if abs(drift_return_gap) > 20 or abs(drift_win_rate_gap) > 20 or abs(drift_accuracy_gap) > 20:
            severity = "high"
        elif abs(drift_return_gap) > 10 or abs(drift_win_rate_gap) > 10 or abs(drift_accuracy_gap) > 10:
            severity = "medium"

        return {
            "backtest_summary": backtest_summary,
            "walkforward_summary": walkforward_summary,
            "runtime_metrics": performance_metrics,
            "runtime_return_pct": runtime_return_pct,
            "drift_return_gap": drift_return_gap,
            "drift_win_rate_gap": drift_win_rate_gap,
            "drift_accuracy_gap": drift_accuracy_gap,
            "severity": severity,
        }

    @staticmethod
    def render(report: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        return "\n".join(
            [
                text_for(lang, "# 漂移报告", "# Drift Report"),
                text_for(lang, f"- 严重程度: {report['severity']}", f"- Severity: {report['severity']}"),
                text_for(lang, f"- 收益漂移差: {report['drift_return_gap']:.2f}", f"- Drift Return Gap: {report['drift_return_gap']:.2f}"),
                text_for(lang, f"- 运行时收益率: {report['runtime_return_pct']:.2f}%", f"- Runtime Return: {report['runtime_return_pct']:.2f}%"),
                text_for(lang, f"- 胜率漂移差: {report['drift_win_rate_gap']:.2f}%", f"- Drift Win Rate Gap: {report['drift_win_rate_gap']:.2f}%"),
                text_for(lang, f"- 准确率漂移差: {report['drift_accuracy_gap']:.2f}%", f"- Drift Accuracy Gap: {report['drift_accuracy_gap']:.2f}%"),
                "",
                text_for(lang, "## 回测摘要", "## Backtest Summary"),
                text_for(lang, f"- 总收益率: {report['backtest_summary'].get('total_return_pct', 0.0):.2f}%", f"- Total Return: {report['backtest_summary'].get('total_return_pct', 0.0):.2f}%"),
                text_for(lang, f"- 胜率: {report['backtest_summary'].get('win_rate', 0.0):.2f}%", f"- Win Rate: {report['backtest_summary'].get('win_rate', 0.0):.2f}%"),
                "",
                text_for(lang, "## Walk-Forward 摘要", "## Walk-Forward Summary"),
                text_for(lang, f"- 总收益率: {report['walkforward_summary'].get('total_return_pct', 0.0):.2f}%", f"- Total Return: {report['walkforward_summary'].get('total_return_pct', 0.0):.2f}%"),
                text_for(lang, f"- 平均胜率: {report['walkforward_summary'].get('avg_win_rate', 0.0):.2f}%", f"- Avg Win Rate: {report['walkforward_summary'].get('avg_win_rate', 0.0):.2f}%"),
                "",
                text_for(lang, "## 运行时指标", "## Runtime Metrics"),
                text_for(lang, f"- 胜率: {report['runtime_metrics'].get('Win Rate', '0%')}", f"- Win Rate: {report['runtime_metrics'].get('Win Rate', '0%')}"),
                text_for(lang, f"- XGBoost 方向准确率: {report['runtime_metrics'].get('XGBoost Direction Accuracy', '0%')}", f"- XGBoost Direction Accuracy: {report['runtime_metrics'].get('XGBoost Direction Accuracy', '0%')}"),
                text_for(lang, f"- 融合信号准确率: {report['runtime_metrics'].get('Fusion Signal Accuracy', '0%')}", f"- Fusion Signal Accuracy: {report['runtime_metrics'].get('Fusion Signal Accuracy', '0%')}"),
            ]
        )
    @staticmethod
    def _parse_pct(value: str) -> float:
        return float((value or "0").replace("%", "").replace(",", "").strip() or 0.0)
