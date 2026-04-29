"""Runtime helpers for report rendering and artifact persistence."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class ReportRuntimeService:
    """Own report rendering, artifact persistence, and report command entrypoints."""

    def __init__(
        self,
        storage,
        report_dir: Path,
        *,
        current_language,
        reflector=None,
        health=None,
        performance=None,
        maintenance=None,
        guard_runtime=None,
        guards=None,
        ab_tests=None,
        drift=None,
        failures=None,
        incidents=None,
        ops=None,
        pool_attribution=None,
        alpha_diagnostics=None,
        validation=None,
        daily_focus=None,
        backtest_live_consistency=None,
        evaluate_live_readiness=None,
        get_execution_symbols=None,
    ):
        self.storage = storage
        self.report_dir = Path(report_dir)
        self.current_language = current_language
        self.reflector = reflector
        self.health = health
        self.performance = performance
        self.maintenance = maintenance
        self.guard_runtime = guard_runtime
        self.guards = guards
        self.ab_tests = ab_tests
        self.drift = drift
        self.failures = failures
        self.incidents = incidents
        self.ops = ops
        self.pool_attribution = pool_attribution
        self.alpha_diagnostics = alpha_diagnostics
        self.validation = validation
        self.daily_focus = daily_focus
        self.backtest_live_consistency = backtest_live_consistency
        self.evaluate_live_readiness = evaluate_live_readiness
        self.get_execution_symbols = get_execution_symbols or (lambda: [])

    def render_with_language(self, renderer, *args):
        try:
            return renderer(*args, lang=self.current_language())
        except TypeError:
            return renderer(*args)

    def save_report_artifact(
        self,
        report_type: str,
        symbol: str | None,
        content: str,
        extension: str,
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        symbol_suffix = f"_{symbol.replace('/', '_')}" if symbol else ""
        path = self.report_dir / f"{report_type}{symbol_suffix}_{stamp}.{extension}"
        path.write_text(content, encoding="utf-8")
        self.storage.insert_report_artifact(report_type, content, symbol=symbol)

    def record_rendered_report(
        self,
        *,
        report_type: str,
        symbol: str | None,
        renderer,
        payload,
        extension: str = "md",
    ) -> str:
        report = self.render_with_language(renderer, payload)
        self.save_report_artifact(report_type, symbol, report, extension)
        return report

    @staticmethod
    def _attach_report(result, report: str) -> dict:
        if isinstance(result, dict):
            return {**result, "report": report}
        if hasattr(result, "__dict__"):
            return {**result.__dict__, "report": report}
        return {"result": result, "report": report}

    def generate_reports(self) -> dict[str, str]:
        reports = {
            "daily": self.reflector.generate_daily_summary(),
            "weekly": self.reflector.generate_weekly_review(),
        }
        if self.daily_focus is not None:
            focus_payload = self.daily_focus.build(self.get_execution_symbols())
            reports["daily_focus"] = self.render_with_language(
                self.daily_focus.render,
                focus_payload,
            )
        if self.backtest_live_consistency is not None:
            consistency_payload = self.backtest_live_consistency.build(
                self.get_execution_symbols()
            )
            reports["backtest_live_consistency"] = self.render_with_language(
                self.backtest_live_consistency.render,
                consistency_payload,
            )
        self.save_report_artifact("daily", None, reports["daily"], "md")
        self.save_report_artifact("weekly", None, reports["weekly"], "md")
        if "daily_focus" in reports:
            self.save_report_artifact("daily_focus", None, reports["daily_focus"], "md")
        if "backtest_live_consistency" in reports:
            self.save_report_artifact(
                "backtest_live_consistency",
                None,
                reports["backtest_live_consistency"],
                "md",
            )
        return reports

    def run_health_check(self) -> dict:
        status = self.health.run()
        report = self.record_rendered_report(
            report_type="health",
            symbol=None,
            renderer=self.health.render_report,
            payload=status,
        )
        return self._attach_report(status, report)

    def run_metrics(self) -> dict:
        snapshot = self.performance.build()
        report = self.record_rendered_report(
            report_type="performance",
            symbol=None,
            renderer=self.performance.render,
            payload=snapshot,
        )
        return self._attach_report(snapshot, report)

    def run_live_readiness_check(self) -> dict:
        readiness = self.evaluate_live_readiness()
        report = self.guard_runtime.render_live_readiness_report(
            {
                "ready": readiness.ready,
                "reasons": readiness.reasons,
                "metrics": readiness.metrics,
            }
        )
        self.save_report_artifact("live_readiness", None, report, "md")
        return {
            "ready": readiness.ready,
            "reasons": readiness.reasons,
            "metrics": readiness.metrics,
            "report": report,
        }

    def run_guard_report(self) -> dict:
        result = self.guards.build()
        report = self.record_rendered_report(
            report_type="guard",
            symbol=None,
            renderer=self.guards.render,
            payload=result,
        )
        return self._attach_report(result, report)

    def run_ab_test_report(self) -> dict:
        result = self.ab_tests.build()
        report = self.record_rendered_report(
            report_type="ab_test",
            symbol=None,
            renderer=self.ab_tests.render,
            payload=result,
        )
        return self._attach_report(result, report)

    def run_drift_report(self) -> dict:
        result = self.drift.build()
        report = self.record_rendered_report(
            report_type="drift",
            symbol=None,
            renderer=self.drift.render,
            payload=result,
        )
        return self._attach_report(result, report)

    def run_maintenance(self) -> dict:
        result = self.maintenance.run()
        report = "\n".join(f"{key}: {value}" for key, value in result.items())
        self.save_report_artifact("maintenance", None, report, "md")
        return {"summary": result, "report": report}

    def run_failure_report(self) -> dict:
        result = self.failures.build()
        report = self.record_rendered_report(
            report_type="failure",
            symbol=None,
            renderer=self.failures.render,
            payload=result,
        )
        return self._attach_report(result, report)

    def run_incident_report(self) -> dict:
        result = self.incidents.build()
        report = self.record_rendered_report(
            report_type="incident",
            symbol=None,
            renderer=self.incidents.render,
            payload=result,
        )
        return self._attach_report(result, report)

    def run_ops_overview(self) -> dict:
        result = self.ops.build()
        report = self.record_rendered_report(
            report_type="ops_overview",
            symbol=None,
            renderer=self.ops.render,
            payload=result,
        )
        return self._attach_report(result, report)

    def run_pool_attribution_report(self, symbols: list[str] | None = None) -> dict:
        selected_symbols = symbols or self.get_execution_symbols()
        result = self.pool_attribution.build(selected_symbols)
        report = self.record_rendered_report(
            report_type="pool_attribution",
            symbol=None,
            renderer=self.pool_attribution.render,
            payload=result,
        )
        return self._attach_report(result, report)

    def run_alpha_diagnostics_report(self, symbols: list[str] | None = None) -> dict:
        selected_symbols = symbols or self.get_execution_symbols()
        result = self.alpha_diagnostics.build(selected_symbols)
        report = self.record_rendered_report(
            report_type="alpha_diagnostics",
            symbol=None,
            renderer=self.alpha_diagnostics.render,
            payload=result,
        )
        return self._attach_report(result, report)

    def run_validation_sprint(self, symbols: list[str] | None = None) -> dict:
        selected_symbols = symbols or self.get_execution_symbols()
        result = self.validation.run(selected_symbols)
        report = self.record_rendered_report(
            report_type="validation_sprint",
            symbol=None,
            renderer=self.validation.render,
            payload=result,
        )
        return self._attach_report(result, report)

    def run_daily_focus_report(self, symbols: list[str] | None = None) -> dict:
        selected_symbols = symbols or self.get_execution_symbols()
        result = self.daily_focus.build(selected_symbols)
        report = self.record_rendered_report(
            report_type="daily_focus",
            symbol=None,
            renderer=self.daily_focus.render,
            payload=result,
        )
        return self._attach_report(result, report)

    def run_backtest_live_consistency_report(
        self,
        symbols: list[str] | None = None,
    ) -> dict:
        selected_symbols = symbols or self.get_execution_symbols()
        result = self.backtest_live_consistency.build(selected_symbols)
        report = self.record_rendered_report(
            report_type="backtest_live_consistency",
            symbol=None,
            renderer=self.backtest_live_consistency.render,
            payload=result,
        )
        return self._attach_report(result, report)
