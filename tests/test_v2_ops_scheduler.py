import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import get_settings
from monitor.guard_report import GuardReporter
from monitor.health_check import HealthChecker
from monitor.notifier import (
    CriticalFeishuWebhookChannel,
    CriticalWebhookChannel,
    FeishuWebhookChannel,
    Notifier,
)
from monitor.ops_overview import OpsOverviewService
from monitor.scheduler_service import SchedulerService
from tests.v2_architecture_support import V2ArchitectureTestCase


class V2OpsSchedulerTests(V2ArchitectureTestCase):
    def test_health_checker_reads_market_latency_state(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.data_latency_warning_seconds = 3
        settings.exchange.data_latency_circuit_breaker_seconds = 5
        self.storage.set_state("latest_market_latency_seconds", "6.500")
        checker = HealthChecker(self.storage, settings)
        status = checker.run()
        self.assertEqual(status.market_latency_status, "degraded")

    def test_guard_reporter_includes_model_degradation(self):
        self.storage.insert_execution_event(
            "model_degradation",
            "SYSTEM",
            {"reason": "live_accuracy_below_disable_floor"},
        )
        report_data = GuardReporter(self.storage).build()
        report_text = GuardReporter.render(report_data)
        self.assertGreaterEqual(report_data["guard_event_count"], 1)
        self.assertIn("model_degradation", report_text)

    def test_guard_reporter_includes_operational_guard_events(self):
        self.storage.insert_execution_event(
            "data_quality_failure",
            "BTC/USDT",
            {"reason": "missing_ratio_too_high"},
        )
        self.storage.insert_execution_event(
            "manual_recovery_required",
            "SYSTEM",
            {"reason": "market_data_latency"},
        )
        report_data = GuardReporter(self.storage).build()
        event_counts = dict(report_data["event_counts"])
        self.assertIn("data_quality_failure", event_counts)
        self.assertIn("manual_recovery_required", event_counts)

    def test_notifier_routes_only_error_and_critical_to_critical_channel(self):
        delivered = []

        class CaptureChannel:
            name = "capture"

            def accepts(self, level: str) -> bool:
                return True

            def send(self, event_type, title, body, level="info"):
                delivered.append((event_type, level))

        class CaptureCritical(CriticalWebhookChannel):
            def __init__(self):
                pass

            def send(self, event_type, title, body, level="info"):
                delivered.append((f"critical:{event_type}", level))

        notifier = Notifier(self.storage)
        notifier.add_channel(CaptureChannel())
        notifier.add_channel(CaptureCritical())
        notifier.notify("info_evt", "t", "b", level="info")
        notifier.notify("err_evt", "t", "b", level="error")
        notifier.notify("crit_evt", "t", "b", level="critical")
        self.assertIn(("info_evt", "info"), delivered)
        self.assertIn(("err_evt", "error"), delivered)
        self.assertIn(("crit_evt", "critical"), delivered)
        self.assertNotIn(("critical:info_evt", "info"), delivered)
        self.assertIn(("critical:err_evt", "error"), delivered)
        self.assertIn(("critical:crit_evt", "critical"), delivered)

    def test_feishu_webhook_channel_builds_signed_payload(self):
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

        def fake_post(url, json=None, timeout=10):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("requests.post", side_effect=fake_post):
            channel = FeishuWebhookChannel("https://open.feishu.cn/fake", "secret")
            channel.send("evt", "Title", "Body", "warning")

        self.assertEqual(captured["url"], "https://open.feishu.cn/fake")
        self.assertEqual(captured["json"]["msg_type"], "text")
        self.assertIn("[WARNING] Title", captured["json"]["content"]["text"])
        self.assertIn("timestamp", captured["json"])
        self.assertIn("sign", captured["json"])

    def test_critical_feishu_channel_filters_info(self):
        channel = CriticalFeishuWebhookChannel("https://open.feishu.cn/fake", "")
        self.assertFalse(channel.accepts("info"))
        self.assertTrue(channel.accepts("error"))
        self.assertTrue(channel.accepts("critical"))

    def test_ops_overview_service_renders_snapshot(self):
        self.storage.insert_cycle_run(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "ok",
                "symbols": ["BTC/USDT"],
                "opened_positions": 1,
                "closed_positions": 0,
                "circuit_breaker_active": False,
                "reconciliation_status": "ok",
                "notes": "",
            }
        )
        self.storage.insert_scheduler_run(
            {
                "job_name": "health",
                "status": "ok",
                "output": "",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        overview = OpsOverviewService(self.storage).build()
        report = OpsOverviewService.render(overview)
        self.assertIn("运维总览", report)

    def test_ops_overview_includes_runtime_override_state(self):
        self.storage.set_state("runtime_settings_override_status", "mixed")
        self.storage.set_json_state(
            "runtime_settings_effective",
            {
                "manual_overrides": {"xgboost_probability_threshold": 0.76},
                "learning_overrides": {"min_liquidity_ratio": 0.9},
            },
        )
        self.storage.set_json_state(
            "runtime_settings_override_conflicts",
            {
                "conflict_keys": ["xgboost_probability_threshold"],
                "blocked_learning_overrides": {"xgboost_probability_threshold": 0.73},
            },
        )
        overview = OpsOverviewService(self.storage).build()
        report = OpsOverviewService.render(overview)
        self.assertIn("Runtime 覆盖状态", report)
        self.assertIn("被拦截的学习字段", report)

    def test_ops_overview_includes_nextgen_live_state(self):
        self.storage.set_json_state(
            "nextgen_autonomy_live_operator_request",
            {
                "requested_live": True,
                "whitelist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                "max_active_runtimes": 2,
                "reason": "cli",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.storage.set_json_state(
            "nextgen_autonomy_live_status",
            {
                "requested_live": True,
                "effective_live": False,
                "allow_entries": False,
                "allow_managed_closes": True,
                "force_flatten": True,
                "kill_switch_active": True,
                "kill_switch_reason": "ops_pause",
                "reasons": ["live_kill_switch_active"],
            },
        )
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_run",
            "SYSTEM",
            {
                "requested_live": True,
                "trigger": "scheduler",
                "status": "ok",
                "reason": "rotation_cleanup",
                "effective_live": False,
                "force_flatten": True,
                "autonomy_cycle_id": 101,
                "repair_queue_requested_size": 3,
                "repair_queue_dropped_count": 1,
                "repair_queue_dropped_runtime_ids": ["runtime_z:seed"],
                "repair_queue_hold_priority_count": 2,
                "repair_queue_postponed_rebuild_count": 1,
                "repair_queue_reprioritized_count": 3,
            },
        )
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_run",
            "SYSTEM",
            {
                "requested_live": True,
                "trigger": "manual_recovery_required",
                "status": "ok",
                "reason": "market_data_latency",
                "effective_live": False,
                "force_flatten": True,
                "autonomy_cycle_id": 102,
                "repair_queue_requested_size": 2,
                "repair_queue_dropped_count": 0,
                "repair_queue_hold_priority_count": 1,
                "repair_queue_postponed_rebuild_count": 0,
                "repair_queue_reprioritized_count": 1,
            },
        )
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_guard_callback_failed",
            "SYSTEM",
            {
                "trigger": "manual_recovery_required",
                "reason": "market_data_latency",
                "error": "nextgen_guard_boom",
            },
        )

        overview = OpsOverviewService(self.storage).build()
        report = OpsOverviewService.render(overview)

        self.assertTrue(overview["nextgen_live"]["requested_live"])
        self.assertFalse(overview["nextgen_live"]["effective_live"])
        self.assertTrue(overview["nextgen_live"]["force_flatten"])
        self.assertEqual(overview["nextgen_live"]["last_run_status"], "ok")
        self.assertEqual(overview["nextgen_live"]["repair_queue_requested_size"], 2)
        self.assertEqual(overview["nextgen_live"]["repair_queue_dropped_count"], 0)
        self.assertEqual(overview["nextgen_live"]["repair_queue_hold_priority_count"], 1)
        self.assertEqual(overview["nextgen_live"]["repair_queue_postponed_rebuild_count"], 0)
        self.assertEqual(len(overview["nextgen_live"]["recent_repair_queue_runs"]), 2)
        self.assertEqual(
            overview["nextgen_live"]["recent_repair_queue_runs"][0]["autonomy_cycle_id"],
            102,
        )
        self.assertEqual(
            overview["nextgen_live"]["recent_repair_queue_runs"][0]["reason"],
            "market_data_latency",
        )
        self.assertEqual(
            overview["nextgen_live"]["recent_repair_queue_runs"][0]["latest_issue_event_type"],
            "nextgen_autonomy_live_guard_callback_failed",
        )
        self.assertEqual(
            overview["nextgen_live"]["recent_repair_queue_runs"][0]["latest_issue_reason"],
            "nextgen_guard_boom",
        )
        self.assertIn("Nextgen Live 请求启用: True", report)
        self.assertIn("Nextgen Live Kill Switch: True", report)
        self.assertIn("Nextgen Live 最近运行: ok", report)
        self.assertIn("Nextgen Live 请求 Repair 数: 2", report)
        self.assertIn("Nextgen Live 被丢弃 Repair 数: 0", report)
        self.assertIn("Nextgen Live Hold Repair 数: 1", report)
        self.assertIn("Nextgen Live 延后 Rebuild 数: 0", report)
        self.assertIn("Nextgen Live 最近队列趋势:", report)

    def test_health_checker_includes_market_data_provider_details(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        self.storage.set_json_state(
            "market_data_last_route",
            {
                "operation": "fetch_latest_price",
                "selected_provider": "okx",
                "fallback_used": False,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        checker = HealthChecker(self.storage, settings)

        status = checker.run()

        self.assertEqual(status.latest_market_data_provider, "okx")
        self.assertEqual(status.latest_market_data_operation, "fetch_latest_price")
        self.assertFalse(status.market_data_failover_active)

    def test_guard_reporter_renders_snapshot(self):
        self.storage.insert_execution_event(
            "model_accuracy_guard",
            "BTC/USDT",
            {"reason": "xgboost_accuracy=40.00"},
        )
        self.storage.insert_execution_event(
            "live_open_limit_timeout",
            "BTC/USDT",
            {"reason": "limit order timeout after 300s"},
        )
        self.storage.insert_account_snapshot(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "equity": 10000.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "daily_loss_pct": 0.0,
                "weekly_loss_pct": 0.0,
                "drawdown_pct": 0.0,
                "total_exposure_pct": 0.0,
                "open_positions": 0,
                "cooldown_until": datetime.now(timezone.utc).isoformat(),
                "circuit_breaker_active": False,
            }
        )
        report_data = GuardReporter(self.storage).build()
        report_text = GuardReporter.render(report_data)
        self.assertGreaterEqual(report_data["guard_event_count"], 2)
        self.assertIn("风控告警报告", report_text)

    def test_guard_reporter_includes_nextgen_live_guard_failures(self):
        self.storage.set_json_state(
            "nextgen_autonomy_live_operator_request",
            {
                "requested_live": True,
                "whitelist": ["BTC/USDT:USDT"],
                "max_active_runtimes": 1,
                "reason": "cli",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.storage.set_json_state(
            "nextgen_autonomy_live_status",
            {
                "requested_live": True,
                "effective_live": False,
                "allow_entries": False,
                "allow_managed_closes": True,
                "force_flatten": True,
                "kill_switch_active": True,
                "kill_switch_reason": "ops_pause",
                "reasons": ["live_kill_switch_active"],
            },
        )
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_run",
            "SYSTEM",
            {
                "requested_live": True,
                "trigger": "scheduler",
                "status": "ok",
                "force_flatten": True,
                "repair_queue_requested_size": 4,
                "repair_queue_dropped_count": 2,
                "repair_queue_hold_priority_count": 1,
                "repair_queue_postponed_rebuild_count": 2,
                "repair_queue_reprioritized_count": 2,
            },
        )
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_guard_callback_failed",
            "SYSTEM",
            {
                "trigger": "manual_recovery_required",
                "reason": "market_data_latency",
                "error": "nextgen_guard_boom",
            },
        )
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_run_failed",
            "SYSTEM",
            {
                "trigger": "scheduler",
                "error": "runner_boom",
            },
        )

        report_data = GuardReporter(self.storage).build()
        report_text = GuardReporter.render(report_data)
        event_counts = dict(report_data["event_counts"])
        top_reasons = dict(report_data["top_reasons"])

        self.assertIn("nextgen_autonomy_live_guard_callback_failed", event_counts)
        self.assertIn("nextgen_autonomy_live_run_failed", event_counts)
        self.assertIn("nextgen_guard_boom", top_reasons)
        self.assertIn("Nextgen Live Kill Switch: True", report_text)
        self.assertIn(
            "Nextgen Live 最近问题: nextgen_autonomy_live_run_failed",
            report_text,
        )
        self.assertIn("Nextgen Live 被丢弃 Repair 数: 2", report_text)
        self.assertIn("Nextgen Live Hold Repair 数: 1", report_text)
        self.assertIn("Nextgen Live 延后 Rebuild 数: 2", report_text)
        self.assertIn("Nextgen Live 最近队列趋势:", report_text)

    def test_failure_reporter_filters_non_failure_events(self):
        from monitor.failure_report import FailureReporter

        self.storage.insert_execution_event(
            "analysis",
            "BTC/USDT",
            {"reason": "normal_analysis"},
        )
        self.storage.insert_execution_event(
            "live_open_rejected",
            "BTC/USDT",
            {"reason": "slippage too high"},
        )
        report_data = FailureReporter(self.storage).build()
        reasons = dict(report_data["top_execution_failures"])
        self.assertIn("slippage too high", reasons)
        self.assertNotIn("normal_analysis", reasons)

    def test_failure_reporter_and_maintenance(self):
        from monitor.failure_report import FailureReporter
        from monitor.incident_report import IncidentReporter
        from monitor.maintenance_service import MaintenanceService

        self.storage.insert_cycle_run(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "symbols": ["BTC/USDT"],
                "opened_positions": 0,
                "closed_positions": 0,
                "circuit_breaker_active": True,
                "reconciliation_status": "mismatch",
                "notes": "reconciliation_mismatch",
            }
        )
        self.storage.insert_execution_event(
            "live_open_rejected",
            "BTC/USDT",
            {"reason": "slippage too high"},
        )
        report_data = FailureReporter(self.storage).build()
        report_text = FailureReporter.render(report_data)
        self.assertIn("故障报告", report_text)
        self.assertGreaterEqual(report_data["failed_cycle_count"], 1)

        incident_data = IncidentReporter(self.storage).build()
        incident_text = IncidentReporter.render(incident_data)
        self.assertIn("事故报告", incident_text)
        self.assertGreaterEqual(incident_data["failed_cycles"], 1)

        settings = get_settings().maintenance
        maintenance = MaintenanceService(self.storage, settings)
        summary = maintenance.run()
        self.assertIn("feature_snapshots", summary)

    def test_scheduler_service_records_run(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.project_root = Path(self.db_path).with_suffix("")
        fake_engine = SimpleNamespace(
            settings=settings,
            storage=self.storage,
            run_once=lambda: {"ok": True},
            train_models=lambda: [],
            run_walkforward=lambda symbol: {"symbol": symbol},
            generate_reports=lambda: {"daily": "x", "weekly": "y"},
            run_health_check=lambda: {"status": "ok"},
            run_guard_report=lambda: {"status": "ok"},
            run_ab_test_report=lambda: {"status": "ok"},
            run_drift_report=lambda: {"status": "ok"},
            run_metrics=lambda: {"status": "ok"},
            run_maintenance=lambda: {"status": "ok"},
            run_failure_report=lambda: {"status": "ok"},
            run_incident_report=lambda: {"status": "ok"},
            run_reconciliation=lambda: {"status": "ok"},
        )
        scheduler = SchedulerService(fake_engine)
        self.assertNotIn("ops", scheduler.jobs)
        self.assertIn("guards", scheduler.jobs)
        self.assertIn("abtest", scheduler.jobs)
        self.assertIn("drift", scheduler.jobs)
        result = scheduler.run_job("once")
        self.assertEqual(result["status"], "ok")
        with self.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM scheduler_runs"
            ).fetchone()["c"]
        self.assertEqual(count, 1)

    def test_scheduler_service_registers_optional_ops_job(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.project_root = Path(self.db_path).with_suffix("")
        fake_engine = SimpleNamespace(
            settings=settings,
            storage=self.storage,
            run_once=lambda: {"ok": True},
            train_models=lambda: [],
            run_walkforward=lambda symbol: {"symbol": symbol},
            generate_reports=lambda: {"daily": "x", "weekly": "y"},
            run_health_check=lambda: {"status": "ok"},
            run_guard_report=lambda: {"status": "ok"},
            run_ab_test_report=lambda: {"status": "ok"},
            run_drift_report=lambda: {"status": "ok"},
            run_metrics=lambda: {"status": "ok"},
            run_maintenance=lambda: {"status": "ok"},
            run_failure_report=lambda: {"status": "ok"},
            run_incident_report=lambda: {"status": "ok"},
            run_ops_overview=lambda: {"status": "ok"},
            run_reconciliation=lambda: {"status": "ok"},
        )
        scheduler = SchedulerService(fake_engine)
        self.assertIn("ops", scheduler.jobs)

    def test_scheduler_service_registers_optional_nextgen_live_job(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.project_root = Path(self.db_path).with_suffix("")
        fake_engine = SimpleNamespace(
            settings=settings,
            storage=self.storage,
            run_once=lambda: {"ok": True},
            train_models=lambda: [],
            run_walkforward=lambda symbol: {"symbol": symbol},
            generate_reports=lambda: {"daily": "x", "weekly": "y"},
            run_health_check=lambda: {"status": "ok"},
            run_guard_report=lambda: {"status": "ok"},
            run_ab_test_report=lambda: {"status": "ok"},
            run_drift_report=lambda: {"status": "ok"},
            run_metrics=lambda: {"status": "ok"},
            run_maintenance=lambda: {"status": "ok"},
            run_failure_report=lambda: {"status": "ok"},
            run_incident_report=lambda: {"status": "ok"},
            run_reconciliation=lambda: {"status": "ok"},
            run_nextgen_autonomy_live=lambda: {"status": "ok"},
        )

        scheduler = SchedulerService(fake_engine)
        scheduled_job_names = [job[0] for job in scheduler._scheduled_jobs()]

        self.assertIn("nextgen_live", scheduler.jobs)
        self.assertIn("nextgen_live", scheduled_job_names)

    def test_scheduler_service_skips_heavy_jobs_in_low_resource_mode(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.low_resource_mode = True
        settings.app.project_root = Path(self.db_path).with_suffix("")
        fake_engine = SimpleNamespace(
            settings=settings,
            storage=self.storage,
            run_once=lambda: {"ok": True},
            train_models=lambda: [],
            run_walkforward=lambda symbol: {"symbol": symbol},
            generate_reports=lambda: {"daily": "x", "weekly": "y"},
            run_health_check=lambda: {"status": "ok"},
            run_guard_report=lambda: {"status": "ok"},
            run_ab_test_report=lambda: {"status": "ok"},
            run_drift_report=lambda: {"status": "ok"},
            run_metrics=lambda: {"status": "ok"},
            run_maintenance=lambda: {"status": "ok"},
            run_failure_report=lambda: {"status": "ok"},
            run_incident_report=lambda: {"status": "ok"},
            run_ops_overview=lambda: {"status": "ok"},
            run_reconciliation=lambda: {"status": "ok"},
        )
        scheduler = SchedulerService(fake_engine)
        scheduled_job_names = [job[0] for job in scheduler._scheduled_jobs()]
        self.assertNotIn("train", scheduled_job_names)
        self.assertNotIn("walkforward", scheduled_job_names)
        self.assertNotIn("abtest", scheduled_job_names)
        self.assertNotIn("drift", scheduled_job_names)
        self.assertIn("once", scheduled_job_names)

    def test_scheduler_service_start_blocking_routes_through_run_job(self):
        import monitor.scheduler_service as scheduler_module

        class FakeBlockingScheduler:
            def __init__(self, timezone="UTC"):
                self.jobs = []

            def add_job(self, func, trigger, id=None, replace_existing=True, **schedule_kwargs):
                self.jobs.append((id, func, trigger, schedule_kwargs))

            def start(self):
                for _job_id, func, _trigger, _schedule_kwargs in self.jobs:
                    func()

        settings = get_settings().model_copy(deep=True)
        settings.scheduler.analysis_cron_minutes = 0
        settings.scheduler.training_cron_hours = 0
        settings.scheduler.walkforward_cron_hours = 0
        settings.scheduler.report_cron_hours = 0
        settings.scheduler.guard_cron_hours = 0
        settings.scheduler.ops_cron_minutes = 0
        settings.scheduler.reconcile_cron_hours = 0
        settings.scheduler.maintenance_cron_hours = 0
        settings.scheduler.failure_cron_hours = 0
        settings.scheduler.incident_cron_hours = 0
        settings.scheduler.health_cron_minutes = 1

        fake_engine = SimpleNamespace(
            settings=settings,
            storage=self.storage,
            run_health_check=lambda: {"status": "ok"},
        )
        scheduler = scheduler_module.SchedulerService(fake_engine)

        with patch.object(scheduler_module, "BlockingScheduler", FakeBlockingScheduler):
            scheduler.start_blocking()

        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT job_name, status FROM scheduler_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(row["job_name"], "health")
        self.assertEqual(row["status"], "ok")

    def test_scheduler_service_once_respects_cycle_lock(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.project_root = Path(self.db_path).with_suffix("")
        cycle_lock = settings.app.project_root / "data" / "runtime" / "cycle.lock"
        cycle_lock.parent.mkdir(parents=True, exist_ok=True)
        cycle_lock.write_text(str(os.getpid()), encoding="utf-8")
        called = []

        fake_engine = SimpleNamespace(
            settings=settings,
            storage=self.storage,
            run_once=lambda: called.append("run"),
            train_models=lambda: [],
            run_walkforward=lambda symbol: {"symbol": symbol},
            generate_reports=lambda: {"daily": "x", "weekly": "y"},
            run_health_check=lambda: {"status": "ok"},
            run_guard_report=lambda: {"status": "ok"},
            run_ab_test_report=lambda: {"status": "ok"},
            run_drift_report=lambda: {"status": "ok"},
            run_metrics=lambda: {"status": "ok"},
            run_maintenance=lambda: {"status": "ok"},
            run_failure_report=lambda: {"status": "ok"},
            run_incident_report=lambda: {"status": "ok"},
            run_reconciliation=lambda: {"status": "ok"},
        )
        scheduler = SchedulerService(fake_engine)
        result = scheduler.run_job("once")
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(called, [])

    def test_scheduler_service_cycle_lock_handles_create_race(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.project_root = Path(self.db_path).with_suffix("")
        cycle_lock = settings.app.project_root / "data" / "runtime" / "cycle.lock"
        cycle_lock.parent.mkdir(parents=True, exist_ok=True)
        original_open = Path.open
        raced = {"done": False}

        fake_engine = SimpleNamespace(
            settings=settings,
            storage=self.storage,
            run_once=lambda: {"ok": True},
            train_models=lambda: [],
            run_walkforward=lambda symbol: {"symbol": symbol},
            generate_reports=lambda: {"daily": "x", "weekly": "y"},
            run_health_check=lambda: {"status": "ok"},
            run_guard_report=lambda: {"status": "ok"},
            run_ab_test_report=lambda: {"status": "ok"},
            run_drift_report=lambda: {"status": "ok"},
            run_metrics=lambda: {"status": "ok"},
            run_maintenance=lambda: {"status": "ok"},
            run_failure_report=lambda: {"status": "ok"},
            run_incident_report=lambda: {"status": "ok"},
            run_reconciliation=lambda: {"status": "ok"},
        )
        scheduler = SchedulerService(fake_engine)

        def racing_open(path_obj, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if path_obj == cycle_lock and mode == "x" and not raced["done"]:
                raced["done"] = True
                with original_open(cycle_lock, "w", encoding="utf-8") as handle:
                    handle.write(str(os.getpid()))
                raise FileExistsError
            return original_open(path_obj, *args, **kwargs)

        with patch.object(Path, "open", autospec=True, side_effect=racing_open):
            result = scheduler.run_job("once")

        self.assertEqual(result["status"], "skipped")

    def test_scheduler_service_walkforward_uses_active_symbol(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.project_root = Path(self.db_path).with_suffix("")
        settings.exchange.symbols = ["BTC/USDT"]
        captured: list[str] = []

        fake_engine = SimpleNamespace(
            settings=settings,
            storage=self.storage,
            run_once=lambda: {"ok": True},
            train_models=lambda: [],
            run_walkforward=lambda symbol: captured.append(symbol) or {"symbol": symbol},
            get_active_symbols=lambda force_refresh=False: ["SOL/USDT"],
            get_execution_symbols=lambda: ["ETH/USDT"],
            generate_reports=lambda: {"daily": "x", "weekly": "y"},
            run_health_check=lambda: {"status": "ok"},
            run_guard_report=lambda: {"status": "ok"},
            run_ab_test_report=lambda: {"status": "ok"},
            run_drift_report=lambda: {"status": "ok"},
            run_metrics=lambda: {"status": "ok"},
            run_maintenance=lambda: {"status": "ok"},
            run_failure_report=lambda: {"status": "ok"},
            run_incident_report=lambda: {"status": "ok"},
            run_reconciliation=lambda: {"status": "ok"},
        )
        scheduler = SchedulerService(fake_engine)
        result = scheduler.run_job("walkforward")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(captured, ["SOL/USDT"])
