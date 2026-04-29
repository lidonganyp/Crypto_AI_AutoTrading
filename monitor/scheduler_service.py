"""Scheduler service for CryptoAI v3."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from loguru import logger

from config import resolve_project_path

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
except Exception:  # pragma: no cover
    BlockingScheduler = None

if TYPE_CHECKING:
    from core.engine import CryptoAIV2Engine


class SchedulerService:
    """Register and run periodic jobs for the engine."""

    def __init__(self, engine: CryptoAIV2Engine):
        self.engine = engine
        self.jobs: dict[str, Callable[[], object]] = {}

        self.jobs["once"] = self._run_once_guarded
        self._register_job("train", "train_models")
        self._register_job("report", "generate_reports")
        self._register_job("health", "run_health_check")
        self._register_job("guards", "run_guard_report")
        self._register_job("abtest", "run_ab_test_report")
        self._register_job("drift", "run_drift_report")
        self._register_job("metrics", "run_metrics")
        self._register_job("maintenance", "run_maintenance")
        self._register_job("failures", "run_failure_report")
        self._register_job("incidents", "run_incident_report")
        self._register_job("ops", "run_ops_overview")
        self._register_job("reconcile", "run_reconciliation")
        self._register_job("nextgen_live", "run_nextgen_autonomy_live")

        walkforward = getattr(self.engine, "run_walkforward", None)
        if callable(walkforward):
            self.jobs["walkforward"] = self._run_walkforward_job

    def _register_job(self, name: str, attr_name: str):
        handler = getattr(self.engine, attr_name, None)
        if callable(handler):
            self.jobs[name] = handler

    def _scheduled_job_runner(self, job_name: str):
        return lambda: self.run_job(job_name)

    def _scheduled_jobs(self) -> list[tuple[str, str, dict[str, int]]]:
        settings = self.engine.settings.scheduler
        jobs = [
            ("once", "interval", {"minutes": settings.analysis_cron_minutes}),
            ("health", "interval", {"minutes": settings.health_cron_minutes}),
            ("guards", "interval", {"hours": settings.guard_cron_hours}),
            ("abtest", "interval", {"hours": settings.report_cron_hours}),
            ("drift", "interval", {"hours": settings.report_cron_hours}),
            ("ops", "interval", {"minutes": settings.ops_cron_minutes}),
            ("nextgen_live", "interval", {"hours": settings.guard_cron_hours}),
            ("train", "interval", {"hours": settings.training_cron_hours}),
            ("walkforward", "interval", {"hours": settings.walkforward_cron_hours}),
            ("report", "interval", {"hours": settings.report_cron_hours}),
            ("reconcile", "interval", {"hours": settings.reconcile_cron_hours}),
            ("maintenance", "interval", {"hours": settings.maintenance_cron_hours}),
            ("failures", "interval", {"hours": settings.failure_cron_hours}),
            ("incidents", "interval", {"hours": settings.incident_cron_hours}),
        ]
        if self.engine.settings.app.low_resource_mode:
            disabled = {"train", "walkforward", "abtest", "drift"}
            jobs = [job for job in jobs if job[0] not in disabled]
        return jobs

    def _cycle_lock_path(self) -> Path:
        path = resolve_project_path("data/runtime/cycle.lock", self.engine.settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _process_exists(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _acquire_cycle_lock(self) -> tuple[bool, Path, int | None]:
        path = self._cycle_lock_path()
        for _ in range(2):
            try:
                with path.open("x", encoding="utf-8") as handle:
                    handle.write(str(os.getpid()))
                return True, path, None
            except FileExistsError:
                try:
                    existing_pid = int(path.read_text(encoding="utf-8").strip() or "0")
                except (OSError, ValueError):
                    return False, path, None
                if self._process_exists(existing_pid):
                    return False, path, existing_pid
                path.unlink(missing_ok=True)
        return False, path, None

    @staticmethod
    def _release_cycle_lock(path: Path | None) -> None:
        if path is None:
            return
        path.unlink(missing_ok=True)

    def _run_once_guarded(self) -> dict:
        acquired, lock_path, existing_pid = self._acquire_cycle_lock()
        if not acquired:
            return {
                "status": "skipped",
                "reason": "cycle_already_running",
                "pid": existing_pid,
            }
        try:
            result = self.engine.run_once()
            if result is None:
                return {"status": "ok"}
            return result
        finally:
            self._release_cycle_lock(lock_path)

    def _walkforward_symbol(self) -> str | None:
        get_active_symbols = getattr(self.engine, "get_active_symbols", None)
        if callable(get_active_symbols):
            try:
                active_symbols = list(get_active_symbols(force_refresh=False))
            except TypeError:
                active_symbols = list(get_active_symbols())
            if active_symbols:
                return active_symbols[0]

        get_execution_symbols = getattr(self.engine, "get_execution_symbols", None)
        if callable(get_execution_symbols):
            execution_symbols = list(get_execution_symbols())
            if execution_symbols:
                return execution_symbols[0]

        symbols = list(getattr(self.engine.settings.exchange, "symbols", []) or [])
        return symbols[0] if symbols else None

    def _run_walkforward_job(self) -> dict:
        symbol = self._walkforward_symbol()
        if not symbol:
            return {"status": "skipped", "reason": "no_symbols"}
        return self.engine.run_walkforward(symbol)

    def run_job(self, job_name: str) -> dict:
        if job_name not in self.jobs:
            raise ValueError(f"Unknown scheduler job: {job_name}")

        started_at = datetime.now(timezone.utc).isoformat()
        run_id = self.engine.storage.insert_scheduler_run(
            {
                "job_name": job_name,
                "status": "running",
                "output": "",
                "started_at": started_at,
                "completed_at": None,
            }
        )
        try:
            result = self.jobs[job_name]()
            output = str(result)
            status = "ok"
            if isinstance(result, dict):
                result_status = str(result.get("status") or "").lower()
                if result_status in {"ok", "failed", "skipped"}:
                    status = result_status
        except Exception as exc:
            logger.exception(f"Scheduled job failed: {job_name}")
            output = str(exc)
            status = "failed"
        completed_at = datetime.now(timezone.utc).isoformat()
        self.engine.storage.update_scheduler_run(
            run_id,
            {
                "status": status,
                "output": output,
                "completed_at": completed_at,
            },
        )
        return {
            "job_name": job_name,
            "status": status,
            "output": output,
            "started_at": started_at,
            "completed_at": completed_at,
        }

    def start_blocking(self):
        if BlockingScheduler is None:
            raise RuntimeError("APScheduler is not available")

        scheduler = BlockingScheduler(timezone="UTC")
        for job_name, trigger, schedule_kwargs in self._scheduled_jobs():
            if job_name not in self.jobs:
                logger.info(f"Skipping unavailable scheduler job: {job_name}")
                continue
            if any(value <= 0 for value in schedule_kwargs.values()):
                logger.info(f"Skipping disabled scheduler job: {job_name}")
                continue
            scheduler.add_job(
                self._scheduled_job_runner(job_name),
                trigger,
                id=job_name,
                replace_existing=True,
                **schedule_kwargs,
            )
        logger.info("Starting blocking scheduler")
        scheduler.start()
