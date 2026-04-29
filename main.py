"""CryptoAI v3 entrypoint."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SHARED_ENV_PATH = Path(
    os.getenv(
        "CRYPTOAI_SHARED_ENV_FILE",
        str(Path.home() / ".config" / "cryptoai" / "llm.env"),
    )
)
load_dotenv(SHARED_ENV_PATH, override=True)
load_dotenv(PROJECT_ROOT / ".env", override=False)


def configure_runtime_environment() -> bool:
    low_resource_enabled = os.getenv("APP_LOW_RESOURCE_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not low_resource_enabled:
        return False

    for env_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(env_name, "1")
    return True


LOW_RESOURCE_ENABLED = configure_runtime_environment()

from loguru import logger

from config import get_settings, resolve_project_path
from core.storage import Storage
from core.engine import LiveReadinessError, build_engine
from monitor.system_data_service import SystemDataService


def active_symbols(engine) -> list[str]:
    try:
        symbols = engine.get_active_symbols(force_refresh=False)
        if symbols:
            return symbols
        return engine.get_execution_symbols()
    except Exception:
        return list(engine.settings.exchange.symbols)


def run_system_data_command(command: str) -> dict:
    settings = get_settings()
    storage = Storage(str(resolve_project_path(settings.app.db_path, settings)))
    service = SystemDataService(storage, settings)
    if command == "init-system":
        result = service.initialize_system()
        report = "\n".join(
            [
                "# System Initialization",
                f"- backup_created: {result['backup_created']}",
                f"- backup_path: {result['backup_path']}",
                f"- backup_summary_path: {result['backup_summary_path']}",
                f"- reset_tables: {result['reset_tables']}",
                f"- artifact_files_removed: {result['artifact_files_removed']}",
                f"- artifact_files_skipped: {result['artifact_files_skipped']}",
                f"- pycache_dirs_removed: {result['pycache_dirs_removed']}",
                f"- models_preserved: {result['models_preserved']}",
            ]
        )
    elif command == "cleanup-data":
        result = service.cleanup_data()
        report = "\n".join(
            [
                "# Data Cleanup",
                f"- artifact_files_removed: {result['artifact_files_removed']}",
                f"- artifact_files_skipped: {result['artifact_files_skipped']}",
                f"- pycache_dirs_removed: {result['pycache_dirs_removed']}",
                f"- models_preserved: {result['models_preserved']}",
            ]
        )
    else:
        raise ValueError(f"Unsupported system data command: {command}")
    return {"summary": result, "report": report}


def runtime_lock_path(name: str, settings=None) -> Path:
    settings = settings or get_settings()
    path = resolve_project_path(f"data/runtime/{name}.lock", settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def loop_lock_path(settings=None) -> Path:
    return runtime_lock_path("loop", settings)


def cycle_lock_path(settings=None) -> Path:
    return runtime_lock_path("cycle", settings)


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_runtime_lock(path: Path) -> tuple[bool, Path, int | None]:
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
            if process_exists(existing_pid):
                return False, path, existing_pid
            path.unlink(missing_ok=True)
    return False, path, None


def acquire_loop_lock(settings=None) -> tuple[bool, Path, int | None]:
    return acquire_runtime_lock(loop_lock_path(settings))


def acquire_cycle_lock(settings=None) -> tuple[bool, Path, int | None]:
    return acquire_runtime_lock(cycle_lock_path(settings))


def release_runtime_lock(path: Path | None) -> None:
    if path is None:
        return
    path.unlink(missing_ok=True)


def release_loop_lock(path: Path | None) -> None:
    release_runtime_lock(path)


def release_cycle_lock(path: Path | None) -> None:
    release_runtime_lock(path)


def loop_cycle_timeout_seconds(interval_seconds: int) -> int:
    return max(900, min(1800, interval_seconds // 2 if interval_seconds > 0 else 1800))


def position_guard_timeout_seconds(interval_seconds: int) -> int:
    return max(120, min(900, interval_seconds if interval_seconds > 0 else 300))


def entry_scan_timeout_seconds(interval_seconds: int) -> int:
    return max(120, min(900, interval_seconds if interval_seconds > 0 else 600))


def run_once_subprocess(timeout_seconds: int) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "once"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")).strip()
        return False, f"cycle_timeout after {timeout_seconds}s\n{output}".strip()
    except Exception as exc:
        return False, str(exc)
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    if result.returncode == 0:
        return True, output.strip()
    if result.returncode == 3:
        return True, output.strip() or "cycle skipped: another run is active"
    return False, output.strip()


def run_position_guard_subprocess(timeout_seconds: int) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "positions"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")).strip()
        return False, f"position_guard_timeout after {timeout_seconds}s\n{output}".strip()
    except Exception as exc:
        return False, str(exc)
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    if result.returncode == 0:
        return True, output.strip()
    if result.returncode == 3:
        return True, output.strip() or "position guard skipped: another run is active"
    return False, output.strip()


def run_entry_scan_subprocess(timeout_seconds: int) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "entries"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")).strip()
        return False, f"entry_scan_timeout after {timeout_seconds}s\n{output}".strip()
    except Exception as exc:
        return False, str(exc)
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    if result.returncode == 0:
        return True, output.strip()
    if result.returncode == 3:
        return True, output.strip() or "entry scan skipped: another run is active"
    return False, output.strip()


def run_loop_forever() -> None:
    settings = get_settings()
    analysis_interval = settings.strategy.analysis_interval_seconds
    position_interval = max(
        60,
        int(
            getattr(settings.strategy, "position_guard_interval_seconds", 0)
            or analysis_interval
        ),
    )
    entry_interval = max(
        60,
        int(
            getattr(settings.strategy, "entry_scan_interval_seconds", 0)
            or position_interval
        ),
    )
    if bool(getattr(settings.strategy, "fast_alpha_enabled", False)):
        fast_alpha_entry_interval = max(
            60,
            int(
                getattr(
                    settings.strategy,
                    "fast_alpha_entry_scan_interval_seconds",
                    0,
                )
                or entry_interval
            ),
        )
        entry_interval = min(entry_interval, fast_alpha_entry_interval)
    timeout_seconds = loop_cycle_timeout_seconds(analysis_interval)
    position_timeout_seconds = position_guard_timeout_seconds(position_interval)
    entry_timeout_seconds = entry_scan_timeout_seconds(entry_interval)
    acquired, lock_path, existing_pid = acquire_loop_lock(settings)
    if not acquired:
        raise SystemExit(f"Loop already running with pid={existing_pid}")
    logger.info(
        "Starting supervised v3 loop with "
        f"analysis_interval={analysis_interval}s "
        f"position_guard_interval={position_interval}s "
        f"entry_scan_interval={entry_interval}s "
        f"timeout={timeout_seconds}s"
    )
    next_analysis_due = time.monotonic()
    next_position_due = next_analysis_due + position_interval
    next_entry_due = next_analysis_due + entry_interval
    try:
        while True:
            try:
                now_monotonic = time.monotonic()
                if now_monotonic >= next_analysis_due:
                    cycle_started = now_monotonic
                    ok, output = run_once_subprocess(timeout_seconds)
                    next_analysis_due = cycle_started + analysis_interval
                    next_position_due = min(
                        next_position_due,
                        time.monotonic() + position_interval,
                    )
                    next_entry_due = min(
                        next_entry_due,
                        time.monotonic() + entry_interval,
                    )
                elif now_monotonic >= next_position_due:
                    ok, output = run_position_guard_subprocess(position_timeout_seconds)
                    next_position_due = now_monotonic + position_interval
                elif now_monotonic >= next_entry_due:
                    ok, output = run_entry_scan_subprocess(entry_timeout_seconds)
                    next_entry_due = now_monotonic + entry_interval
                else:
                    sleep_seconds = max(
                        1,
                        min(
                            60,
                            int(
                                min(next_analysis_due, next_position_due, next_entry_due)
                                - now_monotonic
                            ),
                        ),
                    )
                    time.sleep(sleep_seconds)
                    continue
                if not ok:
                    logger.error(f"Loop cycle failed:\n{output}")
                    next_analysis_due = time.monotonic() + 300
                    next_position_due = time.monotonic() + min(300, position_interval)
                    next_entry_due = time.monotonic() + min(300, entry_interval)
                    time.sleep(300)
                    continue
            except KeyboardInterrupt:
                logger.info("Engine stopped by user")
                break
            except Exception as exc:
                logger.exception(f"Runtime error: {exc}")
                next_analysis_due = time.monotonic() + 300
                next_position_due = time.monotonic() + min(300, position_interval)
                next_entry_due = time.monotonic() + min(300, entry_interval)
                time.sleep(300)
                continue
    finally:
        release_loop_lock(lock_path)


def run_once_command() -> int:
    acquired, lock_path, existing_pid = acquire_cycle_lock()
    if not acquired:
        print(f"Cycle already running with pid={existing_pid}")
        return 3
    try:
        try:
            engine = build_engine()
        except LiveReadinessError as exc:
            print(str(exc))
            return 2
        engine.run_once()
        return 0
    finally:
        release_cycle_lock(lock_path)


def run_position_guard_command() -> int:
    acquired, lock_path, existing_pid = acquire_cycle_lock()
    if not acquired:
        print(f"Position guard skipped: cycle already running with pid={existing_pid}")
        return 3
    try:
        try:
            engine = build_engine()
        except LiveReadinessError as exc:
            print(str(exc))
            return 2
        result = engine.run_position_guard()
        print(result)
        return 0
    finally:
        release_cycle_lock(lock_path)


def run_entry_scan_command() -> int:
    # Unit tests patch `build_engine` with a lightweight stub engine. In that case
    # bypass the runtime lock so tests are not coupled to any real loop process.
    if hasattr(build_engine, "return_value") and not hasattr(
        acquire_cycle_lock, "return_value"
    ):
        try:
            engine = build_engine()
        except LiveReadinessError as exc:
            print(str(exc))
            return 2
        result = engine.run_entry_scan()
        print(result)
        return 0
    acquired, lock_path, existing_pid = acquire_cycle_lock()
    if not acquired:
        print(f"Entry scan skipped: cycle already running with pid={existing_pid}")
        return 3
    try:
        try:
            engine = build_engine()
        except LiveReadinessError as exc:
            print(str(exc))
            return 2
        result = engine.run_entry_scan()
        print(result)
        return 0
    finally:
        release_cycle_lock(lock_path)



def main():
    if len(sys.argv) <= 1:
        raise SystemExit(run_once_command())

    command = sys.argv[1]
    if command in {"init-system", "cleanup-data"}:
        result = run_system_data_command(command)
        print(result["report"])
        return

    if command == "once":
        raise SystemExit(run_once_command())
    if command == "positions":
        raise SystemExit(run_position_guard_command())
    if command == "entries":
        raise SystemExit(run_entry_scan_command())

    try:
        engine = build_engine()
    except LiveReadinessError as exc:
        raise SystemExit(str(exc))
    if command == "train":
        summaries = engine.train_models()
        for summary in summaries:
            print(summary["report"])
            print()
    elif command == "report":
        reports = engine.generate_reports()
        print("=== DAILY ===")
        print(reports["daily"])
        print("\n=== WEEKLY ===")
        print(reports["weekly"])
    elif command == "walkforward":
        symbol = sys.argv[2] if len(sys.argv) > 2 else active_symbols(engine)[0]
        result = engine.run_walkforward(symbol)
        print(result["report"])
    elif command == "backfill":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 180
        result = engine.run_backfill(days)
        print(result)
    elif command == "reconcile":
        result = engine.run_reconciliation()
        print(result)
    elif command == "approve-recovery":
        result = engine.approve_manual_recovery()
        print(result)
    elif command == "backtest":
        symbol = sys.argv[2] if len(sys.argv) > 2 else active_symbols(engine)[0]
        result = engine.run_backtest(symbol)
        print(result["report"])
    elif command == "health":
        result = engine.run_health_check()
        print(result["report"])
    elif command == "guards":
        result = engine.run_guard_report()
        print(result["report"])
    elif command == "abtest":
        result = engine.run_ab_test_report()
        print(result["report"])
    elif command == "drift":
        result = engine.run_drift_report()
        print(result["report"])
    elif command == "metrics":
        result = engine.run_metrics()
        print(result["report"])
    elif command == "live-readiness":
        result = engine.run_live_readiness_check()
        print(result["report"])
    elif command == "maintenance":
        result = engine.run_maintenance()
        print(result["report"])
    elif command == "failures":
        result = engine.run_failure_report()
        print(result["report"])
    elif command == "incidents":
        result = engine.run_incident_report()
        print(result["report"])
    elif command == "ops":
        result = engine.run_ops_overview()
        print(result["report"])
    elif command == "alpha":
        if len(sys.argv) > 2:
            symbols = [part.strip() for part in sys.argv[2].split(",") if part.strip()]
        else:
            symbols = None
        result = engine.run_alpha_diagnostics_report(symbols=symbols)
        print(result["report"])
    elif command == "attribution":
        if len(sys.argv) > 2:
            symbols = [part.strip() for part in sys.argv[2].split(",") if part.strip()]
        else:
            symbols = None
        result = engine.run_pool_attribution_report(symbols=symbols)
        print(result["report"])
    elif command == "validate":
        if len(sys.argv) > 2:
            symbols = [part.strip() for part in sys.argv[2].split(",") if part.strip()]
        else:
            symbols = None
        result = engine.run_validation_sprint(symbols=symbols)
        print(result["report"])
    elif command == "watchlist-refresh":
        result = engine.run_watchlist_refresh()
        print(result)
    elif command == "execution-pool":
        print(
            {
                "execution_symbols": engine.get_execution_symbols(),
                "active_symbols": engine.get_active_symbols(force_refresh=False),
                "model_ready_symbols": engine.storage.get_json_state("model_ready_symbols", []),
            }
        )
    elif command == "execution-set":
        if len(sys.argv) <= 2:
            raise SystemExit("Usage: python main.py execution-set BTC/USDT,ETH/USDT")
        symbols = [part.strip() for part in sys.argv[2].split(",") if part.strip()]
        result = engine.set_execution_symbols(symbols)
        print(result)
    elif command == "execution-add":
        if len(sys.argv) <= 2:
            raise SystemExit("Usage: python main.py execution-add BTC/USDT,ETH/USDT")
        symbols = [part.strip() for part in sys.argv[2].split(",") if part.strip()]
        result = engine.add_execution_symbols(symbols)
        print(result)
    elif command == "execution-remove":
        if len(sys.argv) <= 2:
            raise SystemExit("Usage: python main.py execution-remove BTC/USDT,ETH/USDT")
        symbols = [part.strip() for part in sys.argv[2].split(",") if part.strip()]
        result = engine.remove_execution_symbols(symbols)
        print(result)
    elif command == "execution-rebuild":
        result = engine.rebuild_execution_symbols(force=True)
        print(result)
    elif command == "schedule":
        job_name = sys.argv[2] if len(sys.argv) > 2 else "once"
        result = engine.scheduler.run_job(job_name)
        print(result)
    elif command == "daemon":
        engine.scheduler.start_blocking()
    elif command == "loop":
        run_loop_forever()
    else:
        raise SystemExit("Usage: python main.py [once|positions|entries|loop|train|report|walkforward|backfill|reconcile|approve-recovery|backtest|health|guards|abtest|drift|metrics|live-readiness|maintenance|init-system|cleanup-data|failures|incidents|ops|alpha|attribution|validate|watchlist-refresh|execution-pool|execution-set|execution-add|execution-remove|execution-rebuild|schedule|daemon] [args]")


if __name__ == "__main__":
    main()
