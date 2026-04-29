import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dashboard as dashboard_module
import main as main_module
from config import get_settings
from core.i18n import LANGUAGE_STATE_KEY
from core.storage import Storage
from monitor.system_data_service import SystemDataService


class _DummyContext:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self._st

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, name):
        return getattr(self._st, name)


class _DummySidebar(_DummyContext):
    def __init__(self, st, page: str = "Overview"):
        super().__init__(st)
        self._page = page

    def radio(self, _label, options, format_func=None):
        return self._page if self._page in options else options[0]

    def selectbox(self, _label, options, index=0, format_func=None, key=None):
        if "zh" in options:
            return "zh"
        return options[index if index < len(options) else 0]


class DummyStreamlit:
    def __init__(self, page: str = "Overview"):
        self.sidebar = _DummySidebar(self, page=page)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def title(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def code(self, *args, **kwargs):
        return None

    def json(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def line_chart(self, *args, **kwargs):
        return None

    def bar_chart(self, *args, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        return None

    def divider(self, *args, **kwargs):
        return None

    def rerun(self, *args, **kwargs):
        return None

    def tabs(self, labels):
        return [_DummyContext(self) for _ in labels]

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [_DummyContext(self) for _ in range(count)]

    def form(self, _name):
        return _DummyContext(self)

    def button(self, *args, **kwargs):
        return False

    def form_submit_button(self, *args, **kwargs):
        return False

    def text_area(self, _label, value="", **kwargs):
        return value

    def text_input(self, _label, value="", **kwargs):
        return value

    def number_input(self, _label, min_value=None, max_value=None, value=0, **kwargs):
        return value

    def slider(self, _label, min_value=None, max_value=None, value=0, **kwargs):
        return value

    def selectbox(self, _label, options, index=0, format_func=None, key=None):
        return options[index if index < len(options) else 0]

    def radio(self, _label, options, format_func=None):
        return options[0]


class SystemToolsAndSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.db_path = self.project_root / "data" / "cryptoai.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage = Storage(str(self.db_path))
        self.settings = get_settings().model_copy(deep=True)
        self.settings.app.project_root = self.project_root
        self.settings.app.db_path = str(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_system_data_service_initialize_resets_runtime_data_and_preserves_language(self):
        reports_dir = self.project_root / "data" / "reports"
        logs_dir = self.project_root / "data" / "logs"
        notifications_dir = self.project_root / "data" / "notifications"
        for directory in (reports_dir, logs_dir, notifications_dir):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "old.txt").write_text("artifact", encoding="utf-8")
        model_file = self.project_root / "data" / "models" / "model.json"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_text("{}", encoding="utf-8")
        pycache_dir = self.project_root / "__pycache__"
        pycache_dir.mkdir(parents=True, exist_ok=True)
        (pycache_dir / "x.pyc").write_bytes(b"123")

        self.storage.set_state(LANGUAGE_STATE_KEY, "zh")
        self.storage.insert_report_artifact("health", "x")
        self.storage.insert_execution_event("evt", "BTC/USDT", {"reason": "x"})

        service = SystemDataService(self.storage, self.settings)
        result = service.initialize_system()

        self.assertTrue(result["backup_created"])
        self.assertTrue(Path(result["backup_path"]).exists())
        self.assertTrue(Path(result["backup_summary_path"]).exists())
        self.assertGreaterEqual(result["artifact_files_removed"]["reports"], 1)
        self.assertGreaterEqual(result["artifact_files_removed"]["logs"], 1)
        self.assertGreaterEqual(result["artifact_files_removed"]["notifications"], 1)
        self.assertEqual(result["artifact_files_skipped"]["reports"], [])
        self.assertGreaterEqual(result["pycache_dirs_removed"], 1)
        self.assertTrue(model_file.exists())
        self.assertEqual(self.storage.get_state(LANGUAGE_STATE_KEY), "zh")
        with sqlite3.connect(result["backup_path"]) as backup_conn:
            backup_report_count = backup_conn.execute(
                "SELECT COUNT(*) FROM report_artifacts"
            ).fetchone()[0]
            backup_event_count = backup_conn.execute(
                "SELECT COUNT(*) FROM execution_events"
            ).fetchone()[0]
        self.assertEqual(backup_report_count, 1)
        self.assertEqual(backup_event_count, 1)

        with self.storage._conn() as conn:
            for table in ("report_artifacts", "execution_events", "positions", "prediction_runs"):
                count = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
                self.assertEqual(count, 0)

    def test_system_data_service_cleanup_removes_small_artifacts_and_pycache(self):
        reports_dir = self.project_root / "data" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        tiny = reports_dir / "tiny.md"
        keep = reports_dir / "keep.md"
        tiny.write_text("wf-report", encoding="utf-8")
        keep.write_text("this is a longer report body", encoding="utf-8")
        pycache_dir = self.project_root / "analysis" / "__pycache__"
        pycache_dir.mkdir(parents=True, exist_ok=True)
        (pycache_dir / "x.pyc").write_bytes(b"1")

        service = SystemDataService(self.storage, self.settings)
        result = service.cleanup_data()

        self.assertFalse(tiny.exists())
        self.assertFalse(keep.exists())
        self.assertEqual(result["artifact_files_skipped"]["reports"], [])
        self.assertGreaterEqual(result["pycache_dirs_removed"], 1)

    def test_system_data_service_skips_venv_pycache(self):
        venv_pycache_dir = self.project_root / ".venv" / "Lib" / "site-packages" / "__pycache__"
        venv_pycache_dir.mkdir(parents=True, exist_ok=True)
        marker = venv_pycache_dir / "x.pyc"
        marker.write_bytes(b"1")

        service = SystemDataService(self.storage, self.settings)
        service.cleanup_data()

        self.assertTrue(marker.exists())

    def test_main_supports_init_and_cleanup_commands(self):
        with patch.object(main_module, "run_system_data_command", return_value={"report": "init-report"}), patch(
            "sys.argv", ["main.py", "init-system"]
        ), patch("builtins.print") as fake_print:
            main_module.main()
        fake_print.assert_called_with("init-report")

        with patch.object(main_module, "run_system_data_command", return_value={"report": "cleanup-report"}), patch(
            "sys.argv", ["main.py", "cleanup-data"]
        ), patch("builtins.print") as fake_print:
            main_module.main()
        fake_print.assert_called_with("cleanup-report")

    def test_acquire_runtime_lock_handles_create_race_as_existing_lock(self):
        lock_path = self.project_root / "data" / "runtime" / "loop.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        original_open = Path.open
        raced = {"done": False}

        def racing_open(path_obj, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if path_obj == lock_path and mode == "x" and not raced["done"]:
                raced["done"] = True
                with original_open(lock_path, "w", encoding="utf-8") as handle:
                    handle.write(str(os.getpid()))
                raise FileExistsError
            return original_open(path_obj, *args, **kwargs)

        with patch.object(Path, "open", autospec=True, side_effect=racing_open):
            acquired, returned_path, existing_pid = main_module.acquire_runtime_lock(lock_path)

        self.assertFalse(acquired)
        self.assertEqual(returned_path, lock_path)
        self.assertEqual(existing_pid, os.getpid())

    def test_main_loop_command_uses_supervised_loop(self):
        with patch.object(main_module, "run_loop_forever") as run_loop, patch.object(
            main_module, "build_engine"
        ) as build_engine, patch("sys.argv", ["main.py", "loop"]):
            main_module.main()
        run_loop.assert_called_once()
        build_engine.assert_called_once()

    def test_run_once_subprocess_invokes_once_command(self):
        fake_completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
        with patch.object(main_module.subprocess, "run", return_value=fake_completed) as fake_run:
            ok, output = main_module.run_once_subprocess(123)
        self.assertTrue(ok)
        self.assertEqual(output, "ok")
        args, kwargs = fake_run.call_args
        self.assertEqual(args[0], [main_module.sys.executable, "main.py", "once"])
        self.assertEqual(kwargs["timeout"], 123)

    def test_run_position_guard_subprocess_invokes_positions_command(self):
        fake_completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
        with patch.object(main_module.subprocess, "run", return_value=fake_completed) as fake_run:
            ok, output = main_module.run_position_guard_subprocess(321)
        self.assertTrue(ok)
        self.assertEqual(output, "ok")
        args, kwargs = fake_run.call_args
        self.assertEqual(args[0], [main_module.sys.executable, "main.py", "positions"])
        self.assertEqual(kwargs["timeout"], 321)

    def test_run_entry_scan_subprocess_invokes_entries_command(self):
        fake_completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
        with patch.object(main_module.subprocess, "run", return_value=fake_completed) as fake_run:
            ok, output = main_module.run_entry_scan_subprocess(222)
        self.assertTrue(ok)
        self.assertEqual(output, "ok")
        args, kwargs = fake_run.call_args
        self.assertEqual(args[0], [main_module.sys.executable, "main.py", "entries"])
        self.assertEqual(kwargs["timeout"], 222)

    def test_main_positions_command_uses_position_guard(self):
        fake_engine = SimpleNamespace(run_position_guard=lambda: {"status": "ok"})
        with patch.object(main_module, "build_engine", return_value=fake_engine), patch.object(
            main_module,
            "acquire_cycle_lock",
            return_value=(True, Path("/tmp/test-cycle.lock"), None),
        ), patch.object(
            main_module,
            "release_cycle_lock",
        ) as release_lock, patch(
            "sys.argv", ["main.py", "positions"]
        ), patch("builtins.print") as fake_print:
            with self.assertRaises(SystemExit) as raised:
                main_module.main()
        self.assertEqual(raised.exception.code, 0)
        fake_print.assert_called_once_with({"status": "ok"})
        release_lock.assert_called_once()

    def test_main_entries_command_uses_entry_scan(self):
        fake_engine = SimpleNamespace(run_entry_scan=lambda: {"status": "ok"})
        with patch.object(main_module, "build_engine", return_value=fake_engine), patch.object(
            main_module,
            "acquire_cycle_lock",
            return_value=(True, Path("/tmp/test-cycle.lock"), None),
        ), patch.object(
            main_module,
            "release_cycle_lock",
        ) as release_lock, patch(
            "sys.argv", ["main.py", "entries"]
        ), patch("builtins.print") as fake_print:
            with self.assertRaises(SystemExit) as raised:
                main_module.main()
        self.assertEqual(raised.exception.code, 0)
        fake_print.assert_called_once_with({"status": "ok"})
        release_lock.assert_called_once()

    def test_acquire_loop_lock_blocks_when_pid_is_alive(self):
        with patch.object(main_module, "get_settings", return_value=self.settings), patch.object(
            main_module, "process_exists", return_value=True
        ):
            lock_path = main_module.loop_lock_path(self.settings)
            lock_path.write_text("99999", encoding="utf-8")
            acquired, _, existing_pid = main_module.acquire_loop_lock(self.settings)
        self.assertFalse(acquired)
        self.assertEqual(existing_pid, 99999)

    def test_dashboard_render_functions_smoke(self):
        self.storage.set_state(LANGUAGE_STATE_KEY, "zh")
        (self.project_root / "data" / "logs").mkdir(parents=True, exist_ok=True)
        (self.project_root / "data" / "logs" / "app.log").write_text("line1\nline2", encoding="utf-8")

        fake_st = DummyStreamlit()
        render_functions = [
            dashboard_module.render_overview,
            dashboard_module.render_settings,
            dashboard_module.render_ops,
            dashboard_module.render_predictions,
            dashboard_module.render_watchlist,
            dashboard_module.render_training,
            dashboard_module.render_walkforward,
            dashboard_module.render_backtest,
            dashboard_module.render_reports,
            dashboard_module.render_execution,
            dashboard_module.render_health,
            dashboard_module.render_metrics,
            dashboard_module.render_drift,
            dashboard_module.render_abtest,
            dashboard_module.render_guards,
            dashboard_module.render_failures,
            dashboard_module.render_scheduler,
            dashboard_module.render_logs,
            dashboard_module.render_operations,
        ]
        with patch.object(dashboard_module, "st", fake_st), patch.object(
            dashboard_module, "get_settings", return_value=self.settings
        ):
            for render in render_functions:
                render()

    def test_dashboard_render_ops_includes_nextgen_live_queue_table(self):
        self.storage.set_state(LANGUAGE_STATE_KEY, "zh")
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_run",
            "SYSTEM",
            {
                "status": "ok",
                "trigger": "scheduler",
                "repair_queue_hold_priority_count": 2,
                "repair_queue_postponed_rebuild_count": 1,
                "repair_queue_reprioritized_count": 3,
            },
        )
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_run",
            "SYSTEM",
            {
                "status": "ok",
                "trigger": "manual_recovery_required",
                "repair_queue_hold_priority_count": 1,
                "repair_queue_postponed_rebuild_count": 0,
                "repair_queue_reprioritized_count": 1,
            },
        )

        fake_st = DummyStreamlit()
        captured_frames = []
        fake_st.dataframe = lambda data, *args, **kwargs: captured_frames.append(data)

        with patch.object(dashboard_module, "st", fake_st), patch.object(
            dashboard_module, "get_settings", return_value=self.settings
        ):
            dashboard_module.render_ops()

        self.assertTrue(captured_frames)
        queue_frames = [
            frame
            for frame in captured_frames
            if hasattr(frame, "columns")
            and "hold_repair_count" in list(frame.columns)
            and "postponed_rebuild_count" in list(frame.columns)
            and "reprioritized_count" in list(frame.columns)
        ]
        self.assertTrue(queue_frames)

    def test_overview_renders_model_lifecycle_summary_caption(self):
        self.storage.set_state(LANGUAGE_STATE_KEY, "zh")
        self.storage.insert_training_run(
            {
                "symbol": "BTC/USDT",
                "rows": 500,
                "feature_count": 10,
                "positives": 250,
                "negatives": 250,
                "model_path": str(self.project_root / "data" / "models" / "xgboost_challenger_BTC_USDT.json"),
                "active_model_path": str(self.project_root / "data" / "models" / "xgboost_v2_BTC_USDT.json"),
                "challenger_model_path": str(self.project_root / "data" / "models" / "xgboost_challenger_BTC_USDT.json"),
                "trained_with_xgboost": True,
                "promoted_to_active": False,
                "promotion_status": "canary_pending",
                "promotion_reason": "candidate_higher_walkforward_return",
            }
        )
        self.storage.set_json_state(
            "model_promotion_observations",
            {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "status": "observing",
                    "active_model_path": str(self.project_root / "data" / "models" / "xgboost_v2_BTC_USDT.json"),
                    "challenger_model_path": str(self.project_root / "data" / "models" / "xgboost_challenger_BTC_USDT.json"),
                }
            },
        )
        self.storage.insert_execution_event(
            "model_rollback",
            "BTC/USDT",
            {"reason": "post_promotion_accuracy_0.25_below_0.54"},
        )

        fake_st = DummyStreamlit()
        captured_captions: list[str] = []
        fake_st.caption = lambda text, *args, **kwargs: captured_captions.append(str(text))

        with patch.object(dashboard_module, "st", fake_st), patch.object(
            dashboard_module, "get_settings", return_value=self.settings
        ):
            dashboard_module.render_overview()

        combined = "\n".join(captured_captions)
        self.assertIn("模型生命周期摘要", combined)
        self.assertIn("xgboost_v2_BTC_USDT.json", combined)
        self.assertIn("xgboost_challenger_BTC_USDT.json", combined)
        self.assertIn("观察中", combined)
        self.assertIn("post_promotion_accuracy_0.25_below_0.54", combined)

    def test_dashboard_render_overview_filters_unsupported_renderer_kwargs(self):
        called = {}

        def legacy_renderer(
            *,
            st,
            t,
            current_language,
            query_df,
            query_one,
            load_json,
            parse_markdown_metrics,
            parse_report_history,
            to_numeric_percent,
            display_df,
        ):
            called["ok"] = True

        with patch.object(dashboard_module, "render_overview_page", legacy_renderer):
            dashboard_module.render_overview()

        self.assertTrue(called.get("ok"))
