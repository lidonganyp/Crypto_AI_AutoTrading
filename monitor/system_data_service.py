"""System data initialization and cleanup helpers."""
from __future__ import annotations

import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

from config import Settings
from core.i18n import LANGUAGE_STATE_KEY
from core.storage import Storage


class SystemDataService:
    """Reset runtime data and clean generated artifacts safely."""

    ARTIFACT_DIRS = ("reports", "logs", "notifications")

    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings
        self.project_root = Path(settings.app.project_root)
        self.data_root = self.project_root / "data"

    def initialize_system(self, preserve_language: bool = True) -> dict:
        backup_result = self._backup_runtime_db()
        preserve_state_keys = [LANGUAGE_STATE_KEY] if preserve_language else []
        reset_summary = self.storage.reset_runtime_data(
            preserve_state_keys=preserve_state_keys,
        )
        artifact_result = self._purge_artifact_dirs(remove_all=True)
        pycache_removed = self._remove_pycache_dirs()
        self.storage.vacuum()
        return {
            "backup_created": backup_result["created"],
            "backup_path": backup_result["path"],
            "backup_summary_path": backup_result["summary_path"],
            "reset_tables": reset_summary,
            "artifact_files_removed": artifact_result["removed"],
            "artifact_files_skipped": artifact_result["skipped"],
            "pycache_dirs_removed": pycache_removed,
            "models_preserved": True,
        }

    def cleanup_data(self) -> dict:
        artifact_result = self._purge_artifact_dirs(remove_all=False)
        pycache_removed = self._remove_pycache_dirs()
        self.storage.vacuum()
        return {
            "artifact_files_removed": artifact_result["removed"],
            "artifact_files_skipped": artifact_result["skipped"],
            "pycache_dirs_removed": pycache_removed,
            "models_preserved": True,
        }

    def _purge_artifact_dirs(self, remove_all: bool) -> dict[str, dict]:
        removed_summary: dict[str, int] = {}
        skipped_summary: dict[str, list[str]] = {}
        today_token = datetime.now().strftime("%Y-%m-%d")
        for name in self.ARTIFACT_DIRS:
            directory = self.data_root / name
            removed = 0
            skipped: list[str] = []
            if directory.exists():
                for file_path in directory.glob("*"):
                    if not file_path.is_file():
                        continue
                    if remove_all or self._should_delete_cleanup_artifact(
                        name=name,
                        path=file_path,
                        today_token=today_token,
                    ):
                        try:
                            file_path.unlink(missing_ok=True)
                            removed += 1
                        except PermissionError:
                            skipped.append(file_path.name)
                            continue
            directory.mkdir(parents=True, exist_ok=True)
            removed_summary[name] = removed
            skipped_summary[name] = skipped
        return {"removed": removed_summary, "skipped": skipped_summary}

    def _backup_runtime_db(self) -> dict[str, str | bool]:
        source_db = Path(self.storage.db_path)
        if not source_db.exists():
            return {
                "created": False,
                "path": "",
                "summary_path": "",
            }

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_dir = self.data_root / "backups" / f"pre-init-system-{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{source_db.stem}-pre-init-system.db"
        with self.storage._conn() as source_conn:
            dest_conn = sqlite3.connect(str(backup_path))
            try:
                source_conn.backup(dest_conn)
                dest_conn.commit()
            finally:
                dest_conn.close()

        summary_path = backup_dir / "summary.txt"
        summary_path.write_text(
            "\n".join(
                [
                    f"timestamp={timestamp}",
                    f"source_db={source_db}",
                    f"backup_db={backup_path}",
                    "reason=init-system-auto-backup",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "created": True,
            "path": str(backup_path),
            "summary_path": str(summary_path),
        }

    @staticmethod
    def _should_delete_cleanup_artifact(name: str, path: Path, today_token: str) -> bool:
        if name == "reports":
            return True
        if name in {"logs", "notifications"}:
            return today_token not in path.name
        return False

    def _remove_pycache_dirs(self) -> int:
        removed = 0
        for path in self.project_root.rglob("__pycache__"):
            if ".venv" in path.parts:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed
