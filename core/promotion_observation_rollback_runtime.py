"""Post-promotion rollback helpers."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


class PromotionObservationRollbackRuntime:
    """Rollback promoted models that fail observation."""

    def __init__(self, observation_runtime):
        self.observation_runtime = observation_runtime
        self.runtime = observation_runtime.runtime

    def rollback_promoted_model(
        self,
        symbol: str,
        observation: dict,
        now: datetime,
        reason: str,
        metrics: dict[str, float | int],
    ) -> None:
        active_model_path = Path(str(observation.get("active_model_path") or ""))
        backup_model_path = Path(str(observation.get("backup_model_path") or ""))
        backup_meta_path = Path(str(observation.get("backup_meta_path") or ""))
        challenger_model_path = Path(
            str(observation.get("challenger_model_path") or "")
        )
        active_model_id = self.runtime.observation_active_model_id(observation)
        backup_model_id = self.runtime.observation_backup_model_id(observation)
        if not str(active_model_path) or not backup_model_path.exists():
            return
        active_meta_path = active_model_path.with_suffix(".meta.json")
        archived_challenger_meta_path = challenger_model_path.with_suffix(
            ".meta.json"
        )
        if str(challenger_model_path):
            challenger_model_path.parent.mkdir(parents=True, exist_ok=True)
            if active_model_path.exists():
                shutil.copy2(active_model_path, challenger_model_path)
            rollback_metadata = dict(observation.get("training_metadata", {}) or {})
            rollback_metadata.update(
                {
                    "symbol": symbol,
                    "model_id": active_model_id,
                    "model_path": str(challenger_model_path),
                    "active_model_path": str(
                        backup_model_path
                        if backup_model_path.exists()
                        else active_model_path
                    ),
                    "active_model_id": backup_model_id,
                    "challenger_model_path": str(challenger_model_path),
                    "challenger_model_id": active_model_id,
                    "promoted_to_active": False,
                    "promotion_status": "rolled_back",
                    "rollback_reason": reason,
                    "rollback_at": now.isoformat(),
                    "post_promotion_eval_count": int(
                        metrics.get("eval_count", 0) or 0
                    ),
                    "post_promotion_accuracy": float(
                        metrics.get("accuracy", 0.0) or 0.0
                    ),
                    "post_promotion_expectancy_pct": float(
                        metrics.get("expectancy_pct", 0.0) or 0.0
                    ),
                    "post_promotion_profit_factor": float(
                        metrics.get("profit_factor", 0.0) or 0.0
                    ),
                    "post_promotion_max_drawdown_pct": float(
                        metrics.get("max_drawdown_pct", 0.0) or 0.0
                    ),
                    "post_promotion_avg_trade_return_pct": float(
                        metrics.get("avg_trade_return_pct", 0.0) or 0.0
                    ),
                }
            )
            self.observation_runtime.write_json_file_atomic(
                archived_challenger_meta_path,
                rollback_metadata,
            )
        active_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_model_path, active_model_path)
        if backup_meta_path.exists():
            shutil.copy2(backup_meta_path, active_meta_path)
        self.runtime.clear_symbol_models(symbol)
        scorecard = self.runtime.record_model_scorecard(
            symbol=symbol,
            model_id=active_model_id,
            model_path=str(active_model_path),
            stage="rolled_back",
            evaluation_type="execution",
            started_at=str(observation.get("promoted_at") or ""),
            extra_metadata={"rolled_back_at": now.isoformat(), "reason": reason},
        )
        self.runtime.upsert_model_registry_entry(
            symbol=symbol,
            model_id=active_model_id,
            model_path=str(challenger_model_path),
            role="challenger",
            stage="rolled_back",
            active=False,
            metadata={
                "rolled_back_at": now.isoformat(),
                "reason": reason,
                "scorecard": scorecard,
            },
        )
        self.runtime.upsert_model_registry_entry(
            symbol=symbol,
            model_id=backup_model_id,
            model_path=str(active_model_path),
            role="active",
            stage="restored",
            active=True,
            metadata={
                "restored_at": now.isoformat(),
                "restored_from_backup": str(backup_model_path),
            },
        )
        self.runtime.storage.insert_execution_event(
            "model_rollback",
            symbol,
            {
                "rolled_back_at": now.isoformat(),
                "reason": reason,
                "eval_count": int(metrics.get("eval_count", 0) or 0),
                "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
                "expectancy_pct": float(metrics.get("expectancy_pct", 0.0) or 0.0),
                "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
                "max_drawdown_pct": float(
                    metrics.get("max_drawdown_pct", 0.0) or 0.0
                ),
                "active_model_path": str(active_model_path),
                "restored_from": str(backup_model_path),
                "archived_challenger_model_path": str(challenger_model_path),
                "archived_challenger_meta_path": str(
                    archived_challenger_meta_path
                ),
            },
        )
        self.runtime.report_runtime.save_report_artifact(
            report_type="model_rollback",
            symbol=symbol,
            content="\n".join(
                [
                    f"# Model Rollback: {symbol}",
                    f"- rolled_back_at: {now.isoformat()}",
                    f"- reason: {reason}",
                    f"- eval_count: {int(metrics.get('eval_count', 0) or 0)}",
                    f"- accuracy: {float(metrics.get('accuracy', 0.0) or 0.0):.4f}",
                    f"- expectancy_pct: {float(metrics.get('expectancy_pct', 0.0) or 0.0):.4f}",
                    f"- profit_factor: {float(metrics.get('profit_factor', 0.0) or 0.0):.4f}",
                    f"- max_drawdown_pct: {float(metrics.get('max_drawdown_pct', 0.0) or 0.0):.4f}",
                    f"- restored_from: {backup_model_path}",
                    f"- archived_challenger_model_path: {challenger_model_path}",
                ]
            ),
            extension="md",
        )
        self.runtime.notifier.notify(
            "model_rollback",
            "模型已自动回滚",
            f"{symbol} | {reason}",
            level="warning",
        )
        self.observation_runtime.remove_file_if_exists(str(backup_model_path))
        self.observation_runtime.remove_file_if_exists(str(backup_meta_path))
