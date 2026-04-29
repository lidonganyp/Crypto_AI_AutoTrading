"""Post-promotion observation helpers."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from core.promotion_observation_accept_runtime import (
    PromotionObservationAcceptRuntime,
)
from core.promotion_observation_decision_runtime import (
    PromotionObservationDecisionRuntime,
)
from core.promotion_observation_rollback_runtime import (
    PromotionObservationRollbackRuntime,
)
from strategy.model_trainer import model_path_for_symbol


class PromotionObservationRuntime:
    """Manage promoted-model observation, acceptance, and rollback."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.decision_runtime = PromotionObservationDecisionRuntime(self)
        self.accept_runtime = PromotionObservationAcceptRuntime(self)
        self.rollback_runtime = PromotionObservationRollbackRuntime(self)

    def register_promoted_model_observation(
        self,
        symbol: str,
        summary,
        now: datetime,
    ) -> None:
        backup_model_path = str(
            getattr(summary, "previous_active_backup_path", "") or ""
        ).strip()
        if not backup_model_path:
            return
        active_model_path = str(
            getattr(summary, "active_model_path", "") or summary.model_path or ""
        ).strip()
        if not active_model_path:
            return
        active_model_id = self.runtime.resolved_model_id(
            str(getattr(summary, "model_id", "") or ""),
            active_model_path,
        )
        backup_model_id = self.runtime.resolved_model_id(
            str(getattr(summary, "incumbent_model_id", "") or ""),
            backup_model_path,
        )
        training_metadata = dict(getattr(summary, "__dict__", {}) or {})
        baseline = self.runtime.training_objective_baseline(training_metadata)
        requirements = self.runtime.promotion_adaptive_requirements(
            symbol,
            training_metadata=training_metadata,
        )
        observations = self.runtime.get_model_promotion_observations()
        observations[symbol] = {
            "symbol": symbol,
            "promoted_at": now.isoformat(),
            "active_model_path": active_model_path,
            "active_model_id": active_model_id,
            "challenger_model_path": str(
                model_path_for_symbol(
                    self.runtime.challenger_predictor_base_path, symbol
                )
            ),
            "active_model_signature": list(
                self.runtime.model_file_signature(Path(active_model_path))
            ),
            "backup_model_path": backup_model_path,
            "backup_model_id": backup_model_id,
            "backup_meta_path": str(
                getattr(summary, "previous_active_backup_meta_path", "") or ""
            ).strip(),
            "baseline_holdout_accuracy": float(
                getattr(summary, "candidate_holdout_accuracy", 0.0) or 0.0
            ),
            "baseline_objective_score": float(
                baseline.get("baseline_objective_score", 0.0) or 0.0
            ),
            "baseline_objective_quality": float(
                baseline.get("baseline_objective_quality", 0.0) or 0.0
            ),
            "baseline_expectancy_pct": float(
                baseline.get("baseline_expectancy_pct", 0.0) or 0.0
            ),
            "baseline_profit_factor": float(
                baseline.get("baseline_profit_factor", 0.0) or 0.0
            ),
            "baseline_max_drawdown_pct": float(
                baseline.get("baseline_max_drawdown_pct", 0.0) or 0.0
            ),
            "baseline_trade_win_rate": float(
                baseline.get("baseline_trade_win_rate", 0.0) or 0.0
            ),
            "baseline_avg_trade_return_pct": float(
                baseline.get("baseline_avg_trade_return_pct", 0.0) or 0.0
            ),
            "baseline_walkforward_summary": dict(
                getattr(summary, "candidate_walkforward_summary", {}) or {}
            ),
            "recent_walkforward_baseline_summary": dict(
                getattr(summary, "recent_walkforward_baseline_summary", {}) or {}
            ),
            "min_evaluations": int(
                requirements.get("observation_min_evaluations", 0) or 0
            ),
            "max_observation_age_hours": int(
                requirements.get("observation_max_age_hours", 72) or 72
            ),
            "adaptive_requirements_source": str(
                requirements.get("requirements_source", "default") or "default"
            ),
            "adaptive_volatility_pct": float(
                requirements.get("volatility_pct", 0.0) or 0.0
            ),
            "adaptive_reference_holding_hours": float(
                requirements.get("reference_holding_hours", 0.0) or 0.0
            ),
            "adaptive_reference_trade_count": int(
                requirements.get("reference_trade_count", 0) or 0
            ),
            "adaptive_requirement_scale": float(
                requirements.get("requirement_scale", 1.0) or 1.0
            ),
            "status": "observing",
            "training_metadata": training_metadata,
        }
        self.runtime.set_model_promotion_observations(observations)
        self.runtime.upsert_model_registry_entry(
            symbol=symbol,
            model_id=active_model_id,
            model_path=active_model_path,
            role="active",
            stage="observing",
            active=True,
            metadata={
                "promoted_at": now.isoformat(),
                "backup_model_id": backup_model_id,
            },
        )
        self.runtime.storage.insert_execution_event(
            "model_observation_started",
            symbol,
            {
                "promoted_at": now.isoformat(),
                "active_model_path": active_model_path,
                "backup_model_path": backup_model_path,
                "baseline_holdout_accuracy": getattr(
                    summary,
                    "candidate_holdout_accuracy",
                    0.0,
                ),
                "baseline_expectancy_pct": baseline.get(
                    "baseline_expectancy_pct", 0.0
                ),
                "baseline_profit_factor": baseline.get(
                    "baseline_profit_factor", 0.0
                ),
                "baseline_max_drawdown_pct": baseline.get(
                    "baseline_max_drawdown_pct",
                    0.0,
                ),
                "baseline_walkforward_summary": getattr(
                    summary,
                    "candidate_walkforward_summary",
                    {},
                ),
                "min_evaluations": int(
                    requirements.get("observation_min_evaluations", 0) or 0
                ),
                "max_observation_age_hours": int(
                    requirements.get("observation_max_age_hours", 72) or 72
                ),
                "adaptive_volatility_pct": float(
                    requirements.get("volatility_pct", 0.0) or 0.0
                ),
                "adaptive_reference_holding_hours": float(
                    requirements.get("reference_holding_hours", 0.0) or 0.0
                ),
                "adaptive_requirement_scale": float(
                    requirements.get("requirement_scale", 1.0) or 1.0
                ),
            },
        )

    def observe_promoted_models(self, now: datetime) -> dict[str, int]:
        return self.decision_runtime.observe_promoted_models(now)

    def model_self_heal_interval_hours(self) -> int:
        scheduled = int(
            getattr(self.runtime.settings.scheduler, "training_cron_hours", 0) or 0
        )
        return max(1, min(24, scheduled if scheduled > 0 else 1))

    def runtime_model_path_for_symbol(self, symbol: str) -> Path:
        with self.runtime.storage._conn() as conn:
            row = conn.execute(
                "SELECT metadata_json, model_path FROM training_runs "
                "WHERE symbol = ? ORDER BY created_at DESC, id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        if row is not None:
            try:
                metadata = json.loads(row["metadata_json"])
            except Exception:
                metadata = {}
            model_path_raw = str(
                metadata.get("active_model_path")
                or metadata.get("model_path")
                or row["model_path"]
                or ""
            ).strip()
            if model_path_raw:
                return Path(model_path_raw)
        return model_path_for_symbol(self.runtime.predictor_base_path, symbol)

    def post_promotion_execution_metrics(
        self,
        symbol: str,
        promoted_at: str,
        limit: int = 200,
    ) -> dict[str, float | int]:
        with self.runtime.storage._conn() as conn:
            rows = conn.execute(
                """SELECT pe.is_correct
                   FROM prediction_evaluations pe
                   JOIN prediction_runs pr
                     ON pr.symbol = pe.symbol
                    AND pr.timestamp = pe.timestamp
                  WHERE pe.symbol = ?
                    AND pe.evaluation_type = 'execution'
                    AND pr.timestamp >= ?
                  ORDER BY pe.created_at DESC
                  LIMIT ?""",
                (symbol, promoted_at, limit),
            ).fetchall()
        eval_count = len(rows)
        accuracy = (
            sum(int(row["is_correct"] or 0) for row in rows) / eval_count
            if eval_count
            else 0.0
        )
        return {"eval_count": eval_count, "accuracy": accuracy}

    @staticmethod
    def remove_file_if_exists(path_str: str) -> None:
        path = Path(path_str or "")
        if not str(path):
            return
        path.unlink(missing_ok=True)

    @staticmethod
    def write_json_file_atomic(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def accept_promoted_model(
        self,
        symbol: str,
        observation: dict,
        now: datetime,
        metrics: dict[str, float | int],
    ) -> None:
        self.accept_runtime.accept_promoted_model(
            symbol,
            observation,
            now,
            metrics,
        )

    def rollback_promoted_model(
        self,
        symbol: str,
        observation: dict,
        now: datetime,
        reason: str,
        metrics: dict[str, float | int],
    ) -> None:
        self.rollback_runtime.rollback_promoted_model(
            symbol,
            observation,
            now,
            reason,
            metrics,
        )
