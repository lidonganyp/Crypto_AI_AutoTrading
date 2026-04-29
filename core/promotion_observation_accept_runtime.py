"""Post-promotion acceptance helpers."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


class PromotionObservationAcceptRuntime:
    """Accept promoted models that pass observation."""

    def __init__(self, observation_runtime):
        self.observation_runtime = observation_runtime
        self.runtime = observation_runtime.runtime

    def accept_promoted_model(
        self,
        symbol: str,
        observation: dict,
        now: datetime,
        metrics: dict[str, float | int],
    ) -> None:
        active_model_id = self.runtime.observation_active_model_id(observation)
        active_model_path = Path(str(observation.get("active_model_path") or ""))
        self.observation_runtime.remove_file_if_exists(
            str(observation.get("backup_model_path") or "")
        )
        self.observation_runtime.remove_file_if_exists(
            str(observation.get("backup_meta_path") or "")
        )
        scorecard = self.runtime.record_model_scorecard(
            symbol=symbol,
            model_id=active_model_id,
            model_path=str(active_model_path),
            stage="observation_accepted",
            evaluation_type="execution",
            started_at=str(observation.get("promoted_at") or ""),
            extra_metadata={"accepted_at": now.isoformat()},
        )
        if str(active_model_path):
            active_metadata = self.runtime.read_model_metadata(active_model_path)
            active_metadata.update(
                {
                    "observation_accepted_at": now.isoformat(),
                    "post_promotion_accept_eval_count": int(
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
                    "post_promotion_objective_score": float(
                        metrics.get("objective_score", 0.0) or 0.0
                    ),
                    "post_promotion_objective_quality": float(
                        metrics.get("objective_quality", 0.0) or 0.0
                    ),
                }
            )
            self.observation_runtime.write_json_file_atomic(
                active_model_path.with_suffix(".meta.json"),
                active_metadata,
            )
        self.runtime.upsert_model_registry_entry(
            symbol=symbol,
            model_id=active_model_id,
            model_path=str(active_model_path),
            role="active",
            stage="accepted",
            active=True,
            metadata={
                "accepted_at": now.isoformat(),
                "scorecard": scorecard,
            },
        )
        self.runtime.storage.insert_execution_event(
            "model_observation_accepted",
            symbol,
            {
                "accepted_at": now.isoformat(),
                "eval_count": int(metrics.get("eval_count", 0) or 0),
                "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
                "expectancy_pct": float(metrics.get("expectancy_pct", 0.0) or 0.0),
                "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
                "max_drawdown_pct": float(
                    metrics.get("max_drawdown_pct", 0.0) or 0.0
                ),
                "active_model_path": str(active_model_path),
            },
        )
