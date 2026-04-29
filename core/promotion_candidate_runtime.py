"""Candidate-stage model lifecycle helpers."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from core.promotion_candidate_live_runtime import PromotionCandidateLiveRuntime
from core.promotion_candidate_shadow_runtime import PromotionCandidateShadowRuntime


class PromotionCandidateRuntime:
    """Manage canary candidate registration, scoring, and promotion decisions."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.shadow_runtime = PromotionCandidateShadowRuntime(self)
        self.live_runtime = PromotionCandidateLiveRuntime(self)

    def register_promotion_candidate(
        self,
        symbol: str,
        summary,
        now: datetime,
    ) -> None:
        active_model_path = str(
            getattr(summary, "active_model_path", "") or ""
        ).strip()
        challenger_model_path = str(
            getattr(summary, "challenger_model_path", "")
            or getattr(summary, "model_path", "")
            or ""
        ).strip()
        if not active_model_path or not challenger_model_path:
            return
        active_path = Path(active_model_path)
        challenger_path = Path(challenger_model_path)
        if not active_path.exists() or not challenger_path.exists():
            return
        active_model_id = self.runtime.resolved_model_id(
            str(getattr(summary, "active_model_id", "") or ""),
            active_model_path,
        )
        challenger_model_id = self.runtime.resolved_model_id(
            str(getattr(summary, "model_id", "") or ""),
            challenger_model_path,
        )
        training_metadata = dict(getattr(summary, "__dict__", {}) or {})
        baseline = self.runtime.training_objective_baseline(training_metadata)
        requirements = self.runtime.promotion_adaptive_requirements(
            symbol,
            training_metadata=training_metadata,
        )
        candidates = self.runtime.get_model_promotion_candidates()
        candidates[symbol] = {
            "symbol": symbol,
            "registered_at": now.isoformat(),
            "status": "shadow",
            "active_model_path": active_model_path,
            "active_model_id": active_model_id,
            "active_model_signature": list(
                self.runtime.model_file_signature(active_path)
            ),
            "challenger_model_path": challenger_model_path,
            "challenger_model_id": challenger_model_id,
            "challenger_model_signature": list(
                self.runtime.model_file_signature(challenger_path)
            ),
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
            "promotion_reason": str(
                getattr(summary, "promotion_reason", "") or ""
            ),
            "min_shadow_evaluations": int(
                requirements.get("shadow_min_evaluations", 0) or 0
            ),
            "min_live_evaluations": int(
                requirements.get("live_min_evaluations", 0) or 0
            ),
            "max_shadow_age_hours": int(
                requirements.get("shadow_max_age_hours", 72) or 72
            ),
            "max_live_age_hours": int(
                requirements.get("live_max_age_hours", 72) or 72
            ),
            "live_allocation_pct": self.runtime.promotion_live_allocation_pct(),
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
            "training_metadata": training_metadata,
        }
        self.runtime.set_model_promotion_candidates(candidates)
        self.runtime.storage.insert_execution_event(
            "model_promotion_candidate_started",
            symbol,
            {
                "registered_at": now.isoformat(),
                "active_model_path": active_model_path,
                "challenger_model_path": challenger_model_path,
                "promotion_reason": getattr(summary, "promotion_reason", ""),
                "candidate_holdout_accuracy": getattr(
                    summary,
                    "candidate_holdout_accuracy",
                    0.0,
                ),
                "min_shadow_evaluations": int(
                    requirements.get("shadow_min_evaluations", 0) or 0
                ),
                "min_live_evaluations": int(
                    requirements.get("live_min_evaluations", 0) or 0
                ),
                "max_shadow_age_hours": int(
                    requirements.get("shadow_max_age_hours", 72) or 72
                ),
                "max_live_age_hours": int(
                    requirements.get("live_max_age_hours", 72) or 72
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
        self.runtime.upsert_model_registry_entry(
            symbol=symbol,
            model_id=challenger_model_id,
            model_path=challenger_model_path,
            role="challenger",
            stage="canary_shadow",
            active=False,
            metadata={
                "promotion_status": getattr(summary, "promotion_status", ""),
                "promotion_reason": getattr(summary, "promotion_reason", ""),
                "registered_at": now.isoformat(),
                **baseline,
            },
            created_at=now.isoformat(),
        )
        self.runtime.upsert_model_registry_entry(
            symbol=symbol,
            model_id=active_model_id,
            model_path=active_model_path,
            role="active",
            stage="active",
            active=True,
            metadata={
                "registered_at": now.isoformat(),
                "source": "pre_canary_incumbent",
            },
        )

    def sync_model_registry_from_training_summary(
        self,
        symbol: str,
        summary,
        now: datetime,
    ) -> None:
        model_path = str(getattr(summary, "model_path", "") or "")
        model_id = self.runtime.resolved_model_id(
            str(getattr(summary, "model_id", "") or ""),
            model_path,
        )
        active_model_path = str(getattr(summary, "active_model_path", "") or "")
        active_model_id = self.runtime.resolved_model_id(
            str(getattr(summary, "active_model_id", "") or ""),
            active_model_path,
        )
        role = (
            "active" if getattr(summary, "promoted_to_active", False) else "challenger"
        )
        stage = str(
            getattr(summary, "promotion_status", "")
            or getattr(summary, "reason", "")
            or "trained"
        )
        self.runtime.upsert_model_registry_entry(
            symbol=symbol,
            model_id=model_id,
            model_path=model_path,
            role=role,
            stage=stage,
            active=bool(getattr(summary, "promoted_to_active", False)),
            metadata={
                "promotion_reason": str(
                    getattr(summary, "promotion_reason", "") or ""
                ),
                "trained_at": now.isoformat(),
                "holdout_accuracy": float(
                    getattr(summary, "candidate_holdout_accuracy", 0.0) or 0.0
                ),
            },
            created_at=now.isoformat(),
        )
        if active_model_path and active_model_id and active_model_id != model_id:
            incumbent_stage = (
                "active"
                if not getattr(summary, "promoted_to_active", False)
                else "superseded"
            )
            self.runtime.upsert_model_registry_entry(
                symbol=symbol,
                model_id=active_model_id,
                model_path=active_model_path,
                role="active" if incumbent_stage == "active" else "backup",
                stage=incumbent_stage,
                active=incumbent_stage == "active",
                metadata={"synced_at": now.isoformat()},
            )

    def observe_promotion_candidates(self, now: datetime) -> dict[str, int]:
        candidates = self.runtime.get_model_promotion_candidates()
        if not candidates:
            return {"shadow_to_live": 0, "promoted": 0, "rejected": 0}
        shadow_to_live = 0
        promoted = 0
        rejected = 0
        changed = False
        for symbol, candidate in list(candidates.items()):
            registered_at = str(candidate.get("registered_at") or "")
            if not registered_at:
                candidates.pop(symbol, None)
                changed = True
                continue
            active_model_path = Path(str(candidate.get("active_model_path") or ""))
            challenger_model_path = Path(str(candidate.get("challenger_model_path") or ""))
            if (
                list(self.runtime.model_file_signature(active_model_path))
                != list(candidate.get("active_model_signature") or [])
                or list(self.runtime.model_file_signature(challenger_model_path))
                != list(candidate.get("challenger_model_signature") or [])
            ):
                candidates.pop(symbol, None)
                changed = True
                continue

            status = str(candidate.get("status") or "shadow")
            challenger_model_id = self.runtime.candidate_challenger_model_id(candidate)
            active_model_id = self.runtime.candidate_active_model_id(candidate)
            baseline_accuracy = float(
                candidate.get("baseline_holdout_accuracy", 0.0) or 0.0
            )
            adaptive_requirements = self.runtime.promotion_adaptive_requirements(
                symbol,
                training_metadata=dict(candidate.get("training_metadata", {}) or {}),
            )
            if status == "live":
                action, updated_candidate = self.live_runtime.observe_live_candidate(
                    symbol=symbol,
                    candidate=candidate,
                    now=now,
                    registered_at=registered_at,
                    challenger_model_id=challenger_model_id,
                    active_model_id=active_model_id,
                    baseline_accuracy=baseline_accuracy,
                    adaptive_requirements=adaptive_requirements,
                )
            else:
                action, updated_candidate = self.shadow_runtime.observe_shadow_candidate(
                    symbol=symbol,
                    candidate=candidate,
                    now=now,
                    registered_at=registered_at,
                    challenger_model_id=challenger_model_id,
                    active_model_id=active_model_id,
                    baseline_accuracy=baseline_accuracy,
                    adaptive_requirements=adaptive_requirements,
                )
            if action == "continue":
                continue
            if action == "promoted":
                candidates.pop(symbol, None)
                changed = True
                promoted += 1
                continue
            if action == "shadow_to_live":
                candidates[symbol] = updated_candidate or dict(candidate)
                changed = True
                shadow_to_live += 1
                continue
            if action == "rejected":
                candidates.pop(symbol, None)
                changed = True
                rejected += 1
                continue
        if changed:
            self.runtime.set_model_promotion_candidates(candidates)
        return {
            "shadow_to_live": shadow_to_live,
            "promoted": promoted,
            "rejected": rejected,
        }

    def advance_promotion_candidate_to_live(
        self,
        symbol: str,
        candidate: dict,
        now: datetime,
        shadow_metrics: dict[str, float | int],
        champion_metrics: dict[str, float | int],
    ) -> dict:
        updated = dict(candidate)
        challenger_model_id = self.runtime.candidate_challenger_model_id(updated)
        updated["status"] = "live"
        updated["live_started_at"] = now.isoformat()
        updated["challenger_model_id"] = challenger_model_id
        updated["active_model_id"] = self.runtime.candidate_active_model_id(updated)
        updated["shadow_eval_count"] = int(
            shadow_metrics.get("eval_count", shadow_metrics.get("sample_count", 0))
            or 0
        )
        updated["shadow_executed_count"] = int(
            shadow_metrics.get("executed_count", 0) or 0
        )
        updated["shadow_accuracy"] = float(
            shadow_metrics.get("accuracy", 0.0) or 0.0
        )
        updated["shadow_expectancy_pct"] = float(
            shadow_metrics.get("expectancy_pct", 0.0) or 0.0
        )
        updated["shadow_profit_factor"] = float(
            shadow_metrics.get("profit_factor", 0.0) or 0.0
        )
        updated["shadow_max_drawdown_pct"] = float(
            shadow_metrics.get("max_drawdown_pct", 0.0) or 0.0
        )
        updated["shadow_champion_accuracy"] = float(
            champion_metrics.get("accuracy", 0.0) or 0.0
        )
        self.runtime.storage.insert_execution_event(
            "model_promotion_live_started",
            symbol,
            {
                "live_started_at": now.isoformat(),
                "shadow_eval_count": updated["shadow_eval_count"],
                "shadow_accuracy": updated["shadow_accuracy"],
                "shadow_expectancy_pct": updated["shadow_expectancy_pct"],
                "shadow_profit_factor": updated["shadow_profit_factor"],
                "shadow_max_drawdown_pct": updated["shadow_max_drawdown_pct"],
                "champion_accuracy": updated["shadow_champion_accuracy"],
                "challenger_model_path": updated.get("challenger_model_path"),
            },
        )
        self.runtime.upsert_model_registry_entry(
            symbol=symbol,
            model_id=challenger_model_id,
            model_path=str(updated.get("challenger_model_path") or ""),
            role="challenger",
            stage="canary_live",
            active=False,
            metadata={
                "live_started_at": now.isoformat(),
                "shadow_eval_count": updated["shadow_eval_count"],
                "shadow_accuracy": updated["shadow_accuracy"],
                "shadow_expectancy_pct": updated["shadow_expectancy_pct"],
                "shadow_profit_factor": updated["shadow_profit_factor"],
                "shadow_max_drawdown_pct": updated["shadow_max_drawdown_pct"],
                "shadow_champion_accuracy": updated["shadow_champion_accuracy"],
            },
        )
        return updated

    def reject_promotion_candidate(
        self,
        symbol: str,
        candidate: dict,
        now: datetime,
        reason: str,
        metrics: dict[str, float | int],
    ) -> None:
        challenger_model_path = Path(
            str(candidate.get("challenger_model_path") or "")
        )
        challenger_meta_path = challenger_model_path.with_suffix(".meta.json")
        challenger_model_id = self.runtime.candidate_challenger_model_id(candidate)
        active_model_id = self.runtime.candidate_active_model_id(candidate)
        metadata = dict(candidate.get("training_metadata", {}) or {})
        metadata.update(
            {
                "symbol": symbol,
                "model_id": challenger_model_id,
                "model_path": str(challenger_model_path),
                "active_model_path": str(candidate.get("active_model_path") or ""),
                "challenger_model_path": str(challenger_model_path),
                "challenger_model_id": challenger_model_id,
                "active_model_id": active_model_id,
                "promoted_to_active": False,
                "promotion_status": "canary_rejected",
                "promotion_reason": reason,
                "canary_rejected_at": now.isoformat(),
                "canary_metrics": dict(metrics),
            }
        )
        if str(challenger_model_path):
            self.runtime.write_json_file_atomic(challenger_meta_path, metadata)
        self.runtime.storage.insert_execution_event(
            "model_promotion_candidate_rejected",
            symbol,
            {
                "rejected_at": now.isoformat(),
                "reason": reason,
                "status": candidate.get("status", "shadow"),
                "challenger_model_path": str(challenger_model_path),
                "metrics": dict(metrics),
            },
        )
        self.runtime.upsert_model_registry_entry(
            symbol=symbol,
            model_id=challenger_model_id,
            model_path=str(challenger_model_path),
            role="challenger",
            stage="canary_rejected",
            active=False,
            metadata={
                "rejected_at": now.isoformat(),
                "reason": reason,
                "metrics": dict(metrics),
            },
        )

    def promote_candidate_to_active(
        self,
        symbol: str,
        candidate: dict,
        now: datetime,
        metrics: dict[str, float | int],
    ) -> None:
        active_model_path = Path(str(candidate.get("active_model_path") or ""))
        challenger_model_path = Path(
            str(candidate.get("challenger_model_path") or "")
        )
        if not str(active_model_path) or not challenger_model_path.exists():
            return
        challenger_model_id = self.runtime.candidate_challenger_model_id(candidate)
        active_model_id = self.runtime.candidate_active_model_id(candidate)
        training_metadata = dict(candidate.get("training_metadata", {}) or {})
        dataset_end_timestamp_ms = int(
            training_metadata.get("dataset_end_timestamp_ms", 0) or 0
        )
        backup_paths = self.runtime.trainer._backup_active_model_artifacts(
            symbol=symbol,
            active_model_path=active_model_path,
            dataset_end_timestamp_ms=dataset_end_timestamp_ms,
        )
        active_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(challenger_model_path, active_model_path)
        promoted_metadata = dict(training_metadata)
        promoted_metadata.update(
            {
                "symbol": symbol,
                "model_id": challenger_model_id,
                "model_path": str(active_model_path),
                "active_model_path": str(active_model_path),
                "active_model_id": challenger_model_id,
                "challenger_model_path": "",
                "challenger_model_id": "",
                "incumbent_model_id": active_model_id,
                "promoted_to_active": True,
                "promotion_status": "promoted",
                "promotion_reason": str(
                    candidate.get("promotion_reason")
                    or promoted_metadata.get("promotion_reason")
                    or ""
                ),
                "previous_active_backup_path": str(
                    backup_paths.get("model_path") or ""
                ),
                "previous_active_backup_meta_path": str(
                    backup_paths.get("meta_path") or ""
                ),
                "canary_promoted_at": now.isoformat(),
                "canary_shadow_eval_count": int(
                    candidate.get("shadow_eval_count", 0) or 0
                ),
                "canary_shadow_accuracy": float(
                    candidate.get("shadow_accuracy", 0.0) or 0.0
                ),
                "canary_shadow_expectancy_pct": float(
                    candidate.get("shadow_expectancy_pct", 0.0) or 0.0
                ),
                "canary_shadow_profit_factor": float(
                    candidate.get("shadow_profit_factor", 0.0) or 0.0
                ),
                "canary_shadow_max_drawdown_pct": float(
                    candidate.get("shadow_max_drawdown_pct", 0.0) or 0.0
                ),
                "canary_live_eval_count": int(
                    metrics.get("eval_count", metrics.get("sample_count", 0)) or 0
                ),
                "canary_live_accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
                "canary_live_expectancy_pct": float(
                    metrics.get("expectancy_pct", 0.0) or 0.0
                ),
                "canary_live_profit_factor": float(
                    metrics.get("profit_factor", 0.0) or 0.0
                ),
                "canary_live_max_drawdown_pct": float(
                    metrics.get("max_drawdown_pct", 0.0) or 0.0
                ),
                "canary_live_trade_count": int(
                    metrics.get("live_trade_count", 0) or 0
                ),
                "canary_live_realized_trade_count": int(
                    metrics.get("live_pnl_closed_trade_count", 0) or 0
                ),
                "canary_live_realized_net_pnl": float(
                    metrics.get("live_pnl_realized_net_pnl", 0.0) or 0.0
                ),
                "canary_live_net_expectancy_pct": float(
                    metrics.get("live_pnl_expectancy_pct", 0.0) or 0.0
                ),
                "canary_live_net_profit_factor": float(
                    metrics.get("live_pnl_profit_factor", 0.0) or 0.0
                ),
                "canary_live_net_max_drawdown_pct": float(
                    metrics.get("live_pnl_max_drawdown_pct", 0.0) or 0.0
                ),
                "canary_live_net_avg_trade_return_pct": float(
                    metrics.get("live_pnl_avg_trade_return_pct", 0.0) or 0.0
                ),
                "canary_live_net_avg_holding_hours": float(
                    metrics.get("live_pnl_avg_holding_hours", 0.0) or 0.0
                ),
                "canary_live_net_objective_score": float(
                    metrics.get("live_pnl_objective_score", 0.0) or 0.0
                ),
                "canary_live_net_objective_quality": float(
                    metrics.get("live_pnl_objective_quality", 0.0) or 0.0
                ),
            }
        )
        self.runtime.write_json_file_atomic(
            active_model_path.with_suffix(".meta.json"),
            promoted_metadata,
        )
        self.runtime.remove_file_if_exists(str(challenger_model_path))
        self.runtime.remove_file_if_exists(
            str(challenger_model_path.with_suffix(".meta.json"))
        )
        self.runtime.clear_symbol_models(symbol)
        self.runtime.storage.insert_execution_event(
            "model_promotion_promoted",
            symbol,
            {
                "promoted_at": now.isoformat(),
                "active_model_path": str(active_model_path),
                "promotion_reason": promoted_metadata.get("promotion_reason", ""),
                "shadow_accuracy": promoted_metadata.get(
                    "canary_shadow_accuracy", 0.0
                ),
                "live_accuracy": promoted_metadata.get("canary_live_accuracy", 0.0),
                "shadow_expectancy_pct": promoted_metadata.get(
                    "canary_shadow_expectancy_pct",
                    0.0,
                ),
                "shadow_profit_factor": promoted_metadata.get(
                    "canary_shadow_profit_factor",
                    0.0,
                ),
                "shadow_max_drawdown_pct": promoted_metadata.get(
                    "canary_shadow_max_drawdown_pct",
                    0.0,
                ),
                "live_expectancy_pct": promoted_metadata.get(
                    "canary_live_expectancy_pct",
                    0.0,
                ),
                "live_profit_factor": promoted_metadata.get(
                    "canary_live_profit_factor",
                    0.0,
                ),
                "live_max_drawdown_pct": promoted_metadata.get(
                    "canary_live_max_drawdown_pct",
                    0.0,
                ),
                "live_trade_count": promoted_metadata.get(
                    "canary_live_trade_count", 0
                ),
                "live_realized_trade_count": promoted_metadata.get(
                    "canary_live_realized_trade_count",
                    0,
                ),
                "live_realized_net_pnl": promoted_metadata.get(
                    "canary_live_realized_net_pnl",
                    0.0,
                ),
                "live_net_expectancy_pct": promoted_metadata.get(
                    "canary_live_net_expectancy_pct",
                    0.0,
                ),
                "live_net_profit_factor": promoted_metadata.get(
                    "canary_live_net_profit_factor",
                    0.0,
                ),
                "live_net_max_drawdown_pct": promoted_metadata.get(
                    "canary_live_net_max_drawdown_pct",
                    0.0,
                ),
                "live_net_avg_trade_return_pct": promoted_metadata.get(
                    "canary_live_net_avg_trade_return_pct",
                    0.0,
                ),
                "live_net_avg_holding_hours": promoted_metadata.get(
                    "canary_live_net_avg_holding_hours",
                    0.0,
                ),
                "live_net_objective_score": promoted_metadata.get(
                    "canary_live_net_objective_score",
                    0.0,
                ),
                "live_net_objective_quality": promoted_metadata.get(
                    "canary_live_net_objective_quality",
                    0.0,
                ),
            },
        )
        self.runtime.upsert_model_registry_entry(
            symbol=symbol,
            model_id=challenger_model_id,
            model_path=str(active_model_path),
            role="active",
            stage="promoted",
            active=True,
            metadata={
                "promoted_at": now.isoformat(),
                "shadow_accuracy": promoted_metadata.get(
                    "canary_shadow_accuracy", 0.0
                ),
                "live_accuracy": promoted_metadata.get("canary_live_accuracy", 0.0),
                "shadow_expectancy_pct": promoted_metadata.get(
                    "canary_shadow_expectancy_pct",
                    0.0,
                ),
                "shadow_profit_factor": promoted_metadata.get(
                    "canary_shadow_profit_factor",
                    0.0,
                ),
                "shadow_max_drawdown_pct": promoted_metadata.get(
                    "canary_shadow_max_drawdown_pct",
                    0.0,
                ),
                "live_expectancy_pct": promoted_metadata.get(
                    "canary_live_expectancy_pct",
                    0.0,
                ),
                "live_profit_factor": promoted_metadata.get(
                    "canary_live_profit_factor",
                    0.0,
                ),
                "live_max_drawdown_pct": promoted_metadata.get(
                    "canary_live_max_drawdown_pct",
                    0.0,
                ),
                "live_trade_count": promoted_metadata.get(
                    "canary_live_trade_count", 0
                ),
                "live_realized_trade_count": promoted_metadata.get(
                    "canary_live_realized_trade_count",
                    0,
                ),
                "live_realized_net_pnl": promoted_metadata.get(
                    "canary_live_realized_net_pnl",
                    0.0,
                ),
                "live_net_expectancy_pct": promoted_metadata.get(
                    "canary_live_net_expectancy_pct",
                    0.0,
                ),
                "live_net_profit_factor": promoted_metadata.get(
                    "canary_live_net_profit_factor",
                    0.0,
                ),
                "live_net_max_drawdown_pct": promoted_metadata.get(
                    "canary_live_net_max_drawdown_pct",
                    0.0,
                ),
                "live_net_avg_trade_return_pct": promoted_metadata.get(
                    "canary_live_net_avg_trade_return_pct",
                    0.0,
                ),
                "live_net_avg_holding_hours": promoted_metadata.get(
                    "canary_live_net_avg_holding_hours",
                    0.0,
                ),
                "live_net_objective_score": promoted_metadata.get(
                    "canary_live_net_objective_score",
                    0.0,
                ),
                "live_net_objective_quality": promoted_metadata.get(
                    "canary_live_net_objective_quality",
                    0.0,
                ),
            },
        )
        self.runtime.upsert_model_registry_entry(
            symbol=symbol,
            model_id=active_model_id,
            model_path=str(candidate.get("active_model_path") or ""),
            role="backup",
            stage="superseded_backup",
            active=False,
            metadata={
                "superseded_at": now.isoformat(),
                "replaced_by": challenger_model_id,
            },
        )
        if promoted_metadata.get("candidate_walkforward_summary"):
            walkforward_result = {
                "symbol": symbol,
                "summary": dict(
                    promoted_metadata.get("candidate_walkforward_summary", {}) or {}
                ),
                "splits": list(
                    promoted_metadata.get("candidate_walkforward_splits", []) or []
                ),
            }
            self.runtime.storage.insert_walkforward_run(walkforward_result)
            self.runtime.report_runtime.record_rendered_report(
                report_type="walkforward",
                symbol=symbol,
                renderer=self.runtime.walkforward.render_report,
                payload=walkforward_result,
            )
        self.runtime.register_promoted_model_observation(
            symbol,
            SimpleNamespace(**promoted_metadata),
            now,
        )
