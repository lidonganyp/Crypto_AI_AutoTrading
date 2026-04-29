"""Runtime coordinator for model promotion, observation, and rollback lifecycle."""
from __future__ import annotations

from datetime import datetime

from core.model_evidence_policy import ModelEvidencePolicy
from core.promotion_candidate_runtime import PromotionCandidateRuntime
from core.promotion_observation_runtime import PromotionObservationRuntime


class ModelLifecycleRuntimeService:
    """Thin coordinator that delegates lifecycle work to focused runtime modules."""

    def __init__(
        self,
        storage,
        settings,
        *,
        trainer,
        walkforward,
        report_runtime,
        notifier,
        predictor_base_path,
        challenger_predictor_base_path,
        get_model_promotion_candidates,
        set_model_promotion_candidates,
        get_model_promotion_observations,
        set_model_promotion_observations,
        resolved_model_id,
        model_file_signature,
        training_objective_baseline,
        promotion_adaptive_requirements,
        promotion_live_allocation_pct,
        upsert_model_registry_entry,
        candidate_active_model_id,
        candidate_challenger_model_id,
        observation_active_model_id,
        observation_backup_model_id,
        clear_symbol_models,
        read_model_metadata,
        record_model_scorecard,
        build_model_scorecard,
        build_model_live_pnl_summary,
        objective_score_from_metrics,
        objective_score_quality,
        accuracy_safety_floor,
    ):
        self.storage = storage
        self.settings = settings
        self.trainer = trainer
        self.walkforward = walkforward
        self.report_runtime = report_runtime
        self.notifier = notifier
        self.predictor_base_path = predictor_base_path
        self.challenger_predictor_base_path = challenger_predictor_base_path

        self.get_model_promotion_candidates = get_model_promotion_candidates
        self.set_model_promotion_candidates = set_model_promotion_candidates
        self.get_model_promotion_observations = get_model_promotion_observations
        self.set_model_promotion_observations = set_model_promotion_observations

        self.resolved_model_id = resolved_model_id
        self.model_file_signature = model_file_signature
        self.training_objective_baseline = training_objective_baseline
        self.promotion_adaptive_requirements = promotion_adaptive_requirements
        self.promotion_live_allocation_pct = promotion_live_allocation_pct
        self.upsert_model_registry_entry = upsert_model_registry_entry
        self.candidate_active_model_id = candidate_active_model_id
        self.candidate_challenger_model_id = candidate_challenger_model_id
        self.observation_active_model_id = observation_active_model_id
        self.observation_backup_model_id = observation_backup_model_id
        self.clear_symbol_models = clear_symbol_models
        self.read_model_metadata = read_model_metadata
        self.record_model_scorecard = record_model_scorecard
        self.build_model_scorecard = build_model_scorecard
        self.build_model_live_pnl_summary = build_model_live_pnl_summary
        self.objective_score_from_metrics = objective_score_from_metrics
        self.objective_score_quality = objective_score_quality
        self.accuracy_safety_floor = accuracy_safety_floor

        self.evidence_policy = ModelEvidencePolicy(self)
        self.candidate_runtime = PromotionCandidateRuntime(self)
        self.observation_runtime = PromotionObservationRuntime(self)

    @staticmethod
    def _parse_iso_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _ab_variant_count(
        self,
        symbol: str,
        started_at: str,
        selected_variant: str,
    ) -> int:
        with self.storage._conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS c
                   FROM ab_test_runs
                  WHERE symbol = ?
                    AND timestamp >= ?
                    AND selected_variant = ?""",
                (symbol, started_at, selected_variant),
            ).fetchone()
        return int(row["c"] if row is not None else 0)

    def evidence_scale_from_metrics(
        self,
        payload: dict[str, float | int],
        *,
        source: str,
    ) -> dict[str, float | int | str]:
        return self.evidence_policy.evidence_scale_from_metrics(
            payload,
            source=source,
        )

    def active_model_evidence_scale(
        self,
        symbol: str,
    ) -> dict[str, float | int | str]:
        return self.evidence_policy.active_model_evidence_scale(symbol)

    def active_model_registry_entry(
        self,
        symbol: str,
    ) -> dict[str, str | int | bool] | None:
        with self.storage._conn() as conn:
            row = conn.execute(
                """SELECT symbol, model_id, model_version, model_path, role, stage,
                          active, metadata_json, created_at, updated_at
                   FROM model_registry
                   WHERE symbol = ?
                     AND active = 1
                     AND role = 'active'
                   ORDER BY updated_at DESC, created_at DESC
                   LIMIT 1""",
                (symbol,),
            ).fetchone()
        return dict(row) if row else None

    def active_model_exit_policy(
        self,
        symbol: str,
    ) -> dict[str, float | int | str | bool]:
        return self.evidence_policy.active_model_exit_policy(symbol)

    def candidate_live_evidence_scale(
        self,
        symbol: str,
        candidate: dict,
    ) -> dict[str, float | int | str]:
        return self.evidence_policy.candidate_live_evidence_scale(symbol, candidate)

    def register_promotion_candidate(
        self,
        symbol: str,
        summary,
        now: datetime,
    ) -> None:
        self.candidate_runtime.register_promotion_candidate(symbol, summary, now)

    def sync_model_registry_from_training_summary(
        self,
        symbol: str,
        summary,
        now: datetime,
    ) -> None:
        self.candidate_runtime.sync_model_registry_from_training_summary(
            symbol,
            summary,
            now,
        )

    def advance_promotion_candidate_to_live(
        self,
        symbol: str,
        candidate: dict,
        now: datetime,
        shadow_metrics: dict[str, float | int],
        champion_metrics: dict[str, float | int],
    ) -> dict:
        return self.candidate_runtime.advance_promotion_candidate_to_live(
            symbol,
            candidate,
            now,
            shadow_metrics,
            champion_metrics,
        )

    def reject_promotion_candidate(
        self,
        symbol: str,
        candidate: dict,
        now: datetime,
        reason: str,
        metrics: dict[str, float | int],
    ) -> None:
        self.candidate_runtime.reject_promotion_candidate(
            symbol,
            candidate,
            now,
            reason,
            metrics,
        )

    def promote_candidate_to_active(
        self,
        symbol: str,
        candidate: dict,
        now: datetime,
        metrics: dict[str, float | int],
    ) -> None:
        self.candidate_runtime.promote_candidate_to_active(
            symbol,
            candidate,
            now,
            metrics,
        )

    def register_promoted_model_observation(
        self,
        symbol: str,
        summary,
        now: datetime,
    ) -> None:
        self.observation_runtime.register_promoted_model_observation(
            symbol,
            summary,
            now,
        )

    def model_self_heal_interval_hours(self) -> int:
        return self.observation_runtime.model_self_heal_interval_hours()

    def runtime_model_path_for_symbol(self, symbol: str):
        return self.observation_runtime.runtime_model_path_for_symbol(symbol)

    def post_promotion_execution_metrics(
        self,
        symbol: str,
        promoted_at: str,
        limit: int = 200,
    ) -> dict[str, float | int]:
        return self.observation_runtime.post_promotion_execution_metrics(
            symbol,
            promoted_at,
            limit=limit,
        )

    @staticmethod
    def remove_file_if_exists(path_str: str) -> None:
        PromotionObservationRuntime.remove_file_if_exists(path_str)

    @staticmethod
    def write_json_file_atomic(path, payload: dict) -> None:
        PromotionObservationRuntime.write_json_file_atomic(path, payload)

    def accept_promoted_model(
        self,
        symbol: str,
        observation: dict,
        now: datetime,
        metrics: dict[str, float | int],
    ) -> None:
        self.observation_runtime.accept_promoted_model(
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
        self.observation_runtime.rollback_promoted_model(
            symbol,
            observation,
            now,
            reason,
            metrics,
        )

    def observe_promotion_candidates(self, now: datetime) -> dict[str, int]:
        return self.candidate_runtime.observe_promotion_candidates(now)

    def observe_promoted_models(self, now: datetime) -> dict[str, int]:
        return self.observation_runtime.observe_promoted_models(now)
