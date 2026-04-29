"""Summarize historical repair outcomes for adaptive autonomy planning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json

from .config import EvolutionConfig
from .models import PromotionStage, RepairActionType


@dataclass(slots=True)
class RepairFeedbackSummary:
    """Aggregated repair history for one runtime lineage."""

    source_strategy_id: str
    source_runtime_id: str = ""
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    latest_status: str = ""
    latest_action: RepairActionType | None = None
    preferred_action: RepairActionType | None = None
    suggested_validation_stage: PromotionStage = PromotionStage.SHADOW
    probation_required: bool = False
    retire_recommended: bool = False
    notes: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.attempts <= 0:
            return 0.0
        return round(self.successes / float(self.attempts), 4)


class RepairFeedbackEngine:
    """Turn persisted repair execution history into planner-friendly feedback."""

    SUCCESS_STAGES = {
        PromotionStage.SHADOW.value,
        PromotionStage.PAPER.value,
        PromotionStage.LIVE.value,
    }

    def __init__(self, config: EvolutionConfig | None = None):
        self.config = config or EvolutionConfig()

    def build(
        self,
        rows: list[dict],
        *,
        runtime_ids: list[str] | None = None,
        strategy_ids: list[str] | None = None,
        autonomy_outcome_rows: list[dict] | None = None,
    ) -> dict[str, RepairFeedbackSummary]:
        runtime_filter = {
            str(item).strip()
            for item in (runtime_ids or [])
            if str(item).strip()
        }
        strategy_filter = {
            str(item).strip()
            for item in (strategy_ids or [])
            if str(item).strip()
        }
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            runtime_id = str(row.get("source_runtime_id") or "").strip()
            strategy_id = str(row.get("source_strategy_id") or "").strip()
            if runtime_filter and runtime_id not in runtime_filter:
                if not strategy_filter or strategy_id not in strategy_filter:
                    continue
            elif strategy_filter and strategy_id not in strategy_filter and runtime_id not in runtime_filter:
                    continue
            grouped[(runtime_id, strategy_id)].append(row)
        for row in (autonomy_outcome_rows or []):
            metadata = self._row_metadata(row)
            runtime_id = str(metadata.get("runtime_id") or "").strip()
            strategy_id = str(metadata.get("strategy_id") or "").strip()
            if runtime_filter and runtime_id not in runtime_filter:
                if not strategy_filter or strategy_id not in strategy_filter:
                    continue
            elif strategy_filter and strategy_id not in strategy_filter and runtime_id not in runtime_filter:
                continue
            grouped.setdefault((runtime_id, strategy_id), [])

        summaries: dict[str, RepairFeedbackSummary] = {}
        for (runtime_id, strategy_id), grouped_rows in grouped.items():
            summary = self._summarize_group(
                runtime_id=runtime_id,
                strategy_id=strategy_id,
                rows=grouped_rows,
                autonomy_outcome_rows=autonomy_outcome_rows or [],
            )
            if runtime_id:
                summaries[runtime_id] = summary
            if strategy_id and strategy_id not in summaries:
                summaries[strategy_id] = summary
        return summaries

    def _summarize_group(
        self,
        *,
        runtime_id: str,
        strategy_id: str,
        rows: list[dict],
        autonomy_outcome_rows: list[dict],
    ) -> RepairFeedbackSummary:
        ordered = sorted(
            rows,
            key=lambda item: (
                str(item.get("created_at") or ""),
                int(item.get("id") or 0),
            ),
        )
        attempts = len(ordered)
        successes = 0
        failures = 0
        consecutive_failures = 0
        latest_status = ""
        latest_action: RepairActionType | None = None
        preferred_action: RepairActionType | None = None
        highest_success_stage = PromotionStage.SHADOW
        latest_candidate_strategy_id = ""
        successful_actions: list[RepairActionType] = []
        outcome_summary = self._summarize_autonomy_outcomes(
            runtime_id=runtime_id,
            strategy_id=strategy_id,
            rows=autonomy_outcome_rows,
        )

        for row in ordered:
            latest_status = str(row.get("status") or "")
            latest_action = self._parse_action(row.get("action"))
            latest_candidate_strategy_id = str(row.get("candidate_strategy_id") or "")
            if self._is_success(latest_status):
                successes += 1
                if latest_action is not None:
                    successful_actions.append(latest_action)
                    preferred_action = latest_action
                stage = self._parse_stage(latest_status)
                if stage is not None and self._stage_rank(stage) > self._stage_rank(highest_success_stage):
                    highest_success_stage = stage
            else:
                failures += 1

        for row in reversed(ordered):
            status = str(row.get("status") or "")
            if self._is_success(status):
                break
            consecutive_failures += 1

        suggested_validation_stage = PromotionStage.SHADOW
        if (
            successes >= self.config.autonomy_repair_promote_after_successes
            and consecutive_failures == 0
            and highest_success_stage in {PromotionStage.PAPER, PromotionStage.LIVE}
        ):
            suggested_validation_stage = PromotionStage.PAPER

        probation_required = failures > 0 or successes == 0
        retire_recommended = (
            consecutive_failures >= self.config.autonomy_repair_retire_after_failures
        )
        if preferred_action is None:
            preferred_action = self._preferred_action_from_outcomes(outcome_summary)

        return RepairFeedbackSummary(
            source_strategy_id=strategy_id,
            source_runtime_id=runtime_id,
            attempts=attempts,
            successes=successes,
            failures=failures,
            consecutive_failures=consecutive_failures,
            latest_status=latest_status,
            latest_action=latest_action,
            preferred_action=preferred_action,
            suggested_validation_stage=suggested_validation_stage,
            probation_required=probation_required,
            retire_recommended=retire_recommended,
            notes={
                "highest_success_stage": highest_success_stage.value,
                "latest_candidate_strategy_id": latest_candidate_strategy_id,
                "successful_actions": [item.value for item in successful_actions],
                "success_rate": (
                    round(successes / float(attempts), 4)
                    if attempts > 0
                    else 0.0
                ),
                "autonomy_outcomes": outcome_summary,
            },
        )

    @staticmethod
    def _row_metadata(row: dict) -> dict:
        raw = row.get("metadata_json")
        if isinstance(raw, dict):
            return dict(raw)
        try:
            payload = json.loads(raw or "{}")
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def _summarize_autonomy_outcomes(
        self,
        *,
        runtime_id: str,
        strategy_id: str,
        rows: list[dict],
    ) -> dict[str, float | int]:
        summary = {
            "profit_lock_harvest_count": 0,
            "profit_lock_exit_count": 0,
            "profit_lock_net_pnl": 0.0,
            "forced_exit_count": 0,
            "forced_exit_net_pnl": 0.0,
        }
        for row in rows:
            metadata = self._row_metadata(row)
            row_runtime_id = str(metadata.get("runtime_id") or "").strip()
            row_strategy_id = str(metadata.get("strategy_id") or "").strip()
            if runtime_id and row_runtime_id != runtime_id:
                if not strategy_id or row_strategy_id != strategy_id:
                    continue
            elif strategy_id and row_strategy_id != strategy_id:
                continue
            reason = str(metadata.get("reason") or "").strip()
            net_pnl = float(row.get("net_pnl") or 0.0)
            if reason in {
                "nextgen_autonomy_profit_lock_reduce",
                "nextgen_autonomy_live_profit_lock_reduce",
            }:
                summary["profit_lock_harvest_count"] += 1
                summary["profit_lock_net_pnl"] = round(
                    float(summary["profit_lock_net_pnl"]) + net_pnl,
                    4,
                )
            elif reason in {
                "nextgen_autonomy_close",
                "nextgen_autonomy_live_close",
            } and "profit_lock_exit" in {
                str(item).strip() for item in (metadata.get("reasons") or [])
            }:
                summary["profit_lock_exit_count"] += 1
                summary["profit_lock_net_pnl"] = round(
                    float(summary["profit_lock_net_pnl"]) + net_pnl,
                    4,
                )
            elif reason in {
                "nextgen_autonomy_close",
                "nextgen_autonomy_live_close",
            }:
                summary["forced_exit_count"] += 1
                summary["forced_exit_net_pnl"] = round(
                    float(summary["forced_exit_net_pnl"]) + net_pnl,
                    4,
                )
        return summary

    @staticmethod
    def _preferred_action_from_outcomes(outcomes: dict[str, float | int]) -> RepairActionType | None:
        if int(outcomes.get("profit_lock_exit_count") or 0) > 0:
            return RepairActionType.TIGHTEN_RISK
        if int(outcomes.get("forced_exit_count") or 0) >= 2:
            return RepairActionType.TIGHTEN_RISK
        if (
            int(outcomes.get("profit_lock_harvest_count") or 0) >= 2
            and float(outcomes.get("profit_lock_net_pnl") or 0.0) > 0.0
        ):
            return RepairActionType.RAISE_SELECTIVITY
        return None

    @classmethod
    def _is_success(cls, status: str) -> bool:
        return str(status or "") in cls.SUCCESS_STAGES

    @staticmethod
    def _parse_action(value) -> RepairActionType | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return RepairActionType(text)
        except ValueError:
            return None

    @staticmethod
    def _parse_stage(value) -> PromotionStage | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return PromotionStage(text)
        except ValueError:
            return None

    @staticmethod
    def _stage_rank(stage: PromotionStage) -> int:
        return {
            PromotionStage.REJECT: 0,
            PromotionStage.SHADOW: 1,
            PromotionStage.PAPER: 2,
            PromotionStage.LIVE: 3,
        }[stage]
