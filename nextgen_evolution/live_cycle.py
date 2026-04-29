"""Reusable runner for guarded nextgen autonomy-live cycles."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace

from config import Settings, get_settings

from .config import EvolutionConfig
from .director import AutonomousDirector
from .execution_bridge import AutonomyPaperBridge
from .experiment_lab import ExperimentLab
from .live_bridge import AutonomyLiveBridge
from .live_runtime import AutonomyLiveRuntime
from .lineage_rebuild import LineageRebuildPlanner
from .models import AutonomyDirective, RepairActionType, RepairPlan, RuntimeLifecycleState
from .portfolio_allocator import PortfolioAllocator
from .promotion_registry import PromotionRegistry
from .repair_cycle import RepairCycleRunner
from .repair_feedback import RepairFeedbackEngine
from .repair_reentry import RepairReentryPlanner
from .runtime_evidence import RuntimeEvidenceCollector
from .runtime_override_policy import latest_managed_close_time_index
from .scheduler import ExperimentScheduler
from .data_feed import SQLiteOHLCVFeed


class AutonomyLiveCycleRunner:
    """Run nextgen autonomy-live from the existing v3 runtime shell."""

    DEFAULT_TIMEFRAME = "5m"
    DEFAULT_MIN_ROWS = 200
    DEFAULT_MAX_SYMBOLS = 4
    DEFAULT_TOTAL_CAPITAL = 10000.0

    def __init__(
        self,
        storage,
        *,
        settings: Settings | None = None,
        feed_cls=SQLiteOHLCVFeed,
        registry_cls=PromotionRegistry,
        lab_cls=ExperimentLab,
        scheduler_cls=ExperimentScheduler,
        evidence_collector_cls=RuntimeEvidenceCollector,
        director_cls=AutonomousDirector,
        allocator_cls=PortfolioAllocator,
        live_runtime_cls=AutonomyLiveRuntime,
        live_bridge_cls=AutonomyLiveBridge,
        repair_cycle_cls=RepairCycleRunner,
        repair_reentry_cls=RepairReentryPlanner,
        lineage_rebuild_cls=LineageRebuildPlanner,
    ):
        self.storage = storage
        self.settings = settings or get_settings()
        self.feed_cls = feed_cls
        self.registry_cls = registry_cls
        self.lab_cls = lab_cls
        self.scheduler_cls = scheduler_cls
        self.evidence_collector_cls = evidence_collector_cls
        self.director_cls = director_cls
        self.allocator_cls = allocator_cls
        self.live_runtime_cls = live_runtime_cls
        self.live_bridge_cls = live_bridge_cls
        self.repair_cycle_cls = repair_cycle_cls
        self.repair_reentry_cls = repair_reentry_cls
        self.lineage_rebuild_cls = lineage_rebuild_cls

    def run(
        self,
        *,
        requested_live: bool | None = None,
        trigger: str = "manual",
        trigger_reason: str = "",
        trigger_details: str = "",
        whitelist: tuple[str, ...] | list[str] | None = None,
        max_active_runtimes: int | None = None,
    ) -> dict:
        feed = self.feed_cls(str(self.storage.db_path))
        registry = self.registry_cls(str(self.storage.db_path))
        live_runtime = self.live_runtime_cls(feed.storage, settings=self.settings)
        operator_request = live_runtime.resolve_operator_request(
            requested_live=requested_live,
            whitelist=whitelist,
            max_active_runtimes=max_active_runtimes,
        )
        requested_live = bool(operator_request["requested_live"])
        lab = self.lab_cls(feed) if requested_live else None
        base_config = lab.engine.config if lab is not None else EvolutionConfig()
        live_config = live_runtime.build_config(
            base_config=base_config,
            requested_live=requested_live,
            whitelist=operator_request["whitelist"],
            max_active_runtimes=operator_request["max_active_runtimes"],
        )
        live_status = live_runtime.evaluate(
            requested_live=requested_live,
            config=live_config,
        )

        if not requested_live and not live_status.force_flatten:
            return self._skipped_result(
                reason="live_not_requested",
                trigger=trigger,
                trigger_reason=trigger_reason,
                trigger_details=trigger_details,
                requested_live=requested_live,
                operator_request=operator_request,
                live_status=live_status,
            )

        if requested_live:
            cycle = self._run_requested_live_cycle(
                lab=lab,
                feed=feed,
                registry=registry,
                live_runtime=live_runtime,
                live_status=live_status,
                live_config=live_config,
                trigger=trigger,
                trigger_reason=trigger_reason,
                trigger_details=trigger_details,
            )
        else:
            cycle = self._run_emergency_flatten(
                feed=feed,
                registry=registry,
                live_runtime=live_runtime,
                live_status=live_status,
                live_config=live_config,
                trigger=trigger,
                trigger_reason=trigger_reason,
                trigger_details=trigger_details,
            )

        return {
            "status": "ok",
            "reason": "",
            "trigger": trigger,
            "trigger_reason": trigger_reason,
            "trigger_details": trigger_details,
            "requested_live": bool(requested_live),
            "effective_live": bool(live_status.effective_live),
            "force_flatten": bool(live_status.force_flatten),
            "operator_request": operator_request,
            "operator_reasons": list(live_status.reasons),
            "operator_status": asdict(live_status),
            **cycle,
        }

    def _run_requested_live_cycle(
        self,
        *,
        lab,
        feed,
        registry,
        live_runtime,
        live_status,
        live_config,
        trigger: str,
        trigger_reason: str,
        trigger_details: str,
    ) -> dict:
        jobs = [
            job
            for job in self.scheduler_cls(lab).default_jobs(
                timeframe=self.DEFAULT_TIMEFRAME,
                min_rows=self.DEFAULT_MIN_ROWS,
                max_symbols=self.DEFAULT_MAX_SYMBOLS,
            )
            if job.symbol.endswith("USDT:USDT")
        ]
        if not jobs and not live_status.force_flatten:
            return {
                **self._count_summary([], []),
                **self._repair_queue_summary(),
                "status": "skipped",
                "reason": "no_nextgen_jobs",
                "job_count": 0,
                "experiment_count": 0,
                "runtime_state_count": 0,
                "autonomy_cycle_id": None,
                "evidence_count": 0,
                "repair_execution_count": 0,
                "repair_run_ids": [],
                "repair_reentry_count": 0,
                "repair_reentry_runtime_ids": [],
            }

        previous_states = registry.load_runtime_states(hydrate_legacy=False)
        carryforward_index = self._active_candidate_index(
            registry=registry,
            previous_states=previous_states,
        )
        persisted_results = []
        for index, job in enumerate(jobs, start=1):
            result = lab.run_for_symbol(
                symbol=job.symbol,
                timeframe=job.timeframe,
                total_capital=job.total_capital,
                candle_limit=job.candle_limit,
                extra_genomes=carryforward_index.get((job.symbol, job.timeframe), []),
                notes={
                    "carryforward_candidates": len(
                        carryforward_index.get((job.symbol, job.timeframe), [])
                    ),
                },
            )
            persisted_results.append(
                registry.persist_experiment(
                    result,
                    notes={
                        "source": "v3_engine",
                        "mode": "autonomy-live",
                        "trigger": trigger,
                        "trigger_reason": trigger_reason,
                        "trigger_details": trigger_details,
                        "requested_live": True,
                        "batch_index": index,
                        "batch_size": len(jobs),
                    },
                )
            )

        evidence_collector = self.evidence_collector_cls(feed, lab.engine.config)
        evidence_snapshots = list(
            evidence_collector.collect(
                [
                    AutonomyPaperBridge.runtime_id(
                        result.symbol,
                        result.timeframe,
                        card.genome.strategy_id,
                    )
                    for result in persisted_results
                    for card in result.scorecards
                ],
                previous_states=previous_states,
            ).values()
        )
        runtime_evidence_index = {
            item.runtime_id: item
            for item in evidence_snapshots
        }
        latest_close_time_by_runtime = latest_managed_close_time_index(
            feed.storage,
            managed_source=AutonomyPaperBridge.MANAGED_SOURCE,
        )
        if evidence_snapshots:
            registry.persist_runtime_evidence(evidence_snapshots)
        directive = self.director_cls(
            lab.engine.config,
            evidence_collector=evidence_collector,
        ).plan_from_experiments(
            persisted_results,
            repair_feedback=self._repair_feedback_index(
                registry=registry,
                results=persisted_results,
                config=lab.engine.config,
            ),
            previous_states=previous_states,
        )
        directive = self.lineage_rebuild_cls(lab.engine.config).expand(
            directive,
            results=persisted_results,
        )
        directive = self._prioritize_repairs(directive)
        portfolio_allocations = self.allocator_cls(lab.engine.config).allocate(
            persisted_results,
            total_capital=self.DEFAULT_TOTAL_CAPITAL,
            previous_states=previous_states,
            runtime_evidence=runtime_evidence_index,
            latest_close_time_by_runtime=latest_close_time_by_runtime,
            directive=directive,
        )
        return self._apply_cycle(
            lab=lab,
            feed=feed,
            registry=registry,
            live_runtime=live_runtime,
            live_status=live_status,
            live_config=live_config,
            results=persisted_results,
            directive=directive,
            portfolio_allocations=portfolio_allocations,
            trigger=trigger,
            trigger_reason=trigger_reason,
            trigger_details=trigger_details,
            evidence_count=len(evidence_snapshots),
            job_count=len(jobs),
            previous_states=previous_states,
            runtime_evidence_index=runtime_evidence_index,
            latest_close_time_by_runtime=latest_close_time_by_runtime,
        )

    def _run_emergency_flatten(
        self,
        *,
        feed,
        registry,
        live_runtime,
        live_status,
        live_config,
        trigger: str,
        trigger_reason: str,
        trigger_details: str,
    ) -> dict:
        return self._apply_cycle(
            lab=None,
            feed=feed,
            registry=registry,
            live_runtime=live_runtime,
            live_status=live_status,
            live_config=live_config,
            results=[],
            directive=AutonomyDirective(
                notes={
                    "emergency_flatten": True,
                }
            ),
            portfolio_allocations=[],
            trigger=trigger,
            trigger_reason=trigger_reason,
            trigger_details=trigger_details,
            evidence_count=0,
            job_count=0,
            previous_states=registry.load_runtime_states(hydrate_legacy=False),
            runtime_evidence_index={},
            latest_close_time_by_runtime={},
        )

    def _apply_cycle(
        self,
        *,
        lab,
        feed,
        registry,
        live_runtime,
        live_status,
        live_config,
        results,
        directive,
        portfolio_allocations,
        trigger: str,
        trigger_reason: str,
        trigger_details: str,
        evidence_count: int,
        job_count: int,
        previous_states,
        runtime_evidence_index,
        latest_close_time_by_runtime,
    ) -> dict:
        notes = {
            "source": "v3_engine",
            "mode": "autonomy-live",
            "trigger": trigger,
            "trigger_reason": trigger_reason,
            "trigger_details": trigger_details,
            "requested_live": bool(live_status.requested_live),
            "effective_live": bool(live_status.effective_live),
            "force_flatten": bool(live_status.force_flatten),
            "operator_reasons": list(live_status.reasons),
            "job_count": int(job_count),
            "experiment_count": len(results),
        }
        autonomy_cycle_id = registry.persist_autonomy_cycle(
            directive,
            notes=notes,
        )
        repair_results = []
        if lab is not None and directive.repairs:
            repair_results = self.repair_cycle_cls(
                lab,
                registry=registry,
            ).run(
                list(directive.repairs),
                autonomy_cycle_id=autonomy_cycle_id,
                total_capital=self.DEFAULT_TOTAL_CAPITAL,
            )
        reentry_results: list = []
        reentry_directives: list = []
        effective_results = list(results)
        effective_directive = directive
        effective_allocations = list(portfolio_allocations)
        if repair_results:
            reentry_results, reentry_directives = self.repair_reentry_cls().plan(
                repair_results
            )
            if reentry_directives:
                registry.append_execution_directives(
                    autonomy_cycle_id,
                    reentry_directives,
                    strategy_count_delta=len(reentry_results),
                    notes={
                        "repair_reentry_count": len(reentry_directives),
                    },
                )
                effective_results = [*effective_results, *reentry_results]
                effective_directive = AutonomyDirective(
                    execution=[*directive.execution, *reentry_directives],
                    repairs=list(directive.repairs),
                    quarantined=list(directive.quarantined),
                    retired=list(directive.retired),
                    notes={
                        **dict(directive.notes or {}),
                        "repair_reentry_count": len(reentry_directives),
                    },
                )
                effective_allocations = self.allocator_cls(live_config).allocate(
                    effective_results,
                    total_capital=self.DEFAULT_TOTAL_CAPITAL,
                    previous_states=previous_states,
                    runtime_evidence=runtime_evidence_index,
                    latest_close_time_by_runtime=latest_close_time_by_runtime,
                    directive=effective_directive,
                )
        live_config.autonomy_live_enabled = live_status.effective_live
        bridge = self.live_bridge_cls(
            feed,
            registry=registry,
            live_trader=live_runtime.build_live_trader(status=live_status),
            config=live_config,
            operator_status=live_status,
        )
        runtime_states, intents = bridge.apply(
            results=list(effective_results),
            directive=effective_directive,
            portfolio_allocations=list(effective_allocations),
            total_capital=self.DEFAULT_TOTAL_CAPITAL,
            autonomy_cycle_id=autonomy_cycle_id,
            previous_states=previous_states,
        )
        return {
            **self._count_summary(runtime_states, intents),
            **self._repair_queue_summary(effective_directive),
            "job_count": int(job_count),
            "experiment_count": len(effective_results),
            "runtime_state_count": len(runtime_states),
            "autonomy_cycle_id": autonomy_cycle_id,
            "evidence_count": int(evidence_count),
            "repair_execution_count": len(repair_results),
            "repair_run_ids": [
                item.experiment.registry_run_id
                for item in repair_results
                if item.experiment.registry_run_id is not None
            ],
            "repair_reentry_count": len(reentry_directives),
            "repair_reentry_runtime_ids": [
                item.strategy_id
                for item in reentry_directives
            ],
        }

    def _skipped_result(
        self,
        *,
        reason: str,
        trigger: str,
        trigger_reason: str,
        trigger_details: str,
        requested_live: bool,
        operator_request: dict,
        live_status,
    ) -> dict:
        return {
            **self._repair_queue_summary(),
            "status": "skipped",
            "reason": reason,
            "trigger": trigger,
            "trigger_reason": trigger_reason,
            "trigger_details": trigger_details,
            "requested_live": bool(requested_live),
            "effective_live": bool(live_status.effective_live),
            "force_flatten": bool(live_status.force_flatten),
            "operator_request": dict(operator_request),
            "operator_reasons": list(live_status.reasons),
            "operator_status": asdict(live_status),
            "job_count": 0,
            "experiment_count": 0,
            "runtime_state_count": 0,
            "autonomy_cycle_id": None,
            "evidence_count": 0,
            "repair_execution_count": 0,
            "repair_run_ids": [],
            "repair_reentry_count": 0,
            "repair_reentry_runtime_ids": [],
            "intent_count": 0,
            "action_counts": {},
            "intent_status_counts": {},
        }

    @classmethod
    def _prioritize_repairs(
        cls,
        directive: AutonomyDirective,
    ) -> AutonomyDirective:
        repairs = list(directive.repairs or [])
        if not repairs:
            return directive
        adjusted: list[RepairPlan] = []
        hold_priority_count = 0
        postponed_rebuild_count = 0
        reprioritized_count = 0
        for item in repairs:
            recovery_mode = cls._repair_recovery_mode(item)
            if recovery_mode == "hold":
                hold_priority_count += 1
            adjusted_item = cls._reprioritized_repair(item)
            if int(adjusted_item.priority) != int(item.priority):
                reprioritized_count += 1
            if (
                item.action == RepairActionType.REBUILD_LINEAGE
                and int(adjusted_item.priority) < int(item.priority)
            ):
                postponed_rebuild_count += 1
            adjusted.append(adjusted_item)
        adjusted.sort(
            key=lambda item: (
                int(item.priority),
                item.action != RepairActionType.REBUILD_LINEAGE,
            ),
            reverse=True,
        )
        return replace(
            directive,
            repairs=adjusted,
            notes={
                **dict(directive.notes or {}),
                "repair_queue_requested_size": int(
                    dict(directive.notes or {}).get("repair_queue_requested_size") or 0
                ),
                "repair_queue_dropped_count": int(
                    dict(directive.notes or {}).get("repair_queue_dropped_count") or 0
                ),
                "repair_queue_dropped_runtime_ids": [
                    str(item).strip()
                    for item in list(
                        dict(directive.notes or {}).get("repair_queue_dropped_runtime_ids") or []
                    )
                    if str(item).strip()
                ],
                "repair_queue_runtime_ids": [item.strategy_id for item in adjusted],
                "repair_queue_actions": [item.action.value for item in adjusted],
                "repair_queue_priorities": [int(item.priority) for item in adjusted],
                "repair_queue_hold_priority_count": int(hold_priority_count),
                "repair_queue_postponed_rebuild_count": int(postponed_rebuild_count),
                "repair_queue_reprioritized_count": int(reprioritized_count),
            },
        )

    @classmethod
    def _reprioritized_repair(cls, plan: RepairPlan) -> RepairPlan:
        adjusted_priority = max(1, int(plan.priority))
        recovery_mode = cls._repair_recovery_mode(plan)
        if recovery_mode == "hold":
            adjusted_priority += 2
        elif recovery_mode == "accelerate":
            adjusted_priority = max(1, adjusted_priority - 1)
        elif recovery_mode == "release":
            adjusted_priority = max(1, adjusted_priority - 2)
        if plan.action == RepairActionType.REBUILD_LINEAGE:
            adjusted_priority = max(1, adjusted_priority - 4)
            if recovery_mode == "accelerate":
                adjusted_priority = max(1, adjusted_priority - 1)
            elif recovery_mode == "release":
                adjusted_priority = max(1, adjusted_priority - 2)
        if adjusted_priority == int(plan.priority):
            return plan
        return replace(plan, priority=adjusted_priority)

    @staticmethod
    def _repair_recovery_mode(plan: RepairPlan) -> str:
        prefix = "repair_runtime_override_recovery_mode:"
        for reason in list(plan.reasons or []):
            text = str(reason).strip().lower()
            if text.startswith(prefix):
                return text[len(prefix):].strip()
        return ""

    @staticmethod
    def _repair_queue_summary(
        directive: AutonomyDirective | None = None,
    ) -> dict:
        notes = dict((directive.notes if directive is not None else {}) or {})
        requested_size = int(notes.get("repair_queue_requested_size") or 0)
        dropped_count = int(notes.get("repair_queue_dropped_count") or 0)
        dropped_runtime_ids = [
            str(item).strip()
            for item in list(notes.get("repair_queue_dropped_runtime_ids") or [])
            if str(item).strip()
        ]
        hold_count = int(notes.get("repair_queue_hold_priority_count") or 0)
        postponed_rebuild_count = int(notes.get("repair_queue_postponed_rebuild_count") or 0)
        reprioritized_count = int(notes.get("repair_queue_reprioritized_count") or 0)
        return {
            "repair_queue_requested_size": requested_size,
            "repair_queue_dropped_count": dropped_count,
            "repair_queue_dropped_runtime_ids": dropped_runtime_ids,
            "repair_queue_hold_priority_count": hold_count,
            "repair_queue_postponed_rebuild_count": postponed_rebuild_count,
            "repair_queue_reprioritized_count": reprioritized_count,
            "repair_queue_dropped_active": dropped_count > 0,
            "repair_queue_hold_priority_active": hold_count > 0,
            "repair_queue_postponed_rebuild_active": postponed_rebuild_count > 0,
            "repair_queue_reprioritized_active": reprioritized_count > 0,
        }

    @staticmethod
    def _count_summary(runtime_states, intents) -> dict:
        action_counts = Counter(item.action.value for item in intents)
        intent_status_counts = Counter(str(item.status or "") for item in intents)
        return {
            "runtime_state_count": len(runtime_states),
            "intent_count": len(intents),
            "action_counts": dict(action_counts),
            "intent_status_counts": dict(intent_status_counts),
        }

    @staticmethod
    def _active_candidate_index(
        *,
        registry,
        previous_states,
    ) -> dict[tuple[str, str], list]:
        active_states = [
            item
            for item in (previous_states or [])
            if item.lifecycle_state
            in {
                RuntimeLifecycleState.SHADOW,
                RuntimeLifecycleState.PAPER,
                RuntimeLifecycleState.LIMITED_LIVE,
                RuntimeLifecycleState.LIVE,
            }
        ]
        genomes = registry.load_candidate_genomes(
            [item.strategy_id for item in active_states]
        )
        indexed: dict[tuple[str, str], list] = {}
        for item in active_states:
            genome = genomes.get(item.strategy_id)
            if genome is None:
                continue
            indexed.setdefault((item.symbol, item.timeframe), []).append(genome)
        return indexed

    @staticmethod
    def _repair_feedback_index(
        *,
        registry,
        results,
        config,
    ) -> dict:
        rows = registry.latest_repair_executions(
            limit=int(config.autonomy_repair_history_limit),
        )
        autonomy_outcome_rows = registry.latest_pnl_ledger(
            limit=max(200, int(config.autonomy_repair_history_limit) * 10),
        )
        return RepairFeedbackEngine(config).build(
            rows,
            runtime_ids=[
                AutonomyPaperBridge.runtime_id(
                    result.symbol,
                    result.timeframe,
                    card.genome.strategy_id,
                )
                for result in results
                for card in result.scorecards
            ],
            strategy_ids=[
                value
                for result in results
                for card in result.scorecards
                for value in (
                    card.genome.strategy_id,
                    card.genome.mutation_of or "",
                )
                if str(value).strip()
            ],
            autonomy_outcome_rows=autonomy_outcome_rows,
        )
