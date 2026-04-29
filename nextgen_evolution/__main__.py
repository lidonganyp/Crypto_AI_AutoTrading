"""Tiny demo entrypoint for the next-generation scaffold."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

from config import get_settings

from .director import AutonomousDirector
from .alpha_factory import StrategyPrimitive
from .data_feed import SQLiteOHLCVFeed
from .engine import NextGenEvolutionEngine
from .execution_bridge import AutonomyPaperBridge
from .live_bridge import AutonomyLiveBridge
from .live_runtime import AutonomyLiveRuntime
from .experiment_lab import ExperimentLab
from .lineage_rebuild import LineageRebuildPlanner
from .models import AutonomyDirective, RuntimeLifecycleState, ValidationMetrics
from .portfolio_allocator import PortfolioAllocator
from .portfolio_monitor import PortfolioMonitor
from .portfolio_tracker import PortfolioTracker
from .promotion_registry import PromotionRegistry
from .repair_cycle import RepairCycleRunner
from .repair_feedback import RepairFeedbackEngine
from .repair_reentry import RepairReentryPlanner
from .runtime_evidence import RuntimeEvidenceCollector
from .runtime_override_policy import latest_managed_close_time_index
from .scheduler import ExperimentScheduler


def _family_by_strategy(result) -> dict[str, str]:
    return {
        card.genome.strategy_id: card.genome.family
        for card in result.scorecards
    }


def _active_runtime_candidate_index(registry, previous_states) -> dict[tuple[str, str], list]:
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


def _repair_feedback_index(registry, results, config) -> dict:
    return RepairFeedbackEngine(config).build(
        registry.latest_repair_executions(
            limit=int(config.autonomy_repair_history_limit),
        ),
        runtime_ids=[
            f"{result.symbol}|{result.timeframe}|{card.genome.strategy_id}"
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
    )


def _print_batch_summary(results) -> None:
    total_promoted = sum(len(result.promoted) for result in results)
    total_allocations = sum(len(result.allocations) for result in results)
    family_capital: dict[str, float] = defaultdict(float)
    for result in results:
        families = _family_by_strategy(result)
        for item in result.allocations:
            family = families.get(item.strategy_id, "unknown")
            family_capital[family] += item.allocated_capital

    print(f"Batch experiments: {len(results)} symbols")
    print(f"Total promoted candidates: {total_promoted}")
    print(f"Total capital allocations: {total_allocations}")
    if family_capital:
        print("Capital by family:")
        for family, capital in sorted(
            family_capital.items(),
            key=lambda entry: entry[1],
            reverse=True,
        ):
            print(f"- {family}: capital={round(capital, 2)}")

    for result in results:
        print(f"{result.symbol} {result.timeframe} run_id={result.registry_run_id}")
        print(
            f"  promoted={len(result.promoted)} allocations={len(result.allocations)} feature_points={result.notes.get('feature_summary', {}).get('feature_points', 0)}"
        )
        for card in result.promoted[:3]:
            print(
                f"  candidate {card.genome.strategy_id}: stage={card.stage.value} total={card.total_score}"
            )
        for item in result.allocations:
            print(
                f"  allocation {item.strategy_id}: capital={item.allocated_capital} weight={item.weight}"
            )


def _print_portfolio_plan(allocations, total_capital: float) -> None:
    allocated = round(sum(item.allocated_capital for item in allocations), 2)
    reserve = round(max(0.0, total_capital - allocated), 2)
    print(f"Unified portfolio capital: {round(total_capital, 2)}")
    print(f"Allocated capital: {allocated}")
    print(f"Reserve capital: {reserve}")
    print("Portfolio allocations:")
    for item in allocations:
        print(
            f"- {item.symbol} {item.strategy_id}: family={item.family} stage={item.stage.value} capital={item.allocated_capital} weight={item.weight}"
        )


def _print_refresh_summary(portfolio_run_id: int, snapshots) -> None:
    print(f"Portfolio refresh run id: {portfolio_run_id}")
    print(f"Snapshots written: {len(snapshots)}")
    for index, snapshot in enumerate(snapshots, start=1):
        print(
            f"- cycle={index} equity={snapshot.equity} unrealized_pnl={snapshot.unrealized_pnl} gross_exposure={snapshot.gross_exposure} open_positions={snapshot.open_positions} status={snapshot.status}"
        )


def _print_health_summary(status: dict) -> None:
    print(f"Health: {status['health']}")
    print(f"Freshness: {status['freshness']}")
    print(f"Max drawdown pct: {status['max_drawdown_pct']}")
    if status["alerts"]:
        print("Alerts:")
        for alert in status["alerts"]:
            print(f"- [{alert['severity']}] {alert['code']}: {alert['message']}")


def _print_autonomy_summary(autonomy_cycle_id: int, directive) -> None:
    print(f"Autonomy cycle id: {autonomy_cycle_id}")
    print(f"Execution directives: {len(directive.execution)}")
    print(f"Repair plans: {len(directive.repairs)}")
    _print_repair_queue_summary(dict(directive.notes or {}))
    if directive.quarantined:
        print(f"Quarantined: {','.join(directive.quarantined)}")
    if directive.retired:
        print(f"Retired: {','.join(directive.retired)}")
    for item in directive.execution:
        print(
            f"- execution {item.strategy_id}: action={item.action.value} from={item.from_stage.value} to={item.target_stage.value} capital_multiplier={item.capital_multiplier}"
        )
    for item in directive.repairs:
        candidate = item.candidate_genome.strategy_id if item.candidate_genome else ""
        print(
            f"- repair {item.strategy_id}: action={item.action.value} priority={item.priority} validation_stage={item.validation_stage.value} candidate={candidate}"
        )


def _print_repair_queue_summary(notes: dict) -> None:
    payload = dict(notes or {})
    print(
        "Repair queue summary:"
        f" hold_priority_count={int(payload.get('repair_queue_hold_priority_count') or 0)}"
        f" postponed_rebuild_count={int(payload.get('repair_queue_postponed_rebuild_count') or 0)}"
        f" reprioritized_count={int(payload.get('repair_queue_reprioritized_count') or 0)}"
    )


def _print_execution_summary(label: str, runtime_states, intents) -> None:
    print(f"{label} runtime states synced: {len(runtime_states)}")
    print(f"{label} execution intents: {len(intents)}")
    for item in intents:
        print(
            f"- intent {item.symbol} {item.strategy_id}: action={item.action.value} lifecycle={item.lifecycle_state.value} status={item.status} desired_capital={item.desired_capital}"
        )


def _print_paper_execution_summary(runtime_states, intents) -> None:
    _print_execution_summary("Paper", runtime_states, intents)


def _print_live_execution_summary(runtime_states, intents) -> None:
    _print_execution_summary("Live", runtime_states, intents)


def _print_repair_execution_summary(repair_results) -> None:
    print(f"Repair executions: {len(repair_results)}")
    for item in repair_results:
        run_id = item.experiment.registry_run_id
        outcome = (
            item.experiment.scorecards[0].stage.value
            if item.experiment.scorecards
            else "no_score"
        )
        print(
            f"- repair validation {item.source_runtime_id}: candidate={item.plan.candidate_genome.strategy_id if item.plan.candidate_genome else ''} outcome={outcome} run_id={run_id}"
        )


def _print_repair_reentry_summary(reentry_directives) -> None:
    print(f"Repair reentries: {len(reentry_directives)}")
    for item in reentry_directives:
        print(
            f"- repair reentry {item.strategy_id}: action={item.action.value} target_stage={item.target_stage.value} capital_multiplier={item.capital_multiplier}"
        )


def _print_live_operator_summary(status) -> None:
    print(f"Live requested: {status.requested_live}")
    print(f"Live effective: {status.effective_live}")
    print(f"Live mode: {'live' if status.effective_live else 'dry_run'}")
    print(f"Allow entries: {status.allow_entries}")
    print(f"Allow managed closes: {status.allow_managed_closes}")
    print(f"Force flatten: {status.force_flatten}")
    print(f"Exchange provider: {status.provider}")
    print(f"Runtime mode: {status.runtime_mode}")
    print(f"Allow live orders: {status.allow_live_orders}")
    print(f"Live whitelist: {','.join(status.whitelist) if status.whitelist else '(empty)'}")
    print(f"Max active live runtimes: {status.max_active_runtimes}")
    print(f"Kill switch active: {status.kill_switch_active}")
    if status.kill_switch_reason:
        print(f"Kill switch reason: {status.kill_switch_reason}")
    print(f"Manual recovery required: {status.manual_recovery_required}")
    print(f"Manual recovery approved: {status.manual_recovery_approved}")
    print(f"Model degradation status: {status.model_degradation_status}")
    if status.reasons:
        print("Live gate reasons:")
        for item in status.reasons:
            print(f"- {item}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Next-generation evolution runner")
    parser.add_argument(
        "mode",
        nargs="?",
        default="batch",
        choices=("batch", "autonomy", "autonomy-paper", "autonomy-live", "refresh", "status"),
    )
    parser.add_argument("--portfolio-run-id", type=int, default=None)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float, default=0.0)
    parser.add_argument("--stale-after-minutes", type=int, default=10)
    parser.add_argument("--fail-on-alert", action="store_true")
    live_toggle = parser.add_mutually_exclusive_group()
    live_toggle.add_argument(
        "--enable-live",
        dest="requested_live_override",
        action="store_const",
        const=True,
        default=None,
    )
    live_toggle.add_argument(
        "--disable-live",
        dest="requested_live_override",
        action="store_const",
        const=False,
    )
    parser.add_argument("--live-whitelist", nargs="*", default=None)
    parser.add_argument("--live-max-active-runtimes", type=int, default=None)
    parser.add_argument("--kill-live", action="store_true")
    parser.add_argument("--clear-live-kill-switch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "data" / "cryptoai.db"
    if db_path.exists():
        feed = SQLiteOHLCVFeed(str(db_path))
        registry = PromotionRegistry(str(db_path))
        monitor = PortfolioMonitor(PortfolioTracker(feed, registry), registry)
        if args.mode == "refresh":
            snapshots = monitor.run_cycles(
                portfolio_run_id=args.portfolio_run_id,
                cycles=args.cycles,
                interval_seconds=args.interval_seconds,
            )
            target_run = args.portfolio_run_id
            if target_run is None:
                latest = registry.latest_portfolio_run()
                target_run = int(latest["id"]) if latest else 0
            _print_refresh_summary(target_run or 0, snapshots)
            status = monitor.status(
                portfolio_run_id=target_run,
                stale_after_minutes=args.stale_after_minutes,
            )
            if status is not None:
                _print_health_summary(status)
                if args.fail_on_alert and status["alerts"]:
                    raise SystemExit(2)
            return
        if args.mode == "status":
            status = monitor.status(
                portfolio_run_id=args.portfolio_run_id,
                stale_after_minutes=args.stale_after_minutes,
            )
            if status is None:
                print("No portfolio run available")
                return
            print(f"Portfolio status run id: {status['portfolio_run_id']}")
            print(f"Status: {status['status']}")
            print(f"Snapshot count: {status['snapshot_count']}")
            print(f"Latest snapshot at: {status['latest_snapshot_at']}")
            print(f"Snapshot age seconds: {status['snapshot_age_seconds']}")
            print(f"Equity: {status['equity']}")
            print(f"Realized PnL: {status['realized_pnl']}")
            print(f"Unrealized PnL: {status['unrealized_pnl']}")
            print(f"Gross exposure: {status['gross_exposure']}")
            print(f"Net exposure: {status['net_exposure']}")
            print(f"Open positions: {status['open_positions']}")
            print(f"Closed positions: {status['closed_positions']}")
            print(f"Symbols: {','.join(status['symbols'])}")
            _print_health_summary(status)
            if args.fail_on_alert and status["alerts"]:
                raise SystemExit(2)
            return

        lab = ExperimentLab(feed)
        scheduler = ExperimentScheduler(lab)
        jobs = [
            job
            for job in scheduler.default_jobs(timeframe="5m", min_rows=200, max_symbols=4)
            if job.symbol.endswith("USDT:USDT")
        ]
        if jobs:
            portfolio_total_capital = 10000.0
            previous_states = registry.load_runtime_states(hydrate_legacy=False)
            carryforward_index = _active_runtime_candidate_index(
                registry,
                previous_states,
            )
            persisted_results = []
            latest_prices: dict[str, float] = {}
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
                candles = feed.load_candles(job.symbol, job.timeframe, limit=1)
                if candles:
                    latest_prices[job.symbol] = float(candles[-1]["close"])
                persisted_results.append(
                    registry.persist_experiment(
                        result,
                        notes={
                            "source": "module_demo",
                            "mode": "real_data_batch",
                            "batch_index": index,
                            "batch_size": len(jobs),
                        },
                    )
                )
            _print_batch_summary(persisted_results)
            if args.mode in {"autonomy", "autonomy-paper", "autonomy-live"}:
                evidence_collector = RuntimeEvidenceCollector(feed, lab.engine.config)
                evidence_snapshots = list(
                    evidence_collector.collect(
                        [
                            f"{result.symbol}|{result.timeframe}|{card.genome.strategy_id}"
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
                registry.persist_runtime_evidence(evidence_snapshots)
                autonomy = AutonomousDirector(
                    lab.engine.config,
                    evidence_collector=evidence_collector,
                )
                directive = autonomy.plan_from_experiments(
                    persisted_results,
                    repair_feedback=_repair_feedback_index(
                        registry,
                        persisted_results,
                        lab.engine.config,
                    ),
                    previous_states=previous_states,
                )
                directive = LineageRebuildPlanner(lab.engine.config).expand(
                    directive,
                    results=persisted_results,
                )
                from .live_cycle import AutonomyLiveCycleRunner

                directive = AutonomyLiveCycleRunner._prioritize_repairs(directive)
                autonomy_cycle_id = registry.persist_autonomy_cycle(
                    directive,
                    notes={"source": "module_demo", "mode": args.mode},
                )
                _print_autonomy_summary(autonomy_cycle_id, directive)
                repair_results = RepairCycleRunner(lab, registry=registry).run(
                    directive.repairs,
                    autonomy_cycle_id=autonomy_cycle_id,
                    total_capital=portfolio_total_capital,
                )
                execution_results = list(persisted_results)
                execution_directive = directive
                if repair_results:
                    _print_repair_execution_summary(repair_results)
                    reentry_results, reentry_directives = RepairReentryPlanner().plan(
                        repair_results
                    )
                    if reentry_directives:
                        registry.append_execution_directives(
                            autonomy_cycle_id,
                            reentry_directives,
                            strategy_count_delta=len(reentry_results),
                            notes={"repair_reentry_count": len(reentry_directives)},
                        )
                        execution_results.extend(reentry_results)
                        execution_directive = AutonomyDirective(
                            execution=[*directive.execution, *reentry_directives],
                            repairs=list(directive.repairs),
                            quarantined=list(directive.quarantined),
                            retired=list(directive.retired),
                            notes={
                                **dict(directive.notes or {}),
                                "repair_reentry_count": len(reentry_directives),
                            },
                        )
                        _print_repair_reentry_summary(reentry_directives)
                if args.mode in {"autonomy-paper", "autonomy-live"}:
                    portfolio_allocations = PortfolioAllocator(lab.engine.config).allocate(
                        execution_results,
                        total_capital=portfolio_total_capital,
                        previous_states=previous_states,
                        runtime_evidence=runtime_evidence_index,
                        latest_close_time_by_runtime=latest_close_time_by_runtime,
                        directive=execution_directive,
                    )
                    if args.mode == "autonomy-paper":
                        bridge = AutonomyPaperBridge(
                            feed,
                            registry=registry,
                            config=lab.engine.config,
                        )
                        runtime_states, intents = bridge.apply(
                            results=execution_results,
                            directive=execution_directive,
                            portfolio_allocations=portfolio_allocations,
                            total_capital=portfolio_total_capital,
                            autonomy_cycle_id=autonomy_cycle_id,
                            previous_states=previous_states,
                        )
                        _print_paper_execution_summary(runtime_states, intents)
                    else:
                        settings = get_settings()
                        live_runtime = AutonomyLiveRuntime(feed.storage, settings=settings)
                        if args.kill_live:
                            live_runtime.set_kill_switch(True, reason="cli")
                        if args.clear_live_kill_switch:
                            live_runtime.set_kill_switch(False, reason="cli")
                        if (
                            args.requested_live_override is not None
                            or args.live_whitelist is not None
                            or args.live_max_active_runtimes is not None
                        ):
                            live_runtime.set_operator_request(
                                requested_live=args.requested_live_override,
                                whitelist=(
                                    tuple(args.live_whitelist)
                                    if args.live_whitelist is not None
                                    else None
                                ),
                                max_active_runtimes=args.live_max_active_runtimes,
                                reason="cli",
                            )
                        operator_request = live_runtime.resolve_operator_request(
                            requested_live=args.requested_live_override,
                            whitelist=(
                                tuple(args.live_whitelist)
                                if args.live_whitelist is not None
                                else None
                            ),
                            max_active_runtimes=args.live_max_active_runtimes,
                        )
                        live_config = live_runtime.build_config(
                            base_config=lab.engine.config,
                            requested_live=bool(operator_request["requested_live"]),
                            whitelist=operator_request["whitelist"],
                            max_active_runtimes=operator_request["max_active_runtimes"],
                        )
                        live_status = live_runtime.evaluate(
                            requested_live=bool(operator_request["requested_live"]),
                            config=live_config,
                        )
                        live_config.autonomy_live_enabled = live_status.effective_live
                        _print_live_operator_summary(live_status)
                        bridge = AutonomyLiveBridge(
                            feed,
                            registry=registry,
                            live_trader=live_runtime.build_live_trader(status=live_status),
                            config=live_config,
                            operator_status=live_status,
                        )
                        runtime_states, intents = bridge.apply(
                            results=execution_results,
                            directive=execution_directive,
                            portfolio_allocations=portfolio_allocations,
                            total_capital=portfolio_total_capital,
                            autonomy_cycle_id=autonomy_cycle_id,
                            previous_states=previous_states,
                        )
                        _print_live_execution_summary(runtime_states, intents)
                return
            portfolio_allocations = PortfolioAllocator(lab.engine.config).allocate(
                persisted_results,
                total_capital=portfolio_total_capital,
            )
            portfolio_run_id = registry.persist_portfolio(
                portfolio_allocations,
                total_capital=portfolio_total_capital,
                experiment_results=persisted_results,
                price_by_symbol=latest_prices,
                notes={"source": "module_demo", "mode": "portfolio_batch"},
            )
            monitor.run_cycles(portfolio_run_id=portfolio_run_id, cycles=1, interval_seconds=0.0)
            _print_portfolio_plan(
                portfolio_allocations,
                total_capital=portfolio_total_capital,
            )
            print(f"Portfolio run id: {portfolio_run_id}")
            return

    engine = NextGenEvolutionEngine()
    primitives = [
        StrategyPrimitive(
            family="microtrend_breakout",
            base_params={"lookback": 20.0, "vol_filter": 1.2, "stop": 0.6},
            tags=("trend", "momentum"),
        ),
        StrategyPrimitive(
            family="mean_revert_imbalance",
            base_params={"zscore": 2.0, "inventory_bias": 0.4, "stop": 0.5},
            tags=("reversion", "market_making"),
        ),
    ]
    metrics = {
        "microtrend_breakout:seed": ValidationMetrics(
            backtest_expectancy=0.42,
            walkforward_expectancy=0.28,
            shadow_expectancy=0.17,
            live_expectancy=0.06,
            max_drawdown_pct=6.0,
            trade_count=180,
            cost_drag_pct=0.12,
            latency_ms=80.0,
            regime_consistency=0.74,
        ),
        "mean_revert_imbalance:seed": ValidationMetrics(
            backtest_expectancy=0.22,
            walkforward_expectancy=0.16,
            shadow_expectancy=0.08,
            live_expectancy=0.01,
            max_drawdown_pct=8.0,
            trade_count=120,
            cost_drag_pct=0.18,
            latency_ms=65.0,
            regime_consistency=0.66,
        ),
    }
    promoted, allocations = engine.build_deployment_plan(
        metrics_by_strategy=metrics,
        primitives=primitives,
        total_capital=10000.0,
    )
    print("Promoted candidates:")
    for card in promoted:
        print(
            f"- {card.genome.strategy_id}: stage={card.stage.value} total={card.total_score} reasons={','.join(card.reasons)}"
        )
    print("Capital plan:")
    for item in allocations:
        print(
            f"- {item.strategy_id}: capital={item.allocated_capital} weight={item.weight}"
        )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BrokenPipeError:
        sys.exit(1)
