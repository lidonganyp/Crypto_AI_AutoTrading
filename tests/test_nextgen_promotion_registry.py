import tempfile
import unittest
from pathlib import Path
import json

import pandas as pd
from core.dashboard_page_renderers import (
    _coerce_selectbox_value,
    _default_nextgen_focus_mode,
    _build_nextgen_cycle_detail_view,
    _build_nextgen_cycle_focus_view,
    _filter_nextgen_queue_rows,
    _filter_nextgen_cycle_detail_tables,
    _filter_nextgen_focus_view,
    _limit_nextgen_focus_rows,
    _build_nextgen_selected_queue_context,
)
from core.storage import Storage
from nextgen_evolution import (
    AutonomyDirective,
    EvolutionConfig,
    ExecutionAction,
    ExecutionDirective,
    ExperimentLab,
    NextGenEvolutionEngine,
    PortfolioAllocator,
    PortfolioMonitor,
    PortfolioPerformanceSnapshot,
    PortfolioTracker,
    PromotionRegistry,
    RepairActionType,
    RepairPlan,
    SQLiteOHLCVFeed,
)
from nextgen_evolution.models import (
    PortfolioAllocation,
    PromotionStage,
    RuntimeEvidenceSnapshot,
    RuntimeLifecycleState,
    RuntimeState,
    StrategyGenome,
)
from nextgen_evolution.runtime_override_policy import compose_runtime_policy_notes


def make_intraday_candles(count: int, base: float):
    candles = []
    for idx in range(count):
        swing = ((idx % 10) - 5) * 0.2
        shock = -1.0 if idx % 29 == 0 and idx > 0 else 0.0
        price = base + idx * 0.12 + swing + shock
        candles.append(
            {
                "timestamp": 1700000000000 + idx * 300000,
                "open": price,
                "high": price + 0.6,
                "low": price - 0.6,
                "close": price + (0.18 if idx % 4 else -0.08),
                "volume": 900 + idx,
            }
        )
    return candles


class NextGenPromotionRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "cryptoai.db"
        self.storage = Storage(str(self.db_path))
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "5m",
            make_intraday_candles(320, 100.0),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _rows_to_frame(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_registry_persists_experiment_run_scores_and_allocations(self):
        feed = SQLiteOHLCVFeed(str(self.db_path))
        lab = ExperimentLab(
            feed,
            engine=NextGenEvolutionEngine(
                EvolutionConfig(
                    min_trade_count=8,
                    shadow_threshold=0.05,
                    paper_threshold=0.10,
                    live_threshold=0.18,
                )
            ),
        )
        registry = PromotionRegistry(str(self.db_path))

        result = lab.run_for_symbol(symbol="BTC/USDT", timeframe="5m", total_capital=10000.0)
        persisted = registry.persist_experiment(result, notes={"source": "test"})

        self.assertIsNotNone(persisted.registry_run_id)
        latest_run = registry.latest_run()
        self.assertIsNotNone(latest_run)
        self.assertEqual(latest_run["symbol"], "BTC/USDT")
        self.assertEqual(latest_run["timeframe"], "5m")
        self.assertEqual(latest_run["evaluated_candidates"], len(result.scorecards))
        self.assertEqual(latest_run["promoted_candidates"], len(result.promoted))
        notes = json.loads(latest_run["notes_json"])
        self.assertIn("feature_summary", notes)
        latest_scores = registry.latest_scores(limit=50)
        self.assertEqual(len(latest_scores), len(result.scorecards))

    def test_registry_loads_latest_candidate_genomes_by_strategy_id(self):
        feed = SQLiteOHLCVFeed(str(self.db_path))
        lab = ExperimentLab(
            feed,
            engine=NextGenEvolutionEngine(
                EvolutionConfig(
                    min_trade_count=8,
                    shadow_threshold=0.05,
                    paper_threshold=0.10,
                    live_threshold=0.18,
                )
            ),
        )
        registry = PromotionRegistry(str(self.db_path))

        result = registry.persist_experiment(
            lab.run_for_symbol(symbol="BTC/USDT", timeframe="5m", total_capital=10000.0)
        )
        target = result.scorecards[0].genome.strategy_id

        genomes = registry.load_candidate_genomes([target, "missing:strategy"])

        self.assertIn(target, genomes)
        self.assertEqual(genomes[target].strategy_id, target)
        self.assertEqual(
            genomes[target].params,
            result.scorecards[0].genome.params,
        )
        self.assertNotIn("missing:strategy", genomes)

    def test_registry_persists_portfolio_allocations(self):
        feed = SQLiteOHLCVFeed(str(self.db_path))
        engine = NextGenEvolutionEngine(
            EvolutionConfig(
                min_trade_count=8,
                shadow_threshold=0.05,
                paper_threshold=0.10,
                live_threshold=0.18,
            )
        )
        lab = ExperimentLab(feed, engine=engine)
        registry = PromotionRegistry(str(self.db_path))

        result = lab.run_for_symbol(symbol="BTC/USDT", timeframe="5m", total_capital=10000.0)
        persisted = registry.persist_experiment(result, notes={"source": "test"})
        allocations = PortfolioAllocator(engine.config).allocate(
            [persisted],
            total_capital=10000.0,
        )
        portfolio_run_id = registry.persist_portfolio(
            allocations,
            total_capital=10000.0,
            experiment_results=[persisted],
            price_by_symbol={"BTC/USDT:USDT": 138.38},
            notes={"scope": "test_portfolio"},
        )

        self.assertGreater(portfolio_run_id, 0)
        latest_portfolio_run = registry.latest_portfolio_run()
        self.assertIsNotNone(latest_portfolio_run)
        self.assertEqual(latest_portfolio_run["allocation_count"], len(allocations))
        self.assertEqual(latest_portfolio_run["symbol_count"], len({item.symbol for item in allocations}))
        portfolio_notes = json.loads(latest_portfolio_run["notes_json"])
        self.assertEqual(portfolio_notes["scope"], "test_portfolio")
        latest_portfolio_allocations = registry.latest_portfolio_allocations(limit=20)
        self.assertEqual(len(latest_portfolio_allocations), len(allocations))
        self.assertGreaterEqual(latest_portfolio_allocations[0]["entry_price"], 0.0)
        latest_snapshots = registry.latest_portfolio_snapshots(limit=20)
        self.assertEqual(len(latest_snapshots), 1)
        self.assertEqual(latest_snapshots[0]["portfolio_run_id"], portfolio_run_id)
        self.assertEqual(latest_snapshots[0]["status"], "active")

    def test_registry_updates_portfolio_performance_summary_from_snapshot(self):
        registry = PromotionRegistry(str(self.db_path))
        portfolio_run_id = registry.persist_portfolio(
            allocations=[],
            total_capital=10000.0,
            notes={"scope": "snapshot_update"},
        )

        snapshot_id = registry.persist_portfolio_snapshot(
            portfolio_run_id,
            PortfolioPerformanceSnapshot(
                realized_pnl=125.5,
                unrealized_pnl=-20.0,
                equity=10105.5,
                gross_exposure=4800.0,
                net_exposure=3500.0,
                open_positions=2,
                closed_positions=3,
                win_rate=0.6667,
                max_drawdown_pct=2.4,
                status="monitoring",
                notes={"cycle": "mark_to_market"},
            ),
        )

        self.assertGreater(snapshot_id, 0)
        latest_portfolio_run = registry.latest_portfolio_run()
        self.assertEqual(latest_portfolio_run["id"], portfolio_run_id)
        self.assertEqual(latest_portfolio_run["status"], "monitoring")
        self.assertEqual(latest_portfolio_run["latest_realized_pnl"], 125.5)
        self.assertEqual(latest_portfolio_run["latest_unrealized_pnl"], -20.0)
        self.assertEqual(latest_portfolio_run["latest_equity"], 10105.5)
        self.assertEqual(latest_portfolio_run["latest_open_positions"], 2)
        self.assertEqual(latest_portfolio_run["latest_closed_positions"], 3)
        self.assertEqual(latest_portfolio_run["latest_win_rate"], 0.6667)
        latest_snapshots = registry.latest_portfolio_snapshots(limit=5)
        self.assertEqual(latest_snapshots[0]["portfolio_run_id"], portfolio_run_id)
        snapshot_notes = json.loads(latest_snapshots[0]["notes_json"])
        self.assertEqual(snapshot_notes["cycle"], "mark_to_market")

    def test_portfolio_tracker_marks_to_market_from_latest_close(self):
        feed = SQLiteOHLCVFeed(str(self.db_path))
        registry = PromotionRegistry(str(self.db_path))
        portfolio_run_id = registry.persist_portfolio(
            allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="volatility_reclaim:seed",
                    family="volatility_reclaim",
                    stage=PromotionStage.PAPER,
                    allocated_capital=2000.0,
                    weight=0.2,
                    score=0.35,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            price_by_symbol={"BTC/USDT:USDT": 138.38},
            notes={"scope": "mark_test"},
        )
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "5m",
            [
                {
                    "timestamp": 1700000000000 + 320 * 300000,
                    "open": 145.0,
                    "high": 146.0,
                    "low": 144.0,
                    "close": 145.0,
                    "volume": 1500.0,
                }
            ],
        )
        tracker = PortfolioTracker(feed, registry)
        snapshot = tracker.refresh(portfolio_run_id)

        self.assertIsNotNone(snapshot)
        self.assertGreater(snapshot.unrealized_pnl, 0.0)
        self.assertGreater(snapshot.equity, 10000.0)
        latest_portfolio_run = registry.portfolio_run(portfolio_run_id)
        self.assertEqual(latest_portfolio_run["latest_equity"], snapshot.equity)
        allocations = registry.portfolio_allocations(portfolio_run_id)
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0]["last_price"], 145.0)
        self.assertGreater(allocations[0]["unrealized_pnl"], 0.0)

    def test_portfolio_monitor_runs_multiple_refresh_cycles(self):
        feed = SQLiteOHLCVFeed(str(self.db_path))
        registry = PromotionRegistry(str(self.db_path))
        portfolio_run_id = registry.persist_portfolio(
            allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="trend_pullback_continuation:seed",
                    family="trend_pullback_continuation",
                    stage=PromotionStage.PAPER,
                    allocated_capital=1500.0,
                    weight=0.15,
                    score=0.31,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            price_by_symbol={"BTC/USDT:USDT": 138.38},
            notes={"scope": "monitor_test"},
        )
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "5m",
            [
                {
                    "timestamp": 1700000000000 + 321 * 300000,
                    "open": 146.0,
                    "high": 147.0,
                    "low": 145.0,
                    "close": 146.0,
                    "volume": 1600.0,
                }
            ],
        )
        monitor = PortfolioMonitor(PortfolioTracker(feed, registry), registry)

        snapshots = monitor.run_cycles(
            portfolio_run_id=portfolio_run_id,
            cycles=2,
            interval_seconds=0.0,
        )

        self.assertEqual(len(snapshots), 2)
        self.assertGreaterEqual(snapshots[-1].equity, snapshots[0].equity)
        latest_snapshots = registry.latest_portfolio_snapshots(limit=10)
        tagged = [row for row in latest_snapshots if row["portfolio_run_id"] == portfolio_run_id]
        self.assertGreaterEqual(len(tagged), 3)

    def test_portfolio_monitor_reports_status_summary(self):
        feed = SQLiteOHLCVFeed(str(self.db_path))
        registry = PromotionRegistry(str(self.db_path))
        portfolio_run_id = registry.persist_portfolio(
            allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="volatility_reclaim:seed",
                    family="volatility_reclaim",
                    stage=PromotionStage.PAPER,
                    allocated_capital=1200.0,
                    weight=0.12,
                    score=0.33,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            price_by_symbol={"BTC/USDT:USDT": 138.38},
            notes={"scope": "status_test"},
        )
        monitor = PortfolioMonitor(PortfolioTracker(feed, registry), registry)

        status = monitor.status(portfolio_run_id=portfolio_run_id, stale_after_minutes=10)

        self.assertIsNotNone(status)
        self.assertEqual(status["portfolio_run_id"], portfolio_run_id)
        self.assertEqual(status["health"], "healthy")
        self.assertEqual(status["freshness"], "fresh")
        self.assertEqual(status["allocation_count"], 1)
        self.assertEqual(status["alerts"], [])
        self.assertIn("BTC/USDT:USDT", status["symbols"])

    def test_portfolio_monitor_reports_health_alerts(self):
        feed = SQLiteOHLCVFeed(str(self.db_path))
        registry = PromotionRegistry(str(self.db_path))
        portfolio_run_id = registry.persist_portfolio(
            allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="trend_pullback_continuation:seed",
                    family="trend_pullback_continuation",
                    stage=PromotionStage.LIVE,
                    allocated_capital=9000.0,
                    weight=0.9,
                    score=0.51,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            price_by_symbol={"BTC/USDT:USDT": 138.38},
            notes={"scope": "alert_test"},
        )
        registry.persist_portfolio_snapshot(
            portfolio_run_id,
            PortfolioPerformanceSnapshot(
                equity=9300.0,
                gross_exposure=9000.0,
                net_exposure=9000.0,
                open_positions=1,
                max_drawdown_pct=9.5,
                status="active",
            ),
        )
        monitor = PortfolioMonitor(PortfolioTracker(feed, registry), registry)

        status = monitor.status(portfolio_run_id=portfolio_run_id, stale_after_minutes=10)

        self.assertIsNotNone(status)
        self.assertEqual(status["health"], "alerting")
        alert_codes = {item["code"] for item in status["alerts"]}
        self.assertEqual(
            alert_codes,
            {"drawdown_limit", "equity_floor", "gross_exposure_limit"},
        )

    def test_registry_persists_autonomy_cycles_execution_and_repairs(self):
        registry = PromotionRegistry(str(self.db_path))

        autonomy_cycle_id = registry.persist_autonomy_cycle(
            AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id="trend_pullback_continuation:seed",
                        action=ExecutionAction.PROMOTE_TO_PAPER,
                        from_stage=PromotionStage.SHADOW,
                        target_stage=PromotionStage.PAPER,
                        capital_multiplier=1.0,
                        reasons=["shadow_requirements_met"],
                    )
                ],
                repairs=[
                    RepairPlan(
                        strategy_id="volatility_reclaim:seed",
                        action=RepairActionType.QUARANTINE,
                        priority=4,
                        candidate_genome=StrategyGenome(
                            "volatility_reclaim:seed:repair",
                            "volatility_reclaim",
                            {"lookback": 22.0, "hold_bars": 6.0},
                            mutation_of="volatility_reclaim:seed",
                            tags=("repair", "quarantine"),
                        ),
                        validation_stage=PromotionStage.SHADOW,
                        capital_multiplier=0.5,
                        runtime_overrides={"max_weight_multiplier": 0.5},
                        reasons=["drawdown_breach"],
                    )
                ],
                quarantined=["volatility_reclaim:seed"],
                retired=["fragile_momentum:seed"],
                notes={
                    "strategy_count": 2,
                    "source": "test",
                    "repair_queue_runtime_ids": [
                        "volatility_reclaim:seed",
                        "trend_pullback_continuation:seed",
                    ],
                    "repair_queue_actions": ["quarantine", "promote_to_paper"],
                    "repair_queue_priorities": [4, 1],
                },
            )
        )

        self.assertGreater(autonomy_cycle_id, 0)
        latest_cycle = registry.latest_autonomy_cycle()
        self.assertIsNotNone(latest_cycle)
        self.assertEqual(latest_cycle["repair_count"], 1)
        cycle_notes = json.loads(latest_cycle["notes_json"])
        self.assertEqual(cycle_notes["source"], "test")
        directives = registry.latest_execution_directives(limit=10)
        self.assertEqual(len(directives), 1)
        self.assertEqual(directives[0]["action"], ExecutionAction.PROMOTE_TO_PAPER.value)
        repairs = registry.latest_repair_plans(limit=10)
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0]["action"], RepairActionType.QUARANTINE.value)
        repair_params = json.loads(repairs[0]["candidate_params_json"])
        self.assertEqual(repair_params["hold_bars"], 6.0)

        cycle_meta, cycle_notes, execution_df, repair_df = _build_nextgen_cycle_detail_view(
            cycle_row=latest_cycle,
            execution_rows=self._rows_to_frame(directives),
            repair_rows=self._rows_to_frame(repairs),
            load_json=lambda value, default=None: json.loads(value) if value else (default or {}),
        )
        self.assertEqual(cycle_meta["id"], autonomy_cycle_id)
        self.assertEqual(cycle_notes["source"], "test")
        self.assertIn("shadow_requirements_met", execution_df.iloc[0]["reasons"])
        self.assertIn("drawdown_breach", repair_df.iloc[0]["reasons"])
        self.assertIn("max_weight_multiplier", repair_df.iloc[0]["runtime_overrides"])
        focus_meta, focused_execution_df, focused_repair_df = _build_nextgen_cycle_focus_view(
            cycle_notes=cycle_notes,
            execution_df=execution_df,
            repair_df=repair_df,
        )
        self.assertEqual(
            focus_meta["runtime_ids"],
            ["volatility_reclaim:seed", "trend_pullback_continuation:seed"],
        )
        self.assertIn("quarantine", focus_meta["actions_text"])
        self.assertIn("promote_to_paper", focus_meta["actions_text"])
        self.assertIn("4", focus_meta["priorities_text"])
        self.assertEqual(len(focused_execution_df), 1)
        self.assertEqual(len(focused_repair_df), 1)
        self.assertEqual(int(focused_execution_df.iloc[0]["queue_priority"]), 1)
        self.assertEqual(focused_execution_df.iloc[0]["queue_action"], "promote_to_paper")
        self.assertEqual(int(focused_repair_df.iloc[0]["queue_priority"]), 4)
        self.assertEqual(focused_repair_df.iloc[0]["queue_action"], "quarantine")
        all_execution_df, all_repair_df = _filter_nextgen_focus_view(
            mode="all",
            execution_df=focused_execution_df,
            repair_df=focused_repair_df,
        )
        self.assertEqual(
            _coerce_selectbox_value([1, 2, 3], 2, fallback=1),
            2,
        )
        self.assertEqual(
            _coerce_selectbox_value([1, 2, 3], 9, fallback=1),
            1,
        )
        self.assertEqual(
            _coerce_selectbox_value([1, 2, 3], 9, fallback=7),
            1,
        )
        self.assertEqual(
            _default_nextgen_focus_mode(
                execution_df=focused_execution_df,
                repair_df=focused_repair_df,
            ),
            "repair",
        )
        self.assertEqual(
            _default_nextgen_focus_mode(
                execution_df=focused_execution_df,
                repair_df=pd.DataFrame(),
            ),
            "execution",
        )
        self.assertEqual(
            _default_nextgen_focus_mode(
                execution_df=pd.DataFrame(),
                repair_df=pd.DataFrame(),
            ),
            "all",
        )
        filtered_queue_rows = _filter_nextgen_queue_rows(
            query="market_data_latency",
            rows=[
                {
                    "autonomy_cycle_id": 101,
                    "trigger": "scheduler",
                    "reason": "rotation_cleanup",
                    "latest_issue_event_type": "",
                    "latest_issue_reason": "",
                },
                {
                    "autonomy_cycle_id": 102,
                    "trigger": "manual_recovery_required",
                    "reason": "market_data_latency",
                    "latest_issue_event_type": "nextgen_autonomy_live_guard_callback_failed",
                    "latest_issue_reason": "nextgen_guard_boom",
                },
            ],
        )
        self.assertEqual(len(filtered_queue_rows), 1)
        self.assertEqual(filtered_queue_rows[0]["autonomy_cycle_id"], 102)
        filtered_cycle_execution_df, filtered_cycle_repair_df = _filter_nextgen_cycle_detail_tables(
            query="volatility",
            execution_df=focused_execution_df,
            repair_df=focused_repair_df,
        )
        self.assertTrue(filtered_cycle_execution_df.empty)
        self.assertEqual(len(filtered_cycle_repair_df), 1)
        filtered_cycle_execution_df, filtered_cycle_repair_df = _filter_nextgen_cycle_detail_tables(
            query="promote_to_paper",
            execution_df=focused_execution_df,
            repair_df=focused_repair_df,
        )
        self.assertEqual(len(filtered_cycle_execution_df), 1)
        self.assertTrue(filtered_cycle_repair_df.empty)
        execution_only_df, repair_only_empty_df = _filter_nextgen_focus_view(
            mode="execution",
            execution_df=focused_execution_df,
            repair_df=focused_repair_df,
        )
        execution_empty_df, repair_only_df = _filter_nextgen_focus_view(
            mode="repair",
            execution_df=focused_execution_df,
            repair_df=focused_repair_df,
        )
        self.assertEqual(len(all_execution_df), 1)
        self.assertEqual(len(all_repair_df), 1)
        self.assertEqual(len(execution_only_df), 1)
        self.assertTrue(repair_only_empty_df.empty)
        self.assertTrue(execution_empty_df.empty)
        self.assertEqual(len(repair_only_df), 1)
        limited_execution_df, hidden_execution = _limit_nextgen_focus_rows(
            pd.concat([focused_execution_df] * 6, ignore_index=True),
            limit=5,
        )
        self.assertEqual(len(limited_execution_df), 5)
        self.assertEqual(hidden_execution, 1)
        queue_context = _build_nextgen_selected_queue_context(
            {
                "autonomy_cycle_id": autonomy_cycle_id,
                "created_at": latest_cycle["created_at"],
                "trigger": "scheduler",
                "reason": "rotation_cleanup",
                "latest_issue_event_type": "nextgen_autonomy_live_guard_callback_failed",
                "latest_issue_reason": "nextgen_guard_boom",
            }
        )
        self.assertEqual(queue_context["autonomy_cycle_id"], autonomy_cycle_id)
        self.assertEqual(queue_context["trigger"], "scheduler")
        self.assertEqual(
            queue_context["latest_issue_event_type"],
            "nextgen_autonomy_live_guard_callback_failed",
        )
        self.assertEqual(queue_context["latest_issue_reason"], "nextgen_guard_boom")

    def test_registry_appends_execution_directives_to_existing_cycle(self):
        registry = PromotionRegistry(str(self.db_path))
        autonomy_cycle_id = registry.persist_autonomy_cycle(
            AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id="BTC/USDT:USDT|5m|microtrend_breakout:seed",
                        action=ExecutionAction.PROMOTE_TO_SHADOW,
                        from_stage=PromotionStage.REJECT,
                        target_stage=PromotionStage.SHADOW,
                        reasons=["candidate_enters_shadow"],
                    )
                ],
                notes={
                    "strategy_count": 1,
                    "source": "append_test",
                    "repair_queue_hold_priority_count": 1,
                    "repair_queue_postponed_rebuild_count": 2,
                },
            )
        )

        registry.append_execution_directives(
            autonomy_cycle_id,
            [
                ExecutionDirective(
                    strategy_id="BTC/USDT:USDT|5m|microtrend_breakout@BTC_USDT_USDT_5m:repair",
                    action=ExecutionAction.PROMOTE_TO_PAPER,
                    from_stage=PromotionStage.REJECT,
                    target_stage=PromotionStage.PAPER,
                    capital_multiplier=0.5,
                    reasons=["repair_revalidation_passed"],
                )
            ],
            strategy_count_delta=1,
            notes={"repair_reentry_count": 1},
        )

        latest_cycle = registry.latest_autonomy_cycle()
        self.assertEqual(latest_cycle["execution_count"], 2)
        self.assertEqual(latest_cycle["strategy_count"], 2)
        cycle_notes = json.loads(latest_cycle["notes_json"])
        self.assertEqual(cycle_notes["repair_reentry_count"], 1)
        self.assertEqual(cycle_notes["repair_queue_hold_priority_count"], 1)
        self.assertEqual(cycle_notes["repair_queue_postponed_rebuild_count"], 2)
        directives = registry.latest_execution_directives(limit=10)
        self.assertEqual(len(directives), 2)
        self.assertEqual(directives[0]["action"], ExecutionAction.PROMOTE_TO_PAPER.value)

    def test_registry_persists_runtime_evidence_snapshots(self):
        registry = PromotionRegistry(str(self.db_path))

        registry.persist_runtime_evidence(
            [
                RuntimeEvidenceSnapshot(
                    runtime_id="BTC/USDT:USDT|5m|volatility_reclaim:seed",
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="volatility_reclaim:seed",
                    family="volatility_reclaim",
                    open_position=True,
                    current_capital=980.0,
                    realized_pnl=-25.0,
                    unrealized_pnl=-10.0,
                    total_net_pnl=-35.0,
                    current_drawdown_pct=3.5,
                    max_drawdown_pct=4.2,
                    closed_trade_count=2,
                    win_rate=0.0,
                    consecutive_losses=2,
                    health_status="degraded",
                    notes={"source": "test"},
                )
            ]
        )

        evidence_rows = registry.latest_runtime_evidence(limit=10)
        self.assertEqual(len(evidence_rows), 1)
        self.assertEqual(evidence_rows[0]["health_status"], "degraded")
        self.assertEqual(evidence_rows[0]["runtime_id"], "BTC/USDT:USDT|5m|volatility_reclaim:seed")
        evidence_notes = json.loads(evidence_rows[0]["notes_json"])
        self.assertEqual(evidence_notes["source"], "test")

    def test_registry_strips_legacy_runtime_policy_fields_on_persist_and_hydrates_on_load(self):
        registry = PromotionRegistry(str(self.db_path))
        state = RuntimeState(
            runtime_id="BTC/USDT:USDT|5m|trend_pullback_continuation:seed",
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            strategy_id="trend_pullback_continuation:seed",
            family="trend_pullback_continuation",
            lifecycle_state=RuntimeLifecycleState.PAPER,
            promotion_stage=PromotionStage.PAPER,
            target_stage=PromotionStage.PAPER,
            last_directive_action=ExecutionAction.KEEP,
            score=0.42,
            allocated_capital=1000.0,
            desired_capital=550.0,
            current_capital=0.0,
            current_weight=0.10,
            notes=compose_runtime_policy_notes(
                base_notes={"tag": "runtime_policy"},
                repair_reentry_notes={
                    "source_runtime_id": "BTC/USDT:USDT|5m|seed_runtime",
                    "runtime_overrides": {
                        "max_weight_multiplier": 0.55,
                    },
                },
                runtime_overrides={
                    "max_weight_multiplier": 0.55,
                },
                runtime_override_state={
                    "cycles_since_refresh": 1,
                    "fresh_refresh": False,
                    "recovery_mode": "neutral",
                },
                reentry_state={
                    "mode": "repair_reentry",
                    "phase": "probation",
                    "source_runtime_id": "BTC/USDT:USDT|5m|seed_runtime",
                    "active_overrides": ["max_weight_multiplier"],
                },
            ),
        )

        registry.persist_runtime_states([state])

        stored = registry.latest_runtime_states(limit=1)[0]
        stored_notes = json.loads(stored["notes_json"])
        self.assertIn("runtime_lifecycle_policy", stored_notes)
        self.assertNotIn("runtime_overrides", stored_notes)
        self.assertNotIn("runtime_override_state", stored_notes)
        self.assertNotIn("reentry_state", stored_notes)
        self.assertNotIn("repair_reentry", stored_notes)

        loaded = registry.load_runtime_states()[0]
        self.assertEqual(loaded.notes["runtime_overrides"]["max_weight_multiplier"], 0.55)
        self.assertEqual(loaded.notes["runtime_override_state"]["recovery_mode"], "neutral")
        self.assertEqual(loaded.notes["reentry_state"]["phase"], "probation")
        self.assertEqual(
            loaded.notes["repair_reentry"]["source_runtime_id"],
            "BTC/USDT:USDT|5m|seed_runtime",
        )


if __name__ == "__main__":
    unittest.main()
