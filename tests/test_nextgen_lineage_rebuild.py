import tempfile
import unittest
from pathlib import Path

from core.storage import Storage
from nextgen_evolution import (
    AutonomyDirective,
    AutonomyPlanner,
    EvolutionConfig,
    ExperimentLab,
    LineageRebuildPlanner,
    NextGenEvolutionEngine,
    PromotionRegistry,
    RepairActionType,
    RepairCycleRunner,
    RepairFeedbackSummary,
    RepairPlan,
    RepairReentryPlanner,
    SQLiteOHLCVFeed,
)
from nextgen_evolution.experiment_lab import ExperimentResult
from nextgen_evolution.models import PromotionStage, ScoreCard, StrategyGenome, StrategyRuntimeSnapshot, ValidationMetrics


def make_intraday_candles(count: int, base: float, step: float = 0.15):
    candles = []
    for idx in range(count):
        swing = ((idx % 12) - 6) * 0.18
        shock = -1.1 if idx % 37 == 0 and idx > 0 else 0.0
        reclaim = 0.9 if idx % 37 == 3 else 0.0
        price = base + idx * step + swing + shock + reclaim
        candles.append(
            {
                "timestamp": 1700000000000 + idx * 300000,
                "open": price,
                "high": price + 0.55,
                "low": price - 0.55,
                "close": price + (0.16 if idx % 5 else -0.12),
                "volume": 1000 + idx * 2,
            }
        )
    return candles


class NextGenLineageRebuildTests(unittest.TestCase):
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

    def test_rebuild_planner_generates_alternative_family_repairs(self):
        runtime_id = "BTC/USDT:USDT|5m|microtrend_breakout:seed"
        result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                ScoreCard(
                    genome=StrategyGenome(
                        "microtrend_breakout:seed",
                        "microtrend_breakout",
                        {"lookback": 18.0, "breakout_buffer": 0.002, "hold_bars": 6.0},
                    ),
                    stage=PromotionStage.PAPER,
                    edge_score=0.18,
                    robustness_score=0.52,
                    deployment_score=0.16,
                    total_score=0.20,
                    reasons=["promote_shadow"],
                )
            ],
            promoted=[],
            allocations=[],
            candle_count=250,
            metrics_by_strategy={
                "microtrend_breakout:seed": ValidationMetrics(
                    backtest_expectancy=0.15,
                    walkforward_expectancy=-0.02,
                    shadow_expectancy=0.01,
                    live_expectancy=-0.03,
                    max_drawdown_pct=8.0,
                    trade_count=24,
                    cost_drag_pct=0.20,
                    latency_ms=35.0,
                    regime_consistency=0.48,
                )
            },
        )
        directive = AutonomyDirective(
            repairs=[
                RepairPlan(
                    strategy_id=runtime_id,
                    action=RepairActionType.RETIRE,
                    priority=6,
                    candidate_genome=None,
                    validation_stage=PromotionStage.REJECT,
                    capital_multiplier=0.0,
                    reasons=["repair_lineage_exhausted", "regime_consistency_low"],
                )
            ],
            retired=[runtime_id],
        )

        rebuilds = LineageRebuildPlanner(EvolutionConfig()).plan(
            directive,
            results=[result],
        )

        self.assertEqual(len(rebuilds), 2)
        self.assertTrue(all(item.action == RepairActionType.REBUILD_LINEAGE for item in rebuilds))
        self.assertTrue(all(item.candidate_genome is not None for item in rebuilds))
        self.assertTrue(
            all(item.candidate_genome.family != "microtrend_breakout" for item in rebuilds)
        )
        self.assertTrue(
            all(item.candidate_genome.mutation_of is None for item in rebuilds)
        )
        self.assertTrue(
            all("lineage_rebuild_requested" in item.reasons for item in rebuilds)
        )

    def test_rebuild_candidate_flows_through_repair_cycle_and_reentry(self):
        feed = SQLiteOHLCVFeed(str(self.db_path))
        config = EvolutionConfig(
            min_trade_count=8,
            shadow_threshold=0.0,
            paper_threshold=0.08,
            live_threshold=0.15,
        )
        lab = ExperimentLab(
            feed,
            engine=NextGenEvolutionEngine(config),
        )
        registry = PromotionRegistry(str(self.db_path))
        source_candidate = StrategyGenome(
            "microtrend_breakout:seed",
            "microtrend_breakout",
            {
                "lookback": 18.0,
                "breakout_buffer": 0.002,
                "hold_bars": 6.0,
            },
        )
        source_result = lab.run_candidates_for_symbol(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            genomes=[source_candidate],
            total_capital=5000.0,
            candle_limit=250,
        )
        runtime_id = "BTC/USDT:USDT|5m|microtrend_breakout:seed"
        directive = LineageRebuildPlanner(config).expand(
            AutonomyDirective(
                repairs=[
                    RepairPlan(
                        strategy_id=runtime_id,
                        action=RepairActionType.RETIRE,
                        priority=6,
                        candidate_genome=None,
                        validation_stage=PromotionStage.REJECT,
                        capital_multiplier=0.0,
                        reasons=["repair_lineage_exhausted", "repair_failures:3"],
                    )
                ],
                retired=[runtime_id],
            ),
            results=[source_result],
        )
        autonomy_cycle_id = registry.persist_autonomy_cycle(
            directive,
            notes={"source": "lineage_rebuild_test"},
        )

        repair_results = RepairCycleRunner(lab, registry=registry).run(
            directive.repairs,
            autonomy_cycle_id=autonomy_cycle_id,
            total_capital=5000.0,
            candle_limit=250,
        )
        reentry_results, reentry_directives = RepairReentryPlanner().plan(repair_results)

        self.assertTrue(any(item.action == RepairActionType.REBUILD_LINEAGE for item in directive.repairs))
        self.assertTrue(repair_results)
        self.assertTrue(
            all(item.plan.action == RepairActionType.REBUILD_LINEAGE for item in repair_results)
        )
        self.assertTrue(
            all(item.plan.candidate_genome.family != "microtrend_breakout" for item in repair_results)
        )
        self.assertTrue(reentry_results)
        self.assertTrue(reentry_directives)
        self.assertTrue(
            all("repair_action:rebuild_lineage" in item.reasons for item in reentry_directives)
        )

    def test_rebuild_candidate_does_not_inherit_parent_retire_feedback(self):
        source_runtime_id = "BTC/USDT:USDT|5m|microtrend_breakout:seed"
        source_result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                ScoreCard(
                    genome=StrategyGenome(
                        "microtrend_breakout:seed",
                        "microtrend_breakout",
                        {"lookback": 18.0, "breakout_buffer": 0.002, "hold_bars": 6.0},
                    ),
                    stage=PromotionStage.PAPER,
                    edge_score=0.18,
                    robustness_score=0.52,
                    deployment_score=0.16,
                    total_score=0.20,
                    reasons=["promote_shadow"],
                )
            ],
            promoted=[],
            allocations=[],
            candle_count=250,
            metrics_by_strategy={},
        )
        rebuild = LineageRebuildPlanner(EvolutionConfig()).plan(
            AutonomyDirective(
                repairs=[
                    RepairPlan(
                        strategy_id=source_runtime_id,
                        action=RepairActionType.RETIRE,
                        priority=6,
                        candidate_genome=None,
                        validation_stage=PromotionStage.REJECT,
                        reasons=["repair_lineage_exhausted"],
                    )
                ],
                retired=[source_runtime_id],
            ),
            results=[source_result],
        )[0]
        candidate = rebuild.candidate_genome
        self.assertIsNotNone(candidate)
        snapshot = StrategyRuntimeSnapshot(
            scorecard=ScoreCard(
                genome=candidate,
                stage=PromotionStage.SHADOW,
                edge_score=0.40,
                robustness_score=0.72,
                deployment_score=0.44,
                total_score=0.41,
                reasons=["candidate_enters_shadow"],
            ),
            metrics=ValidationMetrics(
                backtest_expectancy=0.28,
                walkforward_expectancy=0.18,
                shadow_expectancy=0.12,
                live_expectancy=0.05,
                max_drawdown_pct=4.0,
                trade_count=32,
                cost_drag_pct=0.12,
                latency_ms=40.0,
                regime_consistency=0.82,
            ),
            runtime_id=f"BTC/USDT:USDT|5m|{candidate.strategy_id}",
            current_drawdown_pct=1.0,
            consecutive_losses=0,
            notes={"symbol": "BTC/USDT:USDT", "timeframe": "5m"},
        )

        directive = AutonomyPlanner(
            EvolutionConfig(
                paper_threshold=0.30,
                autonomy_min_runtime_trades=24,
                autonomy_repair_expectancy_floor=0.0,
            )
        ).plan(
            [snapshot],
            repair_feedback={
                source_runtime_id: RepairFeedbackSummary(
                    source_strategy_id="microtrend_breakout:seed",
                    source_runtime_id=source_runtime_id,
                    attempts=3,
                    failures=3,
                    consecutive_failures=3,
                    probation_required=True,
                    retire_recommended=True,
                ),
                "microtrend_breakout:seed": RepairFeedbackSummary(
                    source_strategy_id="microtrend_breakout:seed",
                    source_runtime_id=source_runtime_id,
                    attempts=3,
                    failures=3,
                    consecutive_failures=3,
                    probation_required=True,
                    retire_recommended=True,
                ),
            },
        )

        self.assertEqual(directive.retired, [])
        self.assertEqual(directive.repairs, [])
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.PAPER)


if __name__ == "__main__":
    unittest.main()
