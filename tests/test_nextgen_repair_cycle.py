import json
import tempfile
import unittest
from pathlib import Path

from core.storage import Storage
from nextgen_evolution import (
    AutonomyDirective,
    EvolutionConfig,
    ExperimentLab,
    NextGenEvolutionEngine,
    PromotionRegistry,
    RepairActionType,
    RepairCycleRunner,
    RepairPlan,
    SQLiteOHLCVFeed,
)
from nextgen_evolution.models import PromotionStage, StrategyGenome


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


class NextGenRepairCycleTests(unittest.TestCase):
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

    def test_repair_cycle_revalidates_candidate_and_persists_lineage(self):
        feed = SQLiteOHLCVFeed(str(self.db_path))
        lab = ExperimentLab(
            feed,
            engine=NextGenEvolutionEngine(
                EvolutionConfig(
                    min_trade_count=8,
                    shadow_threshold=0.05,
                    paper_threshold=0.12,
                    live_threshold=0.20,
                )
            ),
        )
        registry = PromotionRegistry(str(self.db_path))
        autonomy_cycle_id = registry.persist_autonomy_cycle(
            AutonomyDirective(notes={"source": "repair_test"})
        )
        source_runtime_id = "BTC/USDT:USDT|5m|microtrend_breakout:seed"
        candidate = StrategyGenome(
            "microtrend_breakout@BTC_USDT_USDT_5m:repair",
            "microtrend_breakout",
            {
                "lookback": 18.0,
                "breakout_buffer": 0.0022,
                "hold_bars": 5.0,
            },
            mutation_of="microtrend_breakout:seed",
            tags=("repair", "mutate_and_revalidate"),
        )
        runner = RepairCycleRunner(lab, registry=registry)

        results = runner.run(
            [
                RepairPlan(
                    strategy_id=source_runtime_id,
                    action=RepairActionType.MUTATE_AND_REVALIDATE,
                    priority=3,
                    candidate_genome=candidate,
                    validation_stage=PromotionStage.SHADOW,
                    capital_multiplier=0.5,
                    runtime_overrides={"max_weight_multiplier": 0.5},
                    reasons=["live_expectancy_below_floor", "loss_streak"],
                )
            ],
            autonomy_cycle_id=autonomy_cycle_id,
            total_capital=5000.0,
            candle_limit=250,
        )

        self.assertEqual(len(results), 1)
        executed = results[0]
        self.assertEqual(executed.source_runtime_id, source_runtime_id)
        self.assertEqual(executed.symbol, "BTC/USDT:USDT")
        self.assertEqual(executed.timeframe, "5m")
        self.assertIsNotNone(executed.experiment.registry_run_id)
        self.assertEqual(len(executed.experiment.scorecards), 1)
        self.assertEqual(
            executed.experiment.scorecards[0].genome.strategy_id,
            candidate.strategy_id,
        )
        lineage = executed.experiment.notes["repair_validation"]
        self.assertEqual(lineage["source_runtime_id"], source_runtime_id)
        self.assertEqual(lineage["candidate_strategy_id"], candidate.strategy_id)
        self.assertEqual(lineage["repair_action"], RepairActionType.MUTATE_AND_REVALIDATE.value)

        latest_run = registry.latest_run()
        self.assertEqual(latest_run["id"], executed.experiment.registry_run_id)
        run_notes = json.loads(latest_run["notes_json"])
        self.assertEqual(run_notes["repair_validation"]["source_runtime_id"], source_runtime_id)

        repair_rows = registry.latest_repair_executions(limit=10)
        self.assertEqual(len(repair_rows), 1)
        self.assertEqual(repair_rows[0]["autonomy_cycle_id"], autonomy_cycle_id)
        self.assertEqual(repair_rows[0]["source_runtime_id"], source_runtime_id)
        self.assertEqual(repair_rows[0]["candidate_strategy_id"], candidate.strategy_id)
        self.assertEqual(repair_rows[0]["experiment_run_id"], executed.experiment.registry_run_id)
        self.assertEqual(
            repair_rows[0]["status"],
            executed.experiment.scorecards[0].stage.value,
        )
        repair_notes = json.loads(repair_rows[0]["notes_json"])
        self.assertEqual(repair_notes["source_strategy_id"], "microtrend_breakout:seed")


if __name__ == "__main__":
    unittest.main()
