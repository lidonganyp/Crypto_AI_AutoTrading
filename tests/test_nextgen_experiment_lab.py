import tempfile
import unittest
from pathlib import Path

from core.storage import Storage
from nextgen_evolution import (
    EvolutionConfig,
    ExperimentLab,
    ExperimentScheduler,
    NextGenEvolutionEngine,
    SQLiteOHLCVFeed,
)
from nextgen_evolution.models import StrategyGenome


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


class NextGenExperimentLabTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "cryptoai.db"
        self.storage = Storage(str(self.db_path))
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "5m",
            make_intraday_candles(320, 100.0),
        )
        self.storage.insert_ohlcv(
            "ETH/USDT:USDT",
            "5m",
            make_intraday_candles(280, 200.0, step=0.10),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sqlite_feed_lists_symbols_and_loads_candles(self):
        feed = SQLiteOHLCVFeed(str(self.db_path))

        symbols = feed.list_symbols("5m", min_rows=200)
        candles = feed.load_candles("BTC/USDT", "5m", limit=50)

        self.assertIn("BTC/USDT:USDT", symbols)
        self.assertEqual(len(candles), 50)
        self.assertLess(candles[0]["timestamp"], candles[-1]["timestamp"])

    def test_experiment_lab_runs_on_real_sqlite_candles(self):
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

        result = lab.run_for_symbol(
            symbol="BTC/USDT",
            timeframe="5m",
            total_capital=10000.0,
            candle_limit=250,
        )

        self.assertEqual(result.symbol, "BTC/USDT")
        self.assertEqual(result.timeframe, "5m")
        self.assertTrue(result.promoted)
        self.assertIn("feature_summary", result.notes)
        self.assertTrue(any(card.total_score != result.promoted[0].total_score for card in result.scorecards[1:]))

    def test_default_primitives_include_trend_pullback_family(self):
        families = {primitive.family for primitive in ExperimentLab.default_primitives()}

        self.assertIn("trend_pullback_continuation", families)
        self.assertIn("volatility_reclaim", families)

    def test_experiment_lab_runs_explicit_candidates_without_regenerating_population(self):
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
        candidate = StrategyGenome(
            "microtrend_breakout:repair1",
            "microtrend_breakout",
            {
                "lookback": 18.0,
                "breakout_buffer": 0.0022,
                "hold_bars": 6.0,
            },
            mutation_of="microtrend_breakout:seed",
            tags=("repair", "mutate_and_revalidate"),
        )

        result = lab.run_candidates_for_symbol(
            symbol="BTC/USDT",
            timeframe="5m",
            genomes=[candidate],
            total_capital=5000.0,
            candle_limit=250,
            notes={"source": "repair_test"},
        )

        self.assertEqual(len(result.scorecards), 1)
        self.assertEqual(result.scorecards[0].genome.strategy_id, candidate.strategy_id)
        self.assertEqual(result.scorecards[0].genome.params, candidate.params)
        self.assertEqual(result.scorecards[0].genome.mutation_of, candidate.mutation_of)
        self.assertIn(candidate.strategy_id, result.metrics_by_strategy)
        self.assertEqual(result.notes["source"], "repair_test")

    def test_run_for_symbol_merges_extra_genomes_into_default_population(self):
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
        candidate = StrategyGenome(
            "microtrend_breakout@BTC_USDT_USDT_5m:repair",
            "microtrend_breakout",
            {
                "lookback": 18.0,
                "breakout_buffer": 0.0022,
                "hold_bars": 5.0,
            },
            mutation_of="microtrend_breakout:seed",
            tags=("repair",),
        )

        result = lab.run_for_symbol(
            symbol="BTC/USDT",
            timeframe="5m",
            total_capital=5000.0,
            candle_limit=250,
            extra_genomes=[candidate],
        )

        strategy_ids = {card.genome.strategy_id for card in result.scorecards}
        self.assertIn(candidate.strategy_id, strategy_ids)
        self.assertIn(candidate.strategy_id, result.metrics_by_strategy)

    def test_scheduler_builds_default_jobs_from_feed(self):
        feed = SQLiteOHLCVFeed(str(self.db_path))
        lab = ExperimentLab(feed)
        scheduler = ExperimentScheduler(lab)

        jobs = scheduler.default_jobs(timeframe="5m", min_rows=200, max_symbols=2)
        results = scheduler.run(jobs)

        self.assertEqual(len(jobs), 2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.promoted is not None for result in results))


if __name__ == "__main__":
    unittest.main()
