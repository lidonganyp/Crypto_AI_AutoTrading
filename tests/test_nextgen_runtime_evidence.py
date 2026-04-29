import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.models import SignalDirection
from core.storage import Storage
from execution.paper_trader import PaperTrader
from nextgen_evolution import (
    AutonomousDirector,
    EvolutionConfig,
    ExecutionAction,
    PromotionStage,
    RuntimeEvidenceCollector,
    RuntimeEvidenceSnapshot,
    SQLiteOHLCVFeed,
)
from nextgen_evolution.experiment_lab import ExperimentResult
from nextgen_evolution.models import ScoreCard, StrategyGenome, ValidationMetrics


def make_intraday_candles(count: int, base: float, final_close: float | None = None):
    candles = []
    for idx in range(count):
        close = base + idx * 0.1 + ((idx % 6) - 3) * 0.12
        candles.append(
            {
                "timestamp": 1700000000000 + idx * 300000,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1200 + idx,
            }
        )
    if final_close is not None and candles:
        candles[-1]["close"] = final_close
        candles[-1]["high"] = max(candles[-1]["high"], final_close)
        candles[-1]["low"] = min(candles[-1]["low"], final_close)
    return candles


class NextGenRuntimeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "cryptoai.db"
        self.storage = Storage(str(self.db_path))
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "5m",
            make_intraday_candles(240, 100.0, final_close=95.0),
        )
        self.feed = SQLiteOHLCVFeed(str(self.db_path))
        self.paper = PaperTrader(self.storage)
        self.collector = RuntimeEvidenceCollector(self.feed, EvolutionConfig())

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_runtime_evidence_tracks_open_position_unrealized_pnl(self):
        runtime_id = "BTC/USDT:USDT|5m|trend_pullback_continuation:seed"
        self.paper.execute_open(
            "BTC/USDT:USDT",
            SignalDirection.LONG,
            100.0,
            0.7,
            "autonomy open",
            position_value=1000.0,
            metadata={
                "source": "nextgen_autonomy",
                "runtime_id": runtime_id,
                "strategy_id": "trend_pullback_continuation:seed",
                "family": "trend_pullback_continuation",
                "timeframe": "5m",
                "lifecycle_state": "paper",
                "execution_action": "promote_to_paper",
            },
        )
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "5m",
            [
                {
                    "timestamp": 1700000000000 + 241 * 300000,
                    "open": 99.0,
                    "high": 99.2,
                    "low": 98.8,
                    "close": 99.0,
                    "volume": 1600.0,
                }
            ],
        )

        evidence = self.collector.collect([runtime_id])
        snapshot = evidence[runtime_id]

        self.assertIsInstance(snapshot, RuntimeEvidenceSnapshot)
        self.assertTrue(snapshot.open_position)
        self.assertLess(snapshot.unrealized_pnl, 0.0)
        self.assertGreater(snapshot.current_capital, 0.0)
        self.assertEqual(snapshot.health_status, "active")
        self.assertGreater(float(snapshot.notes.get("holding_minutes") or 0.0), 0.0)
        self.assertEqual(float(snapshot.notes.get("timeframe_minutes") or 0.0), 5.0)
        self.assertGreater(float(snapshot.notes.get("holding_bars") or 0.0), 0.0)
        self.assertGreaterEqual(float(snapshot.notes.get("peak_unrealized_pnl") or 0.0), 0.0)

    def test_director_uses_runtime_evidence_to_exit_failing_live_runtime(self):
        runtime_id = "BTC/USDT:USDT|5m|volatility_reclaim:seed"
        for price_in, price_out in [(100.0, 97.0), (98.0, 95.0), (96.0, 93.0)]:
            self.paper.execute_open(
                "BTC/USDT:USDT",
                SignalDirection.LONG,
                price_in,
                0.6,
                "loss trade",
                position_value=1000.0,
                metadata={
                    "source": "nextgen_autonomy",
                    "runtime_id": runtime_id,
                    "strategy_id": "volatility_reclaim:seed",
                    "family": "volatility_reclaim",
                    "timeframe": "5m",
                    "lifecycle_state": "live",
                    "execution_action": "keep",
                },
            )
            self.paper.execute_close(
                "BTC/USDT:USDT",
                price_out,
                reason="loss close",
            )

        result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                ScoreCard(
                    genome=StrategyGenome(
                        "volatility_reclaim:seed",
                        "volatility_reclaim",
                        {"lookback": 20.0, "hold_bars": 8.0},
                    ),
                    stage=PromotionStage.LIVE,
                    edge_score=0.55,
                    robustness_score=0.75,
                    deployment_score=0.60,
                    total_score=0.56,
                    reasons=["promote_live"],
                )
            ],
            promoted=[],
            allocations=[],
            candle_count=200,
            metrics_by_strategy={
                "volatility_reclaim:seed": ValidationMetrics(
                    backtest_expectancy=0.45,
                    walkforward_expectancy=0.32,
                    shadow_expectancy=0.18,
                    live_expectancy=0.10,
                    max_drawdown_pct=4.0,
                    trade_count=30,
                    cost_drag_pct=0.12,
                    latency_ms=40.0,
                    regime_consistency=0.78,
                )
            },
        )
        director = AutonomousDirector(
            EvolutionConfig(),
            evidence_collector=self.collector,
        )

        directive = director.plan_from_experiments([result])

        self.assertEqual(len(directive.execution), 1)
        self.assertEqual(directive.execution[0].strategy_id, runtime_id)
        self.assertEqual(directive.execution[0].action, ExecutionAction.EXIT)
        self.assertEqual(len(directive.repairs), 1)

    def test_director_exits_overstayed_losing_runtime(self):
        runtime_id = "BTC/USDT:USDT|5m|microtrend_breakout:seed"
        opened = self.paper.execute_open(
            "BTC/USDT:USDT",
            SignalDirection.LONG,
            100.0,
            0.7,
            "overstay runtime",
            position_value=1000.0,
            metadata={
                "source": "nextgen_autonomy",
                "runtime_id": runtime_id,
                "strategy_id": "microtrend_breakout:seed",
                "family": "microtrend_breakout",
                "timeframe": "5m",
                "lifecycle_state": "paper",
                "execution_action": "keep",
            },
        )
        self.assertIsNotNone(opened)
        stale_entry = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
        with self.storage._conn() as conn:
            conn.execute(
                "UPDATE trades SET entry_time = ? WHERE id = ?",
                (stale_entry, opened["trade_id"]),
            )
            conn.execute(
                "UPDATE positions SET entry_time = ? WHERE symbol = ?",
                (stale_entry, "BTC/USDT:USDT"),
            )

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
                    edge_score=0.42,
                    robustness_score=0.74,
                    deployment_score=0.44,
                    total_score=0.43,
                    reasons=["promote_paper"],
                )
            ],
            promoted=[],
            allocations=[],
            candle_count=200,
            metrics_by_strategy={
                "microtrend_breakout:seed": ValidationMetrics(
                    backtest_expectancy=0.32,
                    walkforward_expectancy=0.18,
                    shadow_expectancy=0.12,
                    live_expectancy=0.05,
                    max_drawdown_pct=4.0,
                    trade_count=32,
                    cost_drag_pct=0.12,
                    latency_ms=35.0,
                    regime_consistency=0.78,
                )
            },
        )
        director = AutonomousDirector(
            EvolutionConfig(
                autonomy_repair_expectancy_floor=0.0,
                autonomy_overstay_soft_multiplier=1.25,
                autonomy_overstay_hard_multiplier=2.0,
            ),
            evidence_collector=self.collector,
        )

        directive = director.plan_from_experiments([result])

        self.assertEqual(len(directive.execution), 1)
        self.assertEqual(directive.execution[0].strategy_id, runtime_id)
        self.assertEqual(directive.execution[0].action, ExecutionAction.PAUSE_NEW)
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.SHADOW)
        self.assertEqual(len(directive.repairs), 1)

    def test_director_scales_down_profitable_runtime_after_deep_profit_retrace(self):
        runtime_id = "BTC/USDT:USDT|5m|trend_pullback_continuation:seed"
        opened = self.paper.execute_open(
            "BTC/USDT:USDT",
            SignalDirection.LONG,
            100.0,
            0.7,
            "profit lock runtime",
            position_value=1000.0,
            metadata={
                "source": "nextgen_autonomy",
                "runtime_id": runtime_id,
                "strategy_id": "trend_pullback_continuation:seed",
                "family": "trend_pullback_continuation",
                "timeframe": "5m",
                "lifecycle_state": "live",
                "execution_action": "keep",
            },
        )
        self.assertIsNotNone(opened)
        aligned_entry = datetime.fromtimestamp(
            (1700000000000 + 499 * 300000) / 1000.0,
            tz=timezone.utc,
        ).isoformat()
        with self.storage._conn() as conn:
            conn.execute(
                "UPDATE trades SET entry_time = ? WHERE id = ?",
                (aligned_entry, opened["trade_id"]),
            )
            conn.execute(
                "UPDATE positions SET entry_time = ? WHERE symbol = ?",
                (aligned_entry, "BTC/USDT:USDT"),
            )
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "5m",
            [
                {
                    "timestamp": 1700000000000 + 500 * 300000,
                    "open": 108.5,
                    "high": 110.0,
                    "low": 108.0,
                    "close": 109.0,
                    "volume": 1600.0,
                },
                {
                    "timestamp": 1700000000000 + 501 * 300000,
                    "open": 106.0,
                    "high": 106.5,
                    "low": 105.5,
                    "close": 106.0,
                    "volume": 1600.0,
                },
            ],
        )

        result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                ScoreCard(
                    genome=StrategyGenome(
                        "trend_pullback_continuation:seed",
                        "trend_pullback_continuation",
                        {"lookback": 18.0, "hold_bars": 6.0},
                    ),
                    stage=PromotionStage.LIVE,
                    edge_score=0.58,
                    robustness_score=0.78,
                    deployment_score=0.61,
                    total_score=0.57,
                    reasons=["promote_live"],
                )
            ],
            promoted=[],
            allocations=[],
            candle_count=200,
            metrics_by_strategy={
                "trend_pullback_continuation:seed": ValidationMetrics(
                    backtest_expectancy=0.42,
                    walkforward_expectancy=0.28,
                    shadow_expectancy=0.18,
                    live_expectancy=0.09,
                    max_drawdown_pct=4.0,
                    trade_count=36,
                    cost_drag_pct=0.10,
                    latency_ms=35.0,
                    regime_consistency=0.81,
                )
            },
        )
        director = AutonomousDirector(
            EvolutionConfig(
                autonomy_repair_expectancy_floor=0.0,
                autonomy_profit_lock_min_return_pct=5.0,
                autonomy_profit_lock_soft_retrace_pct=35.0,
                autonomy_profit_lock_hard_retrace_pct=80.0,
            ),
            evidence_collector=self.collector,
        )

        directive = director.plan_from_experiments([result])

        self.assertEqual(len(directive.execution), 1)
        self.assertEqual(directive.execution[0].strategy_id, runtime_id)
        self.assertEqual(directive.execution[0].action, ExecutionAction.SCALE_DOWN)
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.LIVE)
        self.assertIn("profit_lock_harvest", directive.execution[0].reasons)
        self.assertEqual(directive.repairs, [])

    def test_director_exits_and_repairs_runtime_after_severe_profit_retrace(self):
        runtime_id = "BTC/USDT:USDT|5m|trend_exhaustion:seed"
        opened = self.paper.execute_open(
            "BTC/USDT:USDT",
            SignalDirection.LONG,
            100.0,
            0.7,
            "profit lock runtime",
            position_value=1000.0,
            metadata={
                "source": "nextgen_autonomy",
                "runtime_id": runtime_id,
                "strategy_id": "trend_exhaustion:seed",
                "family": "trend_exhaustion",
                "timeframe": "5m",
                "lifecycle_state": "live",
                "execution_action": "keep",
            },
        )
        self.assertIsNotNone(opened)
        aligned_entry = datetime.fromtimestamp(
            (1700000000000 + 599 * 300000) / 1000.0,
            tz=timezone.utc,
        ).isoformat()
        with self.storage._conn() as conn:
            conn.execute(
                "UPDATE trades SET entry_time = ? WHERE id = ?",
                (aligned_entry, opened["trade_id"]),
            )
            conn.execute(
                "UPDATE positions SET entry_time = ? WHERE symbol = ?",
                (aligned_entry, "BTC/USDT:USDT"),
            )
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "5m",
            [
                {
                    "timestamp": 1700000000000 + 600 * 300000,
                    "open": 109.0,
                    "high": 110.0,
                    "low": 108.5,
                    "close": 109.0,
                    "volume": 1600.0,
                },
                {
                    "timestamp": 1700000000000 + 601 * 300000,
                    "open": 102.2,
                    "high": 102.5,
                    "low": 101.8,
                    "close": 102.0,
                    "volume": 1600.0,
                },
            ],
        )

        result = ExperimentResult(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                ScoreCard(
                    genome=StrategyGenome(
                        "trend_exhaustion:seed",
                        "trend_exhaustion",
                        {"lookback": 18.0, "hold_bars": 6.0},
                    ),
                    stage=PromotionStage.LIVE,
                    edge_score=0.57,
                    robustness_score=0.74,
                    deployment_score=0.58,
                    total_score=0.56,
                    reasons=["promote_live"],
                )
            ],
            promoted=[],
            allocations=[],
            candle_count=200,
            metrics_by_strategy={
                "trend_exhaustion:seed": ValidationMetrics(
                    backtest_expectancy=0.40,
                    walkforward_expectancy=0.24,
                    shadow_expectancy=0.17,
                    live_expectancy=0.08,
                    max_drawdown_pct=4.0,
                    trade_count=34,
                    cost_drag_pct=0.10,
                    latency_ms=35.0,
                    regime_consistency=0.81,
                )
            },
        )
        director = AutonomousDirector(
            EvolutionConfig(
                autonomy_repair_expectancy_floor=0.0,
                autonomy_profit_lock_min_return_pct=5.0,
                autonomy_profit_lock_soft_retrace_pct=35.0,
                autonomy_profit_lock_hard_retrace_pct=75.0,
            ),
            evidence_collector=self.collector,
        )

        directive = director.plan_from_experiments([result])

        self.assertEqual(len(directive.execution), 1)
        self.assertEqual(directive.execution[0].strategy_id, runtime_id)
        self.assertEqual(directive.execution[0].action, ExecutionAction.EXIT)
        self.assertEqual(directive.execution[0].target_stage, PromotionStage.SHADOW)
        self.assertIn("profit_lock_exit", directive.execution[0].reasons)
        self.assertEqual(len(directive.repairs), 1)


if __name__ == "__main__":
    unittest.main()
