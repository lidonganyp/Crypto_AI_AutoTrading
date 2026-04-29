import json
import tempfile
import unittest
from pathlib import Path

from config import Settings
from core.models import SignalDirection
from core.storage import Storage
from execution.paper_trader import PaperTrader
from nextgen_evolution import (
    AutonomyDirective,
    AutonomyLiveCycleRunner,
    AutonomyLiveRuntime,
    ExecutionAction,
    PromotionStage,
    PromotionRegistry,
    RepairActionType,
    RepairPlan,
    RuntimeLifecycleState,
    StrategyGenome,
)


class NextGenLiveCycleRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "cryptoai.db"
        self.storage = Storage(str(self.db_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_runner_skips_when_live_not_requested_and_no_emergency(self):
        runner = AutonomyLiveCycleRunner(self.storage, settings=Settings())

        result = runner.run(requested_live=False, trigger="scheduler")

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "live_not_requested")
        self.assertEqual(result["intent_count"], 0)
        self.assertEqual(result["repair_execution_count"], 0)
        self.assertEqual(result["repair_queue_requested_size"], 0)
        self.assertEqual(result["repair_queue_dropped_count"], 0)
        self.assertEqual(result["repair_queue_dropped_runtime_ids"], [])
        self.assertEqual(result["repair_queue_hold_priority_count"], 0)
        self.assertFalse(result["repair_queue_hold_priority_active"])
        self.assertFalse(result["repair_queue_dropped_active"])
        self.assertEqual(result["repair_queue_postponed_rebuild_count"], 0)
        self.assertFalse(result["repair_queue_postponed_rebuild_active"])
        self.assertIsNone(PromotionRegistry(str(self.db_path)).latest_autonomy_cycle())

    def test_runner_uses_persisted_requested_live_state(self):
        settings = Settings()
        AutonomyLiveRuntime(self.storage, settings=settings).set_operator_request(
            requested_live=True,
            reason="cli",
        )
        runner = AutonomyLiveCycleRunner(self.storage, settings=settings)

        result = runner.run(requested_live=None, trigger="scheduler")

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "no_nextgen_jobs")
        self.assertTrue(result["requested_live"])
        self.assertTrue(result["operator_request"]["requested_live"])
        self.assertEqual(result["repair_execution_count"], 0)

    def test_runner_force_flattens_live_managed_position_without_jobs(self):
        settings = Settings()
        self.storage.set_state("manual_recovery_required", "true")
        self.storage.set_state("manual_recovery_approved", "false")
        PaperTrader(self.storage).execute_open(
            "BTC/USDT:USDT",
            SignalDirection.LONG,
            120.0,
            0.6,
            "live managed runtime",
            position_value=1000.0,
            metadata={
                "source": "nextgen_autonomy",
                "bridge_mode": "live",
                "runtime_id": "BTC/USDT:USDT|5m|flatten_runtime:seed",
                "strategy_id": "flatten_runtime:seed",
                "family": "flatten_runtime",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.LIVE.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        runner = AutonomyLiveCycleRunner(self.storage, settings=settings)

        result = runner.run(
            requested_live=False,
            trigger="manual_recovery_required",
            trigger_reason="market_data_latency",
            trigger_details="latency_seconds=6.000",
        )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["force_flatten"])
        self.assertEqual(result["intent_count"], 1)
        self.assertEqual(result["action_counts"], {"close": 1})
        self.assertEqual(result["intent_status_counts"], {"dry_run": 1})
        self.assertIsNotNone(result["autonomy_cycle_id"])

        with self.storage._conn() as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM execution_events
                WHERE event_type = 'nextgen_autonomy_live_intent'
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["action"], "close")
        self.assertIn("operator_forced_flatten", payload["reasons"])

    def test_runner_prioritizes_hold_repairs_and_postpones_release_rebuilds(self):
        runner = AutonomyLiveCycleRunner(self.storage, settings=Settings())
        directive = AutonomyDirective(
            repairs=[
                RepairPlan(
                    strategy_id="BTC/USDT:USDT|5m|hold_runtime:seed",
                    action=RepairActionType.TIGHTEN_RISK,
                    priority=3,
                    candidate_genome=StrategyGenome(
                        "hold_runtime:repair",
                        "hold_runtime",
                        {"lookback": 18.0, "hold_bars": 6.0},
                    ),
                    validation_stage=PromotionStage.SHADOW,
                    reasons=["repair_runtime_override_recovery_mode:hold"],
                ),
                RepairPlan(
                    strategy_id="BTC/USDT:USDT|5m|accelerate_runtime:seed",
                    action=RepairActionType.RAISE_SELECTIVITY,
                    priority=4,
                    candidate_genome=StrategyGenome(
                        "accelerate_runtime:repair",
                        "accelerate_runtime",
                        {"lookback": 18.0, "hold_bars": 6.0},
                    ),
                    validation_stage=PromotionStage.SHADOW,
                    reasons=["repair_runtime_override_recovery_mode:accelerate"],
                ),
                RepairPlan(
                    strategy_id="BTC/USDT:USDT|5m|release_runtime:seed",
                    action=RepairActionType.REBUILD_LINEAGE,
                    priority=6,
                    candidate_genome=StrategyGenome(
                        "release_runtime:rebuild",
                        "release_runtime",
                        {"lookback": 20.0, "hold_bars": 8.0},
                    ),
                    validation_stage=PromotionStage.SHADOW,
                    reasons=["repair_runtime_override_recovery_mode:release"],
                ),
            ]
        )

        prioritized = runner._prioritize_repairs(directive)

        self.assertEqual(
            [item.action for item in prioritized.repairs],
            [
                RepairActionType.TIGHTEN_RISK,
                RepairActionType.RAISE_SELECTIVITY,
                RepairActionType.REBUILD_LINEAGE,
            ],
        )
        self.assertEqual(
            [item.priority for item in prioritized.repairs],
            [5, 3, 1],
        )
        self.assertEqual(
            prioritized.notes["repair_queue_actions"],
            ["tighten_risk", "raise_selectivity", "rebuild_lineage"],
        )
        self.assertEqual(prioritized.notes["repair_queue_requested_size"], 0)
        self.assertEqual(prioritized.notes["repair_queue_dropped_count"], 0)
        self.assertEqual(prioritized.notes["repair_queue_hold_priority_count"], 1)
        self.assertEqual(prioritized.notes["repair_queue_postponed_rebuild_count"], 1)
        self.assertEqual(prioritized.notes["repair_queue_reprioritized_count"], 3)
        summary = runner._repair_queue_summary(prioritized)
        self.assertEqual(summary["repair_queue_requested_size"], 0)
        self.assertEqual(summary["repair_queue_dropped_count"], 0)
        self.assertEqual(summary["repair_queue_dropped_runtime_ids"], [])
        self.assertEqual(summary["repair_queue_hold_priority_count"], 1)
        self.assertTrue(summary["repair_queue_hold_priority_active"])
        self.assertFalse(summary["repair_queue_dropped_active"])
        self.assertEqual(summary["repair_queue_postponed_rebuild_count"], 1)
        self.assertTrue(summary["repair_queue_postponed_rebuild_active"])
        self.assertEqual(summary["repair_queue_reprioritized_count"], 3)
        self.assertTrue(summary["repair_queue_reprioritized_active"])


if __name__ == "__main__":
    unittest.main()
