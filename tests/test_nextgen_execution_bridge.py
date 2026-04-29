import tempfile
import unittest
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.models import SignalDirection
from core.storage import Storage
from execution.paper_trader import PaperTrader
from nextgen_evolution import (
    AutonomyDirective,
    AutonomyPaperBridge,
    EvolutionConfig,
    ExecutionAction,
    ExecutionDirective,
    PortfolioAllocator,
    PromotionRegistry,
    RuntimeEvidenceCollector,
    RolloutStateMachine,
    SQLiteOHLCVFeed,
)
from nextgen_evolution.experiment_lab import ExperimentResult
from nextgen_evolution.runtime_override_policy import (
    build_repair_reentry_notes,
    compose_runtime_policy_notes,
    latest_managed_close_time_index,
    lifecycle_policy_reentry_state,
    lifecycle_policy_repair_reentry_notes,
    lifecycle_policy_runtime_override_state,
    lifecycle_policy_runtime_overrides,
    lifecycle_policy_staged_exit_state,
)
from nextgen_evolution.models import (
    PortfolioAllocation,
    PromotionStage,
    RuntimeEvidenceSnapshot,
    RuntimeLifecycleState,
    RuntimeState,
    ScoreCard,
    StrategyGenome,
)


def make_intraday_candles(count: int, base: float):
    candles = []
    for idx in range(count):
        price = base + idx * 0.1 + ((idx % 8) - 4) * 0.15
        candles.append(
            {
                "timestamp": 1700000000000 + idx * 300000,
                "open": price,
                "high": price + 0.4,
                "low": price - 0.4,
                "close": price + 0.12,
                "volume": 1000 + idx,
            }
        )
    return candles


class NextGenExecutionBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "cryptoai.db"
        self.storage = Storage(str(self.db_path))
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "5m",
            make_intraday_candles(240, 100.0),
        )
        self.feed = SQLiteOHLCVFeed(str(self.db_path))
        self.registry = PromotionRegistry(str(self.db_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_rollout_state_machine_uses_limited_live_before_live(self):
        machine = RolloutStateMachine(
            EvolutionConfig(
                autonomy_limited_live_cycles=2,
                autonomy_limited_live_max_weight=0.05,
                autonomy_live_blast_radius_capital_pct=0.10,
            )
        )
        directive = ExecutionDirective(
            strategy_id="BTC/USDT:USDT|5m|volatility_reclaim:seed",
            action=ExecutionAction.PROMOTE_TO_LIVE,
            from_stage=PromotionStage.PAPER,
            target_stage=PromotionStage.LIVE,
        )

        first_state, first_cycles = machine.resolve(None, directive)
        second_state, second_cycles = machine.resolve(
            RuntimeState(
                runtime_id="BTC/USDT:USDT|5m|volatility_reclaim:seed",
                symbol="BTC/USDT:USDT",
                timeframe="5m",
                strategy_id="volatility_reclaim:seed",
                family="volatility_reclaim",
                lifecycle_state=RuntimeLifecycleState.LIMITED_LIVE,
                promotion_stage=PromotionStage.LIVE,
                target_stage=PromotionStage.LIVE,
                last_directive_action=ExecutionAction.PROMOTE_TO_LIVE,
                limited_live_cycles=1,
            ),
            directive,
        )
        bounded_capital = machine.bounded_capital(
            lifecycle_state=RuntimeLifecycleState.LIMITED_LIVE,
            allocated_capital=1800.0,
            total_capital=10000.0,
            capital_multiplier=1.0,
        )

        self.assertEqual(first_state, RuntimeLifecycleState.LIMITED_LIVE)
        self.assertEqual(first_cycles, 1)
        self.assertEqual(second_state, RuntimeLifecycleState.LIVE)
        self.assertEqual(second_cycles, 2)
        self.assertEqual(bounded_capital, 500.0)

    def test_paper_bridge_opens_primary_runtime_and_persists_state(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            config=EvolutionConfig(),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation:seed",
        )
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation:seed",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.42,
                )
            ],
        )
        directive = AutonomyDirective(
            execution=[
                ExecutionDirective(
                    strategy_id=runtime_id,
                    action=ExecutionAction.PROMOTE_TO_PAPER,
                    from_stage=PromotionStage.SHADOW,
                    target_stage=PromotionStage.PAPER,
                    reasons=["shadow_requirements_met"],
                )
            ]
        )
        allocations = [
            PortfolioAllocation(
                symbol="BTC/USDT:USDT",
                strategy_id="trend_pullback_continuation:seed",
                family="trend_pullback_continuation",
                stage=PromotionStage.PAPER,
                allocated_capital=1800.0,
                weight=0.18,
                score=0.42,
                timeframe="5m",
            )
        ]

        runtime_states, intents = bridge.apply(
            results=[result],
            directive=directive,
            portfolio_allocations=allocations,
            total_capital=10000.0,
            autonomy_cycle_id=None,
        )

        self.assertEqual(len(runtime_states), 1)
        self.assertEqual(runtime_states[0].lifecycle_state, RuntimeLifecycleState.PAPER)
        self.assertEqual(intents[0].action.value, "open")
        self.assertEqual(intents[0].status, "executed")
        open_trades = self.storage.get_open_trades()
        self.assertEqual(len(open_trades), 1)
        self.assertIn(runtime_id, open_trades[0]["metadata_json"])
        latest_states = self.registry.latest_runtime_states(limit=10)
        self.assertEqual(len(latest_states), 1)
        latest_intents = self.registry.latest_execution_intents(limit=10)
        self.assertEqual(len(latest_intents), 1)
        self.assertEqual(latest_intents[0]["status"], "executed")

    def test_paper_bridge_rotates_managed_runtime_on_same_symbol(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(),
        )
        old_runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "volatility_reclaim:seed",
        )
        bridge.paper_trader.execute_open(
            "BTC/USDT:USDT",
            direction=SignalDirection.LONG,
            price=123.0,
            confidence=0.6,
            rationale="old runtime",
            position_value=1200.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "runtime_id": old_runtime_id,
                "strategy_id": "volatility_reclaim:seed",
                "family": "volatility_reclaim",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.PAPER.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        new_runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation:seed",
        )
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation:seed",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.44,
                )
            ],
        )
        directive = AutonomyDirective(
            execution=[
                ExecutionDirective(
                    strategy_id=new_runtime_id,
                    action=ExecutionAction.PROMOTE_TO_PAPER,
                    from_stage=PromotionStage.SHADOW,
                    target_stage=PromotionStage.PAPER,
                )
            ]
        )
        allocations = [
            PortfolioAllocation(
                symbol="BTC/USDT:USDT",
                strategy_id="trend_pullback_continuation:seed",
                family="trend_pullback_continuation",
                stage=PromotionStage.PAPER,
                allocated_capital=1500.0,
                weight=0.15,
                score=0.44,
                timeframe="5m",
            )
        ]

        _, intents = bridge.apply(
            results=[result],
            directive=directive,
            portfolio_allocations=allocations,
            total_capital=10000.0,
        )

        self.assertEqual([item.action.value for item in intents[:2]], ["close", "open"])
        open_trades = self.storage.get_open_trades()
        self.assertEqual(len(open_trades), 1)
        self.assertIn(new_runtime_id, open_trades[0]["metadata_json"])
        closed_trades = self.storage.get_closed_trades()
        self.assertEqual(len(closed_trades), 1)

    def test_paper_bridge_adds_to_existing_managed_runtime_when_target_capital_increases(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation:seed",
        )
        seed_open = bridge.paper_trader.execute_open(
            "BTC/USDT:USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.6,
            rationale="seed runtime",
            quantity=5.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": bridge.BRIDGE_MODE,
                "runtime_id": runtime_id,
                "strategy_id": "trend_pullback_continuation:seed",
                "family": "trend_pullback_continuation",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.PAPER.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        self.assertIsNotNone(seed_open)
        initial_qty = float(self.storage.get_positions()[0]["quantity"])

        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation:seed",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.42,
                )
            ],
        )
        directive = AutonomyDirective(
            execution=[
                ExecutionDirective(
                    strategy_id=runtime_id,
                    action=ExecutionAction.KEEP,
                    from_stage=PromotionStage.PAPER,
                    target_stage=PromotionStage.PAPER,
                )
            ]
        )
        allocations = [
            PortfolioAllocation(
                symbol="BTC/USDT:USDT",
                strategy_id="trend_pullback_continuation:seed",
                family="trend_pullback_continuation",
                stage=PromotionStage.PAPER,
                allocated_capital=1800.0,
                weight=0.18,
                score=0.42,
                timeframe="5m",
            )
        ]

        _, intents = bridge.apply(
            results=[result],
            directive=directive,
            portfolio_allocations=allocations,
            total_capital=10000.0,
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].action.value, "open")
        self.assertEqual(intents[0].status, "executed")
        self.assertIn("rebalance_up", intents[0].reasons)
        self.assertEqual(len(self.storage.get_open_trades()), 1)
        self.assertGreater(float(self.storage.get_positions()[0]["quantity"]), initial_qty)

    def test_paper_bridge_marks_profit_lock_harvest_reduce_explicitly(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation:seed",
        )
        seed_open = bridge.paper_trader.execute_open(
            "BTC/USDT:USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.6,
            rationale="seed runtime",
            quantity=10.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": bridge.BRIDGE_MODE,
                "runtime_id": runtime_id,
                "strategy_id": "trend_pullback_continuation:seed",
                "family": "trend_pullback_continuation",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.PAPER.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        self.assertIsNotNone(seed_open)

        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation:seed",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.42,
                )
            ],
        )
        directive = AutonomyDirective(
            execution=[
                ExecutionDirective(
                    strategy_id=runtime_id,
                    action=ExecutionAction.SCALE_DOWN,
                    from_stage=PromotionStage.PAPER,
                    target_stage=PromotionStage.PAPER,
                    reasons=["profit_lock_harvest"],
                )
            ]
        )
        allocations = [
            PortfolioAllocation(
                symbol="BTC/USDT:USDT",
                strategy_id="trend_pullback_continuation:seed",
                family="trend_pullback_continuation",
                stage=PromotionStage.PAPER,
                allocated_capital=500.0,
                weight=0.05,
                score=0.42,
                timeframe="5m",
            )
        ]

        _, intents = bridge.apply(
            results=[result],
            directive=directive,
            portfolio_allocations=allocations,
            total_capital=10000.0,
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].action.value, "reduce")
        self.assertEqual(intents[0].status, "executed")
        self.assertEqual(intents[0].reasons, ["profit_lock_harvest"])
        self.assertEqual(intents[0].notes.get("position_adjustment"), "profit_lock_harvest")
        self.assertEqual(
            intents[0].notes.get("autonomy_scale_down_reason"),
            "profit_lock_harvest",
        )
        latest_ledger = self.storage.get_pnl_ledger(limit=1)[0]
        latest_metadata = json.loads(latest_ledger["metadata_json"])
        self.assertEqual(
            latest_metadata.get("reason"),
            "nextgen_autonomy_profit_lock_reduce",
        )

    def test_paper_bridge_keeps_profit_lock_staged_exit_active_across_keep_cycles(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(
                autonomy_profit_lock_soft_scale_down_factor=0.75,
                autonomy_profit_lock_deep_scale_down_factor=0.40,
            ),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "profit_lock_runtime:seed",
        )
        opened = bridge.paper_trader.execute_open(
            "BTC/USDT:USDT",
            direction=SignalDirection.LONG,
            price=124.47,
            confidence=0.6,
            rationale="profit lock runtime",
            quantity=10.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": bridge.BRIDGE_MODE,
                "runtime_id": runtime_id,
                "strategy_id": "profit_lock_runtime:seed",
                "family": "profit_lock_runtime",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.LIVE.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        self.assertIsNotNone(opened)
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "profit_lock_runtime:seed",
                    "profit_lock_runtime",
                    PromotionStage.LIVE,
                    0.57,
                )
            ],
        )
        allocations = [
            PortfolioAllocation(
                symbol="BTC/USDT:USDT",
                strategy_id="profit_lock_runtime:seed",
                family="profit_lock_runtime",
                stage=PromotionStage.LIVE,
                allocated_capital=1000.0,
                weight=0.10,
                score=0.57,
                timeframe="5m",
            )
        ]

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.SCALE_DOWN,
                        from_stage=PromotionStage.LIVE,
                        target_stage=PromotionStage.LIVE,
                        capital_multiplier=0.75,
                        reasons=["profit_lock_harvest"],
                    )
                ]
            ),
            portfolio_allocations=allocations,
            total_capital=10000.0,
            previous_states=[
                RuntimeState(
                    runtime_id=runtime_id,
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="profit_lock_runtime:seed",
                    family="profit_lock_runtime",
                    lifecycle_state=RuntimeLifecycleState.LIVE,
                    promotion_stage=PromotionStage.LIVE,
                    target_stage=PromotionStage.LIVE,
                    last_directive_action=ExecutionAction.KEEP,
                    score=0.57,
                    allocated_capital=1000.0,
                    desired_capital=1000.0,
                    current_capital=1000.0,
                    current_weight=0.10,
                    limited_live_cycles=2,
                )
            ],
        )

        self.assertEqual(runtime_states[0].desired_capital, 750.0)
        self.assertEqual(
            lifecycle_policy_staged_exit_state(runtime_states[0].notes)["phase"],
            "harvest",
        )
        self.assertEqual(
            lifecycle_policy_staged_exit_state(runtime_states[0].notes)["target_multiplier"],
            0.75,
        )
        self.assertEqual(
            runtime_states[0].notes["runtime_lifecycle_policy"]["staged_exit"]["phase"],
            "harvest",
        )
        self.assertTrue(
            runtime_states[0].notes["runtime_lifecycle_policy"]["staged_exit"]["active"]
        )

        next_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.LIVE,
                        target_stage=PromotionStage.LIVE,
                        reasons=["live_strategy_stable"],
                    )
                ]
            ),
            portfolio_allocations=allocations,
            total_capital=10000.0,
            previous_states=runtime_states,
        )
        intents = bridge.build_execution_intents(next_states)

        self.assertEqual(next_states[0].desired_capital, 750.0)
        self.assertEqual(
            lifecycle_policy_staged_exit_state(next_states[0].notes)["phase"],
            "harvest",
        )
        self.assertEqual(intents[0].action.value, "reduce")
        self.assertEqual(intents[0].reasons, ["profit_lock_harvest"])
        self.assertEqual(intents[0].notes.get("staged_exit_phase"), "harvest")

    def test_paper_bridge_reads_profit_lock_staged_exit_from_unified_lifecycle_policy(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            config=EvolutionConfig(),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "profit_lock_runtime:seed",
        )
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "profit_lock_runtime:seed",
                    "profit_lock_runtime",
                    PromotionStage.LIVE,
                    0.57,
                )
            ],
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.LIVE,
                        target_stage=PromotionStage.LIVE,
                        reasons=["live_strategy_stable"],
                    )
                ]
            ),
            portfolio_allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="profit_lock_runtime:seed",
                    family="profit_lock_runtime",
                    stage=PromotionStage.LIVE,
                    allocated_capital=1000.0,
                    weight=0.10,
                    score=0.57,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            previous_states=[
                RuntimeState(
                    runtime_id=runtime_id,
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="profit_lock_runtime:seed",
                    family="profit_lock_runtime",
                    lifecycle_state=RuntimeLifecycleState.LIVE,
                    promotion_stage=PromotionStage.LIVE,
                    target_stage=PromotionStage.LIVE,
                    last_directive_action=ExecutionAction.KEEP,
                    score=0.57,
                    allocated_capital=1000.0,
                    desired_capital=750.0,
                    current_capital=750.0,
                    current_weight=0.075,
                    limited_live_cycles=2,
                    notes=compose_runtime_policy_notes(
                        staged_exit_state={
                            "mode": "profit_lock",
                            "phase": "harvest",
                            "target_multiplier": 0.75,
                            "last_reason": "profit_lock_harvest",
                        }
                    ),
                )
            ],
        )

        self.assertEqual(runtime_states[0].desired_capital, 750.0)
        self.assertEqual(lifecycle_policy_staged_exit_state(runtime_states[0].notes)["phase"], "harvest")
        self.assertEqual(
            runtime_states[0].notes["runtime_lifecycle_policy"]["staged_exit"]["phase"],
            "harvest",
        )

    def test_paper_bridge_gradually_reenters_after_profit_lock_staged_exit_recovery(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "profit_lock_runtime:seed",
        )
        opened = bridge.paper_trader.execute_open(
            "BTC/USDT:USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.6,
            rationale="profit lock runtime",
            quantity=5.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": bridge.BRIDGE_MODE,
                "runtime_id": runtime_id,
                "strategy_id": "profit_lock_runtime:seed",
                "family": "profit_lock_runtime",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.LIVE.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        self.assertIsNotNone(opened)
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "profit_lock_runtime:seed",
                    "profit_lock_runtime",
                    PromotionStage.LIVE,
                    0.57,
                )
            ],
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.LIVE,
                        target_stage=PromotionStage.LIVE,
                        reasons=["live_strategy_stable"],
                    )
                ]
            ),
            portfolio_allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="profit_lock_runtime:seed",
                    family="profit_lock_runtime",
                    stage=PromotionStage.LIVE,
                    allocated_capital=1000.0,
                    weight=0.10,
                    score=0.57,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            previous_states=[
                RuntimeState(
                    runtime_id=runtime_id,
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="profit_lock_runtime:seed",
                    family="profit_lock_runtime",
                    lifecycle_state=RuntimeLifecycleState.LIVE,
                    promotion_stage=PromotionStage.LIVE,
                    target_stage=PromotionStage.LIVE,
                    last_directive_action=ExecutionAction.KEEP,
                    score=0.57,
                    allocated_capital=1000.0,
                    desired_capital=400.0,
                    current_capital=400.0,
                    current_weight=0.04,
                    limited_live_cycles=2,
                    notes=compose_runtime_policy_notes(
                        staged_exit_state={
                            "mode": "profit_lock",
                            "phase": "deep_harvest",
                            "target_multiplier": 0.4,
                            "trigger_count": 2,
                            "recovery_count": 0,
                            "last_reason": "profit_lock_harvest",
                        }
                    ),
                )
            ],
        )

        self.assertEqual(runtime_states[0].desired_capital, 700.0)
        self.assertEqual(
            lifecycle_policy_staged_exit_state(runtime_states[0].notes)["phase"],
            "reentry",
        )
        self.assertEqual(
            lifecycle_policy_staged_exit_state(runtime_states[0].notes)["target_multiplier"],
            0.7,
        )
        self.assertEqual(
            lifecycle_policy_staged_exit_state(runtime_states[0].notes)["recovery_count"],
            1,
        )

    def test_paper_bridge_releases_profit_lock_staged_exit_after_confirmed_recovery(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "profit_lock_runtime:seed",
        )
        for entry_price, exit_price in ((100.0, 110.0), (101.0, 108.0)):
            opened = bridge.paper_trader.execute_open(
                "BTC/USDT:USDT",
                direction=SignalDirection.LONG,
                price=entry_price,
                confidence=0.6,
                rationale="profit lock recovery",
                quantity=1.0,
                metadata={
                    "source": bridge.MANAGED_SOURCE,
                    "bridge_mode": bridge.BRIDGE_MODE,
                    "runtime_id": runtime_id,
                    "strategy_id": "profit_lock_runtime:seed",
                    "family": "profit_lock_runtime",
                    "timeframe": "5m",
                    "lifecycle_state": RuntimeLifecycleState.LIVE.value,
                    "execution_action": ExecutionAction.KEEP.value,
                },
            )
            self.assertIsNotNone(opened)
            bridge.paper_trader.execute_close(
                "BTC/USDT:USDT",
                exit_price,
                reason="profit lock recovery",
            )
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "profit_lock_runtime:seed",
                    "profit_lock_runtime",
                    PromotionStage.LIVE,
                    0.57,
                )
            ],
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.LIVE,
                        target_stage=PromotionStage.LIVE,
                        reasons=["live_strategy_stable"],
                    )
                ]
            ),
            portfolio_allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="profit_lock_runtime:seed",
                    family="profit_lock_runtime",
                    stage=PromotionStage.LIVE,
                    allocated_capital=1000.0,
                    weight=0.10,
                    score=0.57,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            previous_states=[
                RuntimeState(
                    runtime_id=runtime_id,
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="profit_lock_runtime:seed",
                    family="profit_lock_runtime",
                    lifecycle_state=RuntimeLifecycleState.LIVE,
                    promotion_stage=PromotionStage.LIVE,
                    target_stage=PromotionStage.LIVE,
                    last_directive_action=ExecutionAction.KEEP,
                    score=0.57,
                    allocated_capital=1000.0,
                    desired_capital=700.0,
                    current_capital=700.0,
                    current_weight=0.07,
                    limited_live_cycles=2,
                    notes=compose_runtime_policy_notes(
                        staged_exit_state={
                            "mode": "profit_lock",
                            "phase": "reentry",
                            "target_multiplier": 0.7,
                            "trigger_count": 2,
                            "recovery_count": 1,
                            "last_reason": "profit_lock_reentry",
                        }
                    ),
                )
            ],
        )

        self.assertEqual(runtime_states[0].desired_capital, 1000.0)
        self.assertFalse(
            runtime_states[0].notes["runtime_lifecycle_policy"]["staged_exit"]["active"]
        )

    def test_paper_bridge_does_not_double_apply_profit_lock_staged_exit_cap_from_allocator(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            config=EvolutionConfig(),
        )
        allocator = PortfolioAllocator(EvolutionConfig())
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "profit_lock_runtime:seed",
        )
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "profit_lock_runtime:seed",
                    "profit_lock_runtime",
                    PromotionStage.LIVE,
                    0.57,
                )
            ],
        )
        allocations = allocator.allocate(
            [result],
            total_capital=10000.0,
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.SCALE_DOWN,
                        from_stage=PromotionStage.LIVE,
                        target_stage=PromotionStage.LIVE,
                        capital_multiplier=0.75,
                        reasons=["profit_lock_harvest"],
                    )
                ]
            ),
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.SCALE_DOWN,
                        from_stage=PromotionStage.LIVE,
                        target_stage=PromotionStage.LIVE,
                        capital_multiplier=0.75,
                        reasons=["profit_lock_harvest"],
                    )
                ]
            ),
            portfolio_allocations=allocations,
            total_capital=10000.0,
            previous_states=[
                RuntimeState(
                    runtime_id=runtime_id,
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="profit_lock_runtime:seed",
                    family="profit_lock_runtime",
                    lifecycle_state=RuntimeLifecycleState.LIVE,
                    promotion_stage=PromotionStage.LIVE,
                    target_stage=PromotionStage.LIVE,
                    last_directive_action=ExecutionAction.KEEP,
                    score=0.57,
                    allocated_capital=3500.0,
                    desired_capital=3500.0,
                    current_capital=3500.0,
                    current_weight=0.35,
                    limited_live_cycles=2,
                )
            ],
        )

        self.assertEqual(allocations[0].allocated_capital, 2625.0)
        self.assertIn(
            PortfolioAllocator.STAGED_EXIT_CAP_REASON,
            allocations[0].reasons,
        )
        self.assertEqual(runtime_states[0].allocated_capital, 2625.0)
        self.assertEqual(runtime_states[0].desired_capital, 1000.0)
        self.assertEqual(
            lifecycle_policy_staged_exit_state(runtime_states[0].notes)["phase"],
            "harvest",
        )
        self.assertIn(
            PortfolioAllocator.STAGED_EXIT_CAP_REASON,
            runtime_states[0].notes["allocation_reasons"],
        )

    def test_paper_bridge_skips_symbol_managed_elsewhere(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(),
        )
        bridge.paper_trader.execute_open(
            "BTC/USDT:USDT",
            direction=SignalDirection.LONG,
            price=123.0,
            confidence=0.5,
            rationale="external runtime",
            position_value=1000.0,
            metadata={"source": "manual"},
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation:seed",
        )
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation:seed",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.40,
                )
            ],
        )
        directive = AutonomyDirective(
            execution=[
                ExecutionDirective(
                    strategy_id=runtime_id,
                    action=ExecutionAction.PROMOTE_TO_PAPER,
                    from_stage=PromotionStage.SHADOW,
                    target_stage=PromotionStage.PAPER,
                )
            ]
        )
        allocations = [
            PortfolioAllocation(
                symbol="BTC/USDT:USDT",
                strategy_id="trend_pullback_continuation:seed",
                family="trend_pullback_continuation",
                stage=PromotionStage.PAPER,
                allocated_capital=1400.0,
                weight=0.14,
                score=0.40,
                timeframe="5m",
            )
        ]

        _, intents = bridge.apply(
            results=[result],
            directive=directive,
            portfolio_allocations=allocations,
            total_capital=10000.0,
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].action.value, "skip")
        self.assertEqual(intents[0].status, "skipped")
        open_trades = self.storage.get_open_trades()
        self.assertEqual(len(open_trades), 1)
        self.assertIn("manual", open_trades[0]["metadata_json"])

    def test_paper_bridge_applies_repair_runtime_overrides_on_reentry(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            config=EvolutionConfig(),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
        )
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.42,
                )
            ],
        )
        result.notes = compose_runtime_policy_notes(
            base_notes=result.notes,
            repair_reentry_notes=build_repair_reentry_notes(
                source_runtime_id="BTC/USDT:USDT|5m|trend_pullback_continuation:seed",
                runtime_overrides={
                    "max_weight_multiplier": 0.4,
                    "take_profit_bias": 1.1,
                },
            ),
        )
        directive = AutonomyDirective(
            execution=[
                ExecutionDirective(
                    strategy_id=runtime_id,
                    action=ExecutionAction.PROMOTE_TO_PAPER,
                    from_stage=PromotionStage.REJECT,
                    target_stage=PromotionStage.PAPER,
                    reasons=["repair_revalidation_passed"],
                )
            ]
        )
        allocations = [
            PortfolioAllocation(
                symbol="BTC/USDT:USDT",
                strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                family="trend_pullback_continuation",
                stage=PromotionStage.PAPER,
                allocated_capital=1000.0,
                weight=0.10,
                score=0.42,
                timeframe="5m",
            )
        ]

        runtime_states, intents = bridge.apply(
            results=[result],
            directive=directive,
            portfolio_allocations=allocations,
            total_capital=10000.0,
        )

        self.assertEqual(runtime_states[0].desired_capital, 400.0)
        self.assertEqual(
            lifecycle_policy_runtime_overrides(runtime_states[0].notes)["max_weight_multiplier"],
            0.4,
        )
        self.assertEqual(lifecycle_policy_runtime_overrides(intents[0].notes)["take_profit_bias"], 1.1)
        open_trades = self.storage.get_open_trades()
        metadata = json.loads(open_trades[0]["metadata_json"])
        self.assertIn("runtime_lifecycle_policy", metadata)
        self.assertNotIn("runtime_overrides", metadata)
        self.assertEqual(
            metadata["runtime_lifecycle_policy"]["runtime_override"]["values"]["max_weight_multiplier"],
            0.4,
        )

    def test_paper_bridge_applies_take_profit_bias_and_reentry_cooldown(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(),
        )
        source_runtime_id = "BTC/USDT:USDT|5m|trend_pullback_continuation:seed"
        closed = bridge.paper_trader.execute_open(
            "BTC/USDT:USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.6,
            rationale="seed runtime",
            quantity=5.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": bridge.BRIDGE_MODE,
                "runtime_id": source_runtime_id,
                "strategy_id": "trend_pullback_continuation:seed",
                "family": "trend_pullback_continuation",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.PAPER.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        self.assertIsNotNone(closed)
        bridge.paper_trader.execute_close(
            "BTC/USDT:USDT",
            101.0,
            reason="repair rotation",
        )

        repair_runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
        )
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.42,
                )
            ],
        )
        result.notes = compose_runtime_policy_notes(
            base_notes=result.notes,
            repair_reentry_notes=build_repair_reentry_notes(
                source_runtime_id=source_runtime_id,
                runtime_overrides={
                    "take_profit_bias": 1.2,
                    "entry_cooldown_bars_multiplier": 2.0,
                },
            ),
        )
        directive = AutonomyDirective(
            execution=[
                ExecutionDirective(
                    strategy_id=repair_runtime_id,
                    action=ExecutionAction.PROMOTE_TO_PAPER,
                    from_stage=PromotionStage.REJECT,
                    target_stage=PromotionStage.PAPER,
                    reasons=["repair_revalidation_passed"],
                )
            ]
        )
        allocations = [
            PortfolioAllocation(
                symbol="BTC/USDT:USDT",
                strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                family="trend_pullback_continuation",
                stage=PromotionStage.PAPER,
                allocated_capital=600.0,
                weight=0.06,
                score=0.42,
                timeframe="5m",
            )
        ]

        runtime_states, intents = bridge.apply(
            results=[result],
            directive=directive,
            portfolio_allocations=allocations,
            total_capital=10000.0,
        )

        self.assertEqual(lifecycle_policy_reentry_state(runtime_states[0].notes)["phase"], "cooldown")
        self.assertEqual(intents[0].action.value, "skip")
        self.assertIn("repair_reentry_cooldown_active", intents[0].reasons)

        stale_exit = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        with self.storage._conn() as conn:
            conn.execute(
                "UPDATE pnl_ledger SET event_time = ? WHERE event_type = 'close'",
                (stale_exit,),
            )

        runtime_states, intents = bridge.apply(
            results=[result],
            directive=directive,
            portfolio_allocations=allocations,
            total_capital=10000.0,
        )

        self.assertEqual(lifecycle_policy_reentry_state(runtime_states[0].notes)["phase"], "probation")
        self.assertEqual(lifecycle_policy_runtime_overrides(runtime_states[0].notes)["take_profit_bias"], 1.2)
        self.assertEqual(intents[0].action.value, "open")
        self.assertEqual(intents[0].status, "executed")
        position = self.storage.get_positions()[0]
        expected_take_profit = round(float(position["entry_price"]) * 1.12, 8)
        self.assertEqual(float(position["take_profit"]), expected_take_profit)

    def test_paper_bridge_carries_runtime_overrides_across_cycles(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            config=EvolutionConfig(
                autonomy_runtime_override_decay_rate=0.25,
                autonomy_runtime_override_cooldown_decay_step=0.5,
            ),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
        )
        first_result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.42,
                )
            ],
        )
        first_result.notes = compose_runtime_policy_notes(
            base_notes=first_result.notes,
            repair_reentry_notes=build_repair_reentry_notes(
                source_runtime_id="BTC/USDT:USDT|5m|trend_pullback_continuation:seed",
                runtime_overrides={
                    "max_weight_multiplier": 0.4,
                    "take_profit_bias": 1.1,
                },
            ),
        )
        directive = AutonomyDirective(
            execution=[
                ExecutionDirective(
                    strategy_id=runtime_id,
                    action=ExecutionAction.PROMOTE_TO_PAPER,
                    from_stage=PromotionStage.REJECT,
                    target_stage=PromotionStage.PAPER,
                    reasons=["repair_revalidation_passed"],
                )
            ]
        )
        allocations = [
            PortfolioAllocation(
                symbol="BTC/USDT:USDT",
                strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                family="trend_pullback_continuation",
                stage=PromotionStage.PAPER,
                allocated_capital=1000.0,
                weight=0.10,
                score=0.42,
                timeframe="5m",
            )
        ]

        runtime_states = bridge.build_runtime_states(
            results=[first_result],
            directive=directive,
            portfolio_allocations=allocations,
            total_capital=10000.0,
        )

        second_result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.43,
                )
            ],
        )
        runtime_states = bridge.build_runtime_states(
            results=[second_result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.PAPER,
                        reasons=["paper_validation_continues"],
                    )
                ]
            ),
            portfolio_allocations=allocations,
            total_capital=10000.0,
            previous_states=runtime_states,
        )

        self.assertEqual(runtime_states[0].desired_capital, 550.0)
        self.assertEqual(lifecycle_policy_runtime_overrides(runtime_states[0].notes)["max_weight_multiplier"], 0.55)
        self.assertEqual(lifecycle_policy_runtime_overrides(runtime_states[0].notes)["take_profit_bias"], 1.075)
        self.assertEqual(
            lifecycle_policy_repair_reentry_notes(runtime_states[0].notes)["source_runtime_id"],
            "BTC/USDT:USDT|5m|trend_pullback_continuation:seed",
        )
        self.assertEqual(
            lifecycle_policy_runtime_override_state(runtime_states[0].notes)["recovery_mode"],
            "neutral",
        )
        self.assertEqual(lifecycle_policy_runtime_override_state(runtime_states[0].notes)["decay_steps_applied"], 1)
        self.assertEqual(lifecycle_policy_reentry_state(runtime_states[0].notes)["phase"], "probation")

    def test_paper_bridge_decays_runtime_overrides_without_fresh_reentry(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            config=EvolutionConfig(
                autonomy_runtime_override_decay_rate=0.25,
                autonomy_runtime_override_cooldown_decay_step=0.5,
            ),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
        )
        previous_state = RuntimeState(
            runtime_id=runtime_id,
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
            family="trend_pullback_continuation",
            lifecycle_state=RuntimeLifecycleState.PAPER,
            promotion_stage=PromotionStage.PAPER,
            target_stage=PromotionStage.PAPER,
            last_directive_action=ExecutionAction.KEEP,
            score=0.42,
            allocated_capital=1000.0,
            desired_capital=400.0,
            current_capital=0.0,
            current_weight=0.10,
            capital_multiplier=1.0,
            limited_live_cycles=0,
            notes=compose_runtime_policy_notes(
                runtime_overrides={
                    "max_weight_multiplier": 0.4,
                    "take_profit_bias": 1.2,
                    "entry_cooldown_bars_multiplier": 2.0,
                },
                runtime_override_state={
                    "cycles_since_refresh": 0,
                },
                repair_reentry_notes=build_repair_reentry_notes(
                    source_runtime_id="BTC/USDT:USDT|5m|trend_pullback_continuation:seed",
                ),
            ),
        )
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.43,
                )
            ],
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.PAPER,
                        reasons=["paper_validation_continues"],
                    )
                ]
            ),
            portfolio_allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    family="trend_pullback_continuation",
                    stage=PromotionStage.PAPER,
                    allocated_capital=1000.0,
                    weight=0.10,
                    score=0.43,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            previous_states=[previous_state],
        )

        self.assertEqual(lifecycle_policy_runtime_overrides(runtime_states[0].notes)["max_weight_multiplier"], 0.55)
        self.assertEqual(lifecycle_policy_runtime_overrides(runtime_states[0].notes)["take_profit_bias"], 1.15)
        self.assertEqual(
            lifecycle_policy_runtime_overrides(runtime_states[0].notes)["entry_cooldown_bars_multiplier"],
            1.5,
        )
        self.assertEqual(lifecycle_policy_runtime_override_state(runtime_states[0].notes)["cycles_since_refresh"], 1)
        self.assertEqual(runtime_states[0].desired_capital, 550.0)

    def test_paper_bridge_accelerates_runtime_override_decay_when_runtime_recovers(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(
                autonomy_runtime_override_decay_rate=0.25,
                autonomy_runtime_override_cooldown_decay_step=0.5,
            ),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
        )
        previous_state = self._runtime_state_with_overrides(runtime_id)
        opened = bridge.paper_trader.execute_open(
            "BTC/USDT:USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.6,
            rationale="repair recovery",
            quantity=5.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": bridge.BRIDGE_MODE,
                "runtime_id": runtime_id,
                "strategy_id": "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                "family": "trend_pullback_continuation",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.PAPER.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        self.assertIsNotNone(opened)
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.43,
                )
            ],
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.PAPER,
                        reasons=["paper_validation_continues"],
                    )
                ]
            ),
            portfolio_allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    family="trend_pullback_continuation",
                    stage=PromotionStage.PAPER,
                    allocated_capital=1000.0,
                    weight=0.10,
                    score=0.43,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            previous_states=[previous_state],
        )

        self.assertEqual(
            lifecycle_policy_runtime_override_state(runtime_states[0].notes)["recovery_mode"],
            "accelerate",
        )
        self.assertEqual(lifecycle_policy_runtime_override_state(runtime_states[0].notes)["decay_steps_applied"], 2)
        self.assertEqual(lifecycle_policy_runtime_overrides(runtime_states[0].notes)["max_weight_multiplier"], 0.6625)
        self.assertEqual(lifecycle_policy_runtime_overrides(runtime_states[0].notes)["take_profit_bias"], 1.1125)
        self.assertEqual(
            lifecycle_policy_runtime_overrides(runtime_states[0].notes)["entry_cooldown_bars_multiplier"],
            1.0,
        )
        self.assertGreater(
            lifecycle_policy_runtime_override_state(runtime_states[0].notes)["performance_snapshot"]["unrealized_pnl"],
            0.0,
        )
        self.assertEqual(runtime_states[0].desired_capital, 662.5)

    def test_paper_bridge_holds_runtime_overrides_when_runtime_is_still_degraded(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(
                autonomy_runtime_override_decay_rate=0.25,
                autonomy_runtime_override_cooldown_decay_step=0.5,
            ),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
        )
        previous_state = self._runtime_state_with_overrides(runtime_id)

        for entry_price, exit_price in ((130.0, 120.0), (128.0, 118.0)):
            opened = bridge.paper_trader.execute_open(
                "BTC/USDT:USDT",
                direction=SignalDirection.LONG,
                price=entry_price,
                confidence=0.6,
                rationale="repair recovery test",
                quantity=1.0,
                metadata={
                    "source": bridge.MANAGED_SOURCE,
                    "bridge_mode": bridge.BRIDGE_MODE,
                    "runtime_id": runtime_id,
                    "strategy_id": "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "family": "trend_pullback_continuation",
                    "timeframe": "5m",
                    "lifecycle_state": RuntimeLifecycleState.PAPER.value,
                    "execution_action": ExecutionAction.KEEP.value,
                },
            )
            self.assertIsNotNone(opened)
            bridge.paper_trader.execute_close(
                "BTC/USDT:USDT",
                exit_price,
                reason="repair recovery failure",
            )

        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.43,
                )
            ],
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.PAPER,
                        reasons=["paper_validation_continues"],
                    )
                ]
            ),
            portfolio_allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    family="trend_pullback_continuation",
                    stage=PromotionStage.PAPER,
                    allocated_capital=1000.0,
                    weight=0.10,
                    score=0.43,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            previous_states=[previous_state],
        )

        self.assertEqual(lifecycle_policy_runtime_override_state(runtime_states[0].notes)["recovery_mode"], "hold")
        self.assertEqual(lifecycle_policy_runtime_override_state(runtime_states[0].notes)["decay_steps_applied"], 0)
        self.assertEqual(lifecycle_policy_runtime_overrides(runtime_states[0].notes)["max_weight_multiplier"], 0.4)
        self.assertEqual(lifecycle_policy_runtime_overrides(runtime_states[0].notes)["take_profit_bias"], 1.2)
        self.assertEqual(
            lifecycle_policy_runtime_overrides(runtime_states[0].notes)["entry_cooldown_bars_multiplier"],
            2.0,
        )
        self.assertEqual(
            lifecycle_policy_runtime_override_state(runtime_states[0].notes)["performance_snapshot"]["consecutive_losses"],
            2,
        )
        self.assertEqual(runtime_states[0].desired_capital, 400.0)

    def test_paper_bridge_releases_runtime_overrides_after_confirmed_recovery(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(
                autonomy_runtime_override_decay_rate=0.25,
                autonomy_runtime_override_cooldown_decay_step=0.5,
            ),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
        )
        previous_state = self._runtime_state_with_overrides(runtime_id)

        for entry_price, exit_price in ((100.0, 110.0), (101.0, 108.0)):
            opened = bridge.paper_trader.execute_open(
                "BTC/USDT:USDT",
                direction=SignalDirection.LONG,
                price=entry_price,
                confidence=0.6,
                rationale="repair recovery success",
                quantity=1.0,
                metadata={
                    "source": bridge.MANAGED_SOURCE,
                    "bridge_mode": bridge.BRIDGE_MODE,
                    "runtime_id": runtime_id,
                    "strategy_id": "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "family": "trend_pullback_continuation",
                    "timeframe": "5m",
                    "lifecycle_state": RuntimeLifecycleState.PAPER.value,
                    "execution_action": ExecutionAction.KEEP.value,
                },
            )
            self.assertIsNotNone(opened)
            bridge.paper_trader.execute_close(
                "BTC/USDT:USDT",
                exit_price,
                reason="repair recovery success",
            )

        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.43,
                )
            ],
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.PAPER,
                        reasons=["paper_validation_continues"],
                    )
                ]
            ),
            portfolio_allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    family="trend_pullback_continuation",
                    stage=PromotionStage.PAPER,
                    allocated_capital=1000.0,
                    weight=0.10,
                    score=0.43,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            previous_states=[previous_state],
        )

        self.assertEqual(lifecycle_policy_runtime_override_state(runtime_states[0].notes)["recovery_mode"], "release")
        self.assertEqual(lifecycle_policy_runtime_overrides(runtime_states[0].notes), {})
        self.assertFalse(
            runtime_states[0].notes["runtime_lifecycle_policy"]["repair_reentry"]["active"]
        )
        self.assertEqual(
            lifecycle_policy_runtime_override_state(runtime_states[0].notes)["performance_snapshot"]["closed_trade_count"],
            2,
        )
        self.assertEqual(runtime_states[0].desired_capital, 1000.0)

    def test_paper_bridge_reads_reentry_state_from_unified_lifecycle_policy(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(
                autonomy_runtime_override_decay_rate=0.25,
                autonomy_runtime_override_cooldown_decay_step=0.5,
            ),
        )
        source_runtime_id = "BTC/USDT:USDT|5m|trend_pullback_continuation:seed"
        opened = bridge.paper_trader.execute_open(
            "BTC/USDT:USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.6,
            rationale="seed runtime",
            quantity=5.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": bridge.BRIDGE_MODE,
                "runtime_id": source_runtime_id,
                "strategy_id": "trend_pullback_continuation:seed",
                "family": "trend_pullback_continuation",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.PAPER.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        self.assertIsNotNone(opened)
        bridge.paper_trader.execute_close(
            "BTC/USDT:USDT",
            101.0,
            reason="repair rotation",
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
        )
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.43,
                )
            ],
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.PAPER,
                        reasons=["paper_validation_continues"],
                    )
                ]
            ),
            portfolio_allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    family="trend_pullback_continuation",
                    stage=PromotionStage.PAPER,
                    allocated_capital=1000.0,
                    weight=0.10,
                    score=0.43,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            previous_states=[
                RuntimeState(
                    runtime_id=runtime_id,
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    family="trend_pullback_continuation",
                    lifecycle_state=RuntimeLifecycleState.PAPER,
                    promotion_stage=PromotionStage.PAPER,
                    target_stage=PromotionStage.PAPER,
                    last_directive_action=ExecutionAction.KEEP,
                    score=0.43,
                    allocated_capital=1000.0,
                    desired_capital=400.0,
                    current_capital=0.0,
                    current_weight=0.10,
                    notes=compose_runtime_policy_notes(
                        repair_reentry_notes=build_repair_reentry_notes(
                            source_runtime_id=source_runtime_id,
                            source_strategy_id="trend_pullback_continuation:seed",
                            raw_stage="paper",
                            effective_target_stage="paper",
                            requested_validation_stage="paper",
                            runtime_overrides={
                                "max_weight_multiplier": 0.4,
                                "entry_cooldown_bars_multiplier": 2.0,
                            },
                        ),
                        runtime_overrides={
                            "max_weight_multiplier": 0.4,
                            "entry_cooldown_bars_multiplier": 2.0,
                        },
                        runtime_override_state={
                            "recovery_mode": "neutral",
                        },
                        reentry_state={
                            "mode": "repair_reentry",
                            "phase": "cooldown",
                            "recovery_mode": "neutral",
                            "cooldown_active": True,
                            "active_overrides": [
                                "entry_cooldown_bars_multiplier",
                                "max_weight_multiplier",
                            ],
                            "source_runtime_id": source_runtime_id,
                            "source_strategy_id": "trend_pullback_continuation:seed",
                            "effective_target_stage": "paper",
                            "requested_validation_stage": "paper",
                            "raw_stage": "paper",
                        },
                    ),
                )
            ],
        )
        intents = bridge.build_execution_intents(runtime_states)

        self.assertEqual(lifecycle_policy_runtime_override_state(runtime_states[0].notes)["cycles_since_refresh"], 1)
        self.assertEqual(lifecycle_policy_reentry_state(runtime_states[0].notes)["phase"], "cooldown")
        self.assertEqual(intents[0].action.value, "skip")
        self.assertIn("repair_reentry_cooldown_active", intents[0].reasons)

    def test_paper_bridge_reads_runtime_override_decay_from_unified_lifecycle_policy(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            config=EvolutionConfig(
                autonomy_runtime_override_decay_rate=0.25,
                autonomy_runtime_override_cooldown_decay_step=0.5,
            ),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
        )
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.43,
                )
            ],
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.PAPER,
                        reasons=["paper_validation_continues"],
                    )
                ]
            ),
            portfolio_allocations=[
                PortfolioAllocation(
                    symbol="BTC/USDT:USDT",
                    strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    family="trend_pullback_continuation",
                    stage=PromotionStage.PAPER,
                    allocated_capital=1000.0,
                    weight=0.10,
                    score=0.43,
                    timeframe="5m",
                )
            ],
            total_capital=10000.0,
            previous_states=[
                RuntimeState(
                    runtime_id=runtime_id,
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    family="trend_pullback_continuation",
                    lifecycle_state=RuntimeLifecycleState.PAPER,
                    promotion_stage=PromotionStage.PAPER,
                    target_stage=PromotionStage.PAPER,
                    last_directive_action=ExecutionAction.KEEP,
                    score=0.42,
                    allocated_capital=1000.0,
                    desired_capital=400.0,
                    current_capital=0.0,
                    current_weight=0.10,
                    notes=compose_runtime_policy_notes(
                        repair_reentry_notes=build_repair_reentry_notes(
                            source_runtime_id="BTC/USDT:USDT|5m|trend_pullback_continuation:seed",
                            source_strategy_id="trend_pullback_continuation:seed",
                            raw_stage="paper",
                            effective_target_stage="paper",
                            requested_validation_stage="paper",
                            runtime_overrides={
                                "max_weight_multiplier": 0.4,
                                "take_profit_bias": 1.2,
                                "entry_cooldown_bars_multiplier": 2.0,
                            },
                        ),
                        runtime_overrides={
                            "max_weight_multiplier": 0.4,
                            "take_profit_bias": 1.2,
                            "entry_cooldown_bars_multiplier": 2.0,
                        },
                        runtime_override_state={
                            "recovery_mode": "neutral",
                        },
                        reentry_state={
                            "mode": "repair_reentry",
                            "phase": "probation",
                            "recovery_mode": "neutral",
                            "active_overrides": [
                                "entry_cooldown_bars_multiplier",
                                "max_weight_multiplier",
                                "take_profit_bias",
                            ],
                            "source_runtime_id": "BTC/USDT:USDT|5m|trend_pullback_continuation:seed",
                            "source_strategy_id": "trend_pullback_continuation:seed",
                            "effective_target_stage": "paper",
                            "requested_validation_stage": "paper",
                            "raw_stage": "paper",
                        },
                    ),
                )
            ],
        )

        self.assertEqual(lifecycle_policy_runtime_overrides(runtime_states[0].notes)["max_weight_multiplier"], 0.55)
        self.assertEqual(lifecycle_policy_runtime_overrides(runtime_states[0].notes)["take_profit_bias"], 1.15)
        self.assertEqual(
            lifecycle_policy_runtime_overrides(runtime_states[0].notes)["entry_cooldown_bars_multiplier"],
            1.5,
        )
        self.assertEqual(lifecycle_policy_runtime_override_state(runtime_states[0].notes)["cycles_since_refresh"], 1)
        self.assertEqual(runtime_states[0].desired_capital, 550.0)

    def test_paper_bridge_does_not_double_apply_weight_cap_already_consumed_in_allocator(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            config=EvolutionConfig(),
        )
        allocator = PortfolioAllocator(EvolutionConfig())
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
        )
        previous_state = self._runtime_state_with_overrides(runtime_id)
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.43,
                )
            ],
        )
        allocations = allocator.allocate(
            [result],
            total_capital=10000.0,
            previous_states=[previous_state],
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.PAPER,
                        reasons=["paper_validation_continues"],
                    )
                ]
            ),
            portfolio_allocations=allocations,
            total_capital=10000.0,
            previous_states=[previous_state],
        )

        self.assertEqual(allocations[0].allocated_capital, 1100.0)
        self.assertIn(
            PortfolioAllocator.RUNTIME_OVERRIDE_WEIGHT_CAP_REASON,
            allocations[0].reasons,
        )
        self.assertEqual(runtime_states[0].allocated_capital, 1100.0)
        self.assertEqual(runtime_states[0].desired_capital, 1100.0)
        self.assertIn(
            PortfolioAllocator.RUNTIME_OVERRIDE_WEIGHT_CAP_REASON,
            runtime_states[0].notes["allocation_reasons"],
        )

    def test_paper_bridge_tracks_allocator_relaxed_weight_cap_after_recovery(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            config=EvolutionConfig(
                autonomy_runtime_override_decay_rate=0.25,
                autonomy_runtime_override_cooldown_decay_step=0.5,
            ),
        )
        allocator = PortfolioAllocator(
            EvolutionConfig(
                autonomy_runtime_override_decay_rate=0.25,
                autonomy_runtime_override_cooldown_decay_step=0.5,
            )
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
        )
        previous_state = self._runtime_state_with_overrides(runtime_id)
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.43,
                )
            ],
        )
        opened = bridge.paper_trader.execute_open(
            "BTC/USDT:USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.6,
            rationale="repair recovery accelerate",
            quantity=5.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": bridge.BRIDGE_MODE,
                "runtime_id": runtime_id,
                "strategy_id": "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                "family": "trend_pullback_continuation",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.PAPER.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        self.assertIsNotNone(opened)
        runtime_evidence = RuntimeEvidenceCollector(
            self.feed,
            bridge.config,
        ).collect(
            [runtime_id],
            previous_states=[previous_state],
        )
        allocations = allocator.allocate(
            [result],
            total_capital=10000.0,
            previous_states=[previous_state],
            runtime_evidence=runtime_evidence,
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.PAPER,
                        reasons=["paper_validation_continues"],
                    )
                ]
            ),
            portfolio_allocations=allocations,
            total_capital=10000.0,
            previous_states=[previous_state],
        )

        self.assertEqual(allocations[0].allocated_capital, 1325.0)
        self.assertEqual(runtime_states[0].allocated_capital, 1325.0)
        self.assertEqual(runtime_states[0].desired_capital, 1325.0)
        self.assertEqual(
            lifecycle_policy_runtime_override_state(runtime_states[0].notes)["recovery_mode"],
            "accelerate",
        )

    def test_paper_bridge_respects_portfolio_level_reentry_cooldown_reserve(self):
        bridge = AutonomyPaperBridge(
            self.feed,
            registry=self.registry,
            paper_trader=PaperTrader(self.storage),
            config=EvolutionConfig(),
        )
        allocator = PortfolioAllocator(EvolutionConfig())
        source_runtime_id = "BTC/USDT:USDT|5m|trend_pullback_continuation:seed"
        opened = bridge.paper_trader.execute_open(
            "BTC/USDT:USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.6,
            rationale="seed runtime",
            quantity=5.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": bridge.BRIDGE_MODE,
                "runtime_id": source_runtime_id,
                "strategy_id": "trend_pullback_continuation:seed",
                "family": "trend_pullback_continuation",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.PAPER.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        self.assertIsNotNone(opened)
        bridge.paper_trader.execute_close(
            "BTC/USDT:USDT",
            101.0,
            reason="repair rotation",
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
        )
        result = self._experiment_result(
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            scorecards=[
                self._scorecard(
                    "trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
                    "trend_pullback_continuation",
                    PromotionStage.PAPER,
                    0.43,
                )
            ],
        )
        result.notes = compose_runtime_policy_notes(
            base_notes=result.notes,
            repair_reentry_notes=build_repair_reentry_notes(
                source_runtime_id=source_runtime_id,
                runtime_overrides={
                    "entry_cooldown_bars_multiplier": 2.0,
                },
            ),
        )
        allocations = allocator.allocate(
            [result],
            total_capital=10000.0,
            latest_close_time_by_runtime=latest_managed_close_time_index(
                self.storage,
                managed_source=bridge.MANAGED_SOURCE,
            ),
        )

        runtime_states = bridge.build_runtime_states(
            results=[result],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.PAPER,
                        reasons=["paper_validation_continues"],
                    )
                ]
            ),
            portfolio_allocations=allocations,
            total_capital=10000.0,
        )
        intents = bridge.build_execution_intents(runtime_states)

        self.assertEqual(allocations, [])
        self.assertEqual(runtime_states[0].allocated_capital, 0.0)
        self.assertEqual(runtime_states[0].desired_capital, 0.0)
        self.assertEqual(intents[0].action.value, "skip")
        self.assertIn("repair_reentry_cooldown_active", intents[0].reasons)

    @staticmethod
    def _scorecard(strategy_id: str, family: str, stage: PromotionStage, score: float) -> ScoreCard:
        return ScoreCard(
            genome=StrategyGenome(strategy_id, family, {"lookback": 18.0, "hold_bars": 6.0}),
            stage=stage,
            edge_score=score,
            robustness_score=0.70,
            deployment_score=score,
            total_score=score,
            reasons=["test"],
        )

    @staticmethod
    def _experiment_result(
        *,
        symbol: str,
        timeframe: str,
        scorecards: list[ScoreCard],
    ) -> ExperimentResult:
        return ExperimentResult(
            symbol=symbol,
            timeframe=timeframe,
            scorecards=scorecards,
            promoted=scorecards,
            allocations=[],
            candle_count=200,
        )

    @staticmethod
    def _runtime_state_with_overrides(runtime_id: str) -> RuntimeState:
        return RuntimeState(
            runtime_id=runtime_id,
            symbol="BTC/USDT:USDT",
            timeframe="5m",
            strategy_id="trend_pullback_continuation@BTC_USDT_USDT_5m:repair",
            family="trend_pullback_continuation",
            lifecycle_state=RuntimeLifecycleState.PAPER,
            promotion_stage=PromotionStage.PAPER,
            target_stage=PromotionStage.PAPER,
            last_directive_action=ExecutionAction.KEEP,
            score=0.42,
            allocated_capital=1000.0,
            desired_capital=400.0,
            current_capital=0.0,
            current_weight=0.10,
            capital_multiplier=1.0,
            limited_live_cycles=0,
            notes=compose_runtime_policy_notes(
                runtime_overrides={
                    "max_weight_multiplier": 0.4,
                    "take_profit_bias": 1.2,
                    "entry_cooldown_bars_multiplier": 2.0,
                },
                runtime_override_state={
                    "cycles_since_refresh": 0,
                },
                repair_reentry_notes=build_repair_reentry_notes(
                    source_runtime_id="BTC/USDT:USDT|5m|trend_pullback_continuation:seed",
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
