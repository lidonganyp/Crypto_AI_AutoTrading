import tempfile
import unittest
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.models import SignalDirection
from core.storage import Storage
from execution.live_trader import LiveTrader
from execution.paper_trader import PaperTrader
from nextgen_evolution import (
    AutonomyDirective,
    AutonomyLiveBridge,
    AutonomyLiveStatus,
    EvolutionConfig,
    ExecutionAction,
    ExecutionDirective,
    PromotionRegistry,
    SQLiteOHLCVFeed,
)
from nextgen_evolution.experiment_lab import ExperimentResult
from nextgen_evolution.runtime_override_policy import (
    build_repair_reentry_notes,
    compose_runtime_policy_notes,
)
from nextgen_evolution.models import (
    PortfolioAllocation,
    PromotionStage,
    RuntimeLifecycleState,
    RuntimeState,
    ScoreCard,
    StrategyGenome,
)


def make_intraday_candles(count: int, base: float):
    candles = []
    for idx in range(count):
        price = base + idx * 0.1 + ((idx % 6) - 3) * 0.12
        candles.append(
            {
                "timestamp": 1700000000000 + idx * 300000,
                "open": price - 0.1,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price,
                "volume": 1000 + idx,
            }
        )
    return candles


class NextGenLiveBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "cryptoai.db"
        self.storage = Storage(str(self.db_path))
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "5m",
            make_intraday_candles(240, 100.0),
        )
        self.storage.insert_ohlcv(
            "ETH/USDT:USDT",
            "5m",
            make_intraday_candles(240, 200.0),
        )
        self.feed = SQLiteOHLCVFeed(str(self.db_path))
        self.registry = PromotionRegistry(str(self.db_path))
        self.paper = PaperTrader(self.storage)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_live_bridge_defaults_to_dry_run_and_caps_limited_live_capital(self):
        config = EvolutionConfig(
            autonomy_live_enabled=False,
            autonomy_live_require_explicit_enable=True,
            autonomy_live_whitelist=("BTC/USDT:USDT",),
            autonomy_limited_live_max_weight=0.05,
            autonomy_live_blast_radius_capital_pct=0.04,
        )
        bridge = AutonomyLiveBridge(
            self.feed,
            registry=self.registry,
            live_trader=LiveTrader(self.storage, exchange=object(), enabled=True),
            config=config,
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation:seed",
        )

        runtime_states, intents = bridge.apply(
            results=[self._experiment_result("BTC/USDT:USDT", "trend_pullback_continuation:seed")],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.PROMOTE_TO_LIVE,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.LIVE,
                    )
                ]
            ),
            portfolio_allocations=[
                self._allocation("BTC/USDT:USDT", "trend_pullback_continuation:seed", 2500.0, 0.25)
            ],
            total_capital=10000.0,
        )

        self.assertEqual(runtime_states[0].lifecycle_state, RuntimeLifecycleState.LIMITED_LIVE)
        self.assertEqual(runtime_states[0].desired_capital, 400.0)
        self.assertEqual(intents[0].action.value, "skip")
        self.assertEqual(intents[0].status, "skipped")
        self.assertIn("operator_gate_blocked", intents[0].reasons)
        self.assertEqual(len(self.storage.get_open_trades()), 0)

    def test_live_bridge_skips_non_whitelisted_symbol(self):
        config = EvolutionConfig(
            autonomy_live_whitelist=("ETH/USDT:USDT",),
        )
        bridge = AutonomyLiveBridge(
            self.feed,
            registry=self.registry,
            config=config,
            operator_status=AutonomyLiveStatus(
                requested_live=True,
                effective_live=True,
                dry_run=False,
                allow_entries=True,
                allow_managed_closes=True,
                force_flatten=False,
                runtime_mode="live",
                allow_live_orders=True,
                provider="okx",
                whitelist=("BTC/USDT:USDT", "ETH/USDT:USDT"),
                reasons=[],
            ),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "volatility_reclaim:seed",
        )

        _, intents = bridge.apply(
            results=[self._experiment_result("BTC/USDT:USDT", "volatility_reclaim:seed")],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.PROMOTE_TO_LIVE,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.LIVE,
                    )
                ]
            ),
            portfolio_allocations=[
                self._allocation("BTC/USDT:USDT", "volatility_reclaim:seed", 1800.0, 0.18)
            ],
            total_capital=10000.0,
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].action.value, "skip")
        self.assertEqual(intents[0].status, "skipped")
        self.assertIn("symbol_not_whitelisted", intents[0].reasons)

    def test_live_bridge_refuses_multiple_new_live_runtimes_in_same_cycle(self):
        config = EvolutionConfig(
            autonomy_live_whitelist=("BTC/USDT:USDT", "ETH/USDT:USDT"),
            autonomy_live_max_active_runtimes=1,
        )
        bridge = AutonomyLiveBridge(
            self.feed,
            registry=self.registry,
            config=config,
            operator_status=AutonomyLiveStatus(
                requested_live=True,
                effective_live=True,
                dry_run=False,
                allow_entries=True,
                allow_managed_closes=True,
                force_flatten=False,
                runtime_mode="live",
                allow_live_orders=True,
                provider="okx",
                whitelist=("BTC/USDT:USDT", "ETH/USDT:USDT"),
                reasons=[],
            ),
        )
        btc_runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "btc_breakout:seed",
        )
        eth_runtime_id = bridge.runtime_id(
            "ETH/USDT:USDT",
            "5m",
            "eth_reclaim:seed",
        )

        _, intents = bridge.apply(
            results=[
                self._experiment_result("BTC/USDT:USDT", "btc_breakout:seed"),
                self._experiment_result("ETH/USDT:USDT", "eth_reclaim:seed"),
            ],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=btc_runtime_id,
                        action=ExecutionAction.PROMOTE_TO_LIVE,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.LIVE,
                    ),
                    ExecutionDirective(
                        strategy_id=eth_runtime_id,
                        action=ExecutionAction.PROMOTE_TO_LIVE,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.LIVE,
                    ),
                ]
            ),
            portfolio_allocations=[
                self._allocation("BTC/USDT:USDT", "btc_breakout:seed", 1800.0, 0.18),
                self._allocation("ETH/USDT:USDT", "eth_reclaim:seed", 1700.0, 0.17),
            ],
            total_capital=10000.0,
        )

        open_intents = [item for item in intents if item.action.value == "open"]
        skip_intents = [item for item in intents if item.action.value == "skip"]
        self.assertEqual(len(open_intents), 1)
        self.assertEqual(len(skip_intents), 1)
        self.assertEqual(open_intents[0].symbol, "BTC/USDT:USDT")
        self.assertEqual(open_intents[0].status, "dry_run")
        self.assertEqual(skip_intents[0].symbol, "ETH/USDT:USDT")
        self.assertIn("live_runtime_cap_reached", skip_intents[0].reasons)

    def test_live_bridge_closes_live_managed_runtime_on_rollback(self):
        config = EvolutionConfig(
            autonomy_live_whitelist=("BTC/USDT:USDT",),
        )
        bridge = AutonomyLiveBridge(
            self.feed,
            registry=self.registry,
            config=config,
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "trend_pullback_continuation:seed",
        )
        self.paper.execute_open(
            "BTC/USDT:USDT",
            SignalDirection.LONG,
            120.0,
            0.6,
            "seed live runtime",
            position_value=1200.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": bridge.BRIDGE_MODE,
                "runtime_id": runtime_id,
                "strategy_id": "trend_pullback_continuation:seed",
                "family": "trend_pullback_continuation",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.LIMITED_LIVE.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )

        runtime_states, intents = bridge.apply(
            results=[self._experiment_result("BTC/USDT:USDT", "trend_pullback_continuation:seed")],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.EXIT,
                        from_stage=PromotionStage.LIVE,
                        target_stage=PromotionStage.SHADOW,
                    )
                ]
            ),
            portfolio_allocations=[],
            total_capital=10000.0,
        )

        self.assertEqual(runtime_states[0].lifecycle_state, RuntimeLifecycleState.ROLLBACK)
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].action.value, "close")
        self.assertEqual(intents[0].status, "dry_run")

    def test_live_bridge_does_not_take_over_paper_managed_position(self):
        config = EvolutionConfig(
            autonomy_live_whitelist=("BTC/USDT:USDT",),
        )
        bridge = AutonomyLiveBridge(
            self.feed,
            registry=self.registry,
            config=config,
        )
        self.paper.execute_open(
            "BTC/USDT:USDT",
            SignalDirection.LONG,
            118.0,
            0.5,
            "paper managed runtime",
            position_value=1000.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": "paper",
                "runtime_id": "BTC/USDT:USDT|5m|paper_runtime:seed",
                "strategy_id": "paper_runtime:seed",
                "family": "paper_runtime",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.PAPER.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "live_runtime:seed",
        )

        _, intents = bridge.apply(
            results=[self._experiment_result("BTC/USDT:USDT", "live_runtime:seed")],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.PROMOTE_TO_LIVE,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.LIVE,
                    )
                ]
            ),
            portfolio_allocations=[
                self._allocation("BTC/USDT:USDT", "live_runtime:seed", 1600.0, 0.16)
            ],
            total_capital=10000.0,
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].action.value, "skip")
        self.assertEqual(intents[0].status, "skipped")
        self.assertIn("symbol_managed_elsewhere", intents[0].reasons)

    def test_live_bridge_adds_to_existing_managed_runtime_when_target_capital_increases(self):
        class FakeAdapter:
            def fetch_free_balance(self, asset):
                return 10000.0

            def estimate_slippage(self, symbol, side, quantity, reference_price):
                from execution.exchange_adapter import SlippageEstimate

                return SlippageEstimate(
                    symbol=symbol,
                    reference_price=reference_price,
                    expected_price=reference_price,
                    slippage_pct=0.0005,
                )

            def place_market_order(self, symbol, side, quantity):
                return {
                    "exchange_order_id": f"ex-{side}",
                    "status": "closed",
                    "price": 100.0,
                    "average_price": 100.0,
                    "filled_qty": quantity,
                    "raw": {"id": f"ex-{side}"},
                }

        config = EvolutionConfig(
            autonomy_live_whitelist=("BTC/USDT:USDT",),
        )
        bridge = AutonomyLiveBridge(
            self.feed,
            registry=self.registry,
            live_trader=LiveTrader(
                self.storage,
                exchange=FakeAdapter(),
                enabled=True,
            ),
            config=config,
            operator_status=AutonomyLiveStatus(
                requested_live=True,
                effective_live=True,
                dry_run=False,
                allow_entries=True,
                allow_managed_closes=True,
                force_flatten=False,
                runtime_mode="live",
                allow_live_orders=True,
                provider="okx",
                whitelist=("BTC/USDT:USDT",),
                reasons=[],
            ),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "scale_runtime:seed",
        )
        self.paper.execute_open(
            "BTC/USDT:USDT",
            SignalDirection.LONG,
            100.0,
            0.6,
            "seed live runtime",
            quantity=4.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": bridge.BRIDGE_MODE,
                "runtime_id": runtime_id,
                "strategy_id": "scale_runtime:seed",
                "family": "scale_runtime",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.LIMITED_LIVE.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        initial_qty = float(self.storage.get_positions()[0]["quantity"])

        _, intents = bridge.apply(
            results=[self._experiment_result("BTC/USDT:USDT", "scale_runtime:seed")],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.KEEP,
                        from_stage=PromotionStage.LIVE,
                        target_stage=PromotionStage.LIVE,
                    )
                ]
            ),
            portfolio_allocations=[
                self._allocation("BTC/USDT:USDT", "scale_runtime:seed", 1800.0, 0.18)
            ],
            total_capital=10000.0,
            previous_states=[
                RuntimeState(
                    runtime_id=runtime_id,
                    symbol="BTC/USDT:USDT",
                    timeframe="5m",
                    strategy_id="scale_runtime:seed",
                    family="scale_runtime",
                    lifecycle_state=RuntimeLifecycleState.LIMITED_LIVE,
                    promotion_stage=PromotionStage.LIVE,
                    target_stage=PromotionStage.LIVE,
                    last_directive_action=ExecutionAction.PROMOTE_TO_LIVE,
                    desired_capital=500.0,
                    current_capital=400.0,
                    limited_live_cycles=1,
                )
            ],
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].action.value, "open")
        self.assertEqual(intents[0].status, "executed")
        self.assertIn("rebalance_up", intents[0].reasons)
        self.assertEqual(len(self.storage.get_open_trades()), 1)
        self.assertGreater(float(self.storage.get_positions()[0]["quantity"]), initial_qty)

    def test_live_bridge_marks_profit_lock_harvest_reduce_explicitly(self):
        class FakeAdapter:
            def fetch_free_balance(self, asset):
                return 10000.0

            def estimate_slippage(self, symbol, side, quantity, reference_price):
                from execution.exchange_adapter import SlippageEstimate

                return SlippageEstimate(
                    symbol=symbol,
                    reference_price=reference_price,
                    expected_price=reference_price,
                    slippage_pct=0.0005,
                )

            def place_market_order(self, symbol, side, quantity):
                return {
                    "exchange_order_id": f"ex-{side}",
                    "status": "closed",
                    "price": 100.0,
                    "average_price": 100.0,
                    "filled_qty": quantity,
                    "raw": {"id": f"ex-{side}"},
                }

        config = EvolutionConfig(
            autonomy_live_whitelist=("BTC/USDT:USDT",),
        )
        bridge = AutonomyLiveBridge(
            self.feed,
            registry=self.registry,
            live_trader=LiveTrader(
                self.storage,
                exchange=FakeAdapter(),
                enabled=True,
            ),
            config=config,
            operator_status=AutonomyLiveStatus(
                requested_live=True,
                effective_live=True,
                dry_run=False,
                allow_entries=True,
                allow_managed_closes=True,
                force_flatten=False,
                runtime_mode="live",
                allow_live_orders=True,
                provider="okx",
                whitelist=("BTC/USDT:USDT",),
                reasons=[],
            ),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "profit_lock_runtime:seed",
        )
        self.paper.execute_open(
            "BTC/USDT:USDT",
            SignalDirection.LONG,
            100.0,
            0.6,
            "seed live runtime",
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

        _, intents = bridge.apply(
            results=[self._experiment_result("BTC/USDT:USDT", "profit_lock_runtime:seed")],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.SCALE_DOWN,
                        from_stage=PromotionStage.LIVE,
                        target_stage=PromotionStage.LIVE,
                        reasons=["profit_lock_harvest"],
                    )
                ]
            ),
            portfolio_allocations=[
                self._allocation("BTC/USDT:USDT", "profit_lock_runtime:seed", 500.0, 0.05)
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
                    desired_capital=1000.0,
                    current_capital=1000.0,
                    limited_live_cycles=2,
                )
            ],
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
            "nextgen_autonomy_live_profit_lock_reduce",
        )

    def test_live_bridge_applies_take_profit_bias_and_reentry_cooldown(self):
        class FakeAdapter:
            def fetch_free_balance(self, asset):
                return 10000.0

            def estimate_slippage(self, symbol, side, quantity, reference_price):
                from execution.exchange_adapter import SlippageEstimate

                return SlippageEstimate(
                    symbol=symbol,
                    reference_price=reference_price,
                    expected_price=reference_price,
                    slippage_pct=0.0005,
                )

            def place_market_order(self, symbol, side, quantity):
                return {
                    "exchange_order_id": f"ex-{side}",
                    "status": "closed",
                    "price": 100.0,
                    "average_price": 100.0,
                    "filled_qty": quantity,
                    "raw": {"id": f"ex-{side}"},
                }

        config = EvolutionConfig(
            autonomy_live_whitelist=("BTC/USDT:USDT",),
        )
        bridge = AutonomyLiveBridge(
            self.feed,
            registry=self.registry,
            live_trader=LiveTrader(
                self.storage,
                exchange=FakeAdapter(),
                enabled=True,
            ),
            config=config,
            operator_status=AutonomyLiveStatus(
                requested_live=True,
                effective_live=True,
                dry_run=False,
                allow_entries=True,
                allow_managed_closes=True,
                force_flatten=False,
                runtime_mode="live",
                allow_live_orders=True,
                provider="okx",
                whitelist=("BTC/USDT:USDT",),
                reasons=[],
            ),
        )
        source_runtime_id = "BTC/USDT:USDT|5m|live_runtime:seed"
        self.paper.execute_open(
            "BTC/USDT:USDT",
            SignalDirection.LONG,
            100.0,
            0.6,
            "seed runtime",
            quantity=5.0,
            metadata={
                "source": bridge.MANAGED_SOURCE,
                "bridge_mode": bridge.BRIDGE_MODE,
                "runtime_id": source_runtime_id,
                "strategy_id": "live_runtime:seed",
                "family": "live_runtime",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.LIVE.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        self.paper.execute_close(
            "BTC/USDT:USDT",
            101.0,
            reason="repair rotation",
        )

        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "live_runtime@BTC_USDT_USDT_5m:repair",
        )
        result = self._experiment_result("BTC/USDT:USDT", "live_runtime@BTC_USDT_USDT_5m:repair")
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
                    strategy_id=runtime_id,
                    action=ExecutionAction.PROMOTE_TO_LIVE,
                    from_stage=PromotionStage.REJECT,
                    target_stage=PromotionStage.LIVE,
                    reasons=["repair_revalidation_passed"],
                )
            ]
        )
        allocations = [
            self._allocation(
                "BTC/USDT:USDT",
                "live_runtime@BTC_USDT_USDT_5m:repair",
                600.0,
                0.06,
            )
        ]

        _, intents = bridge.apply(
            results=[result],
            directive=directive,
            portfolio_allocations=allocations,
            total_capital=10000.0,
        )

        self.assertEqual(intents[0].action.value, "skip")
        self.assertIn("repair_reentry_cooldown_active", intents[0].reasons)

        stale_exit = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        with self.storage._conn() as conn:
            conn.execute(
                "UPDATE pnl_ledger SET event_time = ? WHERE event_type = 'close'",
                (stale_exit,),
            )

        _, intents = bridge.apply(
            results=[result],
            directive=directive,
            portfolio_allocations=allocations,
            total_capital=10000.0,
        )

        self.assertEqual(intents[0].action.value, "open")
        self.assertEqual(intents[0].status, "executed")
        position = self.storage.get_positions()[0]
        expected_take_profit = round(float(position["entry_price"]) * 1.12, 8)
        self.assertEqual(float(position["take_profit"]), expected_take_profit)
        open_trade = self.storage.get_open_trades()[0]
        metadata = json.loads(open_trade["metadata_json"])
        self.assertIn("runtime_lifecycle_policy", metadata)
        self.assertNotIn("runtime_overrides", metadata)

    def test_live_bridge_skips_new_entry_when_operator_gate_blocks_entries(self):
        config = EvolutionConfig(
            autonomy_live_enabled=False,
            autonomy_live_whitelist=("BTC/USDT:USDT",),
        )
        bridge = AutonomyLiveBridge(
            self.feed,
            registry=self.registry,
            config=config,
            operator_status=AutonomyLiveStatus(
                requested_live=False,
                effective_live=False,
                dry_run=True,
                allow_entries=False,
                allow_managed_closes=False,
                force_flatten=False,
                runtime_mode="live",
                allow_live_orders=True,
                provider="okx",
                whitelist=("BTC/USDT:USDT",),
                reasons=[],
            ),
        )
        runtime_id = bridge.runtime_id(
            "BTC/USDT:USDT",
            "5m",
            "blocked_runtime:seed",
        )

        _, intents = bridge.apply(
            results=[self._experiment_result("BTC/USDT:USDT", "blocked_runtime:seed")],
            directive=AutonomyDirective(
                execution=[
                    ExecutionDirective(
                        strategy_id=runtime_id,
                        action=ExecutionAction.PROMOTE_TO_LIVE,
                        from_stage=PromotionStage.PAPER,
                        target_stage=PromotionStage.LIVE,
                    )
                ]
            ),
            portfolio_allocations=[
                self._allocation("BTC/USDT:USDT", "blocked_runtime:seed", 1600.0, 0.16)
            ],
            total_capital=10000.0,
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].action.value, "skip")
        self.assertEqual(intents[0].status, "skipped")
        self.assertIn("operator_gate_blocked", intents[0].reasons)

    def test_live_bridge_force_flattens_live_managed_position_on_operator_kill_switch(self):
        config = EvolutionConfig(
            autonomy_live_whitelist=("BTC/USDT:USDT",),
        )
        runtime_id = "BTC/USDT:USDT|5m|flatten_runtime:seed"
        self.paper.execute_open(
            "BTC/USDT:USDT",
            SignalDirection.LONG,
            119.0,
            0.6,
            "live managed runtime",
            position_value=1000.0,
            metadata={
                "source": "nextgen_autonomy",
                "bridge_mode": "live",
                "runtime_id": runtime_id,
                "strategy_id": "flatten_runtime:seed",
                "family": "flatten_runtime",
                "timeframe": "5m",
                "lifecycle_state": RuntimeLifecycleState.LIVE.value,
                "execution_action": ExecutionAction.KEEP.value,
            },
        )
        bridge = AutonomyLiveBridge(
            self.feed,
            registry=self.registry,
            config=config,
            operator_status=AutonomyLiveStatus(
                requested_live=False,
                effective_live=False,
                dry_run=True,
                allow_entries=False,
                allow_managed_closes=False,
                force_flatten=True,
                runtime_mode="paper",
                allow_live_orders=False,
                provider="okx",
                whitelist=("BTC/USDT:USDT",),
                kill_switch_active=True,
                kill_switch_reason="ops_pause",
                reasons=["live_kill_switch_active"],
            ),
        )

        _, intents = bridge.apply(
            results=[],
            directive=AutonomyDirective(),
            portfolio_allocations=[],
            total_capital=10000.0,
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].action.value, "close")
        self.assertEqual(intents[0].status, "dry_run")
        self.assertIn("operator_forced_flatten", intents[0].reasons)
        self.assertIn("live_kill_switch_active", intents[0].reasons)

    @staticmethod
    def _scorecard(strategy_id: str) -> ScoreCard:
        return ScoreCard(
            genome=StrategyGenome(
                strategy_id,
                strategy_id.split(":", 1)[0],
                {"lookback": 18.0, "hold_bars": 6.0},
            ),
            stage=PromotionStage.LIVE,
            edge_score=0.61,
            robustness_score=0.78,
            deployment_score=0.64,
            total_score=0.62,
            reasons=["test_live_bridge"],
        )

    @classmethod
    def _experiment_result(cls, symbol: str, strategy_id: str) -> ExperimentResult:
        scorecard = cls._scorecard(strategy_id)
        return ExperimentResult(
            symbol=symbol,
            timeframe="5m",
            scorecards=[scorecard],
            promoted=[scorecard],
            allocations=[],
            candle_count=240,
        )

    @staticmethod
    def _allocation(symbol: str, strategy_id: str, capital: float, weight: float) -> PortfolioAllocation:
        return PortfolioAllocation(
            symbol=symbol,
            strategy_id=strategy_id,
            family=strategy_id.split(":", 1)[0],
            stage=PromotionStage.LIVE,
            allocated_capital=capital,
            weight=weight,
            score=0.62,
            timeframe="5m",
        )


if __name__ == "__main__":
    unittest.main()
