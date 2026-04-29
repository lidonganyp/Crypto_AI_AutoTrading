from datetime import datetime, timezone
from types import SimpleNamespace

from core.models import AccountState, SignalDirection
from execution.exchange_adapter import OKXExchangeAdapter, SlippageEstimate, SlippageGuard
from execution.live_trader import LiveTrader
from execution.reconciler import Reconciler
from strategy.risk_manager import RiskManager
from tests.v2_architecture_support import V2ArchitectureTestCase


class V2ExecutionTests(V2ArchitectureTestCase):
    def test_reconciler_detects_quantity_mismatch(self):
        self.storage.insert_trade(
            {
                "id": "t1",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        self.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 0.5,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "stop_loss": 99.0,
                "take_profit": 105.0,
            }
        )
        result = Reconciler(self.storage).run()
        self.assertEqual(result.status, "mismatch")
        self.assertGreater(result.mismatch_count, 0)
        self.assertGreater(result.mismatch_ratio_pct, 0.0)

    def test_reconciler_detects_exchange_balance_mismatch(self):
        self.storage.insert_trade(
            {
                "id": "t1",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        self.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "stop_loss": 99.0,
                "take_profit": 105.0,
            }
        )

        class FakeExchange:
            def fetch_total_balance(self, asset: str):
                return 0.2 if asset == "BTC" else 0.0

        result = Reconciler(self.storage, exchange=FakeExchange()).run()
        self.assertEqual(result.status, "mismatch")
        self.assertTrue(result.details["exchange_state_checked"])
        self.assertEqual(len(result.details["exchange_balance_mismatches"]), 1)

    def test_reconciler_detects_exchange_open_order_mismatch(self):
        self.storage.insert_order(
            {
                "order_id": "o1",
                "symbol": "BTC/USDT",
                "side": "LONG",
                "order_type": "LIMIT",
                "status": "SUBMITTED",
                "price": 100.0,
                "quantity": 1.0,
                "reason": "x",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        class FakeExchange:
            def fetch_open_orders(self, symbol=None):
                return []

        result = Reconciler(self.storage, exchange=FakeExchange()).run()
        self.assertEqual(result.status, "mismatch")
        self.assertTrue(result.details["exchange_order_state_checked"])
        self.assertEqual(len(result.details["missing_exchange_orders"]), 1)

    def test_reconciler_ignores_exchange_balance_excess_for_tracked_symbol(self):
        self.storage.insert_trade(
            {
                "id": "t1",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        self.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "stop_loss": 99.0,
                "take_profit": 105.0,
            }
        )

        class FakeExchange:
            def fetch_total_balance(self, asset: str):
                return 1.2 if asset == "BTC" else 0.0

        result = Reconciler(self.storage, exchange=FakeExchange()).run()
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.details["exchange_balance_mismatches"], [])

    def test_reconciler_ignores_unexpected_external_open_orders(self):
        class FakeExchange:
            def fetch_open_orders(self, symbol=None):
                return [
                    {
                        "symbol": "ETH/USDT",
                        "side": "buy",
                        "requested_qty": 1.0,
                    }
                ]

        result = Reconciler(self.storage, exchange=FakeExchange()).run()
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.details["unexpected_exchange_orders"], [])

    def test_reconciler_flags_exchange_fetch_errors_as_mismatch(self):
        class FakeExchange:
            def fetch_open_orders(self, symbol=None):
                raise RuntimeError("exchange unavailable")

        result = Reconciler(self.storage, exchange=FakeExchange()).run()
        self.assertEqual(result.status, "mismatch")
        self.assertTrue(result.details["exchange_errors"])

    def test_paper_trader_creates_order_records(self):
        from execution.paper_trader import PaperTrader

        trader = PaperTrader(self.storage)
        opened = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.9,
            "test open",
            position_value=1000.0,
        )
        self.assertIsNotNone(opened)
        closed = trader.execute_close("BTC/USDT", 105.0, "manual")
        self.assertIsNotNone(closed)

        with self.storage._conn() as conn:
            order_count = conn.execute(
                "SELECT COUNT(*) AS c FROM orders"
            ).fetchone()["c"]
        self.assertEqual(order_count, 2)

    def test_paper_trader_balance_includes_realized_pnl(self):
        from execution.paper_trader import PaperTrader

        trader = PaperTrader(self.storage, initial_balance=1000.0)
        trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.9,
            "test open",
            position_value=500.0,
        )
        trader.execute_close("BTC/USDT", 110.0, "manual")
        self.assertAlmostEqual(trader.get_balance(), 1049.2125, places=6)

    def test_paper_trader_records_pnl_ledger_with_trade_metadata(self):
        from execution.paper_trader import PaperTrader

        trader = PaperTrader(self.storage, initial_balance=1000.0)
        opened = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.9,
            "test open",
            position_value=500.0,
            metadata={"model_id": "model-1", "pipeline_mode": "execution"},
        )
        self.assertIsNotNone(opened)
        closed = trader.execute_close("BTC/USDT", 110.0, "manual")
        self.assertIsNotNone(closed)

        with self.storage._conn() as conn:
            rows = conn.execute(
                "SELECT event_type, model_id, fee_cost, net_pnl "
                "FROM pnl_ledger ORDER BY id ASC"
            ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["event_type"], "open")
        self.assertEqual(rows[0]["model_id"], "model-1")
        self.assertGreater(float(rows[0]["fee_cost"]), 0.0)
        self.assertLess(float(rows[0]["net_pnl"]), 0.0)
        self.assertEqual(rows[1]["event_type"], "close")
        self.assertEqual(rows[1]["model_id"], "model-1")
        self.assertGreater(float(rows[1]["net_pnl"]), 0.0)

    def test_paper_trader_allows_explicit_managed_position_add(self):
        from execution.paper_trader import PaperTrader

        trader = PaperTrader(self.storage, initial_balance=2000.0)
        first = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.8,
            "seed runtime",
            quantity=2.0,
            metadata={
                "source": "nextgen_autonomy",
                "runtime_id": "BTC/USDT|5m|scale_runtime:seed",
                "strategy_id": "scale_runtime:seed",
            },
        )
        second = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            102.0,
            0.9,
            "rebalance_up",
            quantity=1.0,
            metadata={
                "source": "nextgen_autonomy",
                "runtime_id": "BTC/USDT|5m|scale_runtime:seed",
                "strategy_id": "scale_runtime:seed",
                "allow_position_add": True,
                "position_adjustment": "rebalance_up",
            },
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(len(self.storage.get_open_trades()), 1)
        self.assertEqual(len(self.storage.get_positions()), 1)
        self.assertAlmostEqual(float(self.storage.get_positions()[0]["quantity"]), 3.0)

    def test_partial_close_updates_remaining_trade_quantity_for_reconciliation(self):
        from execution.paper_trader import PaperTrader

        trader = PaperTrader(self.storage)
        trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.9,
            "test open",
            quantity=10.0,
        )
        closed = trader.execute_close(
            "BTC/USDT",
            105.0,
            "take_profit",
            close_qty=4.0,
        )
        self.assertIsNotNone(closed)
        self.assertFalse(closed["is_full_close"])

        with self.storage._conn() as conn:
            trade = conn.execute(
                "SELECT quantity, initial_quantity, status FROM trades ORDER BY entry_time DESC LIMIT 1"
            ).fetchone()
        self.assertAlmostEqual(float(trade["quantity"]), 6.0)
        self.assertAlmostEqual(float(trade["initial_quantity"]), 10.0)
        self.assertEqual(trade["status"], "open")

        reconciliation = Reconciler(self.storage).run()
        self.assertEqual(reconciliation.status, "ok")

    def test_paper_trader_rejects_short_entries(self):
        from execution.paper_trader import PaperTrader

        trader = PaperTrader(self.storage)
        opened = trader.execute_open(
            "BTC/USDT",
            SignalDirection.SHORT,
            100.0,
            0.9,
            "unsupported short",
            position_value=1000.0,
        )

        self.assertIsNone(opened)
        self.assertEqual(self.storage.get_positions(), [])
        self.assertEqual(self.storage.get_open_trades(), [])

        with self.storage._conn() as conn:
            order_count = conn.execute(
                "SELECT COUNT(*) AS c FROM orders"
            ).fetchone()["c"]
        self.assertEqual(order_count, 0)

    def test_paper_trader_rejects_non_positive_entry_price(self):
        from execution.paper_trader import PaperTrader

        trader = PaperTrader(self.storage)
        opened = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            0.0,
            0.9,
            "invalid price",
            position_value=1000.0,
        )

        self.assertIsNone(opened)
        self.assertEqual(self.storage.get_positions(), [])
        self.assertEqual(self.storage.get_open_trades(), [])

        with self.storage._conn() as conn:
            order_count = conn.execute(
                "SELECT COUNT(*) AS c FROM orders"
            ).fetchone()["c"]
        self.assertEqual(order_count, 0)

    def test_slippage_guard_rejects_large_slippage(self):
        guard = SlippageGuard(max_slippage_pct=0.001)
        allowed, reason = guard.check(
            SlippageEstimate(
                symbol="BTC/USDT:USDT",
                reference_price=100.0,
                expected_price=100.3,
                slippage_pct=0.003,
            )
        )
        self.assertFalse(allowed)
        self.assertIn("exceeds limit", reason)

    def test_risk_manager_cuts_position_on_weak_recent_net_performance(self):
        settings = self.settings.model_copy(deep=True)
        manager = RiskManager(settings.risk, settings.strategy)
        account = AccountState(
            equity=10000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            daily_loss_pct=0.0,
            weekly_loss_pct=0.0,
            drawdown_pct=0.01,
            total_exposure_pct=0.0,
            open_positions=0,
        )
        performance_snapshot = SimpleNamespace(
            recent_closed_trades=5,
            recent_expectancy_pct=-0.5,
            recent_profit_factor=0.8,
            recent_max_drawdown_pct=12.0,
            recent_sortino_like=-0.3,
            equity_return_pct=-1.0,
        )

        result = manager.can_open_position(
            account=account,
            positions=[],
            symbol="BTC/USDT",
            atr=2.0,
            entry_price=100.0,
            liquidity_ratio=1.5,
            performance_snapshot=performance_snapshot,
        )

        self.assertTrue(result.allowed)
        self.assertLess(result.dynamic_position_factor, 1.0)
        self.assertAlmostEqual(result.dynamic_position_factor, 0.3, places=6)

    def test_risk_manager_cuts_total_budget_when_portfolio_heat_rises(self):
        settings = self.settings.model_copy(deep=True)
        settings.risk.max_total_exposure_pct = 0.30
        settings.risk.max_positions = 4
        manager = RiskManager(settings.risk, settings.strategy)
        positions = [
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "current_price": 100.0,
                "quantity": 14.0,
            }
        ]
        account = manager.build_account_state(
            equity=10000.0,
            positions=positions,
            realized_pnl_today=0.0,
            realized_pnl_week=0.0,
        )
        performance_snapshot = SimpleNamespace(
            recent_closed_trades=10,
            recent_expectancy_pct=-0.4,
            recent_profit_factor=0.92,
            recent_max_drawdown_pct=6.0,
            recent_sortino_like=-0.2,
            equity_return_pct=-1.0,
            recent_return_volatility_pct=2.2,
            recent_loss_cluster_ratio_pct=55.0,
            recent_drawdown_velocity_pct=0.8,
        )

        result = manager.can_open_position(
            account=account,
            positions=positions,
            symbol="ETH/USDT",
            atr=2.0,
            entry_price=100.0,
            liquidity_ratio=1.5,
            performance_snapshot=performance_snapshot,
        )

        self.assertTrue(result.allowed)
        self.assertLess(result.allowed_position_value, 1000.0)
        self.assertLess(result.portfolio_heat_factor, 1.0)
        self.assertLess(
            result.effective_max_total_exposure_pct,
            settings.risk.max_total_exposure_pct,
        )
        self.assertEqual(result.effective_max_positions, 3)
        self.assertIn("portfolio heat", result.reason)

    def test_risk_manager_softens_portfolio_heat_during_paper_canary_exploration(self):
        settings = self.settings.model_copy(deep=True)
        manager = RiskManager(settings.risk, settings.strategy)
        account = manager.build_account_state(
            equity=10000.0,
            positions=[],
            realized_pnl_today=0.0,
            realized_pnl_week=0.0,
        )
        performance_snapshot = SimpleNamespace(
            runtime_mode="paper",
            recent_closed_trades=8,
            recent_expectancy_pct=-0.28,
            recent_profit_factor=0.03,
            recent_max_drawdown_pct=1.96,
            recent_sortino_like=-0.68,
            equity_return_pct=-0.02,
            recent_return_volatility_pct=0.24,
            recent_loss_cluster_ratio_pct=87.5,
            recent_drawdown_velocity_pct=0.25,
            paper_canary_open_count=8,
        )

        result = manager.can_open_position(
            account=account,
            positions=[],
            symbol="ETH/USDT",
            atr=2.0,
            entry_price=100.0,
            liquidity_ratio=1.5,
            performance_snapshot=performance_snapshot,
        )

        self.assertTrue(result.allowed)
        self.assertGreaterEqual(result.portfolio_heat_factor, 0.70)
        self.assertIn("paper portfolio heat", result.reason)

    def test_risk_manager_blocks_new_position_when_heat_shrinks_slot_budget(self):
        settings = self.settings.model_copy(deep=True)
        settings.risk.max_positions = 4
        manager = RiskManager(settings.risk, settings.strategy)
        positions = [
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "current_price": 100.0,
                "quantity": 10.0,
            },
            {
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "current_price": 100.0,
                "quantity": 10.0,
            },
        ]
        account = manager.build_account_state(
            equity=10000.0,
            positions=positions,
            realized_pnl_today=0.0,
            realized_pnl_week=0.0,
        )
        performance_snapshot = SimpleNamespace(
            recent_closed_trades=12,
            recent_expectancy_pct=-1.1,
            recent_profit_factor=0.60,
            recent_max_drawdown_pct=12.0,
            recent_sortino_like=-0.6,
            equity_return_pct=-3.0,
            recent_return_volatility_pct=3.4,
            recent_loss_cluster_ratio_pct=80.0,
            recent_drawdown_velocity_pct=1.8,
        )

        result = manager.can_open_position(
            account=account,
            positions=positions,
            symbol="SOL/USDT",
            atr=2.0,
            entry_price=100.0,
            liquidity_ratio=1.5,
            performance_snapshot=performance_snapshot,
        )

        self.assertFalse(result.allowed)
        self.assertIn("portfolio heat max positions reached", result.reason)
        self.assertAlmostEqual(
            result.portfolio_heat_factor,
            settings.risk.portfolio_heat_exposure_floor_pct,
            places=6,
        )
        self.assertEqual(result.effective_max_positions, 2)

    def test_live_trader_rejects_order_when_slippage_too_high(self):
        class FakeAdapter:
            def estimate_slippage(self, symbol, side, quantity, reference_price):
                return SlippageEstimate(
                    symbol=symbol,
                    reference_price=reference_price,
                    expected_price=reference_price * 1.01,
                    slippage_pct=0.01,
                )

        trader = LiveTrader(
            self.storage,
            exchange=FakeAdapter(),
            enabled=True,
            slippage_guard=SlippageGuard(max_slippage_pct=0.001),
        )
        result = trader.execute_open(
            "BTC/USDT:USDT",
            SignalDirection.LONG,
            100.0,
            0.8,
            "test",
            quantity=1.0,
        )
        self.assertIsNone(result)
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT status FROM orders ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(row["status"], "REJECTED")

    def test_live_trader_rejects_non_positive_entry_price(self):
        class FakeAdapter:
            def fetch_free_balance(self, asset):
                return 10000.0

            def estimate_slippage(self, symbol, side, quantity, reference_price):
                raise AssertionError("estimate_slippage should not be called")

        trader = LiveTrader(
            self.storage,
            exchange=FakeAdapter(),
            enabled=True,
            slippage_guard=SlippageGuard(max_slippage_pct=0.001),
        )
        result = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            0.0,
            0.8,
            "invalid live price",
            quantity=1.0,
        )
        self.assertIsNone(result)
        self.assertEqual(self.storage.get_positions(), [])
        self.assertEqual(self.storage.get_open_trades(), [])

        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT status FROM orders ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(row["status"], "REJECTED")

    def test_live_trader_rejects_duplicate_symbol_open(self):
        class FakeAdapter:
            def fetch_free_balance(self, asset):
                return 10000.0

            def estimate_slippage(self, symbol, side, quantity, reference_price):
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

        trader = LiveTrader(
            self.storage,
            exchange=FakeAdapter(),
            enabled=True,
            slippage_guard=SlippageGuard(max_slippage_pct=0.001),
        )
        first = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.8,
            "first open",
            quantity=1.0,
        )
        second = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            101.0,
            0.8,
            "duplicate open",
            quantity=1.0,
        )

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(self.storage.get_open_trades()), 1)
        self.assertEqual(len(self.storage.get_positions()), 1)

        with self.storage._conn() as conn:
            order_rows = conn.execute(
                "SELECT status, reason FROM orders ORDER BY created_at ASC"
            ).fetchall()
        self.assertEqual(len(order_rows), 2)
        self.assertEqual(order_rows[-1]["status"], "REJECTED")
        self.assertEqual(order_rows[-1]["reason"], "position already open")

    def test_live_trader_allows_explicit_managed_position_add(self):
        class FakeAdapter:
            def fetch_free_balance(self, asset):
                return 10000.0

            def estimate_slippage(self, symbol, side, quantity, reference_price):
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

        trader = LiveTrader(
            self.storage,
            exchange=FakeAdapter(),
            enabled=True,
            slippage_guard=SlippageGuard(max_slippage_pct=0.001),
        )
        first = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.8,
            "seed runtime",
            quantity=1.0,
            metadata={
                "source": "nextgen_autonomy",
                "runtime_id": "BTC/USDT|5m|scale_runtime:seed",
                "strategy_id": "scale_runtime:seed",
            },
        )
        second = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            101.0,
            0.9,
            "rebalance_up",
            quantity=0.5,
            metadata={
                "source": "nextgen_autonomy",
                "runtime_id": "BTC/USDT|5m|scale_runtime:seed",
                "strategy_id": "scale_runtime:seed",
                "allow_position_add": True,
                "position_adjustment": "rebalance_up",
            },
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(len(self.storage.get_open_trades()), 1)
        self.assertEqual(len(self.storage.get_positions()), 1)
        self.assertAlmostEqual(float(self.storage.get_positions()[0]["quantity"]), 1.5)

    def test_live_trader_places_and_closes_order_when_enabled(self):
        class FakeAdapter:
            def fetch_free_balance(self, asset):
                return 10000.0

            def estimate_slippage(self, symbol, side, quantity, reference_price):
                return SlippageEstimate(
                    symbol=symbol,
                    reference_price=reference_price,
                    expected_price=reference_price,
                    slippage_pct=0.0005,
                )

            def place_market_order(self, symbol, side, quantity):
                price = 100.2 if side == "buy" else 104.8
                return {
                    "exchange_order_id": f"ex-{side}",
                    "status": "closed",
                    "price": price,
                    "average_price": price,
                    "filled_qty": quantity,
                    "raw": {"id": f"ex-{side}"},
                }

        trader = LiveTrader(
            self.storage,
            exchange=FakeAdapter(),
            enabled=True,
            slippage_guard=SlippageGuard(max_slippage_pct=0.001),
        )
        opened = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.8,
            "test live open",
            quantity=1.5,
        )
        self.assertIsNotNone(opened)
        self.assertFalse(opened["dry_run"])

        closed = trader.execute_close("BTC/USDT", 105.0, "take_profit")
        self.assertIsNotNone(closed)
        self.assertFalse(closed["dry_run"])
        self.assertTrue(closed["is_full_close"])

        with self.storage._conn() as conn:
            order_count = conn.execute(
                "SELECT COUNT(*) AS c FROM orders"
            ).fetchone()["c"]
            event_count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type IN ('live_open', 'live_close')"
            ).fetchone()["c"]
        self.assertEqual(order_count, 2)
        self.assertEqual(event_count, 2)

    def test_live_trader_handles_exchange_error_during_open(self):
        class FakeAdapter:
            def fetch_free_balance(self, asset):
                return 10000.0

            def estimate_slippage(self, symbol, side, quantity, reference_price):
                raise RuntimeError("slippage feed offline")

        trader = LiveTrader(
            self.storage,
            exchange=FakeAdapter(),
            enabled=True,
            slippage_guard=SlippageGuard(max_slippage_pct=0.001),
        )
        result = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.8,
            "test exchange failure",
            quantity=1.0,
        )
        self.assertIsNone(result)
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT event_type, payload_json FROM execution_events ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(row["event_type"], "live_open_rejected")
        self.assertIn("exchange error", row["payload_json"])

    def test_live_trader_handles_exchange_error_during_close(self):
        class FakeAdapter:
            def fetch_free_balance(self, asset):
                return 10000.0

            def estimate_slippage(self, symbol, side, quantity, reference_price):
                if side == "BUY":
                    return SlippageEstimate(
                        symbol=symbol,
                        reference_price=reference_price,
                        expected_price=reference_price,
                        slippage_pct=0.0005,
                    )
                raise RuntimeError("order book unavailable")

            def place_market_order(self, symbol, side, quantity):
                return {
                    "exchange_order_id": f"ex-{side}",
                    "status": "closed",
                    "price": 100.0,
                    "average_price": 100.0,
                    "filled_qty": quantity,
                    "raw": {"id": f"ex-{side}"},
                }

        trader = LiveTrader(
            self.storage,
            exchange=FakeAdapter(),
            enabled=True,
            slippage_guard=SlippageGuard(max_slippage_pct=0.001),
        )
        opened = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.8,
            "seed position",
            quantity=1.0,
        )
        self.assertIsNotNone(opened)
        closed = trader.execute_close("BTC/USDT", 101.0, "test exchange failure")
        self.assertIsNone(closed)
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT event_type, payload_json FROM execution_events ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(row["event_type"], "live_close_rejected")
        self.assertIn("exchange error", row["payload_json"])

    def test_live_trader_falls_back_to_limit_order_and_polls_fill(self):
        class FakeAdapter:
            def __init__(self):
                self.fetch_count = 0

            def fetch_free_balance(self, asset):
                return 10000.0

            def estimate_slippage(self, symbol, side, quantity, reference_price):
                return SlippageEstimate(
                    symbol=symbol,
                    reference_price=reference_price,
                    expected_price=reference_price * 1.01,
                    slippage_pct=0.01,
                )

            def place_limit_order(self, symbol, side, quantity, price):
                return {
                    "exchange_order_id": "limit-1",
                    "status": "open",
                    "price": price,
                    "average_price": 0.0,
                    "filled_qty": 0.0,
                    "raw": {"id": "limit-1"},
                }

            def fetch_order(self, exchange_order_id, symbol):
                self.fetch_count += 1
                return {
                    "exchange_order_id": exchange_order_id,
                    "status": "closed" if self.fetch_count >= 1 else "open",
                    "price": 100.1,
                    "average_price": 100.1,
                    "filled_qty": 1.0,
                    "raw": {"id": exchange_order_id},
                }

            def cancel_order(self, exchange_order_id, symbol):
                return {"exchange_order_id": exchange_order_id, "status": "canceled"}

        trader = LiveTrader(
            self.storage,
            exchange=FakeAdapter(),
            enabled=True,
            slippage_guard=SlippageGuard(max_slippage_pct=0.001),
            order_poll_interval_seconds=0,
            limit_order_timeout_seconds=1,
            sleep_fn=lambda _: None,
        )
        opened = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.8,
            "limit fallback",
            quantity=1.0,
        )
        self.assertIsNotNone(opened)
        self.assertEqual(opened["execution_type"], "LIMIT")
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT order_type, status FROM orders ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(row["order_type"], "LIMIT")
        self.assertEqual(row["status"], "FILLED")

    def test_live_trader_cancels_limit_order_when_timeout_expires(self):
        class FakeAdapter:
            def __init__(self):
                self.cancelled = False

            def fetch_free_balance(self, asset):
                return 10000.0

            def estimate_slippage(self, symbol, side, quantity, reference_price):
                return SlippageEstimate(
                    symbol=symbol,
                    reference_price=reference_price,
                    expected_price=reference_price * 1.01,
                    slippage_pct=0.01,
                )

            def place_limit_order(self, symbol, side, quantity, price):
                return {
                    "exchange_order_id": "limit-timeout",
                    "status": "open",
                    "price": price,
                    "average_price": 0.0,
                    "filled_qty": 0.0,
                    "raw": {"id": "limit-timeout"},
                }

            def fetch_order(self, exchange_order_id, symbol):
                return {
                    "exchange_order_id": exchange_order_id,
                    "status": "open",
                    "price": 100.1,
                    "average_price": 0.0,
                    "filled_qty": 0.0,
                    "raw": {"id": exchange_order_id},
                }

            def cancel_order(self, exchange_order_id, symbol):
                self.cancelled = True
                return {"exchange_order_id": exchange_order_id, "status": "canceled"}

        adapter = FakeAdapter()
        trader = LiveTrader(
            self.storage,
            exchange=adapter,
            enabled=True,
            slippage_guard=SlippageGuard(max_slippage_pct=0.001),
            order_poll_interval_seconds=0,
            limit_order_timeout_seconds=0,
            limit_order_retry_count=0,
            sleep_fn=lambda _: None,
        )
        opened = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.8,
            "limit timeout",
            quantity=1.0,
        )
        self.assertIsNone(opened)
        self.assertTrue(adapter.cancelled)
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT status FROM orders ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(row["status"], "CANCELLED")

    def test_live_trader_records_partial_limit_fill_and_cancels_residual(self):
        class FakeAdapter:
            def __init__(self):
                self.cancelled = False

            def fetch_free_balance(self, asset):
                return 10000.0

            def estimate_slippage(self, symbol, side, quantity, reference_price):
                return SlippageEstimate(
                    symbol=symbol,
                    reference_price=reference_price,
                    expected_price=reference_price * 1.01,
                    slippage_pct=0.01,
                )

            def place_limit_order(self, symbol, side, quantity, price):
                return {
                    "exchange_order_id": "limit-partial",
                    "status": "open",
                    "price": price,
                    "average_price": 100.1,
                    "filled_qty": 0.4,
                    "remaining_qty": quantity - 0.4,
                    "raw": {"id": "limit-partial"},
                }

            def fetch_order(self, exchange_order_id, symbol):
                return {
                    "exchange_order_id": exchange_order_id,
                    "status": "open",
                    "price": 100.1,
                    "average_price": 100.1,
                    "filled_qty": 0.4,
                    "remaining_qty": 0.6,
                    "raw": {"id": exchange_order_id},
                }

            def cancel_order(self, exchange_order_id, symbol):
                self.cancelled = True
                return {"exchange_order_id": exchange_order_id, "status": "canceled"}

        adapter = FakeAdapter()
        trader = LiveTrader(
            self.storage,
            exchange=adapter,
            enabled=True,
            slippage_guard=SlippageGuard(max_slippage_pct=0.001),
            order_poll_interval_seconds=0,
            limit_order_timeout_seconds=0,
            limit_order_retry_count=0,
            sleep_fn=lambda _: None,
        )
        opened = trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.8,
            "partial limit",
            quantity=1.0,
        )
        self.assertIsNotNone(opened)
        self.assertFalse(opened["is_full_fill"])
        self.assertAlmostEqual(opened["quantity"], 0.4, places=6)
        self.assertTrue(adapter.cancelled)

        with self.storage._conn() as conn:
            order = conn.execute(
                "SELECT status FROM orders ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            trade = conn.execute(
                "SELECT quantity, initial_quantity FROM trades ORDER BY entry_time DESC LIMIT 1"
            ).fetchone()
            position = conn.execute(
                "SELECT quantity FROM positions WHERE symbol = ?",
                ("BTC/USDT",),
            ).fetchone()
        self.assertEqual(order["status"], "PARTIALLY_FILLED")
        self.assertAlmostEqual(float(trade["quantity"]), 0.4, places=6)
        self.assertAlmostEqual(float(trade["initial_quantity"]), 0.4, places=6)
        self.assertAlmostEqual(float(position["quantity"]), 0.4, places=6)

    def test_okx_exchange_adapter_summarizes_order_book_depth(self):
        class FakeExchange:
            def fetch_order_book(self, symbol, limit=20):
                return {
                    "bids": [[100.0, 2.0], [99.0, 1.0]],
                    "asks": [[101.0, 1.5], [102.0, 2.0]],
                }

        adapter = OKXExchangeAdapter(exchange=FakeExchange())
        snapshot = adapter.summarize_order_book_depth("BTC/USDT", depth=2)

        self.assertAlmostEqual(snapshot.bid_notional_top5, 299.0, places=6)
        self.assertAlmostEqual(snapshot.ask_notional_top5, 355.5, places=6)
        self.assertAlmostEqual(snapshot.bid_ask_spread_pct, 0.009950248756218905, places=12)
        self.assertAlmostEqual(snapshot.large_order_net_notional, -4.0, places=6)
