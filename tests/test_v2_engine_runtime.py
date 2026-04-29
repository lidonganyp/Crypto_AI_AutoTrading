import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import get_settings
from core.models import (
    MarketRegime,
    PredictionResult,
    ResearchInsight,
    SuggestedAction,
    SignalDirection,
)
from core.storage import Storage
from strategy.model_trainer import model_path_for_symbol
from tests.v2_architecture_support import V2ArchitectureTestCase, make_candles


class V2EngineRuntimeTests(V2ArchitectureTestCase):
    def test_engine_run_position_guard_only_manages_existing_positions(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        now = datetime.now(timezone.utc).isoformat()
        engine.storage.insert_trade(
            {
                "id": "guard-1",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        managed = []
        engine._manage_open_positions = lambda now, positions, account: managed.append(
            [position["symbol"] for position in positions]
        ) or 1

        result = engine.run_position_guard()

        self.assertEqual(result["checked_positions"], 1)
        self.assertEqual(result["closed_positions"], 1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(managed, [["BTC/USDT"]])

    def test_engine_live_account_state_uses_exchange_quote_balance(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 110.0

        class FakeExchangeAdapter:
            def __init__(self, *args, **kwargs):
                pass

            def fetch_total_balance(self, asset: str):
                return 5000.0 if asset == "USDT" else 0.0

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "live"
        settings.app.allow_live_orders = False

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), patch.object(
            engine_module,
            "OKXExchangeAdapter",
            FakeExchangeAdapter,
        ):
            engine = engine_module.CryptoAIV2Engine(settings)

        now = datetime.now(timezone.utc).isoformat()
        engine.storage.insert_trade(
            {
                "id": "live-1",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )

        account = engine._account_state(datetime.now(timezone.utc), engine.storage.get_positions())

        self.assertAlmostEqual(account.equity, 5110.0, places=6)
        self.assertAlmostEqual(account.drawdown_pct, 0.0, places=6)

    def test_snapshot_runtime_uses_market_depth_for_adaptive_liquidity_in_paper_mode(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, limit=240):
                base = 100 if timeframe == "1h" else 120 if timeframe == "4h" else 140
                return make_candles(limit, base)

            def fetch_funding_rate(self, symbol):
                return None

            def summarize_order_book_depth(self, symbol, depth=5):
                return SimpleNamespace(
                    symbol=symbol,
                    bid_notional_top5=15000.0,
                    ask_notional_top5=15000.0,
                    bid_ask_spread_pct=0.001,
                    depth_imbalance=0.0,
                    large_bid_notional=4000.0,
                    large_ask_notional=3800.0,
                    large_order_net_notional=200.0,
                )

        class FakeSentiment:
            def __init__(self, storage, settings=None):
                self.storage = storage

            def get_latest_sentiment(self, symbol="BTC/USDT"):
                return {"value": 55, "summary": "neutral"}

        class FakeResearch:
            def __init__(self, settings, clients=None):
                pass

            def analyze(self, **kwargs):
                return ResearchInsight(
                    symbol=kwargs["symbol"],
                    market_regime=MarketRegime.UNKNOWN,
                    sentiment_score=0.0,
                    confidence=0.6,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                )

        class FakePredictor:
            def __init__(self, model_path, enable_fallback=True):
                self.model_path = model_path

            def predict(self, snapshot):
                return PredictionResult(
                    symbol=snapshot.symbol,
                    up_probability=0.72,
                    feature_count=len(snapshot.values),
                    model_version="test-model",
                )

        class FakeNews:
            def __init__(self, settings=None):
                self.settings = settings

            def get_summary(self, symbol):
                return SimpleNamespace(
                    summary="neutral",
                    sources=["CoinDesk"],
                    coverage_score=0.5,
                    service_health_score=1.0,
                )

        class FakeMacro:
            def get_summary(self, fear_greed=None):
                return SimpleNamespace(summary="macro ok")

        class FakeOnchain:
            def __init__(self, settings=None):
                pass

            def get_summary(self, symbol):
                return SimpleNamespace(summary="neutral", netflow_score=0.0, whale_score=0.0)

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.strategy.min_liquidity_ratio = 0.80
        settings.strategy.adaptive_liquidity_floor_min_ratio = 0.35
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "SentimentCollector", FakeSentiment), \
             patch.object(engine_module, "NewsService", FakeNews), \
             patch.object(engine_module, "MacroService", FakeMacro), \
             patch.object(engine_module, "OnchainService", FakeOnchain), \
             patch.object(engine_module, "ResearchLLMAnalyzer", FakeResearch), \
             patch.object(engine_module, "XGBoostPredictor", FakePredictor):
            engine = engine_module.CryptoAIV2Engine(settings)

        now = datetime.now(timezone.utc)
        for idx in range(8):
            engine.storage.insert_feature_snapshot(
                {
                    "symbol": "BTC/USDT",
                    "timeframe": "4h",
                    "timestamp": (now - timedelta(hours=idx)).isoformat(),
                    "features": {"volume_ratio_1h": 0.5},
                    "valid": True,
                }
            )

        snapshot = engine._prepare_symbol_snapshot("BTC/USDT", now, include_blocked=True)
        self.assertIsNotNone(snapshot)
        features = snapshot[0]
        self.assertIn("adaptive_min_liquidity_ratio", features.values)
        self.assertLess(features.values["adaptive_min_liquidity_ratio"], 0.5)
        self.assertAlmostEqual(features.values["bid_ask_spread_pct"], 0.001, places=6)

    def test_engine_account_state_counts_partial_realized_pnl_in_equity(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 110.0

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        opened = engine.executor.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.9,
            "seed",
            quantity=1.0,
        )
        self.assertIsNotNone(opened)
        partial = engine.executor.execute_close(
            "BTC/USDT",
            110.0,
            "take_profit",
            close_qty=0.5,
        )
        self.assertIsNotNone(partial)
        self.assertFalse(partial["is_full_close"])

        account = engine._account_state(datetime.now(timezone.utc), engine.storage.get_positions())

        fee_pct = float(engine.executor.ESTIMATED_FEE_PCT)
        expected_equity = (
            10000.0
            - (100.0 * 1.0 * fee_pct)
            + (10.0 * 0.5 - (110.0 * 0.5 * fee_pct))
            + (10.0 * 0.5)
        )
        self.assertAlmostEqual(account.equity, expected_equity, places=6)

    def test_analysis_runtime_continues_after_abnormal_move_and_refreshes_cooldown(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult, SignalDirection

        settings = get_settings().model_copy(deep=True)
        cycle_now = datetime(2026, 3, 28, tzinfo=timezone.utc)
        state = {"cooldown_until": None}
        prepared_symbols = []
        persisted_symbols = []
        risk_accounts = []

        def build_account() -> AccountState:
            return AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
                cooldown_until=state["cooldown_until"],
                circuit_breaker_active=False,
            )

        class FakeRisk:
            def can_open_position(
                self,
                account,
                positions,
                symbol,
                atr,
                entry_price,
                liquidity_ratio,
                consecutive_wins=0,
                consecutive_losses=0,
            ):
                risk_accounts.append((symbol, account.cooldown_until))
                return RiskCheckResult(
                    allowed=account.cooldown_until is None,
                    reason="" if account.cooldown_until is None else "cooldown active",
                    allowed_position_value=100.0 if account.cooldown_until is None else 0.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.8

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(
                        direction=SignalDirection.LONG,
                        final_score=0.9,
                    ),
                    SimpleNamespace(
                        should_execute=bool(risk_result.allowed),
                        reason=risk_result.reason or "ok",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=risk_result.allowed_position_value,
                        final_score=0.9,
                    ),
                )

        class FakeExecutor:
            def __init__(self):
                self.calls = []

            def execute_open(self, **kwargs):
                self.calls.append(kwargs["symbol"])
                return {
                    "price": kwargs["price"],
                    "dry_run": False,
                }

        class FakeNotifier:
            def notify_analysis_result(self, *args, **kwargs):
                return None

            def notify_trade_open(self, *args, **kwargs):
                return None

        executor = FakeExecutor()

        def prepare_symbol_snapshot(symbol, now, include_blocked=True):
            prepared_symbols.append(symbol)
            return (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 1.0,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.2,
                    confidence=0.7,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.82,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(reasons=[], raw_action="OPEN_LONG"),
            )

        def detect_abnormal_move(symbol, now):
            if symbol == "BTC/USDT":
                state["cooldown_until"] = now + timedelta(minutes=30)
                return True
            return False

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=FakeRisk(),
            executor=executor,
            notifier=FakeNotifier(),
            prepare_symbol_snapshot=prepare_symbol_snapshot,
            detect_abnormal_move=detect_abnormal_move,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda symbol, *args, **kwargs: persisted_symbols.append(symbol),
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: build_account(),
            get_circuit_breaker_reason=lambda: "",
        )

        result = service.run_active_symbols(
            now=cycle_now,
            active_symbols=["BTC/USDT", "ETH/USDT"],
            positions=[],
            account=build_account(),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(prepared_symbols, ["BTC/USDT", "ETH/USDT"])
        self.assertEqual(persisted_symbols, ["ETH/USDT"])
        self.assertEqual(executor.calls, [])
        self.assertEqual(result["opened_positions"], 0)
        self.assertEqual(risk_accounts, [("ETH/USDT", state["cooldown_until"])])

    def test_analysis_runtime_builds_correlation_price_data_from_storage_variants(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        seen = {}

        class FakeRisk:
            def can_open_position(
                self,
                account,
                positions,
                symbol,
                atr,
                entry_price,
                liquidity_ratio,
                consecutive_wins=0,
                consecutive_losses=0,
                correlation_price_data=None,
            ):
                seen["symbol"] = symbol
                seen["price_data"] = correlation_price_data
                return RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )

        service = AnalysisRuntimeService(
            storage=self.storage,
            settings=get_settings().model_copy(deep=True),
            decision_engine=SimpleNamespace(),
            risk=FakeRisk(),
            executor=SimpleNamespace(),
            notifier=SimpleNamespace(),
            prepare_symbol_snapshot=lambda *args, **kwargs: None,
            detect_abnormal_move=lambda *args, **kwargs: False,
            evaluate_ab_test=lambda *args, **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda *args, **kwargs: "",
            get_positions=lambda: [],
            account_state=lambda now, positions: None,
            get_circuit_breaker_reason=lambda: "",
        )
        self.storage.insert_ohlcv("ETH/USDT:USDT", "4h", make_candles(120, 100))
        self.storage.insert_ohlcv("BTC/USDT:USDT", "4h", make_candles(120, 200))

        service._can_open_position(
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.2,
                open_positions=1,
            ),
            positions=[
                {
                    "symbol": "BTC/USDT",
                    "direction": "LONG",
                    "entry_price": 200.0,
                    "quantity": 1.0,
                }
            ],
            symbol="ETH/USDT",
            atr=2.0,
            entry_price=100.0,
            liquidity_ratio=1.0,
            consecutive_wins=0,
            consecutive_losses=0,
            performance_snapshot=None,
        )

        self.assertEqual(seen["symbol"], "ETH/USDT")
        self.assertEqual(set(seen["price_data"].keys()), {"ETH/USDT", "BTC/USDT"})
        self.assertEqual(len(seen["price_data"]["ETH/USDT"]), 120)
        self.assertEqual(len(seen["price_data"]["BTC/USDT"]), 120)

    def test_engine_maybe_rebuild_execution_pool_respects_rebuild_interval(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.risk.execution_pool_rebuild_interval_hours = 24
        engine = engine_module.CryptoAIV2Engine(settings)
        now = datetime(2026, 3, 26, tzinfo=timezone.utc)
        engine.storage.set_state(
            engine.EXECUTION_POOL_LAST_REBUILD_AT_STATE_KEY,
            now.isoformat(),
        )

        with patch.object(engine, "rebuild_execution_symbols") as mocked_rebuild:
            result = engine._maybe_rebuild_execution_pool(
                now + timedelta(hours=1),
                ["BTC/USDT"],
            )

        self.assertIsNone(result)
        mocked_rebuild.assert_not_called()

    def test_engine_run_loop_model_maintenance_triggers_for_broken_models(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        engine.storage.set_json_state(
            engine.BROKEN_MODEL_SYMBOLS_STATE_KEY,
            {"BTC/USDT": {"failure_kind": "model_load_failed"}},
        )
        calls = []
        engine._train_models_if_due = lambda now, force, reason, record_skips=True: calls.append(
            (force, reason, record_skips)
        ) or []

        engine._run_loop_model_maintenance(datetime.now(timezone.utc))

        self.assertEqual(calls, [(False, "loop_auto", False)])

    def test_engine_predictor_cache_refreshes_after_model_file_changes(self):
        import os
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.model.xgboost_model_path = str(
            Path(self.db_path).parent / "models" / "xgboost_runtime_test.json"
        )
        engine = engine_module.CryptoAIV2Engine(settings)
        model_path = model_path_for_symbol(engine._predictor_base_path, "BTC/USDT")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("version-one", encoding="utf-8")

        created_payloads = []

        class FakePredictor:
            def __init__(self, loaded_model_path, enable_fallback=True):
                self.model_path = loaded_model_path
                self.payload = Path(loaded_model_path).read_text(encoding="utf-8")
                created_payloads.append(self.payload)

        engine._predictor_factory = FakePredictor

        first = engine._predictor_for_symbol("BTC/USDT")

        previous_mtime_ns = model_path.stat().st_mtime_ns
        model_path.write_text("version-two-updated", encoding="utf-8")
        os.utime(
            model_path,
            ns=(previous_mtime_ns + 1_000_000, previous_mtime_ns + 1_000_000),
        )

        second = engine._predictor_for_symbol("BTC/USDT")

        self.assertIsNot(first, second)
        self.assertEqual(first.payload, "version-one")
        self.assertEqual(second.payload, "version-two-updated")
        self.assertEqual(created_payloads, ["version-one", "version-two-updated"])

    def test_engine_predictor_uses_latest_trained_model_path(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        trained_model_path = Path(self.db_path).parent / "trained_BTC_model.json"
        trained_model_path.write_text("trained-model", encoding="utf-8")
        engine.storage.insert_training_run(
            {
                "symbol": "BTC/USDT",
                "rows": settings.training.minimum_training_rows,
                "feature_count": 10,
                "positives": 5,
                "negatives": 5,
                "model_path": str(trained_model_path),
                "trained_with_xgboost": True,
                "holdout_accuracy": 0.6,
            }
        )
        created_paths = []

        class FakePredictor:
            def __init__(self, loaded_model_path, enable_fallback=True):
                created_paths.append(loaded_model_path)

        engine._predictor_factory = FakePredictor

        engine._predictor_for_symbol("BTC/USDT")

        self.assertEqual(created_paths, [str(trained_model_path)])

    def test_engine_remove_execution_symbols_clears_cached_predictors(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        engine.storage.set_json_state(engine.EXECUTION_SYMBOLS_STATE_KEY, ["BTC/USDT"])
        engine._predictors_by_symbol["BTC/USDT"] = object()
        engine._predictor_signatures_by_symbol["BTC/USDT"] = ("main", 1, 1)
        engine._challenger_predictors_by_symbol["BTC/USDT"] = object()
        engine._challenger_predictor_signatures_by_symbol["BTC/USDT"] = ("challenger", 1, 1)

        result = engine.remove_execution_symbols(["BTC/USDT"])

        self.assertEqual(result["execution_symbols"], [])
        self.assertNotIn("BTC/USDT", engine._predictors_by_symbol)
        self.assertNotIn("BTC/USDT", engine._predictor_signatures_by_symbol)
        self.assertNotIn("BTC/USDT", engine._challenger_predictors_by_symbol)
        self.assertNotIn("BTC/USDT", engine._challenger_predictor_signatures_by_symbol)

    def test_engine_active_symbols_follow_latest_training_status_per_symbol(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        engine.storage.set_json_state(engine.EXECUTION_SYMBOLS_STATE_KEY, ["BTC/USDT"])

        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        btc_model = model_dir / "xgboost_v2_BTC_USDT.json"
        btc_model.write_text("{}", encoding="utf-8")

        engine.storage.insert_training_run(
            {
                "symbol": "BTC/USDT",
                "rows": settings.training.minimum_training_rows,
                "feature_count": 10,
                "positives": 5,
                "negatives": 5,
                "model_path": str(btc_model),
                "trained_with_xgboost": True,
                "holdout_accuracy": 0.6,
            }
        )
        engine.storage.insert_training_run(
            {
                "symbol": "BTC/USDT",
                "rows": settings.training.minimum_training_rows - 1,
                "feature_count": 10,
                "positives": 5,
                "negatives": 5,
                "model_path": "",
                "trained_with_xgboost": False,
                "holdout_accuracy": 0.0,
            }
        )

        self.assertEqual(engine.get_active_symbols(force_refresh=False), [])

    def test_engine_active_symbols_exclude_broken_model_symbols_until_artifact_changes(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        engine.storage.set_json_state(engine.EXECUTION_SYMBOLS_STATE_KEY, ["BTC/USDT"])

        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        btc_model = model_dir / "xgboost_v2_BTC_USDT.json"
        btc_model.write_text("{}", encoding="utf-8")
        signature = (
            str(btc_model),
            int(btc_model.stat().st_mtime_ns),
            int(btc_model.stat().st_size),
        )

        engine.storage.insert_training_run(
            {
                "symbol": "BTC/USDT",
                "rows": settings.training.minimum_training_rows,
                "feature_count": 10,
                "positives": 5,
                "negatives": 5,
                "model_path": str(btc_model),
                "trained_with_xgboost": True,
                "holdout_accuracy": 0.6,
            }
        )
        engine.storage.set_json_state(
            engine.BROKEN_MODEL_SYMBOLS_STATE_KEY,
            {
                "BTC/USDT": {
                    "model_path": str(btc_model),
                    "signature": list(signature),
                    "failure_kind": "model_load_failed",
                }
            },
        )

        self.assertEqual(engine.get_active_symbols(force_refresh=False), [])

        btc_model.write_text("{\"fixed\":true}", encoding="utf-8")
        self.assertEqual(engine.get_active_symbols(force_refresh=False), ["BTC/USDT"])

    def test_engine_prepare_symbol_snapshot_queues_broken_model_for_repair(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                base = 100 if timeframe == "1h" else 120 if timeframe == "4h" else 140
                return make_candles(240, base)

        class FakeSentiment:
            def __init__(self, storage, settings=None):
                self.storage = storage

            def get_latest_sentiment(self, symbol="BTC/USDT"):
                return {"value": 60, "summary": "bullish"}

        class FakeResearch:
            def __init__(self, settings, clients=None):
                pass

            def analyze(self, **kwargs):
                return ResearchInsight(
                    symbol=kwargs["symbol"],
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.2,
                    confidence=0.5,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                )

        class FakePredictor:
            def __init__(self, model_path, enable_fallback=True):
                self.model_path = Path(model_path)
                self.model_loaded = False
                self.load_failure_kind = "model_load_failed"
                self.load_error = "corrupt model"

            def predict(self, snapshot):
                return PredictionResult(
                    symbol=snapshot.symbol,
                    up_probability=0.82,
                    feature_count=len(snapshot.values),
                    model_version="fallback_v2",
                )

        class FakeRegimeDetector:
            def detect(self, candles, fear_greed=None):
                return SimpleNamespace(state="BULL_TREND")

        class FakeNews:
            def __init__(self, settings=None):
                self.settings = settings

            def get_summary(self, symbol):
                return SimpleNamespace(
                    summary=f"{symbol} news",
                    trending_symbols=[symbol],
                    sources=["CoinDesk"],
                    coverage_score=1.0,
                    service_health_score=1.0,
                )

        class FakeMacro:
            def get_summary(self, fear_greed=None):
                return SimpleNamespace(summary="macro ok", score=0.0, position_adjustment=1.0)

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "SentimentCollector", FakeSentiment), \
             patch.object(engine_module, "NewsService", FakeNews), \
             patch.object(engine_module, "MacroService", FakeMacro), \
             patch.object(engine_module, "ResearchLLMAnalyzer", FakeResearch), \
             patch.object(engine_module, "XGBoostPredictor", FakePredictor), \
             patch.object(engine_module, "MarketRegimeDetector", lambda: FakeRegimeDetector()):
            engine = engine_module.CryptoAIV2Engine(settings)

        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        model_path.write_text("{}", encoding="utf-8")
        engine.storage.insert_training_run(
            {
                "symbol": "BTC/USDT",
                "rows": settings.training.minimum_training_rows,
                "feature_count": 10,
                "positives": 5,
                "negatives": 5,
                "model_path": str(model_path),
                "trained_with_xgboost": True,
                "holdout_accuracy": 0.6,
            }
        )
        train_calls = []
        engine.trainer.train_symbol = lambda symbol: (
            train_calls.append(symbol),
            SimpleNamespace(
                symbol=symbol,
                rows=settings.training.minimum_training_rows,
                feature_count=10,
                positives=5,
                negatives=5,
                model_path=str(model_path),
                trained_with_xgboost=True,
                holdout_accuracy=0.61,
                reason="trained",
            ),
        )[1]
        engine.trainer.render_report = lambda summary, lang=None: f"trained {summary.symbol}"

        snapshot = engine._prepare_symbol_snapshot(
            "BTC/USDT",
            datetime.now(timezone.utc),
            include_blocked=True,
        )

        self.assertIsNone(snapshot)
        self.assertEqual(train_calls, [])
        broken_state = engine.storage.get_json_state(
            engine.BROKEN_MODEL_SYMBOLS_STATE_KEY,
            {},
        )
        self.assertIn("BTC/USDT", broken_state)
        self.assertEqual(
            broken_state["BTC/USDT"]["failure_kind"],
            "model_load_failed",
        )

    def test_engine_train_models_repairs_queued_broken_model(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.symbols = []
        engine = engine_module.CryptoAIV2Engine(settings)

        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        model_path.write_text("{}", encoding="utf-8")
        signature = list(engine._model_file_signature(model_path))
        engine.storage.insert_training_run(
            {
                "symbol": "BTC/USDT",
                "rows": settings.training.minimum_training_rows,
                "feature_count": 10,
                "positives": 5,
                "negatives": 5,
                "model_path": str(model_path),
                "trained_with_xgboost": True,
                "holdout_accuracy": 0.6,
            }
        )
        engine.storage.set_json_state(
            engine.BROKEN_MODEL_SYMBOLS_STATE_KEY,
            {
                "BTC/USDT": {
                    "model_path": str(model_path),
                    "signature": signature,
                    "failure_kind": "model_load_failed",
                }
            },
        )
        train_calls = []
        engine.trainer.train_symbol = lambda symbol: (
            train_calls.append(symbol),
            SimpleNamespace(
                symbol=symbol,
                rows=settings.training.minimum_training_rows,
                feature_count=10,
                positives=5,
                negatives=5,
                model_path=str(model_path),
                trained_with_xgboost=True,
                holdout_accuracy=0.61,
                reason="trained",
            ),
        )[1]
        engine.trainer.render_report = lambda summary, lang=None: f"trained {summary.symbol}"

        summaries = engine._train_models_if_due(
            datetime.now(timezone.utc),
            force=False,
            reason="scheduled",
        )

        self.assertEqual(train_calls, ["BTC/USDT"])
        self.assertEqual(len(summaries), 1)
        self.assertEqual(
            engine.storage.get_json_state(engine.BROKEN_MODEL_SYMBOLS_STATE_KEY, {}),
            {},
        )

    def test_engine_train_models_persists_walkforward_for_promoted_model(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        engine.get_execution_symbols = lambda: ["BTC/USDT"]
        engine._broken_model_symbols = lambda: {}
        model_path = Path(self.db_path).parent / "models" / "xgboost_v2_BTC_USDT.json"
        backup_model_path = Path(self.db_path).parent / "models" / "backups" / "prev_BTC.json"
        backup_model_path.parent.mkdir(parents=True, exist_ok=True)
        backup_model_path.write_text("stable-model", encoding="utf-8")
        backup_meta_path = backup_model_path.with_suffix(".meta.json")
        backup_meta_path.write_text("{}", encoding="utf-8")
        engine.trainer.train_symbol = lambda symbol: SimpleNamespace(
            symbol=symbol,
            rows=settings.training.minimum_training_rows,
            feature_count=10,
            positives=5,
            negatives=5,
            model_path=str(model_path),
            active_model_path=str(model_path),
            challenger_model_path="",
            trained_with_xgboost=True,
            promoted_to_active=True,
            promotion_status="promoted",
            promotion_reason="candidate_higher_walkforward_return",
            holdout_rows=20,
            holdout_accuracy=0.62,
            holdout_logloss=0.45,
            candidate_holdout_accuracy=0.62,
            candidate_holdout_logloss=0.45,
            incumbent_holdout_accuracy=0.60,
            incumbent_holdout_logloss=0.50,
            candidate_walkforward_summary={
                "symbol": symbol,
                "total_splits": 2,
                "avg_win_rate": 55.0,
                "avg_trade_return_pct": 0.8,
                "total_return_pct": 1.6,
                "profit_factor": 1.3,
                "sharpe_like": 0.5,
            },
            candidate_walkforward_splits=[
                {
                    "train_rows": 300,
                    "test_rows": 60,
                    "win_rate": 55.0,
                    "avg_trade_return_pct": 0.8,
                    "total_return_pct": 1.6,
                }
            ],
            incumbent_walkforward_summary={
                "symbol": symbol,
                "total_splits": 2,
                "avg_win_rate": 52.0,
                "avg_trade_return_pct": 0.5,
                "total_return_pct": 1.0,
                "profit_factor": 1.1,
                "sharpe_like": 0.3,
            },
            top_features=["f1"],
            dataset_start_timestamp_ms=1,
            dataset_end_timestamp_ms=2,
            dataset_start_at="1970-01-01T00:00:00+00:00",
            dataset_end_at="1970-01-01T00:00:00+00:00",
            previous_active_backup_path=str(backup_model_path),
            previous_active_backup_meta_path=str(backup_meta_path),
            reason="trained",
        )
        engine.trainer.render_report = lambda summary, lang=None: f"trained {summary.symbol}"
        engine.walkforward.render_report = lambda result, lang=None: f"wf {result['symbol']}"

        summaries = engine._train_models_if_due(
            datetime.now(timezone.utc),
            force=True,
            reason="manual",
        )

        self.assertEqual(len(summaries), 1)
        with engine.storage._conn() as conn:
            walkforward_count = conn.execute(
                "SELECT COUNT(*) AS c FROM walkforward_runs"
            ).fetchone()["c"]
            walkforward_report_count = conn.execute(
                "SELECT COUNT(*) AS c FROM report_artifacts WHERE report_type='walkforward'"
            ).fetchone()["c"]
        self.assertEqual(walkforward_count, 1)
        self.assertEqual(walkforward_report_count, 1)
        observation_state = engine.storage.get_json_state(
            engine.MODEL_PROMOTION_OBSERVATION_STATE_KEY,
            {},
        )
        self.assertIn("BTC/USDT", observation_state)

    def test_engine_train_models_registers_canary_candidate_for_stronger_challenger(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        engine.get_execution_symbols = lambda: ["BTC/USDT"]
        engine._broken_model_symbols = lambda: {}
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        challenger_model_path = model_dir / "xgboost_challenger_BTC_USDT.json"
        active_model_path.write_text("stable-model", encoding="utf-8")
        challenger_model_path.write_text("candidate-model", encoding="utf-8")
        engine.trainer.train_symbol = lambda symbol: SimpleNamespace(
            symbol=symbol,
            rows=settings.training.minimum_training_rows,
            feature_count=10,
            positives=5,
            negatives=5,
            model_path=str(challenger_model_path),
            model_id="candidate-model-id",
            active_model_path=str(active_model_path),
            active_model_id="active-model-id",
            challenger_model_path=str(challenger_model_path),
            challenger_model_id="candidate-model-id",
            incumbent_model_id="active-model-id",
            trained_with_xgboost=True,
            promoted_to_active=False,
            promotion_status="canary_pending",
            promotion_reason="candidate_higher_walkforward_return",
            holdout_rows=20,
            holdout_accuracy=0.60,
            holdout_logloss=0.48,
            candidate_holdout_accuracy=0.64,
            candidate_holdout_logloss=0.42,
            incumbent_holdout_accuracy=0.60,
            incumbent_holdout_logloss=0.48,
            candidate_walkforward_summary={
                "symbol": symbol,
                "total_splits": 2,
                "avg_win_rate": 56.0,
                "avg_trade_return_pct": 0.9,
                "total_return_pct": 1.8,
                "profit_factor": 1.35,
                "sharpe_like": 0.55,
            },
            candidate_walkforward_splits=[],
            top_features=["f1"],
            dataset_start_timestamp_ms=1,
            dataset_end_timestamp_ms=2,
            dataset_start_at="1970-01-01T00:00:00+00:00",
            dataset_end_at="1970-01-01T00:00:00+00:00",
            previous_active_backup_path="",
            previous_active_backup_meta_path="",
            reason="trained",
        )
        engine.trainer.render_report = lambda summary, lang=None: f"trained {summary.symbol}"

        engine._train_models_if_due(
            datetime.now(timezone.utc),
            force=True,
            reason="manual",
        )

        candidate_state = engine.storage.get_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {},
        )
        self.assertIn("BTC/USDT", candidate_state)
        self.assertEqual(candidate_state["BTC/USDT"]["status"], "shadow")
        self.assertEqual(
            engine.storage.get_json_state(engine.MODEL_PROMOTION_OBSERVATION_STATE_KEY, {}),
            {},
        )
        with engine.storage._conn() as conn:
            walkforward_count = conn.execute(
                "SELECT COUNT(*) AS c FROM walkforward_runs"
            ).fetchone()["c"]
            registry_count = conn.execute(
                "SELECT COUNT(*) AS c FROM model_registry WHERE symbol='BTC/USDT'"
            ).fetchone()["c"]
        self.assertEqual(walkforward_count, 0)
        self.assertGreaterEqual(registry_count, 2)

    def test_engine_register_promotion_candidate_expands_canary_for_high_volatility_and_long_holds(self):
        import core.engine as engine_module

        def make_high_vol_candles(count: int) -> list[dict]:
            candles = []
            price = 100.0
            for idx in range(count):
                open_price = price
                close_price = open_price * (1.06 if idx % 2 == 0 else 0.95)
                high_price = max(open_price, close_price) * 1.04
                low_price = min(open_price, close_price) * 0.96
                candles.append(
                    {
                        "timestamp": 1700000000000 + idx * 4 * 3600000,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": 1000 + idx * 10,
                    }
                )
                price = close_price
            return candles

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.risk.execution_symbol_min_samples = 6
        engine = engine_module.CryptoAIV2Engine(settings)

        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        challenger_model_path = model_dir / "xgboost_challenger_BTC_USDT.json"
        active_model_path.write_text("stable-model", encoding="utf-8")
        challenger_model_path.write_text("candidate-model", encoding="utf-8")
        engine.storage.insert_ohlcv("BTC/USDT", "4h", make_high_vol_candles(80))
        for idx, holding_hours in enumerate((24.0, 36.0, 42.0), start=1):
            engine.storage.insert_pnl_ledger_entry(
                {
                    "trade_id": f"historic-hold-{idx}",
                    "symbol": "BTC/USDT",
                    "event_type": "close",
                    "event_time": (
                        datetime.now(timezone.utc) - timedelta(hours=idx)
                    ).isoformat(),
                    "quantity": 1.0,
                    "notional_value": 100.0,
                    "reference_price": 100.0,
                    "fill_price": 102.0,
                    "gross_pnl": 2.0,
                    "fee_cost": 0.1,
                    "slippage_cost": 0.0,
                    "net_pnl": 1.9,
                    "net_return_pct": 1.9,
                    "holding_hours": holding_hours,
                    "model_id": "champion",
                }
            )

        summary = SimpleNamespace(
            symbol="BTC/USDT",
            model_path=str(challenger_model_path),
            model_id="challenger",
            active_model_path=str(active_model_path),
            active_model_id="champion",
            challenger_model_path=str(challenger_model_path),
            challenger_model_id="challenger",
            incumbent_model_id="champion",
            promotion_status="canary_pending",
            promotion_reason="candidate_higher_walkforward_expectancy",
            candidate_holdout_accuracy=0.64,
            holdout_rows=40,
            candidate_walkforward_summary={
                "symbol": "BTC/USDT",
                "total_splits": 3,
                "avg_win_rate": 56.0,
                "avg_trade_return_pct": 1.1,
                "expectancy_pct": 1.1,
                "total_return_pct": 3.3,
                "profit_factor": 1.4,
                "max_drawdown_pct": 1.2,
                "trade_count": 6,
            },
        )

        engine._register_promotion_candidate(
            "BTC/USDT",
            summary,
            datetime.now(timezone.utc),
        )

        candidate = engine.storage.get_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {},
        )["BTC/USDT"]
        self.assertEqual(candidate["adaptive_requirements_source"], "adaptive")
        self.assertGreater(
            candidate["min_shadow_evaluations"],
            engine._promotion_shadow_min_evaluations(),
        )
        self.assertGreater(
            candidate["min_live_evaluations"],
            engine._promotion_live_min_evaluations(),
        )
        self.assertGreater(candidate["max_shadow_age_hours"], 72)
        self.assertGreater(candidate["max_live_age_hours"], 72)
        self.assertGreater(candidate["adaptive_volatility_pct"], 4.0)
        self.assertGreater(candidate["adaptive_reference_holding_hours"], 24.0)

    def test_engine_register_promoted_model_observation_compresses_window_for_low_volatility_and_short_holds(self):
        import core.engine as engine_module

        def make_low_vol_candles(count: int) -> list[dict]:
            candles = []
            price = 100.0
            for idx in range(count):
                open_price = price
                close_price = open_price * (1.0008 if idx % 2 == 0 else 0.9994)
                high_price = max(open_price, close_price) * 1.0015
                low_price = min(open_price, close_price) * 0.9985
                candles.append(
                    {
                        "timestamp": 1700000000000 + idx * 4 * 3600000,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": 1000 + idx * 5,
                    }
                )
                price = close_price
            return candles

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.risk.execution_symbol_min_samples = 6
        engine = engine_module.CryptoAIV2Engine(settings)

        model_dir = Path(self.db_path).parent / "models"
        backup_dir = model_dir / "backups"
        model_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        backup_model_path = backup_dir / "prev_BTC_USDT.json"
        backup_meta_path = backup_model_path.with_suffix(".meta.json")
        active_model_path.write_text("promoted-model", encoding="utf-8")
        backup_model_path.write_text("stable-model", encoding="utf-8")
        backup_meta_path.write_text("{}", encoding="utf-8")
        engine.storage.insert_ohlcv("BTC/USDT", "4h", make_low_vol_candles(80))

        summary = SimpleNamespace(
            symbol="BTC/USDT",
            model_path=str(active_model_path),
            model_id="challenger",
            active_model_path=str(active_model_path),
            active_model_id="challenger",
            incumbent_model_id="champion",
            previous_active_backup_path=str(backup_model_path),
            previous_active_backup_meta_path=str(backup_meta_path),
            candidate_holdout_accuracy=0.61,
            candidate_walkforward_summary={
                "symbol": "BTC/USDT",
                "total_splits": 3,
                "avg_win_rate": 54.0,
                "avg_trade_return_pct": 0.4,
                "expectancy_pct": 0.4,
                "total_return_pct": 1.2,
                "profit_factor": 1.2,
                "max_drawdown_pct": 0.9,
                "trade_count": 18,
            },
            recent_walkforward_baseline_summary={
                "symbol": "BTC/USDT",
                "history_count": 3,
                "avg_expectancy_pct": 0.35,
                "avg_profit_factor": 1.15,
                "avg_max_drawdown_pct": 0.8,
            },
            canary_live_net_avg_holding_hours=1.5,
        )

        engine._register_promoted_model_observation(
            "BTC/USDT",
            summary,
            datetime.now(timezone.utc),
        )

        observation = engine.storage.get_json_state(
            engine.MODEL_PROMOTION_OBSERVATION_STATE_KEY,
            {},
        )["BTC/USDT"]
        self.assertEqual(observation["adaptive_requirements_source"], "adaptive")
        self.assertLess(
            observation["min_evaluations"],
            max(4, int(settings.risk.execution_symbol_min_samples)),
        )
        self.assertLess(observation["max_observation_age_hours"], 72)
        self.assertLess(observation["adaptive_volatility_pct"], 0.8)
        self.assertAlmostEqual(
            observation["adaptive_reference_holding_hours"],
            1.5,
            places=6,
        )

    def test_engine_set_execution_symbols_registers_canary_candidate_for_added_symbol(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.symbols = ["ETH/USDT", "BTC/USDT"]
        settings.exchange.candidate_symbols = ["ETH/USDT", "BTC/USDT"]
        engine = engine_module.CryptoAIV2Engine(settings)
        engine.storage.set_json_state(engine.EXECUTION_SYMBOLS_STATE_KEY, ["ETH/USDT"])
        engine.market.fetch_historical_ohlcv = (
            lambda symbol, timeframe, since=None, limit=500: []
        )
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        challenger_model_path = model_dir / "xgboost_challenger_BTC_USDT.json"
        active_model_path.write_text("stable-model", encoding="utf-8")
        challenger_model_path.write_text("candidate-model", encoding="utf-8")

        engine.trainer.train_symbol = lambda symbol: SimpleNamespace(
            symbol=symbol,
            rows=settings.training.minimum_training_rows,
            feature_count=10,
            positives=5,
            negatives=5,
            model_path=str(challenger_model_path),
            model_id="candidate-model-id",
            active_model_path=str(active_model_path),
            active_model_id="active-model-id",
            challenger_model_path=str(challenger_model_path),
            challenger_model_id="candidate-model-id",
            incumbent_model_id="active-model-id",
            trained_with_xgboost=True,
            promoted_to_active=False,
            promotion_status="canary_pending",
            promotion_reason="candidate_higher_walkforward_return",
            holdout_rows=20,
            holdout_accuracy=0.60,
            candidate_holdout_accuracy=0.64,
            candidate_holdout_logloss=0.42,
            top_features=["f1"],
            dataset_start_timestamp_ms=1,
            dataset_end_timestamp_ms=2,
            dataset_start_at="1970-01-01T00:00:00+00:00",
            dataset_end_at="1970-01-01T00:00:00+00:00",
            previous_active_backup_path="",
            previous_active_backup_meta_path="",
            reason="trained",
        )
        engine.trainer.render_report = lambda summary, lang=None: f"trained {summary.symbol}"

        result = engine.set_execution_symbols(
            ["ETH/USDT", "BTC/USDT"],
            backfill_days=0,
            train_models=True,
            action="set",
        )

        candidate_state = engine.storage.get_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {},
        )
        self.assertEqual(result["execution_symbols"], ["ETH/USDT", "BTC/USDT"])
        self.assertIn("BTC/USDT", candidate_state)
        self.assertEqual(candidate_state["BTC/USDT"]["status"], "shadow")
        self.assertEqual(
            engine.storage.get_json_state(engine.MODEL_PROMOTION_OBSERVATION_STATE_KEY, {}),
            {},
        )

    def test_engine_observe_promoted_models_rolls_back_underperforming_model(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        active_model_path.write_text("new-model", encoding="utf-8")
        active_meta_path = active_model_path.with_suffix(".meta.json")
        active_meta_path.write_text("new-meta", encoding="utf-8")
        backup_model_path = model_dir / "backups" / "stable_BTC.json"
        backup_model_path.parent.mkdir(parents=True, exist_ok=True)
        backup_model_path.write_text("stable-model", encoding="utf-8")
        backup_meta_path = backup_model_path.with_suffix(".meta.json")
        backup_meta_path.write_text("stable-meta", encoding="utf-8")
        challenger_model_path = model_path_for_symbol(
            engine._challenger_predictor_base_path,
            "BTC/USDT",
        )

        promoted_at = datetime.now(timezone.utc) - timedelta(hours=10)
        engine.storage.set_json_state(
            engine.MODEL_PROMOTION_OBSERVATION_STATE_KEY,
            {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "promoted_at": promoted_at.isoformat(),
                    "active_model_path": str(active_model_path),
                    "active_model_id": "test",
                    "challenger_model_path": str(challenger_model_path),
                    "active_model_signature": list(engine._model_file_signature(active_model_path)),
                    "backup_model_path": str(backup_model_path),
                    "backup_model_id": "stable",
                    "backup_meta_path": str(backup_meta_path),
                    "baseline_holdout_accuracy": 0.62,
                    "baseline_objective_quality": 1.0,
                    "baseline_expectancy_pct": 0.60,
                    "baseline_profit_factor": 1.30,
                    "baseline_max_drawdown_pct": 0.50,
                    "baseline_trade_win_rate": 0.55,
                    "baseline_walkforward_summary": {
                        "total_return_pct": 1.6,
                        "expectancy_pct": 0.60,
                        "profit_factor": 1.30,
                        "max_drawdown_pct": 0.50,
                    },
                    "recent_walkforward_baseline_summary": {
                        "history_count": 3,
                        "avg_total_return_pct": 1.4,
                        "avg_expectancy_pct": 0.55,
                        "avg_profit_factor": 1.25,
                        "avg_max_drawdown_pct": 0.45,
                    },
                    "min_evaluations": 4,
                    "max_observation_age_hours": 72,
                    "status": "observing",
                    "training_metadata": {
                        "symbol": "BTC/USDT",
                        "model_path": str(active_model_path),
                        "active_model_path": str(active_model_path),
                        "challenger_model_path": "",
                        "model_id": "test",
                        "active_model_id": "test",
                        "incumbent_model_id": "stable",
                        "promotion_status": "promoted",
                        "promotion_reason": "candidate_higher_walkforward_return",
                    },
                }
            },
        )
        for idx, is_correct in enumerate((True, False, False, False), start=1):
            ts = (promoted_at + timedelta(hours=idx)).isoformat()
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "test",
                    "up_probability": 0.8,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT", "suggested_action": "HOLD"},
                    "decision": {
                        "pipeline_mode": "execution",
                        "final_score": 0.8,
                        "model_id": "test",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "execution",
                    "actual_up": bool(is_correct),
                    "predicted_up": True,
                    "is_correct": bool(is_correct),
                    "entry_close": 100.0,
                    "future_close": 101.0 if is_correct else 99.0,
                    "metadata": {},
                }
            )

        result = engine._observe_promoted_models(datetime.now(timezone.utc))

        self.assertEqual(result["rolled_back"], 1)
        self.assertEqual(active_model_path.read_text(encoding="utf-8"), "stable-model")
        self.assertEqual(active_meta_path.read_text(encoding="utf-8"), "stable-meta")
        self.assertEqual(challenger_model_path.read_text(encoding="utf-8"), "new-model")
        challenger_meta = json.loads(
            challenger_model_path.with_suffix(".meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(challenger_meta["promotion_status"], "rolled_back")
        self.assertTrue(challenger_meta["rollback_reason"].startswith("post_promotion_"))
        self.assertEqual(
            engine.storage.get_json_state(engine.MODEL_PROMOTION_OBSERVATION_STATE_KEY, {}),
            {},
        )
        with engine.storage._conn() as conn:
            rollback_count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='model_rollback'"
            ).fetchone()["c"]
            rollback_report_count = conn.execute(
                "SELECT COUNT(*) AS c FROM report_artifacts WHERE report_type='model_rollback'"
            ).fetchone()["c"]
            scorecard_count = conn.execute(
                "SELECT COUNT(*) AS c FROM model_scorecards WHERE symbol='BTC/USDT'"
            ).fetchone()["c"]
            rollback_scorecard = conn.execute(
                "SELECT expectancy_pct, profit_factor, max_drawdown_pct "
                "FROM model_scorecards WHERE symbol='BTC/USDT' AND stage='rolled_back' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            rolled_back_stage = conn.execute(
                "SELECT stage FROM model_registry WHERE model_id='test'"
            ).fetchone()
            restored_stage = conn.execute(
                "SELECT stage FROM model_registry WHERE model_id='stable'"
            ).fetchone()
        self.assertEqual(rollback_count, 1)
        self.assertEqual(rollback_report_count, 1)
        self.assertGreaterEqual(scorecard_count, 1)
        self.assertLess(rollback_scorecard["expectancy_pct"], 0.0)
        self.assertLess(rollback_scorecard["profit_factor"], 1.0)
        self.assertGreater(rollback_scorecard["max_drawdown_pct"], 0.0)
        self.assertEqual(rolled_back_stage["stage"], "rolled_back")
        self.assertEqual(restored_stage["stage"], "restored")

    def test_engine_observe_promoted_models_accepts_stable_model(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        active_model_path.write_text("stable-new-model", encoding="utf-8")
        backup_model_path = model_dir / "backups" / "stable_BTC.json"
        backup_model_path.parent.mkdir(parents=True, exist_ok=True)
        backup_model_path.write_text("stable-old-model", encoding="utf-8")
        backup_meta_path = backup_model_path.with_suffix(".meta.json")
        backup_meta_path.write_text("stable-old-meta", encoding="utf-8")

        promoted_at = datetime.now(timezone.utc) - timedelta(hours=10)
        engine.storage.set_json_state(
            engine.MODEL_PROMOTION_OBSERVATION_STATE_KEY,
            {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "promoted_at": promoted_at.isoformat(),
                    "active_model_path": str(active_model_path),
                    "active_model_id": "test",
                    "active_model_signature": list(engine._model_file_signature(active_model_path)),
                    "backup_model_path": str(backup_model_path),
                    "backup_model_id": "stable-old",
                    "backup_meta_path": str(backup_meta_path),
                    "baseline_holdout_accuracy": 0.62,
                    "baseline_objective_quality": 1.0,
                    "baseline_expectancy_pct": 0.60,
                    "baseline_profit_factor": 1.30,
                    "baseline_max_drawdown_pct": 0.50,
                    "baseline_trade_win_rate": 0.55,
                    "baseline_walkforward_summary": {
                        "total_return_pct": 1.6,
                        "expectancy_pct": 0.60,
                        "profit_factor": 1.30,
                        "max_drawdown_pct": 0.50,
                    },
                    "recent_walkforward_baseline_summary": {
                        "history_count": 3,
                        "avg_total_return_pct": 1.4,
                        "avg_expectancy_pct": 0.55,
                        "avg_profit_factor": 1.25,
                        "avg_max_drawdown_pct": 0.45,
                    },
                    "min_evaluations": 4,
                    "max_observation_age_hours": 72,
                    "status": "observing",
                }
            },
        )
        for idx in range(4):
            ts = (promoted_at + timedelta(hours=idx + 1)).isoformat()
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "test",
                    "up_probability": 0.8,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT", "suggested_action": "HOLD"},
                    "decision": {
                        "pipeline_mode": "execution",
                        "final_score": 0.8,
                        "model_id": "test",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "execution",
                    "actual_up": True,
                    "predicted_up": True,
                    "is_correct": True,
                    "entry_close": 100.0,
                    "future_close": 101.0,
                    "metadata": {},
                }
            )

        result = engine._observe_promoted_models(datetime.now(timezone.utc))

        self.assertEqual(result["accepted"], 1)
        self.assertFalse(backup_model_path.exists())
        self.assertFalse(backup_meta_path.exists())
        self.assertEqual(
            engine.storage.get_json_state(engine.MODEL_PROMOTION_OBSERVATION_STATE_KEY, {}),
            {},
        )
        with engine.storage._conn() as conn:
            accepted_count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='model_observation_accepted'"
            ).fetchone()["c"]
            accepted_stage = conn.execute(
                "SELECT stage FROM model_registry WHERE model_id='test'"
            ).fetchone()
        self.assertEqual(accepted_count, 1)
        self.assertEqual(accepted_stage["stage"], "accepted")

    def test_engine_observe_promoted_models_accepts_positive_edge_despite_lower_accuracy(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        active_model_path.write_text("positive-edge-model", encoding="utf-8")
        backup_model_path = model_dir / "backups" / "stable_BTC.json"
        backup_model_path.parent.mkdir(parents=True, exist_ok=True)
        backup_model_path.write_text("stable-old-model", encoding="utf-8")
        backup_meta_path = backup_model_path.with_suffix(".meta.json")
        backup_meta_path.write_text("stable-old-meta", encoding="utf-8")

        promoted_at = datetime.now(timezone.utc) - timedelta(hours=10)
        engine.storage.set_json_state(
            engine.MODEL_PROMOTION_OBSERVATION_STATE_KEY,
            {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "promoted_at": promoted_at.isoformat(),
                    "active_model_path": str(active_model_path),
                    "active_model_id": "test",
                    "active_model_signature": list(engine._model_file_signature(active_model_path)),
                    "backup_model_path": str(backup_model_path),
                    "backup_model_id": "stable-old",
                    "backup_meta_path": str(backup_meta_path),
                    "baseline_holdout_accuracy": 0.62,
                    "baseline_objective_quality": 1.0,
                    "baseline_expectancy_pct": 0.60,
                    "baseline_profit_factor": 1.30,
                    "baseline_max_drawdown_pct": 0.50,
                    "baseline_trade_win_rate": 0.55,
                    "baseline_avg_trade_return_pct": 0.80,
                    "baseline_walkforward_summary": {
                        "total_return_pct": 1.6,
                        "expectancy_pct": 0.60,
                        "profit_factor": 1.30,
                        "max_drawdown_pct": 0.50,
                    },
                    "recent_walkforward_baseline_summary": {
                        "history_count": 3,
                        "avg_total_return_pct": 1.4,
                        "avg_expectancy_pct": 0.55,
                        "avg_profit_factor": 1.25,
                        "avg_max_drawdown_pct": 0.45,
                    },
                    "min_evaluations": 4,
                    "max_observation_age_hours": 72,
                    "status": "observing",
                }
            },
        )
        evaluation_rows = (
            (True, True, 101.0),
            (False, True, 101.0),
            (True, True, 101.0),
            (False, True, 101.0),
        )
        for idx, (predicted_up, actual_up, future_close) in enumerate(
            evaluation_rows,
            start=1,
        ):
            ts = (promoted_at + timedelta(hours=idx)).isoformat()
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "test",
                    "up_probability": 0.8,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT", "suggested_action": "HOLD"},
                    "decision": {
                        "pipeline_mode": "execution",
                        "final_score": 0.8,
                        "model_id": "test",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "execution",
                    "actual_up": bool(actual_up),
                    "predicted_up": bool(predicted_up),
                    "is_correct": bool(predicted_up) == bool(actual_up),
                    "entry_close": 100.0,
                    "future_close": future_close,
                    "metadata": {},
                }
            )

        result = engine._observe_promoted_models(datetime.now(timezone.utc))

        self.assertEqual(result["accepted"], 1)
        self.assertFalse(backup_model_path.exists())
        self.assertFalse(backup_meta_path.exists())
        self.assertEqual(
            engine.storage.get_json_state(engine.MODEL_PROMOTION_OBSERVATION_STATE_KEY, {}),
            {},
        )
        with engine.storage._conn() as conn:
            accepted_event = conn.execute(
                "SELECT payload_json FROM execution_events "
                "WHERE event_type='model_observation_accepted' "
                "ORDER BY created_at DESC, id DESC LIMIT 1"
            ).fetchone()
        accepted_payload = json.loads(accepted_event["payload_json"])
        self.assertLess(accepted_payload["accuracy"], 0.6)
        self.assertGreater(accepted_payload["expectancy_pct"], 0.4)
        self.assertGreaterEqual(accepted_payload["profit_factor"], 1.0)

    def test_engine_observe_promotion_candidates_advances_and_promotes_candidate(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        challenger_model_path = model_dir / "xgboost_challenger_BTC_USDT.json"
        active_model_path.write_text("stable-model", encoding="utf-8")
        challenger_model_path.write_text("candidate-model", encoding="utf-8")

        registered_at = datetime.now(timezone.utc) - timedelta(hours=12)
        training_metadata = {
            "symbol": "BTC/USDT",
            "model_path": str(challenger_model_path),
            "model_id": "challenger",
            "active_model_path": str(active_model_path),
            "active_model_id": "champion",
            "challenger_model_path": str(challenger_model_path),
            "challenger_model_id": "challenger",
            "incumbent_model_id": "champion",
            "promoted_to_active": False,
            "promotion_status": "canary_pending",
            "promotion_reason": "candidate_higher_walkforward_return",
            "candidate_holdout_accuracy": 0.62,
            "candidate_walkforward_summary": {
                "symbol": "BTC/USDT",
                "total_splits": 2,
                "avg_win_rate": 55.0,
                "avg_trade_return_pct": 0.8,
                "total_return_pct": 1.6,
                "profit_factor": 1.3,
                "sharpe_like": 0.5,
            },
            "candidate_walkforward_splits": [],
            "dataset_end_timestamp_ms": 123456,
        }
        engine.storage.set_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "registered_at": registered_at.isoformat(),
                    "status": "shadow",
                    "active_model_path": str(active_model_path),
                    "active_model_id": "champion",
                    "active_model_signature": list(engine._model_file_signature(active_model_path)),
                    "challenger_model_path": str(challenger_model_path),
                    "challenger_model_id": "challenger",
                    "challenger_model_signature": list(engine._model_file_signature(challenger_model_path)),
                    "baseline_holdout_accuracy": 0.62,
                    "baseline_objective_score": 1.0,
                    "baseline_objective_quality": 1.0,
                    "baseline_expectancy_pct": 0.50,
                    "baseline_profit_factor": 1.10,
                    "baseline_max_drawdown_pct": 0.80,
                    "baseline_trade_win_rate": 0.50,
                    "baseline_avg_trade_return_pct": 0.8,
                    "promotion_reason": "candidate_higher_walkforward_return",
                    "min_shadow_evaluations": 4,
                    "min_live_evaluations": 3,
                    "max_shadow_age_hours": 72,
                    "max_live_age_hours": 72,
                    "live_allocation_pct": 0.03,
                    "training_metadata": training_metadata,
                }
            },
        )
        for idx, (champion_correct, challenger_correct) in enumerate(
            ((True, True), (True, True), (False, True), (True, True)),
            start=1,
        ):
            ts = (registered_at + timedelta(hours=idx)).isoformat()
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "champion",
                    "up_probability": 0.8,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT"},
                    "decision": {
                        "pipeline_mode": "execution",
                        "final_score": 0.8,
                        "model_id": "champion",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "execution",
                    "actual_up": bool(champion_correct),
                    "predicted_up": True,
                    "is_correct": bool(champion_correct),
                    "entry_close": 100.0,
                    "future_close": 101.0 if champion_correct else 99.0,
                    "metadata": {},
                }
            )
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "challenger",
                    "up_probability": 0.85,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT"},
                    "decision": {
                        "pipeline_mode": "challenger_shadow",
                        "final_score": 0.85,
                        "model_id": "challenger",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "challenger_shadow",
                    "actual_up": bool(challenger_correct),
                    "predicted_up": True,
                    "is_correct": bool(challenger_correct),
                    "entry_close": 100.0,
                    "future_close": 101.0 if challenger_correct else 99.0,
                    "metadata": {},
                }
            )

        shadow_result = engine._observe_promoted_models(datetime.now(timezone.utc))

        self.assertEqual(shadow_result["shadow_to_live"], 1)
        candidate_state = engine.storage.get_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {},
        )
        self.assertEqual(candidate_state["BTC/USDT"]["status"], "live")
        live_started_at = datetime.fromisoformat(candidate_state["BTC/USDT"]["live_started_at"])
        for idx, (champion_correct, challenger_correct) in enumerate(
            ((True, True), (False, True), (True, True)),
            start=1,
        ):
            ts = (live_started_at + timedelta(hours=idx)).isoformat()
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "champion",
                    "up_probability": 0.8,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT"},
                    "decision": {
                        "pipeline_mode": "execution",
                        "final_score": 0.8,
                        "model_id": "champion",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "execution",
                    "actual_up": bool(champion_correct),
                    "predicted_up": True,
                    "is_correct": bool(champion_correct),
                    "entry_close": 100.0,
                    "future_close": 101.0 if champion_correct else 99.0,
                    "metadata": {},
                }
            )
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "challenger",
                    "up_probability": 0.9,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT"},
                    "decision": {
                        "pipeline_mode": "challenger_live",
                        "final_score": 0.9,
                        "model_id": "challenger",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "challenger_live",
                    "actual_up": bool(challenger_correct),
                    "predicted_up": True,
                    "is_correct": bool(challenger_correct),
                    "entry_close": 100.0,
                    "future_close": 101.0 if challenger_correct else 99.0,
                    "metadata": {},
                }
            )
            engine.storage.insert_ab_test_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "champion_model_version": "champion",
                    "challenger_model_version": "challenger",
                    "champion_probability": 0.8,
                    "challenger_probability": 0.9,
                    "champion_execute": False,
                    "challenger_execute": True,
                    "selected_variant": "challenger_live",
                    "allocation_pct": 0.03,
                    "notes": "candidate canary",
                }
            )

        live_result = engine._observe_promoted_models(datetime.now(timezone.utc) + timedelta(hours=1))

        self.assertEqual(live_result["promoted"], 1)
        self.assertEqual(active_model_path.read_text(encoding="utf-8"), "candidate-model")
        self.assertFalse(challenger_model_path.exists())
        self.assertEqual(
            engine.storage.get_json_state(engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY, {}),
            {},
        )
        observation_state = engine.storage.get_json_state(
            engine.MODEL_PROMOTION_OBSERVATION_STATE_KEY,
            {},
        )
        self.assertIn("BTC/USDT", observation_state)
        active_meta = json.loads(
            active_model_path.with_suffix(".meta.json").read_text(encoding="utf-8")
        )
        self.assertTrue(active_meta["promoted_to_active"])
        self.assertEqual(active_meta["promotion_status"], "promoted")
        self.assertEqual(active_meta["model_id"], "challenger")
        self.assertEqual(active_meta["canary_live_trade_count"], 3)
        with engine.storage._conn() as conn:
            scorecard_count = conn.execute(
                "SELECT COUNT(*) AS c FROM model_scorecards WHERE symbol='BTC/USDT'"
            ).fetchone()["c"]
            challenger_live_scorecard = conn.execute(
                "SELECT expectancy_pct, profit_factor, max_drawdown_pct "
                "FROM model_scorecards WHERE symbol='BTC/USDT' "
                "AND evaluation_type='challenger_live' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            active_stage = conn.execute(
                "SELECT stage FROM model_registry WHERE model_id='challenger'"
            ).fetchone()
        self.assertGreaterEqual(scorecard_count, 3)
        self.assertGreater(challenger_live_scorecard["expectancy_pct"], 0.0)
        self.assertGreaterEqual(challenger_live_scorecard["profit_factor"], 1.0)
        self.assertGreaterEqual(challenger_live_scorecard["max_drawdown_pct"], 0.0)
        self.assertIn(active_stage["stage"], {"promoted", "observing"})

    def test_engine_observe_promotion_candidates_rejects_live_candidate_with_negative_realized_pnl(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        challenger_model_path = model_dir / "xgboost_challenger_BTC_USDT.json"
        active_model_path.write_text("stable-model", encoding="utf-8")
        challenger_model_path.write_text("candidate-model", encoding="utf-8")

        registered_at = datetime.now(timezone.utc) - timedelta(hours=12)
        live_started_at = registered_at + timedelta(hours=6)
        engine.storage.set_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "registered_at": registered_at.isoformat(),
                    "status": "live",
                    "live_started_at": live_started_at.isoformat(),
                    "active_model_path": str(active_model_path),
                    "active_model_id": "champion",
                    "active_model_signature": list(engine._model_file_signature(active_model_path)),
                    "challenger_model_path": str(challenger_model_path),
                    "challenger_model_id": "challenger",
                    "challenger_model_signature": list(engine._model_file_signature(challenger_model_path)),
                    "baseline_holdout_accuracy": 0.62,
                    "baseline_objective_score": 1.0,
                    "baseline_objective_quality": 1.0,
                    "baseline_expectancy_pct": 0.50,
                    "baseline_profit_factor": 1.10,
                    "baseline_max_drawdown_pct": 0.80,
                    "baseline_trade_win_rate": 0.50,
                    "baseline_avg_trade_return_pct": 0.80,
                    "shadow_eval_count": 4,
                    "shadow_executed_count": 4,
                    "shadow_accuracy": 0.75,
                    "shadow_objective_score": 1.2,
                    "shadow_objective_quality": 1.2,
                    "shadow_expectancy_pct": 0.70,
                    "shadow_profit_factor": 1.40,
                    "shadow_max_drawdown_pct": 0.50,
                    "shadow_avg_trade_return_pct": 0.90,
                    "promotion_reason": "candidate_higher_walkforward_expectancy",
                    "min_shadow_evaluations": 4,
                    "min_live_evaluations": 3,
                    "max_shadow_age_hours": 72,
                    "max_live_age_hours": 72,
                    "live_allocation_pct": 0.03,
                    "training_metadata": {
                        "symbol": "BTC/USDT",
                        "model_path": str(challenger_model_path),
                        "model_id": "challenger",
                        "active_model_path": str(active_model_path),
                        "active_model_id": "champion",
                        "challenger_model_path": str(challenger_model_path),
                        "challenger_model_id": "challenger",
                        "incumbent_model_id": "champion",
                        "promoted_to_active": False,
                        "promotion_status": "canary_pending",
                        "promotion_reason": "candidate_higher_walkforward_expectancy",
                    },
                }
            },
        )
        for idx in range(3):
            ts = (live_started_at + timedelta(hours=idx + 1)).isoformat()
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "champion",
                    "up_probability": 0.8,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT"},
                    "decision": {
                        "pipeline_mode": "execution",
                        "final_score": 0.8,
                        "model_id": "champion",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "execution",
                    "actual_up": True,
                    "predicted_up": True,
                    "is_correct": True,
                    "entry_close": 100.0,
                    "future_close": 101.0,
                    "metadata": {},
                }
            )
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "challenger",
                    "up_probability": 0.9,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT"},
                    "decision": {
                        "pipeline_mode": "challenger_live",
                        "final_score": 0.9,
                        "model_id": "challenger",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "challenger_live",
                    "actual_up": True,
                    "predicted_up": True,
                    "is_correct": True,
                    "entry_close": 100.0,
                    "future_close": 101.0,
                    "metadata": {},
                }
            )
            engine.storage.insert_ab_test_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "champion_model_version": "champion",
                    "challenger_model_version": "challenger",
                    "champion_probability": 0.8,
                    "challenger_probability": 0.9,
                    "champion_execute": False,
                    "challenger_execute": True,
                    "selected_variant": "challenger_live",
                    "allocation_pct": 0.03,
                    "notes": "candidate canary",
                }
            )
        for idx, net_pnl in enumerate((-2.0, -1.5), start=1):
            trade_id = f"canary-live-{idx}"
            open_time = (live_started_at + timedelta(hours=idx)).isoformat()
            close_time = (live_started_at + timedelta(hours=idx, minutes=30)).isoformat()
            engine.storage.insert_pnl_ledger_entry(
                {
                    "trade_id": trade_id,
                    "symbol": "BTC/USDT",
                    "event_type": "open",
                    "event_time": open_time,
                    "quantity": 1.0,
                    "notional_value": 100.0,
                    "reference_price": 100.0,
                    "fill_price": 100.0,
                    "gross_pnl": 0.0,
                    "fee_cost": 0.1,
                    "slippage_cost": 0.0,
                    "net_pnl": -0.1,
                    "net_return_pct": -0.1,
                    "holding_hours": 0.0,
                    "model_id": "challenger",
                }
            )
            engine.storage.insert_pnl_ledger_entry(
                {
                    "trade_id": trade_id,
                    "symbol": "BTC/USDT",
                    "event_type": "close",
                    "event_time": close_time,
                    "quantity": 1.0,
                    "notional_value": 100.0,
                    "reference_price": 100.0,
                    "fill_price": 98.0,
                    "gross_pnl": net_pnl,
                    "fee_cost": 0.0,
                    "slippage_cost": 0.0,
                    "net_pnl": net_pnl,
                    "net_return_pct": net_pnl,
                    "holding_hours": 0.5,
                    "model_id": "challenger",
                }
            )

        result = engine._observe_promoted_models(datetime.now(timezone.utc))

        self.assertEqual(result["rejected"], 1)
        self.assertEqual(
            engine.storage.get_json_state(engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY, {}),
            {},
        )
        challenger_meta = json.loads(
            challenger_model_path.with_suffix(".meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(challenger_meta["promotion_status"], "canary_rejected")
        self.assertTrue(
            challenger_meta["promotion_reason"].startswith("live_realized_")
        )
        canary_metrics = challenger_meta["canary_metrics"]
        self.assertEqual(canary_metrics["live_trade_count"], 3)
        self.assertEqual(canary_metrics["live_pnl_closed_trade_count"], 2)
        self.assertLess(canary_metrics["live_pnl_realized_net_pnl"], 0.0)

    def test_engine_observe_promotion_candidates_advances_shadow_candidate_with_lower_accuracy_but_better_edge(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        challenger_model_path = model_dir / "xgboost_challenger_BTC_USDT.json"
        active_model_path.write_text("stable-model", encoding="utf-8")
        challenger_model_path.write_text("candidate-model", encoding="utf-8")

        registered_at = datetime.now(timezone.utc) - timedelta(hours=12)
        engine.storage.set_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "registered_at": registered_at.isoformat(),
                    "status": "shadow",
                    "active_model_path": str(active_model_path),
                    "active_model_id": "champion",
                    "active_model_signature": list(engine._model_file_signature(active_model_path)),
                    "challenger_model_path": str(challenger_model_path),
                    "challenger_model_id": "challenger",
                    "challenger_model_signature": list(engine._model_file_signature(challenger_model_path)),
                    "baseline_holdout_accuracy": 0.62,
                    "baseline_objective_score": 1.0,
                    "baseline_objective_quality": 1.0,
                    "baseline_expectancy_pct": 0.50,
                    "baseline_profit_factor": 1.10,
                    "baseline_max_drawdown_pct": 0.80,
                    "baseline_trade_win_rate": 0.50,
                    "baseline_avg_trade_return_pct": 0.80,
                    "promotion_reason": "candidate_higher_walkforward_expectancy",
                    "min_shadow_evaluations": 4,
                    "min_live_evaluations": 3,
                    "max_shadow_age_hours": 72,
                    "max_live_age_hours": 72,
                    "live_allocation_pct": 0.03,
                    "training_metadata": {
                        "symbol": "BTC/USDT",
                        "model_path": str(challenger_model_path),
                        "model_id": "challenger",
                        "active_model_path": str(active_model_path),
                        "active_model_id": "champion",
                        "challenger_model_path": str(challenger_model_path),
                        "challenger_model_id": "challenger",
                        "incumbent_model_id": "champion",
                        "promoted_to_active": False,
                        "promotion_status": "canary_pending",
                        "promotion_reason": "candidate_higher_walkforward_expectancy",
                    },
                }
            },
        )
        champion_rows = (
            (True, True, 101.0),
            (True, True, 101.0),
            (True, False, 99.0),
            (True, True, 101.0),
        )
        challenger_rows = (
            (True, True, 101.0),
            (False, True, 101.0),
            (True, True, 101.0),
            (False, True, 101.0),
        )
        for idx, (champion_row, challenger_row) in enumerate(
            zip(champion_rows, challenger_rows),
            start=1,
        ):
            ts = (registered_at + timedelta(hours=idx)).isoformat()
            champion_predicted_up, champion_actual_up, champion_future_close = champion_row
            challenger_predicted_up, challenger_actual_up, challenger_future_close = challenger_row
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "champion",
                    "up_probability": 0.8,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT"},
                    "decision": {
                        "pipeline_mode": "execution",
                        "final_score": 0.8,
                        "model_id": "champion",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "execution",
                    "actual_up": bool(champion_actual_up),
                    "predicted_up": bool(champion_predicted_up),
                    "is_correct": bool(champion_predicted_up) == bool(champion_actual_up),
                    "entry_close": 100.0,
                    "future_close": champion_future_close,
                    "metadata": {},
                }
            )
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "challenger",
                    "up_probability": 0.85,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT"},
                    "decision": {
                        "pipeline_mode": "challenger_shadow",
                        "final_score": 0.85,
                        "model_id": "challenger",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "challenger_shadow",
                    "actual_up": bool(challenger_actual_up),
                    "predicted_up": bool(challenger_predicted_up),
                    "is_correct": bool(challenger_predicted_up) == bool(challenger_actual_up),
                    "entry_close": 100.0,
                    "future_close": challenger_future_close,
                    "metadata": {},
                }
            )

        result = engine._observe_promoted_models(datetime.now(timezone.utc))

        self.assertEqual(result["shadow_to_live"], 1)
        candidate_state = engine.storage.get_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {},
        )
        self.assertEqual(candidate_state["BTC/USDT"]["status"], "live")
        self.assertLess(candidate_state["BTC/USDT"]["shadow_accuracy"], 0.6)
        self.assertGreater(candidate_state["BTC/USDT"]["shadow_expectancy_pct"], 0.4)
        self.assertGreater(candidate_state["BTC/USDT"]["shadow_objective_quality"], 1.0)

    def test_engine_observe_promotion_candidates_rejects_underperforming_shadow_candidate(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        challenger_model_path = model_dir / "xgboost_challenger_BTC_USDT.json"
        active_model_path.write_text("stable-model", encoding="utf-8")
        challenger_model_path.write_text("candidate-model", encoding="utf-8")

        registered_at = datetime.now(timezone.utc) - timedelta(hours=12)
        engine.storage.set_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "registered_at": registered_at.isoformat(),
                    "status": "shadow",
                    "active_model_path": str(active_model_path),
                    "active_model_id": "champion",
                    "active_model_signature": list(engine._model_file_signature(active_model_path)),
                    "challenger_model_path": str(challenger_model_path),
                    "challenger_model_id": "challenger",
                    "challenger_model_signature": list(engine._model_file_signature(challenger_model_path)),
                    "baseline_holdout_accuracy": 0.62,
                    "baseline_objective_score": 1.0,
                    "baseline_objective_quality": 1.0,
                    "baseline_expectancy_pct": 0.50,
                    "baseline_profit_factor": 1.10,
                    "baseline_max_drawdown_pct": 0.80,
                    "baseline_trade_win_rate": 0.50,
                    "baseline_avg_trade_return_pct": 0.8,
                    "promotion_reason": "candidate_higher_walkforward_return",
                    "min_shadow_evaluations": 4,
                    "min_live_evaluations": 3,
                    "max_shadow_age_hours": 72,
                    "max_live_age_hours": 72,
                    "live_allocation_pct": 0.03,
                    "training_metadata": {
                        "symbol": "BTC/USDT",
                        "model_path": str(challenger_model_path),
                        "model_id": "challenger",
                        "active_model_path": str(active_model_path),
                        "active_model_id": "champion",
                        "challenger_model_path": str(challenger_model_path),
                        "challenger_model_id": "challenger",
                        "incumbent_model_id": "champion",
                        "promoted_to_active": False,
                        "promotion_status": "canary_pending",
                        "promotion_reason": "candidate_higher_walkforward_return",
                    },
                }
            },
        )
        for idx, (champion_correct, challenger_correct) in enumerate(
            ((True, False), (True, False), (True, True), (False, False)),
            start=1,
        ):
            ts = (registered_at + timedelta(hours=idx)).isoformat()
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "champion",
                    "up_probability": 0.8,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT"},
                    "decision": {
                        "pipeline_mode": "execution",
                        "final_score": 0.8,
                        "model_id": "champion",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "execution",
                    "actual_up": bool(champion_correct),
                    "predicted_up": True,
                    "is_correct": bool(champion_correct),
                    "entry_close": 100.0,
                    "future_close": 101.0 if champion_correct else 99.0,
                    "metadata": {},
                }
            )
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "challenger",
                    "up_probability": 0.7,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT"},
                    "decision": {
                        "pipeline_mode": "challenger_shadow",
                        "final_score": 0.7,
                        "model_id": "challenger",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "challenger_shadow",
                    "actual_up": bool(challenger_correct),
                    "predicted_up": True,
                    "is_correct": bool(challenger_correct),
                    "entry_close": 100.0,
                    "future_close": 101.0 if challenger_correct else 99.0,
                    "metadata": {},
                }
            )

        result = engine._observe_promoted_models(datetime.now(timezone.utc))

        self.assertEqual(result["rejected"], 1)
        self.assertEqual(
            engine.storage.get_json_state(engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY, {}),
            {},
        )
        challenger_meta = json.loads(
            challenger_model_path.with_suffix(".meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(challenger_meta["promotion_status"], "canary_rejected")
        self.assertTrue(challenger_meta["promotion_reason"].startswith("shadow_"))
        self.assertEqual(active_model_path.read_text(encoding="utf-8"), "stable-model")
        with engine.storage._conn() as conn:
            rejected_stage = conn.execute(
                "SELECT stage FROM model_registry WHERE model_id='challenger'"
            ).fetchone()
        self.assertEqual(rejected_stage["stage"], "canary_rejected")

    def test_engine_uses_live_trader_when_runtime_mode_is_live(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "live"

        engine = engine_module.CryptoAIV2Engine(settings)
        self.assertEqual(engine.executor.__class__.__name__, "LiveTrader")
        self.assertFalse(engine.executor.enabled)

    def test_engine_manage_open_positions_ignores_live_dry_run_close(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 110.0

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "live"
        settings.app.allow_live_orders = False

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        now = datetime.now(timezone.utc).isoformat()
        engine.storage.insert_trade(
            {
                "id": "open-1",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "stop_loss": 95.0,
                "take_profit": 105.0,
            }
        )
        engine.feature_pipeline.build = lambda payload: SimpleNamespace(
            symbol=payload.symbol,
            valid=True,
            values={},
        )
        engine._predictor_for_symbol = lambda symbol: SimpleNamespace(
            predict=lambda snapshot: PredictionResult(
                symbol=snapshot.symbol,
                up_probability=0.8,
                feature_count=0,
                model_version="fake",
            )
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: ["take_profit_1"]
        engine.notifier = SimpleNamespace(
            notify_trade_close=lambda *args, **kwargs: None
        )

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )
        self.assertEqual(closed, 0)
        self.assertEqual(len(engine.storage.get_positions()), 1)

    def test_engine_manage_open_positions_uses_stored_take_profit(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 104.0

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        now = datetime.now(timezone.utc).isoformat()
        engine.storage.insert_trade(
            {
                "id": "open-stored-tp",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "stop_loss": 95.0,
                "take_profit": 103.0,
            }
        )
        engine.feature_pipeline.build = lambda payload: SimpleNamespace(
            symbol=payload.symbol,
            valid=True,
            values={},
        )
        engine._predictor_for_symbol = lambda symbol: SimpleNamespace(
            predict=lambda snapshot: PredictionResult(
                symbol=snapshot.symbol,
                up_probability=0.8,
                feature_count=0,
                model_version="fake",
            )
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine.notifier = SimpleNamespace(
            notify_trade_close=lambda *args, **kwargs: None
        )

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 0)
        positions = engine.storage.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["quantity"], 0.5)
        self.assertAlmostEqual(engine.storage.get_open_trades()[0]["quantity"], 0.5)

    def test_engine_manage_open_positions_uses_stored_stop_loss(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 94.0

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        now = datetime.now(timezone.utc).isoformat()
        engine.storage.insert_trade(
            {
                "id": "open-stored-stop",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.feature_pipeline.build = lambda payload: SimpleNamespace(
            symbol=payload.symbol,
            valid=True,
            values={},
        )
        engine._predictor_for_symbol = lambda symbol: SimpleNamespace(
            predict=lambda snapshot: PredictionResult(
                symbol=snapshot.symbol,
                up_probability=0.8,
                feature_count=0,
                model_version="fake",
            )
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine.notifier = SimpleNamespace(
            notify_trade_close=lambda *args, **kwargs: None
        )

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 1)
        self.assertEqual(engine.storage.get_positions(), [])
        self.assertEqual(engine.storage.get_open_trades(), [])

    def test_engine_manage_open_positions_fully_closes_take_profit_when_evidence_is_weak(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 104.0

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        now = datetime.now(timezone.utc).isoformat()
        engine.storage.insert_trade(
            {
                "id": "weak-evidence-tp",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "stop_loss": 95.0,
                "take_profit": 103.0,
            }
        )
        engine.feature_pipeline.build = lambda payload: SimpleNamespace(
            symbol=payload.symbol,
            valid=True,
            values={},
        )
        engine._predictor_for_symbol = lambda symbol: SimpleNamespace(
            predict=lambda snapshot: PredictionResult(
                symbol=snapshot.symbol,
                up_probability=0.8,
                feature_count=0,
                model_version="fake",
            )
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._active_model_exit_policy = lambda symbol: {
            "adaptive_active": True,
            "scale": 0.55,
            "source": "post_promotion_observation",
            "reason": "weak",
            "time_stop_hours": 48.0,
            "de_risk_min_hours": 2.0,
            "de_risk_min_pnl_ratio": 0.004,
            "force_full_take_profit": True,
        }
        engine.notifier = SimpleNamespace(
            notify_trade_close=lambda *args, **kwargs: None
        )

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 1)
        self.assertEqual(engine.storage.get_positions(), [])
        self.assertEqual(engine.storage.get_open_trades(), [])

    def test_engine_manage_open_positions_de_risks_earlier_when_evidence_is_weak(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 101.0

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        engine.storage.insert_trade(
            {
                "id": "weak-evidence-derisk",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.feature_pipeline.build = lambda payload: SimpleNamespace(
            symbol=payload.symbol,
            valid=True,
            values={},
        )
        engine._predictor_for_symbol = lambda symbol: SimpleNamespace(
            predict=lambda snapshot: PredictionResult(
                symbol=snapshot.symbol,
                up_probability=0.8,
                feature_count=0,
                model_version="fake",
            )
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: None
        engine._active_model_exit_policy = lambda symbol: {
            "adaptive_active": True,
            "scale": 0.55,
            "source": "post_promotion_observation",
            "reason": "weak",
            "time_stop_hours": 48.0,
            "de_risk_min_hours": 2.0,
            "de_risk_min_pnl_ratio": 0.005,
            "force_full_take_profit": False,
        }
        engine.notifier = SimpleNamespace(
            notify_trade_close=lambda *args, **kwargs: None
        )

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 0)
        positions = engine.storage.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["quantity"], 0.5)

    def test_engine_manage_open_positions_uses_shorter_time_stop_when_evidence_is_weak(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 100.4

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        engine.storage.insert_trade(
            {
                "id": "weak-evidence-time-stop",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.feature_pipeline.build = lambda payload: SimpleNamespace(
            symbol=payload.symbol,
            valid=True,
            values={},
        )
        engine._predictor_for_symbol = lambda symbol: SimpleNamespace(
            predict=lambda snapshot: PredictionResult(
                symbol=snapshot.symbol,
                up_probability=0.8,
                feature_count=0,
                model_version="fake",
            )
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: None
        engine._active_model_exit_policy = lambda symbol: {
            "adaptive_active": True,
            "scale": 0.35,
            "source": "post_promotion_observation",
            "reason": "very_weak",
            "time_stop_hours": 4.0,
            "de_risk_min_hours": 1.5,
            "de_risk_min_pnl_ratio": 0.003,
            "force_full_take_profit": True,
        }
        engine.notifier = SimpleNamespace(
            notify_trade_close=lambda *args, **kwargs: None
        )

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 1)
        self.assertEqual(engine.storage.get_positions(), [])
        self.assertEqual(engine.storage.get_open_trades(), [])

    def test_engine_manage_open_positions_generates_reflection_on_full_close(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 94.0

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        now = datetime.now(timezone.utc).isoformat()
        engine.storage.insert_trade(
            {
                "id": "open-reflect",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "rationale": "entry rationale",
                "confidence": 0.8,
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": now,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine._prepare_position_review = lambda symbol, now: None
        engine.notifier = SimpleNamespace(
            notify_trade_close=lambda *args, **kwargs: None
        )

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 1)
        with engine.storage._conn() as conn:
            reflection_count = conn.execute(
                "SELECT COUNT(*) AS c FROM reflections"
            ).fetchone()["c"]
        self.assertEqual(reflection_count, 1)

    def test_engine_position_review_reasons_trigger_de_risk_on_profitable_hold(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        reasons = engine._position_review_exit_reasons(
            position={"symbol": "BTC/USDT", "entry_price": 100.0},
            current_price=102.0,
            hours_held=9,
            review_snapshot={
                "ok": True,
                "insight": ResearchInsight(
                    symbol="BTC/USDT",
                    suggested_action=SuggestedAction.HOLD,
                ),
                "validation": SimpleNamespace(ok=True),
                "decision": SimpleNamespace(portfolio_rating="HOLD"),
            },
        )
        self.assertIn("portfolio_de_risk", reasons)

    def test_engine_position_review_auto_pause_does_not_trigger_de_risk(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        reasons = engine._position_review_exit_reasons(
            position={"symbol": "BTC/USDT", "entry_price": 100.0},
            current_price=102.0,
            hours_held=9,
            review_snapshot={
                "ok": True,
                "insight": ResearchInsight(
                    symbol="BTC/USDT",
                    suggested_action=SuggestedAction.HOLD,
                ),
                "validation": SimpleNamespace(ok=True),
                "decision": SimpleNamespace(portfolio_rating="HOLD"),
                "review": SimpleNamespace(reasons=["setup_auto_pause"]),
            },
        )
        self.assertNotIn("portfolio_de_risk", reasons)

    def test_engine_review_exit_close_quantity_fully_closes_after_second_downgrade(self):
        import core.engine as engine_module

        self.assertEqual(
            engine_module.CryptoAIV2Engine._review_exit_close_quantity(
                initial_quantity=1.0,
                current_quantity=0.5,
                exit_reasons=["portfolio_de_risk"],
            ),
            0.5,
        )

    def test_engine_manage_open_positions_de_risks_on_first_research_exit(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 100.2

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = datetime.now(timezone.utc).isoformat()
        trade_id = "research-de-risk"
        engine.storage.insert_trade(
            {
                "id": trade_id,
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "entry rationale",
                "confidence": 0.7,
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: {
            "ok": True,
            "insight": ResearchInsight(
                symbol=symbol,
                suggested_action=SuggestedAction.CLOSE,
            ),
            "prediction": PredictionResult(
                symbol=symbol,
                up_probability=0.55,
                feature_count=10,
                model_version="test",
            ),
            "validation": SimpleNamespace(ok=True),
            "decision": SimpleNamespace(portfolio_rating="HOLD"),
            "review": SimpleNamespace(review_score=-0.2, reasons=[]),
        }
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 0)
        positions = engine.storage.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["quantity"], 0.5)
        state = engine.storage.get_json_state(engine.POSITION_REVIEW_STATE_KEY, {})
        self.assertEqual(state[trade_id]["research_exit_count"], 1)
        with engine.storage._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM execution_events WHERE event_type='close' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIn("research_de_risk", row["payload_json"])

    def test_engine_manage_open_positions_fully_closes_on_second_research_exit(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 100.1

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = datetime.now(timezone.utc).isoformat()
        trade_id = "research-full-close"
        engine.storage.insert_trade(
            {
                "id": trade_id,
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 0.5,
                "initial_quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "entry rationale",
                "confidence": 0.7,
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 0.5,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.storage.set_json_state(
            engine.POSITION_REVIEW_STATE_KEY,
            {
                trade_id: {
                    "trade_id": trade_id,
                    "symbol": "ETH/USDT",
                    "research_exit_count": 1,
                }
            },
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: {
            "ok": True,
            "insight": ResearchInsight(
                symbol=symbol,
                suggested_action=SuggestedAction.CLOSE,
            ),
            "prediction": PredictionResult(
                symbol=symbol,
                up_probability=0.55,
                feature_count=10,
                model_version="test",
            ),
            "validation": SimpleNamespace(ok=True),
            "decision": SimpleNamespace(portfolio_rating="HOLD"),
            "review": SimpleNamespace(review_score=-0.2, reasons=[]),
        }
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 1)
        self.assertEqual(engine.storage.get_positions(), [])
        state = engine.storage.get_json_state(engine.POSITION_REVIEW_STATE_KEY, {})
        self.assertNotIn(trade_id, state)

    def test_engine_manage_open_positions_keeps_offensive_canary_runner_on_second_downgrade(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 99.6

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        trade_id = "research-offensive-watch"
        engine.storage.insert_trade(
            {
                "id": trade_id,
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 0.5,
                "initial_quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "entry rationale",
                "confidence": 0.7,
                "metadata": {
                    "pipeline_mode": "paper_canary",
                    "paper_canary_mode": "offensive_review",
                },
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 0.5,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.storage.set_json_state(
            engine.POSITION_REVIEW_STATE_KEY,
            {
                trade_id: {
                    "trade_id": trade_id,
                    "symbol": "ETH/USDT",
                    "research_exit_count": 1,
                }
            },
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: {
            "ok": True,
            "insight": ResearchInsight(
                symbol=symbol,
                suggested_action=SuggestedAction.CLOSE,
            ),
            "prediction": PredictionResult(
                symbol=symbol,
                up_probability=0.58,
                feature_count=10,
                model_version="test",
            ),
            "validation": SimpleNamespace(ok=True),
            "decision": SimpleNamespace(portfolio_rating="HOLD"),
            "review": SimpleNamespace(
                raw_action="OPEN_LONG",
                reviewed_action="CLOSE",
                review_score=-0.22,
                reasons=["extreme_fear_offensive_setup"],
            ),
        }
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 0)
        positions = engine.storage.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["quantity"], 0.5)
        state = engine.storage.get_json_state(engine.POSITION_REVIEW_STATE_KEY, {})
        self.assertEqual(state[trade_id]["research_exit_count"], 2)
        with engine.storage._conn() as conn:
            close_count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='close'"
            ).fetchone()["c"]
            watch_row = conn.execute(
                "SELECT payload_json FROM execution_events WHERE event_type='position_review_watch' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(close_count, 0)
        self.assertIn('"research_exit_count": 2', watch_row["payload_json"])

    def test_engine_manage_open_positions_closes_offensive_canary_on_third_downgrade(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 99.6

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        trade_id = "research-offensive-close"
        engine.storage.insert_trade(
            {
                "id": trade_id,
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 0.5,
                "initial_quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "entry rationale",
                "confidence": 0.7,
                "metadata": {
                    "pipeline_mode": "paper_canary",
                    "paper_canary_mode": "offensive_review",
                },
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 0.5,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.storage.set_json_state(
            engine.POSITION_REVIEW_STATE_KEY,
            {
                trade_id: {
                    "trade_id": trade_id,
                    "symbol": "ETH/USDT",
                    "research_exit_count": 2,
                }
            },
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: {
            "ok": True,
            "insight": ResearchInsight(
                symbol=symbol,
                suggested_action=SuggestedAction.CLOSE,
            ),
            "prediction": PredictionResult(
                symbol=symbol,
                up_probability=0.58,
                feature_count=10,
                model_version="test",
            ),
            "validation": SimpleNamespace(ok=True),
            "decision": SimpleNamespace(portfolio_rating="HOLD"),
            "review": SimpleNamespace(
                raw_action="OPEN_LONG",
                reviewed_action="CLOSE",
                review_score=-0.22,
                reasons=["extreme_fear_offensive_setup"],
            ),
        }
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 1)
        self.assertEqual(engine.storage.get_positions(), [])
        state = engine.storage.get_json_state(engine.POSITION_REVIEW_STATE_KEY, {})
        self.assertNotIn(trade_id, state)

    def test_engine_manage_open_positions_keeps_primary_review_runner_before_min_hold_even_after_multiple_downgrades(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 99.7

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        trade_id = "research-primary-watch"
        engine.storage.insert_trade(
            {
                "id": trade_id,
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 0.5,
                "initial_quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "entry rationale",
                "confidence": 0.7,
                "metadata": {
                    "pipeline_mode": "paper_canary",
                    "paper_canary_mode": "primary_review",
                    "raw_action": "OPEN_LONG",
                    "reviewed_action": "OPEN_LONG",
                    "review_score": 0.28,
                    "review_reasons": ["xgb_strong", "liquidity_supportive"],
                },
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 0.5,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.storage.set_json_state(
            engine.POSITION_REVIEW_STATE_KEY,
            {
                trade_id: {
                    "trade_id": trade_id,
                    "symbol": "ETH/USDT",
                    "research_exit_count": 3,
                }
            },
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: {
            "ok": True,
            "insight": ResearchInsight(
                symbol=symbol,
                suggested_action=SuggestedAction.CLOSE,
            ),
            "prediction": PredictionResult(
                symbol=symbol,
                up_probability=0.61,
                feature_count=10,
                model_version="test",
            ),
            "validation": SimpleNamespace(ok=True),
            "decision": SimpleNamespace(portfolio_rating="HOLD"),
            "review": SimpleNamespace(
                raw_action="OPEN_LONG",
                reviewed_action="CLOSE",
                review_score=-0.18,
                reasons=[],
            ),
        }
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 0)
        positions = engine.storage.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["quantity"], 0.5)
        state = engine.storage.get_json_state(engine.POSITION_REVIEW_STATE_KEY, {})
        self.assertEqual(state[trade_id]["research_exit_count"], 4)
        with engine.storage._conn() as conn:
            close_count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='close'"
            ).fetchone()["c"]
        self.assertEqual(close_count, 0)

    def test_engine_manage_open_positions_keeps_primary_review_full_runner_on_first_downgrade(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 99.7

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        trade_id = "research-primary-full-watch"
        engine.storage.insert_trade(
            {
                "id": trade_id,
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "initial_quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "primary review runner",
                "confidence": 0.7,
                "metadata": {
                    "pipeline_mode": "paper_canary",
                    "paper_canary_mode": "primary_review",
                    "raw_action": "HOLD",
                    "reviewed_action": "OPEN_LONG",
                    "review_score": 0.20,
                    "review_reasons": ["trend_supportive"],
                },
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: {
            "ok": True,
            "insight": ResearchInsight(
                symbol=symbol,
                suggested_action=SuggestedAction.CLOSE,
            ),
            "prediction": PredictionResult(
                symbol=symbol,
                up_probability=0.61,
                feature_count=10,
                model_version="test",
            ),
            "validation": SimpleNamespace(ok=True),
            "decision": SimpleNamespace(portfolio_rating="HOLD"),
            "review": SimpleNamespace(
                raw_action="HOLD",
                reviewed_action="CLOSE",
                review_score=-0.18,
                reasons=[],
            ),
        }
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 0)
        positions = engine.storage.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["quantity"], 1.0)
        state = engine.storage.get_json_state(engine.POSITION_REVIEW_STATE_KEY, {})
        self.assertEqual(state[trade_id]["research_exit_count"], 1)
        with engine.storage._conn() as conn:
            close_count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='close'"
            ).fetchone()["c"]
            watch_count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='position_review_watch'"
            ).fetchone()["c"]
        self.assertEqual(close_count, 0)
        self.assertEqual(watch_count, 1)

    def test_engine_manage_open_positions_keeps_fast_alpha_close_override_runner_before_confirmation(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 99.9

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        trade_id = "research-fast-alpha-watch"
        engine.storage.insert_trade(
            {
                "id": trade_id,
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "initial_quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "fast alpha",
                "confidence": 0.7,
                "metadata": {
                    "pipeline_mode": "fast_alpha",
                    "entry_thesis": "fast_alpha_short_horizon",
                    "reviewed_action": "CLOSE",
                    "review_score": -0.35,
                    "fast_alpha_close_override": True,
                    "horizon_hours": 6,
                },
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: {
            "ok": True,
            "insight": ResearchInsight(
                symbol=symbol,
                suggested_action=SuggestedAction.CLOSE,
            ),
            "prediction": PredictionResult(
                symbol=symbol,
                up_probability=0.6,
                feature_count=10,
                model_version="test",
            ),
            "validation": SimpleNamespace(ok=True),
            "decision": SimpleNamespace(portfolio_rating="HOLD"),
            "review": SimpleNamespace(
                raw_action="HOLD",
                reviewed_action="CLOSE",
                review_score=-0.35,
                reasons=["trend_supportive"],
            ),
        }
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 0)
        positions = engine.storage.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["quantity"], 1.0)
        state = engine.storage.get_json_state(engine.POSITION_REVIEW_STATE_KEY, {})
        self.assertEqual(state[trade_id]["research_exit_count"], 1)
        with engine.storage._conn() as conn:
            close_count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='close'"
            ).fetchone()["c"]
            watch_count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='position_review_watch'"
            ).fetchone()["c"]
        self.assertEqual(close_count, 0)
        self.assertEqual(watch_count, 1)

    def test_engine_manage_open_positions_closes_fast_alpha_close_override_runner_on_material_deterioration(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 99.4

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
        trade_id = "research-fast-alpha-close"
        engine.storage.insert_trade(
            {
                "id": trade_id,
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "initial_quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "fast alpha",
                "confidence": 0.7,
                "metadata": {
                    "pipeline_mode": "fast_alpha",
                    "entry_thesis": "fast_alpha_short_horizon",
                    "reviewed_action": "CLOSE",
                    "review_score": -0.35,
                    "fast_alpha_close_override": True,
                    "horizon_hours": 6,
                },
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: {
            "ok": True,
            "insight": ResearchInsight(
                symbol=symbol,
                suggested_action=SuggestedAction.CLOSE,
            ),
            "prediction": PredictionResult(
                symbol=symbol,
                up_probability=0.55,
                feature_count=10,
                model_version="test",
            ),
            "validation": SimpleNamespace(ok=True),
            "decision": SimpleNamespace(portfolio_rating="HOLD"),
            "review": SimpleNamespace(
                raw_action="HOLD",
                reviewed_action="CLOSE",
                review_score=-0.55,
                reasons=["trend_supportive"],
            ),
        }
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 1)
        self.assertEqual(engine.storage.get_positions(), [])

    def test_engine_manage_open_positions_fast_alpha_de_risks_earlier_when_evidence_is_weak(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 100.3

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = (datetime.now(timezone.utc) - timedelta(minutes=36)).isoformat()
        trade_id = "fast-alpha-evidence-derisk"
        engine.storage.insert_trade(
            {
                "id": trade_id,
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "initial_quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "fast alpha evidence",
                "confidence": 0.7,
                "metadata": {
                    "pipeline_mode": "fast_alpha",
                    "entry_thesis": "fast_alpha_short_horizon",
                    "horizon_hours": 6,
                    "fast_alpha_de_risk_hours": 1.0,
                    "fast_alpha_de_risk_pnl_ratio": 0.004,
                    "model_evidence_scale": 0.55,
                    "reviewed_action": "HOLD",
                    "review_score": 0.0,
                },
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: None
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 0)
        positions = engine.storage.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["quantity"], 0.5)
        with engine.storage._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM execution_events WHERE event_type='close' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIn("evidence_de_risk", row["payload_json"])

    def test_engine_manage_open_positions_fast_alpha_uses_shorter_time_stop_when_evidence_is_weak(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 100.2

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
        trade_id = "fast-alpha-evidence-time-stop"
        engine.storage.insert_trade(
            {
                "id": trade_id,
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "initial_quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "fast alpha evidence",
                "confidence": 0.7,
                "metadata": {
                    "pipeline_mode": "fast_alpha",
                    "entry_thesis": "fast_alpha_short_horizon",
                    "horizon_hours": 6,
                    "fast_alpha_de_risk_hours": 1.0,
                    "fast_alpha_de_risk_pnl_ratio": 0.004,
                    "model_evidence_scale": 0.55,
                    "reviewed_action": "HOLD",
                    "review_score": 0.0,
                },
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: None
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 1)
        self.assertEqual(engine.storage.get_positions(), [])
        with engine.storage._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM execution_events WHERE event_type='close' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIn("evidence_time_stop", row["payload_json"])

    def test_engine_manage_open_positions_keeps_execution_runner_before_min_hold_with_strong_entry_thesis(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 99.7

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        trade_id = "research-execution-watch"
        engine.storage.insert_trade(
            {
                "id": trade_id,
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 0.5,
                "initial_quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "entry rationale",
                "confidence": 0.7,
                "metadata": {
                    "pipeline_mode": "execution",
                    "raw_action": "OPEN_LONG",
                    "reviewed_action": "OPEN_LONG",
                    "review_score": 0.28,
                    "review_reasons": ["xgb_strong", "liquidity_supportive"],
                    "entry_thesis": "high_conviction_long",
                    "entry_thesis_strength": "strong",
                    "entry_open_bias": True,
                },
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 0.5,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.storage.set_json_state(
            engine.POSITION_REVIEW_STATE_KEY,
            {
                trade_id: {
                    "trade_id": trade_id,
                    "symbol": "ETH/USDT",
                    "research_exit_count": 2,
                }
            },
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: {
            "ok": True,
            "insight": ResearchInsight(
                symbol=symbol,
                suggested_action=SuggestedAction.CLOSE,
            ),
            "prediction": PredictionResult(
                symbol=symbol,
                up_probability=0.58,
                feature_count=10,
                model_version="test",
            ),
            "validation": SimpleNamespace(ok=True),
            "decision": SimpleNamespace(portfolio_rating="HOLD"),
            "review": SimpleNamespace(
                raw_action="CLOSE",
                reviewed_action="CLOSE",
                review_score=-0.18,
                reasons=[],
            ),
        }
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 0)
        positions = engine.storage.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["quantity"], 0.5)
        state = engine.storage.get_json_state(engine.POSITION_REVIEW_STATE_KEY, {})
        self.assertEqual(state[trade_id]["research_exit_count"], 3)
        with engine.storage._conn() as conn:
            close_count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='close'"
            ).fetchone()["c"]
        self.assertEqual(close_count, 0)

    def test_engine_manage_open_positions_keeps_fast_alpha_strong_open_runner_before_confirmation(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 99.7

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = (datetime.now(timezone.utc) - timedelta(minutes=25)).isoformat()
        trade_id = "fast-alpha-strong-open-watch"
        engine.storage.insert_trade(
            {
                "id": trade_id,
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "initial_quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "fast alpha strong open",
                "confidence": 0.8,
                "metadata": {
                    "pipeline_mode": "fast_alpha",
                    "raw_action": "HOLD",
                    "reviewed_action": "OPEN_LONG",
                    "review_score": 0.35,
                    "review_reasons": [
                        "xgb_strong",
                        "trend_supportive",
                        "core_extreme_fear_liquidity_repair",
                    ],
                    "entry_thesis": "fast_alpha_short_horizon",
                    "entry_thesis_strength": "strong",
                    "entry_open_bias": True,
                    "horizon_hours": 6,
                },
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine._prepare_position_review = lambda symbol, now: {
            "ok": True,
            "insight": ResearchInsight(
                symbol=symbol,
                suggested_action=SuggestedAction.CLOSE,
            ),
            "prediction": PredictionResult(
                symbol=symbol,
                up_probability=0.72,
                feature_count=10,
                model_version="test",
            ),
            "validation": SimpleNamespace(ok=True),
            "decision": SimpleNamespace(portfolio_rating="HOLD"),
            "review": SimpleNamespace(
                raw_action="HOLD",
                reviewed_action="CLOSE",
                review_score=-0.20,
                reasons=[
                    "liquidity_weak",
                    "trend_supportive",
                    "regime_extreme_fear",
                    "fear_greed_extreme_fear",
                    "risk_warning_present",
                ],
            ),
        }
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 0)
        positions = engine.storage.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["quantity"], 1.0)
        state = engine.storage.get_json_state(engine.POSITION_REVIEW_STATE_KEY, {})
        self.assertEqual(state[trade_id]["research_exit_count"], 1)
        with engine.storage._conn() as conn:
            close_count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='close'"
            ).fetchone()["c"]
            watch_count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='position_review_watch'"
            ).fetchone()["c"]
        self.assertEqual(close_count, 0)
        self.assertEqual(watch_count, 1)

    def test_engine_position_review_fallback_reads_storage_symbol_variants(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 100.0

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        entry_time = datetime.now(timezone.utc).isoformat()
        engine.storage.insert_trade(
            {
                "id": "fallback-variant",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "rationale": "entry rationale",
                "confidence": 0.7,
            }
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": entry_time,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        for timeframe, base in (("1h", 100), ("4h", 120), ("1d", 140)):
            engine.storage.insert_ohlcv("BTC/USDT", timeframe, make_candles(20, base))

        captured = {}
        engine._prepare_position_review = lambda symbol, now: {
            "ok": False,
            "reason": "missing_candles",
        }
        engine.position_runtime.feature_pipeline = SimpleNamespace(
            build=lambda payload: captured.setdefault(
                "candle_lengths",
                (
                    len(payload.candles_1h),
                    len(payload.candles_4h),
                    len(payload.candles_1d),
                ),
            )
            or SimpleNamespace()
        )
        engine.position_runtime.predictor_for_symbol = lambda symbol: SimpleNamespace(
            predict=lambda snapshot: PredictionResult(
                symbol=symbol,
                up_probability=0.5,
                feature_count=1,
                model_version="fallback-test",
            )
        )
        engine.position_runtime.fallback_research = lambda symbol: ResearchInsight(
            symbol=symbol,
            suggested_action=SuggestedAction.HOLD,
        )
        engine.decision_engine.evaluate_exit = lambda **kwargs: []
        engine.notifier = SimpleNamespace(notify_trade_close=lambda *args, **kwargs: None)

        closed = engine._manage_open_positions(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
            SimpleNamespace(),
        )

        self.assertEqual(closed, 0)
        self.assertEqual(captured["candle_lengths"], (20, 20, 20))

    def test_engine_enables_live_orders_only_when_gate_is_open(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "live"
        settings.app.allow_live_orders = True

        with patch.object(engine_module.CryptoAIV2Engine, "_ensure_live_readiness", lambda self: None):
            engine = engine_module.CryptoAIV2Engine(settings)
        self.assertEqual(engine.executor.__class__.__name__, "LiveTrader")
        self.assertTrue(engine.executor.enabled)

    def test_engine_blocks_live_orders_when_live_readiness_fails(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "live"
        settings.app.allow_live_orders = True

        with self.assertRaises(engine_module.LiveReadinessError):
            engine_module.CryptoAIV2Engine(settings)

    def test_engine_registers_feishu_channels_when_configured(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.notifications.feishu_webhook_url = "https://open.feishu.cn/fake"
        settings.notifications.feishu_critical_webhook_url = "https://open.feishu.cn/fake-critical"

        engine = engine_module.CryptoAIV2Engine(settings)
        channel_names = [channel.name for channel in engine.notifier.channels]
        self.assertIn("feishu_webhook", channel_names)
        self.assertIn("critical_feishu_webhook", channel_names)

    def test_engine_uses_binance_market_collector_when_provider_is_binance(self):
        import core.engine as engine_module

        class FakeBinanceMarket:
            def __init__(self, storage, proxy=None, api_key="", api_secret=""):
                self.storage = storage
                self.proxy = proxy
                self.api_key = api_key
                self.api_secret = api_secret

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.provider = "binance"

        with patch.object(engine_module, "BinanceMarketDataCollector", FakeBinanceMarket):
            engine = engine_module.CryptoAIV2Engine(settings)
        self.assertEqual(engine.market.primary_provider, "binance")
        self.assertEqual(engine.market.primary.__class__.__name__, "FakeBinanceMarket")

    def test_engine_market_failover_uses_secondary_provider_for_snapshot_reads(self):
        import core.engine as engine_module

        class FailingPrimaryMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def measure_latency(self, symbol):
                raise RuntimeError("okx_latency_down")

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                raise RuntimeError("okx_ohlcv_down")

            def fetch_funding_rate(self, symbol):
                raise RuntimeError("okx_funding_down")

            def summarize_order_book_depth(self, symbol, depth=5):
                raise RuntimeError("okx_depth_down")

        class SecondaryMarket:
            def __init__(self, storage, proxy=None, api_key="", api_secret=""):
                self.storage = storage

            def measure_latency(self, symbol):
                return {"latency_seconds": 0.0, "status": "ok"}

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                base = 100 if timeframe == "1h" else 120 if timeframe == "4h" else 140
                return make_candles(240, base)

            def fetch_funding_rate(self, symbol):
                return 0.0

            def summarize_order_book_depth(self, symbol, depth=5):
                return SimpleNamespace(
                    symbol=symbol,
                    bid_notional_top5=25000.0,
                    ask_notional_top5=24000.0,
                    bid_ask_spread_pct=0.001,
                    depth_imbalance=0.02,
                    large_bid_notional=6000.0,
                    large_ask_notional=5800.0,
                    large_order_net_notional=200.0,
                )

        class FakeSentiment:
            def __init__(self, storage, settings=None):
                self.storage = storage

            def get_latest_sentiment(self, symbol="BTC/USDT"):
                return {"value": 58, "summary": "neutral"}

        class FakeNews:
            def __init__(self, settings=None):
                self.settings = settings

            def get_summary(self, symbol):
                return SimpleNamespace(
                    summary="market stable",
                    sources=["CoinDesk"],
                    coverage_score=0.7,
                    service_health_score=1.0,
                    source_status={"CoinDesk": "matched"},
                )

        class FakeOnchain:
            def __init__(self, settings=None):
                self.settings = settings

            def get_summary(self, symbol):
                return SimpleNamespace(
                    summary="onchain supportive",
                    netflow_score=0.2,
                    whale_score=0.1,
                    source="coinmetrics",
                )

        class FakeResearch:
            def __init__(self, settings, clients=None):
                pass

            def analyze(self, **kwargs):
                return ResearchInsight(
                    symbol=kwargs["symbol"],
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.2,
                    confidence=0.6,
                    risk_warning=[],
                    key_reason=["raw_signal"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                )

        class FakeRegimeDetector:
            def detect(self, candles, fear_greed=None):
                return SimpleNamespace(state="BULL_TREND")

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FailingPrimaryMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", SecondaryMarket), \
             patch.object(engine_module, "SentimentCollector", FakeSentiment), \
             patch.object(engine_module, "NewsService", FakeNews), \
             patch.object(engine_module, "OnchainService", FakeOnchain), \
             patch.object(engine_module, "ResearchLLMAnalyzer", FakeResearch), \
             patch.object(engine_module, "MarketRegimeDetector", lambda: FakeRegimeDetector()):
            engine = engine_module.CryptoAIV2Engine(settings)
            engine.predictor = SimpleNamespace(
                predict=lambda snapshot: PredictionResult(
                    symbol=snapshot.symbol,
                    up_probability=0.82,
                    feature_count=len(snapshot.values),
                    model_version="xgboost_v3_BTC_USDT.json",
                )
            )
            snapshot = engine._prepare_symbol_snapshot("BTC/USDT", datetime.now(timezone.utc))

        self.assertIsNotNone(snapshot)
        route = self.storage.get_json_state("market_data_last_route", {})
        self.assertEqual(route["selected_provider"], "binance")
        stats = self.storage.get_json_state("market_data_failover_stats", {})
        self.assertGreaterEqual(stats["fetch_historical_ohlcv"]["fallback_used"], 1)
        with self.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='market_data_failover'"
            ).fetchone()["c"]
        self.assertGreaterEqual(count, 1)

    def test_engine_records_ab_test_run_when_enabled(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                base = 100 if timeframe == "1h" else 120 if timeframe == "4h" else 140
                return make_candles(240, base)

            def measure_latency(self, symbol):
                return {"latency_seconds": 0.0}

        class FakeSentiment:
            def __init__(self, storage, settings=None):
                self.storage = storage

            def get_latest_sentiment(self, symbol="BTC/USDT"):
                return {"value": 60, "summary": "bullish"}

        class FakeNews:
            def __init__(self, settings=None):
                self.settings = settings

            def get_summary(self, symbol):
                return SimpleNamespace(
                    summary=f"{symbol} news",
                    trending_symbols=[symbol],
                    sources=["CoinDesk"],
                )

        class FakeResearch:
            def __init__(self, settings, clients=None):
                pass

            def analyze(self, **kwargs):
                return ResearchInsight(
                    symbol=kwargs["symbol"],
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.2,
                    confidence=0.5,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                )

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.ab_testing.enabled = True
        settings.ab_testing.execute_challenger_live = False

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "SentimentCollector", FakeSentiment), \
             patch.object(engine_module, "NewsService", FakeNews), \
             patch.object(engine_module, "ResearchLLMAnalyzer", FakeResearch):
            engine = engine_module.CryptoAIV2Engine(settings)
            model_dir = Path(self.db_path).parent / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            btc_model = model_dir / "xgboost_v2_BTC_USDT.json"
            btc_model.write_text("{}", encoding="utf-8")
            engine.storage.insert_training_run(
                {
                    "symbol": "BTC/USDT",
                    "rows": settings.training.minimum_training_rows,
                    "feature_count": 10,
                    "positives": 5,
                    "negatives": 5,
                    "model_path": str(btc_model),
                    "trained_with_xgboost": True,
                    "holdout_accuracy": 0.6,
                }
            )
            engine.storage.set_json_state(
                engine.EXECUTION_SYMBOLS_STATE_KEY,
                ["BTC/USDT"],
            )
            engine.performance = SimpleNamespace(
                build=lambda: SimpleNamespace(
                    prediction_eval_count=10,
                    xgboost_accuracy_pct=70.0,
                    degradation_status="healthy",
                    degradation_reason="",
                    recommended_xgboost_threshold=engine.decision_engine.xgboost_threshold,
                    recommended_final_score_threshold=0.55,
                    fusion_accuracy_pct=70.0,
                )
            )
            engine.predictor = SimpleNamespace(
                predict=lambda snapshot: PredictionResult(
                    symbol=snapshot.symbol,
                    up_probability=0.60,
                    feature_count=len(snapshot.values),
                    model_version="champion",
                )
            )
            engine.challenger_predictor = SimpleNamespace(
                predict=lambda snapshot: PredictionResult(
                    symbol=snapshot.symbol,
                    up_probability=0.90,
                    feature_count=len(snapshot.values),
                    model_version="challenger",
                )
            )
            engine.run_once()

        with self.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM ab_test_runs"
            ).fetchone()["c"]
            prediction_modes = conn.execute(
                "SELECT json_extract(decision_json, '$.pipeline_mode') AS pipeline_mode "
                "FROM prediction_runs ORDER BY id ASC"
            ).fetchall()
        self.assertGreaterEqual(count, 1)
        self.assertIn("execution", [row["pipeline_mode"] for row in prediction_modes])
        self.assertIn(
            "challenger_shadow",
            [row["pipeline_mode"] for row in prediction_modes],
        )

    def test_engine_accuracy_guard_triggers_retraining_and_cooldown(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.risk.model_accuracy_min_samples = 5
        settings.risk.model_accuracy_floor_pct = 55.0
        engine = engine_module.CryptoAIV2Engine(settings)

        retrain_calls = []
        engine.performance = SimpleNamespace(
            build=lambda: SimpleNamespace(
                prediction_eval_count=8,
                xgboost_accuracy_pct=40.0,
            )
        )
        engine._train_models_if_due = lambda now, force, reason: retrain_calls.append(
            (force, reason)
        )
        engine._cooldown_until = None

        now = datetime.now(timezone.utc)
        engine._enforce_accuracy_guard(now)

        self.assertEqual(retrain_calls, [(False, "accuracy_guard")])
        self.assertIsNone(engine._cooldown_until)
        self.assertEqual(engine.settings.app.runtime_mode, "paper")
        self.assertIsNotNone(engine.storage.get_state("last_accuracy_guard_triggered"))

    def test_engine_should_retrain_symbol_requires_new_rows_or_stale_model(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.training.retrain_interval_days = 30
        engine = engine_module.CryptoAIV2Engine(settings)
        model_path = Path(self.db_path).parent / "existing_model.json"
        model_path.write_text("{}", encoding="utf-8")
        training_window = {
            "start_timestamp": 1_710_000_000_000,
            "end_timestamp": 1_710_007_200_000,
        }
        engine.storage.insert_training_run(
            {
                "symbol": "BTC/USDT",
                "rows": 500,
                "feature_count": 10,
                "positives": 250,
                "negatives": 250,
                "model_path": str(model_path),
                "trained_with_xgboost": True,
                "holdout_accuracy": 0.6,
                "dataset_start_timestamp_ms": training_window["start_timestamp"],
                "dataset_end_timestamp_ms": training_window["end_timestamp"],
            }
        )
        engine.trainer = SimpleNamespace(
            training_data_signature=lambda symbol: {
                "rows": 500,
                "start_timestamp": training_window["start_timestamp"],
                "end_timestamp": training_window["end_timestamp"],
            }
        )

        should_train, reason = engine._should_retrain_symbol(
            "BTC/USDT",
            datetime.now(timezone.utc),
            force=False,
        )

        self.assertFalse(should_train)
        self.assertEqual(reason, "no_new_training_data")

    def test_engine_should_retrain_symbol_when_training_window_advances(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.training.retrain_interval_days = 30
        engine = engine_module.CryptoAIV2Engine(settings)
        model_path = Path(self.db_path).parent / "rolling_window_model.json"
        model_path.write_text("{}", encoding="utf-8")
        engine.storage.insert_training_run(
            {
                "symbol": "BTC/USDT",
                "rows": 500,
                "feature_count": 10,
                "positives": 250,
                "negatives": 250,
                "model_path": str(model_path),
                "trained_with_xgboost": True,
                "holdout_accuracy": 0.6,
                "dataset_start_timestamp_ms": 1_710_000_000_000,
                "dataset_end_timestamp_ms": 1_710_007_200_000,
            }
        )
        engine.trainer = SimpleNamespace(
            training_data_signature=lambda symbol: {
                "rows": 500,
                "start_timestamp": 1_710_014_400_000,
                "end_timestamp": 1_710_021_600_000,
            }
        )

        should_train, reason = engine._should_retrain_symbol(
            "BTC/USDT",
            datetime.now(timezone.utc),
            force=False,
        )

        self.assertTrue(should_train)
        self.assertTrue(reason.startswith("new_training_window:"))

    def test_engine_should_retrain_symbol_when_model_artifact_is_missing(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        engine.storage.insert_training_run(
            {
                "symbol": "BTC/USDT",
                "rows": settings.training.minimum_training_rows,
                "feature_count": 10,
                "positives": 250,
                "negatives": 250,
                "model_path": str(Path(self.db_path).parent / "missing_model.json"),
                "trained_with_xgboost": True,
                "holdout_accuracy": 0.6,
            }
        )
        engine.trainer = SimpleNamespace(
            count_training_rows=lambda symbol: settings.training.minimum_training_rows
        )

        should_train, reason = engine._should_retrain_symbol(
            "BTC/USDT",
            datetime.now(timezone.utc),
            force=False,
        )

        self.assertTrue(should_train)
        self.assertEqual(reason, "missing_model_artifact")

    def test_engine_runtime_model_path_prefers_active_model_from_latest_training_row(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        challenger_model_path = model_dir / "xgboost_challenger_BTC_USDT.json"
        active_model_path.write_text("active", encoding="utf-8")
        challenger_model_path.write_text("candidate", encoding="utf-8")

        engine.storage.insert_training_run(
            {
                "symbol": "BTC/USDT",
                "rows": settings.training.minimum_training_rows,
                "feature_count": 10,
                "positives": 250,
                "negatives": 250,
                "model_path": str(challenger_model_path),
                "active_model_path": str(active_model_path),
                "challenger_model_path": str(challenger_model_path),
                "trained_with_xgboost": True,
                "promoted_to_active": False,
                "promotion_status": "rejected",
                "promotion_reason": "candidate_lower_holdout_accuracy",
                "holdout_accuracy": 0.6,
                "candidate_holdout_accuracy": 0.5,
            }
        )

        self.assertEqual(
            engine._runtime_model_path_for_symbol("BTC/USDT"),
            active_model_path,
        )

    def test_engine_run_once_does_not_reopen_symbol_closed_in_same_cycle(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        captured_active_symbols = []
        engine.preflight_runtime.prepare_cycle_symbols = lambda now: {
            "active_symbols": ["BTC/USDT"],
            "shadow_symbols": [],
        }
        engine.preflight_runtime.run_preflight = lambda **kwargs: {
            "abort": False,
            "positions": [],
            "account": SimpleNamespace(circuit_breaker_active=False),
            "reconciliation": SimpleNamespace(status="ok"),
        }
        engine.cycle_runtime.start_cycle = lambda **kwargs: 1
        engine.cycle_runtime.complete_cycle = lambda *args, **kwargs: None
        engine._account_state = lambda now, positions: SimpleNamespace(
            circuit_breaker_active=False
        )

        def fake_manage(now, positions, account):
            engine.storage.insert_execution_event(
                "close",
                "BTC/USDT",
                {"reason": "same_cycle_close"},
            )
            return 1

        engine._manage_open_positions = fake_manage
        engine.get_active_symbols = lambda force_refresh=False, now=None: ["BTC/USDT"]
        engine.analysis_runtime.run_active_symbols = lambda **kwargs: (
            captured_active_symbols.append(list(kwargs["active_symbols"]))
            or {
                "opened_positions": 0,
                "positions": [],
                "account": SimpleNamespace(circuit_breaker_active=False),
            }
        )
        engine.analysis_runtime.run_shadow_symbols = lambda **kwargs: None
        engine._evaluate_matured_predictions = lambda now: {"evaluated_count": 0}
        engine._evaluate_shadow_trades = lambda now: {"evaluated_count": 0}
        engine._generate_reports = lambda now: None
        engine._run_loop_model_maintenance = lambda now: None

        engine.run_once()

        self.assertEqual(captured_active_symbols, [[]])

    def test_engine_approve_manual_recovery_clears_flags(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        engine.storage.set_state("manual_recovery_required", "true")
        engine.storage.set_state("manual_recovery_approved", "false")
        engine.storage.set_state("manual_recovery_reason", "daily_loss_limit")
        engine._circuit_breaker_active = True
        result = engine.approve_manual_recovery()
        self.assertEqual(result["status"], "approved")
        self.assertEqual(engine.storage.get_state("manual_recovery_required"), "false")
        self.assertEqual(engine.storage.get_state("manual_recovery_approved"), "true")
        self.assertFalse(engine._circuit_breaker_active)

    def test_engine_sets_nextgen_live_operator_request(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        request = engine.set_nextgen_autonomy_live_operator_request(
            requested_live=True,
            whitelist=["BTC/USDT"],
            max_active_runtimes=2,
            reason="test",
        )

        self.assertTrue(request["requested_live"])
        self.assertEqual(request["whitelist"], ("BTC/USDT:USDT",))
        self.assertEqual(request["max_active_runtimes"], 2)
        self.assertEqual(
            engine.get_nextgen_autonomy_live_operator_request()["requested_live"],
            True,
        )

    def test_engine_manual_recovery_triggers_nextgen_live_guard_callback(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        calls = []

        def fake_run_nextgen_autonomy_live(**kwargs):
            calls.append(dict(kwargs))
            return {"status": "ok"}

        engine.run_nextgen_autonomy_live = fake_run_nextgen_autonomy_live

        engine._trigger_manual_recovery("market_data_latency", "latency_seconds=6.000")

        self.assertEqual(
            calls,
            [
                {
                    "requested_live": False,
                    "trigger": "manual_recovery_required",
                    "trigger_reason": "market_data_latency",
                    "trigger_details": "latency_seconds=6.000",
                }
            ],
        )

    def test_engine_manual_recovery_callback_failure_is_captured(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        def fail_run_nextgen_autonomy_live(**kwargs):
            raise RuntimeError("nextgen_guard_boom")

        engine.run_nextgen_autonomy_live = fail_run_nextgen_autonomy_live

        engine._trigger_manual_recovery("market_data_latency", "latency_seconds=6.000")

        with engine.storage._conn() as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM execution_events
                WHERE event_type = 'nextgen_autonomy_live_guard_callback_failed'
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["trigger"], "manual_recovery_required")
        self.assertEqual(payload["reason"], "market_data_latency")
        self.assertIn("nextgen_guard_boom", payload["error"])

    def test_engine_applies_runtime_overrides_from_state(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        storage = Storage(self.db_path)
        storage.set_json_state(
            engine_module.CryptoAIV2Engine.RUNTIME_OVERRIDE_STATE_KEY,
            {
                "xgboost_probability_threshold": 0.76,
                "final_score_threshold": 0.84,
                "min_liquidity_ratio": 0.95,
                "sentiment_weight": 0.35,
                "fixed_stop_loss_pct": 0.006,
                "take_profit_levels": [0.06, 0.1],
            },
        )

        engine = engine_module.CryptoAIV2Engine(settings)
        self.assertEqual(engine.settings.model.xgboost_probability_threshold, 0.76)
        self.assertEqual(engine.settings.model.final_score_threshold, 0.84)
        self.assertEqual(engine.decision_engine.min_liquidity_ratio, 0.95)
        self.assertEqual(engine.decision_engine.sentiment_weight, 0.35)
        self.assertEqual(engine.settings.strategy.fixed_stop_loss_pct, 0.006)
        self.assertEqual(engine.settings.strategy.take_profit_levels, [0.06, 0.1])
        effective = engine.storage.get_json_state(
            engine_module.CryptoAIV2Engine.RUNTIME_EFFECTIVE_STATE_KEY,
            {},
        )
        self.assertEqual(effective["overrides"]["xgboost_probability_threshold"], 0.76)

    def test_engine_model_degradation_disables_new_entries(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        engine.performance = SimpleNamespace(
            build=lambda: SimpleNamespace(
                degradation_status="disabled",
                degradation_reason="live_accuracy_below_disable_floor",
                recommended_xgboost_threshold=0.80,
                recommended_final_score_threshold=0.90,
                xgboost_accuracy_pct=42.0,
                fusion_accuracy_pct=41.0,
            )
        )
        now = datetime.now(timezone.utc)
        engine._apply_model_degradation(now)
        self.assertTrue(engine._model_trading_disabled)
        self.assertEqual(engine.storage.get_state("model_degradation_status"), "disabled")
        self.assertEqual(engine.decision_engine.xgboost_threshold, 0.80)
        self.assertEqual(engine.decision_engine.final_score_threshold, 0.90)

    def test_engine_model_disable_triggers_nextgen_live_guard_callback(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        calls = []
        engine.performance = SimpleNamespace(
            build=lambda: SimpleNamespace(
                degradation_status="disabled",
                degradation_reason="live_accuracy_below_disable_floor",
                recommended_xgboost_threshold=0.80,
                recommended_final_score_threshold=0.90,
                xgboost_accuracy_pct=42.0,
                fusion_accuracy_pct=41.0,
            )
        )

        def fake_run_nextgen_autonomy_live(**kwargs):
            calls.append(dict(kwargs))
            return {"status": "ok"}

        engine.run_nextgen_autonomy_live = fake_run_nextgen_autonomy_live

        engine._apply_model_degradation(datetime.now(timezone.utc))

        self.assertEqual(
            calls,
            [
                {
                    "requested_live": False,
                    "trigger": "model_degradation_disabled",
                    "trigger_reason": "live_accuracy_below_disable_floor",
                    "trigger_details": (
                        "xgboost_accuracy_pct=42.00;"
                        "fusion_accuracy_pct=41.00"
                    ),
                }
            ],
        )

    def test_engine_persists_nextgen_live_queue_summary_in_execution_event(self):
        import core.engine as engine_module
        import nextgen_evolution.live_cycle as live_cycle_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        class FakeRunner:
            def __init__(self, storage, settings=None):
                self.storage = storage
                self.settings = settings

            def run(self, **kwargs):
                return {
                    "status": "ok",
                    "reason": "",
                    "requested_live": True,
                    "operator_request": {"requested_live": True},
                    "effective_live": False,
                    "force_flatten": False,
                    "autonomy_cycle_id": 12,
                    "intent_count": 3,
                    "action_counts": {"hold": 2, "close": 1},
                    "intent_status_counts": {"dry_run": 3},
                    "repair_queue_requested_size": 5,
                    "repair_queue_dropped_count": 1,
                    "repair_queue_dropped_runtime_ids": ["runtime_x:seed"],
                    "repair_queue_hold_priority_count": 2,
                    "repair_queue_postponed_rebuild_count": 1,
                    "repair_queue_reprioritized_count": 4,
                    "repair_queue_dropped_active": True,
                    "repair_queue_hold_priority_active": True,
                    "repair_queue_postponed_rebuild_active": True,
                    "repair_queue_reprioritized_active": True,
                }

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket), \
             patch.object(live_cycle_module, "AutonomyLiveCycleRunner", FakeRunner):
            engine = engine_module.CryptoAIV2Engine(settings)
            result = engine.run_nextgen_autonomy_live(
                requested_live=True,
                trigger="scheduler",
                trigger_reason="periodic",
                trigger_details="job=nextgen_live",
            )

        self.assertEqual(result["repair_queue_hold_priority_count"], 2)
        self.assertEqual(result["repair_queue_requested_size"], 5)
        self.assertEqual(result["repair_queue_dropped_count"], 1)
        with engine.storage._conn() as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM execution_events
                WHERE event_type = 'nextgen_autonomy_live_run'
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["repair_queue_requested_size"], 5)
        self.assertEqual(payload["repair_queue_dropped_count"], 1)
        self.assertEqual(payload["repair_queue_dropped_runtime_ids"], ["runtime_x:seed"])
        self.assertEqual(payload["repair_queue_hold_priority_count"], 2)
        self.assertEqual(payload["repair_queue_postponed_rebuild_count"], 1)
        self.assertEqual(payload["repair_queue_reprioritized_count"], 4)
        self.assertTrue(payload["repair_queue_dropped_active"])
        self.assertTrue(payload["repair_queue_hold_priority_active"])
        self.assertTrue(payload["repair_queue_postponed_rebuild_active"])
        self.assertTrue(payload["repair_queue_reprioritized_active"])

    def test_engine_model_degradation_tightens_thresholds_in_paper_mode(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "paper"
        engine = engine_module.CryptoAIV2Engine(settings)
        engine.performance = SimpleNamespace(
            build=lambda: SimpleNamespace(
                degradation_status="degraded",
                degradation_reason="paper_mode_accuracy_guard",
                recommended_xgboost_threshold=0.78,
                recommended_final_score_threshold=0.60,
                xgboost_accuracy_pct=49.0,
                fusion_accuracy_pct=47.0,
            )
        )

        engine._apply_model_degradation(datetime.now(timezone.utc))

        self.assertFalse(engine._model_trading_disabled)
        self.assertEqual(engine.storage.get_state("model_degradation_status"), "degraded")
        self.assertEqual(engine.decision_engine.xgboost_threshold, 0.78)
        self.assertEqual(engine.decision_engine.final_score_threshold, 0.60)

    def test_engine_blocks_bearish_news_entries(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                base = 100 if timeframe == "1h" else 120 if timeframe == "4h" else 140
                return make_candles(240, base)

            def fetch_latest_price(self, symbol):
                return 100.0

        class FakeSentiment:
            def __init__(self, storage, settings=None):
                self.storage = storage

            def get_latest_sentiment(self, symbol="BTC/USDT"):
                return {"value": 60, "summary": "bullish"}

        class FakeNews:
            def __init__(self, settings=None):
                self.settings = settings

            def get_summary(self, symbol):
                return SimpleNamespace(
                    summary=f"{symbol} hit by major hack and liquidation risk",
                    trending_symbols=[symbol],
                )

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "SentimentCollector", FakeSentiment), \
             patch.object(engine_module, "NewsService", FakeNews):
            engine = engine_module.CryptoAIV2Engine(settings)
            snapshot = engine._prepare_symbol_snapshot("BTC/USDT", datetime.now(timezone.utc))
        self.assertIsNone(snapshot)
        with engine.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='bearish_news_block'"
            ).fetchone()["c"]
        self.assertEqual(count, 1)

    def test_engine_does_not_block_unrelated_bearish_news(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        summary = "Cointelegraph headlines: Resolv says no assets lost as DeFi protocols respond to $24M USR exploit"
        self.assertFalse(engine._contains_bearish_news_risk("SOL/USDT", summary))

    def test_engine_does_not_block_bullish_bear_liquidation_headline(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        summary = (
            "Cointelegraph headlines: Bitcoin tops $72K after $280M liquidation targets bears: "
            "Will the fragile truce hold? | Bitcoin has 3-5 years to prepare for quantum risk, "
            "says Bernstein | Price predictions 4/8: BTC, ETH, XRP, BNB, SOL, DOGE, HYPE, ADA, BCH, LINK"
        )
        self.assertFalse(engine._contains_bearish_news_risk("BTC/USDT", summary))

    def test_engine_blocks_marketwide_bearish_news_without_symbol_mention(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        summary = "Crypto market faces liquidation risk after macro shock and stablecoin stress"
        self.assertTrue(engine._contains_bearish_news_risk("SOL/USDT", summary))

    def test_engine_rejects_poor_data_quality(self):
        import core.engine as engine_module

        class GapMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                candles = make_candles(10, 100)
                if timeframe == "1h":
                    candles[5]["timestamp"] += 3 * 3600000
                return candles

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.data_quality_max_missing_ratio = 0.001
        with patch.object(engine_module, "OKXMarketDataCollector", GapMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", GapMarket):
            engine = engine_module.CryptoAIV2Engine(settings)
            snapshot = engine._prepare_symbol_snapshot("BTC/USDT", datetime.now(timezone.utc))
        self.assertIsNone(snapshot)
        with self.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='data_quality_failure'"
            ).fetchone()["c"]
        self.assertEqual(count, 1)

    def test_engine_blocks_high_funding_rate_entries(self):
        import core.engine as engine_module

        class FundingMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                base = 100 if timeframe == "1h" else 120 if timeframe == "4h" else 140
                return make_candles(240, base)

            def fetch_funding_rate(self, symbol):
                return 0.05

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FundingMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FundingMarket):
            engine = engine_module.CryptoAIV2Engine(settings)
            snapshot = engine._prepare_symbol_snapshot("BTC/USDT", datetime.now(timezone.utc))
        self.assertIsNone(snapshot)
        with self.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='funding_rate_block'"
            ).fetchone()["c"]
        self.assertEqual(count, 1)

    def test_engine_blocks_symbol_when_market_latency_is_high(self):
        import core.engine as engine_module

        class SlowMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def measure_latency(self, symbol):
                return {"latency_seconds": 6.0}

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                return make_candles(50, 100)

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.data_latency_warning_seconds = 5
        with patch.object(engine_module, "OKXMarketDataCollector", SlowMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", SlowMarket):
            engine = engine_module.CryptoAIV2Engine(settings)
            snapshot = engine._prepare_symbol_snapshot(
                "BTC/USDT",
                datetime.now(timezone.utc),
            )
        self.assertIsNone(snapshot)
        with self.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events "
                "WHERE event_type='symbol_market_latency_block'"
            ).fetchone()["c"]
        self.assertEqual(count, 1)

    def test_engine_retries_market_data_fetch_before_failing_snapshot(self):
        import core.engine as engine_module

        class FlakyMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage
                self.calls = {}

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                key = (symbol, timeframe)
                self.calls[key] = self.calls.get(key, 0) + 1
                if self.calls[key] == 1:
                    raise RuntimeError(f"temporary_{timeframe}_failure")
                base = 100 if timeframe == "1h" else 120 if timeframe == "4h" else 140
                return make_candles(240, base)

            def fetch_funding_rate(self, symbol):
                return 0.0

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.market_data_retry_count = 2
        settings.exchange.market_data_retry_delay_seconds = 0.0
        with patch.object(engine_module, "OKXMarketDataCollector", FlakyMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FlakyMarket):
            engine = engine_module.CryptoAIV2Engine(settings)
            engine.news.get_summary = lambda symbol: SimpleNamespace(
                summary="market calm",
                sources=[],
                coverage_score=0.0,
                service_health_score=1.0,
            )
            snapshot = engine._prepare_symbol_snapshot(
                "BTC/USDT",
                datetime.now(timezone.utc),
            )
        self.assertIsNotNone(snapshot)

    def test_engine_blocks_cross_validation_conflict(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                base = 100 if timeframe == "1h" else 120 if timeframe == "4h" else 140
                return make_candles(240, base)

        class FakeSentiment:
            def __init__(self, storage, settings=None):
                self.storage = storage

            def get_latest_sentiment(self, symbol="BTC/USDT"):
                return {
                    "value": 80,
                    "summary": "bullish",
                    "lunarcrush_sentiment": -0.7,
                }

        class FakeNews:
            def __init__(self, settings=None):
                self.settings = settings

            def get_summary(self, symbol):
                return SimpleNamespace(
                    summary=f"{symbol} news",
                    trending_symbols=[symbol],
                    sources=["CoinDesk"],
                )

        class FakeOnchain:
            def __init__(self, settings=None):
                self.settings = settings

            def get_summary(self, symbol):
                return SimpleNamespace(
                    summary="onchain weak",
                    netflow_score=-0.4,
                    whale_score=0.2,
                )

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "SentimentCollector", FakeSentiment), \
             patch.object(engine_module, "NewsService", FakeNews), \
             patch.object(engine_module, "OnchainService", FakeOnchain):
            engine = engine_module.CryptoAIV2Engine(settings)
            engine.predictor = SimpleNamespace(
                predict=lambda snapshot: PredictionResult(
                    symbol=snapshot.symbol,
                    up_probability=0.82,
                    feature_count=len(snapshot.values),
                    model_version="xgboost_v3_BTC_USDT.json",
                )
            )
            snapshot = engine._prepare_symbol_snapshot("BTC/USDT", datetime.now(timezone.utc))
        self.assertIsNone(snapshot)
        with self.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='cross_validation'"
            ).fetchone()["c"]
        self.assertEqual(count, 1)

    def test_engine_blocks_degraded_research_input_consistency(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                base = 100 if timeframe == "1h" else 120 if timeframe == "4h" else 140
                return make_candles(240, base)

        class FakeSentiment:
            def __init__(self, storage, settings=None):
                self.storage = storage

            def get_latest_sentiment(self, symbol="BTC/USDT"):
                return {"value": 52, "summary": "neutral"}

        class FakeNews:
            def __init__(self, settings=None):
                self.settings = settings

            def get_summary(self, symbol):
                return SimpleNamespace(
                    summary="neutral",
                    sources=[],
                    coverage_score=0.0,
                    service_health_score=0.0,
                    source_status={
                        "CoinDesk": "unavailable",
                        "Cointelegraph": "unavailable",
                        "Jin10": "unavailable",
                    },
                )

        class FakeOnchain:
            def __init__(self, settings=None):
                self.settings = settings

            def get_summary(self, symbol):
                return SimpleNamespace(
                    summary="fallback onchain",
                    netflow_score=0.0,
                    whale_score=0.0,
                    source="fallback",
                )

        class FakeResearch:
            def __init__(self, settings, clients=None):
                pass

            def analyze(self, **kwargs):
                raise AssertionError("research should not run on degraded input")

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "SentimentCollector", FakeSentiment), \
             patch.object(engine_module, "NewsService", FakeNews), \
             patch.object(engine_module, "OnchainService", FakeOnchain), \
             patch.object(engine_module, "ResearchLLMAnalyzer", FakeResearch):
            engine = engine_module.CryptoAIV2Engine(settings)
            snapshot = engine._prepare_symbol_snapshot("BTC/USDT", datetime.now(timezone.utc))

        self.assertIsNone(snapshot)
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM execution_events "
                "WHERE event_type='research_input_consistency' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIsNotNone(row)
        payload = json.loads(row["payload_json"])
        self.assertFalse(payload["ok"])
        self.assertIn("news_services_unavailable", payload["details"]["reasons"])

    def test_engine_records_research_review_for_snapshot(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                base = 100 if timeframe == "1h" else 120 if timeframe == "4h" else 140
                return make_candles(240, base)

        class FakeSentiment:
            def __init__(self, storage, settings=None):
                self.storage = storage

            def get_latest_sentiment(self, symbol="BTC/USDT"):
                return {"value": 60, "summary": "bullish", "lunarcrush_sentiment": 0.3}

        class FakeNews:
            def __init__(self, settings=None):
                self.settings = settings

            def get_summary(self, symbol):
                return SimpleNamespace(
                    summary=f"{symbol} adoption momentum improving",
                    trending_symbols=[symbol],
                    sources=["CoinDesk"],
                )

        class FakeResearch:
            def __init__(self, settings, clients=None):
                pass

            def analyze(self, **kwargs):
                return ResearchInsight(
                    symbol=kwargs["symbol"],
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.2,
                    confidence=0.6,
                    risk_warning=[],
                    key_reason=["raw_signal"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                )

        class FakeRegimeDetector:
            def detect(self, candles, fear_greed=None):
                return SimpleNamespace(state="BULL_TREND")

        class FakeOnchain:
            def __init__(self, settings=None):
                self.settings = settings

            def get_summary(self, symbol):
                return SimpleNamespace(
                    summary="onchain supportive",
                    netflow_score=1.0,
                    whale_score=0.2,
                )

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "SentimentCollector", FakeSentiment), \
             patch.object(engine_module, "NewsService", FakeNews), \
             patch.object(engine_module, "ResearchLLMAnalyzer", FakeResearch), \
             patch.object(engine_module, "OnchainService", FakeOnchain), \
             patch.object(engine_module, "MarketRegimeDetector", lambda: FakeRegimeDetector()):
            engine = engine_module.CryptoAIV2Engine(settings)
            engine.predictor = SimpleNamespace(
                predict=lambda snapshot: PredictionResult(
                    symbol=snapshot.symbol,
                    up_probability=0.82,
                    feature_count=len(snapshot.values),
                    model_version="xgboost_v3_BTC_USDT.json",
                )
            )
            snapshot = engine._prepare_symbol_snapshot("BTC/USDT", datetime.now(timezone.utc))
        self.assertIsNotNone(snapshot)
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM execution_events "
                "WHERE event_type='research_review' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertIn('"raw_action": "OPEN_LONG"', row["payload_json"])

    def test_engine_shadow_observation_symbols_exclude_execution_pool(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.candidate_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            model_path = model_dir / f"xgboost_v2_{symbol.replace('/', '_')}.json"
            model_path.write_text("{}", encoding="utf-8")
            engine.storage.insert_training_run(
                {
                    "symbol": symbol,
                    "rows": settings.training.minimum_training_rows,
                    "feature_count": 10,
                    "positives": 5,
                    "negatives": 5,
                    "model_path": str(model_path),
                    "trained_with_xgboost": True,
                    "holdout_accuracy": 0.6,
                }
            )
        engine.storage.set_json_state(engine.EXECUTION_SYMBOLS_STATE_KEY, ["BTC/USDT"])
        engine.performance = SimpleNamespace(
            build_symbol_accuracy_summary=lambda limit=1000: {
                "BTC/USDT": {"count": 20, "correct": 10, "accuracy_pct": 50.0},
                "ETH/USDT": {"count": 12, "correct": 6, "accuracy_pct": 50.0},
            },
            build_symbol_edge_summary=lambda limit=1000: {
                "BTC/USDT": {
                    "count": 20,
                    "sample_count": 20,
                    "accuracy_pct": 50.0,
                    "expectancy_pct": 0.08,
                    "profit_factor": 1.05,
                    "max_drawdown_pct": 2.0,
                    "objective_score": 0.5,
                },
                "ETH/USDT": {
                    "count": 12,
                    "sample_count": 12,
                    "accuracy_pct": 50.0,
                    "expectancy_pct": 0.08,
                    "profit_factor": 1.05,
                    "max_drawdown_pct": 2.0,
                    "objective_score": 0.5,
                },
            },
        )
        symbols = engine.get_shadow_observation_symbols()
        self.assertEqual(symbols, ["ETH/USDT"])

    def test_engine_shadow_observation_symbols_prioritize_positive_shadow_feedback(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        settings.exchange.candidate_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
            model_path = model_dir / f"xgboost_v2_{symbol.replace('/', '_')}.json"
            model_path.write_text("{}", encoding="utf-8")
            engine.storage.insert_training_run(
                {
                    "symbol": symbol,
                    "rows": settings.training.minimum_training_rows,
                    "feature_count": 10,
                    "positives": 5,
                    "negatives": 5,
                    "model_path": str(model_path),
                    "trained_with_xgboost": True,
                    "holdout_accuracy": 0.6,
                }
            )
        engine.storage.set_json_state(engine.EXECUTION_SYMBOLS_STATE_KEY, ["BTC/USDT"])
        engine.performance = SimpleNamespace(
            build_symbol_accuracy_summary=lambda limit=1000: {
                "BTC/USDT": {"count": 20, "correct": 10, "accuracy_pct": 50.0},
                "ETH/USDT": {"count": 12, "correct": 6, "accuracy_pct": 50.0},
                "SOL/USDT": {"count": 12, "correct": 6, "accuracy_pct": 50.0},
            },
            build_symbol_edge_summary=lambda limit=1000: {
                "BTC/USDT": {
                    "count": 20,
                    "sample_count": 20,
                    "accuracy_pct": 50.0,
                    "expectancy_pct": 0.08,
                    "profit_factor": 1.05,
                    "max_drawdown_pct": 2.0,
                    "objective_score": 0.5,
                },
                "ETH/USDT": {
                    "count": 12,
                    "sample_count": 12,
                    "accuracy_pct": 50.0,
                    "expectancy_pct": 0.08,
                    "profit_factor": 1.05,
                    "max_drawdown_pct": 2.0,
                    "objective_score": 0.5,
                },
                "SOL/USDT": {
                    "count": 12,
                    "sample_count": 12,
                    "accuracy_pct": 50.0,
                    "expectancy_pct": 0.12,
                    "profit_factor": 1.15,
                    "max_drawdown_pct": 1.8,
                    "objective_score": 0.8,
                },
            },
        )
        for idx in range(3):
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "SOL/USDT",
                    "timestamp": (datetime.now(timezone.utc) - timedelta(hours=idx * 4 + 24)).isoformat(),
                    "evaluation_type": "shadow_observation",
                    "actual_up": True,
                    "predicted_up": True,
                    "is_correct": True,
                    "entry_close": 100.0,
                    "future_close": 105.0,
                    "metadata": {"regime": "BULL_TREND", "setup_profile": {}},
                }
            )
        symbols = engine.get_shadow_observation_symbols()
        self.assertEqual(symbols[:2], ["SOL/USDT", "ETH/USDT"])

    def test_engine_evaluate_matured_predictions_inserts_evaluations(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        prediction_time = datetime.now(timezone.utc) - timedelta(hours=8)
        current_candle_ts = int((prediction_time - timedelta(minutes=1)).timestamp() * 1000)
        future_candle_ts = int((prediction_time + timedelta(hours=4, minutes=1)).timestamp() * 1000)
        engine.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "4h",
            [
                {
                    "timestamp": current_candle_ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1.0,
                },
                {
                    "timestamp": future_candle_ts,
                    "open": 101.0,
                    "high": 111.0,
                    "low": 100.0,
                    "close": 110.0,
                    "volume": 1.5,
                },
            ],
        )
        engine.storage.insert_prediction_run(
            {
                "symbol": "BTC/USDT",
                "timestamp": prediction_time.isoformat(),
                "model_version": "test",
                "up_probability": 0.82,
                "feature_count": 56,
                "research": {"symbol": "BTC/USDT", "market_regime": "UPTREND"},
                "decision": {
                    "pipeline_mode": "execution",
                    "regime": "UPTREND",
                    "xgboost_threshold": 0.7,
                    "final_score": 0.9,
                },
            }
        )
        result = engine._evaluate_matured_predictions(datetime.now(timezone.utc))
        self.assertEqual(result["evaluated_count"], 1)
        with engine.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM prediction_evaluations"
            ).fetchone()["c"]
            evaluation_row = conn.execute(
                "SELECT metadata_json FROM prediction_evaluations LIMIT 1"
            ).fetchone()
        self.assertEqual(count, 1)
        metadata = json.loads(evaluation_row["metadata_json"])
        self.assertIn("trade_net_return_pct", metadata)
        self.assertIn("favorable_excursion_pct", metadata)
        self.assertIn("adverse_excursion_pct", metadata)

    def test_engine_evaluate_matured_predictions_processes_oldest_pending_backlog(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        older_time = datetime.now(timezone.utc) - timedelta(hours=12)
        newer_time = datetime.now(timezone.utc) - timedelta(hours=8)
        candles = [
            {
                "timestamp": int((older_time - timedelta(minutes=1)).timestamp() * 1000),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            },
            {
                "timestamp": int((older_time + timedelta(hours=4, minutes=1)).timestamp() * 1000),
                "open": 101.0,
                "high": 106.0,
                "low": 100.0,
                "close": 105.0,
                "volume": 1.0,
            },
            {
                "timestamp": int((newer_time - timedelta(minutes=1)).timestamp() * 1000),
                "open": 200.0,
                "high": 202.0,
                "low": 198.0,
                "close": 200.0,
                "volume": 1.0,
            },
            {
                "timestamp": int((newer_time + timedelta(hours=4, minutes=1)).timestamp() * 1000),
                "open": 202.0,
                "high": 212.0,
                "low": 201.0,
                "close": 210.0,
                "volume": 1.0,
            },
        ]
        engine.storage.insert_ohlcv("BTC/USDT:USDT", "4h", candles)
        engine.storage.insert_ohlcv("ETH/USDT:USDT", "4h", candles[2:])

        engine.storage.insert_prediction_run(
            {
                "symbol": "BTC/USDT",
                "timestamp": older_time.isoformat(),
                "model_version": "older",
                "up_probability": 0.82,
                "feature_count": 56,
                "research": {"symbol": "BTC/USDT", "market_regime": "UPTREND"},
                "decision": {
                    "pipeline_mode": "execution",
                    "regime": "UPTREND",
                    "xgboost_threshold": 0.7,
                    "final_score": 0.9,
                },
            }
        )
        engine.storage.insert_prediction_run(
            {
                "symbol": "ETH/USDT",
                "timestamp": newer_time.isoformat(),
                "model_version": "newer",
                "up_probability": 0.82,
                "feature_count": 56,
                "research": {"symbol": "ETH/USDT", "market_regime": "UPTREND"},
                "decision": {
                    "pipeline_mode": "execution",
                    "regime": "UPTREND",
                    "xgboost_threshold": 0.7,
                    "final_score": 0.9,
                },
            }
        )
        engine.storage.insert_prediction_evaluation(
            {
                "symbol": "ETH/USDT",
                "timestamp": newer_time.isoformat(),
                "evaluation_type": "execution",
                "actual_up": True,
                "predicted_up": True,
                "is_correct": True,
                "entry_close": 200.0,
                "future_close": 210.0,
                "metadata": {},
            }
        )

        result = engine._evaluate_matured_predictions(
            datetime.now(timezone.utc),
            limit=1,
        )

        self.assertEqual(result["evaluated_count"], 1)
        with engine.storage._conn() as conn:
            btc_eval = conn.execute(
                "SELECT 1 FROM prediction_evaluations "
                "WHERE symbol = ? AND timestamp = ?",
                ("BTC/USDT", older_time.isoformat()),
            ).fetchone()
        self.assertIsNotNone(btc_eval)

    def test_engine_evaluate_matured_predictions_keeps_distinct_pipeline_modes(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        prediction_time = datetime.now(timezone.utc) - timedelta(hours=8)
        current_candle_ts = int((prediction_time - timedelta(minutes=1)).timestamp() * 1000)
        future_candle_ts = int((prediction_time + timedelta(hours=4, minutes=1)).timestamp() * 1000)
        engine.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "4h",
            [
                {
                    "timestamp": current_candle_ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1.0,
                },
                {
                    "timestamp": future_candle_ts,
                    "open": 101.0,
                    "high": 111.0,
                    "low": 100.0,
                    "close": 110.0,
                    "volume": 1.5,
                },
            ],
        )
        for pipeline_mode, probability in (
            ("execution", 0.82),
            ("shadow_observation", 0.68),
        ):
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": prediction_time.isoformat(),
                    "model_version": pipeline_mode,
                    "up_probability": probability,
                    "feature_count": 56,
                    "research": {"symbol": "BTC/USDT", "market_regime": "UPTREND"},
                    "decision": {
                        "pipeline_mode": pipeline_mode,
                        "regime": "UPTREND",
                        "xgboost_threshold": 0.7,
                        "final_score": 0.9,
                    },
                }
            )

        result = engine._evaluate_matured_predictions(datetime.now(timezone.utc))

        self.assertEqual(result["evaluated_count"], 2)
        with engine.storage._conn() as conn:
            rows = conn.execute(
                "SELECT evaluation_type FROM prediction_evaluations "
                "WHERE symbol = ? AND timestamp = ? "
                "ORDER BY evaluation_type ASC",
                ("BTC/USDT", prediction_time.isoformat()),
            ).fetchall()
        self.assertEqual(
            [row["evaluation_type"] for row in rows],
            ["execution", "shadow_observation"],
        )

    def test_engine_records_blocked_shadow_trade(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        engine._record_shadow_trade_if_blocked(
            symbol="BTC/USDT",
            features=SimpleNamespace(timestamp=datetime.now(timezone.utc), values={}),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.8,
                feature_count=10,
                model_version="test",
            ),
            decision=SimpleNamespace(final_score=0.9),
            validation=SimpleNamespace(ok=True, reason="ok"),
            review=SimpleNamespace(
                reasons=["setup_auto_pause"],
                raw_action="OPEN_LONG",
                setup_profile={"regime": "EXTREME_FEAR", "liquidity_bucket": "weak"},
                review_score=-0.2,
            ),
            risk_result=SimpleNamespace(allowed=True, reason=""),
            entry_price=100.0,
        )
        with engine.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM shadow_trade_runs"
            ).fetchone()["c"]
        self.assertEqual(count, 1)

    def test_engine_records_near_miss_shadow_trade(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        engine._record_shadow_trade_if_blocked(
            symbol="BTC/USDT",
            features=SimpleNamespace(timestamp=datetime.now(timezone.utc), values={}),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=engine.decision_engine.xgboost_threshold - 0.05,
                feature_count=10,
                model_version="test",
            ),
            decision=SimpleNamespace(
                final_score=engine.decision_engine.final_score_threshold - 0.3,
                portfolio_rating="HOLD",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            review=SimpleNamespace(
                reasons=[],
                raw_action="OPEN_LONG",
                reviewed_action="OPEN_LONG",
                setup_profile={},
                review_score=0.2,
            ),
            risk_result=SimpleNamespace(allowed=True, reason=""),
            entry_price=100.0,
        )
        with engine.storage._conn() as conn:
            row = conn.execute(
                "SELECT block_reason FROM shadow_trade_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(row["block_reason"], "near_miss:xgboost_threshold")

    def test_engine_records_risk_guard_near_miss_shadow_trade(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        engine._record_shadow_trade_if_blocked(
            symbol="BTC/USDT",
            features=SimpleNamespace(timestamp=datetime.now(timezone.utc), values={}),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=engine.decision_engine.xgboost_threshold - 0.1,
                feature_count=10,
                model_version="test",
            ),
            decision=SimpleNamespace(
                final_score=engine.decision_engine.final_score_threshold - 0.2,
                portfolio_rating="HOLD",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            review=SimpleNamespace(
                reasons=[],
                raw_action="HOLD",
                reviewed_action="CLOSE",
                setup_profile={},
                review_score=-0.9,
            ),
            risk_result=SimpleNamespace(allowed=False, reason="insufficient liquidity"),
            entry_price=100.0,
        )
        with engine.storage._conn() as conn:
            row = conn.execute(
                "SELECT block_reason FROM shadow_trade_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(
            row["block_reason"],
            "near_miss:risk_guard:insufficient liquidity",
        )

    def test_analysis_runtime_applies_position_value_adjuster_before_open(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.paper_canary_enabled = False
        opened = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.7
            final_score_threshold = 0.8

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.LONG, final_score=0.91),
                    SimpleNamespace(
                        should_execute=True,
                        reason="go",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=100.0,
                        final_score=0.91,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={"atr_4h": 1.0, "close_4h": 100.0, "volume_ratio_1h": 1.0},
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.2,
                    confidence=0.7,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.82,
                    feature_count=10,
                    model_version="test",
                    model_id="mdl_test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=[],
                    raw_action="OPEN_LONG",
                    reviewed_action="OPEN_LONG",
                    review_score=0.3,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            position_value_adjuster=lambda **kwargs: {
                "position_value": 60.0,
                "scale": 0.60,
                "source": "test_evidence",
                "reason": "test_evidence:limited_sample",
            },
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertAlmostEqual(opened[0]["position_value"], 60.0, places=6)
        self.assertIn("evidence_scale=0.60", opened[0]["rationale"])
        self.assertAlmostEqual(
            opened[0]["metadata"]["model_evidence_scale"],
            0.60,
            places=6,
        )
        self.assertEqual(
            opened[0]["metadata"]["model_evidence_source"],
            "test_evidence",
        )

    def test_analysis_runtime_does_not_double_scale_ab_live_position(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.paper_canary_enabled = False
        opened = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.7
            final_score_threshold = 0.8

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.75),
                    SimpleNamespace(
                        should_execute=False,
                        reason="champion_hold",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.75,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={"atr_4h": 1.0, "close_4h": 100.0, "volume_ratio_1h": 1.0},
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.2,
                    confidence=0.7,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.74,
                    feature_count=10,
                    model_version="champion",
                    model_id="champion",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=[],
                    raw_action="OPEN_LONG",
                    reviewed_action="OPEN_LONG",
                    review_score=0.3,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: {
                "execute_live": True,
                "selected_variant": "challenger_live",
                "position_value": 60.0,
                "reason": "challenger_edge",
                "final_score": 0.88,
                "analysis_pipeline_mode": "challenger_live",
                "challenger_prediction": PredictionResult(
                    symbol="BTC/USDT",
                    up_probability=0.88,
                    feature_count=10,
                    model_version="challenger",
                    model_id="challenger",
                ),
                "challenger_decision": SimpleNamespace(
                    should_execute=True,
                    reason="challenger_edge",
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                ),
                "challenger_final_score": 0.88,
                "evidence_scale": 0.60,
                "evidence_source": "candidate_live_prediction",
                "evidence_reason": "candidate_live_prediction:limited_sample",
            },
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            position_value_adjuster=lambda **kwargs: {
                "position_value": 30.0,
                "scale": 0.30,
                "source": "should_not_apply_to_ab_live",
                "reason": "should_not_apply_to_ab_live",
            },
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertAlmostEqual(opened[0]["position_value"], 60.0, places=6)
        self.assertIn("evidence_scale=0.60", opened[0]["rationale"])
        self.assertEqual(
            opened[0]["metadata"]["model_evidence_source"],
            "candidate_live_prediction",
        )

    def test_analysis_runtime_opens_paper_canary_when_enabled(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.paper_canary_enabled = True
        settings.strategy.paper_canary_xgboost_gap_pct = 0.2
        settings.strategy.paper_canary_final_score_gap_pct = 0.2
        opened = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.7
            final_score_threshold = 0.8

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.69),
                    SimpleNamespace(
                        should_execute=False,
                        reason="near miss",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.69,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={"atr_4h": 1.0, "close_4h": 100.0, "volume_ratio_1h": 1.0},
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.2,
                    confidence=0.7,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.61,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=[],
                    raw_action="OPEN_LONG",
                    reviewed_action="OPEN_LONG",
                    review_score=0.2,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["direction"], SignalDirection.LONG)
        self.assertAlmostEqual(
            opened[0]["position_value"],
            100.0 * float(settings.strategy.paper_canary_position_scale),
            places=6,
        )

    def test_analysis_runtime_opens_soft_paper_canary_for_research_hold(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.paper_canary_enabled = True
        settings.strategy.paper_canary_soft_enabled = True
        settings.strategy.paper_canary_soft_review_min_score = 0.0
        settings.strategy.paper_canary_soft_position_scale = 0.35
        opened = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.7
            final_score_threshold = 0.8

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.76),
                    SimpleNamespace(
                        should_execute=False,
                        reason="research_hold",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.76,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=120.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={"atr_4h": 1.0, "close_4h": 100.0, "volume_ratio_1h": 1.0},
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.15,
                    confidence=0.7,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.74,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=["xgb_pass", "liquidity_supportive"],
                    raw_action="HOLD",
                    reviewed_action="HOLD",
                    review_score=0.02,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertAlmostEqual(
            opened[0]["position_value"],
            120.0
            * float(settings.strategy.paper_canary_position_scale)
            * float(settings.strategy.paper_canary_soft_position_scale),
            places=6,
        )
        self.assertEqual(opened[0]["metadata"]["paper_canary_mode"], "soft_review")
        self.assertEqual(opened[0]["direction"], SignalDirection.LONG)

    def test_analysis_runtime_opens_execution_soft_entry_for_research_hold(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.paper_canary_enabled = False
        settings.strategy.fast_alpha_enabled = False
        settings.strategy.execution_soft_entry_enabled = True
        settings.strategy.execution_soft_entry_review_min_score = 0.08
        settings.strategy.execution_soft_entry_position_scale = 0.50
        settings.strategy.execution_soft_entry_max_final_score_gap_pct = 0.03
        opened = []
        persisted = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.60),
                    SimpleNamespace(
                        should_execute=False,
                        reason="research_hold_mainline_softened",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.60,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=120.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={"atr_4h": 1.0, "close_4h": 100.0, "volume_ratio_1h": 1.0},
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.10,
                    confidence=0.7,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.75,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=["xgb_pass", "liquidity_supportive", "trend_supportive"],
                    raw_action="HOLD",
                    reviewed_action="HOLD",
                    review_score=0.12,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: persisted.append(kwargs),
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertAlmostEqual(
            opened[0]["position_value"],
            120.0 * float(settings.strategy.execution_soft_entry_position_scale),
            places=6,
        )
        self.assertEqual(opened[0]["metadata"]["pipeline_mode"], "execution_soft_entry")
        self.assertEqual(opened[0]["direction"], SignalDirection.LONG)
        self.assertEqual(len(persisted), 2)
        self.assertEqual(persisted[-1]["pipeline_mode"], "execution_soft_entry")

    def test_analysis_runtime_keeps_hold_signal_closed_when_execution_soft_entry_disabled(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.paper_canary_enabled = False
        settings.strategy.fast_alpha_enabled = False
        settings.strategy.execution_soft_entry_enabled = False
        opened = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.60),
                    SimpleNamespace(
                        should_execute=False,
                        reason="research_hold_mainline_softened",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.60,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=120.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={"atr_4h": 1.0, "close_4h": 100.0, "volume_ratio_1h": 1.0},
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.10,
                    confidence=0.7,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.75,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=["xgb_pass", "liquidity_supportive", "trend_supportive"],
                    raw_action="HOLD",
                    reviewed_action="HOLD",
                    review_score=0.12,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 0)
        self.assertEqual(opened, [])

    def test_analysis_runtime_opens_fast_alpha_for_core_near_miss(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.strategy.fast_alpha_position_scale = 0.4
        settings.strategy.fast_alpha_min_probability_pct = 0.66
        settings.strategy.fast_alpha_min_final_score = 0.54
        settings.strategy.fast_alpha_min_review_score = 0.12
        settings.strategy.fast_alpha_max_hold_hours = 6
        opened = []
        persisted = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.57),
                    SimpleNamespace(
                        should_execute=False,
                        reason="near_miss_fast_alpha",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.57,
                    ),
                )

        reviewed_insight = ResearchInsight(
            symbol="BTC/USDT",
            market_regime=MarketRegime.UPTREND,
            sentiment_score=0.2,
            confidence=0.65,
            risk_warning=[],
            key_reason=["ok"],
            suggested_action=SuggestedAction.HOLD,
        )
        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 1.05,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.2,
                    confidence=0.65,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.69,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=["xgb_pass", "liquidity_supportive", "trend_supportive"],
                    raw_action="OPEN_LONG",
                    reviewed_action="HOLD",
                    reviewed_insight=reviewed_insight,
                    review_score=0.18,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: persisted.append(kwargs),
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(build=lambda: None),
            position_value_adjuster=lambda **kwargs: {
                "position_value": kwargs["base_position_value"],
                "scale": 1.0,
                "source": "none",
                "reason": "",
            },
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["metadata"]["pipeline_mode"], "fast_alpha")
        self.assertEqual(opened[0]["metadata"]["horizon_hours"], 6)
        self.assertAlmostEqual(
            opened[0]["position_value"],
            100.0 * float(settings.strategy.fast_alpha_position_scale),
            places=6,
        )
        self.assertEqual(len(persisted), 2)
        self.assertEqual(persisted[-1]["pipeline_mode"], "fast_alpha")

    def test_analysis_runtime_blocks_fast_alpha_for_non_core_symbol(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        opened = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.58),
                    SimpleNamespace(
                        should_execute=False,
                        reason="near_miss_fast_alpha",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.58,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 1.05,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.2,
                    confidence=0.65,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.70,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=["xgb_pass", "liquidity_supportive", "trend_supportive"],
                    raw_action="OPEN_LONG",
                    reviewed_action="HOLD",
                    reviewed_insight=ResearchInsight(
                        symbol=symbol,
                        market_regime=MarketRegime.UPTREND,
                        sentiment_score=0.2,
                        confidence=0.65,
                        risk_warning=[],
                        key_reason=["ok"],
                        suggested_action=SuggestedAction.HOLD,
                    ),
                    review_score=0.18,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(build=lambda: None),
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["SOL/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 0)
        self.assertEqual(opened, [])

    def test_analysis_runtime_eth_fast_alpha_uses_relaxed_probability_floor(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.strategy.fast_alpha_min_probability_pct = 0.58
        settings.strategy.fast_alpha_eth_min_probability_pct = 0.54
        opened = []

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.67),
                    SimpleNamespace(
                        should_execute=False,
                        reason="near_miss_fast_alpha",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.67,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=SimpleNamespace(
                execute_open=lambda **kwargs: opened.append(kwargs)
                or {"price": kwargs["price"], "dry_run": False}
            ),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 0.8,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.EXTREME_FEAR,
                    sentiment_score=0.15,
                    confidence=0.4,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.55,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=["liquidity_supportive", "trend_supportive"],
                    raw_action="OPEN_LONG",
                    reviewed_action="HOLD",
                    reviewed_insight=ResearchInsight(symbol=symbol),
                    review_score=0.18,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(build=lambda: None),
            position_value_adjuster=lambda **kwargs: {
                "position_value": kwargs["base_position_value"],
                "scale": 1.0,
                "source": "none",
                "reason": "",
            },
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["ETH/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["metadata"]["pipeline_mode"], "fast_alpha")

    def test_analysis_runtime_opens_fast_alpha_with_liquidity_risk_override(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.strategy.fast_alpha_min_probability_pct = 0.58
        settings.strategy.fast_alpha_liquidity_override_enabled = True
        settings.strategy.fast_alpha_liquidity_floor_ratio = 0.25
        opened = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.76),
                    SimpleNamespace(
                        should_execute=False,
                        reason="near_miss_fast_alpha",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.76,
                    ),
                )

        def can_open_position(**kwargs):
            floor = float(kwargs.get("liquidity_floor_override", 0.8) or 0.8)
            allowed = floor <= 0.25
            return RiskCheckResult(
                allowed=allowed,
                reason="" if allowed else "insufficient liquidity",
                allowed_position_value=100.0 if allowed else 0.0,
                stop_loss_pct=0.05,
                take_profit_levels=[0.05],
                trailing_stop_drawdown_pct=0.3,
            )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(can_open_position=can_open_position),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 0.26,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.EXTREME_FEAR,
                    sentiment_score=0.15,
                    confidence=0.5,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.63,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=["liquidity_supportive", "trend_supportive"],
                    raw_action="OPEN_LONG",
                    reviewed_action="HOLD",
                    reviewed_insight=ResearchInsight(
                        symbol=symbol,
                        market_regime=MarketRegime.EXTREME_FEAR,
                        sentiment_score=0.15,
                        confidence=0.5,
                        risk_warning=[],
                        key_reason=["ok"],
                        suggested_action=SuggestedAction.CLOSE,
                    ),
                    review_score=0.18,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(build=lambda: None),
            position_value_adjuster=lambda **kwargs: {
                "position_value": kwargs["base_position_value"],
                "scale": 1.0,
                "source": "none",
                "reason": "",
            },
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["ETH/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["metadata"]["pipeline_mode"], "fast_alpha")

    def test_analysis_runtime_opens_fast_alpha_for_core_extreme_fear_liquidity_repair(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.strategy.fast_alpha_min_probability_pct = 0.58
        opened = []

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.521),
                    SimpleNamespace(
                        should_execute=False,
                        reason="near_miss_fast_alpha",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.521,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=SimpleNamespace(
                execute_open=lambda **kwargs: opened.append(kwargs)
                or {"price": kwargs["price"], "dry_run": False}
            ),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 0.21,
                        "adaptive_min_liquidity_ratio": 0.38,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.EXTREME_FEAR,
                    sentiment_score=0.1,
                    confidence=0.55,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.723,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=[
                        "trend_supportive",
                        "regime_extreme_fear_core_repair_discounted",
                        "fear_greed_extreme_fear_core_repair_discounted",
                        "trend_supportive",
                        "core_extreme_fear_liquidity_repair",
                        "core_extreme_fear_liquidity_repair_open",
                    ],
                    raw_action="HOLD",
                    reviewed_action="OPEN_LONG",
                    reviewed_insight=ResearchInsight(symbol=symbol),
                    review_score=0.18,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(build=lambda: None),
            position_value_adjuster=lambda **kwargs: {
                "position_value": kwargs["base_position_value"],
                "scale": 1.0,
                "source": "none",
                "reason": "",
            },
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["metadata"]["pipeline_mode"], "fast_alpha")

    def test_analysis_runtime_blocks_fast_alpha_on_reviewed_close_even_with_regime_reversal_confirmation(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.strategy.fast_alpha_min_probability_pct = 0.58
        opened = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.78),
                    SimpleNamespace(
                        should_execute=False,
                        reason="near_miss_fast_alpha",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.78,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    reason="",
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 0.5,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.DOWNTREND,
                    sentiment_score=0.15,
                    confidence=0.5,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.63,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(
                    ok=True,
                    reason="ok",
                    details={
                        "reasons": [
                            "onchain_regime_conflict",
                            "price_regime_conflict",
                            "regime_reversal_confirmation",
                        ]
                    },
                ),
                SimpleNamespace(
                    reasons=["trend_supportive"],
                    raw_action="HOLD",
                    reviewed_action="CLOSE",
                    reviewed_insight=ResearchInsight(
                        symbol=symbol,
                        market_regime=MarketRegime.DOWNTREND,
                        sentiment_score=0.15,
                        confidence=0.5,
                        risk_warning=[],
                        key_reason=["ok"],
                        suggested_action=SuggestedAction.CLOSE,
                    ),
                    review_score=-0.50,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(build=lambda: None),
            position_value_adjuster=lambda **kwargs: {
                "position_value": kwargs["base_position_value"],
                "scale": 1.0,
                "source": "none",
                "reason": "",
            },
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 0)
        self.assertEqual(opened, [])
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM execution_events "
                "WHERE event_type='fast_alpha_blocked' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIsNotNone(row)
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["reason"], "reviewed_action_close")

    def test_execution_pool_fast_alpha_active_symbols_do_not_bypass_edge_filter(self):
        from core.execution_pool_runtime_service import ExecutionPoolRuntimeService

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "paper"
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT"]
        settings.exchange.max_active_symbols = 2

        service = ExecutionPoolRuntimeService(
            self.storage,
            settings,
            performance_getter=lambda: SimpleNamespace(),
            shadow_feedback_getter=lambda: SimpleNamespace(),
            consistency_getter=lambda: SimpleNamespace(),
            watchlist_getter=lambda force_refresh=False: {},
            market_getter=lambda: None,
            trainer_getter=lambda: None,
            notifier=SimpleNamespace(),
            current_language=lambda: "en",
            handle_training_summary=None,
            clear_symbol_models=lambda symbol: None,
            clear_broken_model_symbol=lambda symbol: None,
            runtime_model_path_for_symbol=lambda symbol: Path(self.temp_dir) / "model.json",
            broken_model_symbols_state_key="broken_model_symbols",
            execution_symbols_state_key="execution_symbols",
            execution_pool_last_rebuild_at_state_key="execution_pool_last_rebuild_at",
            parse_iso_datetime=lambda value: datetime.fromisoformat(value),
            rebuild_execution_symbols_callback=lambda **kwargs: None,
        )

        selected = service._fast_alpha_core_active_symbols(
            ready=["ETH/USDT", "BTC/USDT"],
            filtered=["ETH/USDT"],
        )

        self.assertEqual(selected, ["ETH/USDT"])
        self.assertEqual(
            self.storage.get_json_state("fast_alpha_active_symbols", []),
            [],
        )

    def test_execution_pool_fast_alpha_rebuild_does_not_readd_disqualified_symbol(self):
        from core.execution_pool_runtime_service import ExecutionPoolRuntimeService

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "paper"
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT"]

        service = ExecutionPoolRuntimeService(
            self.storage,
            settings,
            performance_getter=lambda: SimpleNamespace(),
            shadow_feedback_getter=lambda: SimpleNamespace(),
            consistency_getter=lambda: SimpleNamespace(),
            watchlist_getter=lambda force_refresh=False: {},
            market_getter=lambda: None,
            trainer_getter=lambda: None,
            notifier=SimpleNamespace(),
            current_language=lambda: "en",
            handle_training_summary=None,
            clear_symbol_models=lambda symbol: None,
            clear_broken_model_symbol=lambda symbol: None,
            runtime_model_path_for_symbol=lambda symbol: Path(self.temp_dir) / "model.json",
            broken_model_symbols_state_key="broken_model_symbols",
            execution_symbols_state_key="execution_symbols",
            execution_pool_last_rebuild_at_state_key="execution_pool_last_rebuild_at",
            parse_iso_datetime=lambda value: datetime.fromisoformat(value),
            rebuild_execution_symbols_callback=lambda **kwargs: None,
        )

        selected = service._fast_alpha_core_execution_symbols(
            ranked_candidates=[
                {
                    "symbol": "ETH/USDT",
                    "status": "qualified",
                    "has_model": True,
                    "consistency_flags": [],
                },
                {
                    "symbol": "BTC/USDT",
                    "status": "disqualified",
                    "has_model": True,
                    "consistency_flags": [],
                },
            ],
            selected_symbols=["ETH/USDT"],
            target_size=2,
        )

        self.assertEqual(selected, ["ETH/USDT"])

    def test_analysis_runtime_records_fast_alpha_blocked_event(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.55),
                    SimpleNamespace(
                        should_execute=False,
                        reason="near_miss_fast_alpha",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.55,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=SimpleNamespace(execute_open=lambda **kwargs: None),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 0.5,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.1,
                    confidence=0.4,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.50,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=["trend_supportive"],
                    raw_action="HOLD",
                    reviewed_action="HOLD",
                    reviewed_insight=ResearchInsight(symbol=symbol),
                    review_score=0.2,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(build=lambda: None),
        )

        service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM execution_events "
                "WHERE event_type='fast_alpha_blocked' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIsNotNone(row)
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["reason"], "probability_below_min")

    def test_analysis_runtime_fast_alpha_relaxes_thresholds_when_short_horizon_edge_is_positive(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.strategy.paper_canary_enabled = False
        settings.strategy.fast_alpha_min_probability_pct = 0.66
        settings.strategy.fast_alpha_min_final_score = 0.54
        settings.strategy.fast_alpha_min_review_score = 0.12
        settings.strategy.short_horizon_adaptive_min_closed_trades = 3
        settings.strategy.short_horizon_adaptive_positive_expectancy_pct = 0.05
        settings.strategy.short_horizon_adaptive_positive_profit_factor = 1.05
        opened = []

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.52),
                    SimpleNamespace(
                        should_execute=False,
                        reason="near_miss_fast_alpha",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.52,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=SimpleNamespace(
                execute_open=lambda **kwargs: opened.append(kwargs)
                or {"price": kwargs["price"], "dry_run": False}
            ),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 1.0,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.2,
                    confidence=0.6,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.63,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok", details={"reasons": []}),
                SimpleNamespace(
                    reasons=["xgb_pass", "liquidity_supportive", "trend_supportive"],
                    raw_action="OPEN_LONG",
                    reviewed_action="HOLD",
                    reviewed_insight=ResearchInsight(symbol=symbol),
                    review_score=0.08,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(
                build=lambda: None,
                build_pipeline_mode_summary=lambda pipeline_modes=None, limit=20: {
                    "_combined": {
                        "closed_trade_count": 5,
                        "expectancy_pct": 0.24,
                        "profit_factor": 1.42,
                        "max_drawdown_pct": 1.2,
                        "win_rate_pct": 60.0,
                        "mode_counts": {"fast_alpha": 3, "paper_canary": 2},
                    }
                },
            ),
            position_value_adjuster=lambda **kwargs: {
                "position_value": kwargs["base_position_value"],
                "scale": 1.0,
                "source": "none",
                "reason": "",
            },
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertEqual(
            opened[0]["metadata"]["short_horizon_adaptive_reason"],
            "positive_edge_expand",
        )
        self.assertAlmostEqual(
            opened[0]["metadata"]["fast_alpha_min_probability_pct"],
            0.62,
            places=6,
        )
        self.assertAlmostEqual(
            opened[0]["metadata"]["fast_alpha_min_final_score"],
            0.51,
            places=6,
        )
        self.assertAlmostEqual(
            opened[0]["metadata"]["fast_alpha_min_review_score"],
            0.07,
            places=6,
        )

    def test_analysis_runtime_fast_alpha_pauses_when_short_horizon_edge_is_negative(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.strategy.paper_canary_enabled = False
        settings.strategy.short_horizon_adaptive_min_closed_trades = 3
        opened = []

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.68),
                    SimpleNamespace(
                        should_execute=False,
                        reason="near_miss_fast_alpha",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.68,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=SimpleNamespace(execute_open=lambda **kwargs: opened.append(kwargs)),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 1.0,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.2,
                    confidence=0.6,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.72,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok", details={"reasons": []}),
                SimpleNamespace(
                    reasons=["xgb_pass", "liquidity_supportive", "trend_supportive"],
                    raw_action="OPEN_LONG",
                    reviewed_action="HOLD",
                    reviewed_insight=ResearchInsight(symbol=symbol),
                    review_score=0.20,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(
                build=lambda: None,
                build_pipeline_mode_summary=lambda pipeline_modes=None, limit=20: {
                    "_combined": {
                        "closed_trade_count": 6,
                        "expectancy_pct": -0.35,
                        "profit_factor": 0.62,
                        "max_drawdown_pct": 4.8,
                        "win_rate_pct": 33.3,
                        "mode_counts": {"fast_alpha": 4, "paper_canary": 2},
                    }
                },
            ),
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 0)
        self.assertEqual(opened, [])
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM execution_events "
                "WHERE event_type='fast_alpha_blocked' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIsNotNone(row)
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["reason"], "short_horizon_negative_expectancy_pause")
        self.assertAlmostEqual(payload["adaptive_expectancy_pct"], -0.35, places=6)

    def test_analysis_runtime_fast_alpha_softens_mild_setup_negative_expectancy_while_warming_up(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.strategy.paper_canary_enabled = False
        settings.strategy.short_horizon_adaptive_min_closed_trades = 3
        opened = []

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.74),
                    SimpleNamespace(
                        should_execute=False,
                        reason="setup_negative_expectancy_near_miss",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.74,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=SimpleNamespace(
                execute_open=lambda **kwargs: opened.append(kwargs)
                or {"price": kwargs["price"], "dry_run": False}
            ),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 1.0,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.EXTREME_FEAR,
                    sentiment_score=0.15,
                    confidence=0.6,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.68,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok", details={"reasons": []}),
                SimpleNamespace(
                    reasons=[
                        "xgb_pass",
                        "liquidity_supportive",
                        "trend_supportive",
                        "setup_negative_expectancy",
                        "setup_avg_outcome_-0.14",
                    ],
                    raw_action="OPEN_LONG",
                    reviewed_action="CLOSE",
                    reviewed_insight=ResearchInsight(symbol=symbol),
                    review_score=0.06,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(
                build=lambda: None,
                build_pipeline_mode_summary=lambda pipeline_modes=None, limit=20: {
                    "_combined": {
                        "closed_trade_count": 2,
                        "expectancy_pct": 0.02,
                        "profit_factor": 1.01,
                        "max_drawdown_pct": 0.6,
                        "win_rate_pct": 50.0,
                        "mode_counts": {"paper_canary": 2},
                    }
                },
            ),
            position_value_adjuster=lambda **kwargs: {
                "position_value": kwargs["base_position_value"],
                "scale": 1.0,
                "source": "none",
                "reason": "",
            },
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertEqual(
            opened[0]["metadata"]["fast_alpha_effective_reviewed_action"],
            "HOLD",
        )
        self.assertIn(
            "setup_negative_expectancy",
            opened[0]["metadata"]["fast_alpha_review_policy_relaxed_reasons"],
        )
        self.assertIn(
            "warming_up_soften",
            opened[0]["metadata"]["fast_alpha_review_policy_reason"],
        )
        self.assertGreater(
            float(opened[0]["metadata"]["fast_alpha_effective_review_score"]),
            0.12,
        )

    def test_analysis_runtime_fast_alpha_disables_softening_after_negative_softened_trade_edge(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.strategy.paper_canary_enabled = False
        settings.strategy.short_horizon_adaptive_min_closed_trades = 3
        settings.strategy.fast_alpha_short_horizon_trade_feedback_min_closed_trades = 3

        base_time = datetime.now(timezone.utc) - timedelta(hours=6)
        for idx, pnl_pct in enumerate((-1.0, -0.8, -0.6), start=1):
            trade_id = f"softened-loss-{idx}"
            entry_time = (base_time + timedelta(minutes=idx * 20)).isoformat()
            self.storage.insert_trade(
                {
                    "id": trade_id,
                    "symbol": "BTC/USDT",
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "quantity": 1.0,
                    "entry_time": entry_time,
                    "rationale": "softened entry",
                    "confidence": 0.7,
                    "metadata": {
                        "pipeline_mode": "fast_alpha",
                        "fast_alpha_review_policy_reason": "warming_up_soften|setup_avg_outcome=-0.14",
                    },
                }
            )
            self.storage.update_trade_exit(
                trade_id,
                100.0 * (1.0 + pnl_pct / 100.0),
                (base_time + timedelta(minutes=idx * 20 + 10)).isoformat(),
                pnl_pct,
                pnl_pct,
            )

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.74),
                    SimpleNamespace(
                        should_execute=False,
                        reason="softened_trade_feedback_near_miss",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.74,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=SimpleNamespace(execute_open=lambda **kwargs: None),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 1.0,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.EXTREME_FEAR,
                    sentiment_score=0.15,
                    confidence=0.6,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.68,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok", details={"reasons": []}),
                SimpleNamespace(
                    reasons=[
                        "xgb_pass",
                        "liquidity_supportive",
                        "trend_supportive",
                        "setup_negative_expectancy",
                        "setup_avg_outcome_-0.14",
                    ],
                    raw_action="OPEN_LONG",
                    reviewed_action="CLOSE",
                    reviewed_insight=ResearchInsight(symbol=symbol),
                    review_score=0.04,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(
                build=lambda: None,
                build_pipeline_mode_summary=lambda pipeline_modes=None, limit=20: {
                    "_combined": {
                        "closed_trade_count": 2,
                        "expectancy_pct": 0.02,
                        "profit_factor": 1.01,
                        "max_drawdown_pct": 0.6,
                        "win_rate_pct": 50.0,
                        "mode_counts": {"paper_canary": 2},
                    }
                },
            ),
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 0)
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM execution_events "
                "WHERE event_type='fast_alpha_blocked' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIsNotNone(row)
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["reason"], "review_policy_guard_blocked")
        self.assertEqual(payload["review_policy_feedback_status"], "disabled_negative_edge")
        self.assertLess(payload["review_policy_feedback_expectancy_pct"], 0.0)

    def test_analysis_runtime_fast_alpha_keeps_setup_auto_pause_hard_blocked(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.strategy.paper_canary_enabled = False
        settings.strategy.short_horizon_adaptive_min_closed_trades = 3

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.74),
                    SimpleNamespace(
                        should_execute=False,
                        reason="setup_auto_pause_near_miss",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.74,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=SimpleNamespace(execute_open=lambda **kwargs: None),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 1.0,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.EXTREME_FEAR,
                    sentiment_score=0.15,
                    confidence=0.6,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.68,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok", details={"reasons": []}),
                SimpleNamespace(
                    reasons=[
                        "xgb_pass",
                        "liquidity_supportive",
                        "trend_supportive",
                        "setup_negative_expectancy",
                        "setup_avg_outcome_-0.14",
                        "setup_auto_pause",
                    ],
                    raw_action="OPEN_LONG",
                    reviewed_action="HOLD",
                    reviewed_insight=ResearchInsight(symbol=symbol),
                    review_score=0.14,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(
                build=lambda: None,
                build_pipeline_mode_summary=lambda pipeline_modes=None, limit=20: {
                    "_combined": {
                        "closed_trade_count": 2,
                        "expectancy_pct": 0.02,
                        "profit_factor": 1.01,
                        "max_drawdown_pct": 0.6,
                        "win_rate_pct": 50.0,
                        "mode_counts": {"paper_canary": 2},
                    }
                },
            ),
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 0)
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM execution_events "
                "WHERE event_type='fast_alpha_blocked' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIsNotNone(row)
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["reason"], "hard_review_blocker")

    def test_analysis_runtime_allows_fast_alpha_with_mild_experience_negative_setup(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        opened = []

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.78),
                    SimpleNamespace(
                        should_execute=False,
                        reason="near_miss_fast_alpha",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.78,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=SimpleNamespace(
                execute_open=lambda **kwargs: opened.append(kwargs) or {
                    "price": kwargs["price"],
                    "dry_run": False,
                }
            ),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 0.8,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.EXTREME_FEAR,
                    sentiment_score=0.1,
                    confidence=0.5,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.63,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok", details={"reasons": []}),
                SimpleNamespace(
                    reasons=[
                        "liquidity_supportive",
                        "trend_supportive",
                        "experience_negative_setup",
                        "experience_avg_outcome_-0.09",
                    ],
                    raw_action="OPEN_LONG",
                    reviewed_action="HOLD",
                    reviewed_insight=ResearchInsight(symbol=symbol),
                    review_score=0.18,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(build=lambda: None),
            position_value_adjuster=lambda **kwargs: {
                "position_value": kwargs["base_position_value"],
                "scale": 1.0,
                "source": "none",
                "reason": "",
            },
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)

    def test_analysis_runtime_blocks_fast_alpha_with_strong_experience_negative_setup(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]

        class FakeDecisionEngine:
            xgboost_threshold = 0.74
            final_score_threshold = 0.62

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.78),
                    SimpleNamespace(
                        should_execute=False,
                        reason="near_miss_fast_alpha",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.78,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=100.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=SimpleNamespace(execute_open=lambda **kwargs: None),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={
                        "atr_4h": 1.0,
                        "close_4h": 100.0,
                        "volume_ratio_1h": 0.8,
                        "adaptive_min_liquidity_ratio": 0.35,
                    },
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.EXTREME_FEAR,
                    sentiment_score=0.1,
                    confidence=0.5,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.63,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok", details={"reasons": []}),
                SimpleNamespace(
                    reasons=[
                        "liquidity_supportive",
                        "trend_supportive",
                        "experience_negative_setup",
                        "experience_avg_outcome_-0.25",
                    ],
                    raw_action="OPEN_LONG",
                    reviewed_action="HOLD",
                    reviewed_insight=ResearchInsight(symbol=symbol),
                    review_score=0.18,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
            performance_getter=lambda: SimpleNamespace(build=lambda: None),
            position_value_adjuster=lambda **kwargs: {
                "position_value": kwargs["base_position_value"],
                "scale": 1.0,
                "source": "none",
                "reason": "",
            },
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 0)
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM execution_events "
                "WHERE event_type='fast_alpha_blocked' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertIn("hard_review_blocker", row["payload_json"])

    def test_analysis_runtime_blocks_negative_review_score_soft_canary(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.paper_canary_enabled = True
        settings.strategy.paper_canary_soft_enabled = True
        settings.strategy.paper_canary_soft_review_min_score = -0.05
        opened = []
        shadow_records = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.7
            final_score_threshold = 0.8

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.76),
                    SimpleNamespace(
                        should_execute=False,
                        reason="research_hold_negative_score",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.76,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=120.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={"atr_4h": 1.0, "close_4h": 100.0, "volume_ratio_1h": 1.0},
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.UPTREND,
                    sentiment_score=0.15,
                    confidence=0.7,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.74,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=["xgb_pass", "liquidity_supportive"],
                    raw_action="HOLD",
                    reviewed_action="HOLD",
                    review_score=-0.02,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: shadow_records.append(kwargs),
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 0)
        self.assertEqual(opened, [])
        self.assertEqual(len(shadow_records), 1)

    def test_analysis_runtime_blocks_weak_soft_paper_canary_and_leaves_shadow_only(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.paper_canary_enabled = True
        settings.strategy.paper_canary_soft_enabled = True
        opened = []
        shadow_records = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.7
            final_score_threshold = 0.8

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.71),
                    SimpleNamespace(
                        should_execute=False,
                        reason="weak_soft_review",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.71,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=120.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={"atr_4h": 1.0, "close_4h": 100.0, "volume_ratio_1h": 0.8},
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.EXTREME_FEAR,
                    sentiment_score=0.05,
                    confidence=0.6,
                    risk_warning=[],
                    key_reason=["ok"],
                    suggested_action=SuggestedAction.HOLD,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.71,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=[
                        "regime_extreme_fear",
                        "liquidity_weak",
                        "trend_against",
                    ],
                    raw_action="HOLD",
                    reviewed_action="CLOSE",
                    review_score=-0.04,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: shadow_records.append(kwargs),
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 0)
        self.assertEqual(opened, [])
        self.assertEqual(len(shadow_records), 1)

    def test_analysis_runtime_opens_offensive_paper_canary_for_open_long_downgrade(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.paper_canary_enabled = True
        settings.strategy.paper_canary_offensive_enabled = True
        settings.strategy.paper_canary_offensive_review_min_score = 0.0
        settings.strategy.paper_canary_offensive_position_scale = 0.75
        opened = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.7
            final_score_threshold = 0.8

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.74),
                    SimpleNamespace(
                        should_execute=False,
                        reason="offensive_hold",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.74,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=120.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={"atr_4h": 1.0, "close_4h": 100.0, "volume_ratio_1h": 1.0},
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.EXTREME_FEAR,
                    sentiment_score=-0.1,
                    confidence=0.55,
                    risk_warning=["macro stress"],
                    key_reason=["extreme_fear_offensive_setup"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.71,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=["extreme_fear_offensive_setup"],
                    raw_action="OPEN_LONG",
                    reviewed_action="HOLD",
                    review_score=0.02,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertAlmostEqual(
            opened[0]["position_value"],
            120.0
            * float(settings.strategy.paper_canary_position_scale)
            * float(settings.strategy.paper_canary_offensive_position_scale),
            places=6,
        )
        self.assertEqual(opened[0]["metadata"]["paper_canary_mode"], "offensive_review")
        self.assertEqual(opened[0]["direction"], SignalDirection.LONG)

    def test_analysis_runtime_opens_offensive_paper_canary_for_quant_override_open(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.paper_canary_enabled = True
        settings.strategy.paper_canary_offensive_enabled = True
        settings.strategy.paper_canary_min_review_score = 0.15
        settings.strategy.paper_canary_offensive_review_min_score = -0.02
        opened = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.7
            final_score_threshold = 0.8

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.73),
                    SimpleNamespace(
                        should_execute=False,
                        reason="quant_override_near_miss",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.73,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=120.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={"atr_4h": 1.0, "close_4h": 100.0, "volume_ratio_1h": 1.0},
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.EXTREME_FEAR,
                    sentiment_score=0.08,
                    confidence=0.62,
                    risk_warning=[],
                    key_reason=["quant_reversal"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.72,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=["extreme_fear_quant_override", "extreme_fear_quant_override_open"],
                    raw_action="HOLD",
                    reviewed_action="OPEN_LONG",
                    review_score=0.05,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["metadata"]["paper_canary_mode"], "offensive_review")

    def test_analysis_runtime_opens_offensive_paper_canary_for_quant_repairing_open(self):
        from core.analysis_runtime_service import AnalysisRuntimeService
        from core.models import AccountState, RiskCheckResult

        settings = get_settings().model_copy(deep=True)
        settings.strategy.paper_canary_enabled = True
        settings.strategy.paper_canary_offensive_enabled = True
        settings.strategy.paper_canary_min_review_score = 0.15
        settings.strategy.paper_canary_offensive_review_min_score = -0.02
        opened = []

        class FakeExecutor:
            def execute_open(self, **kwargs):
                opened.append(kwargs)
                return {"price": kwargs["price"], "dry_run": False}

        class FakeDecisionEngine:
            xgboost_threshold = 0.7
            final_score_threshold = 0.8

            def evaluate_entry(self, symbol, prediction, insight, features, risk_result):
                return (
                    SimpleNamespace(direction=SignalDirection.FLAT, final_score=0.74),
                    SimpleNamespace(
                        should_execute=False,
                        reason="quant_repairing_near_miss",
                        stop_loss_pct=0.05,
                        take_profit_levels=[0.05],
                        position_value=0.0,
                        final_score=0.74,
                    ),
                )

        service = AnalysisRuntimeService(
            self.storage,
            settings,
            decision_engine=FakeDecisionEngine(),
            risk=SimpleNamespace(
                can_open_position=lambda **kwargs: RiskCheckResult(
                    allowed=True,
                    allowed_position_value=120.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                    trailing_stop_drawdown_pct=0.3,
                )
            ),
            executor=FakeExecutor(),
            notifier=SimpleNamespace(
                notify_analysis_result=lambda *args, **kwargs: None,
                notify_trade_open=lambda *args, **kwargs: None,
            ),
            prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: (
                SimpleNamespace(
                    values={"atr_4h": 1.0, "close_4h": 100.0, "volume_ratio_1h": 1.0},
                    timestamp=now,
                ),
                ResearchInsight(
                    symbol=symbol,
                    market_regime=MarketRegime.EXTREME_FEAR,
                    sentiment_score=-0.12,
                    confidence=0.58,
                    risk_warning=[],
                    key_reason=["repairing_setup"],
                    suggested_action=SuggestedAction.OPEN_LONG,
                ),
                PredictionResult(
                    symbol=symbol,
                    up_probability=0.72,
                    feature_count=10,
                    model_version="test",
                ),
                SimpleNamespace(ok=True, reason="ok"),
                SimpleNamespace(
                    reasons=["quant_repairing_setup", "quant_repairing_setup_open"],
                    raw_action="HOLD",
                    reviewed_action="OPEN_LONG",
                    review_score=0.06,
                ),
            ),
            detect_abnormal_move=lambda symbol, now: False,
            evaluate_ab_test=lambda **kwargs: None,
            persist_analysis=lambda *args, **kwargs: None,
            record_shadow_trade_if_blocked=lambda **kwargs: None,
            compose_trade_rationale=lambda reason, review: reason,
            get_positions=lambda: [],
            account_state=lambda now, positions: AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            get_circuit_breaker_reason=lambda: "",
        )

        result = service.run_active_symbols(
            now=datetime.now(timezone.utc),
            active_symbols=["BTC/USDT"],
            positions=[],
            account=AccountState(
                equity=10000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                daily_loss_pct=0.0,
                weekly_loss_pct=0.0,
                drawdown_pct=0.0,
                total_exposure_pct=0.0,
                open_positions=0,
            ),
            model_trading_disabled=False,
            consecutive_wins=0,
            consecutive_losses=0,
        )

        self.assertEqual(result["opened_positions"], 1)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["metadata"]["paper_canary_mode"], "offensive_review")

    def test_engine_adjust_position_value_for_model_evidence_cuts_size_for_observed_model(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        active_model_path.write_text("promoted-model", encoding="utf-8")

        promoted_at = datetime.now(timezone.utc) - timedelta(hours=6)
        engine.storage.set_json_state(
            engine.MODEL_PROMOTION_OBSERVATION_STATE_KEY,
            {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "promoted_at": promoted_at.isoformat(),
                    "active_model_path": str(active_model_path),
                    "active_model_id": "challenger",
                    "training_metadata": {
                        "model_id": "challenger",
                        "active_model_path": str(active_model_path),
                        "active_model_id": "challenger",
                    },
                }
            },
        )
        for idx in range(2):
            ts = (promoted_at + timedelta(hours=idx + 1)).isoformat()
            engine.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "challenger",
                    "up_probability": 0.85,
                    "feature_count": 10,
                    "research": {"symbol": "BTC/USDT"},
                    "decision": {
                        "pipeline_mode": "execution",
                        "final_score": 0.85,
                        "model_id": "challenger",
                    },
                }
            )
            engine.storage.insert_prediction_evaluation(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "evaluation_type": "execution",
                    "actual_up": True,
                    "predicted_up": True,
                    "is_correct": True,
                    "entry_close": 100.0,
                    "future_close": 101.0,
                    "metadata": {},
                }
            )

        adjustment = engine._adjust_position_value_for_model_evidence(
            symbol="BTC/USDT",
            pipeline_mode="execution",
            base_position_value=100.0,
        )

        self.assertLess(adjustment["position_value"], 100.0)
        self.assertLess(adjustment["scale"], 1.0)
        self.assertEqual(adjustment["source"], "post_promotion_observation")
        self.assertIn("thin_sample", adjustment["reason"])

    def test_engine_adjust_position_value_for_model_evidence_uses_active_runtime_samples_for_fast_alpha(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model_path = model_dir / "xgboost_v2_BTC_USDT.json"
        active_model_path.write_text("active-model", encoding="utf-8")

        started_at = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        engine.storage.upsert_model_registry(
            {
                "symbol": "BTC/USDT",
                "model_id": "active-runtime-model",
                "model_version": "xgboost_v2_BTC_USDT.json",
                "model_path": str(active_model_path),
                "role": "active",
                "stage": "active",
                "active": True,
                "metadata": {},
                "created_at": started_at,
            }
        )
        for idx in range(2):
            event_time = (datetime.now(timezone.utc) - timedelta(hours=idx + 1)).isoformat()
            engine.storage.insert_pnl_ledger_entry(
                {
                    "trade_id": f"active-evidence-{idx}",
                    "symbol": "BTC/USDT",
                    "event_type": "close",
                    "event_time": event_time,
                    "quantity": 1.0,
                    "notional_value": 100.0,
                    "reference_price": 100.0,
                    "fill_price": 101.0,
                    "gross_pnl": 1.0,
                    "fee_cost": 0.1,
                    "slippage_cost": 0.0,
                    "net_pnl": 0.9,
                    "net_return_pct": 0.9,
                    "holding_hours": 2.0,
                    "model_id": "active-runtime-model",
                }
            )

        adjustment = engine._adjust_position_value_for_model_evidence(
            symbol="BTC/USDT",
            pipeline_mode="fast_alpha",
            base_position_value=100.0,
        )

        self.assertLess(adjustment["position_value"], 100.0)
        self.assertLess(adjustment["scale"], 1.0)
        self.assertEqual(adjustment["source"], "active_runtime_realized")
        self.assertIn("thin_sample", adjustment["reason"])

    def test_engine_evaluate_ab_test_scales_live_canary_allocation_by_candidate_evidence(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        engine._challenger_predictor_for_symbol = lambda symbol: SimpleNamespace(
            predict=lambda features: PredictionResult(
                symbol=symbol,
                up_probability=0.90,
                feature_count=10,
                model_version="challenger",
                model_id="challenger",
            )
        )
        engine.decision_engine.evaluate_entry = (
            lambda symbol, prediction, insight, features, risk_result: (
                SimpleNamespace(final_score=0.90),
                SimpleNamespace(
                    should_execute=True,
                    reason="challenger_edge",
                    position_value=500.0,
                    stop_loss_pct=0.05,
                    take_profit_levels=[0.05],
                ),
            )
        )
        live_started_at = datetime.now(timezone.utc) - timedelta(hours=6)
        engine.storage.set_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "status": "live",
                    "registered_at": (live_started_at - timedelta(hours=2)).isoformat(),
                    "live_started_at": live_started_at.isoformat(),
                    "challenger_model_id": "challenger",
                    "challenger_model_path": "unused",
                    "active_model_id": "champion",
                    "active_model_path": "unused",
                    "live_allocation_pct": 0.03,
                    "shadow_eval_count": 4,
                    "shadow_executed_count": 4,
                    "shadow_accuracy": 0.75,
                    "shadow_expectancy_pct": 0.05,
                    "shadow_profit_factor": 1.05,
                    "shadow_max_drawdown_pct": 1.80,
                    "shadow_avg_trade_return_pct": 0.05,
                    "shadow_objective_score": 0.10,
                    "shadow_objective_quality": 0.10,
                    "training_metadata": {
                        "model_id": "challenger",
                        "challenger_model_id": "challenger",
                    },
                }
            },
        )

        result = engine._evaluate_ab_test(
            now=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            features=SimpleNamespace(symbol="BTC/USDT", values={}),
            insight=ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.UPTREND,
                sentiment_score=0.2,
                confidence=0.7,
                risk_warning=[],
                key_reason=["ok"],
                suggested_action=SuggestedAction.OPEN_LONG,
            ),
            risk_result=SimpleNamespace(allowed=True, allowed_position_value=500.0),
            champion_prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.70,
                feature_count=10,
                model_version="champion",
                model_id="champion",
            ),
            champion_decision=SimpleNamespace(
                should_execute=False,
                reason="champion_hold",
            ),
            account=SimpleNamespace(equity=10000.0),
            positions=[],
        )

        self.assertTrue(result["execute_live"])
        self.assertLess(result["position_value"], 300.0)
        self.assertLess(result["evidence_scale"], 1.0)
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT allocation_pct, notes FROM ab_test_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertLess(float(row["allocation_pct"]), 0.03)
        self.assertIn("evidence_scale=", row["notes"])

    def test_engine_evaluate_shadow_trades_marks_mature_runs(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        trade_time = datetime.now(timezone.utc) - timedelta(hours=8)
        future_candle_ts = int((trade_time + timedelta(hours=4, minutes=1)).timestamp() * 1000)
        engine.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "4h",
            [
                {
                    "timestamp": future_candle_ts,
                    "open": 101.0,
                    "high": 111.0,
                    "low": 100.0,
                    "close": 110.0,
                    "volume": 1.5,
                },
            ],
        )
        engine.storage.insert_shadow_trade_run(
            {
                "symbol": "BTC/USDT",
                "timestamp": trade_time.isoformat(),
                "block_reason": "setup_auto_pause",
                "direction": "LONG",
                "entry_price": 100.0,
                "horizon_hours": 4,
                "status": "open",
                "setup_profile": {"regime": "EXTREME_FEAR"},
                "metadata": {},
            }
        )
        result = engine._evaluate_shadow_trades(datetime.now(timezone.utc))
        self.assertEqual(result["evaluated_count"], 1)
        with engine.storage._conn() as conn:
            row = conn.execute(
                "SELECT status, exit_price, pnl_pct FROM shadow_trade_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            reflection_count = conn.execute(
                "SELECT COUNT(*) AS c FROM reflections"
            ).fetchone()["c"]
            reflection = conn.execute(
                "SELECT trade_id, rationale, outcome_24h FROM reflections ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(row["status"], "evaluated")
        self.assertAlmostEqual(float(row["exit_price"]), 110.0, places=6)
        self.assertAlmostEqual(float(row["pnl_pct"]), 10.0, places=6)
        self.assertEqual(reflection_count, 1)
        self.assertEqual(reflection["trade_id"], "shadow:1")
        self.assertIn("setup_profile=", reflection["rationale"])
        self.assertAlmostEqual(float(reflection["outcome_24h"]), 10.0, places=6)

    def test_engine_evaluate_shadow_trades_backfills_reflections_for_existing_runs(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        engine.storage.insert_shadow_trade_run(
            {
                "symbol": "BTC/USDT",
                "timestamp": (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat(),
                "block_reason": "near_miss:review_hold",
                "direction": "LONG",
                "entry_price": 100.0,
                "horizon_hours": 4,
                "status": "evaluated",
                "exit_price": 104.0,
                "pnl_pct": 4.0,
                "setup_profile": {
                    "symbol": "BTC/USDT",
                    "regime": "EXTREME_FEAR",
                    "validation": "ok",
                    "liquidity_bucket": "weak",
                    "news_bucket": "thin",
                },
                "metadata": {
                    "prediction_up_probability": 0.82,
                    "review_score": -0.2,
                    "validation_reason": "ok",
                    "reviewed_action": "CLOSE",
                },
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = engine._evaluate_shadow_trades(datetime.now(timezone.utc))

        self.assertEqual(result["evaluated_count"], 0)
        self.assertEqual(result["reflected_count"], 1)
        with engine.storage._conn() as conn:
            reflection = conn.execute(
                "SELECT trade_id, symbol, direction, confidence, rationale, outcome_24h, lesson "
                "FROM reflections ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(reflection["trade_id"], "shadow:1")
        self.assertEqual(reflection["symbol"], "BTC/USDT")
        self.assertEqual(reflection["direction"], "LONG")
        self.assertGreaterEqual(float(reflection["confidence"]), 0.8)
        self.assertIn("setup_profile=", reflection["rationale"])
        self.assertIn("near_miss:review_hold", reflection["rationale"])
        self.assertAlmostEqual(float(reflection["outcome_24h"]), 4.0, places=6)
        self.assertIn("Blocked setup would have returned", reflection["lesson"])
        with engine.storage._conn() as conn:
            weighted = conn.execute(
                "SELECT source, experience_weight FROM reflections WHERE trade_id='shadow:1'"
            ).fetchone()
        self.assertEqual(weighted["source"], "shadow_observation")
        self.assertAlmostEqual(float(weighted["experience_weight"]), 0.35, places=6)

    def test_engine_reflect_closed_paper_canary_trade_triggers_rebuild(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        rebuild_calls = []
        learning_refresh_calls = []
        engine.rebuild_execution_symbols = lambda **kwargs: rebuild_calls.append(kwargs) or {
            "status": "ok"
        }
        engine._refresh_runtime_learning_feedback = (
            lambda now, reason: learning_refresh_calls.append((now, reason))
        )

        engine._reflect_closed_trade_result(
            "BTC/USDT",
            {
                "trade_id": "paper-canary-1",
                "confidence": 0.45,
                "rationale": "paper canary close",
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "entry_price": 100.0,
                "exit_price": 104.0,
                "pnl": 4.0,
                "pnl_pct": 4.0,
                "metadata": {
                    "pipeline_mode": "paper_canary",
                    "paper_canary_mode": "soft_review",
                },
            },
        )

        with engine.storage._conn() as conn:
            reflection = conn.execute(
                "SELECT source, experience_weight FROM reflections WHERE trade_id='paper-canary-1'"
            ).fetchone()
        self.assertIsNotNone(reflection)
        self.assertEqual(reflection["source"], "paper_canary")
        self.assertAlmostEqual(float(reflection["experience_weight"]), 0.60, places=6)
        self.assertEqual(len(rebuild_calls), 1)
        self.assertEqual(len(learning_refresh_calls), 1)
        self.assertEqual(learning_refresh_calls[0][1], "paper_canary_trade_close")
        self.assertEqual(rebuild_calls[0]["reason"], "paper_canary_trade_close")

    def test_engine_partial_paper_canary_close_triggers_rebuild(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        rebuild_calls = []
        learning_refresh_calls = []
        engine.rebuild_execution_symbols = lambda **kwargs: rebuild_calls.append(kwargs) or {
            "status": "ok"
        }
        engine._refresh_runtime_learning_feedback = (
            lambda now, reason: learning_refresh_calls.append((now, reason))
        )

        engine._handle_trade_close_feedback(
            "BTC/USDT",
            {
                "is_full_close": False,
                "metadata": {
                    "pipeline_mode": "paper_canary",
                    "paper_canary_mode": "soft_review",
                },
            },
        )

        self.assertEqual(len(rebuild_calls), 1)
        self.assertEqual(len(learning_refresh_calls), 1)
        self.assertEqual(learning_refresh_calls[0][1], "paper_canary_partial_close")
        self.assertEqual(rebuild_calls[0]["reason"], "paper_canary_partial_close")

    def test_engine_triggers_latency_circuit_breaker(self):
        import core.engine as engine_module

        class LatencyMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def measure_latency(self, symbol):
                return {"latency_seconds": 6.0}

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.data_latency_warning_seconds = 3
        settings.exchange.data_latency_circuit_breaker_seconds = 5
        with patch.object(engine_module, "OKXMarketDataCollector", LatencyMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", LatencyMarket):
            engine = engine_module.CryptoAIV2Engine(settings)
            blocked = engine._check_market_latency(datetime.now(timezone.utc), symbols=["BTC/USDT"])
        self.assertTrue(blocked)
        self.assertEqual(engine.storage.get_state("manual_recovery_required"), "true")
        self.assertEqual(engine.storage.get_state("manual_recovery_reason"), "market_data_latency")
