import os
import importlib
import ccxt
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dashboard as dashboard_module
import main as main_module

from config import get_settings
from config import Settings as AppSettingsModel
from core.failover_market_data import FailoverMarketDataCollector
from core.feature_pipeline import FeatureInput, FeaturePipeline
from core.models import MarketRegime, PredictionResult, ResearchInsight, SuggestedAction
from core.okx_market_data import OKXMarketDataCollector
from analysis.research_manager import ResearchManager
from strategy.model_trainer import ModelTrainer, model_path_for_symbol
from strategy.xgboost_predictor import XGBoostPredictor
from backtest.walkforward import WalkForwardBacktester, WalkForwardSplit
from dashboard import (
    command_timeout_seconds,
    db_path as dashboard_db_path,
    scheduler_job_options,
)
from monitor.notifier import Notifier
from monitor.validation_sprint import ValidationSprintService
from execution.exchange_adapter import BinanceExchangeAdapter, OKXExchangeAdapter
from tests.v2_architecture_support import V2ArchitectureTestCase, make_candles


class V2PlatformAndTrainingTests(V2ArchitectureTestCase):
    def test_notifier_uses_readable_chinese_messages(self):
        notifier = Notifier(self.storage)
        captured = []

        class CaptureChannel:
            name = "capture"

            def accepts(self, level: str) -> bool:
                return True

            def send(self, event_type, title, body, level="info"):
                captured.append((event_type, title, body, level))

        notifier.add_channel(CaptureChannel())
        notifier.notify_trade_open("BTC/USDT", "LONG", 100.0, 0.8, "测试理由")
        notifier.notify_analysis_result("BTC/USDT", "LONG", 0.9, "策略通过")
        notifier.notify_daily_report([], [], 1000.0)

        all_text = "\n".join(f"{title}\n{body}" for _, title, body, _ in captured)
        self.assertIn("开仓", all_text)
        self.assertIn("分析结果", all_text)
        self.assertIn("每日交易报告", all_text)
        self.assertNotIn("寮", all_text)

    def test_notifier_daily_report_counts_partial_realized_pnl(self):
        notifier = Notifier(self.storage)
        captured = []

        class CaptureChannel:
            name = "capture"

            def accepts(self, level: str) -> bool:
                return True

            def send(self, event_type, title, body, level="info"):
                captured.append((event_type, title, body, level))

        notifier.add_channel(CaptureChannel())
        self.storage.insert_execution_event(
            "close",
            "BTC/USDT",
            {
                "pnl": 5.0,
                "incremental_pnl": 5.0,
                "is_full_close": False,
            },
        )
        notifier.notify_daily_report([], [], 1000.0)

        body = captured[-1][2]
        self.assertIn("今日盈亏: $+5.00", body)
        self.assertIn("今日交易: 1", body)

    def test_model_trainer_writes_metadata_and_report(self):
        market_symbol = "BTC/USDT:USDT"
        self.storage.insert_ohlcv(market_symbol, "1h", make_candles(600, 100))
        self.storage.insert_ohlcv(market_symbol, "4h", make_candles(220, 120))
        self.storage.insert_ohlcv(market_symbol, "1d", make_candles(240, 140))

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        model_dir = Path(self.db_path).parent / "models"
        settings.model.xgboost_model_path = str(model_dir / "btc_v2.json")

        trainer = ModelTrainer(self.storage, settings)
        summary = trainer.train_symbol("BTC/USDT")
        meta_path = Path(summary.model_path).with_suffix(".meta.json")

        self.assertTrue(meta_path.exists())
        self.assertEqual(
            Path(summary.model_path),
            model_path_for_symbol(model_dir / "btc_v2.json", "BTC/USDT"),
        )
        self.assertGreater(summary.rows, 0)
        self.assertIn(
            summary.reason,
            {"trained", "xgboost_missing", "insufficient_rows"},
        )
        report = trainer.render_report(summary)
        self.assertIn("训练报告", report)
        if summary.trained_with_xgboost:
            self.assertGreater(summary.holdout_rows, 0)
            self.assertGreaterEqual(summary.holdout_accuracy, 0.0)
        else:
            self.assertEqual(summary.holdout_rows, 0)
        self.storage.insert_training_run(summary.__dict__)
        with self.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM training_runs"
            ).fetchone()["c"]
        self.assertEqual(count, 1)

    def test_research_manager_promotes_core_extreme_fear_liquidity_repair_setup(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "paper"
        manager = ResearchManager(settings, self.storage)

        review = manager.review(
            symbol="BTC/USDT",
            insight=ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=0.1,
                confidence=0.55,
                risk_warning=["volatility elevated"],
                key_reason=["ok"],
                suggested_action=SuggestedAction.HOLD,
            ),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.82,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=SimpleNamespace(
                values={
                    "volume_ratio_1h": 0.21,
                    "adx_4h": 30.0,
                    "di_plus_4h": 26.0,
                    "di_minus_4h": 20.0,
                    "rsi_1h": 38.0,
                    "return_24h": -0.015,
                }
            ),
            fear_greed=14.0,
            news_summary="Bitcoin market update remains constructive",
            onchain_summary="neutral",
            adaptive_min_liquidity_ratio=0.38,
        )

        self.assertEqual(review.reviewed_action, SuggestedAction.OPEN_LONG.value)
        self.assertGreater(review.review_score, 0.12)
        self.assertIn("core_extreme_fear_liquidity_repair", review.reasons)

    def test_xgboost_training_and_predictor_integration_when_available(self):
        market_symbol = "BTC/USDT:USDT"
        self.storage.insert_ohlcv(market_symbol, "1h", make_candles(2000, 100))
        self.storage.insert_ohlcv(market_symbol, "4h", make_candles(700, 120))
        self.storage.insert_ohlcv(market_symbol, "1d", make_candles(320, 140))

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.training.minimum_training_rows = 100
        model_dir = Path(self.db_path).parent / "models"
        settings.model.xgboost_model_path = str(model_dir / "btc_real.json")

        trainer = ModelTrainer(self.storage, settings)
        summary = trainer.train_symbol("BTC/USDT")

        if summary.trained_with_xgboost:
            self.assertTrue(Path(summary.model_path).exists())
            predictor = XGBoostPredictor(summary.model_path, enable_fallback=False)
            pipeline = FeaturePipeline()
            snapshot = pipeline.build(
                FeatureInput(
                    symbol="BTC/USDT",
                    candles_1h=make_candles(240, 100),
                    candles_4h=make_candles(240, 120),
                    candles_1d=make_candles(240, 140),
                    sentiment_value=0.2,
                    market_regime_score=0.8,
                )
            )
            prediction = predictor.predict(snapshot)
            self.assertEqual(prediction.model_version, Path(summary.model_path).name)
            self.assertGreaterEqual(prediction.up_probability, 0.0)
            self.assertLessEqual(prediction.up_probability, 1.0)
        else:
            self.skipTest("xgboost unavailable in current environment")

    def test_model_trainer_count_training_rows_matches_dataset_labels(self):
        market_symbol = "BTC/USDT:USDT"
        self.storage.insert_ohlcv(market_symbol, "1h", make_candles(600, 100))
        self.storage.insert_ohlcv(market_symbol, "4h", make_candles(220, 120))
        self.storage.insert_ohlcv(market_symbol, "1d", make_candles(240, 140))

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        trainer = ModelTrainer(self.storage, settings)
        dataset = trainer.build_dataset("BTC/USDT")
        self.assertEqual(trainer.count_training_rows("BTC/USDT"), len(dataset["labels"]))

    def test_model_trainer_uses_fast_alpha_training_profile_for_core_symbols(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.training.fast_alpha_training_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        trainer = ModelTrainer(self.storage, settings)

        btc_profile = trainer._training_profile("BTC/USDT")
        sol_profile = trainer._training_profile("SOL/USDT")

        self.assertEqual(btc_profile["mode"], "fast_alpha")
        self.assertEqual(btc_profile["sample_timeframe"], "1h")
        self.assertGreater(float(btc_profile["label_min_abs_net_return_pct"]), 0.0)
        self.assertEqual(sol_profile["mode"], "default")
        self.assertEqual(sol_profile["sample_timeframe"], "4h")

    def test_model_trainer_fast_alpha_label_skips_neutral_samples(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        trainer = ModelTrainer(self.storage, settings)

        self.assertEqual(trainer._label_from_net_return_pct(0.35, 0.20), 1)
        self.assertEqual(trainer._label_from_net_return_pct(-0.35, 0.20), 0)
        self.assertIsNone(trainer._label_from_net_return_pct(0.10, 0.20))
        self.assertIsNone(trainer._label_from_net_return_pct(-0.10, 0.20))

    def test_model_trainer_rejects_weaker_candidate_and_preserves_active_model(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.training.minimum_training_rows = 5
        model_dir = Path(self.db_path).parent / "models"
        settings.model.xgboost_model_path = str(model_dir / "btc_v2.json")
        settings.ab_testing.challenger_model_path = str(model_dir / "btc_challenger.json")

        trainer = ModelTrainer(self.storage, settings)
        active_path = model_path_for_symbol(model_dir / "btc_v2.json", "BTC/USDT")
        challenger_path = model_path_for_symbol(model_dir / "btc_challenger.json", "BTC/USDT")
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text("incumbent-model", encoding="utf-8")

        class FakeBooster:
            def __init__(self, payload: str):
                self.payload = payload

            def save_model(self, path: str):
                Path(path).write_text(self.payload, encoding="utf-8")

            def get_score(self, importance_type="gain"):
                return {"f1": 1.0}

        trainer.build_dataset = lambda symbol: {
            "rows": [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]],
            "labels": [0, 1, 0, 1, 0, 1],
            "feature_names": ["f1"],
            "timestamps": [1, 2, 3, 4, 5, 6],
            "current_closes": [1.0] * 6,
            "next_closes": [1.1] * 6,
        }
        trainer._train_booster = lambda **kwargs: FakeBooster("candidate-model")
        trainer._evaluate_booster = lambda **kwargs: (0.58, 0.52)
        trainer._evaluate_model_path = lambda **kwargs: {
            "accuracy": 0.65,
            "logloss": 0.50,
        }
        trainer._candidate_walkforward_analysis = lambda **kwargs: {
            "summary": {
                "symbol": "BTC/USDT",
                "total_splits": 1,
                "avg_win_rate": 40.0,
                "avg_trade_return_pct": -1.0,
                "total_return_pct": -1.0,
                "profit_factor": 0.8,
                "max_drawdown_pct": 9.5,
                "sharpe_like": -0.2,
            },
            "splits": [],
        }
        trainer._latest_walkforward_summary = lambda symbol: {
            "symbol": symbol,
            "total_splits": 1,
            "avg_win_rate": 55.0,
            "avg_trade_return_pct": 0.8,
            "total_return_pct": 0.8,
            "profit_factor": 1.2,
            "max_drawdown_pct": 3.5,
            "sharpe_like": 0.3,
        }

        with patch("strategy.model_trainer.load_xgboost", return_value=SimpleNamespace()):
            summary = trainer.train_symbol("BTC/USDT")

        self.assertFalse(summary.promoted_to_active)
        self.assertEqual(summary.promotion_status, "rejected")
        self.assertEqual(
            summary.promotion_reason,
            "candidate_negative_walkforward_quality",
        )
        self.assertEqual(Path(summary.model_path), challenger_path)
        self.assertEqual(Path(summary.active_model_path), active_path)
        self.assertEqual(summary.holdout_accuracy, 0.65)
        self.assertEqual(summary.candidate_holdout_accuracy, 0.58)
        self.assertEqual(
            float(summary.candidate_walkforward_summary["total_return_pct"]),
            -1.0,
        )
        self.assertEqual(active_path.read_text(encoding="utf-8"), "incumbent-model")
        self.assertEqual(challenger_path.read_text(encoding="utf-8"), "candidate-model")
        self.assertTrue(challenger_path.with_suffix(".meta.json").exists())

    def test_model_trainer_allows_lower_holdout_when_walkforward_edge_is_stronger(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.training.minimum_training_rows = 5
        model_dir = Path(self.db_path).parent / "models"
        settings.model.xgboost_model_path = str(model_dir / "btc_v2.json")
        settings.ab_testing.challenger_model_path = str(model_dir / "btc_challenger.json")

        trainer = ModelTrainer(self.storage, settings)
        active_path = model_path_for_symbol(model_dir / "btc_v2.json", "BTC/USDT")
        challenger_path = model_path_for_symbol(model_dir / "btc_challenger.json", "BTC/USDT")
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text("incumbent-model", encoding="utf-8")
        challenger_path.write_text("stale-challenger", encoding="utf-8")
        challenger_path.with_suffix(".meta.json").write_text("{}", encoding="utf-8")

        class FakeBooster:
            def __init__(self, payload: str):
                self.payload = payload

            def save_model(self, path: str):
                Path(path).write_text(self.payload, encoding="utf-8")

            def get_score(self, importance_type="gain"):
                return {"f1": 1.0}

        trainer.build_dataset = lambda symbol: {
            "rows": [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]],
            "labels": [0, 1, 0, 1, 0, 1],
            "feature_names": ["f1"],
            "timestamps": [1, 2, 3, 4, 5, 6],
            "current_closes": [1.0] * 6,
            "next_closes": [1.1] * 6,
        }
        trainer._train_booster = lambda **kwargs: FakeBooster("candidate-model")
        trainer._evaluate_booster = lambda **kwargs: (0.61, 0.55)
        trainer._evaluate_model_path = lambda **kwargs: {
            "accuracy": 0.65,
            "logloss": 0.50,
        }
        trainer._candidate_walkforward_analysis = lambda **kwargs: {
            "summary": {
                "symbol": "BTC/USDT",
                "total_splits": 1,
                "avg_win_rate": 62.0,
                "avg_trade_return_pct": 1.5,
                "total_return_pct": 1.5,
                "profit_factor": 1.4,
                "max_drawdown_pct": 1.8,
                "sharpe_like": 0.6,
            },
            "splits": [],
        }
        trainer._latest_walkforward_summary = lambda symbol: {
            "symbol": symbol,
            "total_splits": 1,
            "avg_win_rate": 55.0,
            "avg_trade_return_pct": 0.8,
            "total_return_pct": 0.8,
            "profit_factor": 1.2,
            "max_drawdown_pct": 3.6,
            "sharpe_like": 0.3,
        }

        with patch("strategy.model_trainer.load_xgboost", return_value=SimpleNamespace()):
            summary = trainer.train_symbol("BTC/USDT")

        self.assertFalse(summary.promoted_to_active)
        self.assertEqual(summary.promotion_status, "canary_pending")
        self.assertEqual(
            summary.promotion_reason,
            "candidate_higher_walkforward_expectancy",
        )
        self.assertEqual(Path(summary.model_path), challenger_path)
        self.assertEqual(summary.holdout_accuracy, 0.65)
        self.assertEqual(summary.candidate_holdout_accuracy, 0.61)
        self.assertTrue(summary.model_id.startswith("mdl_BTC_USDT_"))
        self.assertEqual(summary.challenger_model_id, summary.model_id)
        self.assertEqual(
            float(summary.candidate_walkforward_summary["total_return_pct"]),
            1.5,
        )
        self.assertEqual(active_path.read_text(encoding="utf-8"), "incumbent-model")
        self.assertEqual(challenger_path.read_text(encoding="utf-8"), "candidate-model")
        self.assertTrue(challenger_path.with_suffix(".meta.json").exists())

    def test_model_trainer_rejects_candidate_below_recent_walkforward_baseline(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.training.minimum_training_rows = 5
        model_dir = Path(self.db_path).parent / "models"
        settings.model.xgboost_model_path = str(model_dir / "btc_v2.json")
        settings.ab_testing.challenger_model_path = str(model_dir / "btc_challenger.json")

        trainer = ModelTrainer(self.storage, settings)
        active_path = model_path_for_symbol(model_dir / "btc_v2.json", "BTC/USDT")
        challenger_path = model_path_for_symbol(model_dir / "btc_challenger.json", "BTC/USDT")
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text("incumbent-model", encoding="utf-8")

        class FakeBooster:
            def __init__(self, payload: str):
                self.payload = payload

            def save_model(self, path: str):
                Path(path).write_text(self.payload, encoding="utf-8")

            def get_score(self, importance_type="gain"):
                return {"f1": 1.0}

        trainer.build_dataset = lambda symbol: {
            "rows": [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]],
            "labels": [0, 1, 0, 1, 0, 1],
            "feature_names": ["f1"],
            "timestamps": [1, 2, 3, 4, 5, 6],
            "current_closes": [1.0] * 6,
            "next_closes": [1.1] * 6,
        }
        trainer._train_booster = lambda **kwargs: FakeBooster("candidate-model")
        trainer._evaluate_booster = lambda **kwargs: (0.72, 0.30)
        trainer._evaluate_model_path = lambda **kwargs: {
            "accuracy": 0.65,
            "logloss": 0.50,
        }
        trainer._candidate_walkforward_analysis = lambda **kwargs: {
            "summary": {
                "symbol": "BTC/USDT",
                "total_splits": 2,
                "avg_win_rate": 54.0,
                "avg_trade_return_pct": 0.35,
                "total_return_pct": 0.7,
                "profit_factor": 1.15,
                "sharpe_like": 0.2,
            },
            "splits": [],
        }
        trainer._latest_walkforward_summary = lambda symbol: {
            "symbol": symbol,
            "total_splits": 1,
            "avg_win_rate": 52.0,
            "avg_trade_return_pct": 0.3,
            "total_return_pct": 0.6,
            "profit_factor": 1.1,
            "sharpe_like": 0.1,
        }
        trainer._recent_walkforward_baseline = lambda symbol: {
            "symbol": symbol,
            "history_count": 3,
            "avg_total_return_pct": 1.8,
            "avg_profit_factor": 1.35,
            "avg_win_rate": 60.0,
            "min_total_return_pct": 1.2,
        }

        with patch("strategy.model_trainer.load_xgboost", return_value=SimpleNamespace()):
            summary = trainer.train_symbol("BTC/USDT")

        self.assertFalse(summary.promoted_to_active)
        self.assertEqual(summary.promotion_status, "rejected")
        self.assertEqual(
            summary.promotion_reason,
            "candidate_below_recent_walkforward_baseline",
        )
        self.assertEqual(Path(summary.model_path), challenger_path)

    def test_model_trainer_rejects_candidate_when_walkforward_diverges_from_live(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.training.minimum_training_rows = 5
        model_dir = Path(self.db_path).parent / "models"
        settings.model.xgboost_model_path = str(model_dir / "btc_v2.json")
        settings.ab_testing.challenger_model_path = str(model_dir / "btc_challenger.json")

        trainer = ModelTrainer(self.storage, settings)
        active_path = model_path_for_symbol(model_dir / "btc_v2.json", "BTC/USDT")
        challenger_path = model_path_for_symbol(model_dir / "btc_challenger.json", "BTC/USDT")
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text("incumbent-model", encoding="utf-8")

        for idx in range(5):
            trade_id = f"live-loss-{idx}"
            self.storage.insert_trade(
                {
                    "id": trade_id,
                    "symbol": "BTC/USDT",
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "quantity": 1.0,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "rationale": "live trade",
                    "confidence": 0.7,
                }
            )
            self.storage.update_trade_exit(
                trade_id,
                95.0,
                datetime.now(timezone.utc).isoformat(),
                -5.0,
                -5.0,
            )

        class FakeBooster:
            def __init__(self, payload: str):
                self.payload = payload

            def save_model(self, path: str):
                Path(path).write_text(self.payload, encoding="utf-8")

            def get_score(self, importance_type="gain"):
                return {"f1": 1.0}

        trainer.build_dataset = lambda symbol: {
            "rows": [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]],
            "labels": [0, 1, 0, 1, 0, 1],
            "feature_names": ["f1"],
            "timestamps": [1, 2, 3, 4, 5, 6],
            "current_closes": [1.0] * 6,
            "next_closes": [1.1] * 6,
        }
        trainer._train_booster = lambda **kwargs: FakeBooster("candidate-model")
        trainer._evaluate_booster = lambda **kwargs: (0.68, 0.40)
        trainer._evaluate_model_path = lambda **kwargs: {
            "accuracy": 0.65,
            "logloss": 0.50,
        }
        trainer._candidate_walkforward_analysis = lambda **kwargs: {
            "summary": {
                "symbol": "BTC/USDT",
                "total_splits": 1,
                "trade_count": 6,
                "avg_win_rate": 80.0,
                "avg_trade_return_pct": 95.0,
                "total_return_pct": 950.0,
                "profit_factor": 2.6,
                "max_drawdown_pct": 4.0,
                "sharpe_like": 2.0,
            },
            "splits": [],
        }
        trainer._latest_walkforward_summary = lambda symbol: {
            "symbol": symbol,
            "total_splits": 1,
            "trade_count": 6,
            "avg_win_rate": 55.0,
            "avg_trade_return_pct": 0.8,
            "total_return_pct": 0.8,
            "profit_factor": 1.2,
            "max_drawdown_pct": 3.6,
            "sharpe_like": 0.3,
        }

        with patch("strategy.model_trainer.load_xgboost", return_value=SimpleNamespace()):
            summary = trainer.train_symbol("BTC/USDT")

        self.assertFalse(summary.promoted_to_active)
        self.assertEqual(summary.promotion_status, "rejected")
        self.assertEqual(
            summary.promotion_reason,
            "candidate_walkforward_live_pnl_divergence",
        )
        self.assertEqual(Path(summary.model_path), challenger_path)
        report = trainer.render_report(summary)
        self.assertIn("一致性校验", report)
        self.assertIn("Walk-Forward 收益与最近实盘净盈亏明显背离", report)

    def test_main_train_and_report_commands_work(self):
        class FakeEngine:
            def train_models(self):
                return [{"symbol": "BTC/USDT", "rows": 10, "report": "train-report"}]

            def generate_reports(self):
                return {"daily": "daily-report", "weekly": "weekly-report"}

            def run_walkforward(self, symbol):
                return {
                    "symbol": symbol,
                    "summary": {"total_splits": 1},
                    "report": "wf-report",
                }

            def run_backfill(self, days):
                return {"BTC/USDT": {"1h": 100}}

            def run_reconciliation(self):
                return {"status": "ok", "mismatch_count": 0, "details": {}}

            def approve_manual_recovery(self):
                return {"status": "approved"}

            def run_backtest(self, symbol):
                return {
                    "symbol": symbol,
                    "summary": {"total_trades": 1},
                    "report": "bt-report",
                }

            def run_health_check(self):
                return {"report": "health-report"}

            def run_guard_report(self):
                return {"report": "guard-report"}

            def run_ab_test_report(self):
                return {"report": "abtest-report"}

            def run_drift_report(self):
                return {"report": "drift-report"}

            def run_metrics(self):
                return {"report": "metrics-report"}

            def run_maintenance(self):
                return {"report": "maintenance-report"}

            def run_failure_report(self):
                return {"report": "failure-report"}

            def run_incident_report(self):
                return {"report": "incident-report"}

            def run_ops_overview(self):
                return {"report": "ops-report"}

            def run_alpha_diagnostics_report(self, symbols=None):
                return {"report": "alpha-report", "symbols": symbols or []}

            def run_pool_attribution_report(self, symbols=None):
                return {"report": "attribution-report", "symbols": symbols or []}

            def run_entry_scan(self):
                return {"status": "ok", "opened_positions": 0}

            scheduler = SimpleNamespace(
                run_job=lambda job_name: {"job_name": job_name, "status": "ok"}
            )

            def run_once(self):
                raise AssertionError("run_once should not be called")

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "train"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "report"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "walkforward", "BTC/USDT"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "backfill", "30"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "reconcile"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "approve-recovery"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "backtest", "BTC/USDT"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "health"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "guards"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "abtest"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "drift"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "metrics"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "maintenance"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "failures"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "incidents"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "ops"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "entries"]):
            with self.assertRaises(SystemExit) as raised:
                main_module.main()
            self.assertEqual(raised.exception.code, 0)

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "alpha", "BTC/USDT,ETH/USDT"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "attribution", "BTC/USDT,ETH/USDT"]):
            main_module.main()

        with patch.object(main_module, "build_engine", return_value=FakeEngine()), \
             patch.object(main_module.sys, "argv", ["main.py", "schedule", "health"]):
            main_module.main()

    def test_walkforward_backtest_returns_summary(self):
        market_symbol = "BTC/USDT:USDT"
        self.storage.insert_ohlcv(market_symbol, "1h", make_candles(2000, 100))
        self.storage.insert_ohlcv(market_symbol, "4h", make_candles(700, 120))
        self.storage.insert_ohlcv(market_symbol, "1d", make_candles(320, 140))

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        backtester = WalkForwardBacktester(self.storage, settings)
        result = backtester.run("BTC/USDT")

        self.assertEqual(result["symbol"], "BTC/USDT")
        self.assertIn("summary", result)
        self.assertGreaterEqual(result["summary"]["total_splits"], 1)
        report = backtester.render_report(result)
        self.assertIn("Walk-Forward", report)
        self.assertIn("总切分数", report)

    def test_model_trainer_walkforward_summary_keeps_percentage_units(self):
        summary = ModelTrainer._walkforward_summary(
            [10.0, -5.0, 2.5],
            [
                {"win_rate": 50.0, "avg_trade_return_pct": 2.5, "total_return_pct": 5.0},
                {"win_rate": 100.0, "avg_trade_return_pct": 2.5, "total_return_pct": 2.5},
            ],
            symbol="BTC/USDT",
        )

        self.assertAlmostEqual(summary["avg_trade_return_pct"], 2.5, places=6)
        self.assertAlmostEqual(summary["expectancy_pct"], 2.5, places=6)
        self.assertAlmostEqual(summary["total_return_pct"], 7.5, places=6)
        self.assertAlmostEqual(summary["max_drawdown_pct"], 5.0, places=6)

    def test_walkforward_backtester_summary_keeps_percentage_units(self):
        summary = WalkForwardBacktester._summary(
            [10.0, -5.0, 2.5],
            [
                WalkForwardSplit(
                    train_rows=10,
                    test_rows=5,
                    win_rate=50.0,
                    avg_trade_return_pct=2.5,
                    total_return_pct=5.0,
                ),
                WalkForwardSplit(
                    train_rows=15,
                    test_rows=5,
                    win_rate=100.0,
                    avg_trade_return_pct=2.5,
                    total_return_pct=2.5,
                ),
            ],
        )

        self.assertAlmostEqual(summary["avg_trade_return_pct"], 2.5, places=6)
        self.assertAlmostEqual(summary["total_return_pct"], 7.5, places=6)

    def test_realistic_backtest_engine_applies_funding_cost_and_equity_consistently(self):
        from backtest.realistic_engine import RealisticBacktestEngine

        candles = [
            {
                "timestamp": 1700000000000 + i * 3600000,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            }
            for i in range(140)
        ]
        self.storage.insert_ohlcv("BTC/USDT", "1h", candles)

        engine = RealisticBacktestEngine(self.storage)
        signal_calls = {"count": 0}
        engine.technical_calculate_all = lambda window: {}

        def fake_signal(_tech):
            signal_calls["count"] += 1
            if signal_calls["count"] == 1:
                return {"direction": "LONG", "confidence": 0.7, "reasons": []}
            return {"direction": "FLAT", "confidence": 0.3, "reasons": []}

        engine.generate_signal = fake_signal
        result = engine.run_backtest("BTC/USDT")

        self.assertLess(result["final_balance"], 9994.0)
        self.assertGreater(result["cost_breakdown"]["total_funding"], 0.0)
        self.assertLess(max(point["equity"] for point in result["equity_curve"][1:]), 10000.0)
        self.assertEqual(result["trades"][0]["exit_idx"], len(candles) - 1)

    def test_engine_walkforward_persists_result(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        engine = engine_module.CryptoAIV2Engine(settings)
        fake_result = {
            "symbol": "BTC/USDT",
            "splits": [{"train_rows": 10, "test_rows": 5, "win_rate": 50.0, "avg_trade_return_pct": 1.0, "total_return_pct": 5.0}],
            "summary": {
                "total_splits": 1,
                "avg_win_rate": 50.0,
                "avg_trade_return_pct": 1.0,
                "total_return_pct": 5.0,
                "profit_factor": 1.5,
                "sharpe_like": 0.8,
            },
        }
        engine.walkforward = SimpleNamespace(
            run=lambda symbol: fake_result,
            render_report=lambda result: "wf-report",
        )

        result = engine.run_walkforward("BTC/USDT")
        self.assertEqual(result["report"], "wf-report")
        with engine.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM walkforward_runs"
            ).fetchone()["c"]
            report_count = conn.execute(
                "SELECT COUNT(*) AS c FROM report_artifacts WHERE report_type='walkforward'"
            ).fetchone()["c"]
        self.assertEqual(count, 1)
        self.assertEqual(report_count, 1)

    def test_engine_backtest_persists_result(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        fake_result = {
            "symbol": "BTC/USDT",
            "summary": {
                "total_trades": 1,
                "win_rate": 100.0,
                "total_return_pct": 5.0,
                "max_drawdown_pct": 1.0,
                "profit_factor": 2.0,
                "sharpe_like": 1.0,
            },
            "trades": [{"entry_price": 100.0, "exit_price": 105.0}],
        }
        engine.backtester = SimpleNamespace(
            run=lambda symbol: fake_result,
            render_report=lambda result: "bt-report",
        )
        result = engine.run_backtest("BTC/USDT")
        self.assertEqual(result["report"], "bt-report")
        with engine.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM backtest_runs"
            ).fetchone()["c"]
        self.assertEqual(count, 1)

    def test_engine_validation_sprint_persists_report(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        fake_result = {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "baseline": [
                {
                    "symbol": "BTC/USDT",
                    "backtest": {
                        "total_trades": 3,
                        "total_return_pct": 5.0,
                        "max_drawdown_pct": 1.0,
                        "profit_factor": 1.4,
                    },
                    "walkforward": {
                        "total_splits": 3,
                        "avg_win_rate": 60.0,
                        "total_return_pct": 4.0,
                        "profit_factor": 1.3,
                    },
                }
            ],
            "top_candidates": [
                {
                    "symbol": "BTC/USDT",
                    "xgboost_threshold": 0.64,
                    "final_score_threshold": 0.48,
                    "min_liquidity_ratio": 0.5,
                    "total_trades": 3,
                    "total_return_pct": 5.0,
                    "max_drawdown_pct": 1.0,
                    "profit_factor": 1.4,
                    "sharpe_like": 0.9,
                    "score": 6.0,
                }
            ],
            "scan_count": 18,
            "walkforward_window_days": 10,
        }
        engine.validation = SimpleNamespace(
            run=lambda symbols: fake_result,
            render=lambda result, lang=None: "validation-report",
        )
        result = engine.run_validation_sprint(symbols=["BTC/USDT", "ETH/USDT"])
        self.assertEqual(result["report"], "validation-report")
        with engine.storage._conn() as conn:
            report_count = conn.execute(
                "SELECT COUNT(*) AS c FROM report_artifacts WHERE report_type='validation_sprint'"
            ).fetchone()["c"]
        self.assertEqual(report_count, 1)

    def test_engine_pool_attribution_persists_report(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        fake_result = {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "closed_trade_count": 1,
            "open_trade_count": 0,
            "symbol_summary": [],
            "closed_trades": [],
            "open_trades": [],
        }
        engine.pool_attribution = SimpleNamespace(
            build=lambda symbols: fake_result,
            render=lambda result, lang=None: "attribution-report",
        )

        result = engine.run_pool_attribution_report(symbols=["BTC/USDT", "ETH/USDT"])

        self.assertEqual(result["report"], "attribution-report")
        with engine.storage._conn() as conn:
            report_count = conn.execute(
                "SELECT COUNT(*) AS c FROM report_artifacts WHERE report_type='pool_attribution'"
            ).fetchone()["c"]
        self.assertEqual(report_count, 1)

    def test_engine_alpha_diagnostics_persists_report(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        fake_result = {
            "execution_symbols": ["BTC/USDT", "ETH/USDT"],
            "paper_canary_mix": {"total": {"total": 0}},
            "learning": {"blocked_setup_count": 0},
        }
        engine.alpha_diagnostics = SimpleNamespace(
            build=lambda symbols: fake_result,
            render=lambda result, lang=None: "alpha-report",
        )

        result = engine.run_alpha_diagnostics_report(symbols=["BTC/USDT", "ETH/USDT"])

        self.assertEqual(result["report"], "alpha-report")
        with engine.storage._conn() as conn:
            report_count = conn.execute(
                "SELECT COUNT(*) AS c FROM report_artifacts WHERE report_type='alpha_diagnostics'"
            ).fetchone()["c"]
        self.assertEqual(report_count, 1)

    def test_validation_sprint_prefers_expectancy_and_drawdown_quality(self):
        weaker_high_return = {
            "total_trades": 6,
            "total_return_pct": 12.0,
            "avg_trade_return_pct": 0.25,
            "win_rate": 50.0,
            "max_drawdown_pct": 9.0,
            "profit_factor": 1.05,
            "sharpe_like": 0.4,
        }
        stronger_quality = {
            "total_trades": 6,
            "total_return_pct": 8.0,
            "avg_trade_return_pct": 0.55,
            "win_rate": 66.0,
            "max_drawdown_pct": 2.0,
            "profit_factor": 1.8,
            "sharpe_like": 0.8,
        }

        weaker_score = ValidationSprintService._candidate_score(weaker_high_return)
        stronger_score = ValidationSprintService._candidate_score(stronger_quality)

        self.assertGreater(stronger_score, weaker_score)

    def test_engine_backfill_uses_all_symbols_and_timeframes(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.calls = []

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=500):
                self.calls.append((symbol, timeframe, since, limit))
                return [1, 2, 3]

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.timeframes = ["1h", "4h"]

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)
            result = engine.run_backfill(days=30)
            self.assertEqual(result["BTC/USDT"]["1h"], 3)
            self.assertEqual(result["ETH/USDT"]["4h"], 3)

    def test_engine_watchlist_refresh_returns_snapshot(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        engine.watchlist = SimpleNamespace(
            refresh=lambda force=False, now=None: SimpleNamespace(
                active_symbols=["BTC/USDT", "ETH/USDT"],
                added_symbols=["ETH/USDT"],
                removed_symbols=[],
                whitelist=[],
                blacklist=[],
                candidates=[],
                refreshed_at=datetime.now(timezone.utc).isoformat(),
                refresh_reason="manual_refresh",
            )
        )
        engine.storage.set_json_state(
            engine.EXECUTION_SYMBOLS_STATE_KEY,
            ["BTC/USDT", "ETH/USDT"],
        )
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
        snapshot = engine.run_watchlist_refresh()
        self.assertEqual(snapshot["active_symbols"], ["BTC/USDT", "ETH/USDT"])
        self.assertEqual(snapshot["execution_symbols"], ["BTC/USDT", "ETH/USDT"])
        self.assertEqual(snapshot["refresh_reason"], "manual_refresh")

    def test_engine_rebuild_execution_symbols_prioritizes_recent_edge(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=500):
                return [1, 2, 3, 4]

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.max_active_symbols = 4
        settings.exchange.core_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "AVAX/USDT",
            "POL/USDT",
            "RENDER/USDT",
            "WLD/USDT",
        ]
        settings.exchange.candidate_symbols = list(settings.exchange.symbols)
        settings.risk.execution_pool_target_size = 4
        settings.risk.execution_symbol_min_samples = 8
        settings.risk.execution_symbol_accuracy_floor_pct = 45.0

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        def create_ready_model(symbol: str):
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

        create_ready_model("BTC/USDT")
        create_ready_model("ETH/USDT")
        create_ready_model("WLD/USDT")
        engine.storage.set_json_state(engine.EXECUTION_SYMBOLS_STATE_KEY, ["ETH/USDT"])

        summary = {
            "BTC/USDT": {"count": 20, "correct": 10, "accuracy_pct": 50.0},
            "ETH/USDT": {"count": 30, "correct": 11, "accuracy_pct": 36.67},
            "AVAX/USDT": {"count": 4, "correct": 3, "accuracy_pct": 75.0},
            "POL/USDT": {"count": 4, "correct": 4, "accuracy_pct": 100.0},
            "RENDER/USDT": {"count": 4, "correct": 4, "accuracy_pct": 100.0},
            "WLD/USDT": {"count": 24, "correct": 7, "accuracy_pct": 29.17},
        }
        edge_summary = {
            "BTC/USDT": {
                "count": 20,
                "sample_count": 20,
                "accuracy_pct": 50.0,
                "expectancy_pct": 0.12,
                "profit_factor": 1.15,
                "max_drawdown_pct": 2.0,
                "objective_score": 1.1,
            },
            "ETH/USDT": {
                "count": 30,
                "sample_count": 30,
                "accuracy_pct": 36.67,
                "expectancy_pct": -0.18,
                "profit_factor": 0.72,
                "max_drawdown_pct": 6.0,
                "objective_score": -1.4,
            },
            "AVAX/USDT": {
                "count": 4,
                "sample_count": 4,
                "accuracy_pct": 75.0,
                "expectancy_pct": 0.10,
                "profit_factor": 1.10,
                "max_drawdown_pct": 2.0,
                "objective_score": 0.7,
            },
            "POL/USDT": {
                "count": 4,
                "sample_count": 4,
                "accuracy_pct": 100.0,
                "expectancy_pct": 0.22,
                "profit_factor": 1.40,
                "max_drawdown_pct": 1.0,
                "objective_score": 1.3,
            },
            "RENDER/USDT": {
                "count": 4,
                "sample_count": 4,
                "accuracy_pct": 100.0,
                "expectancy_pct": 0.18,
                "profit_factor": 1.25,
                "max_drawdown_pct": 1.5,
                "objective_score": 1.0,
            },
            "WLD/USDT": {
                "count": 24,
                "sample_count": 24,
                "accuracy_pct": 29.17,
                "expectancy_pct": -0.25,
                "profit_factor": 0.60,
                "max_drawdown_pct": 9.5,
                "objective_score": -2.2,
            },
        }
        engine.performance.build_symbol_accuracy_summary = lambda limit=500: summary
        engine.performance.build_symbol_edge_summary = lambda limit=500: edge_summary

        def fake_train_symbol(symbol: str):
            model_path = model_dir / f"xgboost_v2_{symbol.replace('/', '_')}.json"
            model_path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(
                symbol=symbol,
                rows=settings.training.minimum_training_rows,
                feature_count=10,
                positives=5,
                negatives=5,
                model_path=str(model_path),
                trained_with_xgboost=True,
                holdout_accuracy=0.61,
            )

        engine.trainer.train_symbol = fake_train_symbol
        engine.trainer.render_report = lambda summary, lang=None: f"trained {summary.symbol}"

        result = engine.rebuild_execution_symbols(
            force=True,
            now=datetime(2026, 3, 26, tzinfo=timezone.utc),
            reason="manual_test",
        )

        self.assertEqual(
            result["execution_symbols"],
            ["BTC/USDT", "POL/USDT", "RENDER/USDT", "AVAX/USDT"],
        )
        self.assertEqual(result["removed_symbols"], ["ETH/USDT"])
        self.assertEqual(result["added_symbols"], ["BTC/USDT", "POL/USDT", "RENDER/USDT", "AVAX/USDT"])
        self.assertEqual(
            engine.get_active_symbols(force_refresh=False),
            ["BTC/USDT", "POL/USDT", "RENDER/USDT", "AVAX/USDT"],
        )
        self.assertEqual(
            engine.storage.get_json_state(engine.EXECUTION_SYMBOLS_STATE_KEY),
            ["BTC/USDT", "POL/USDT", "RENDER/USDT", "AVAX/USDT"],
        )

    def test_engine_rebuild_execution_symbols_prioritizes_shadow_candidates(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=500):
                return [1, 2, 3, 4]

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.max_active_symbols = 2
        settings.exchange.core_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.symbols = ["BTC/USDT", "ETH/USDT", "FIL/USDT"]
        settings.exchange.candidate_symbols = list(settings.exchange.symbols)
        settings.risk.execution_pool_target_size = 2
        settings.risk.execution_symbol_min_samples = 8
        settings.risk.execution_symbol_accuracy_floor_pct = 45.0

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        def create_ready_model(symbol: str):
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

        create_ready_model("BTC/USDT")
        create_ready_model("ETH/USDT")
        create_ready_model("FIL/USDT")
        engine.storage.set_json_state(engine.EXECUTION_SYMBOLS_STATE_KEY, ["ETH/USDT"])
        engine.storage.set_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {
                "FIL/USDT": {
                    "symbol": "FIL/USDT",
                    "status": "shadow",
                }
            },
        )

        summary = {
            "BTC/USDT": {"count": 20, "correct": 10, "accuracy_pct": 50.0},
            "ETH/USDT": {"count": 30, "correct": 11, "accuracy_pct": 36.67},
            "SOL/USDT": {"count": 18, "correct": 7, "accuracy_pct": 38.89},
        }
        edge_summary = {
            "BTC/USDT": {
                "count": 20,
                "sample_count": 20,
                "accuracy_pct": 50.0,
                "expectancy_pct": 0.10,
                "profit_factor": 1.10,
                "max_drawdown_pct": 2.2,
                "objective_score": 0.8,
            },
            "ETH/USDT": {
                "count": 30,
                "sample_count": 30,
                "accuracy_pct": 36.67,
                "expectancy_pct": -0.12,
                "profit_factor": 0.85,
                "max_drawdown_pct": 5.5,
                "objective_score": -1.1,
            },
            "SOL/USDT": {
                "count": 18,
                "sample_count": 18,
                "accuracy_pct": 38.89,
                "expectancy_pct": -0.18,
                "profit_factor": 0.82,
                "max_drawdown_pct": 6.5,
                "objective_score": -1.5,
            },
        }
        engine.performance.build_symbol_accuracy_summary = lambda limit=500: summary
        engine.performance.build_symbol_edge_summary = lambda limit=500: edge_summary
        engine.shadow_runtime.build_observation_feedback = lambda limit=500: {
            "FIL/USDT": {
                "shadow_eval_count": 12,
                "shadow_accuracy_pct": 66.0,
                "shadow_trade_count": 4,
                "shadow_positive_ratio": 0.75,
                "shadow_avg_pnl_pct": 0.28,
            },
            "SOL/USDT": {
                "shadow_eval_count": 12,
                "shadow_accuracy_pct": 33.0,
                "shadow_trade_count": 4,
                "shadow_positive_ratio": 0.25,
                "shadow_avg_pnl_pct": -0.21,
            },
        }

        def fake_train_symbol(symbol: str):
            model_path = model_dir / f"xgboost_v2_{symbol.replace('/', '_')}.json"
            model_path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(
                symbol=symbol,
                rows=settings.training.minimum_training_rows,
                feature_count=10,
                positives=5,
                negatives=5,
                model_path=str(model_path),
                trained_with_xgboost=True,
                holdout_accuracy=0.61,
            )

        engine.trainer.train_symbol = fake_train_symbol
        engine.trainer.render_report = lambda summary, lang=None: f"trained {summary.symbol}"

        result = engine.rebuild_execution_symbols(
            force=True,
            now=datetime(2026, 3, 26, tzinfo=timezone.utc),
            reason="manual_test",
        )

        self.assertEqual(result["execution_symbols"], ["FIL/USDT", "BTC/USDT"])
        self.assertEqual(result["removed_symbols"], ["ETH/USDT"])
        self.assertEqual(result["added_symbols"], ["FIL/USDT", "BTC/USDT"])
        self.assertEqual(engine.get_active_symbols(force_refresh=False), ["FIL/USDT", "BTC/USDT"])

    def test_engine_rebuild_execution_symbols_respects_exchange_allowlist(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=500):
                return [1, 2, 3, 4]

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.max_active_symbols = 2
        settings.exchange.core_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.candidate_symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "FIL/USDT",
            "POL/USDT",
        ]
        settings.risk.execution_pool_target_size = 2

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        def create_ready_model(symbol: str):
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

        for symbol in ("BTC/USDT", "ETH/USDT", "FIL/USDT", "POL/USDT"):
            create_ready_model(symbol)

        engine.storage.set_json_state(
            engine.EXECUTION_SYMBOLS_STATE_KEY,
            ["FIL/USDT", "POL/USDT"],
        )
        engine.storage.set_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {
                "FIL/USDT": {
                    "symbol": "FIL/USDT",
                    "status": "shadow",
                }
            },
        )

        edge_summary = {
            "BTC/USDT": {
                "count": 12,
                "sample_count": 12,
                "accuracy_pct": 60.0,
                "expectancy_pct": 0.10,
                "profit_factor": 1.15,
                "max_drawdown_pct": 2.0,
                "objective_score": 1.0,
            },
            "ETH/USDT": {
                "count": 10,
                "sample_count": 10,
                "accuracy_pct": 56.0,
                "expectancy_pct": 0.06,
                "profit_factor": 1.02,
                "max_drawdown_pct": 2.5,
                "objective_score": 0.6,
            },
            "FIL/USDT": {
                "count": 18,
                "sample_count": 18,
                "accuracy_pct": 74.0,
                "expectancy_pct": 0.18,
                "profit_factor": 1.45,
                "max_drawdown_pct": 1.8,
                "objective_score": 2.3,
            },
            "POL/USDT": {
                "count": 16,
                "sample_count": 16,
                "accuracy_pct": 72.0,
                "expectancy_pct": 0.14,
                "profit_factor": 1.30,
                "max_drawdown_pct": 1.5,
                "objective_score": 1.9,
            },
        }
        engine.performance.build_symbol_accuracy_summary = lambda limit=500: edge_summary
        engine.performance.build_symbol_edge_summary = lambda limit=500: edge_summary
        engine.shadow_runtime.build_observation_feedback = lambda limit=500: {
            "FIL/USDT": {
                "shadow_eval_count": 16,
                "shadow_accuracy_pct": 68.0,
                "shadow_trade_count": 5,
                "shadow_positive_ratio": 0.8,
                "shadow_avg_pnl_pct": 0.24,
            }
        }

        result = engine.rebuild_execution_symbols(
            force=True,
            now=datetime(2026, 4, 6, tzinfo=timezone.utc),
            reason="allowlist_gate",
        )

        self.assertEqual(result["execution_symbols"], ["BTC/USDT", "ETH/USDT"])
        self.assertEqual(
            engine.storage.get_json_state(engine.EXECUTION_SYMBOLS_STATE_KEY),
            ["BTC/USDT", "ETH/USDT"],
        )
        ranked = {row["symbol"]: row for row in result["ranked_candidates"]}
        self.assertEqual(set(ranked.keys()), {"BTC/USDT", "ETH/USDT"})

    def test_engine_rebuild_execution_symbols_deprioritizes_recent_negative_training(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=500):
                return [1, 2, 3, 4]

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.max_active_symbols = 4
        settings.exchange.core_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.symbols = [
            "ETH/USDT",
            "FIL/USDT",
            "POL/USDT",
            "ARB/USDT",
            "WLD/USDT",
            "AVAX/USDT",
            "SOL/USDT",
        ]
        settings.exchange.candidate_symbols = list(settings.exchange.symbols)
        settings.risk.execution_pool_target_size = 4
        settings.risk.execution_symbol_min_samples = 8
        settings.risk.execution_symbol_accuracy_floor_pct = 45.0

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        def create_training_run(
            symbol: str,
            *,
            promotion_status: str = "",
            promotion_reason: str = "",
            candidate_walkforward_summary: dict | None = None,
        ):
            model_path = model_dir / f"xgboost_v2_{symbol.replace('/', '_')}.json"
            challenger_path = model_dir / f"xgboost_challenger_{symbol.replace('/', '_')}.json"
            model_path.write_text("{}", encoding="utf-8")
            challenger_path.write_text("{}", encoding="utf-8")
            engine.storage.insert_training_run(
                {
                    "symbol": symbol,
                    "rows": settings.training.minimum_training_rows,
                    "feature_count": 10,
                    "positives": 5,
                    "negatives": 5,
                    "model_path": str(challenger_path if promotion_status else model_path),
                    "active_model_path": str(model_path),
                    "challenger_model_path": str(challenger_path if promotion_status else ""),
                    "trained_with_xgboost": True,
                    "holdout_accuracy": 0.6,
                    "candidate_holdout_accuracy": 0.6,
                    "promotion_status": promotion_status,
                    "promotion_reason": promotion_reason,
                    "candidate_walkforward_summary": candidate_walkforward_summary or {},
                }
            )

        create_training_run(
            "ETH/USDT",
            promotion_status="canary_pending",
            promotion_reason="candidate_higher_walkforward_expectancy",
            candidate_walkforward_summary={
                "total_return_pct": 1455.0,
                "profit_factor": 0.0,
                "avg_win_rate": 100.0,
            },
        )
        create_training_run(
            "FIL/USDT",
            promotion_status="rejected",
            promotion_reason="candidate_negative_walkforward_quality",
            candidate_walkforward_summary={
                "total_return_pct": 0.0,
                "profit_factor": 0.0,
                "avg_win_rate": 0.0,
            },
        )
        create_training_run(
            "POL/USDT",
            promotion_status="rejected",
            promotion_reason="candidate_negative_walkforward_quality",
            candidate_walkforward_summary={
                "total_return_pct": -2651.2326,
                "profit_factor": 0.4508,
                "avg_win_rate": 47.37,
            },
        )
        create_training_run(
            "ARB/USDT",
            promotion_status="rejected",
            promotion_reason="candidate_negative_walkforward_quality",
            candidate_walkforward_summary={
                "total_return_pct": 0.0,
                "profit_factor": 0.0,
                "avg_win_rate": 0.0,
            },
        )
        for symbol in ("WLD/USDT", "AVAX/USDT", "SOL/USDT"):
            create_training_run(symbol)

        engine.storage.set_json_state(
            engine.EXECUTION_SYMBOLS_STATE_KEY,
            ["ETH/USDT", "FIL/USDT", "POL/USDT", "ARB/USDT"],
        )
        engine.storage.set_json_state(
            engine.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {"ETH/USDT": {"symbol": "ETH/USDT", "status": "shadow"}},
        )

        edge_summary = {
            "ETH/USDT": {"count": 5, "sample_count": 5, "accuracy_pct": 20.0, "expectancy_pct": -0.1483, "profit_factor": 0.0657, "max_drawdown_pct": 0.7913, "objective_score": -4.0439},
            "FIL/USDT": {"count": 4, "sample_count": 4, "accuracy_pct": 25.0, "expectancy_pct": -0.1165, "profit_factor": 0.2098, "max_drawdown_pct": 0.5889, "objective_score": -2.3214},
            "POL/USDT": {"count": 1, "sample_count": 1, "accuracy_pct": 0.0, "expectancy_pct": -0.1173, "profit_factor": 0.0, "max_drawdown_pct": 0.1173, "objective_score": -0.7285},
            "ARB/USDT": {"count": 4, "sample_count": 4, "accuracy_pct": 50.0, "expectancy_pct": -0.1246, "profit_factor": 0.0, "max_drawdown_pct": 0.4985, "objective_score": -1.0601},
            "WLD/USDT": {"count": 3, "sample_count": 3, "accuracy_pct": 100.0, "expectancy_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0, "objective_score": -0.3047},
            "AVAX/USDT": {"count": 10, "sample_count": 10, "accuracy_pct": 50.0, "expectancy_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0, "objective_score": -1.0},
            "SOL/USDT": {"count": 16, "sample_count": 16, "accuracy_pct": 43.75, "expectancy_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0, "objective_score": -1.0234},
        }
        engine.performance.build_symbol_edge_summary = lambda limit=500: edge_summary
        engine.performance.build_symbol_accuracy_summary = lambda limit=500: edge_summary
        engine.shadow_runtime.build_observation_feedback = lambda limit=500: {
            "ETH/USDT": {"shadow_eval_count": 34, "shadow_accuracy_pct": 47.06, "shadow_trade_count": 0, "shadow_positive_ratio": 0.0, "shadow_avg_pnl_pct": 0.0},
            "FIL/USDT": {"shadow_eval_count": 34, "shadow_accuracy_pct": 64.71, "shadow_trade_count": 0, "shadow_positive_ratio": 0.0, "shadow_avg_pnl_pct": 0.0},
            "POL/USDT": {"shadow_eval_count": 32, "shadow_accuracy_pct": 50.0, "shadow_trade_count": 0, "shadow_positive_ratio": 0.0, "shadow_avg_pnl_pct": 0.0},
            "ARB/USDT": {"shadow_eval_count": 29, "shadow_accuracy_pct": 44.83, "shadow_trade_count": 4, "shadow_positive_ratio": 0.75, "shadow_avg_pnl_pct": 0.0918},
            "WLD/USDT": {"shadow_eval_count": 28, "shadow_accuracy_pct": 53.57, "shadow_trade_count": 3, "shadow_positive_ratio": 0.67, "shadow_avg_pnl_pct": -0.1185},
            "AVAX/USDT": {"shadow_eval_count": 25, "shadow_accuracy_pct": 72.0, "shadow_trade_count": 2, "shadow_positive_ratio": 0.5, "shadow_avg_pnl_pct": -0.1568},
            "SOL/USDT": {"shadow_eval_count": 16, "shadow_accuracy_pct": 81.25, "shadow_trade_count": 22, "shadow_positive_ratio": 0.55, "shadow_avg_pnl_pct": -0.1649},
        }

        result = engine.rebuild_execution_symbols(
            force=True,
            now=datetime(2026, 4, 2, tzinfo=timezone.utc),
            reason="post_retrain_refresh",
        )

        self.assertEqual(
            result["execution_symbols"],
            ["ETH/USDT", "WLD/USDT", "AVAX/USDT", "SOL/USDT"],
        )
        ranked = {row["symbol"]: row for row in result["ranked_candidates"]}
        self.assertEqual(ranked["ETH/USDT"]["status"], "candidate_shadow")
        self.assertEqual(ranked["FIL/USDT"]["status"], "training_negative")
        self.assertEqual(ranked["POL/USDT"]["status"], "training_negative")
        self.assertEqual(ranked["ARB/USDT"]["status"], "training_negative")

    def test_engine_rebuild_execution_symbols_blocks_consistency_outliers(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=500):
                return [1, 2, 3, 4]

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.max_active_symbols = 2
        settings.exchange.core_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        settings.exchange.candidate_symbols = list(settings.exchange.symbols)
        settings.risk.execution_pool_target_size = 2

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        def create_ready_model(symbol: str):
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

        for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
            create_ready_model(symbol)

        edge_summary = {
            "BTC/USDT": {"count": 12, "sample_count": 12, "accuracy_pct": 62.0, "expectancy_pct": 0.12, "profit_factor": 1.20, "max_drawdown_pct": 1.8, "objective_score": 1.1},
            "ETH/USDT": {"count": 10, "sample_count": 10, "accuracy_pct": 58.0, "expectancy_pct": 0.08, "profit_factor": 1.05, "max_drawdown_pct": 2.0, "objective_score": 0.7},
            "SOL/USDT": {"count": 16, "sample_count": 16, "accuracy_pct": 75.0, "expectancy_pct": 0.18, "profit_factor": 1.40, "max_drawdown_pct": 1.4, "objective_score": 2.1},
        }
        engine.performance.build_symbol_accuracy_summary = lambda limit=500: edge_summary
        engine.performance.build_symbol_edge_summary = lambda limit=500: edge_summary
        engine.shadow_runtime.build_observation_feedback = lambda limit=500: {}
        engine.backtest_live_consistency.build_symbol_consistency_rows = (
            lambda symbols, walkforward_overrides=None: {
                "BTC/USDT": {"flags": []},
                "ETH/USDT": {"flags": []},
                "SOL/USDT": {"flags": ["walkforward_live_pnl_divergence"]},
            }
        )

        result = engine.rebuild_execution_symbols(
            force=True,
            now=datetime(2026, 4, 2, tzinfo=timezone.utc),
            reason="consistency_gate",
        )

        self.assertEqual(result["execution_symbols"], ["BTC/USDT", "ETH/USDT"])
        ranked = {row["symbol"]: row for row in result["ranked_candidates"]}
        self.assertEqual(ranked["SOL/USDT"]["status"], "consistency_blocked")
        self.assertEqual(
            ranked["SOL/USDT"]["consistency_flags"],
            ["walkforward_live_pnl_divergence"],
        )
        self.assertEqual(
            engine.storage.get_json_state("consistency_blocked_symbols", []),
            ["SOL/USDT"],
        )
        self.assertEqual(
            engine.storage.get_json_state("consistency_blocked_details", {}),
            {"SOL/USDT": ["walkforward_live_pnl_divergence"]},
        )

    def test_settings_support_prefixed_live_env_vars(self):
        with patch.dict(
            os.environ,
            {
                "APP_RUNTIME_MODE": "live",
                "APP_ALLOW_LIVE_ORDERS": "true",
                "APP_LOG_LEVEL": "DEBUG",
            },
            clear=False,
        ):
            settings = AppSettingsModel()
        self.assertEqual(settings.app.runtime_mode, "live")
        self.assertTrue(settings.app.allow_live_orders)
        self.assertEqual(settings.app.log_level, "DEBUG")

    def test_low_resource_mode_tightens_training_defaults(self):
        with patch.dict(
            os.environ,
            {
                "APP_LOW_RESOURCE_MODE": "true",
            },
            clear=False,
        ):
            settings = AppSettingsModel()
        self.assertTrue(settings.app.low_resource_mode)
        self.assertEqual(settings.model.xgboost_nthread, 1)
        self.assertEqual(settings.model.xgboost_num_boost_round, 120)
        self.assertEqual(settings.training.dataset_limit_1h, 1200)
        self.assertEqual(settings.training.dataset_limit_4h, 720)
        self.assertEqual(settings.training.dataset_limit_1d, 240)
        self.assertEqual(settings.scheduler.walkforward_cron_hours, 0)

    def test_prefixed_env_vars_for_ab_testing_and_training_are_loaded(self):
        with patch.dict(
            os.environ,
            {
                "AB_TESTING_ENABLED": "true",
                "AB_TESTING_CHALLENGER_ALLOCATION_PCT": "0.25",
                "TRAINING_DATASET_LIMIT_1H": "888",
                "TRAINING_DATASET_LIMIT_4H": "555",
                "TRAINING_DATASET_LIMIT_1D": "222",
            },
            clear=False,
        ):
            settings = AppSettingsModel()
        self.assertTrue(settings.ab_testing.enabled)
        self.assertEqual(settings.ab_testing.challenger_allocation_pct, 0.25)
        self.assertEqual(settings.training.dataset_limit_1h, 888)
        self.assertEqual(settings.training.dataset_limit_4h, 555)
        self.assertEqual(settings.training.dataset_limit_1d, 222)

    def test_ab_testing_settings_ignore_generic_enabled_env(self):
        with patch.dict(
            os.environ,
            {
                "ENABLED": "true",
                "CHALLENGER_MODEL_PATH": "bad.json",
                "CHALLENGER_ALLOCATION_PCT": "0.9",
                "EXECUTE_CHALLENGER_LIVE": "true",
            },
            clear=False,
        ):
            settings = AppSettingsModel()
        self.assertFalse(settings.ab_testing.enabled)
        self.assertEqual(
            settings.ab_testing.challenger_model_path,
            "data/models/xgboost_challenger.json",
        )
        self.assertEqual(settings.ab_testing.challenger_allocation_pct, 0.10)
        self.assertFalse(settings.ab_testing.execute_challenger_live)

    def test_main_configure_runtime_environment_sets_thread_limits(self):
        with patch.dict(
            os.environ,
            {
                "APP_LOW_RESOURCE_MODE": "true",
            },
            clear=False,
        ):
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ):
                os.environ.pop(key, None)
            reloaded_main = importlib.reload(main_module)
            enabled = reloaded_main.configure_runtime_environment()
            self.assertTrue(enabled)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")
            self.assertEqual(os.environ["NUMEXPR_NUM_THREADS"], "1")
            self.assertEqual(os.environ["VECLIB_MAXIMUM_THREADS"], "1")

    def test_dashboard_scheduler_job_options_hide_heavy_jobs_in_low_resource_mode(self):
        normal = scheduler_job_options(False)
        low_resource = scheduler_job_options(True)
        self.assertIn("train", normal)
        self.assertIn("walkforward", normal)
        self.assertIn("abtest", normal)
        self.assertIn("drift", normal)
        self.assertNotIn("train", low_resource)
        self.assertNotIn("walkforward", low_resource)
        self.assertNotIn("abtest", low_resource)
        self.assertNotIn("drift", low_resource)
        self.assertIn("once", low_resource)
        self.assertIn("reconcile", low_resource)

    def test_dashboard_db_path_resolves_relative_to_project_root(self):
        fake_settings = SimpleNamespace(
            app=SimpleNamespace(
                db_path="data/cryptoai.db",
                project_root=Path("/srv/cryptoai"),
            )
        )
        with patch.object(dashboard_module, "get_settings", return_value=fake_settings):
            resolved = dashboard_db_path()
        self.assertEqual(resolved, str(Path("/srv/cryptoai") / "data" / "cryptoai.db"))

    def test_dashboard_command_timeout_seconds_expands_for_heavy_jobs(self):
        self.assertEqual(command_timeout_seconds("train"), 3600)
        self.assertEqual(command_timeout_seconds("walkforward"), 3600)
        self.assertEqual(command_timeout_seconds("backfill"), 3600)
        self.assertEqual(command_timeout_seconds("backtest"), 3600)
        self.assertEqual(command_timeout_seconds("health"), 600)
        self.assertEqual(command_timeout_seconds("unknown"), 300)

    def test_exchange_adapters_skip_empty_proxy_configuration(self):
        class FakeOKX:
            def __init__(self, params):
                self.params = params

        class FakeBinance:
            def __init__(self, params):
                self.params = params

        with patch("execution.exchange_adapter.ccxt.okx", FakeOKX), patch(
            "execution.exchange_adapter.ccxt.binance", FakeBinance
        ):
            okx = OKXExchangeAdapter(proxy_url="")
            binance = BinanceExchangeAdapter(proxy_url="")
        self.assertNotIn("proxies", okx.exchange.params)
        self.assertNotIn("proxies", binance.exchange.params)

    def test_okx_market_data_falls_back_to_spot_for_swap_symbol(self):
        class SwapExchange:
            def fetch_ohlcv(self, symbol, timeframe, since=None, limit=300):
                raise ccxt.NetworkError("swap unavailable")

            def fetch_ticker(self, symbol):
                raise ccxt.NetworkError("swap ticker unavailable")

            def fetch_funding_rate(self, symbol):
                return {"fundingRate": 0.01}

            def load_markets(self):
                self.symbols = ["ETH/USDT:USDT"]

        class SpotExchange:
            def __init__(self):
                self.calls = 0

            def fetch_ohlcv(self, symbol, timeframe, since=None, limit=300):
                self.calls += 1
                self.last_call = (symbol, timeframe, since, limit)
                if self.calls > 1:
                    return []
                return [
                    [1700000000000, 100, 101, 99, 100.5, 1000],
                    [1700003600000, 100.5, 102, 100, 101, 1200],
                ]

            def fetch_ticker(self, symbol):
                return {"last": 101.5, "timestamp": 1700003600000}

        collector = OKXMarketDataCollector(
            self.storage,
            spot_exchange=SpotExchange(),
            swap_exchange=SwapExchange(),
        )
        candles = collector.fetch_historical_ohlcv("ETH/USDT:USDT", "1d", limit=2)
        self.assertEqual(len(candles), 2)
        self.assertAlmostEqual(candles[-1]["close"], 101.0, places=6)
        self.assertAlmostEqual(collector.fetch_latest_price("ETH/USDT:USDT"), 101.5, places=6)
        latency = collector.measure_latency("ETH/USDT:USDT")
        self.assertEqual(latency["status"], "ok")
        with self.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM ohlcv WHERE symbol=? AND timeframe=?",
                ("ETH/USDT:USDT", "1d"),
            ).fetchone()["c"]
        self.assertEqual(count, 2)

    def test_engine_generate_reports_persists_artifacts(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        engine = engine_module.CryptoAIV2Engine(settings)
        reports = engine.generate_reports()
        self.assertIn("daily", reports)
        self.assertIn("weekly", reports)
        self.assertIn("daily_focus", reports)
        self.assertIn("backtest_live_consistency", reports)
        with engine.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM report_artifacts WHERE report_type IN ('daily', 'weekly')"
            ).fetchone()["c"]
            extra_count = conn.execute(
                "SELECT COUNT(*) AS c FROM report_artifacts "
                "WHERE report_type IN ('daily_focus', 'backtest_live_consistency')"
            ).fetchone()["c"]
        self.assertEqual(count, 2)
        self.assertEqual(extra_count, 2)

    def test_okx_market_data_uses_lowercase_ccxt_timeframe_for_since_backfill(self):
        class SpotExchange:
            def __init__(self):
                self.calls = []

            def fetch_ohlcv(self, symbol, timeframe, since=None, limit=300):
                self.calls.append((symbol, timeframe, since, limit))
                if len(self.calls) > 1:
                    return []
                return [
                    [1700000000000, 100, 101, 99, 100.5, 1000],
                    [1700003600000, 100.5, 102, 100, 101, 1200],
                ]

            def fetch_ticker(self, symbol):
                return {"last": 101.5, "timestamp": 1700003600000}

        collector = OKXMarketDataCollector(
            self.storage,
            spot_exchange=SpotExchange(),
            swap_exchange=SpotExchange(),
        )
        collector.fetch_historical_ohlcv(
            "ETH/USDT",
            "1h",
            since=1700000000000,
            limit=2,
        )
        self.assertEqual(collector.spot_exchange.calls[0][1], "1h")

    def test_okx_market_data_caps_since_backfill_page_size_and_paginates(self):
        class SpotExchange:
            def __init__(self):
                self.calls = []

            def fetch_ohlcv(self, symbol, timeframe, since=None, limit=300):
                self.calls.append((symbol, timeframe, since, limit))
                base = 1700000000000
                if len(self.calls) == 1:
                    return [
                        [base + idx * 3600000, 100, 101, 99, 100.5, 1000]
                        for idx in range(300)
                    ]
                if len(self.calls) == 2:
                    start = base + 300 * 3600000
                    return [
                        [start + idx * 3600000, 100, 101, 99, 100.5, 1000]
                        for idx in range(2)
                    ]
                return []

            def fetch_ticker(self, symbol):
                return {"last": 101.5, "timestamp": 1700003600000}

        collector = OKXMarketDataCollector(
            self.storage,
            spot_exchange=SpotExchange(),
            swap_exchange=SpotExchange(),
        )
        candles = collector.fetch_historical_ohlcv(
            "ETH/USDT",
            "1h",
            since=1700000000000,
            limit=500,
        )
        self.assertEqual(len(candles), 302)
        self.assertEqual(collector.spot_exchange.calls[0][3], 300)
        self.assertEqual(collector.spot_exchange.calls[1][3], 300)
        self.assertEqual(
            collector.spot_exchange.calls[1][2],
            1700000000000 + 299 * 3600000 + 1,
        )

    def test_binance_market_data_normalizes_swap_symbols_for_read_calls(self):
        class FakeExchange:
            def __init__(self):
                self.last_ohlcv = None
                self.last_ticker = None
                self.last_order_book = None
                self.ohlcv_calls = 0

            def fetch_ohlcv(self, symbol, timeframe, since=None, limit=300):
                self.ohlcv_calls += 1
                self.last_ohlcv = (symbol, timeframe, since, limit)
                if self.ohlcv_calls > 1:
                    return []
                return [[1700000000000, 100, 101, 99, 100.5, 1000]]

            def fetch_ticker(self, symbol):
                self.last_ticker = symbol
                return {"last": 100.5, "timestamp": 1700000000000}

            def fetch_order_book(self, symbol, limit=20):
                self.last_order_book = (symbol, limit)
                return {
                    "bids": [[100.0, 2.0]],
                    "asks": [[100.2, 2.0]],
                }

        from core.binance_market_data import BinanceMarketDataCollector

        collector = BinanceMarketDataCollector(
            self.storage,
            exchange=FakeExchange(),
        )
        candles = collector.fetch_historical_ohlcv("ETH/USDT:USDT", "1h", limit=1)
        price = collector.fetch_latest_price("ETH/USDT:USDT")
        depth = collector.summarize_order_book_depth("ETH/USDT:USDT", depth=1)

        self.assertEqual(len(candles), 1)
        self.assertEqual(price, 100.5)
        self.assertEqual(depth.symbol, "ETH/USDT:USDT")
        self.assertEqual(collector.exchange.last_ohlcv[0], "ETH/USDT")
        self.assertEqual(collector.exchange.last_ticker, "ETH/USDT")
        self.assertEqual(collector.exchange.last_order_book[0], "ETH/USDT")

    def test_failover_market_data_uses_secondary_provider_and_records_stats(self):
        class PrimaryMarket:
            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                raise RuntimeError("primary_down")

        class SecondaryMarket:
            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=300):
                return make_candles(3, 100.0)

        collector = FailoverMarketDataCollector(
            self.storage,
            primary_provider="okx",
            primary_collector=PrimaryMarket(),
            secondary_provider="binance",
            secondary_factory=lambda: SecondaryMarket(),
        )

        candles = collector.fetch_historical_ohlcv("BTC/USDT:USDT", "1h", limit=3)

        self.assertEqual(len(candles), 3)
        route = self.storage.get_json_state("market_data_last_route", {})
        self.assertEqual(route["selected_provider"], "binance")
        self.assertTrue(route["fallback_used"])
        stats = self.storage.get_json_state("market_data_failover_stats", {})
        self.assertEqual(stats["fetch_historical_ohlcv"]["binance_selected"], 1)
        with self.storage._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='market_data_failover'"
            ).fetchone()["c"]
        self.assertEqual(count, 1)

    def test_execution_pool_does_not_block_sparse_extreme_walkforward_warning(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_historical_ohlcv(self, symbol, timeframe, since=None, limit=500):
                return [1, 2, 3, 4]

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.max_active_symbols = 2
        settings.exchange.core_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.candidate_symbols = list(settings.exchange.symbols)
        settings.risk.execution_pool_target_size = 2

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        def create_ready_model(symbol: str):
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

        for symbol in ("BTC/USDT", "ETH/USDT"):
            create_ready_model(symbol)

        edge_summary = {
            "BTC/USDT": {
                "count": 12,
                "sample_count": 12,
                "accuracy_pct": 62.0,
                "expectancy_pct": 0.12,
                "profit_factor": 1.20,
                "max_drawdown_pct": 1.8,
                "objective_score": 1.1,
            },
            "ETH/USDT": {
                "count": 10,
                "sample_count": 10,
                "accuracy_pct": 58.0,
                "expectancy_pct": 0.08,
                "profit_factor": 1.05,
                "max_drawdown_pct": 2.0,
                "objective_score": 0.7,
            },
        }
        engine.performance.build_symbol_accuracy_summary = lambda limit=500: edge_summary
        engine.performance.build_symbol_edge_summary = lambda limit=500: edge_summary
        engine.shadow_runtime.build_observation_feedback = lambda limit=500: {}
        engine.backtest_live_consistency.build_symbol_consistency_rows = (
            lambda symbols, walkforward_overrides=None: {
                "BTC/USDT": {"flags": []},
                "ETH/USDT": {"flags": ["walkforward_return_extreme_sparse"]},
            }
        )

        result = engine.rebuild_execution_symbols(
            force=True,
            now=datetime(2026, 4, 2, tzinfo=timezone.utc),
            reason="sparse_consistency_warning",
        )

        self.assertEqual(result["execution_symbols"], ["BTC/USDT", "ETH/USDT"])
        ranked = {row["symbol"]: row for row in result["ranked_candidates"]}
        self.assertNotEqual(ranked["ETH/USDT"]["status"], "consistency_blocked")
        self.assertEqual(
            ranked["ETH/USDT"]["consistency_flags"],
            ["walkforward_return_extreme_sparse"],
        )
        self.assertEqual(
            ranked["ETH/USDT"]["blocking_consistency_flags"],
            [],
        )
        self.assertEqual(
            engine.storage.get_json_state("consistency_blocked_symbols", []),
            [],
        )
