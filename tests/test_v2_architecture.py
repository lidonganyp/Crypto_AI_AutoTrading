import ccxt
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dashboard as dashboard_module
import pandas as pd
from analysis.dynamic_watchlist import DynamicWatchlistService
from config import get_settings
from core.feature_pipeline import FeatureInput, FeaturePipeline
from core.okx_market_data import OKXMarketDataCollector
from core.models import (
    MarketRegime,
    PredictionResult,
    ResearchInsight,
    SignalDirection,
    SuggestedAction,
)
from strategy.decision_engine import DecisionEngine
from strategy.risk_manager import RiskManager
from dashboard import (
    parse_symbol_text,
)
from tests.v2_architecture_support import V2ArchitectureTestCase, make_candles


class V2ArchitectureTests(V2ArchitectureTestCase):

    @staticmethod
    def _prices_from_returns(start: float, returns: list[float]) -> list[float]:
        prices = [start]
        for change in returns:
            prices.append(prices[-1] * (1.0 + change))
        return prices

    def test_feature_pipeline_builds_valid_snapshot(self):
        pipeline = FeaturePipeline()
        snapshot = pipeline.build(
            FeatureInput(
                symbol="BTC/USDT",
                candles_1h=make_candles(240, 100),
                candles_4h=make_candles(240, 120),
                candles_1d=make_candles(240, 140),
                funding_rate=0.01,
                depth_imbalance=0.12,
                large_order_net_notional=50000.0,
                sentiment_value=0.3,
                llm_sentiment_score=0.2,
                lunarcrush_sentiment=0.15,
                market_regime_score=1.0,
            )
        )
        self.assertTrue(snapshot.valid)
        self.assertIn("rsi_1h", snapshot.values)
        self.assertIn("ma200_1d", snapshot.values)
        self.assertIn("adx_4h", snapshot.values)
        self.assertIn("obv_4h", snapshot.values)
        self.assertIn("fibonacci_position_1d", snapshot.values)
        self.assertIn("llm_sentiment_score", snapshot.values)
        self.assertIn("lunarcrush_sentiment", snapshot.values)
        self.assertIn("funding_rate", snapshot.values)
        self.assertIn("bid_ask_spread_pct", snapshot.values)
        self.assertIn("top5_depth_notional", snapshot.values)
        self.assertIn("depth_imbalance", snapshot.values)
        self.assertIn("large_order_net_notional", snapshot.values)
        self.assertGreaterEqual(len(snapshot.values), 45)
        self.assertGreater(snapshot.values["volume_ratio_1h"], 0)

    def test_exchange_defaults_focus_on_two_core_symbols(self):
        self.assertEqual(get_settings().exchange.symbols, ["BTC/USDT", "ETH/USDT"])

    def test_dynamic_watchlist_selects_ten_symbols_and_keeps_core(self):
        settings = get_settings().model_copy(deep=True)
        settings.exchange.max_active_symbols = 10
        settings.exchange.dynamic_watchlist_enabled = True
        settings.exchange.symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "AVAX/USDT",
            "POL/USDT",
            "ARB/USDT",
            "UNI/USDT",
            "AAVE/USDT",
            "WLD/USDT",
            "RENDER/USDT",
        ]
        settings.exchange.candidate_symbols = settings.exchange.symbols + [
            "LINK/USDT",
            "DOGE/USDT",
        ]

        class FakeExchange:
            def fetch_ticker(self, symbol):
                base = symbol.replace(":USDT", "")
                volume_map = {
                    "BTC/USDT": 500_000_000,
                    "ETH/USDT": 300_000_000,
                    "SOL/USDT": 120_000_000,
                    "AVAX/USDT": 90_000_000,
                    "POL/USDT": 80_000_000,
                    "ARB/USDT": 70_000_000,
                    "UNI/USDT": 65_000_000,
                    "AAVE/USDT": 60_000_000,
                    "WLD/USDT": 55_000_000,
                    "RENDER/USDT": 52_000_000,
                    "LINK/USDT": 45_000_000,
                    "DOGE/USDT": 40_000_000,
                }
                change_map = {
                    "BTC/USDT": 2.0,
                    "ETH/USDT": 1.0,
                    "SOL/USDT": 6.0,
                    "AVAX/USDT": 4.0,
                    "POL/USDT": 3.0,
                    "ARB/USDT": 3.5,
                    "UNI/USDT": 2.5,
                    "AAVE/USDT": 2.2,
                    "WLD/USDT": 4.5,
                    "RENDER/USDT": 5.0,
                    "LINK/USDT": 1.5,
                    "DOGE/USDT": 0.5,
                }
                return {
                    "quoteVolume": volume_map.get(base, 0.0),
                    "percentage": change_map.get(base, 0.0),
                    "last": 100.0,
                }

        fake_market = SimpleNamespace(
            exchange=FakeExchange(),
            fetch_available_instruments=lambda: settings.exchange.candidate_symbols,
        )
        watchlist = DynamicWatchlistService(self.storage, settings, fake_market)
        snapshot = watchlist.refresh(force=True)
        self.assertEqual(len(snapshot.active_symbols), 10)
        self.assertIn("BTC/USDT", snapshot.active_symbols)
        self.assertIn("ETH/USDT", snapshot.active_symbols)

    def test_dynamic_watchlist_blocks_symbols_outside_exchange_allowlist(self):
        settings = get_settings().model_copy(deep=True)
        settings.exchange.max_active_symbols = 4
        settings.exchange.dynamic_watchlist_enabled = True
        settings.exchange.symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.candidate_symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "FIL/USDT",
            "SOL/USDT",
        ]

        class FakeExchange:
            def fetch_ticker(self, symbol):
                base = symbol.replace(":USDT", "")
                volume_map = {
                    "BTC/USDT": 500_000_000,
                    "ETH/USDT": 300_000_000,
                    "FIL/USDT": 220_000_000,
                    "SOL/USDT": 210_000_000,
                }
                change_map = {
                    "BTC/USDT": 1.0,
                    "ETH/USDT": 0.8,
                    "FIL/USDT": 8.0,
                    "SOL/USDT": 7.0,
                }
                return {
                    "quoteVolume": volume_map.get(base, 0.0),
                    "percentage": change_map.get(base, 0.0),
                    "last": 100.0,
                }

        fake_market = SimpleNamespace(
            exchange=FakeExchange(),
            fetch_available_instruments=lambda: settings.exchange.candidate_symbols,
        )
        watchlist = DynamicWatchlistService(self.storage, settings, fake_market)
        watchlist.set_manual_lists(["FIL/USDT"], [])
        snapshot = watchlist.refresh(force=True)

        self.assertEqual(snapshot.active_symbols, ["BTC/USDT", "ETH/USDT"])
        self.assertEqual(snapshot.whitelist, [])
        self.assertEqual(
            [candidate.symbol for candidate in snapshot.candidates],
            ["BTC/USDT", "ETH/USDT"],
        )
        self.assertEqual(self.storage.get_json_state("active_symbols"), snapshot.active_symbols)

    def test_dynamic_watchlist_respects_manual_lists_and_sector_caps(self):
        settings = get_settings().model_copy(deep=True)
        settings.exchange.max_active_symbols = 6
        settings.exchange.max_symbols_per_sector = 1
        settings.exchange.core_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "AVAX/USDT",
            "ARB/USDT",
            "OP/USDT",
            "UNI/USDT",
            "AAVE/USDT",
            "WLD/USDT",
            "RENDER/USDT",
        ]
        settings.exchange.candidate_symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "AVAX/USDT",
            "ARB/USDT",
            "OP/USDT",
            "UNI/USDT",
            "AAVE/USDT",
            "WLD/USDT",
            "RENDER/USDT",
        ]

        class FakeExchange:
            def fetch_ticker(self, symbol):
                base = symbol.replace(":USDT", "")
                ranking = {
                    "BTC/USDT": (500_000_000, 1.0),
                    "ETH/USDT": (300_000_000, 0.5),
                    "SOL/USDT": (120_000_000, 6.0),
                    "AVAX/USDT": (110_000_000, 5.0),
                    "ARB/USDT": (90_000_000, 4.0),
                    "OP/USDT": (88_000_000, 4.2),
                    "UNI/USDT": (85_000_000, 3.0),
                    "AAVE/USDT": (80_000_000, 2.8),
                    "WLD/USDT": (95_000_000, 3.5),
                    "RENDER/USDT": (94_000_000, 3.8),
                }
                volume, change = ranking.get(base, (0.0, 0.0))
                return {"quoteVolume": volume, "percentage": change, "last": 100.0}

        fake_market = SimpleNamespace(
            exchange=FakeExchange(),
            fetch_available_instruments=lambda: settings.exchange.candidate_symbols,
        )
        watchlist = DynamicWatchlistService(self.storage, settings, fake_market)
        watchlist.set_manual_lists(["AAVE/USDT"], ["AVAX/USDT"])
        snapshot = watchlist.refresh(force=True)
        self.assertIn("AAVE/USDT", snapshot.active_symbols)
        self.assertNotIn("AVAX/USDT", snapshot.active_symbols)
        self.assertEqual(len([s for s in snapshot.active_symbols if s in {"SOL/USDT", "AVAX/USDT"}]), 1)
        self.assertEqual(len([s for s in snapshot.active_symbols if s in {"ARB/USDT", "OP/USDT"}]), 1)

    def test_engine_active_symbols_filter_to_model_ready_set(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.symbols = ["BTC/USDT", "ETH/USDT"]
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        btc_model = model_dir / "xgboost_v2_BTC_USDT.json"
        btc_model.write_text("{}", encoding="utf-8")

        engine = engine_module.CryptoAIV2Engine(settings)
        engine.watchlist = SimpleNamespace(
            refresh=lambda force=False, now=None: SimpleNamespace(
                active_symbols=["BTC/USDT", "ETH/USDT"],
                added_symbols=[],
                removed_symbols=[],
                whitelist=[],
                blacklist=[],
                candidates=[],
                refreshed_at="",
                refresh_reason="",
            )
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
        engine.storage.insert_training_run(
            {
                "symbol": "ETH/USDT",
                "rows": settings.training.minimum_training_rows - 1,
                "feature_count": 10,
                "positives": 5,
                "negatives": 5,
                "model_path": str(model_dir / "xgboost_v2_ETH_USDT.json"),
                "trained_with_xgboost": False,
                "holdout_accuracy": 0.0,
            }
        )

        self.assertEqual(engine.get_active_symbols(), ["BTC/USDT"])

    def test_engine_active_symbols_filter_low_edge_symbols(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.symbols = ["BTC/USDT", "ETH/USDT"]
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for symbol in settings.exchange.symbols:
            model_path = model_dir / f"xgboost_v2_{symbol.replace('/', '_')}.json"
            model_path.write_text("{}", encoding="utf-8")

        engine = engine_module.CryptoAIV2Engine(settings)
        engine.watchlist = SimpleNamespace(
            refresh=lambda force=False, now=None: SimpleNamespace(
                active_symbols=["BTC/USDT", "ETH/USDT"],
                added_symbols=[],
                removed_symbols=[],
                whitelist=[],
                blacklist=[],
                candidates=[],
                refreshed_at="",
                refresh_reason="",
            )
        )
        for symbol in settings.exchange.symbols:
            engine.storage.insert_training_run(
                {
                    "symbol": symbol,
                    "rows": settings.training.minimum_training_rows,
                    "feature_count": 10,
                    "positives": 5,
                    "negatives": 5,
                    "model_path": str(model_dir / f"xgboost_v2_{symbol.replace('/', '_')}.json"),
                    "trained_with_xgboost": True,
                    "holdout_accuracy": 0.6,
                }
            )
        engine.performance = SimpleNamespace(
            build_symbol_accuracy_summary=lambda limit=500: {
                "BTC/USDT": {"count": 12, "correct": 7, "accuracy_pct": 58.33},
                "ETH/USDT": {"count": 12, "correct": 4, "accuracy_pct": 33.33},
            },
            build_symbol_edge_summary=lambda limit=500: {
                "BTC/USDT": {
                    "count": 12,
                    "sample_count": 12,
                    "accuracy_pct": 58.33,
                    "expectancy_pct": 0.12,
                    "profit_factor": 1.15,
                    "max_drawdown_pct": 2.0,
                    "objective_score": 0.9,
                },
                "ETH/USDT": {
                    "count": 12,
                    "sample_count": 12,
                    "accuracy_pct": 33.33,
                    "expectancy_pct": -0.20,
                    "profit_factor": 0.75,
                    "max_drawdown_pct": 6.5,
                    "objective_score": -1.3,
                },
            },
        )
        self.assertEqual(engine.get_active_symbols(), ["BTC/USDT"])

    def test_engine_active_symbols_keep_fast_alpha_core_symbols_in_paper_mode(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "paper"
        settings.exchange.symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.core_symbols = ["BTC/USDT", "ETH/USDT"]
        settings.exchange.max_active_symbols = 2
        settings.strategy.fast_alpha_enabled = True
        settings.strategy.fast_alpha_symbols = ["BTC/USDT", "ETH/USDT"]
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for symbol in settings.exchange.symbols:
            model_path = model_dir / f"xgboost_v2_{symbol.replace('/', '_')}.json"
            model_path.write_text("{}", encoding="utf-8")

        engine = engine_module.CryptoAIV2Engine(settings)
        engine.watchlist = SimpleNamespace(
            refresh=lambda force=False, now=None: SimpleNamespace(
                active_symbols=["BTC/USDT", "ETH/USDT"],
                added_symbols=[],
                removed_symbols=[],
                whitelist=[],
                blacklist=[],
                candidates=[],
                refreshed_at="",
                refresh_reason="",
            )
        )
        for symbol in settings.exchange.symbols:
            engine.storage.insert_training_run(
                {
                    "symbol": symbol,
                    "rows": settings.training.minimum_training_rows,
                    "feature_count": 10,
                    "positives": 5,
                    "negatives": 5,
                    "model_path": str(model_dir / f"xgboost_v2_{symbol.replace('/', '_')}.json"),
                    "trained_with_xgboost": True,
                    "holdout_accuracy": 0.6,
                }
            )
        engine.performance = SimpleNamespace(
            build=lambda: SimpleNamespace(total_closed_trades=10),
            build_symbol_accuracy_summary=lambda limit=500: {
                "BTC/USDT": {"count": 12, "correct": 5, "accuracy_pct": 41.67},
                "ETH/USDT": {"count": 12, "correct": 5, "accuracy_pct": 41.67},
            },
            build_symbol_edge_summary=lambda limit=500: {
                "BTC/USDT": {
                    "count": 12,
                    "sample_count": 12,
                    "accuracy_pct": 41.67,
                    "expectancy_pct": -0.12,
                    "profit_factor": 0.70,
                    "max_drawdown_pct": 2.0,
                    "objective_score": -1.2,
                },
                "ETH/USDT": {
                    "count": 12,
                    "sample_count": 12,
                    "accuracy_pct": 41.67,
                    "expectancy_pct": -0.10,
                    "profit_factor": 0.75,
                    "max_drawdown_pct": 2.0,
                    "objective_score": -1.0,
                },
            },
        )

        self.assertEqual(engine.get_execution_symbols(), ["BTC/USDT", "ETH/USDT"])
        self.assertEqual(engine.get_active_symbols(), [])

    def test_engine_active_symbols_add_core_exploration_slot_in_paper_mode(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.symbols = ["BTC/USDT", "AVAX/USDT"]
        settings.exchange.core_symbols = ["BTC/USDT"]
        model_dir = Path(self.db_path).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for symbol in settings.exchange.symbols:
            model_path = model_dir / f"xgboost_v2_{symbol.replace('/', '_')}.json"
            model_path.write_text("{}", encoding="utf-8")

        engine = engine_module.CryptoAIV2Engine(settings)
        engine.watchlist = SimpleNamespace(
            refresh=lambda force=False, now=None: SimpleNamespace(
                active_symbols=["BTC/USDT", "AVAX/USDT"],
                added_symbols=[],
                removed_symbols=[],
                whitelist=[],
                blacklist=[],
                candidates=[],
                refreshed_at="",
                refresh_reason="",
            )
        )
        for symbol in settings.exchange.symbols:
            engine.storage.insert_training_run(
                {
                    "symbol": symbol,
                    "rows": settings.training.minimum_training_rows,
                    "feature_count": 10,
                    "positives": 5,
                    "negatives": 5,
                    "model_path": str(model_dir / f"xgboost_v2_{symbol.replace('/', '_')}.json"),
                    "trained_with_xgboost": True,
                    "holdout_accuracy": 0.6,
                }
            )
        engine.performance = SimpleNamespace(
            build=lambda: SimpleNamespace(total_closed_trades=0),
            build_symbol_accuracy_summary=lambda limit=500: {
                "BTC/USDT": {"count": 12, "correct": 4, "accuracy_pct": 33.33},
                "AVAX/USDT": {"count": 12, "correct": 7, "accuracy_pct": 58.33},
            },
            build_symbol_edge_summary=lambda limit=500: {
                "BTC/USDT": {
                    "count": 12,
                    "sample_count": 12,
                    "accuracy_pct": 33.33,
                    "expectancy_pct": -0.10,
                    "profit_factor": 0.80,
                    "max_drawdown_pct": 5.0,
                    "objective_score": -0.6,
                },
                "AVAX/USDT": {
                    "count": 12,
                    "sample_count": 12,
                    "accuracy_pct": 58.33,
                    "expectancy_pct": 0.12,
                    "profit_factor": 1.10,
                    "max_drawdown_pct": 2.0,
                    "objective_score": 0.8,
                },
            },
        )

        self.assertEqual(engine.get_active_symbols(), ["AVAX/USDT", "BTC/USDT"])

    def test_dashboard_parse_symbol_text_normalizes_inputs(self):
        parsed = parse_symbol_text("btcusdt, eth-usdt\nSOL/USDT, BTC/USDT")
        self.assertEqual(parsed, ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

    def test_dashboard_parse_markdown_metrics_supports_chinese_labels(self):
        metrics = dashboard_module.parse_markdown_metrics(
            "\n".join(
                [
                    "# 性能报告",
                    "- 胜率: 40.00%",
                    "- 最近预测窗口样本: 6/91",
                    "- 扩展预测窗口样本: 134/245",
                    "- XGBoost 方向准确率: 48.00%",
                    "- 扩展窗口 XGBoost 方向准确率: 61.19%",
                    "- 执行闭环准确率: 51.00% (100)",
                    "- Shadow 观察准确率: 56.19% (226)",
                    "- 融合信号准确率: 45.00%",
                    "- 最近 Holdout 准确率: 70.00%",
                    "- Short-horizon 放行开仓数: 3",
                    "- Short-horizon 放行净收益: $+12.50",
                    "- Short-horizon 放行平仓数: 2",
                    "- Short-horizon 负期望暂停次数: 4",
                    "- Short-horizon 当前状态: warming_up",
                    "- Short-horizon 最近样本: 3/6 (lookback 20)",
                    "- Short-horizon 最近净期望: +0.12%",
                    "- Short-horizon 最近净盈亏比: 1.10",
                    "- Short-horizon 放行后状态: disabled_negative_edge",
                    "- Short-horizon 放行后净期望: -0.18%",
                    "- Short-horizon 放行后净盈亏比: 0.71",
                ]
            )
        )
        self.assertEqual(metrics["Win Rate"], "40.00%")
        self.assertEqual(metrics["Recent Prediction Window"], "6/91")
        self.assertEqual(metrics["Expanded Prediction Window"], "134/245")
        self.assertEqual(metrics["XGBoost Direction Accuracy"], "48.00%")
        self.assertEqual(metrics["Expanded XGBoost Direction Accuracy"], "61.19%")
        self.assertEqual(metrics["Execution Accuracy"], "51.00% (100)")
        self.assertEqual(metrics["Shadow Accuracy"], "56.19% (226)")
        self.assertEqual(metrics["Fusion Signal Accuracy"], "45.00%")
        self.assertEqual(metrics["Latest Holdout Accuracy"], "70.00%")
        self.assertEqual(metrics["Short-horizon Softened Opens"], "3")
        self.assertEqual(metrics["Short-horizon Softened Net PnL"], "$+12.50")
        self.assertEqual(metrics["Short-horizon Softened Closed Trades"], "2")
        self.assertEqual(metrics["Short-horizon Negative-Edge Pauses"], "4")
        self.assertEqual(metrics["Short-Horizon Status"], "warming_up")
        self.assertEqual(metrics["Short-Horizon Recent Samples"], "3/6 (lookback 20)")
        self.assertEqual(metrics["Short-Horizon Recent Expectancy"], "+0.12%")
        self.assertEqual(metrics["Short-Horizon Recent Profit Factor"], "1.10")
        self.assertEqual(metrics["Short-Horizon Softened Status"], "disabled_negative_edge")
        self.assertEqual(metrics["Short-Horizon Softened Expectancy"], "-0.18%")
        self.assertEqual(metrics["Short-Horizon Softened Profit Factor"], "0.71")

    def test_dashboard_display_df_translates_values_in_chinese_mode(self):
        df = pd.DataFrame(
            [
                {
                    "status": "ok",
                    "event_type": "research_review",
                    "evaluation_type": "execution",
                    "is_correct": "yes",
                    "circuit_breaker_active": True,
                    "report_type": "ops_overview",
                }
            ]
        )

        with patch.object(dashboard_module, "current_language", return_value="zh"):
            displayed = dashboard_module.display_df(df)

        row = displayed.iloc[0].to_dict()
        self.assertEqual(row["状态"], "正常")
        self.assertEqual(row["事件类型"], "研究复核")
        self.assertEqual(row["评估类型"], "执行")
        self.assertEqual(row["是否正确"], "是")
        self.assertEqual(row["熔断触发"], "是")
        self.assertEqual(row["报告类型"], "运维总览")

    def test_dashboard_localize_report_text_translates_english_report(self):
        content = "\n".join(
            [
                "# Performance Report",
                "- XGBoost Direction Accuracy: 48.00%",
                "- Degradation Status: warming_up",
            ]
        )

        with patch.object(dashboard_module, "current_language", return_value="zh"):
            localized = dashboard_module.localize_report_text(content)

        self.assertIn("# 性能报告", localized)
        self.assertIn("- XGBoost 方向准确率: 48.00%", localized)
        self.assertIn("- 衰减状态: 预热中", localized)

    def test_dashboard_display_research_text_translates_known_english_prefixes(self):
        content = "\n".join(
            [
                "CoinDesk headlines: BTC reclaims key level | ETH sees inflow",
                "Macro context neutral.",
                "On-chain data unavailable, using neutral on-chain context.",
            ]
        )

        with patch.object(dashboard_module, "current_language", return_value="zh"):
            localized = dashboard_module.display_research_text(content)

        self.assertIn("CoinDesk 头条：BTC reclaims key level | ETH sees inflow", localized)
        self.assertIn("宏观环境中性。", localized)
        self.assertIn("当前未获取到链上数据，使用中性链上上下文。", localized)

    def test_decision_engine_requires_all_entry_conditions(self):
        pipeline = FeaturePipeline()
        features = pipeline.build(
            FeatureInput(
                symbol="BTC/USDT",
                candles_1h=make_candles(240, 100),
                candles_4h=make_candles(240, 120),
                candles_1d=make_candles(240, 140),
                sentiment_value=0.2,
                llm_sentiment_score=0.2,
                market_regime_score=0.8,
            )
        )
        prediction = PredictionResult(
            symbol="BTC/USDT",
            up_probability=0.82,
            feature_count=len(features.values),
            model_version="xgboost_v3_BTC_USDT.json",
        )
        features.values["atr_percentile_4h"] = 0.5
        insight = ResearchInsight(
            symbol="BTC/USDT",
            market_regime=MarketRegime.UPTREND,
            sentiment_score=0.2,
            confidence=0.5,
            suggested_action=SuggestedAction.OPEN_LONG,
            key_reason=["ok"],
        )
        risk_result = SimpleNamespace(
            allowed=True,
            reason="",
            allowed_position_value=1000.0,
            stop_loss_pct=0.005,
            take_profit_levels=[0.05, 0.08],
            trailing_stop_drawdown_pct=0.3,
        )
        engine = DecisionEngine(0.7, 0.8, 0.2, 0.8, 0.4, -0.3)
        context, decision = engine.evaluate_entry(
            "BTC/USDT", prediction, insight, features, risk_result
        )
        self.assertTrue(context.should_open)
        self.assertTrue(decision.should_execute)
        self.assertEqual(decision.portfolio_rating, "OVERWEIGHT")
        self.assertAlmostEqual(decision.position_scale, 0.55)
        self.assertAlmostEqual(decision.position_value, 550.0)
        self.assertGreaterEqual(decision.final_score, 0.8)

    def test_decision_engine_assigns_buy_rating_for_high_conviction_entries(self):
        pipeline = FeaturePipeline()
        features = pipeline.build(
            FeatureInput(
                symbol="BTC/USDT",
                candles_1h=make_candles(240, 100),
                candles_4h=make_candles(240, 120),
                candles_1d=make_candles(240, 140),
                sentiment_value=0.3,
                llm_sentiment_score=0.3,
                market_regime_score=0.8,
            )
        )
        prediction = PredictionResult(
            symbol="BTC/USDT",
            up_probability=0.92,
            feature_count=len(features.values),
            model_version="xgboost_v3_BTC_USDT.json",
        )
        features.values["atr_percentile_4h"] = 0.5
        features.values["volume_ratio_1h"] = 1.2
        insight = ResearchInsight(
            symbol="BTC/USDT",
            market_regime=MarketRegime.UPTREND,
            sentiment_score=0.3,
            confidence=0.7,
            suggested_action=SuggestedAction.OPEN_LONG,
            key_reason=["ok"],
            risk_warning=[],
        )
        risk_result = SimpleNamespace(
            allowed=True,
            reason="",
            allowed_position_value=1000.0,
            stop_loss_pct=0.005,
            take_profit_levels=[0.05, 0.08],
            trailing_stop_drawdown_pct=0.3,
        )
        engine = DecisionEngine(0.7, 0.8, 0.2, 0.8, 0.4, -0.3)
        context, decision = engine.evaluate_entry(
            "BTC/USDT", prediction, insight, features, risk_result
        )
        self.assertTrue(context.should_open)
        self.assertEqual(decision.portfolio_rating, "BUY")
        self.assertAlmostEqual(decision.position_scale, 1.0)
        self.assertAlmostEqual(decision.position_value, 1000.0)

    def test_decision_engine_requires_explicit_open_long_signal(self):
        pipeline = FeaturePipeline()
        features = pipeline.build(
            FeatureInput(
                symbol="BTC/USDT",
                candles_1h=make_candles(240, 100),
                candles_4h=make_candles(240, 120),
                candles_1d=make_candles(240, 140),
                sentiment_value=0.2,
                llm_sentiment_score=0.2,
                market_regime_score=0.8,
            )
        )
        prediction = PredictionResult(
            symbol="BTC/USDT",
            up_probability=0.82,
            feature_count=len(features.values),
            model_version="xgboost_v3_BTC_USDT.json",
        )
        features.values["atr_percentile_4h"] = 0.5
        insight = ResearchInsight(
            symbol="BTC/USDT",
            market_regime=MarketRegime.UPTREND,
            sentiment_score=0.2,
            confidence=0.5,
            suggested_action=SuggestedAction.HOLD,
            key_reason=["wait"],
        )
        risk_result = SimpleNamespace(
            allowed=True,
            reason="",
            allowed_position_value=1000.0,
            stop_loss_pct=0.005,
            take_profit_levels=[0.05, 0.08],
            trailing_stop_drawdown_pct=0.3,
        )
        engine = DecisionEngine(0.7, 0.8, 0.2, 0.8, 0.4, -0.3)
        context, decision = engine.evaluate_entry(
            "BTC/USDT", prediction, insight, features, risk_result
        )
        self.assertFalse(context.should_open)
        self.assertFalse(decision.should_execute)

    def test_decision_engine_blocks_fallback_research_entries(self):
        pipeline = FeaturePipeline()
        features = pipeline.build(
            FeatureInput(
                symbol="BTC/USDT",
                candles_1h=make_candles(240, 100),
                candles_4h=make_candles(240, 120),
                candles_1d=make_candles(240, 140),
                sentiment_value=0.2,
                llm_sentiment_score=0.2,
                market_regime_score=0.8,
            )
        )
        prediction = PredictionResult(
            symbol="BTC/USDT",
            up_probability=0.82,
            feature_count=len(features.values),
            model_version="xgboost_v3_BTC_USDT.json",
        )
        features.values["atr_percentile_4h"] = 0.5
        insight = ResearchInsight(
            symbol="BTC/USDT",
            market_regime=MarketRegime.UPTREND,
            sentiment_score=0.2,
            confidence=0.5,
            suggested_action=SuggestedAction.OPEN_LONG,
            key_reason=["fallback_research_model"],
        )
        risk_result = SimpleNamespace(
            allowed=True,
            reason="",
            allowed_position_value=1000.0,
            stop_loss_pct=0.005,
            take_profit_levels=[0.05, 0.08],
            trailing_stop_drawdown_pct=0.3,
        )
        engine = DecisionEngine(0.7, 0.8, 0.2, 0.8, 0.4, -0.3)
        context, decision = engine.evaluate_entry(
            "BTC/USDT", prediction, insight, features, risk_result
        )
        self.assertFalse(context.should_open)
        self.assertFalse(decision.should_execute)
        self.assertEqual(decision.reason, "research unavailable")

    def test_decision_engine_blocks_fallback_predictor_entries(self):
        pipeline = FeaturePipeline()
        features = pipeline.build(
            FeatureInput(
                symbol="BTC/USDT",
                candles_1h=make_candles(240, 100),
                candles_4h=make_candles(240, 120),
                candles_1d=make_candles(240, 140),
                sentiment_value=0.2,
                llm_sentiment_score=0.2,
                market_regime_score=0.8,
            )
        )
        prediction = PredictionResult(
            symbol="BTC/USDT",
            up_probability=0.82,
            feature_count=len(features.values),
            model_version="fallback_v2",
        )
        features.values["atr_percentile_4h"] = 0.5
        insight = ResearchInsight(
            symbol="BTC/USDT",
            market_regime=MarketRegime.UPTREND,
            sentiment_score=0.2,
            confidence=0.5,
            suggested_action=SuggestedAction.OPEN_LONG,
            key_reason=["ok"],
        )
        risk_result = SimpleNamespace(
            allowed=True,
            reason="",
            allowed_position_value=1000.0,
            stop_loss_pct=0.005,
            take_profit_levels=[0.05, 0.08],
            trailing_stop_drawdown_pct=0.3,
        )
        engine = DecisionEngine(0.7, 0.8, 0.2, 0.8, 0.4, -0.3)
        context, decision = engine.evaluate_entry(
            "BTC/USDT", prediction, insight, features, risk_result
        )
        self.assertFalse(context.should_open)
        self.assertFalse(decision.should_execute)
        self.assertEqual(decision.reason, "model unavailable")

    def test_decision_engine_uses_adaptive_liquidity_floor_for_entry(self):
        pipeline = FeaturePipeline()
        features = pipeline.build(
            FeatureInput(
                symbol="BTC/USDT",
                candles_1h=make_candles(240, 100),
                candles_4h=make_candles(240, 120),
                candles_1d=make_candles(240, 140),
                sentiment_value=0.2,
                llm_sentiment_score=0.2,
                market_regime_score=0.8,
            )
        )
        features.values["atr_percentile_4h"] = 0.5
        features.values["volume_ratio_1h"] = 0.4
        features.values["adaptive_min_liquidity_ratio"] = 0.35
        features.values["adx_4h"] = 28.0
        features.values["di_plus_4h"] = 24.0
        features.values["di_minus_4h"] = 12.0
        prediction = PredictionResult(
            symbol="BTC/USDT",
            up_probability=0.82,
            feature_count=len(features.values),
            model_version="xgboost_v3_BTC_USDT.json",
        )
        insight = ResearchInsight(
            symbol="BTC/USDT",
            market_regime=MarketRegime.UPTREND,
            sentiment_score=0.2,
            confidence=0.6,
            suggested_action=SuggestedAction.OPEN_LONG,
            key_reason=["ok"],
            risk_warning=[],
        )
        risk_result = SimpleNamespace(
            allowed=True,
            reason="",
            allowed_position_value=1000.0,
            stop_loss_pct=0.005,
            take_profit_levels=[0.05, 0.08],
            trailing_stop_drawdown_pct=0.3,
        )
        engine = DecisionEngine(0.7, 0.8, 0.2, 0.8, 0.4, -0.3)
        context, decision = engine.evaluate_entry(
            "BTC/USDT", prediction, insight, features, risk_result
        )
        self.assertTrue(context.should_open)
        self.assertTrue(decision.should_execute)
        self.assertIn("liq_floor=0.35", decision.reason)

    def test_decision_engine_executes_extreme_fear_offensive_setup(self):
        pipeline = FeaturePipeline()
        features = pipeline.build(
            FeatureInput(
                symbol="ETH/USDT",
                candles_1h=make_candles(240, 100),
                candles_4h=make_candles(240, 120),
                candles_1d=make_candles(240, 140),
                sentiment_value=0.1,
                llm_sentiment_score=-0.8,
                market_regime_score=-0.6,
            )
        )
        features.values["atr_percentile_4h"] = 0.5
        features.values["volume_ratio_1h"] = 0.73
        features.values["adaptive_min_liquidity_ratio"] = 0.35
        features.values["adx_4h"] = 23.0
        features.values["di_plus_4h"] = 25.0
        features.values["di_minus_4h"] = 21.0
        prediction = PredictionResult(
            symbol="ETH/USDT",
            up_probability=0.59,
            feature_count=len(features.values),
            model_version="xgboost_v3_ETH_USDT.json",
        )
        insight = ResearchInsight(
            symbol="ETH/USDT",
            market_regime=MarketRegime.EXTREME_FEAR,
            sentiment_score=-0.8,
            confidence=0.48,
            suggested_action=SuggestedAction.OPEN_LONG,
            key_reason=[
                "extreme_fear_offensive_setup",
                "extreme_fear_offensive_open",
            ],
            risk_warning=["macro stress"],
        )
        risk_result = SimpleNamespace(
            allowed=True,
            reason="",
            allowed_position_value=1000.0,
            stop_loss_pct=0.005,
            take_profit_levels=[0.05, 0.08],
            trailing_stop_drawdown_pct=0.3,
        )
        engine = DecisionEngine(0.72, 0.53, 0.1, 0.5, 0.4, -0.3)
        context, decision = engine.evaluate_entry(
            "ETH/USDT", prediction, insight, features, risk_result
        )
        self.assertTrue(context.should_open)
        self.assertTrue(decision.should_execute)
        self.assertEqual(decision.portfolio_rating, "OVERWEIGHT")
        self.assertAlmostEqual(decision.position_scale, 0.4)
        self.assertAlmostEqual(decision.position_value, 400.0)
        self.assertIn("mode=extreme_fear_offensive", decision.reason)

    def test_decision_engine_blocks_moderate_setup_in_extreme_fear_conservative_mode(self):
        pipeline = FeaturePipeline()
        features = pipeline.build(
            FeatureInput(
                symbol="BTC/USDT",
                candles_1h=make_candles(240, 100),
                candles_4h=make_candles(240, 120),
                candles_1d=make_candles(240, 140),
                sentiment_value=0.05,
                llm_sentiment_score=0.05,
                market_regime_score=-0.6,
            )
        )
        features.values["atr_percentile_4h"] = 0.5
        features.values["volume_ratio_1h"] = 0.78
        features.values["adaptive_min_liquidity_ratio"] = 0.35
        features.values["adx_4h"] = 23.0
        features.values["di_plus_4h"] = 25.0
        features.values["di_minus_4h"] = 21.0
        prediction = PredictionResult(
            symbol="BTC/USDT",
            up_probability=0.74,
            feature_count=len(features.values),
            model_version="xgboost_v3_BTC_USDT.json",
        )
        insight = ResearchInsight(
            symbol="BTC/USDT",
            market_regime=MarketRegime.EXTREME_FEAR,
            sentiment_score=0.05,
            confidence=0.56,
            suggested_action=SuggestedAction.OPEN_LONG,
            key_reason=["bounce_setup"],
            risk_warning=[],
        )
        risk_result = SimpleNamespace(
            allowed=True,
            reason="",
            allowed_position_value=1000.0,
            stop_loss_pct=0.005,
            take_profit_levels=[0.05, 0.08],
            trailing_stop_drawdown_pct=0.3,
        )
        engine = DecisionEngine(0.72, 0.53, 0.1, 0.5, 0.4, -0.3)
        context, decision = engine.evaluate_entry(
            "BTC/USDT", prediction, insight, features, risk_result
        )
        self.assertFalse(context.should_open)
        self.assertFalse(decision.should_execute)
        self.assertIn("mode=extreme_fear_conservative", decision.reason)

    def test_decision_engine_caps_position_for_strong_non_override_extreme_fear_setup(self):
        pipeline = FeaturePipeline()
        features = pipeline.build(
            FeatureInput(
                symbol="BTC/USDT",
                candles_1h=make_candles(240, 100),
                candles_4h=make_candles(240, 120),
                candles_1d=make_candles(240, 140),
                sentiment_value=0.2,
                llm_sentiment_score=0.2,
                market_regime_score=-0.6,
            )
        )
        features.values["atr_percentile_4h"] = 0.5
        features.values["volume_ratio_1h"] = 0.95
        features.values["adaptive_min_liquidity_ratio"] = 0.35
        features.values["adx_4h"] = 28.0
        features.values["di_plus_4h"] = 28.0
        features.values["di_minus_4h"] = 18.0
        prediction = PredictionResult(
            symbol="BTC/USDT",
            up_probability=0.92,
            feature_count=len(features.values),
            model_version="xgboost_v3_BTC_USDT.json",
        )
        insight = ResearchInsight(
            symbol="BTC/USDT",
            market_regime=MarketRegime.EXTREME_FEAR,
            sentiment_score=0.20,
            confidence=0.80,
            suggested_action=SuggestedAction.OPEN_LONG,
            key_reason=["high_quality_reversal"],
            risk_warning=[],
        )
        risk_result = SimpleNamespace(
            allowed=True,
            reason="",
            allowed_position_value=1000.0,
            stop_loss_pct=0.005,
            take_profit_levels=[0.05, 0.08],
            trailing_stop_drawdown_pct=0.3,
        )
        engine = DecisionEngine(0.72, 0.53, 0.1, 0.5, 0.4, -0.3)
        context, decision = engine.evaluate_entry(
            "BTC/USDT", prediction, insight, features, risk_result
        )
        self.assertTrue(context.should_open)
        self.assertTrue(decision.should_execute)
        self.assertEqual(decision.portfolio_rating, "BUY")
        self.assertAlmostEqual(decision.position_scale, 0.25)
        self.assertAlmostEqual(decision.position_value, 250.0)
        self.assertIn("mode=extreme_fear_conservative", decision.reason)

    def test_decision_engine_exit_uses_configured_thresholds(self):
        engine = DecisionEngine(
            0.7,
            0.8,
            0.2,
            0.8,
            0.4,
            -0.3,
            fixed_stop_loss_pct=0.01,
            take_profit_levels=[0.03, 0.07],
            max_hold_hours=12,
        )
        reasons = engine.evaluate_exit(
            position={"symbol": "BTC/USDT", "entry_price": 100.0},
            current_price=103.5,
            prediction=PredictionResult(symbol="BTC/USDT", up_probability=0.9, model_version="x"),
            insight=ResearchInsight(symbol="BTC/USDT", sentiment_score=0.2),
            hours_held=12,
        )
        self.assertIn("take_profit_1", reasons)
        self.assertIn("time_stop", reasons)
        self.assertNotIn("fixed_stop_loss", reasons)

    def test_risk_manager_enforces_max_positions_and_exposure(self):
        manager = RiskManager(self.settings.risk, self.settings.strategy)
        positions = [
            {"symbol": "BTC/USDT", "entry_price": 100.0, "quantity": 10.0},
            {"symbol": "ETH/USDT", "entry_price": 100.0, "quantity": 10.0},
            {"symbol": "SOL/USDT", "entry_price": 100.0, "quantity": 10.0},
        ]
        account = manager.build_account_state(
            equity=10000.0,
            positions=positions,
            realized_pnl_today=0.0,
            realized_pnl_week=0.0,
        )
        result = manager.can_open_position(
            account=account,
            positions=positions,
            symbol="AVAX/USDT",
            atr=100.0,
            entry_price=100.0,
            liquidity_ratio=1.0,
        )
        self.assertFalse(result.allowed)
        self.assertIn("max positions", result.reason)

    def test_risk_manager_uses_current_price_for_exposure(self):
        manager = RiskManager(self.settings.risk, self.settings.strategy)
        positions = [
            {
                "symbol": "BTC/USDT",
                "entry_price": 100.0,
                "current_price": 200.0,
                "quantity": 1.0,
            }
        ]
        account = manager.build_account_state(
            equity=1000.0,
            positions=positions,
            realized_pnl_today=0.0,
            realized_pnl_week=0.0,
        )
        self.assertAlmostEqual(account.total_exposure_pct, 0.2, places=6)
        result = manager.can_open_position(
            account=account,
            positions=positions,
            symbol="BTC/USDT",
            atr=10.0,
            entry_price=100.0,
            liquidity_ratio=1.0,
        )
        self.assertFalse(result.allowed)
        self.assertIn("symbol exposure exceeded", result.reason)

    def test_risk_manager_applies_dynamic_position_factor(self):
        manager = RiskManager(self.settings.risk, self.settings.strategy)
        account = manager.build_account_state(
            equity=10000.0,
            positions=[],
            realized_pnl_today=-150.0,
            realized_pnl_week=0.0,
            peak_equity=10000.0,
        )
        result = manager.can_open_position(
            account=account,
            positions=[],
            symbol="BTC/USDT",
            atr=100.0,
            entry_price=100.0,
            liquidity_ratio=1.0,
            consecutive_wins=2,
            consecutive_losses=0,
        )
        self.assertTrue(result.allowed)
        self.assertLessEqual(result.dynamic_position_factor, 0.5)

    def test_risk_manager_cuts_position_for_highly_correlated_existing_exposure(self):
        settings = self.settings.model_copy(deep=True)
        manager = RiskManager(settings.risk, settings.strategy)
        positions = [
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 2500.0,
                "current_price": 2500.0,
                "quantity": 1.0,
            }
        ]
        account = manager.build_account_state(
            equity=10000.0,
            positions=positions,
            realized_pnl_today=0.0,
            realized_pnl_week=0.0,
        )
        candidate_returns = [
            0.002 + 0.0003 * math.sin(index / 5.0)
            for index in range(120)
        ]
        held_returns = [
            value * 0.97 + 0.00005 * math.cos(index / 7.0)
            for index, value in enumerate(candidate_returns)
        ]

        result = manager.can_open_position(
            account=account,
            positions=positions,
            symbol="ETH/USDT",
            atr=2.0,
            entry_price=100.0,
            liquidity_ratio=1.0,
            correlation_price_data={
                "ETH/USDT": self._prices_from_returns(100.0, candidate_returns),
                "BTC/USDT": self._prices_from_returns(200.0, held_returns),
            },
        )

        self.assertTrue(result.allowed)
        self.assertLess(result.allowed_position_value, 1000.0)
        self.assertLess(result.correlation_position_factor, 1.0)
        self.assertGreater(result.correlation_effective_exposure_pct, 0.30)
        self.assertIn("BTC/USDT", result.correlation_crowded_symbols)
        self.assertIn("correlation haircut", result.reason)

    def test_risk_manager_blocks_position_when_correlation_crowding_is_extreme(self):
        settings = self.settings.model_copy(deep=True)
        settings.risk.max_positions = 3
        manager = RiskManager(settings.risk, settings.strategy)
        positions = [
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 2000.0,
                "current_price": 2000.0,
                "quantity": 1.0,
            },
            {
                "symbol": "ETH/USDT",
                "direction": "LONG",
                "entry_price": 2000.0,
                "current_price": 2000.0,
                "quantity": 1.0,
            },
        ]
        account = manager.build_account_state(
            equity=10000.0,
            positions=positions,
            realized_pnl_today=0.0,
            realized_pnl_week=0.0,
        )
        base_returns = [
            0.002 + 0.0002 * math.sin(index / 4.0)
            for index in range(120)
        ]

        result = manager.can_open_position(
            account=account,
            positions=positions,
            symbol="SOL/USDT",
            atr=2.0,
            entry_price=100.0,
            liquidity_ratio=1.0,
            correlation_price_data={
                "SOL/USDT": self._prices_from_returns(100.0, base_returns),
                "BTC/USDT": self._prices_from_returns(200.0, base_returns),
                "ETH/USDT": self._prices_from_returns(250.0, base_returns),
            },
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.allowed_position_value, 0.0)
        self.assertEqual(result.correlation_position_factor, 0.0)
        self.assertIn("correlation crowding block", result.reason)
        self.assertIn("BTC/USDT", result.correlation_crowded_symbols)
        self.assertIn("ETH/USDT", result.correlation_crowded_symbols)

    def test_risk_manager_keeps_position_unchanged_for_weakly_correlated_symbol(self):
        manager = RiskManager(self.settings.risk, self.settings.strategy)
        positions = [
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 2000.0,
                "current_price": 2000.0,
                "quantity": 1.0,
            }
        ]
        account = manager.build_account_state(
            equity=10000.0,
            positions=positions,
            realized_pnl_today=0.0,
            realized_pnl_week=0.0,
        )
        candidate_returns = [
            0.006 * math.sin(index / 3.0)
            for index in range(120)
        ]
        held_returns = [
            0.006 * math.cos(index / 3.0)
            for index in range(120)
        ]

        result = manager.can_open_position(
            account=account,
            positions=positions,
            symbol="XRP/USDT",
            atr=2.0,
            entry_price=100.0,
            liquidity_ratio=1.0,
            correlation_price_data={
                "XRP/USDT": self._prices_from_returns(100.0, candidate_returns),
                "BTC/USDT": self._prices_from_returns(200.0, held_returns),
            },
        )

        self.assertTrue(result.allowed)
        self.assertAlmostEqual(result.allowed_position_value, 1000.0, places=6)
        self.assertAlmostEqual(result.correlation_position_factor, 1.0, places=6)
        self.assertEqual(result.reason, "")
        self.assertEqual(result.correlation_crowded_symbols, [])

    def test_risk_manager_triggers_circuit_breaker(self):
        manager = RiskManager(self.settings.risk, self.settings.strategy)
        account = manager.build_account_state(
            equity=10000.0,
            positions=[],
            realized_pnl_today=-300.0,
            realized_pnl_week=0.0,
            peak_equity=10000.0,
        )
        reason = manager.check_circuit_breaker(account)
        self.assertEqual(reason, "daily_loss_limit")

    def test_risk_manager_applies_weekly_and_drawdown_cooldowns(self):
        manager = RiskManager(self.settings.risk, self.settings.strategy)
        now = datetime.now(timezone.utc)
        weekly_account = manager.build_account_state(
            equity=10000.0,
            positions=[],
            realized_pnl_today=0.0,
            realized_pnl_week=-600.0,
            peak_equity=10000.0,
        )
        weekly_cooldown = manager.apply_account_cooldown(
            weekly_account,
            current_cooldown_until=None,
            now=now,
        )
        self.assertIsNotNone(weekly_cooldown)
        self.assertGreaterEqual(
            (weekly_cooldown - now).total_seconds(),
            3 * 24 * 3600 - 1,
        )

        drawdown_account = manager.build_account_state(
            equity=8900.0,
            positions=[],
            realized_pnl_today=0.0,
            realized_pnl_week=0.0,
            peak_equity=10000.0,
        )
        drawdown_cooldown = manager.apply_account_cooldown(
            drawdown_account,
            current_cooldown_until=None,
            now=now,
        )
        self.assertIsNotNone(drawdown_cooldown)
        self.assertGreaterEqual(
            (drawdown_cooldown - now).total_seconds(),
            7 * 24 * 3600 - 1,
        )

    def test_engine_loss_cooldown_preserves_later_guard_cooldown(self):
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

        later_cooldown = datetime.now(timezone.utc) + timedelta(hours=12)
        earlier_cooldown = later_cooldown - timedelta(hours=6)
        engine._cooldown_until = later_cooldown
        engine._consecutive_losses = max(engine.settings.risk.consecutive_loss_limit - 1, 0)
        engine.risk.update_cooldown_after_losses = lambda losses: earlier_cooldown

        engine._record_trade_result(-25.0)

        self.assertEqual(engine._cooldown_until, later_cooldown)

    def test_engine_cycle_opens_at_most_three_positions(self):
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
                    model_version="fake",
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
                )

        class FakeMacro:
            def get_summary(self, fear_greed=None):
                return SimpleNamespace(summary="macro ok", score=0.0, position_adjustment=1.0)

        custom = self.settings.model_copy(deep=True)
        custom.app.db_path = self.db_path
        custom.exchange.symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "AVAX/USDT",
        ]

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "SentimentCollector", FakeSentiment), \
             patch.object(engine_module, "NewsService", FakeNews), \
             patch.object(engine_module, "MacroService", FakeMacro), \
             patch.object(engine_module, "ResearchLLMAnalyzer", FakeResearch), \
             patch.object(engine_module, "XGBoostPredictor", FakePredictor), \
             patch.object(engine_module, "MarketRegimeDetector", lambda: FakeRegimeDetector()):
            engine = engine_module.CryptoAIV2Engine(custom)
            model_dir = Path(self.db_path).parent / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            for symbol in custom.exchange.symbols:
                model_path = model_dir / f"xgboost_v2_{symbol.replace('/', '_')}.json"
                model_path.write_text("{}", encoding="utf-8")
                engine.storage.insert_training_run(
                    {
                        "symbol": symbol,
                        "rows": custom.training.minimum_training_rows,
                        "feature_count": 10,
                        "positives": 5,
                        "negatives": 5,
                        "model_path": str(model_path),
                        "trained_with_xgboost": True,
                        "holdout_accuracy": 0.6,
                    }
                )
            engine.storage.set_json_state(
                engine.EXECUTION_SYMBOLS_STATE_KEY,
                custom.exchange.symbols,
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
            engine.run_once()
            self.assertLessEqual(len(engine.storage.get_positions()), 3)
            with engine.storage._conn() as conn:
                feature_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM feature_snapshots"
                ).fetchone()["c"]
                prediction_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM prediction_runs"
                ).fetchone()["c"]
                research_input_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM research_inputs"
                ).fetchone()["c"]
                account_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM account_snapshots"
                ).fetchone()["c"]
                event_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM execution_events"
                ).fetchone()["c"]
                cycle_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM cycle_runs"
                ).fetchone()["c"]
            self.assertGreater(feature_count, 0)
            self.assertGreater(prediction_count, 0)
            self.assertGreater(research_input_count, 0)
            self.assertGreater(account_count, 0)
            self.assertGreater(event_count, 0)
            self.assertGreater(cycle_count, 0)

    def test_engine_cycle_does_not_open_position_with_fallback_predictor(self):
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
                pass

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
                )

        class FakeMacro:
            def get_summary(self, fear_greed=None):
                return SimpleNamespace(summary="macro ok", score=0.0, position_adjustment=1.0)

        custom = self.settings.model_copy(deep=True)
        custom.app.db_path = self.db_path
        custom.exchange.symbols = ["BTC/USDT"]
        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "BinanceMarketDataCollector", FakeMarket), \
             patch.object(engine_module, "SentimentCollector", FakeSentiment), \
             patch.object(engine_module, "NewsService", FakeNews), \
             patch.object(engine_module, "MacroService", FakeMacro), \
             patch.object(engine_module, "ResearchLLMAnalyzer", FakeResearch), \
             patch.object(engine_module, "XGBoostPredictor", FakePredictor), \
             patch.object(engine_module, "MarketRegimeDetector", lambda: FakeRegimeDetector()):
            engine = engine_module.CryptoAIV2Engine(custom)
            model_dir = Path(self.db_path).parent / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "xgboost_v2_BTC_USDT.json"
            model_path.write_text("{}", encoding="utf-8")
            engine.storage.insert_training_run(
                {
                    "symbol": "BTC/USDT",
                    "rows": custom.training.minimum_training_rows,
                    "feature_count": 10,
                    "positives": 5,
                    "negatives": 5,
                    "model_path": str(model_path),
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
            engine.run_once()

        self.assertEqual(len(engine.storage.get_positions()), 0)
        with engine.storage._conn() as conn:
            prediction_count = conn.execute(
                "SELECT COUNT(*) AS c FROM prediction_runs"
            ).fetchone()["c"]
            trade_count = conn.execute(
                "SELECT COUNT(*) AS c FROM trades"
            ).fetchone()["c"]
        self.assertEqual(prediction_count, 0)
        self.assertEqual(trade_count, 0)

    def test_engine_account_state_includes_unrealized_pnl(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 80.0

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 2.0,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        account = engine._account_state(
            datetime.now(timezone.utc),
            engine.storage.get_positions(),
        )
        self.assertAlmostEqual(account.unrealized_pnl, -40.0, places=6)
        self.assertAlmostEqual(
            account.equity,
            engine.executor.initial_balance - 40.0,
            places=6,
        )
        self.assertGreater(account.drawdown_pct, 0.0)

    def test_engine_account_state_uses_period_baseline_for_daily_loss(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

            def fetch_latest_price(self, symbol):
                return 90.0

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        engine.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "1h",
            [
                {
                    "timestamp": int((today_start - timedelta(hours=1)).timestamp() * 1000),
                    "open": 95.0,
                    "high": 96.0,
                    "low": 94.0,
                    "close": 95.0,
                    "volume": 1.0,
                }
            ],
        )
        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": (now - timedelta(days=2)).isoformat(),
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        account = engine._account_state(now, engine.storage.get_positions())
        self.assertAlmostEqual(account.unrealized_pnl, -10.0, places=6)
        self.assertAlmostEqual(account.daily_loss_pct * account.equity, 5.0, places=3)

    def test_engine_persist_analysis_uses_feature_timestamp(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        feature_time = datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc)
        engine._persist_analysis(
            "BTC/USDT",
            ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.UPTREND,
                sentiment_score=0.2,
                confidence=0.6,
                suggested_action=SuggestedAction.OPEN_LONG,
            ),
            PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.82,
                feature_count=10,
                model_version="test",
            ),
            0.9,
            analysis_timestamp=feature_time,
        )
        with engine.storage._conn() as conn:
            row = conn.execute(
                "SELECT timestamp, decision_json FROM prediction_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(row["timestamp"], feature_time.isoformat())
        self.assertIn("\"portfolio_rating\": \"HOLD\"", row["decision_json"])

    def test_engine_keeps_cycle_failed_when_circuit_breaker_is_active(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        guarded_account = SimpleNamespace(circuit_breaker_active=True)

        def fake_account_state(now, positions):
            engine._circuit_breaker_active = True
            engine._circuit_breaker_reason = "daily_loss_limit"
            return guarded_account

        engine.get_active_symbols = lambda force_refresh=False, now=None: []
        engine._check_market_latency = lambda now, symbols=None: False
        engine._account_state = fake_account_state
        engine._apply_model_degradation = lambda now: None
        engine._persist_runtime_settings_effective = lambda: None
        engine._enforce_accuracy_guard = lambda now: None
        engine._manage_open_positions = lambda now, positions, account: 0
        engine._generate_reports = lambda now: None
        engine.reconciler = SimpleNamespace(
            run=lambda: SimpleNamespace(mismatch_count=0, status="ok")
        )
        engine.notifier = SimpleNamespace(notify=lambda *args, **kwargs: None)

        engine.run_once()

        self.assertEqual(engine.storage.get_state("last_cycle_status"), "failed")
        with engine.storage._conn() as conn:
            row = conn.execute(
                "SELECT status FROM cycle_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(row["status"], "failed")

    def test_engine_generate_reports_uses_equity_for_daily_balance(self):
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

        engine.storage.upsert_position(
            {
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 2.0,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "stop_loss": 95.0,
                "take_profit": 110.0,
            }
        )
        captured = {}
        engine.notifier = SimpleNamespace(
            notify_daily_report=lambda positions, trades, balance: captured.update(
                {"positions": positions, "trades": trades, "balance": balance}
            )
        )
        engine._generate_reports(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
        self.assertAlmostEqual(captured["balance"], 10020.0, places=6)
        self.assertEqual(len(captured["positions"]), 1)

    def test_engine_generate_reports_runs_once_per_day_without_midnight_alignment(self):
        import core.engine as engine_module

        class FakeMarket:
            def __init__(self, storage, proxy=None):
                self.storage = storage

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path

        with patch.object(engine_module, "OKXMarketDataCollector", FakeMarket):
            engine = engine_module.CryptoAIV2Engine(settings)

        calls = []
        engine.notifier = SimpleNamespace(
            notify_daily_report=lambda positions, trades, balance: calls.append(balance)
        )
        engine._portfolio_equity = lambda positions: 1000.0

        engine._generate_reports(datetime(2026, 1, 1, 4, 0, tzinfo=timezone.utc))
        engine._generate_reports(datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc))
        engine._generate_reports(datetime(2026, 1, 2, 4, 0, tzinfo=timezone.utc))

        self.assertEqual(calls, [1000.0, 1000.0])
        self.assertEqual(
            engine.storage.get_state(engine.DAILY_REPORT_DATE_STATE_KEY),
            "2026-01-02",
        )

    def test_weekly_review_counts_trade_by_exit_time(self):
        from learning.reflector import TradeReflector

        with self.storage._conn() as conn:
            conn.execute(
                "INSERT INTO trades "
                "(id, symbol, direction, entry_price, quantity, entry_time, exit_time, pnl, pnl_pct, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "t1",
                    "BTC/USDT",
                    "LONG",
                    100.0,
                    1.0,
                    (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(),
                    (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
                    10.0,
                    10.0,
                    "closed",
                ),
            )
        report = TradeReflector(self.storage).generate_weekly_review()
        self.assertIn("总交易: 1", report)

    def test_reflector_normalizes_legacy_market_regime_values(self):
        from learning.reflector import TradeReflector

        reflector = TradeReflector(self.storage)
        self.assertEqual(
            reflector._normalize_market_regime("SIDEWAYS"),
            MarketRegime.RANGE,
        )
        self.assertEqual(
            reflector._normalize_market_regime("BULL_NORMAL"),
            MarketRegime.UPTREND,
        )
        self.assertEqual(
            reflector._normalize_market_regime("BEAR_CRASH"),
            MarketRegime.DOWNTREND,
        )

    def test_reflector_builds_okx_exchange_without_proxy_when_unset(self):
        from learning.reflector import TradeReflector

        class FakeOKX:
            def __init__(self, params):
                self.params = params

        fake_settings = SimpleNamespace(exchange=SimpleNamespace(proxy_url=""))
        with patch("learning.reflector.get_settings", return_value=fake_settings), \
             patch("ccxt.okx", FakeOKX):
            exchange = TradeReflector(self.storage)._build_okx_exchange()
        self.assertNotIn("proxies", exchange.params)
