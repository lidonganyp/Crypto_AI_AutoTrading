from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr

from analysis.cross_validation_service import CrossValidationService
from analysis.macro_service import MacroService
from analysis.news_service import NewsService
from analysis.onchain_service import OnchainService
from analysis.research_input_consistency import ResearchInputConsistencyService
from analysis.research_llm import ResearchLLMAnalyzer
from analysis.research_manager import ResearchManager
from config import OnchainSettings, get_settings
from core.feature_pipeline import FeatureInput, FeaturePipeline
from core.models import (
    MarketRegime,
    PredictionResult,
    ResearchInsight,
    SignalDirection,
    SuggestedAction,
    TradeReflection,
)
from core.openai_client_factory import build_openai_client
from core.storage import Storage
from learning.experience_store import ExperienceStore
from learning.reflector import TradeReflector
from learning.strategy_evolver import StrategyEvolver
from strategy.model_trainer import model_path_for_symbol
from strategy.risk_manager import RiskManager
from strategy.xgboost_predictor import XGBoostPredictor
from tests.v2_architecture_support import V2ArchitectureTestCase, make_candles


class V2StrategyResearchTests(V2ArchitectureTestCase):
    def test_predictor_aligns_runtime_features_to_model_feature_names(self):
        import strategy.xgboost_predictor as predictor_module

        captured = {}

        class FakeMatrix:
            def __init__(self, rows, feature_names):
                captured["rows"] = rows
                captured["feature_names"] = feature_names

        class FakeXGB:
            DMatrix = FakeMatrix

        class FakeModel:
            feature_names = ["close_1h", "close_4h"]

            def predict(self, matrix):
                return [0.7]

        pipeline = FeaturePipeline()
        snapshot = pipeline.build(
            FeatureInput(
                symbol="BTC/USDT",
                candles_1h=make_candles(240, 100),
                candles_4h=make_candles(240, 120),
                candles_1d=make_candles(240, 140),
            )
        )
        snapshot.values["new_runtime_feature"] = 123.0
        predictor = XGBoostPredictor("data/models/missing.json", enable_fallback=True)
        predictor.model = FakeModel()
        predictor.model_loaded = True

        with patch.object(predictor_module, "xgb", FakeXGB):
            result = predictor.predict(snapshot)

        self.assertEqual(captured["feature_names"], ["close_1h", "close_4h"])
        self.assertEqual(captured["rows"], [[snapshot.values["close_1h"], snapshot.values["close_4h"]]])
        self.assertEqual(result.feature_count, 2)

    def test_predictor_fallback_returns_probability(self):
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
        predictor = XGBoostPredictor("data/models/missing.json", enable_fallback=True)
        result = predictor.predict(snapshot)
        self.assertGreaterEqual(result.up_probability, 0.0)
        self.assertLessEqual(result.up_probability, 1.0)
        self.assertEqual(result.model_version, "fallback_v2")

    def test_predictor_falls_back_when_model_file_is_corrupted(self):
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
        model_path = Path(self.db_path).with_suffix(".json")
        model_path.write_text("not-a-valid-xgboost-model", encoding="utf-8")
        predictor = XGBoostPredictor(str(model_path), enable_fallback=True)
        result = predictor.predict(snapshot)
        self.assertGreaterEqual(result.up_probability, 0.0)
        self.assertLessEqual(result.up_probability, 1.0)
        self.assertEqual(result.model_version, "fallback_v2")

    def test_model_path_for_symbol_isolated_per_symbol(self):
        base = Path("/tmp/models/xgboost_v2.json")
        self.assertEqual(
            model_path_for_symbol(base, "BTC/USDT"),
            Path("/tmp/models/xgboost_v2_BTC_USDT.json"),
        )
        self.assertEqual(
            model_path_for_symbol(base, "ETH/USDT"),
            Path("/tmp/models/xgboost_v2_ETH_USDT.json"),
        )

    def test_research_fallback_produces_structured_output(self):
        analyzer = ResearchLLMAnalyzer(self.settings.llm, clients={})
        insight = analyzer._fallback("BTC/USDT", 60)
        self.assertEqual(insight.market_regime, MarketRegime.UPTREND)
        self.assertEqual(insight.suggested_action, SuggestedAction.OPEN_LONG)

    def test_research_fallback_uses_quant_context_for_extreme_fear_reversal(self):
        analyzer = ResearchLLMAnalyzer(self.settings.llm, clients={})
        insight = analyzer._fallback(
            "BTC/USDT",
            12.0,
            quant_context={
                "market_regime_hint": "EXTREME_FEAR",
                "liquidity_ratio": 0.92,
                "min_liquidity_ratio": 0.8,
                "trend_supportive": True,
                "oversold_reversal": True,
                "microstructure_supportive": True,
                "funding_rate": -0.01,
                "return_24h": -0.045,
            },
        )
        self.assertEqual(insight.market_regime, MarketRegime.EXTREME_FEAR)
        self.assertEqual(insight.suggested_action, SuggestedAction.OPEN_LONG)
        self.assertGreater(insight.sentiment_score, 0.0)
        self.assertIn("fallback_oversold_reversal", insight.key_reason)

    def test_research_llm_defers_client_init_failures_to_analyze_fallback(self):
        import analysis.research_llm as research_llm_module

        llm_settings = self.settings.llm.model_copy(deep=True)
        llm_settings.deepseek_api_key = SecretStr("test-key")

        with patch.object(
            research_llm_module,
            "build_openai_client",
            side_effect=ValueError("bad proxy"),
        ) as mocked_builder:
            analyzer = ResearchLLMAnalyzer(llm_settings, clients={})
            self.assertEqual(mocked_builder.call_count, 0)

            insight = analyzer.analyze(
                symbol="BTC/USDT",
                timestamp=datetime.now(timezone.utc).isoformat(),
                news_summary="neutral",
                macro_summary="neutral",
                fear_greed=55.0,
                onchain_summary="neutral",
            )

        self.assertEqual(mocked_builder.call_count, 1)
        self.assertIn("fallback_research_model", insight.key_reason)

    def test_research_llm_uses_configured_model_name(self):
        llm_settings = self.settings.llm.model_copy(deep=True)
        llm_settings.deepseek_model = "deepseek-custom-model"
        captured_models = []

        class FakeClient:
            class _Chat:
                class _Completions:
                    def create(self, **kwargs):
                        captured_models.append(kwargs["model"])
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content=(
                                            '{"market_regime":"UPTREND",'
                                            '"sentiment_score":0.2,'
                                            '"confidence":0.7,'
                                            '"risk_warning":[],'
                                            '"key_reason":["ok"],'
                                            '"suggested_action":"OPEN_LONG"}'
                                        )
                                    )
                                )
                            ]
                        )

                completions = _Completions()

            chat = _Chat()

        analyzer = ResearchLLMAnalyzer(
            llm_settings,
            clients={"deepseek": FakeClient()},
        )
        insight = analyzer.analyze(
            symbol="BTC/USDT",
            timestamp=datetime.now(timezone.utc).isoformat(),
            news_summary="neutral",
            macro_summary="neutral",
            fear_greed=60.0,
            onchain_summary="neutral",
        )

        self.assertEqual(captured_models, ["deepseek-custom-model"])
        self.assertEqual(insight.suggested_action, SuggestedAction.OPEN_LONG)

    def test_research_llm_calibrates_premature_close_to_open_long(self):
        class FakeClient:
            class _Chat:
                class _Completions:
                    def create(self, **kwargs):
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content=(
                                            '{"market_regime":"DOWNTREND",'
                                            '"sentiment_score":-0.8,'
                                            '"confidence":0.2,'
                                            '"risk_warning":["macro stress"],'
                                            '"key_reason":["trend down"],'
                                            '"suggested_action":"CLOSE"}'
                                        )
                                    )
                                )
                            ]
                        )

                completions = _Completions()

            chat = _Chat()

        analyzer = ResearchLLMAnalyzer(
            self.settings.llm,
            clients={"deepseek": FakeClient()},
        )
        insight = analyzer.analyze(
            symbol="ETH/USDT",
            timestamp=datetime.now(timezone.utc).isoformat(),
            news_summary="macro fragile but no fresh event shock",
            macro_summary="defensive but stabilizing",
            fear_greed=12.0,
            onchain_summary="mixed",
            quant_context={
                "market_regime_hint": "EXTREME_FEAR",
                "liquidity_ratio": 0.7,
                "min_liquidity_ratio": 0.35,
                "trend_against": True,
                "di_plus_4h": 20.0,
                "di_minus_4h": 24.0,
                "oversold_reversal": True,
                "microstructure_supportive": True,
                "funding_rate": -0.01,
                "return_24h": -0.03,
            },
        )
        self.assertEqual(insight.suggested_action, SuggestedAction.OPEN_LONG)
        self.assertEqual(insight.market_regime, MarketRegime.EXTREME_FEAR)
        self.assertIn("calibrated_extreme_fear_reversal_open", insight.key_reason)
        self.assertGreater(insight.sentiment_score, -0.2)

    def test_research_llm_keeps_close_when_event_risk_is_present(self):
        class FakeClient:
            class _Chat:
                class _Completions:
                    def create(self, **kwargs):
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content=(
                                            '{"market_regime":"EXTREME_FEAR",'
                                            '"sentiment_score":-0.8,'
                                            '"confidence":0.2,'
                                            '"risk_warning":["macro stress"],'
                                            '"key_reason":["trend down"],'
                                            '"suggested_action":"CLOSE"}'
                                        )
                                    )
                                )
                            ]
                        )

                completions = _Completions()

            chat = _Chat()

        analyzer = ResearchLLMAnalyzer(
            self.settings.llm,
            clients={"deepseek": FakeClient()},
        )
        insight = analyzer.analyze(
            symbol="ETH/USDT",
            timestamp=datetime.now(timezone.utc).isoformat(),
            news_summary="ETH faces liquidation cascade after hack",
            macro_summary="defensive",
            fear_greed=12.0,
            onchain_summary="mixed",
            quant_context={
                "market_regime_hint": "EXTREME_FEAR",
                "liquidity_ratio": 0.7,
                "min_liquidity_ratio": 0.35,
                "trend_against": True,
                "di_plus_4h": 20.0,
                "di_minus_4h": 24.0,
                "oversold_reversal": True,
                "microstructure_supportive": True,
                "funding_rate": -0.01,
                "return_24h": -0.03,
            },
        )
        self.assertEqual(insight.suggested_action, SuggestedAction.CLOSE)
        self.assertNotIn("calibrated_extreme_fear_reversal_open", insight.key_reason)

    def test_research_llm_treats_exploit_keyword_as_event_risk(self):
        class FakeClient:
            class _Chat:
                class _Completions:
                    def create(self, **kwargs):
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content=(
                                            '{"market_regime":"EXTREME_FEAR",'
                                            '"sentiment_score":-0.8,'
                                            '"confidence":0.2,'
                                            '"risk_warning":["event risk"],'
                                            '"key_reason":["trend down"],'
                                            '"suggested_action":"CLOSE"}'
                                        )
                                    )
                                )
                            ]
                        )

                completions = _Completions()

            chat = _Chat()

        analyzer = ResearchLLMAnalyzer(
            self.settings.llm,
            clients={"deepseek": FakeClient()},
        )
        insight = analyzer.analyze(
            symbol="ETH/USDT",
            timestamp=datetime.now(timezone.utc).isoformat(),
            news_summary="ETH protocol exploit escalates after funds were stolen",
            macro_summary="defensive",
            fear_greed=12.0,
            onchain_summary="mixed",
            quant_context={
                "market_regime_hint": "EXTREME_FEAR",
                "liquidity_ratio": 0.7,
                "min_liquidity_ratio": 0.35,
                "oversold_reversal": True,
                "microstructure_supportive": True,
                "funding_rate": -0.01,
                "return_24h": -0.03,
            },
        )
        self.assertEqual(insight.suggested_action, SuggestedAction.CLOSE)
        self.assertNotIn("calibrated_extreme_fear_reversal_open", insight.key_reason)

    def test_research_llm_ignores_unrelated_exploit_keyword_for_other_symbol(self):
        class FakeClient:
            class _Chat:
                class _Completions:
                    def create(self, **kwargs):
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content=(
                                            '{"market_regime":"DOWNTREND",'
                                            '"sentiment_score":-0.8,'
                                            '"confidence":0.2,'
                                            '"risk_warning":["macro stress"],'
                                            '"key_reason":["trend down"],'
                                            '"suggested_action":"CLOSE"}'
                                        )
                                    )
                                )
                            ]
                        )

                completions = _Completions()

            chat = _Chat()

        analyzer = ResearchLLMAnalyzer(
            self.settings.llm,
            clients={"deepseek": FakeClient()},
        )
        insight = analyzer.analyze(
            symbol="SOL/USDT",
            timestamp=datetime.now(timezone.utc).isoformat(),
            news_summary="ETH protocol exploit escalates after funds were stolen",
            macro_summary="defensive but stabilizing",
            fear_greed=12.0,
            onchain_summary="mixed",
            quant_context={
                "market_regime_hint": "EXTREME_FEAR",
                "liquidity_ratio": 0.7,
                "min_liquidity_ratio": 0.35,
                "trend_against": True,
                "di_plus_4h": 20.0,
                "di_minus_4h": 24.0,
                "oversold_reversal": True,
                "microstructure_supportive": True,
                "funding_rate": -0.01,
                "return_24h": -0.03,
            },
        )
        self.assertEqual(insight.suggested_action, SuggestedAction.OPEN_LONG)
        self.assertIn("calibrated_extreme_fear_reversal_open", insight.key_reason)

    def test_research_manager_uses_asset_aware_news_event_risk(self):
        manager = ResearchManager(self.settings)
        review = manager.review(
            symbol="SOL/USDT",
            insight=ResearchInsight(
                symbol="SOL/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=0.1,
                confidence=0.6,
                risk_warning=[],
                key_reason=[],
                suggested_action=SuggestedAction.OPEN_LONG,
            ),
            prediction=PredictionResult(
                symbol="SOL/USDT",
                up_probability=0.76,
                feature_count=12,
                model_version="test-model",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=SimpleNamespace(
                values={
                    "volume_ratio_1h": 1.0,
                    "adx_4h": 20.0,
                    "di_plus_4h": 24.0,
                    "di_minus_4h": 22.0,
                    "rsi_1h": 24.0,
                    "return_24h": -0.04,
                }
            ),
            fear_greed=12.0,
            news_summary="ETH exploit escalates after funds were stolen",
            onchain_summary="neutral",
            news_sources=["CoinDesk"],
            news_coverage_score=0.8,
            news_service_health_score=1.0,
            adaptive_min_liquidity_ratio=0.35,
        )
        self.assertNotIn("news_event_risk", review.reasons)
        self.assertIn("extreme_fear_offensive_setup", review.reasons)

    def test_risk_manager_paper_heat_grace_requires_paper_runtime_mode(self):
        manager = RiskManager(self.settings.risk, self.settings.strategy)
        self.assertFalse(
            manager._paper_portfolio_heat_grace(
                SimpleNamespace(
                    runtime_mode="live",
                    paper_canary_open_count=3,
                    recent_closed_trades=0,
                )
            )
        )
        self.assertTrue(
            manager._paper_portfolio_heat_grace(
                SimpleNamespace(
                    runtime_mode="paper",
                    paper_canary_open_count=3,
                    recent_closed_trades=0,
                )
            )
        )

    def test_research_llm_runtime_failure_enters_backoff(self):
        call_count = {"deepseek": 0}

        class FailingClient:
            class _Chat:
                class _Completions:
                    def create(self, **kwargs):
                        call_count["deepseek"] += 1
                        raise RuntimeError("temporary upstream failure")

                completions = _Completions()

            chat = _Chat()

        analyzer = ResearchLLMAnalyzer(
            self.settings.llm,
            clients={"deepseek": FailingClient()},
        )
        frozen_now = datetime(2026, 3, 28, tzinfo=timezone.utc)
        analyzer._now = lambda: frozen_now

        for _ in range(2):
            insight = analyzer.analyze(
                symbol="BTC/USDT",
                timestamp=frozen_now.isoformat(),
                news_summary="neutral",
                macro_summary="neutral",
                fear_greed=55.0,
                onchain_summary="neutral",
            )
            self.assertIn("fallback_research_model", insight.key_reason)

        self.assertEqual(call_count["deepseek"], 1)

    def test_openai_client_factory_ignores_host_proxy_environment(self):
        previous = {
            key: os.environ.get(key)
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
        }
        try:
            os.environ["HTTP_PROXY"] = "socks://127.0.0.1:7897"
            os.environ["HTTPS_PROXY"] = "socks://127.0.0.1:7897"
            os.environ["ALL_PROXY"] = "socks://127.0.0.1:7897"
            client = build_openai_client(
                api_key="test-key",
                base_url="https://api.deepseek.com/v1",
            )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.assertIsNotNone(client)

    def test_research_manager_downgrades_conflicted_open_long(self):
        manager = ResearchManager(self.settings)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 0.2,
                "adx_4h": 10.0,
                "di_plus_4h": 12.0,
                "di_minus_4h": 18.0,
            }
        )
        review = manager.review(
            symbol="BTC/USDT",
            insight=ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=-0.2,
                confidence=0.4,
                risk_warning=["stress"],
                key_reason=["fallback_research_model"],
                suggested_action=SuggestedAction.OPEN_LONG,
            ),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.51,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=False, reason="price_regime_conflict"),
            features=features,
            fear_greed=10.0,
            news_summary="BTC hit by liquidation risk after exploit",
            onchain_summary="On-chain data unavailable",
        )
        self.assertIn(review.reviewed_action, {"HOLD", "CLOSE"})
        self.assertIn(review.approval_rating, {"HOLD", "UNDERWEIGHT"})

    def test_research_manager_penalizes_negative_historical_setups(self):
        self.storage._insert_reflection(
            TradeReflection(
                trade_id="r1",
                symbol="BTC/USDT",
                direction="LONG",
                confidence=0.8,
                rationale="extreme fear breakdown with weak liquidity",
                outcome_24h=-2.5,
                correct_signals=[],
                wrong_signals=["liquidity was weak"],
                lesson="Avoid BTC longs in extreme fear when liquidity is weak.",
                market_regime=MarketRegime.EXTREME_FEAR,
            )
        )
        self.storage._insert_reflection(
            TradeReflection(
                trade_id="r2",
                symbol="BTC/USDT",
                direction="LONG",
                confidence=0.75,
                rationale="extreme fear bounce attempt under weak liquidity",
                outcome_24h=-1.8,
                correct_signals=[],
                wrong_signals=["price reversal failed"],
                lesson="Repeated extreme fear longs lost edge.",
                market_regime=MarketRegime.EXTREME_FEAR,
            )
        )

        manager = ResearchManager(self.settings, storage=self.storage)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 0.2,
                "adx_4h": 12.0,
                "di_plus_4h": 11.0,
                "di_minus_4h": 17.0,
            }
        )
        review = manager.review(
            symbol="BTC/USDT",
            insight=ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=-0.1,
                confidence=0.65,
                risk_warning=[],
                key_reason=["bounce_setup"],
                suggested_action=SuggestedAction.OPEN_LONG,
            ),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.66,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            fear_greed=10.0,
            news_summary="market remains fragile",
            onchain_summary="On-chain data unavailable",
            news_sources=["CoinDesk"],
            news_coverage_score=0.3,
            news_service_health_score=1.0,
        )
        self.assertIn("experience_negative_setup", review.reasons)
        self.assertTrue(review.experience_matches)
        self.assertIn(review.reviewed_action, {"HOLD", "CLOSE"})

    def test_research_manager_does_not_hard_block_mild_mixed_experience_setups(self):
        for trade_id, outcome, rationale in (
            ("r1", -0.14, "extreme fear bounce attempt under weak liquidity"),
            ("r2", 0.01, "extreme fear bounce attempt under weak liquidity"),
            ("r3", -0.13, "extreme fear bounce attempt under weak liquidity"),
        ):
            self.storage._insert_reflection(
                TradeReflection(
                    trade_id=trade_id,
                    symbol="BTC/USDT",
                    direction="LONG",
                    confidence=0.8,
                    rationale=rationale,
                    outcome_24h=outcome,
                    correct_signals=[],
                    wrong_signals=[],
                    lesson="Mixed follow-through on similar BTC bounce setup.",
                    market_regime=MarketRegime.EXTREME_FEAR,
                )
            )

        manager = ResearchManager(self.settings, storage=self.storage)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 0.2,
                "adx_4h": 12.0,
                "di_plus_4h": 11.0,
                "di_minus_4h": 17.0,
            }
        )
        review = manager.review(
            symbol="BTC/USDT",
            insight=ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=-0.1,
                confidence=0.65,
                risk_warning=[],
                key_reason=["bounce_setup"],
                suggested_action=SuggestedAction.OPEN_LONG,
            ),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.66,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            fear_greed=10.0,
            news_summary="market remains fragile",
            onchain_summary="On-chain data unavailable",
            news_sources=["CoinDesk"],
            news_coverage_score=0.3,
            news_service_health_score=1.0,
        )

        self.assertNotIn("experience_negative_setup", review.reasons)

    def test_experience_store_aggregates_setup_performance(self):
        profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="EXTREME_FEAR",
            validation_reason="ok",
            liquidity_ratio=0.9,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
        )
        tagged_rationale = "rating=HOLD; " + ExperienceStore.encode_setup_profile(profile)
        self.storage._insert_reflection(
            TradeReflection(
                trade_id="s1",
                symbol="BTC/USDT",
                direction="LONG",
                confidence=0.7,
                rationale=tagged_rationale,
                outcome_24h=-1.2,
                correct_signals=[],
                wrong_signals=[],
                lesson="setup underperformed",
                market_regime=MarketRegime.EXTREME_FEAR,
            )
        )
        self.storage._insert_reflection(
            TradeReflection(
                trade_id="s2",
                symbol="ETH/USDT",
                direction="LONG",
                confidence=0.6,
                rationale=tagged_rationale,
                outcome_24h=-0.8,
                correct_signals=[],
                wrong_signals=[],
                lesson="setup underperformed again",
                market_regime=MarketRegime.EXTREME_FEAR,
            )
        )
        stats = ExperienceStore(self.storage).aggregate_setup_performance(
            direction="LONG",
            setup_profile=profile,
        )
        self.assertEqual(stats["count"], 2)
        self.assertLess(stats["avg_outcome_24h"], 0.0)
        self.assertEqual(stats["negative_ratio"], 1.0)

    def test_experience_store_downweights_shadow_reflections(self):
        profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="EXTREME_FEAR",
            validation_reason="ok",
            liquidity_ratio=0.9,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
        )
        tagged_rationale = "rating=HOLD; " + ExperienceStore.encode_setup_profile(profile)
        self.storage._insert_reflection(
            TradeReflection(
                trade_id="real-1",
                symbol="BTC/USDT",
                direction="LONG",
                confidence=0.7,
                rationale=tagged_rationale,
                outcome_24h=-1.0,
                correct_signals=[],
                wrong_signals=[],
                lesson="real trade underperformed",
                market_regime=MarketRegime.EXTREME_FEAR,
            )
        )
        self.storage._insert_reflection(
            TradeReflection(
                trade_id="shadow:1",
                symbol="BTC/USDT",
                direction="LONG",
                confidence=0.7,
                rationale=tagged_rationale,
                outcome_24h=-1.0,
                correct_signals=[],
                wrong_signals=[],
                lesson="shadow trade underperformed",
                market_regime=MarketRegime.EXTREME_FEAR,
            )
        )

        stats = ExperienceStore(self.storage).aggregate_setup_performance(
            direction="LONG",
            setup_profile=profile,
        )

        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(float(stats["weighted_count"]), 1.35, places=2)
        self.assertLess(stats["avg_outcome_24h"], 0.0)

    def test_experience_store_scopes_setup_performance_by_symbol_and_news_bucket(self):
        btc_profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="EXTREME_FEAR",
            validation_reason="ok",
            liquidity_ratio=0.9,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
        )
        eth_profile = ExperienceStore.build_setup_profile(
            symbol="ETH/USDT",
            market_regime="EXTREME_FEAR",
            validation_reason="ok",
            liquidity_ratio=0.9,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=["CoinDesk", "Cointelegraph"],
            news_coverage_score=0.8,
            news_service_health_score=1.0,
        )
        self.storage._insert_reflection(
            TradeReflection(
                trade_id="btc-setup-1",
                symbol="BTC/USDT",
                direction="LONG",
                confidence=0.7,
                rationale="entry; " + ExperienceStore.encode_setup_profile(btc_profile),
                outcome_24h=-1.0,
                correct_signals=[],
                wrong_signals=[],
                lesson="btc setup underperformed",
                market_regime=MarketRegime.EXTREME_FEAR,
            )
        )
        self.storage._insert_reflection(
            TradeReflection(
                trade_id="eth-setup-1",
                symbol="ETH/USDT",
                direction="LONG",
                confidence=0.7,
                rationale="entry; " + ExperienceStore.encode_setup_profile(eth_profile),
                outcome_24h=2.0,
                correct_signals=[],
                wrong_signals=[],
                lesson="eth setup outperformed",
                market_regime=MarketRegime.EXTREME_FEAR,
            )
        )

        stats = ExperienceStore(self.storage).aggregate_setup_performance(
            direction="LONG",
            setup_profile=btc_profile,
        )

        self.assertEqual(stats["count"], 1)
        self.assertAlmostEqual(float(stats["avg_outcome_24h"]), -1.0, places=6)

    def test_research_manager_penalizes_negative_setup_expectancy(self):
        profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="EXTREME_FEAR",
            validation_reason="ok",
            liquidity_ratio=0.9,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
        )
        tagged_rationale = "rating=HOLD; " + ExperienceStore.encode_setup_profile(profile)
        for trade_id, symbol, outcome in (
            ("p1", "BTC/USDT", -1.1),
            ("p2", "ETH/USDT", -0.9),
        ):
            self.storage._insert_reflection(
                TradeReflection(
                    trade_id=trade_id,
                    symbol=symbol,
                    direction="LONG",
                    confidence=0.7,
                    rationale=tagged_rationale,
                    outcome_24h=outcome,
                    correct_signals=[],
                    wrong_signals=[],
                    lesson="setup lost edge",
                    market_regime=MarketRegime.EXTREME_FEAR,
                )
            )

        manager = ResearchManager(self.settings, storage=self.storage)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 0.9,
                "adx_4h": 26.0,
                "di_plus_4h": 20.0,
                "di_minus_4h": 14.0,
            }
        )
        review = manager.review(
            symbol="BTC/USDT",
            insight=ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=0.1,
                confidence=0.7,
                risk_warning=[],
                key_reason=["bounce_setup"],
                suggested_action=SuggestedAction.OPEN_LONG,
            ),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.74,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            fear_greed=10.0,
            news_summary="macro remains fragile",
            onchain_summary="On-chain data unavailable",
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
        )
        self.assertIn("setup_negative_expectancy", review.reasons)
        self.assertEqual(review.setup_performance["count"], 2)

    def test_research_manager_does_not_overweight_shadow_reflections(self):
        profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="EXTREME_FEAR",
            validation_reason="ok",
            liquidity_ratio=0.9,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
        )
        tagged_rationale = "rating=HOLD; " + ExperienceStore.encode_setup_profile(profile)
        for trade_id in ("shadow:1", "shadow:2"):
            self.storage._insert_reflection(
                TradeReflection(
                    trade_id=trade_id,
                    symbol="BTC/USDT",
                    direction="LONG",
                    confidence=0.7,
                    rationale=tagged_rationale,
                    outcome_24h=-1.1,
                    correct_signals=[],
                    wrong_signals=[],
                    lesson="shadow setup lost edge",
                    market_regime=MarketRegime.EXTREME_FEAR,
                )
            )

        manager = ResearchManager(self.settings, storage=self.storage)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 0.9,
                "adx_4h": 26.0,
                "di_plus_4h": 20.0,
                "di_minus_4h": 14.0,
            }
        )
        review = manager.review(
            symbol="BTC/USDT",
            insight=ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=0.1,
                confidence=0.7,
                risk_warning=[],
                key_reason=["bounce_setup"],
                suggested_action=SuggestedAction.OPEN_LONG,
            ),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.74,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            fear_greed=10.0,
            news_summary="macro remains fragile",
            onchain_summary="On-chain data unavailable",
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
        )

        self.assertNotIn("setup_negative_expectancy", review.reasons)
        self.assertAlmostEqual(
            float(review.setup_performance["weighted_count"]),
            0.7,
            places=2,
        )

    def test_research_manager_allows_strong_quant_override_in_extreme_fear(self):
        manager = ResearchManager(self.settings)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 1.2,
                "adx_4h": 28.0,
                "di_plus_4h": 24.0,
                "di_minus_4h": 12.0,
            }
        )
        review = manager.review(
            symbol="BTC/USDT",
            insight=ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=0.05,
                confidence=0.6,
                risk_warning=[],
                key_reason=["quant_reversal"],
                suggested_action=SuggestedAction.HOLD,
            ),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.82,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            fear_greed=10.0,
            news_summary="macro fragile but no fresh event shock",
            onchain_summary="On-chain data unavailable",
            news_sources=["CoinDesk"],
            news_coverage_score=0.4,
            news_service_health_score=1.0,
        )
        self.assertEqual(review.reviewed_action, "OPEN_LONG")
        self.assertIn("extreme_fear_quant_override", review.reasons)

    def test_research_manager_promotes_extreme_fear_offensive_setup(self):
        manager = ResearchManager(self.settings)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 0.42,
                "adx_4h": 23.0,
                "di_plus_4h": 25.0,
                "di_minus_4h": 21.0,
                "rsi_1h": 28.0,
                "return_24h": -0.038,
            }
        )
        review = manager.review(
            symbol="ETH/USDT",
            insight=ResearchInsight(
                symbol="ETH/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=-0.7,
                confidence=0.45,
                risk_warning=["Extreme fear and macro stress"],
                key_reason=["oversold_reversal"],
                suggested_action=SuggestedAction.OPEN_LONG,
            ),
            prediction=PredictionResult(
                symbol="ETH/USDT",
                up_probability=0.58,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            fear_greed=12.0,
            news_summary="macro fragile but no fresh event shock",
            onchain_summary="On-chain activity mixed",
            news_sources=["CoinDesk"],
            news_coverage_score=0.4,
            news_service_health_score=1.0,
            adaptive_min_liquidity_ratio=0.35,
        )
        self.assertEqual(review.reviewed_action, "OPEN_LONG")
        self.assertIn("extreme_fear_offensive_setup", review.reasons)
        self.assertIn("extreme_fear_offensive_open", review.reasons)
        self.assertIn("regime_extreme_fear_discounted", review.reasons)
        self.assertGreaterEqual(review.review_score, 0.15)

    def test_research_manager_discounts_trend_against_for_early_extreme_fear_reversal(self):
        manager = ResearchManager(self.settings)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 0.50,
                "adx_4h": 16.0,
                "di_plus_4h": 20.0,
                "di_minus_4h": 24.0,
                "rsi_1h": 29.0,
                "return_24h": -0.032,
            }
        )
        review = manager.review(
            symbol="ETH/USDT",
            insight=ResearchInsight(
                symbol="ETH/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=-0.6,
                confidence=0.45,
                risk_warning=["macro stress"],
                key_reason=["early_reversal"],
                suggested_action=SuggestedAction.OPEN_LONG,
            ),
            prediction=PredictionResult(
                symbol="ETH/USDT",
                up_probability=0.59,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            fear_greed=12.0,
            news_summary="macro fragile but no fresh event shock",
            onchain_summary="On-chain activity mixed",
            news_sources=["CoinDesk"],
            news_coverage_score=0.4,
            news_service_health_score=1.0,
            adaptive_min_liquidity_ratio=0.35,
        )
        self.assertIn("trend_against_discounted", review.reasons)
        self.assertNotIn("trend_against", review.reasons)
        self.assertIn(review.reviewed_action, {"HOLD", "OPEN_LONG"})
        self.assertGreater(review.review_score, -0.15)

    def test_research_manager_promotes_quant_repairing_setup_to_open_long(self):
        settings = self.settings.model_copy(deep=True)
        settings.model.xgboost_probability_threshold = 0.70
        manager = ResearchManager(settings)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 0.55,
                "adx_4h": 18.0,
                "di_plus_4h": 19.0,
                "di_minus_4h": 24.0,
                "rsi_1h": 30.0,
                "return_24h": -0.031,
            }
        )
        review = manager.review(
            symbol="WLD/USDT",
            insight=ResearchInsight(
                symbol="WLD/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=-0.35,
                confidence=0.52,
                risk_warning=[],
                key_reason=["reversal_attempt"],
                suggested_action=SuggestedAction.HOLD,
            ),
            prediction=PredictionResult(
                symbol="WLD/USDT",
                up_probability=0.72,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            fear_greed=12.0,
            news_summary="macro fragile but no fresh event shock",
            onchain_summary="On-chain activity mixed",
            news_sources=["CoinDesk"],
            news_coverage_score=0.4,
            news_service_health_score=1.0,
            adaptive_min_liquidity_ratio=0.35,
        )
        self.assertEqual(review.reviewed_action, "OPEN_LONG")
        self.assertIn("quant_repairing_setup", review.reasons)
        self.assertIn("quant_repairing_setup_open", review.reasons)

    def test_research_manager_keeps_quant_repairing_setup_blocked_on_event_risk(self):
        manager = ResearchManager(self.settings)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 0.55,
                "adx_4h": 18.0,
                "di_plus_4h": 19.0,
                "di_minus_4h": 24.0,
                "rsi_1h": 30.0,
                "return_24h": -0.031,
            }
        )
        review = manager.review(
            symbol="WLD/USDT",
            insight=ResearchInsight(
                symbol="WLD/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=-0.35,
                confidence=0.52,
                risk_warning=[],
                key_reason=["reversal_attempt"],
                suggested_action=SuggestedAction.HOLD,
            ),
            prediction=PredictionResult(
                symbol="WLD/USDT",
                up_probability=0.72,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            fear_greed=12.0,
            news_summary="Worldcoin exploit escalates after funds were stolen",
            onchain_summary="On-chain activity mixed",
            news_sources=["CoinDesk"],
            news_coverage_score=0.4,
            news_service_health_score=1.0,
            adaptive_min_liquidity_ratio=0.35,
        )
        self.assertIn(review.reviewed_action, {"HOLD", "CLOSE"})
        self.assertNotIn("quant_repairing_setup_open", review.reasons)

    def test_research_manager_forces_open_long_for_strong_extreme_fear_quant_signal(self):
        settings = self.settings.model_copy(deep=True)
        settings.strategy.min_liquidity_ratio = 0.8
        manager = ResearchManager(settings)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 0.75,
                "adx_4h": 19.0,
                "di_plus_4h": 17.0,
                "di_minus_4h": 16.0,
            }
        )
        review = manager.review(
            symbol="BTC/USDT",
            insight=ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=0.05,
                confidence=0.6,
                risk_warning=[],
                key_reason=["quant_reversal"],
                suggested_action=SuggestedAction.HOLD,
            ),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.84,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            fear_greed=10.0,
            news_summary="macro fragile but no fresh event shock",
            onchain_summary="On-chain data unavailable",
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
        )
        self.assertEqual(review.reviewed_action, "OPEN_LONG")
        self.assertIn("extreme_fear_quant_override_open", review.reasons)

    def test_research_manager_keeps_extreme_fear_override_blocked_on_event_risk(self):
        manager = ResearchManager(self.settings)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 1.2,
                "adx_4h": 28.0,
                "di_plus_4h": 24.0,
                "di_minus_4h": 12.0,
            }
        )
        review = manager.review(
            symbol="BTC/USDT",
            insight=ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=0.05,
                confidence=0.6,
                risk_warning=[],
                key_reason=["quant_reversal"],
                suggested_action=SuggestedAction.HOLD,
            ),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.84,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            fear_greed=10.0,
            news_summary="BTC faces liquidation cascade after hack",
            onchain_summary="On-chain data unavailable",
            news_sources=["CoinDesk"],
            news_coverage_score=0.4,
            news_service_health_score=1.0,
        )
        self.assertIn(review.reviewed_action, {"HOLD", "CLOSE"})
        self.assertNotIn("extreme_fear_quant_override_open", review.reasons)

    def test_research_manager_uses_adaptive_liquidity_floor(self):
        manager = ResearchManager(self.settings)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 0.75,
                "adx_4h": 20.0,
                "di_plus_4h": 18.0,
                "di_minus_4h": 16.0,
            }
        )
        review = manager.review(
            symbol="BTC/USDT",
            insight=ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.UNKNOWN,
                sentiment_score=0.0,
                confidence=0.6,
                risk_warning=[],
                key_reason=["neutral"],
                suggested_action=SuggestedAction.HOLD,
            ),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.70,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            adaptive_min_liquidity_ratio=0.70,
        )
        self.assertNotIn("liquidity_weak", review.reasons)

    def test_risk_manager_uses_liquidity_floor_override(self):
        from strategy.risk_manager import RiskManager
        from core.models import AccountState

        settings = self.settings.model_copy(deep=True)
        settings.strategy.min_liquidity_ratio = 0.80
        manager = RiskManager(settings.risk, settings.strategy)
        account = AccountState(
            equity=10000.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            daily_loss_pct=0.0,
            weekly_loss_pct=0.0,
            drawdown_pct=0.0,
            total_exposure_pct=0.0,
            open_positions=0,
        )
        result = manager.can_open_position(
            account=account,
            positions=[],
            symbol="BTC/USDT",
            atr=2.0,
            entry_price=100.0,
            liquidity_ratio=0.75,
            liquidity_floor_override=0.70,
        )
        self.assertTrue(result.allowed)
        self.assertAlmostEqual(result.liquidity_floor_used, 0.70, places=6)

    def test_reflector_prefers_setup_regime_from_rationale(self):
        profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="EXTREME_FEAR",
            validation_reason="ok",
            liquidity_ratio=0.8,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=["CoinDesk"],
            news_coverage_score=0.6,
            news_service_health_score=1.0,
        )
        reflection = TradeReflector(self.storage)._rule_based_reflection(
            trade_id="rx1",
            symbol="BTC/USDT",
            direction="LONG",
            confidence=0.8,
            rationale="entry; " + ExperienceStore.encode_setup_profile(profile),
            pnl=10.0,
            pnl_pct=10.0,
        )
        self.assertEqual(reflection.market_regime, MarketRegime.EXTREME_FEAR)

    def test_reflector_persists_explicit_trade_source_and_weight(self):
        reflection = TradeReflector(self.storage)._rule_based_reflection(
            trade_id="trade-1",
            symbol="BTC/USDT",
            direction="LONG",
            confidence=0.8,
            rationale="entry",
            pnl=10.0,
            pnl_pct=10.0,
        )

        self.assertEqual(reflection.source, "trade")
        self.assertEqual(reflection.experience_weight, 1.0)
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT source, experience_weight FROM reflections WHERE trade_id='trade-1'"
            ).fetchone()
        self.assertEqual(row["source"], "trade")
        self.assertAlmostEqual(float(row["experience_weight"]), 1.0, places=6)

    def test_strategy_evolver_suggests_tighter_runtime_overrides(self):
        weak_profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="EXTREME_FEAR",
            validation_reason="ok",
            liquidity_ratio=0.2,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
        )
        tagged_rationale = "entry; " + ExperienceStore.encode_setup_profile(weak_profile)
        for trade_id, symbol, outcome in (
            ("e1", "BTC/USDT", -1.4),
            ("e2", "ETH/USDT", -0.9),
        ):
            self.storage._insert_reflection(
                TradeReflection(
                    trade_id=trade_id,
                    symbol=symbol,
                    direction="LONG",
                    confidence=0.7,
                    rationale=tagged_rationale,
                    outcome_24h=outcome,
                    correct_signals=[],
                    wrong_signals=[],
                    lesson="negative setup",
                    market_regime=MarketRegime.EXTREME_FEAR,
                )
            )

        evolver = StrategyEvolver(self.storage, self.settings)
        suggestion = evolver.suggest_runtime_overrides(
            {
                "xgboost_probability_threshold": 0.68,
                "final_score_threshold": 0.50,
                "min_liquidity_ratio": 0.60,
                "sentiment_weight": 0.5,
                "fixed_stop_loss_pct": 0.05,
                "take_profit_levels": [0.05, 0.08],
            }
        )
        overrides = suggestion["runtime_overrides"]
        self.assertGreater(overrides["xgboost_probability_threshold"], 0.68)
        self.assertGreater(overrides["final_score_threshold"], 0.50)
        self.assertGreater(overrides["min_liquidity_ratio"], 0.60)
        self.assertTrue(suggestion["blocked_setups"])

    def test_strategy_evolver_downweights_shadow_reflections(self):
        weak_profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="EXTREME_FEAR",
            validation_reason="ok",
            liquidity_ratio=0.2,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
        )
        tagged_rationale = "entry; " + ExperienceStore.encode_setup_profile(weak_profile)
        for trade_id in ("shadow:1", "shadow:2"):
            self.storage._insert_reflection(
                TradeReflection(
                    trade_id=trade_id,
                    symbol="BTC/USDT",
                    direction="LONG",
                    confidence=0.7,
                    rationale=tagged_rationale,
                    outcome_24h=-1.2,
                    correct_signals=[],
                    wrong_signals=[],
                    lesson="shadow negative setup",
                    market_regime=MarketRegime.EXTREME_FEAR,
                )
            )

        evolver = StrategyEvolver(self.storage, self.settings)
        suggestion = evolver.suggest_runtime_overrides(
            {
                "xgboost_probability_threshold": 0.68,
                "final_score_threshold": 0.50,
                "min_liquidity_ratio": 0.60,
                "sentiment_weight": 0.5,
                "fixed_stop_loss_pct": 0.05,
                "take_profit_levels": [0.05, 0.08],
            }
        )

        self.assertEqual(suggestion["runtime_overrides"], {})
        self.assertEqual(suggestion["blocked_setups"], [])

    def test_strategy_evolver_bootstraps_from_prediction_accuracy(self):
        base_time = datetime.now(timezone.utc) - timedelta(hours=40)
        for idx in range(20):
            prediction_time = base_time + timedelta(hours=idx * 4)
            current_candle_ts = int((prediction_time - timedelta(minutes=1)).timestamp() * 1000)
            future_candle_ts = int((prediction_time + timedelta(hours=4, minutes=1)).timestamp() * 1000)
            self.storage.insert_ohlcv(
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
                        "open": 99.0,
                        "high": 100.0,
                        "low": 95.0,
                        "close": 96.0,
                        "volume": 1.5,
                    },
                ],
            )
            ts = prediction_time.isoformat()
            self.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": ts,
                    "model_version": "test",
                    "up_probability": 0.82,
                    "feature_count": 56,
                    "research": {
                        "symbol": "BTC/USDT",
                        "market_regime": "EXTREME_FEAR",
                        "sentiment_score": -0.2,
                        "confidence": 0.5,
                        "risk_warning": [],
                        "key_reason": ["liquidity_weak", "news_coverage_thin"],
                        "suggested_action": "HOLD",
                        "raw_content": "",
                        "timestamp": ts,
                    },
                    "decision": {
                        "final_score": 0.91,
                        "regime": "EXTREME_FEAR",
                        "suggested_action": "HOLD",
                        "xgboost_threshold": 0.68,
                    },
                }
            )

        evolver = StrategyEvolver(self.storage, self.settings)
        suggestion = evolver.suggest_runtime_overrides(
            {
                "xgboost_probability_threshold": 0.68,
                "final_score_threshold": 0.50,
                "min_liquidity_ratio": 0.60,
                "sentiment_weight": 0.10,
                "fixed_stop_loss_pct": 0.05,
                "take_profit_levels": [0.05, 0.08],
            }
        )
        overrides = suggestion["runtime_overrides"]
        self.assertGreater(overrides["xgboost_probability_threshold"], 0.68)
        self.assertGreater(overrides["final_score_threshold"], 0.50)
        self.assertEqual(overrides["sentiment_weight"], 0.0)
        self.assertEqual(suggestion["blocked_setups"], [])

    def test_strategy_evolver_tightens_from_shadow_observation_feedback(self):
        base_time = datetime.now(timezone.utc) - timedelta(hours=20)
        for idx in range(3):
            self.storage.insert_prediction_evaluation(
                {
                    "symbol": "SOL/USDT",
                    "timestamp": (base_time + timedelta(hours=idx * 4)).isoformat(),
                    "evaluation_type": "shadow_observation",
                    "actual_up": False,
                    "predicted_up": True,
                    "is_correct": False,
                    "entry_close": 100.0,
                    "future_close": 95.0,
                    "metadata": {
                        "regime": "EXTREME_FEAR",
                        "setup_profile": {
                            "regime": "EXTREME_FEAR",
                            "liquidity_bucket": "weak",
                            "news_bucket": "thin",
                            "validation": "ok",
                        },
                    },
                }
            )

        evolver = StrategyEvolver(self.storage, self.settings)
        suggestion = evolver.suggest_runtime_overrides(
            {
                "xgboost_probability_threshold": 0.68,
                "final_score_threshold": 0.50,
                "min_liquidity_ratio": 0.60,
                "sentiment_weight": 0.10,
                "fixed_stop_loss_pct": 0.05,
                "take_profit_levels": [0.05, 0.08],
            }
        )
        overrides = suggestion["runtime_overrides"]
        self.assertGreater(overrides["xgboost_probability_threshold"], 0.68)
        self.assertGreater(overrides["final_score_threshold"], 0.50)
        self.assertGreater(overrides["min_liquidity_ratio"], 0.60)
        self.assertEqual(overrides["sentiment_weight"], 0.0)
        self.assertGreaterEqual(
            suggestion["stats"].get("shadow_prediction_eval_count", 0),
            3,
        )

    def test_strategy_evolver_rehabilitates_negative_setup_with_shadow_trades(self):
        weak_profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="EXTREME_FEAR",
            validation_reason="ok",
            liquidity_ratio=0.2,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
        )
        tagged_rationale = "entry; " + ExperienceStore.encode_setup_profile(weak_profile)
        for trade_id, symbol, outcome in (
            ("e1", "BTC/USDT", -1.4),
            ("e2", "ETH/USDT", -0.9),
        ):
            self.storage._insert_reflection(
                TradeReflection(
                    trade_id=trade_id,
                    symbol=symbol,
                    direction="LONG",
                    confidence=0.7,
                    rationale=tagged_rationale,
                    outcome_24h=outcome,
                    correct_signals=[],
                    wrong_signals=[],
                    lesson="negative setup",
                    market_regime=MarketRegime.EXTREME_FEAR,
                )
            )
        for idx, pnl_pct in enumerate((4.0, 3.0, 2.0), start=1):
            self.storage.insert_shadow_trade_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": (datetime.now(timezone.utc) - timedelta(hours=idx * 4)).isoformat(),
                    "block_reason": "setup_auto_pause",
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "horizon_hours": 4,
                    "status": "evaluated",
                    "exit_price": 100.0 * (1.0 + pnl_pct / 100.0),
                    "pnl_pct": pnl_pct,
                    "setup_profile": weak_profile,
                    "metadata": {},
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        evolver = StrategyEvolver(self.storage, self.settings)
        suggestion = evolver.suggest_runtime_overrides(
            {
                "xgboost_probability_threshold": 0.68,
                "final_score_threshold": 0.50,
                "min_liquidity_ratio": 0.60,
                "sentiment_weight": 0.5,
                "fixed_stop_loss_pct": 0.05,
                "take_profit_levels": [0.05, 0.08],
            }
        )
        self.assertEqual(suggestion["blocked_setups"], [])
        self.assertEqual(
            suggestion["stats"].get("shadow_rehabilitated_setup_count"),
            1,
        )

    def test_strategy_evolver_generates_recent_symbol_level_setup_pause(self):
        profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="RANGE",
            validation_reason="ok",
            liquidity_ratio=0.75,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=["CoinDesk"],
            news_coverage_score=0.3,
            news_service_health_score=1.0,
        )
        tagged_rationale = "entry; " + ExperienceStore.encode_setup_profile(profile)
        for trade_id, outcome in (("p1", -0.24), ("p2", -0.18)):
            self.storage._insert_reflection(
                TradeReflection(
                    trade_id=trade_id,
                    symbol="BTC/USDT",
                    direction="LONG",
                    confidence=0.7,
                    rationale=tagged_rationale,
                    source="paper_canary",
                    realized_return_pct=outcome,
                    outcome_24h=outcome,
                    correct_signals=[],
                    wrong_signals=[],
                    lesson="recent negative setup",
                    market_regime=MarketRegime.RANGE,
                )
            )

        evolver = StrategyEvolver(self.storage, self.settings)
        suggestion = evolver.suggest_runtime_overrides(
            {
                "xgboost_probability_threshold": 0.68,
                "final_score_threshold": 0.50,
                "min_liquidity_ratio": 0.60,
                "sentiment_weight": 0.5,
                "fixed_stop_loss_pct": 0.05,
                "take_profit_levels": [0.05, 0.08],
            }
        )

        blocked = suggestion["blocked_setups"]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["criteria"]["symbol"], "BTC/USDT")
        self.assertEqual(blocked[0]["criteria"]["regime"], "RANGE")
        self.assertIn("recent_realized_setup_negative_expectancy", blocked[0]["reason"])
        self.assertEqual(suggestion["stats"]["recent_negative_setup_pause_count"], 1)
        self.assertIn("recent_negative_setup_pauses_1", suggestion["reasons"])

    def test_strategy_evolver_does_not_pause_recent_setup_when_drawdown_is_too_shallow(self):
        profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="EXTREME_FEAR",
            validation_reason="ok",
            liquidity_ratio=0.25,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=["CoinDesk"],
            news_coverage_score=0.3,
            news_service_health_score=1.0,
        )
        tagged_rationale = "entry; " + ExperienceStore.encode_setup_profile(profile)
        for trade_id, outcome in (
            ("m1", -0.24),
            ("m2", -0.20),
            ("m3", -0.17),
            ("m4", 0.00),
        ):
            self.storage._insert_reflection(
                TradeReflection(
                    trade_id=trade_id,
                    symbol="BTC/USDT",
                    direction="LONG",
                    confidence=0.7,
                    rationale=tagged_rationale,
                    source="paper_canary",
                    realized_return_pct=outcome,
                    outcome_24h=outcome,
                    correct_signals=[],
                    wrong_signals=[],
                    lesson="recent mixed negative setup",
                    market_regime=MarketRegime.EXTREME_FEAR,
                )
            )

        evolver = StrategyEvolver(self.storage, self.settings)
        suggestion = evolver.suggest_runtime_overrides(
            {
                "xgboost_probability_threshold": 0.68,
                "final_score_threshold": 0.50,
                "min_liquidity_ratio": 0.60,
                "sentiment_weight": 0.5,
                "fixed_stop_loss_pct": 0.05,
                "take_profit_levels": [0.05, 0.08],
            }
        )

        self.assertEqual(suggestion["blocked_setups"], [])
        self.assertEqual(suggestion["stats"]["recent_negative_setup_pause_count"], 0)

    def test_strategy_evolver_ignores_stale_recent_setup_pause_candidates(self):
        profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="RANGE",
            validation_reason="ok",
            liquidity_ratio=0.75,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=["CoinDesk"],
            news_coverage_score=0.3,
            news_service_health_score=1.0,
        )
        tagged_rationale = "entry; " + ExperienceStore.encode_setup_profile(profile)
        for trade_id, outcome in (("old1", -0.24), ("old2", -0.18)):
            self.storage._insert_reflection(
                TradeReflection(
                    trade_id=trade_id,
                    symbol="BTC/USDT",
                    direction="LONG",
                    confidence=0.7,
                    rationale=tagged_rationale,
                    source="paper_canary",
                    realized_return_pct=outcome,
                    outcome_24h=outcome,
                    correct_signals=[],
                    wrong_signals=[],
                    lesson="old negative setup",
                    market_regime=MarketRegime.RANGE,
                )
            )
        stale_created_at = (
            datetime.now(timezone.utc)
            - timedelta(hours=StrategyEvolver.RECENT_SETUP_PAUSE_WINDOW_HOURS + 6)
        ).isoformat()
        with self.storage._conn() as conn:
            conn.execute(
                "UPDATE reflections SET created_at=? WHERE trade_id IN ('old1', 'old2')",
                (stale_created_at,),
            )

        evolver = StrategyEvolver(self.storage, self.settings)
        suggestion = evolver.suggest_runtime_overrides(
            {
                "xgboost_probability_threshold": 0.68,
                "final_score_threshold": 0.50,
                "min_liquidity_ratio": 0.60,
                "sentiment_weight": 0.5,
                "fixed_stop_loss_pct": 0.05,
                "take_profit_levels": [0.05, 0.08],
            }
        )

        self.assertEqual(suggestion["blocked_setups"], [])
        self.assertEqual(suggestion["stats"]["recent_negative_setup_pause_count"], 0)

    def test_strategy_evolver_recent_pause_uses_partial_close_ledger_feedback(self):
        from execution.paper_trader import PaperTrader

        trader = PaperTrader(self.storage, initial_balance=10000.0)
        trader.execute_open(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.7,
            rationale=(
                "entry; setup_profile="
                "symbol:BTC/USDT|regime:RANGE|validation:ok|liquidity_bucket:mid|news_bucket:thin"
            ),
            quantity=1.0,
            metadata={"pipeline_mode": "paper_canary"},
        )
        trader.execute_close("BTC/USDT", 99.4, "research_de_risk", close_qty=0.5)
        trader.execute_close("BTC/USDT", 99.0, "research_exit")

        evolver = StrategyEvolver(self.storage, self.settings)
        suggestion = evolver.suggest_runtime_overrides(
            {
                "xgboost_probability_threshold": 0.68,
                "final_score_threshold": 0.50,
                "min_liquidity_ratio": 0.60,
                "sentiment_weight": 0.5,
                "fixed_stop_loss_pct": 0.05,
                "take_profit_levels": [0.05, 0.08],
            }
        )

        self.assertEqual(suggestion["stats"]["recent_negative_setup_pause_count"], 1)
        self.assertEqual(len(suggestion["blocked_setups"]), 1)

    def test_strategy_evolver_symbol_level_pause_is_not_cleared_by_other_symbol_shadow_rehab(self):
        profile = ExperienceStore.build_setup_profile(
            symbol="BTC/USDT",
            market_regime="EXTREME_FEAR",
            validation_reason="ok",
            liquidity_ratio=0.2,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
        )
        tagged_rationale = "entry; " + ExperienceStore.encode_setup_profile(profile)
        for trade_id, outcome in (("b1", -1.4), ("b2", -0.9)):
            self.storage._insert_reflection(
                TradeReflection(
                    trade_id=trade_id,
                    symbol="BTC/USDT",
                    direction="LONG",
                    confidence=0.7,
                    rationale=tagged_rationale,
                    source="paper_canary",
                    outcome_24h=outcome,
                    correct_signals=[],
                    wrong_signals=[],
                    lesson="negative setup",
                    market_regime=MarketRegime.EXTREME_FEAR,
                )
            )
        eth_profile = dict(profile)
        eth_profile["symbol"] = "ETH/USDT"
        for idx, pnl_pct in enumerate((4.0, 3.0, 2.0), start=1):
            self.storage.insert_shadow_trade_run(
                {
                    "symbol": "ETH/USDT",
                    "timestamp": (datetime.now(timezone.utc) - timedelta(hours=idx * 4)).isoformat(),
                    "block_reason": "setup_auto_pause",
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "horizon_hours": 4,
                    "status": "evaluated",
                    "exit_price": 100.0 * (1.0 + pnl_pct / 100.0),
                    "pnl_pct": pnl_pct,
                    "setup_profile": eth_profile,
                    "metadata": {},
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        evolver = StrategyEvolver(self.storage, self.settings)
        suggestion = evolver.suggest_runtime_overrides(
            {
                "xgboost_probability_threshold": 0.68,
                "final_score_threshold": 0.50,
                "min_liquidity_ratio": 0.60,
                "sentiment_weight": 0.5,
                "fixed_stop_loss_pct": 0.05,
                "take_profit_levels": [0.05, 0.08],
            }
        )

        blocked = suggestion["blocked_setups"]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["criteria"]["symbol"], "BTC/USDT")

    def test_research_manager_pauses_blocked_setup_open(self):
        manager = ResearchManager(self.settings, storage=self.storage)
        features = SimpleNamespace(
            values={
                "volume_ratio_1h": 0.2,
                "adx_4h": 26.0,
                "di_plus_4h": 20.0,
                "di_minus_4h": 14.0,
            }
        )
        review = manager.review(
            symbol="BTC/USDT",
            insight=ResearchInsight(
                symbol="BTC/USDT",
                market_regime=MarketRegime.EXTREME_FEAR,
                sentiment_score=0.1,
                confidence=0.7,
                risk_warning=[],
                key_reason=["bounce_setup"],
                suggested_action=SuggestedAction.OPEN_LONG,
            ),
            prediction=PredictionResult(
                symbol="BTC/USDT",
                up_probability=0.82,
                feature_count=10,
                model_version="test",
            ),
            validation=SimpleNamespace(ok=True, reason="ok"),
            features=features,
            fear_greed=10.0,
            news_summary="fragile macro",
            onchain_summary="On-chain data unavailable",
            news_sources=[],
            news_coverage_score=0.0,
            news_service_health_score=0.5,
            blocked_setups=[
                {
                    "criteria": {
                        "regime": "EXTREME_FEAR",
                        "liquidity_bucket": "weak",
                    },
                    "reason": "prediction_liquidity_weak_accuracy_0.43",
                    "mode": "pause_open",
                }
            ],
        )
        self.assertIn("setup_auto_pause", review.reasons)
        self.assertEqual(review.reviewed_action, "HOLD")

    def test_engine_locked_manual_runtime_overrides_take_precedence_over_learning(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        storage = Storage(self.db_path)
        storage.set_json_state(
            engine_module.CryptoAIV2Engine.RUNTIME_LEARNING_OVERRIDE_STATE_KEY,
            {
                "xgboost_probability_threshold": 0.73,
                "min_liquidity_ratio": 0.9,
            },
        )
        storage.set_json_state(
            engine_module.CryptoAIV2Engine.RUNTIME_OVERRIDE_STATE_KEY,
            {
                "xgboost_probability_threshold": 0.76,
            },
        )
        storage.set_json_state(
            engine_module.CryptoAIV2Engine.RUNTIME_LOCKED_FIELDS_STATE_KEY,
            ["xgboost_probability_threshold"],
        )

        engine = engine_module.CryptoAIV2Engine(settings)
        self.assertEqual(engine.settings.model.xgboost_probability_threshold, 0.76)
        self.assertEqual(engine.decision_engine.min_liquidity_ratio, 0.9)
        effective = engine.storage.get_json_state(
            engine_module.CryptoAIV2Engine.RUNTIME_EFFECTIVE_STATE_KEY,
            {},
        )
        self.assertEqual(effective["manual_overrides"]["xgboost_probability_threshold"], 0.76)
        self.assertEqual(effective["learning_overrides"]["min_liquidity_ratio"], 0.9)
        self.assertEqual(effective["overrides"]["xgboost_probability_threshold"], 0.76)
        self.assertEqual(effective["blocked_learning_overrides"]["xgboost_probability_threshold"], 0.73)
        self.assertEqual(effective["override_sources"]["xgboost_probability_threshold"], "manual_locked")
        self.assertEqual(effective["override_sources"]["min_liquidity_ratio"], "learning")

    def test_engine_learning_overrides_supersede_unlocked_manual(self):
        import core.engine as engine_module

        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        storage = Storage(self.db_path)
        storage.set_json_state(
            engine_module.CryptoAIV2Engine.RUNTIME_LEARNING_OVERRIDE_STATE_KEY,
            {
                "xgboost_probability_threshold": 0.73,
            },
        )
        storage.set_json_state(
            engine_module.CryptoAIV2Engine.RUNTIME_OVERRIDE_STATE_KEY,
            {
                "xgboost_probability_threshold": 0.76,
            },
        )

        with patch.object(
            engine_module.StrategyEvolver,
            "suggest_runtime_overrides",
            return_value={
                "runtime_overrides": {"xgboost_probability_threshold": 0.73},
                "reasons": [],
                "stats": {},
                "blocked_setups": [],
            },
        ):
            engine = engine_module.CryptoAIV2Engine(settings)
        effective = engine.storage.get_json_state(
            engine_module.CryptoAIV2Engine.RUNTIME_EFFECTIVE_STATE_KEY,
            {},
        )
        self.assertEqual(engine.settings.model.xgboost_probability_threshold, 0.73)
        self.assertEqual(
            effective["auto_superseded_manual_overrides"]["xgboost_probability_threshold"],
            0.76,
        )
        self.assertEqual(effective["override_sources"]["xgboost_probability_threshold"], "learning")

    def test_research_llm_parse_normalizes_non_unit_scores(self):
        analyzer = ResearchLLMAnalyzer(self.settings.llm, clients={})
        insight = analyzer._parse(
            """
            {
              "market_regime": "EXTREME_FEAR",
              "sentiment_score": 10,
              "confidence": 85,
              "risk_warning": "stress",
              "key_reason": "fear_greed",
              "suggested_action": "HOLD"
            }
            """,
            "BTC/USDT",
        )
        self.assertAlmostEqual(insight.sentiment_score, 0.1, places=6)
        self.assertAlmostEqual(insight.confidence, 0.85, places=6)
        self.assertEqual(insight.risk_warning, ["stress"])
        self.assertEqual(insight.key_reason, ["fear_greed"])

    def test_research_llm_parse_extracts_wrapped_json_object(self):
        analyzer = ResearchLLMAnalyzer(self.settings.llm, clients={})
        insight = analyzer._parse(
            """
            analysis complete:
            ```json
            {
              "market_regime": "UPTREND",
              "sentiment_score": 0.25,
              "confidence": 0.68,
              "risk_warning": [],
              "key_reason": ["quant_context_supportive"],
              "suggested_action": "OPEN_LONG"
            }
            ```
            """,
            "BTC/USDT",
        )
        self.assertEqual(insight.market_regime, MarketRegime.UPTREND)
        self.assertEqual(insight.suggested_action, SuggestedAction.OPEN_LONG)
        self.assertEqual(insight.key_reason, ["quant_context_supportive"])

    def test_news_and_macro_services_return_structured_summaries(self):
        news = NewsService()
        macro = MacroService()
        onchain = OnchainService()
        with patch.object(news.session, "get", side_effect=Exception("offline")):
            news_summary = news.get_summary("BTC/USDT")
        macro_summary = macro.get_summary(fear_greed=80)
        onchain_summary = onchain.get_summary("BTC/USDT")
        self.assertEqual(news_summary.symbol, "BTC/USDT")
        self.assertTrue(news_summary.summary)
        self.assertGreaterEqual(news_summary.coverage_score, 0.0)
        self.assertGreaterEqual(news_summary.service_health_score, 0.0)
        self.assertTrue(macro_summary.summary)
        self.assertTrue(onchain_summary.summary)

    def test_news_service_uses_jin10_headlines_when_available(self):
        class FakeResponse:
            text = """
            <html>
              <body>
                <div>金十快讯：比特币突破关键阻力位</div>
                <div>加密市场波动加剧，以太坊跟涨</div>
              </body>
            </html>
            """

            def raise_for_status(self):
                return None

        class FakeSession:
            def __init__(self):
                self.headers = {}

            def get(self, url, timeout=10):
                return FakeResponse()

        news = NewsService()
        news.session = FakeSession()
        summary = news.get_summary("BTC/USDT")
        self.assertIn("Jin10", summary.sources)
        self.assertIn("Jin10 headlines", summary.summary)

    def test_news_service_uses_cointelegraph_rss_when_available(self):
        class FakeResponse:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                return None

        class FakeSession:
            def __init__(self):
                self.headers = {}

            def get(self, url, timeout=10, params=None):
                if "cointelegraph" in url:
                    return FakeResponse(
                        """
                        <rss>
                          <channel>
                            <item><title>BTC rebounds as crypto market stabilizes</title></item>
                            <item><title>ETH ecosystem sees renewed capital inflow</title></item>
                          </channel>
                        </rss>
                        """
                    )
                return FakeResponse("<rss><channel></channel></rss>")

        news = NewsService()
        news.session = FakeSession()
        summary = news.get_summary("BTC/USDT")
        self.assertIn("Cointelegraph", summary.sources)
        self.assertIn("Cointelegraph headlines", summary.summary)

    def test_news_service_tracks_source_health_and_coverage(self):
        class FakeResponse:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                return None

        class FakeSession:
            def __init__(self):
                self.headers = {}

            def get(self, url, timeout=10, params=None):
                if "coindesk" in url:
                    return FakeResponse(
                        """
                        <rss>
                          <channel>
                            <item><title>Bitcoin rebounds as buyers step in</title></item>
                          </channel>
                        </rss>
                        """
                    )
                if "cointelegraph" in url:
                    raise RuntimeError("feed unavailable")
                if "jin10" in url:
                    return FakeResponse(
                        """
                        <html><body><div>Macro outlook improves for risk assets</div></body></html>
                        """
                    )
                raise RuntimeError("unexpected url")

        news = NewsService()
        news.session = FakeSession()
        summary = news.get_summary("BTC/USDT")
        self.assertEqual(summary.source_status["CoinDesk"], "matched")
        self.assertEqual(summary.source_status["Cointelegraph"], "unavailable")
        self.assertEqual(summary.source_status["Jin10"], "healthy_no_match")
        self.assertGreater(summary.coverage_score, 0.0)
        self.assertGreater(summary.service_health_score, 0.0)
        self.assertLess(summary.service_health_score, 1.0)

    def test_news_service_avoids_short_ticker_false_positives(self):
        class FakeResponse:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                return None

        class FakeSession:
            def __init__(self):
                self.headers = {}

            def get(self, url, timeout=10, params=None):
                if "cointelegraph" in url:
                    return FakeResponse(
                        """
                        <rss>
                          <channel>
                            <item><title>Top crypto stories for today</title></item>
                            <item><title>Macro outlook improves for risk assets</title></item>
                          </channel>
                        </rss>
                        """
                    )
                return FakeResponse("<rss><channel></channel></rss>")

        news = NewsService()
        news.session = FakeSession()
        summary = news.get_summary("OP/USDT")
        self.assertNotIn("Cointelegraph", summary.sources)
        self.assertIn("neutral news context", summary.summary)

    def test_news_service_avoids_short_ticker_false_positives_in_jin10(self):
        class FakeResponse:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                return None

        class FakeSession:
            def __init__(self):
                self.headers = {}

            def get(self, url, timeout=10, params=None):
                if "jin10" in url:
                    return FakeResponse(
                        """
                        <html>
                          <body>
                            <div>Top crypto stories for today</div>
                            <div>Macro outlook improves for risk assets</div>
                          </body>
                        </html>
                        """
                    )
                return FakeResponse("<rss><channel></channel></rss>")

        news = NewsService()
        news.session = FakeSession()
        summary = news.get_summary("OP/USDT")
        self.assertNotIn("Jin10", summary.sources)
        self.assertIn("neutral news context", summary.summary)

    def test_cross_validation_service_blocks_conflicting_sources(self):
        result = CrossValidationService().validate(
            symbol="BTC/USDT",
            news_sources=["CoinDesk"],
            fear_greed_value=80.0,
            lunarcrush_sentiment=-0.5,
            onchain_netflow_score=-0.3,
            regime_score=0.8,
            price_return_24h=-0.05,
        )
        self.assertFalse(result.ok)
        self.assertLess(result.consistency_score, 0.7)

    def test_cross_validation_service_allows_consistent_setups_with_thin_news(self):
        result = CrossValidationService().validate(
            symbol="BTC/USDT",
            news_sources=[],
            fear_greed_value=10.0,
            lunarcrush_sentiment=-0.4,
            onchain_netflow_score=-0.2,
            regime_score=-0.6,
            price_return_24h=-0.03,
            news_coverage_score=0.0,
            news_service_health_score=1.0,
        )
        self.assertTrue(result.ok)
        self.assertGreaterEqual(result.consistency_score, 0.7)

    def test_cross_validation_service_allows_regime_lagged_reversal_confirmation(self):
        result = CrossValidationService().validate(
            symbol="BTC/USDT",
            news_sources=["CoinDesk"],
            fear_greed_value=10.0,
            lunarcrush_sentiment=-0.4,
            onchain_netflow_score=12.0,
            regime_score=-1.0,
            price_return_24h=0.04,
        )
        self.assertTrue(result.ok)
        self.assertGreaterEqual(result.consistency_score, 0.7)
        self.assertIn(
            "regime_reversal_confirmation",
            result.details.get("reasons", []),
        )

    def test_research_input_consistency_service_blocks_degraded_external_context(self):
        result = ResearchInputConsistencyService().validate(
            symbol="BTC/USDT",
            news_summary=SimpleNamespace(
                sources=[],
                service_health_score=0.0,
                source_status={
                    "CoinDesk": "unavailable",
                    "Cointelegraph": "unavailable",
                    "Jin10": "unavailable",
                },
            ),
            onchain_summary=SimpleNamespace(source="fallback"),
            fear_greed_value=52.0,
            lunarcrush_sentiment=None,
        )

        self.assertFalse(result.ok)
        self.assertLess(result.consistency_score, 0.6)
        self.assertIn("news_services_unavailable", result.details["reasons"])
        self.assertIn("onchain_context_unavailable", result.details["reasons"])
        self.assertIn("external_context_thin", result.details["reasons"])

    def test_research_input_consistency_service_allows_thin_news_when_other_sources_exist(self):
        result = ResearchInputConsistencyService().validate(
            symbol="BTC/USDT",
            news_summary=SimpleNamespace(
                sources=[],
                service_health_score=1.0,
                source_status={"CoinDesk": "healthy_no_match"},
            ),
            onchain_summary=SimpleNamespace(source="coinmetrics"),
            fear_greed_value=55.0,
            lunarcrush_sentiment=None,
        )

        self.assertTrue(result.ok)
        self.assertNotIn("news_services_unavailable", result.details["reasons"])

    def test_sentiment_collector_merges_lunarcrush_when_available(self):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        class FakeSession:
            def __init__(self):
                self.headers = {}

            def get(self, url, timeout=10, headers=None):
                if "alternative.me" in url:
                    return FakeResponse(
                        {
                            "data": [
                                {
                                    "value": "60",
                                    "value_classification": "Greed",
                                }
                            ]
                        }
                    )
                return FakeResponse(
                    {
                        "data": {
                            "sentiment": 0.42,
                            "interactions_24h": 12345,
                        }
                    }
                )

        settings = get_settings().sentiment.model_copy(deep=True)
        settings.lunarcrush_api_key = SecretStr("token")
        from core.sentiment import SentimentCollector

        collector = SentimentCollector(self.storage, settings=settings, session=FakeSession())
        payload = collector.get_latest_sentiment("BTC/USDT")
        self.assertEqual(payload["value"], 60)
        self.assertAlmostEqual(payload["lunarcrush_sentiment"], 0.42)
        self.assertIn("LunarCrush BTC", payload["summary"])

    def test_onchain_service_uses_glassnode_when_configured(self):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        class FakeSession:
            def __init__(self):
                self.headers = {}

            def get(self, url, params=None, timeout=10):
                if "balance_exchanges_relative" in url:
                    return FakeResponse([{"t": 1, "v": 0.20}, {"t": 2, "v": 0.26}])
                return FakeResponse([{"t": 1, "v": 1200.0}, {"t": 2, "v": 1400.0}])

        settings = OnchainSettings()
        settings.glassnode_api_key = SecretStr("token")
        service = OnchainService(settings=settings, session=FakeSession())
        summary = service.get_summary("BTC/USDT")
        self.assertEqual(summary.source, "glassnode")
        self.assertIn("Glassnode", summary.summary)
        self.assertGreater(summary.netflow_score, 0.0)

    def test_onchain_service_uses_coinmetrics_when_glassnode_unavailable(self):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        class FakeSession:
            def __init__(self):
                self.headers = {}

            def get(self, url, params=None, timeout=10):
                if "exchange-asset-metrics" in url:
                    return FakeResponse(
                        {
                            "data": [
                                {"volume_reported_spot_usd_1d": "100.0"},
                                {"volume_reported_spot_usd_1d": "125.0"},
                            ]
                        }
                    )
                return FakeResponse(
                    {
                        "data": [
                            {"TxCnt": "50.0"},
                            {"TxCnt": "55.0"},
                        ]
                    }
                )

        settings = OnchainSettings()
        settings.coinmetrics_api_key = SecretStr("token")
        service = OnchainService(settings=settings, session=FakeSession())
        summary = service.get_summary("BTC/USDT")
        self.assertEqual(summary.source, "coinmetrics")
        self.assertIn("CoinMetrics", summary.summary)
        self.assertGreater(summary.netflow_score, 0.0)

    def test_onchain_service_uses_coinmetrics_community_without_api_key(self):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        class FakeSession:
            def __init__(self):
                self.headers = {}

            def get(self, url, params=None, timeout=10):
                return FakeResponse(
                    {
                        "data": [
                            {"TxCnt": "40.0", "AdrActCnt": "25.0"},
                            {"TxCnt": "48.0", "AdrActCnt": "30.0"},
                        ]
                    }
                )

        service = OnchainService(settings=OnchainSettings(), session=FakeSession())
        summary = service.get_summary("BTC/USDT")
        self.assertEqual(summary.source, "coinmetrics")
        self.assertIn("CoinMetrics", summary.summary)
        self.assertGreater(summary.netflow_score, 0.0)
