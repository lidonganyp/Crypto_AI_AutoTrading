"""Symbol snapshot and position-review runtime helpers."""
from __future__ import annotations

from datetime import datetime, timedelta
import inspect
import json
import time

from loguru import logger

from analysis.research_input_consistency import ResearchInputConsistencyService
from analysis.research_manager import ResearchManager
from config import Settings, get_settings
from core.feature_pipeline import FeatureInput, FeaturePipeline
from core.models import RiskCheckResult
from core.news_risk import (
    asset_aliases,
    contains_bearish_news_risk as shared_contains_bearish_news_risk,
    text_mentions_alias,
)
from core.storage import Storage


class SnapshotRuntimeService:
    """Build symbol analysis snapshots without keeping the logic embedded in engine."""

    REGIME_SCORE_MAP = {
        "BULL_TREND": 1.0,
        "BULL_CONSOL": 0.5,
        "BEAR_TREND": -1.0,
        "BEAR_RALLY": -0.3,
        "EXTREME_FEAR": -0.6,
        "EXTREME_GREED": 0.2,
        "UNKNOWN": 0.0,
    }

    def __init__(
        self,
        storage: Storage,
        settings: Settings | None = None,
        *,
        market,
        executor,
        sentiment,
        news,
        macro,
        onchain,
        regime_detector,
        research,
        feature_pipeline: FeaturePipeline,
        cross_validation,
        research_manager: ResearchManager,
        decision_engine,
        notifier,
        predictor_for_symbol,
        handle_model_unavailable,
        learning_details_state_key: str,
        register_market_data_failure,
        reset_market_data_failures,
        extend_cooldown_until,
    ):
        self.storage = storage
        self.settings = settings or get_settings()
        self.market = market
        self.executor = executor
        self.sentiment = sentiment
        self.news = news
        self.macro = macro
        self.onchain = onchain
        self.regime_detector = regime_detector
        self.research = research
        self.feature_pipeline = feature_pipeline
        self.cross_validation = cross_validation
        self.research_input_consistency = ResearchInputConsistencyService()
        self.research_manager = research_manager
        self.decision_engine = decision_engine
        self.notifier = notifier
        self._predictor_for_symbol = predictor_for_symbol
        self._handle_model_unavailable = handle_model_unavailable
        self.learning_details_state_key = learning_details_state_key
        self._register_market_data_failure = register_market_data_failure
        self._reset_market_data_failures = reset_market_data_failures
        self._extend_cooldown_until = extend_cooldown_until

    def prepare_symbol_snapshot(
        self,
        symbol: str,
        now: datetime,
        include_blocked: bool = False,
    ):
        market_symbol = self._market_symbol(symbol)
        if not self._check_symbol_market_latency(symbol, market_symbol):
            return None
        candles = self._fetch_candles(symbol, market_symbol, swallow_errors=False)
        if candles is None:
            return None
        quality_result = self.validate_market_data_quality(symbol, candles)
        if not quality_result["ok"]:
            return None
        self._reset_market_data_failures()

        funding_rate = None
        depth_snapshot = None
        if hasattr(self.market, "fetch_funding_rate"):
            funding_rate = self.market.fetch_funding_rate(market_symbol)
            if funding_rate is not None and abs(funding_rate) > 0.03:
                self.storage.insert_execution_event(
                    "funding_rate_block",
                    symbol,
                    {"funding_rate": funding_rate},
                )
                self.notifier.notify(
                    "funding_rate_block",
                    f"资金费率过热拦截: {symbol}",
                    f"funding_rate={funding_rate:.4f}",
                    level="warning",
                )
                return None
        depth_snapshot = self._summarize_depth_snapshot(market_symbol)

        news_summary = self.news.get_summary(symbol)
        if self.contains_bearish_news_risk(symbol, news_summary.summary):
            cooldown_until = now + timedelta(
                hours=self.settings.risk.news_risk_cooldown_hours
            )
            current_cooldown = self._extend_cooldown_until(cooldown_until)
            self.storage.insert_execution_event(
                "bearish_news_block",
                symbol,
                {
                    "reason": "bearish_news_detected",
                    "cooldown_until": current_cooldown.isoformat(),
                    "summary": news_summary.summary[:300],
                },
            )
            self.notifier.notify(
                "bearish_news_block",
                f"利空新闻拦截: {symbol}",
                f"cooldown_until={current_cooldown.isoformat()}\n{news_summary.summary[:300]}",
                level="warning",
            )
            return None

        bundle = self._analyze_symbol(
            symbol=symbol,
            now=now,
            candles_1h=candles["1h"],
            candles_4h=candles["4h"],
            candles_1d=candles["1d"],
            funding_rate=funding_rate,
            depth_snapshot=depth_snapshot,
            news_summary=news_summary,
        )
        if not bundle["ok"]:
            return None

        features = bundle["features"]
        prediction = bundle["prediction"]
        validation = bundle["validation"]
        review = bundle["review"]

        self.storage.insert_feature_snapshot(
            {
                "symbol": symbol,
                "timeframe": features.timeframe,
                "timestamp": features.timestamp.isoformat(),
                "features": features.values,
                "valid": features.valid,
            }
        )
        self.storage.insert_research_input(
            {
                "symbol": symbol,
                "timestamp": now.isoformat(),
                "news_summary": bundle["news_summary"].summary,
                "macro_summary": bundle["macro_summary"].summary,
                "fear_greed": bundle["fear_greed"],
                "onchain_summary": bundle["onchain_summary"].summary,
            }
        )
        self.storage.insert_execution_event(
            "cross_validation",
            symbol,
            {
                "ok": validation.ok,
                "consistency_score": validation.consistency_score,
                "reason": validation.reason,
                "details": validation.details,
            },
        )
        self.storage.insert_execution_event(
            "research_review",
            symbol,
            {
                "raw_action": review.raw_action,
                "reviewed_action": review.reviewed_action,
                "approval_rating": review.approval_rating,
                "review_score": review.review_score,
                "reasons": review.reasons,
                "experience_matches": review.experience_matches,
                "setup_profile": review.setup_profile,
                "setup_performance": review.setup_performance,
            },
        )
        if not validation.ok:
            self.notifier.notify(
                "cross_validation",
                f"多源校验未通过: {symbol}",
                f"score={validation.consistency_score:.2f}, reason={validation.reason}",
                level="warning",
            )
            if not include_blocked:
                return None
        return features, review.reviewed_insight, prediction, validation, review

    def prepare_position_review(self, symbol: str, now: datetime) -> dict | None:
        if not hasattr(self.market, "fetch_historical_ohlcv"):
            return None
        market_symbol = self._market_symbol(symbol)
        candles = self._fetch_candles(symbol, market_symbol, swallow_errors=True)
        if candles is None:
            return None
        if not candles["1h"] or not candles["4h"] or not candles["1d"]:
            return {"ok": False, "reason": "missing_candles"}

        news_summary = self.news.get_summary(symbol)
        if self.contains_bearish_news_risk(symbol, news_summary.summary):
            news_summary = self.news.get_summary(symbol)
            return {
                "ok": False,
                "reason": "bearish_news_detected",
                "news_summary": news_summary.summary,
            }

        bundle = self._analyze_symbol(
            symbol=symbol,
            now=now,
            candles_1h=candles["1h"],
            candles_4h=candles["4h"],
            candles_1d=candles["1d"],
            funding_rate=None,
            depth_snapshot=None,
            news_summary=news_summary,
        )
        if not bundle["ok"]:
            return {"ok": False, "reason": bundle["reason"]}

        review = bundle["review"]
        review_risk = RiskCheckResult(
            allowed=True,
            allowed_position_value=1.0,
            stop_loss_pct=self.settings.strategy.fixed_stop_loss_pct,
            take_profit_levels=list(self.settings.strategy.take_profit_levels),
            trailing_stop_drawdown_pct=self.settings.strategy.trailing_stop_drawdown_pct,
        )
        _, decision = bundle["decision_engine"].evaluate_entry(
            symbol=symbol,
            prediction=bundle["prediction"],
            insight=review.reviewed_insight,
            features=bundle["features"],
            risk_result=review_risk,
        )
        return {
            "ok": True,
            "insight": review.reviewed_insight,
            "prediction": bundle["prediction"],
            "validation": bundle["validation"],
            "decision": decision,
            "news_summary": bundle["news_summary"].summary,
            "review": review,
        }

    def review_research_signal(
        self,
        symbol: str,
        insight,
        prediction,
        validation,
        features,
        fear_greed: float | None,
        news_summary: str,
        onchain_summary: str,
        news_sources: list[str],
        news_coverage_score: float,
        news_service_health_score: float,
    ):
        learning_details = self.storage.get_json_state(
            self.learning_details_state_key,
            {},
        ) or {}
        return self.research_manager.review(
            symbol=symbol,
            insight=insight,
            prediction=prediction,
            validation=validation,
            features=features,
            fear_greed=fear_greed,
            news_summary=news_summary,
            onchain_summary=onchain_summary,
            news_sources=news_sources,
            news_coverage_score=news_coverage_score,
            news_service_health_score=news_service_health_score,
            adaptive_min_liquidity_ratio=features.values.get(
                "adaptive_min_liquidity_ratio"
            ),
            blocked_setups=learning_details.get("blocked_setups", []),
        )

    def contains_bearish_news_risk(self, symbol: str, summary: str) -> bool:
        return shared_contains_bearish_news_risk(symbol, summary)

    def validate_market_data_quality(
        self,
        symbol: str,
        candles_by_timeframe: dict[str, list[dict]],
    ) -> dict:
        spacing = {"1h": 3600_000, "4h": 4 * 3600_000, "1d": 24 * 3600_000}
        max_missing_ratio = self.settings.exchange.data_quality_max_missing_ratio
        for timeframe, candles in candles_by_timeframe.items():
            if len(candles) < 3:
                continue
            expected = spacing[timeframe]
            missing_slots = 0
            total_slots = 0
            for left, right in zip(candles[:-1], candles[1:]):
                diff = int(right["timestamp"]) - int(left["timestamp"])
                if diff <= 0:
                    continue
                slots = max(int(round(diff / expected)) - 1, 0)
                total_slots += 1 + slots
                missing_slots += slots
            if total_slots <= 0:
                continue
            missing_ratio = missing_slots / total_slots
            if missing_ratio <= max_missing_ratio:
                continue
            self.storage.insert_execution_event(
                "data_quality_failure",
                symbol,
                {
                    "timeframe": timeframe,
                    "missing_ratio": missing_ratio,
                    "threshold": max_missing_ratio,
                },
            )
            self.notifier.notify(
                "data_quality_failure",
                f"数据质量异常: {symbol} {timeframe}",
                f"missing_ratio={missing_ratio:.4%}, threshold={max_missing_ratio:.4%}",
                level="error",
            )
            return {"ok": False, "missing_ratio": missing_ratio}
        return {"ok": True}

    @staticmethod
    def _asset_aliases(symbol: str) -> set[str]:
        return asset_aliases(symbol)

    @staticmethod
    def _text_mentions_alias(text: str, alias: str) -> bool:
        return text_mentions_alias(text, alias)

    @staticmethod
    def _market_symbol(symbol: str) -> str:
        return f"{symbol}:USDT" if ":USDT" not in symbol else symbol

    def _fetch_candles(
        self,
        symbol: str,
        market_symbol: str,
        *,
        swallow_errors: bool,
    ) -> dict[str, list[dict]] | None:
        retry_count = max(0, int(self.settings.exchange.market_data_retry_count))
        retry_delay = max(
            0.0,
            float(self.settings.exchange.market_data_retry_delay_seconds),
        )
        last_error: Exception | None = None

        def fetch_with_retry(timeframe: str) -> list[dict]:
            nonlocal last_error
            candles: list[dict] = []
            for attempt in range(retry_count + 1):
                try:
                    candles = self.market.fetch_historical_ohlcv(
                        market_symbol,
                        timeframe,
                        limit=240,
                    )
                except Exception as exc:
                    last_error = exc
                    candles = []
                if candles:
                    return candles
                if attempt < retry_count and retry_delay > 0:
                    time.sleep(retry_delay)
            return candles

        candles_1h = fetch_with_retry(self.settings.strategy.lower_timeframe)
        candles_4h = fetch_with_retry(self.settings.strategy.primary_timeframe)
        candles_1d = fetch_with_retry(self.settings.strategy.higher_timeframe)
        if last_error is not None and not (candles_1h and candles_4h and candles_1d):
            if swallow_errors:
                return None
            raise last_error
        if not candles_1h or not candles_4h or not candles_1d:
            if swallow_errors:
                return {"1h": candles_1h or [], "4h": candles_4h or [], "1d": candles_1d or []}
            logger.warning(f"Missing candle data for {symbol}")
            self._register_market_data_failure(f"missing_candles:{symbol}")
            return None
        return {"1h": candles_1h, "4h": candles_4h, "1d": candles_1d}

    def _check_symbol_market_latency(
        self,
        symbol: str,
        market_symbol: str,
    ) -> bool:
        if not hasattr(self.market, "measure_latency"):
            return True
        try:
            result = self.market.measure_latency(market_symbol)
        except Exception:
            return True
        latency_seconds = result.get("latency_seconds")
        if latency_seconds is None:
            return True
        latency_seconds = float(latency_seconds)
        if latency_seconds < float(self.settings.exchange.data_latency_warning_seconds):
            return True
        self.storage.insert_execution_event(
            "symbol_market_latency_block",
            symbol,
            {
                "latency_seconds": latency_seconds,
                "threshold": float(self.settings.exchange.data_latency_warning_seconds),
            },
        )
        self.notifier.notify(
            "market_latency",
            f"行情延迟拦截: {symbol}",
            (
                f"latency_seconds={latency_seconds:.3f}, "
                f"threshold={float(self.settings.exchange.data_latency_warning_seconds):.3f}"
            ),
            level="warning",
        )
        return False

    def _analyze_symbol(
        self,
        *,
        symbol: str,
        now: datetime,
        candles_1h: list[dict],
        candles_4h: list[dict],
        candles_1d: list[dict],
        funding_rate,
        depth_snapshot,
        news_summary=None,
    ) -> dict:
        sentiment = self.sentiment.get_latest_sentiment(symbol=symbol) or {}
        fear_greed = (
            float(sentiment.get("value"))
            if sentiment.get("value") is not None
            else None
        )
        sentiment_value = float(sentiment.get("value") or 0.0) / 100.0
        news_summary = news_summary or self.news.get_summary(symbol)
        macro_summary = self.macro.get_summary(fear_greed=fear_greed)
        onchain_summary = self.onchain.get_summary(symbol)
        regime = self.regime_detector.detect(candles_1d, fear_greed=fear_greed)
        features = self.feature_pipeline.build(
            FeatureInput(
                symbol=symbol,
                candles_1h=candles_1h,
                candles_4h=candles_4h,
                candles_1d=candles_1d,
                funding_rate=funding_rate,
                bid_ask_spread_pct=(
                    depth_snapshot.bid_ask_spread_pct if depth_snapshot else 0.0
                ),
                bid_notional_top5=(
                    depth_snapshot.bid_notional_top5 if depth_snapshot else 0.0
                ),
                ask_notional_top5=(
                    depth_snapshot.ask_notional_top5 if depth_snapshot else 0.0
                ),
                depth_imbalance=depth_snapshot.depth_imbalance if depth_snapshot else 0.0,
                large_order_net_notional=depth_snapshot.large_order_net_notional if depth_snapshot else 0.0,
                sentiment_value=sentiment_value,
                llm_sentiment_score=0.0,
                lunarcrush_sentiment=float(sentiment.get("lunarcrush_sentiment") or 0.0),
                market_regime_score=self.REGIME_SCORE_MAP.get(regime.state, 0.0),
                onchain_netflow_score=onchain_summary.netflow_score,
                onchain_whale_score=onchain_summary.whale_score,
            )
        )
        if not features.valid:
            logger.warning(f"Invalid features for {symbol}")
            return {"ok": False, "reason": "invalid_features"}

        adaptive_liquidity = self._adaptive_liquidity_floor(
            symbol=symbol,
            at=features.timestamp,
            base_floor=float(self.settings.strategy.min_liquidity_ratio),
            depth_snapshot=depth_snapshot,
        )
        features.values["adaptive_min_liquidity_ratio"] = float(
            adaptive_liquidity["floor"]
        )
        features.values["adaptive_liquidity_sample_count"] = float(
            adaptive_liquidity["sample_count"]
        )
        features.values["adaptive_liquidity_relief"] = float(
            max(
                0.0,
                float(self.settings.strategy.min_liquidity_ratio)
                - float(adaptive_liquidity["floor"]),
            )
        )
        research_input_validation = self.research_input_consistency.validate(
            symbol=symbol,
            news_summary=news_summary,
            onchain_summary=onchain_summary,
            fear_greed_value=fear_greed,
            lunarcrush_sentiment=self._optional_float(
                sentiment.get("lunarcrush_sentiment")
            ),
        )
        self.storage.insert_execution_event(
            "research_input_consistency",
            symbol,
            {
                "ok": research_input_validation.ok,
                "consistency_score": research_input_validation.consistency_score,
                "reason": research_input_validation.reason,
                "details": research_input_validation.details,
            },
        )
        if not research_input_validation.ok:
            self.notifier.notify(
                "research_input_consistency",
                f"研究输入一致性未通过: {symbol}",
                (
                    f"score={research_input_validation.consistency_score:.2f}, "
                    f"reason={research_input_validation.reason}"
                ),
                level="warning",
            )
            return {
                "ok": False,
                "reason": "research_input_consistency_failed",
            }

        insight = self._run_research_analysis(
            symbol=symbol,
            timestamp=now.isoformat(),
            news_summary=news_summary.summary,
            macro_summary=macro_summary.summary,
            fear_greed=fear_greed,
            onchain_summary=onchain_summary.summary,
            quant_context=self._research_quant_context(
                symbol=symbol,
                regime=regime,
                features=features,
                funding_rate=funding_rate,
                depth_snapshot=depth_snapshot,
            ),
        )
        features.values["llm_sentiment_score"] = float(
            getattr(insight, "sentiment_score", 0.0) or 0.0
        )

        predictor = self._predictor_for_symbol(symbol)
        prediction = predictor.predict(features)
        if str(getattr(prediction, "model_version", "") or "").startswith("fallback"):
            self._handle_model_unavailable(
                symbol=symbol,
                predictor=predictor,
                prediction=prediction,
                now=now,
            )
            return {"ok": False, "reason": "model_unavailable"}
        validation = self.cross_validation.validate(
            symbol=symbol,
            news_sources=getattr(news_summary, "sources", []),
            fear_greed_value=fear_greed,
            lunarcrush_sentiment=float(sentiment.get("lunarcrush_sentiment") or 0.0),
            onchain_netflow_score=onchain_summary.netflow_score,
            regime_score=self.REGIME_SCORE_MAP.get(regime.state, 0.0),
            price_return_24h=features.values.get("return_24h"),
            news_coverage_score=getattr(news_summary, "coverage_score", 0.0),
            news_service_health_score=getattr(
                news_summary,
                "service_health_score",
                0.0,
            ),
        )
        review = self.review_research_signal(
            symbol=symbol,
            insight=insight,
            prediction=prediction,
            validation=validation,
            features=features,
            fear_greed=fear_greed,
            news_summary=news_summary.summary,
            onchain_summary=onchain_summary.summary,
            news_sources=getattr(news_summary, "sources", []),
            news_coverage_score=getattr(news_summary, "coverage_score", 0.0),
            news_service_health_score=getattr(
                news_summary,
                "service_health_score",
                0.0,
            ),
        )
        return {
            "ok": True,
            "fear_greed": fear_greed,
            "news_summary": news_summary,
            "macro_summary": macro_summary,
            "onchain_summary": onchain_summary,
            "features": features,
            "prediction": prediction,
            "validation": validation,
            "review": review,
            "decision_engine": self.decision_engine,
        }

    @staticmethod
    def _optional_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _research_supports_parameter(self, parameter_name: str) -> bool:
        try:
            signature = inspect.signature(self.research.analyze)
        except (TypeError, ValueError):
            return True
        parameters = signature.parameters.values()
        return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters) or (
            parameter_name in signature.parameters
        )

    def _run_research_analysis(
        self,
        *,
        symbol: str,
        timestamp: str,
        news_summary: str,
        macro_summary: str,
        fear_greed: float | None,
        onchain_summary: str,
        quant_context: dict,
    ):
        kwargs = {
            "symbol": symbol,
            "timestamp": timestamp,
            "news_summary": news_summary,
            "macro_summary": macro_summary,
            "fear_greed": fear_greed,
            "onchain_summary": onchain_summary,
        }
        if self._research_supports_parameter("quant_context"):
            kwargs["quant_context"] = quant_context
        return self.research.analyze(**kwargs)

    def _research_quant_context(
        self,
        *,
        symbol: str,
        regime,
        features,
        funding_rate,
        depth_snapshot,
    ) -> dict[str, float | bool | str]:
        values = getattr(features, "values", {}) or {}
        di_plus = float(values.get("di_plus_4h", 0.0))
        di_minus = float(values.get("di_minus_4h", 0.0))
        adx = float(values.get("adx_4h", 0.0))
        liquidity_ratio = float(values.get("volume_ratio_1h", 0.0))
        liquidity_floor = float(
            values.get(
                "adaptive_min_liquidity_ratio",
                self.settings.strategy.min_liquidity_ratio,
            )
            or self.settings.strategy.min_liquidity_ratio
        )
        return_24h = float(values.get("return_24h", 0.0))
        rsi_1h = float(values.get("rsi_1h", 50.0))
        return {
            "symbol": symbol,
            "market_regime_hint": str(getattr(regime, "state", "UNKNOWN") or "UNKNOWN"),
            "return_1h": float(values.get("return_1h", 0.0)),
            "return_24h": return_24h,
            "return_7d": float(values.get("return_7d", 0.0)),
            "rsi_1h": rsi_1h,
            "rsi_4h": float(values.get("rsi_4h", 50.0)),
            "adx_4h": adx,
            "di_plus_4h": di_plus,
            "di_minus_4h": di_minus,
            "trend_alignment_score": float(values.get("trend_alignment_score", 0.0)),
            "price_structure_score": float(values.get("price_structure_score", 0.0)),
            "volatility_regime_score": float(values.get("volatility_regime_score", 0.0)),
            "liquidity_ratio": liquidity_ratio,
            "min_liquidity_ratio": liquidity_floor,
            "funding_rate": float(funding_rate or 0.0),
            "bid_ask_spread_pct": float(
                getattr(depth_snapshot, "bid_ask_spread_pct", 0.0) or 0.0
            ),
            "top5_depth_notional": float(values.get("top5_depth_notional", 0.0)),
            "depth_imbalance": float(values.get("depth_imbalance", 0.0)),
            "trend_supportive": adx >= 20.0 and di_plus >= di_minus,
            "trend_against": di_minus > di_plus + 2.0,
            "oversold_reversal": (
                return_24h <= -0.03
                and rsi_1h <= 35.0
                and di_plus >= di_minus - 4.0
            ),
            "momentum_breakdown": return_24h <= -0.05 and di_minus > di_plus,
            "microstructure_supportive": self._liquidity_microstructure_supportive(
                depth_snapshot
            ),
        }

    def _adaptive_liquidity_floor(
        self,
        *,
        symbol: str,
        at: datetime,
        base_floor: float,
        depth_snapshot=None,
    ) -> dict[str, float | int | str]:
        if not bool(self.settings.strategy.adaptive_liquidity_enabled):
            return {
                "floor": float(base_floor),
                "sample_count": 0,
                "source": "disabled",
            }
        with self.storage._conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp,
                       CAST(json_extract(features_json, '$.volume_ratio_1h') AS REAL) AS liquidity_ratio
                FROM feature_snapshots
                WHERE symbol = ?
                  AND valid = 1
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (
                    symbol,
                    int(self.settings.strategy.adaptive_liquidity_lookback_snapshots),
                ),
            ).fetchall()
        all_values: list[float] = []
        same_hour_values: list[float] = []
        for row in rows:
            ratio = row["liquidity_ratio"]
            if ratio is None:
                continue
            value = float(ratio)
            if value <= 0:
                continue
            all_values.append(value)
            try:
                timestamp = datetime.fromisoformat(str(row["timestamp"]))
            except ValueError:
                continue
            hour_distance = abs(timestamp.hour - at.hour)
            hour_distance = min(hour_distance, 24 - hour_distance)
            if hour_distance <= int(
                self.settings.strategy.adaptive_liquidity_hour_window_hours
            ):
                same_hour_values.append(value)

        min_samples = max(
            3,
            int(self.settings.strategy.adaptive_liquidity_same_hour_min_samples),
        )
        if len(same_hour_values) >= min_samples:
            sample = sorted(same_hour_values)
            source = "same_hour"
        elif len(all_values) >= min_samples:
            sample = sorted(all_values)
            source = "symbol_recent"
        else:
            return {
                "floor": float(base_floor),
                "sample_count": len(all_values),
                "source": "base",
            }

        percentile = float(self.settings.strategy.adaptive_liquidity_percentile)
        percentile = min(max(percentile, 0.0), 1.0)
        index = min(
            len(sample) - 1,
            max(0, int(round((len(sample) - 1) * percentile))),
        )
        candidate_floor = sample[index]
        effective_floor = min(
            float(base_floor),
            max(
                float(self.settings.strategy.adaptive_liquidity_floor_min_ratio),
                float(candidate_floor),
            ),
        )
        if self._liquidity_microstructure_supportive(depth_snapshot):
            effective_floor = max(
                float(self.settings.strategy.adaptive_liquidity_floor_min_ratio),
                effective_floor * 0.85,
            )
        return {
            "floor": effective_floor,
            "sample_count": len(sample),
            "source": source,
        }

    def _summarize_depth_snapshot(self, market_symbol: str):
        providers = []
        if hasattr(self.executor, "exchange") and hasattr(
            self.executor.exchange,
            "summarize_order_book_depth",
        ):
            providers.append(self.executor.exchange)
        if hasattr(self.market, "summarize_order_book_depth"):
            providers.append(self.market)
        for provider in providers:
            try:
                snapshot = provider.summarize_order_book_depth(market_symbol)
            except Exception:
                continue
            if snapshot is not None:
                return snapshot
        return None

    def _liquidity_microstructure_supportive(self, depth_snapshot) -> bool:
        if depth_snapshot is None:
            return False
        spread_pct = float(getattr(depth_snapshot, "bid_ask_spread_pct", 0.0) or 0.0)
        top5_depth = float(
            getattr(depth_snapshot, "bid_notional_top5", 0.0) or 0.0
        ) + float(getattr(depth_snapshot, "ask_notional_top5", 0.0) or 0.0)
        return (
            spread_pct
            <= float(self.settings.strategy.liquidity_microstructure_support_spread_pct)
            and top5_depth
            >= float(self.settings.strategy.liquidity_microstructure_support_depth_usd)
        )
