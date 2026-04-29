"""LLM research layer for structured market insights."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from loguru import logger
from openai import OpenAI

from config import LLMSettings
from core.models import MarketRegime, ResearchInsight, SuggestedAction
from core.news_risk import contains_bearish_news_risk
from core.openai_client_factory import build_openai_client


SYSTEM_PROMPT = """你是 CryptoAI v3 的研究层。
目标是判断这个标的是否具备可交易 alpha，而不是机械跟随宏观恐慌。
优先使用 quant_context 中的量价、趋势、流动性、资金费率和盘口深度判断 suggested_action。
news_summary、macro_summary、onchain_summary 主要用于修正风险与置信度。
当极端恐慌与量化证据冲突时：
- 若 quant_context 显示超跌反转、趋势修复、流动性/盘口深度支持，可以给出 OPEN_LONG，但必须保留风险警告。
- 若宏观/新闻风险与量化弱势同时成立，应给出 HOLD 或 CLOSE。
- 若只是宏观压力或情绪极端，但没有明确事件风险，且 quant_context 已出现 oversold_reversal / liquidity_supportive / trend_repairing，不要默认给 CLOSE，优先给 HOLD 或 OPEN_LONG。
如果需要输出自然语言短语，优先使用中文。
risk_warning 优先输出简短中文短语。
key_reason 优先输出简短、稳定、可复用的标签；如果必须写自然语言，优先使用中文，不要写长英文句子。
只返回一个 JSON 对象，不要 markdown，不要解释。"""

OUTPUT_SCHEMA = {
    "market_regime": "UPTREND|RANGE|DOWNTREND|EXTREME_FEAR|EXTREME_GREED|UNKNOWN",
    "sentiment_score": 0.0,
    "confidence": 0.0,
    "risk_warning": ["short warning"],
    "key_reason": ["short reason"],
    "suggested_action": "OPEN_LONG|HOLD|CLOSE",
}


class ResearchLLMAnalyzer:
    """Research-oriented LLM wrapper."""

    RUNTIME_FAILURE_BACKOFF_SECONDS = 300
    PROMPT_TEXT_LIMIT = 360

    def __init__(
        self,
        settings: LLMSettings,
        clients: dict[str, OpenAI] | None = None,
    ):
        self.settings = settings
        self.clients = clients.copy() if clients else {}
        self._disabled_sources: set[str] = set()
        self._runtime_backoff_until: dict[str, datetime] = {}

    def analyze(
        self,
        symbol: str,
        timestamp: str,
        news_summary: str,
        macro_summary: str,
        fear_greed: float | None,
        onchain_summary: str = "",
        quant_context: dict | None = None,
    ) -> ResearchInsight:
        providers = self._provider_names()
        if not providers:
            fallback = self._fallback(symbol, fear_greed, quant_context=quant_context)
            return self._calibrate_insight(
                fallback,
                symbol=symbol,
                fear_greed=fear_greed,
                quant_context=quant_context,
                news_summary=news_summary,
            )

        prompt = self._build_prompt(
            symbol=symbol,
            timestamp=timestamp,
            news_summary=news_summary,
            macro_summary=macro_summary,
            fear_greed=fear_greed,
            onchain_summary=onchain_summary,
            quant_context=quant_context,
        )

        for source in providers:
            if not self._source_available(source):
                continue
            client = self._client_for(source)
            if client is None:
                continue
            try:
                response = client.chat.completions.create(
                    model=self._model_for(source),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=320,
                )
                content = response.choices[0].message.content or ""
                if not content.strip():
                    logger.warning(f"Research LLM returned empty content ({source})")
                    continue
                insight = self._parse(content, symbol)
                calibrated = self._calibrate_insight(
                    insight,
                    symbol=symbol,
                    fear_greed=fear_greed,
                    quant_context=quant_context,
                    news_summary=news_summary,
                )
                calibrated.raw_content = content
                return calibrated
            except Exception as exc:
                self._runtime_backoff_until[source] = self._now() + timedelta(
                    seconds=self._runtime_failure_backoff_seconds()
                )
                logger.error(f"Research LLM failed ({source}): {exc}")

        fallback = self._fallback(symbol, fear_greed, quant_context=quant_context)
        return self._calibrate_insight(
            fallback,
            symbol=symbol,
            fear_greed=fear_greed,
            quant_context=quant_context,
            news_summary=news_summary,
        )

    def has_live_clients(self) -> bool:
        return bool(self._provider_names())

    def _provider_names(self) -> list[str]:
        providers: list[str] = []
        for source in ("deepseek", "qwen"):
            if source in self.clients or self._api_key_for(source):
                providers.append(source)
        for source in self.clients:
            if source not in providers:
                providers.append(source)
        return providers

    def _client_for(self, source: str) -> OpenAI | None:
        client = self.clients.get(source)
        if client is not None:
            return client
        if source in self._disabled_sources:
            return None
        try:
            client = self._build_client(source)
        except Exception as exc:
            logger.error(f"Research LLM client init failed ({source}): {exc}")
            self._disabled_sources.add(source)
            return None
        if client is None:
            return None
        self.clients[source] = client
        return client

    def _source_available(self, source: str) -> bool:
        backoff_until = self._runtime_backoff_until.get(source)
        if backoff_until is None:
            return True
        if self._now() >= backoff_until:
            self._runtime_backoff_until.pop(source, None)
            return True
        return False

    def _build_client(self, source: str) -> OpenAI | None:
        api_key = self._api_key_for(source)
        if not api_key:
            return None
        return build_openai_client(
            api_key=api_key,
            base_url=self._base_url_for(source),
            timeout_seconds=float(
                getattr(self.settings, "request_timeout_seconds", 18.0) or 18.0
            ),
            connect_timeout_seconds=float(
                getattr(self.settings, "connect_timeout_seconds", 5.0) or 5.0
            ),
        )

    def _api_key_for(self, source: str) -> str:
        if source == "deepseek":
            return self.settings.deepseek_api_key.get_secret_value()
        if source == "qwen":
            return self.settings.qwen_api_key.get_secret_value()
        return ""

    def _base_url_for(self, source: str) -> str:
        if source == "deepseek":
            return self.settings.deepseek_api_base
        if source == "qwen":
            return self.settings.qwen_api_base
        return ""

    def _model_for(self, source: str) -> str:
        if source == "deepseek":
            return self.settings.deepseek_model
        if source == "qwen":
            return self.settings.qwen_model
        return source

    def _runtime_failure_backoff_seconds(self) -> int:
        configured = int(
            getattr(
                self.settings,
                "runtime_failure_backoff_seconds",
                self.RUNTIME_FAILURE_BACKOFF_SECONDS,
            )
            or self.RUNTIME_FAILURE_BACKOFF_SECONDS
        )
        return max(15, configured)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @classmethod
    def _normalize_sentiment_score(cls, value: object) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if abs(score) > 1.0 and abs(score) <= 100.0:
            score /= 100.0
        return cls._clamp(score, -1.0, 1.0)

    @classmethod
    def _normalize_confidence(cls, value: object) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        if confidence > 1.0 and confidence <= 100.0:
            confidence /= 100.0
        return cls._clamp(confidence, 0.0, 1.0)

    @staticmethod
    def _normalize_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @classmethod
    def _compact_text(cls, value: str | None) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            return "none"
        if len(text) <= cls.PROMPT_TEXT_LIMIT:
            return text
        return text[: cls.PROMPT_TEXT_LIMIT - 3].rstrip() + "..."

    @classmethod
    def _clean_quant_context(cls, quant_context: dict | None) -> dict:
        if not isinstance(quant_context, dict):
            return {}
        cleaned: dict[str, object] = {}
        for key, value in quant_context.items():
            key_text = str(key)
            if isinstance(value, bool):
                cleaned[key_text] = value
                continue
            if isinstance(value, (int, float)):
                cleaned[key_text] = round(float(value), 6)
                continue
            if value is None:
                continue
            text = str(value).strip()
            if text:
                cleaned[key_text] = text
        return cleaned

    def _build_prompt(
        self,
        *,
        symbol: str,
        timestamp: str,
        news_summary: str,
        macro_summary: str,
        fear_greed: float | None,
        onchain_summary: str,
        quant_context: dict | None,
    ) -> str:
        payload = {
            "symbol": symbol,
            "timestamp": timestamp,
            "news_summary": self._compact_text(news_summary),
            "news_event_risk": self._news_event_risk(symbol, news_summary),
            "macro_summary": self._compact_text(macro_summary),
            "fear_greed_index": fear_greed if fear_greed is not None else "unknown",
            "onchain_summary": self._compact_text(onchain_summary),
            "quant_context": self._clean_quant_context(quant_context),
        }
        return (
            "请基于 input_json 返回一个严格 JSON 对象。\n"
            "要求: 不要 markdown，不要解释；risk_warning 和 key_reason 保持简洁；"
            "自然语言内容优先中文；若使用规则标签，尽量保持短标签、可复用；"
            "suggested_action 只能是 OPEN_LONG/HOLD/CLOSE。\n"
            f"input_json={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
            f"output_schema={json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, separators=(',', ':'))}"
        )

    @staticmethod
    def _extract_json_text(content: str) -> str:
        candidate = content.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[1]
        if candidate.endswith("```"):
            candidate = candidate.rsplit("```", 1)[0]
        candidate = candidate.strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                snippet = candidate[start : end + 1].strip()
                json.loads(snippet)
                return snippet
            raise

    def _parse(self, content: str, symbol: str) -> ResearchInsight:
        content = self._extract_json_text(content)
        data = json.loads(content)
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            raise ValueError("research output is not a JSON object")
        try:
            regime = MarketRegime(str(data.get("market_regime", "UNKNOWN")))
        except ValueError:
            regime = MarketRegime.UNKNOWN
        try:
            action = SuggestedAction(str(data.get("suggested_action", "HOLD")))
        except ValueError:
            action = SuggestedAction.HOLD
        return ResearchInsight(
            symbol=symbol,
            market_regime=regime,
            sentiment_score=self._normalize_sentiment_score(
                data.get("sentiment_score", 0.0)
            ),
            confidence=self._normalize_confidence(data.get("confidence", 0.0)),
            risk_warning=self._normalize_list(data.get("risk_warning", [])),
            key_reason=self._normalize_list(data.get("key_reason", [])),
            suggested_action=action,
            raw_content=content,
        )

    @staticmethod
    def _news_event_risk(symbol: str, news_summary: str | None) -> bool:
        return contains_bearish_news_risk(symbol, news_summary)

    @classmethod
    def _calibrate_insight(
        cls,
        insight: ResearchInsight,
        *,
        symbol: str,
        fear_greed: float | None,
        quant_context: dict | None,
        news_summary: str,
    ) -> ResearchInsight:
        quant = cls._clean_quant_context(quant_context)
        if not quant:
            return insight
        if cls._news_event_risk(symbol, news_summary):
            return insight

        regime_hint = str(quant.get("market_regime_hint", "") or "").upper()
        extreme_fear_context = (
            (fear_greed is not None and fear_greed <= 15)
            or insight.market_regime == MarketRegime.EXTREME_FEAR
            or regime_hint == "EXTREME_FEAR"
        )
        if not extreme_fear_context:
            return insight

        quant_score = cls._quant_calibration_score(quant)
        if quant_score <= 0.0:
            return insight

        new_action = insight.suggested_action
        calibration_reason = ""
        if insight.suggested_action == SuggestedAction.CLOSE:
            if quant_score >= 0.28:
                new_action = SuggestedAction.OPEN_LONG
                calibration_reason = "calibrated_extreme_fear_reversal_open"
            elif quant_score >= 0.10:
                new_action = SuggestedAction.HOLD
                calibration_reason = "calibrated_extreme_fear_reversal_hold"
        elif (
            insight.suggested_action == SuggestedAction.HOLD
            and quant_score >= 0.32
        ):
            new_action = SuggestedAction.OPEN_LONG
            calibration_reason = "calibrated_extreme_fear_hold_to_open"

        if new_action == insight.suggested_action:
            return insight

        new_regime = insight.market_regime
        if new_action == SuggestedAction.OPEN_LONG and regime_hint == "EXTREME_FEAR":
            new_regime = MarketRegime.EXTREME_FEAR
        elif new_action != SuggestedAction.CLOSE and insight.market_regime == MarketRegime.DOWNTREND:
            new_regime = MarketRegime.EXTREME_FEAR if regime_hint == "EXTREME_FEAR" else MarketRegime.RANGE

        base_sentiment = float(insight.sentiment_score or 0.0)
        calibrated_sentiment = base_sentiment
        if new_action == SuggestedAction.OPEN_LONG:
            calibrated_sentiment = max(base_sentiment, min(0.18, quant_score))
        elif new_action == SuggestedAction.HOLD:
            calibrated_sentiment = max(base_sentiment, -0.05)

        calibrated_confidence = float(insight.confidence or 0.0)
        if new_action == SuggestedAction.OPEN_LONG:
            calibrated_confidence = max(calibrated_confidence, 0.42)
        elif new_action == SuggestedAction.HOLD:
            calibrated_confidence = max(calibrated_confidence, 0.35)

        return insight.model_copy(
            update={
                "market_regime": new_regime,
                "sentiment_score": cls._clamp(calibrated_sentiment, -1.0, 1.0),
                "confidence": cls._clamp(calibrated_confidence, 0.0, 1.0),
                "suggested_action": new_action,
                "key_reason": list(insight.key_reason) + [calibration_reason],
            }
        )

    @classmethod
    def _quant_calibration_score(cls, quant: dict) -> float:
        liquidity_ratio = cls._quant_float(quant.get("liquidity_ratio"))
        liquidity_floor = cls._quant_float(quant.get("min_liquidity_ratio"), 0.8)
        funding_rate = cls._quant_float(quant.get("funding_rate"))
        return_24h = cls._quant_float(quant.get("return_24h"))
        di_plus = cls._quant_float(quant.get("di_plus_4h"))
        di_minus = cls._quant_float(quant.get("di_minus_4h"))
        score = 0.0
        if cls._quant_flag(quant.get("oversold_reversal")):
            score += 0.22
        if liquidity_ratio >= max(0.15, liquidity_floor * 0.75):
            score += 0.08
        else:
            score -= 0.12
        if cls._quant_flag(quant.get("microstructure_supportive")):
            score += 0.08
        if cls._quant_flag(quant.get("trend_supportive")):
            score += 0.10
        elif cls._quant_flag(quant.get("trend_against")):
            if (di_minus - di_plus) <= 6.0 and cls._quant_flag(quant.get("oversold_reversal")):
                score += 0.02
            else:
                score -= 0.12
        if cls._quant_flag(quant.get("momentum_breakdown")):
            score -= 0.12
        if funding_rate <= -0.005 and cls._quant_flag(quant.get("oversold_reversal")):
            score += 0.04
        elif funding_rate >= 0.015:
            score -= 0.08
        if return_24h <= -0.08 and not cls._quant_flag(quant.get("oversold_reversal")):
            score -= 0.05
        return score

    @staticmethod
    def _quant_flag(value: object) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        return text in {"1", "true", "yes", "y"}

    @staticmethod
    def _quant_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _fallback(
        cls,
        symbol: str,
        fear_greed: float | None,
        quant_context: dict | None = None,
    ) -> ResearchInsight:
        sentiment = 0.0
        regime = MarketRegime.UNKNOWN
        warnings: list[str] = []
        reasons: list[str] = ["fallback_research_model"]
        action = SuggestedAction.HOLD
        score = 0.0
        quant = cls._clean_quant_context(quant_context)

        regime_hint = str(quant.get("market_regime_hint", "") or "").upper()
        if regime_hint in MarketRegime._value2member_map_:
            regime = MarketRegime(regime_hint)

        if fear_greed is not None:
            if fear_greed <= 15:
                score -= 0.05
                regime = (
                    MarketRegime.EXTREME_FEAR
                    if regime == MarketRegime.UNKNOWN
                    else regime
                )
                warnings.append("extreme fear")
            elif fear_greed >= 85:
                score += 0.12
                regime = (
                    MarketRegime.EXTREME_GREED
                    if regime == MarketRegime.UNKNOWN
                    else regime
                )
                warnings.append("extreme greed")
            elif fear_greed >= 55:
                score += 0.18
                regime = MarketRegime.UPTREND if regime == MarketRegime.UNKNOWN else regime
            elif regime == MarketRegime.UNKNOWN:
                regime = MarketRegime.RANGE

        liquidity_ratio = cls._quant_float(quant.get("liquidity_ratio"))
        liquidity_floor = cls._quant_float(quant.get("min_liquidity_ratio"), 0.8)
        funding_rate = cls._quant_float(quant.get("funding_rate"))
        return_24h = cls._quant_float(quant.get("return_24h"))
        if quant:
            if liquidity_ratio >= max(0.15, liquidity_floor * 0.75):
                score += 0.08
                reasons.append("fallback_liquidity_supportive")
            else:
                score -= 0.15
                warnings.append("liquidity weak")
                reasons.append("fallback_liquidity_weak")

            if cls._quant_flag(quant.get("microstructure_supportive")):
                score += 0.08
                reasons.append("fallback_microstructure_supportive")

            if cls._quant_flag(quant.get("trend_supportive")):
                score += 0.14
                reasons.append("fallback_trend_supportive")
            if cls._quant_flag(quant.get("trend_against")):
                score -= 0.16
                warnings.append("trend against")
                reasons.append("fallback_trend_against")

            if cls._quant_flag(quant.get("oversold_reversal")):
                score += 0.18
                reasons.append("fallback_oversold_reversal")
            if cls._quant_flag(quant.get("momentum_breakdown")):
                score -= 0.10
                reasons.append("fallback_momentum_breakdown")

            if funding_rate >= 0.015:
                score -= 0.08
                warnings.append("funding overheated")
                reasons.append("fallback_funding_overheated")
            elif funding_rate <= -0.005 and cls._quant_flag(
                quant.get("oversold_reversal")
            ):
                score += 0.04
                reasons.append("fallback_short_crowding_support")

            if return_24h <= -0.08:
                warnings.append("high downside volatility")
            elif return_24h >= 0.04:
                score += 0.04
                reasons.append("fallback_positive_momentum")

        sentiment = cls._clamp(score, -0.85, 0.85)
        confidence = cls._clamp(
            0.35 + abs(score) * 0.7 + (0.05 if quant else 0.0),
            0.25,
            0.78,
        )
        if score >= 0.18:
            action = SuggestedAction.OPEN_LONG
        elif score <= -0.18:
            action = SuggestedAction.CLOSE

        if regime == MarketRegime.UNKNOWN:
            regime = MarketRegime.RANGE

        return ResearchInsight(
            symbol=symbol,
            market_regime=regime,
            sentiment_score=sentiment,
            confidence=confidence,
            risk_warning=list(dict.fromkeys(warnings)),
            key_reason=list(dict.fromkeys(reasons)),
            suggested_action=action,
        )
