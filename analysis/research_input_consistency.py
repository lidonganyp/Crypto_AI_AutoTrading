"""Research-input consistency checks for the active CryptoAI v3 runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ResearchInputConsistencyResult:
    ok: bool
    consistency_score: float
    reason: str
    details: dict


class ResearchInputConsistencyService:
    """Validate whether external research inputs are strong enough to trust."""

    MIN_OK_SCORE = 0.60
    NEWS_SERVICE_HEALTH_FLOOR = 0.35

    def validate(
        self,
        *,
        symbol: str,
        news_summary: Any,
        onchain_summary: Any,
        fear_greed_value: float | None,
        lunarcrush_sentiment: float | None,
    ) -> ResearchInputConsistencyResult:
        score = 1.0
        reasons: list[str] = []

        news_sources = [
            str(item).strip()
            for item in list(getattr(news_summary, "sources", []) or [])
            if str(item).strip()
        ]
        news_health_score = self._optional_float(
            getattr(news_summary, "service_health_score", None)
        )
        news_source_status = dict(getattr(news_summary, "source_status", {}) or {})
        if (
            not news_sources
            and (
                self._all_news_services_unavailable(news_source_status)
                or (
                    news_health_score is not None
                    and news_health_score < self.NEWS_SERVICE_HEALTH_FLOOR
                )
            )
        ):
            score -= 0.35
            reasons.append("news_services_unavailable")

        onchain_source = None
        onchain_available = None
        if hasattr(onchain_summary, "source"):
            onchain_source = str(getattr(onchain_summary, "source", "") or "").strip()
            onchain_available = onchain_source.lower() not in {
                "",
                "fallback",
                "unknown",
            }
            if not onchain_available:
                score -= 0.25
                reasons.append("onchain_context_unavailable")

        sentiment_available = (
            fear_greed_value is not None or lunarcrush_sentiment is not None
        )
        if (
            fear_greed_value is not None
            and lunarcrush_sentiment is not None
            and ((fear_greed_value - 50.0) / 50.0) * lunarcrush_sentiment < 0
        ):
            score -= 0.10
            reasons.append("sentiment_source_conflict")

        available_context_sources = 0
        if news_sources:
            available_context_sources += 1
        if onchain_available:
            available_context_sources += 1
        if sentiment_available:
            available_context_sources += 1
        if available_context_sources < 2:
            score -= 0.10
            reasons.append("external_context_thin")

        score = max(0.0, min(1.0, score))
        ok = score >= self.MIN_OK_SCORE
        return ResearchInputConsistencyResult(
            ok=ok,
            consistency_score=score,
            reason="ok" if ok else ",".join(reasons) or "input_consistency_too_low",
            details={
                "symbol": symbol,
                "news_sources": news_sources,
                "news_service_health_score": news_health_score,
                "news_source_status": news_source_status,
                "onchain_source": onchain_source or "",
                "fear_greed_value": fear_greed_value,
                "lunarcrush_sentiment": lunarcrush_sentiment,
                "available_context_sources": available_context_sources,
                "reasons": reasons,
            },
        )

    @staticmethod
    def _all_news_services_unavailable(source_status: dict[str, str]) -> bool:
        if not source_status:
            return False
        active_statuses = [
            str(status).strip().lower()
            for status in source_status.values()
            if str(status).strip().lower() != "disabled"
        ]
        return bool(active_statuses) and all(
            status == "unavailable" for status in active_statuses
        )

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
