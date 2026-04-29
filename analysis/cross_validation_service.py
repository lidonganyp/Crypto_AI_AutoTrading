"""Cross-source validation helpers for CryptoAI v3."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossValidationResult:
    ok: bool
    consistency_score: float
    reason: str
    details: dict


class CrossValidationService:
    """Compare multiple source signals for consistency before trading."""

    REGIME_REVERSAL_CONFIRMATION_BONUS = 0.15
    REGIME_REVERSAL_MIN_PRICE_RETURN = 0.02

    def validate(
        self,
        symbol: str,
        news_sources: list[str],
        fear_greed_value: float | None,
        lunarcrush_sentiment: float | None,
        onchain_netflow_score: float | None,
        regime_score: float | None,
        price_return_24h: float | None,
        news_coverage_score: float | None = None,
        news_service_health_score: float | None = None,
    ) -> CrossValidationResult:
        score = 1.0
        reasons: list[str] = []
        coverage_score = max(0.0, min(1.0, float(news_coverage_score or 0.0)))
        service_health_score = max(
            0.0,
            min(1.0, float(news_service_health_score or 0.0)),
        )

        if len(news_sources) < 1:
            if service_health_score < 0.35:
                score -= 0.08
                reasons.append("news_services_unavailable")
            else:
                score -= 0.05
                reasons.append("news_coverage_thin")
        elif coverage_score < 0.25:
            score -= 0.05
            reasons.append("news_coverage_weak")
        elif coverage_score >= 0.6 and len(news_sources) >= 2:
            score += 0.03
            reasons.append("news_coverage_supportive")

        if fear_greed_value is not None and lunarcrush_sentiment is not None:
            normalized_fng = (fear_greed_value - 50.0) / 50.0
            if normalized_fng * lunarcrush_sentiment < 0:
                score -= 0.25
                reasons.append("sentiment_source_conflict")

        onchain_conflict = False
        price_conflict = False
        if onchain_netflow_score is not None and regime_score is not None:
            onchain_conflict = onchain_netflow_score * regime_score < 0
            if onchain_conflict:
                score -= 0.2
                reasons.append("onchain_regime_conflict")

        if price_return_24h is not None and regime_score is not None:
            price_conflict = price_return_24h * regime_score < 0
            if price_conflict:
                score -= 0.15
                reasons.append("price_regime_conflict")

        # When price and on-chain agree with each other against the regime label,
        # treat it as a possible regime-lagging reversal rather than two independent conflicts.
        if (
            onchain_conflict
            and price_conflict
            and onchain_netflow_score is not None
            and price_return_24h is not None
            and onchain_netflow_score * price_return_24h > 0
            and abs(float(price_return_24h)) >= self.REGIME_REVERSAL_MIN_PRICE_RETURN
        ):
            score += self.REGIME_REVERSAL_CONFIRMATION_BONUS
            reasons.append("regime_reversal_confirmation")

        score = max(0.0, min(1.0, score))
        ok = score >= 0.7
        return CrossValidationResult(
            ok=ok,
            consistency_score=score,
            reason="ok" if ok else ",".join(reasons) or "consistency_too_low",
            details={
                "symbol": symbol,
                "news_sources": news_sources,
                "news_coverage_score": coverage_score,
                "news_service_health_score": service_health_score,
                "fear_greed_value": fear_greed_value,
                "lunarcrush_sentiment": lunarcrush_sentiment,
                "onchain_netflow_score": onchain_netflow_score,
                "regime_score": regime_score,
                "price_return_24h": price_return_24h,
                "reasons": reasons,
            },
        )
