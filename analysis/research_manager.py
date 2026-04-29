"""Second-stage research approval for CryptoAI v3."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import Settings, get_settings
from core.models import MarketRegime, PredictionResult, ResearchInsight, SuggestedAction
from core.news_risk import contains_bearish_news_risk
from core.storage import Storage
from learning.experience_store import ExperienceStore


@dataclass
class ResearchReview:
    reviewed_insight: ResearchInsight
    approval_rating: str
    review_score: float
    reasons: list[str]
    raw_action: str
    reviewed_action: str
    experience_matches: list[dict[str, Any]] = field(default_factory=list)
    setup_profile: dict[str, str] = field(default_factory=dict)
    setup_performance: dict[str, float | int | str] = field(default_factory=dict)


class ResearchManager:
    """Apply a lightweight second-stage approval over raw research output."""

    EXPERIENCE_NEGATIVE_SETUP_MIN_WEIGHTED_COUNT = 3.0
    EXPERIENCE_NEGATIVE_SETUP_MIN_AVG_OUTCOME_PCT = -0.10
    EXPERIENCE_NEGATIVE_SETUP_SEVERE_AVG_OUTCOME_PCT = -0.20
    CORE_EXTREME_FEAR_REPAIR_MIN_FEAR_GREED = 15.0
    CORE_EXTREME_FEAR_REPAIR_BONUS = 0.25

    def __init__(
        self,
        settings: Settings | None = None,
        storage: Storage | None = None,
    ):
        self.settings = settings or get_settings()
        self.experience_store = ExperienceStore(storage) if storage else None

    def review(
        self,
        symbol: str,
        insight: ResearchInsight,
        prediction: PredictionResult,
        validation: Any,
        features: Any,
        fear_greed: float | None = None,
        news_summary: str = "",
        onchain_summary: str = "",
        news_sources: list[str] | None = None,
        news_coverage_score: float | None = None,
        news_service_health_score: float | None = None,
        adaptive_min_liquidity_ratio: float | None = None,
        blocked_setups: list[dict[str, Any]] | None = None,
    ) -> ResearchReview:
        reasons: list[str] = []
        score = 0.0
        experience_matches: list[dict[str, Any]] = []
        values = getattr(features, "values", {}) or {}
        setup_profile = ExperienceStore.build_setup_profile(
            symbol=symbol,
            market_regime=insight.market_regime.value,
            validation_reason=getattr(validation, "reason", "ok"),
            liquidity_ratio=float(values.get("volume_ratio_1h", 0.0)),
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            news_sources=news_sources,
            news_coverage_score=news_coverage_score,
            news_service_health_score=news_service_health_score,
        )
        setup_performance: dict[str, float | int | str] = {}

        if "fallback_research_model" in insight.key_reason:
            score -= 0.45
            reasons.append("fallback_research_penalty")

        if insight.suggested_action == SuggestedAction.OPEN_LONG:
            score += 0.20
            reasons.append("raw_open_long")
        elif insight.suggested_action == SuggestedAction.CLOSE:
            score -= 0.35
            reasons.append("raw_close")

        adx = float(values.get("adx_4h", 0.0))
        di_plus = float(values.get("di_plus_4h", 0.0))
        di_minus = float(values.get("di_minus_4h", 0.0))
        rsi_1h = float(values.get("rsi_1h", 50.0))
        return_24h = float(values.get("return_24h", 0.0))
        xgb_threshold = self.settings.model.xgboost_probability_threshold
        if prediction.up_probability >= min(0.99, xgb_threshold + 0.05):
            score += 0.20
            reasons.append("xgb_strong")
        elif prediction.up_probability >= xgb_threshold:
            score += 0.10
            reasons.append("xgb_pass")
        elif prediction.up_probability < 0.55:
            score -= 0.20
            reasons.append("xgb_weak")

        liquidity_ratio = float(values.get("volume_ratio_1h", 0.0))
        min_liquidity = float(
            adaptive_min_liquidity_ratio
            if adaptive_min_liquidity_ratio is not None
            else self.settings.strategy.min_liquidity_ratio
        )
        news_event_risk = contains_bearish_news_risk(symbol, news_summary)
        core_extreme_fear_liquidity_repair = (
            self._core_extreme_fear_liquidity_repair_setup(
                symbol=symbol,
                insight=insight,
                prediction=prediction,
                validation=validation,
                fear_greed=fear_greed,
                news_event_risk=news_event_risk,
                liquidity_ratio=liquidity_ratio,
                min_liquidity=min_liquidity,
                adx=adx,
                di_plus=di_plus,
                di_minus=di_minus,
            )
        )
        if liquidity_ratio >= min_liquidity + 0.10:
            score += 0.10
            reasons.append("liquidity_supportive")
        elif liquidity_ratio < min_liquidity:
            if core_extreme_fear_liquidity_repair:
                score -= 0.05
                reasons.append("liquidity_weak_discounted_core_extreme_fear")
            else:
                score -= 0.15
                reasons.append("liquidity_weak")
        extreme_fear_offensive_setup = self._extreme_fear_offensive_setup(
            insight=insight,
            prediction=prediction,
            validation=validation,
            fear_greed=fear_greed,
            liquidity_ratio=liquidity_ratio,
            min_liquidity=min_liquidity,
            adx=adx,
            di_plus=di_plus,
            di_minus=di_minus,
            rsi_1h=rsi_1h,
            return_24h=return_24h,
            news_event_risk=news_event_risk,
        )
        early_reversal_trend_discount = self._early_reversal_trend_discount(
            insight=insight,
            prediction=prediction,
            validation=validation,
            fear_greed=fear_greed,
            liquidity_ratio=liquidity_ratio,
            min_liquidity=min_liquidity,
            di_plus=di_plus,
            di_minus=di_minus,
            rsi_1h=rsi_1h,
            return_24h=return_24h,
            news_event_risk=news_event_risk,
        )
        quant_repairing_setup = self._quant_repairing_setup(
            insight=insight,
            prediction=prediction,
            validation=validation,
            fear_greed=fear_greed,
            liquidity_ratio=liquidity_ratio,
            min_liquidity=min_liquidity,
            di_plus=di_plus,
            di_minus=di_minus,
            rsi_1h=rsi_1h,
            return_24h=return_24h,
            news_event_risk=news_event_risk,
        )
        if adx > 25.0 and di_plus > di_minus:
            score += 0.10
            reasons.append("trend_supportive")
        elif di_minus > di_plus:
            trend_penalty = (
                0.03
                if quant_repairing_setup
                else 0.05
                if early_reversal_trend_discount
                else 0.15
            )
            score -= trend_penalty
            reasons.append(
                "trend_against_quant_repair_discounted"
                if quant_repairing_setup
                else
                "trend_against_discounted"
                if early_reversal_trend_discount
                else "trend_against"
            )

        if insight.market_regime == MarketRegime.DOWNTREND:
            score -= 0.25
            reasons.append("regime_downtrend")
        elif insight.market_regime == MarketRegime.EXTREME_FEAR:
            regime_penalty = (
                0.03
                if quant_repairing_setup
                else 0.05
                if extreme_fear_offensive_setup or core_extreme_fear_liquidity_repair
                else 0.15
            )
            score -= regime_penalty
            reasons.append(
                "regime_extreme_fear_quant_repair_discounted"
                if quant_repairing_setup
                else
                "regime_extreme_fear_core_repair_discounted"
                if core_extreme_fear_liquidity_repair
                else
                "regime_extreme_fear_discounted"
                if extreme_fear_offensive_setup
                else "regime_extreme_fear"
            )

        if fear_greed is not None and fear_greed <= 15:
            fear_penalty = (
                0.03
                if quant_repairing_setup
                else 0.04
                if extreme_fear_offensive_setup or core_extreme_fear_liquidity_repair
                else 0.10
            )
            score -= fear_penalty
            reasons.append(
                "fear_greed_extreme_fear_quant_repair_discounted"
                if quant_repairing_setup
                else
                "fear_greed_extreme_fear_core_repair_discounted"
                if core_extreme_fear_liquidity_repair
                else
                "fear_greed_extreme_fear_discounted"
                if extreme_fear_offensive_setup
                else "fear_greed_extreme_fear"
            )

        if not getattr(validation, "ok", True):
            score -= 0.35
            reasons.append(f"validation_{getattr(validation, 'reason', 'failed')}")

        risk_penalty = min(0.20, 0.10 * len(insight.risk_warning))
        if risk_penalty > 0 and quant_repairing_setup:
            risk_penalty = min(risk_penalty, 0.08)
            reasons.append("risk_warning_discounted_quant_repair_setup")
        elif risk_penalty > 0 and core_extreme_fear_liquidity_repair:
            risk_penalty = min(risk_penalty, 0.06)
            reasons.append("risk_warning_discounted_core_extreme_fear")
        elif risk_penalty > 0 and extreme_fear_offensive_setup:
            risk_penalty = min(risk_penalty, 0.06)
            reasons.append("risk_warning_discounted_extreme_fear_setup")
        if risk_penalty > 0:
            score -= risk_penalty
            reasons.append("risk_warning_present")

        if extreme_fear_offensive_setup:
            score += 0.12
            reasons.append("extreme_fear_offensive_setup")
        if quant_repairing_setup:
            score += 0.12
            reasons.append("quant_repairing_setup")
        if core_extreme_fear_liquidity_repair:
            score += float(self.CORE_EXTREME_FEAR_REPAIR_BONUS)
            reasons.append("core_extreme_fear_liquidity_repair")

        if news_event_risk:
            score -= 0.10
            reasons.append("news_event_risk")

        if news_sources is not None and len(news_sources) < 1:
            coverage_score = float(news_coverage_score or 0.0)
            service_health_score = float(news_service_health_score or 0.0)
            if service_health_score < 0.35:
                score -= 0.05
                reasons.append("news_services_unavailable")
            elif coverage_score < 0.15:
                score -= 0.03
                reasons.append("news_coverage_thin")

        if "unavailable" in onchain_summary.lower():
            reasons.append("onchain_neutral")

        if self.experience_store:
            setup_performance = self.experience_store.aggregate_setup_performance(
                direction="LONG",
                setup_profile=setup_profile,
            )
            setup_count = float(
                setup_performance.get(
                    "weighted_count",
                    setup_performance.get("count", 0),
                )
                or 0.0
            )
            setup_avg_outcome = float(setup_performance.get("avg_outcome_24h", 0.0))
            setup_negative_ratio = float(setup_performance.get("negative_ratio", 0.0))
            if setup_count >= 2 and setup_avg_outcome < 0 and setup_negative_ratio >= 0.6:
                setup_penalty = min(
                    0.22,
                    0.10 + max(0.0, -setup_avg_outcome) * 0.03 + setup_negative_ratio * 0.05,
                )
                score -= setup_penalty
                reasons.append("setup_negative_expectancy")
                reasons.append(f"setup_avg_outcome_{setup_avg_outcome:.2f}")

            setup_context = " ".join(
                [
                    insight.market_regime.value,
                    getattr(validation, "reason", ""),
                    " ".join(str(item) for item in insight.key_reason),
                    " ".join(str(item) for item in insight.risk_warning),
                    news_summary[:240],
                ]
            )
            matches = self.experience_store.find_similar_setups(
                symbol=symbol,
                direction="LONG",
                market_regime=insight.market_regime.value,
                context=setup_context,
                limit=5,
            )
            strong_matches = [
                match for match in matches
                if float(match.get("similarity_score") or 0.0) >= 0.30
            ]
            experience_matches = strong_matches[:3]
            weighted_matches = [
                (
                    float(match.get("outcome_24h") or 0.0),
                    float(match.get("experience_weight") or 1.0),
                )
                for match in strong_matches
            ]
            weighted_count = sum(weight for _, weight in weighted_matches)
            negative_weight = sum(
                weight
                for outcome, weight in weighted_matches
                if outcome < 0
            )
            if weighted_count >= 2 and negative_weight >= 1.0:
                avg_outcome = (
                    sum(outcome * weight for outcome, weight in weighted_matches)
                    / weighted_count
                )
                negative_ratio = negative_weight / weighted_count
                strong_negative = (
                    avg_outcome <= self.EXPERIENCE_NEGATIVE_SETUP_SEVERE_AVG_OUTCOME_PCT
                )
                sufficient_sample_negative = (
                    weighted_count >= self.EXPERIENCE_NEGATIVE_SETUP_MIN_WEIGHTED_COUNT
                    and avg_outcome <= self.EXPERIENCE_NEGATIVE_SETUP_MIN_AVG_OUTCOME_PCT
                )
                if (
                    avg_outcome < 0
                    and negative_ratio >= 0.6
                    and (strong_negative or sufficient_sample_negative)
                ):
                    experience_penalty = min(
                        0.20,
                        0.08 + max(0.0, -avg_outcome) * 0.03 + negative_ratio * 0.04,
                    )
                    score -= experience_penalty
                    reasons.append("experience_negative_setup")
                    reasons.append(f"experience_avg_outcome_{avg_outcome:.2f}")

        strong_quant_override = self._strong_quant_override(
            insight=insight,
            prediction=prediction,
            validation=validation,
            fear_greed=fear_greed,
            liquidity_ratio=liquidity_ratio,
            adx=adx,
            di_plus=di_plus,
            di_minus=di_minus,
            reasons=reasons,
        )
        if strong_quant_override:
            score += float(
                self.settings.strategy.extreme_fear_quant_override_score_bonus
            )
            reasons.append("extreme_fear_quant_override")

        setup_auto_pause = False
        auto_pause_reason = ""
        for blocked in blocked_setups or []:
            criteria = blocked.get("criteria", blocked)
            if all(
                str(setup_profile.get(key, "")) == str(value)
                for key, value in criteria.items()
            ):
                setup_auto_pause = True
                auto_pause_reason = str(blocked.get("reason") or "blocked_setup")
                reasons.append("setup_auto_pause")
                reasons.append(auto_pause_reason)
                break

        reviewed_action = SuggestedAction.HOLD
        approval_rating = "HOLD"
        open_long_special_case = (
            extreme_fear_offensive_setup
            or strong_quant_override
            or quant_repairing_setup
            or core_extreme_fear_liquidity_repair
        )
        if insight.suggested_action == SuggestedAction.CLOSE or score <= -0.15:
            reviewed_action = SuggestedAction.CLOSE
            approval_rating = "UNDERWEIGHT"
        elif score >= 0.35 or (
            open_long_special_case
            and insight.suggested_action == SuggestedAction.OPEN_LONG
            and score >= 0.08
        ):
            reviewed_action = SuggestedAction.OPEN_LONG
            approval_rating = "BUY" if score >= 0.60 else "OVERWEIGHT"
            if score < 0.35:
                reasons.append(
                    "quant_repairing_setup_open"
                    if quant_repairing_setup and not extreme_fear_offensive_setup
                    else "extreme_fear_offensive_open"
                )
        else:
            reviewed_action = SuggestedAction.HOLD
            approval_rating = "HOLD" if score >= 0 else "UNDERWEIGHT"

        if (
            strong_quant_override
            and reviewed_action != SuggestedAction.OPEN_LONG
            and score >= -0.05
        ):
            reviewed_action = SuggestedAction.OPEN_LONG
            approval_rating = "OVERWEIGHT"
            reasons.append("extreme_fear_quant_override_open")
        elif (
            quant_repairing_setup
            and reviewed_action != SuggestedAction.OPEN_LONG
            and score >= -0.02
            and prediction.up_probability
            >= self.settings.model.xgboost_probability_threshold
        ):
            reviewed_action = SuggestedAction.OPEN_LONG
            approval_rating = "OVERWEIGHT"
            reasons.append("quant_repairing_setup_open")
        elif (
            core_extreme_fear_liquidity_repair
            and reviewed_action != SuggestedAction.OPEN_LONG
            and score >= 0.02
            and prediction.up_probability
            >= max(
                0.70,
                self.settings.model.xgboost_probability_threshold - 0.03,
            )
        ):
            reviewed_action = SuggestedAction.OPEN_LONG
            approval_rating = "OVERWEIGHT"
            reasons.append("core_extreme_fear_liquidity_repair_open")

        if setup_auto_pause and reviewed_action == SuggestedAction.OPEN_LONG:
            reviewed_action = SuggestedAction.HOLD
            approval_rating = "HOLD"

        review_confidence = max(
            0.0,
            min(1.0, (insight.confidence + max(0.0, min(1.0, 0.5 + score))) / 2),
        )
        reviewed_warnings = list(insight.risk_warning)
        if reviewed_action != SuggestedAction.OPEN_LONG:
            reviewed_warnings = reviewed_warnings + ["manager_not_approved"]

        reviewed_insight = insight.model_copy(
            update={
                "symbol": symbol,
                "confidence": review_confidence,
                "suggested_action": reviewed_action,
                "risk_warning": reviewed_warnings,
                "key_reason": list(insight.key_reason)
                + [
                    f"manager_rating={approval_rating}",
                    f"review_score={score:.2f}",
                    *reasons,
                ],
            }
        )

        return ResearchReview(
            reviewed_insight=reviewed_insight,
            approval_rating=approval_rating,
            review_score=round(score, 4),
            reasons=reasons,
            raw_action=insight.suggested_action.value,
            reviewed_action=reviewed_action.value,
            experience_matches=experience_matches,
            setup_profile=setup_profile,
            setup_performance=setup_performance,
        )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return str(symbol or "").strip().upper().replace(" ", "")

    def _core_symbols(self) -> set[str]:
        return {
            self._normalize_symbol(symbol)
            for symbol in (self.settings.exchange.core_symbols or [])
            if str(symbol).strip()
        }

    def _core_extreme_fear_repair_liquidity_floor(self, min_liquidity: float) -> float:
        base_floor = max(
            0.18,
            float(getattr(self.settings.strategy, "adaptive_liquidity_floor_min_ratio", 0.35))
            * 0.55,
        )
        return min(float(min_liquidity), base_floor)

    def _core_extreme_fear_liquidity_repair_setup(
        self,
        *,
        symbol: str,
        insight: ResearchInsight,
        prediction: PredictionResult,
        validation: Any,
        fear_greed: float | None,
        news_event_risk: bool,
        liquidity_ratio: float,
        min_liquidity: float,
        adx: float,
        di_plus: float,
        di_minus: float,
    ) -> bool:
        if str(getattr(self.settings.app, "runtime_mode", "") or "").lower() != "paper":
            return False
        if self._normalize_symbol(symbol) not in self._core_symbols():
            return False
        if insight.market_regime != MarketRegime.EXTREME_FEAR:
            return False
        if fear_greed is None or float(fear_greed) > self.CORE_EXTREME_FEAR_REPAIR_MIN_FEAR_GREED:
            return False
        if not getattr(validation, "ok", True):
            return False
        if news_event_risk:
            return False
        if prediction.up_probability < max(
            0.70,
            self.settings.model.xgboost_probability_threshold - 0.03,
        ):
            return False
        if adx < 20.0 or di_plus < di_minus:
            return False
        return liquidity_ratio >= self._core_extreme_fear_repair_liquidity_floor(
            min_liquidity
        )

    def _extreme_fear_offensive_setup(
        self,
        *,
        insight: ResearchInsight,
        prediction: PredictionResult,
        validation: Any,
        fear_greed: float | None,
        liquidity_ratio: float,
        min_liquidity: float,
        adx: float,
        di_plus: float,
        di_minus: float,
        rsi_1h: float,
        return_24h: float,
        news_event_risk: bool,
    ) -> bool:
        if insight.suggested_action != SuggestedAction.OPEN_LONG:
            return False
        if insight.market_regime != MarketRegime.EXTREME_FEAR:
            return False
        if fear_greed is None or fear_greed > 15:
            return False
        if not getattr(validation, "ok", True):
            return False
        if news_event_risk:
            return False
        if "fallback_research_model" in set(insight.key_reason):
            return False
        if len(insight.risk_warning) > 2:
            return False
        if liquidity_ratio < min_liquidity:
            return False
        xgb_threshold = float(self.settings.model.xgboost_probability_threshold)
        if prediction.up_probability < max(0.55, xgb_threshold - 0.18):
            return False
        oversold_reversal = rsi_1h <= 32.0 and return_24h <= -0.025
        trend_repairing = di_plus >= di_minus - 3.0 and (
            adx >= 18.0 or di_plus >= di_minus
        )
        return oversold_reversal and trend_repairing

    def _early_reversal_trend_discount(
        self,
        *,
        insight: ResearchInsight,
        prediction: PredictionResult,
        validation: Any,
        fear_greed: float | None,
        liquidity_ratio: float,
        min_liquidity: float,
        di_plus: float,
        di_minus: float,
        rsi_1h: float,
        return_24h: float,
        news_event_risk: bool,
    ) -> bool:
        if insight.suggested_action != SuggestedAction.OPEN_LONG:
            return False
        if insight.market_regime != MarketRegime.EXTREME_FEAR:
            return False
        if fear_greed is None or fear_greed > 15:
            return False
        if not getattr(validation, "ok", True):
            return False
        if news_event_risk:
            return False
        if liquidity_ratio < min_liquidity:
            return False
        if prediction.up_probability < max(0.55, self.settings.model.xgboost_probability_threshold - 0.18):
            return False
        if rsi_1h > 35.0 or return_24h > -0.02:
            return False
        return (di_minus - di_plus) <= 6.0

    def _strong_quant_override(
        self,
        *,
        insight: ResearchInsight,
        prediction: PredictionResult,
        validation: Any,
        fear_greed: float | None,
        liquidity_ratio: float,
        adx: float,
        di_plus: float,
        di_minus: float,
        reasons: list[str],
    ) -> bool:
        if insight.market_regime != MarketRegime.EXTREME_FEAR:
            return False
        if fear_greed is None or fear_greed > 15:
            return False
        if not getattr(validation, "ok", True):
            return False
        if any(
            reason in reasons
            for reason in (
                "setup_auto_pause",
                "setup_negative_expectancy",
                "experience_negative_setup",
                "news_event_risk",
                "fallback_research_penalty",
                "risk_warning_present",
            )
        ):
            return False
        xgb_threshold = self.settings.model.xgboost_probability_threshold
        min_gap = float(
            self.settings.strategy.extreme_fear_quant_override_min_probability_gap_pct
        )
        min_probability = max(
            xgb_threshold + min_gap,
            float(self.settings.strategy.extreme_fear_quant_override_min_probability_pct),
        )
        if prediction.up_probability < min_probability:
            return False
        liquidity_floor = (
            self.settings.strategy.min_liquidity_ratio
            * float(self.settings.strategy.extreme_fear_quant_override_liquidity_floor_ratio)
        )
        if liquidity_ratio < liquidity_floor:
            return False
        if not (
            (adx > 20.0 and di_plus > di_minus)
            or (
                adx > 18.0
                and di_plus >= di_minus - 2.0
                and prediction.up_probability >= min_probability + 0.04
            )
        ):
            return False
        return True

    def _quant_repairing_setup(
        self,
        *,
        insight: ResearchInsight,
        prediction: PredictionResult,
        validation: Any,
        fear_greed: float | None,
        liquidity_ratio: float,
        min_liquidity: float,
        di_plus: float,
        di_minus: float,
        rsi_1h: float,
        return_24h: float,
        news_event_risk: bool,
    ) -> bool:
        if insight.market_regime != MarketRegime.EXTREME_FEAR:
            return False
        if fear_greed is None or fear_greed > 15:
            return False
        if not getattr(validation, "ok", True):
            return False
        if news_event_risk:
            return False
        if liquidity_ratio < min_liquidity:
            return False
        if prediction.up_probability < self.settings.model.xgboost_probability_threshold:
            return False
        if rsi_1h > 35.0 or return_24h > -0.02:
            return False
        if (di_minus - di_plus) > 10.0:
            return False
        return insight.suggested_action in {
            SuggestedAction.OPEN_LONG,
            SuggestedAction.HOLD,
        }
