"""Decision engine for CryptoAI v3."""
from __future__ import annotations

from core.models import (
    DecisionContext,
    ExecutionDecision,
    MarketRegime,
    PredictionResult,
    ResearchInsight,
    SignalDirection,
    SuggestedAction,
)


class DecisionEngine:
    """Convert model outputs into executable trade decisions."""

    EXTREME_FEAR_OFFENSIVE_XGB_DISCOUNT = 0.15
    EXTREME_FEAR_OFFENSIVE_XGB_FLOOR = 0.55
    EXTREME_FEAR_OFFENSIVE_FINAL_DISCOUNT = 0.18
    EXTREME_FEAR_OFFENSIVE_FINAL_FLOOR = 0.34
    EXTREME_FEAR_OFFENSIVE_POSITION_SCALE = 0.40

    def __init__(
        self,
        xgboost_threshold: float,
        final_score_threshold: float,
        sentiment_weight: float,
        min_liquidity_ratio: float,
        trend_reversal_probability: float,
        sentiment_exit_threshold: float,
        fixed_stop_loss_pct: float = 0.05,
        take_profit_levels: list[float] | None = None,
        max_hold_hours: int = 48,
        extreme_fear_conservative_enabled: bool = True,
        extreme_fear_conservative_xgboost_bonus_pct: float = 0.08,
        extreme_fear_conservative_final_score_bonus: float = 0.10,
        extreme_fear_conservative_liquidity_bonus_ratio: float = 0.15,
        extreme_fear_conservative_position_scale: float = 0.25,
    ):
        self.xgboost_threshold = xgboost_threshold
        self.final_score_threshold = final_score_threshold
        self.sentiment_weight = sentiment_weight
        self.min_liquidity_ratio = min_liquidity_ratio
        self.trend_reversal_probability = trend_reversal_probability
        self.sentiment_exit_threshold = sentiment_exit_threshold
        self.fixed_stop_loss_pct = fixed_stop_loss_pct
        self.take_profit_levels = list(take_profit_levels or [0.05, 0.08])
        self.max_hold_hours = max_hold_hours
        self.extreme_fear_conservative_enabled = bool(
            extreme_fear_conservative_enabled
        )
        self.extreme_fear_conservative_xgboost_bonus_pct = float(
            extreme_fear_conservative_xgboost_bonus_pct
        )
        self.extreme_fear_conservative_final_score_bonus = float(
            extreme_fear_conservative_final_score_bonus
        )
        self.extreme_fear_conservative_liquidity_bonus_ratio = float(
            extreme_fear_conservative_liquidity_bonus_ratio
        )
        self.extreme_fear_conservative_position_scale = float(
            extreme_fear_conservative_position_scale
        )

    def evaluate_entry(
        self,
        symbol: str,
        prediction: PredictionResult,
        insight: ResearchInsight,
        features,
        risk_result,
    ) -> tuple[DecisionContext, ExecutionDecision]:
        trend_filter = self._trend_filter(features.values)
        trend_factor = self._trend_factor(features.values)
        volatility_factor = self._volatility_factor(features.values)
        liquidity_ratio = float(features.values.get("volume_ratio_1h", 0.0))
        extreme_fear_offensive_setup = self._extreme_fear_offensive_setup(insight)
        extreme_fear_conservative_mode = self._extreme_fear_conservative_mode(
            insight,
            extreme_fear_offensive_setup=extreme_fear_offensive_setup,
        )
        effective_xgb_threshold = self._effective_xgboost_threshold(
            extreme_fear_offensive_setup,
            extreme_fear_conservative_mode,
        )
        effective_final_threshold = self._effective_final_score_threshold(
            extreme_fear_offensive_setup,
            extreme_fear_conservative_mode,
        )
        liquidity_floor = self._effective_liquidity_floor(
            features.values,
            extreme_fear_conservative_mode=extreme_fear_conservative_mode,
        )
        model_ready = self._prediction_model_ready(prediction)
        research_ready = "fallback_research_model" not in set(insight.key_reason)
        sentiment_floor = max(
            insight.sentiment_score,
            -0.05 if extreme_fear_offensive_setup else -0.2,
        )
        final_score = (
            prediction.up_probability
            * (1 + sentiment_floor * self.sentiment_weight)
            * trend_factor
            * volatility_factor
        )

        base_entry_ready = all(
            [
                prediction.up_probability >= effective_xgb_threshold,
                final_score >= effective_final_threshold,
                liquidity_ratio >= liquidity_floor,
                risk_result.allowed,
                model_ready,
                research_ready,
                MarketRegime(insight.market_regime) != MarketRegime.DOWNTREND,
                trend_factor >= 0.6,
                insight.suggested_action == SuggestedAction.OPEN_LONG,
            ]
        )
        portfolio_rating, position_scale = self._portfolio_rating(
            prediction=prediction,
            insight=insight,
            trend_factor=trend_factor,
            liquidity_ratio=liquidity_ratio,
            liquidity_floor=liquidity_floor,
            final_score=final_score,
            base_entry_ready=base_entry_ready,
            extreme_fear_conservative_mode=extreme_fear_conservative_mode,
        )
        if extreme_fear_offensive_setup and portfolio_rating == "OVERWEIGHT":
            position_scale = min(
                position_scale,
                self.EXTREME_FEAR_OFFENSIVE_POSITION_SCALE,
            )
        elif extreme_fear_conservative_mode and portfolio_rating in {"BUY", "OVERWEIGHT"}:
            position_scale = min(
                position_scale,
                self.extreme_fear_conservative_position_scale,
            )
        should_open = base_entry_ready and portfolio_rating in {"BUY", "OVERWEIGHT"}

        direction = SignalDirection.LONG if should_open else SignalDirection.FLAT
        context = DecisionContext(
            symbol=symbol,
            prediction=prediction,
            insight=insight,
            features=features,
            trend_filter=trend_filter,
            volatility_factor=volatility_factor,
            liquidity_ratio=liquidity_ratio,
            final_score=final_score,
            portfolio_rating=portfolio_rating,
            position_scale=position_scale,
            direction=direction,
            should_open=should_open,
        )
        decision = ExecutionDecision(
            symbol=symbol,
            direction=direction,
            should_execute=should_open,
            portfolio_rating=portfolio_rating,
            position_scale=position_scale,
            reason=self._entry_reason(
                prediction,
                insight,
                trend_factor,
                final_score,
                liquidity_ratio,
                model_ready,
                research_ready,
                portfolio_rating,
                position_scale,
                risk_result.reason,
                liquidity_floor,
                effective_xgb_threshold,
                effective_final_threshold,
                extreme_fear_offensive_setup,
                extreme_fear_conservative_mode,
            ),
            position_value=(
                risk_result.allowed_position_value * position_scale
                if should_open
                else 0.0
            ),
            stop_loss_pct=risk_result.stop_loss_pct,
            take_profit_levels=risk_result.take_profit_levels,
            trailing_stop_drawdown_pct=risk_result.trailing_stop_drawdown_pct,
            final_score=final_score,
        )
        return context, decision

    def evaluate_exit(
        self,
        position: dict,
        current_price: float,
        prediction: PredictionResult,
        insight: ResearchInsight,
        hours_held: float,
    ) -> list[str]:
        reasons = []
        entry_price = float(position["entry_price"])
        pnl_pct = (current_price / entry_price) - 1.0

        if pnl_pct <= -self.fixed_stop_loss_pct:
            reasons.append("fixed_stop_loss")
        if self.take_profit_levels and pnl_pct >= self._tp_level(0):
            reasons.append("take_profit_1")
        if len(self.take_profit_levels) > 1 and pnl_pct >= self._tp_level(1):
            reasons.append("take_profit_2")
        if prediction.up_probability < self.trend_reversal_probability and (
            insight.sentiment_score < self.sentiment_exit_threshold
        ):
            reasons.append("trend_reversal")
        if hours_held >= self.max_hold_hours:
            reasons.append("time_stop")
        return reasons

    @staticmethod
    def _trend_filter(values: dict[str, float]) -> bool:
        adx = float(values.get("adx_4h", 0.0))
        di_plus = float(values.get("di_plus_4h", 0.0))
        di_minus = float(values.get("di_minus_4h", 0.0))
        return adx > 25.0 and di_plus > di_minus

    @staticmethod
    def _trend_factor(values: dict[str, float]) -> float:
        adx = float(values.get("adx_4h", 0.0))
        di_plus = float(values.get("di_plus_4h", 0.0))
        di_minus = float(values.get("di_minus_4h", 0.0))
        di_gap = di_plus - di_minus
        if adx > 25.0 and di_gap > 0:
            return 1.0
        if di_gap >= -4.0:
            return 0.6
        return 0.45

    @staticmethod
    def _volatility_factor(values: dict[str, float]) -> float:
        atr_percentile = float(values.get("atr_percentile_4h", 0.5))
        if atr_percentile < 0.3:
            return 1.2
        if atr_percentile > 0.7:
            return 0.7
        return 1.0

    @staticmethod
    def _prediction_model_ready(prediction: PredictionResult) -> bool:
        model_version = str(getattr(prediction, "model_version", "") or "").strip().lower()
        if not model_version:
            return False
        if model_version == "invalid_features":
            return False
        return not model_version.startswith("fallback")

    def _entry_reason(
        self,
        prediction: PredictionResult,
        insight: ResearchInsight,
        trend_factor: float,
        final_score: float,
        liquidity_ratio: float,
        model_ready: bool,
        research_ready: bool,
        portfolio_rating: str,
        position_scale: float,
        risk_reason: str,
        liquidity_floor: float,
        effective_xgb_threshold: float,
        effective_final_threshold: float,
        extreme_fear_offensive_setup: bool,
        extreme_fear_conservative_mode: bool,
    ) -> str:
        if risk_reason:
            return risk_reason
        if not model_ready:
            return "model unavailable"
        if not research_ready:
            return "research unavailable"
        reason = (
            f"rating={portfolio_rating}, "
            f"size={position_scale:.2f}, "
            f"pxgb={prediction.up_probability:.2f}, "
            f"sentiment={insight.sentiment_score:+.2f}, "
            f"trend_factor={trend_factor:.2f}, "
            f"liquidity={liquidity_ratio:.2f}, "
            f"score={final_score:.2f}"
        )
        extras: list[str] = []
        if abs(liquidity_floor - self.min_liquidity_ratio) > 1e-9:
            extras.append(f"liq_floor={liquidity_floor:.2f}")
        if extreme_fear_offensive_setup:
            extras.extend(
                [
                    "mode=extreme_fear_offensive",
                    f"xgb_thr={effective_xgb_threshold:.2f}",
                    f"score_thr={effective_final_threshold:.2f}",
                ]
            )
        elif extreme_fear_conservative_mode:
            extras.extend(
                [
                    "mode=extreme_fear_conservative",
                    f"xgb_thr={effective_xgb_threshold:.2f}",
                    f"score_thr={effective_final_threshold:.2f}",
                ]
            )
        if extras:
            reason = f"{reason}, " + ", ".join(extras)
        return reason

    def _portfolio_rating(
        self,
        prediction: PredictionResult,
        insight: ResearchInsight,
        trend_factor: float,
        liquidity_ratio: float,
        liquidity_floor: float,
        final_score: float,
        base_entry_ready: bool,
        extreme_fear_conservative_mode: bool = False,
    ) -> tuple[str, float]:
        if not base_entry_ready:
            return "HOLD", 0.0

        buy_checks = [
            prediction.up_probability >= min(0.99, self.xgboost_threshold + 0.08),
            final_score >= min(0.99, self.final_score_threshold + 0.08),
            insight.confidence >= 0.55,
            insight.sentiment_score > 0,
            trend_factor >= 1.0,
            liquidity_ratio >= liquidity_floor + 0.2,
            len(insight.risk_warning) == 0,
        ]
        if all(buy_checks):
            return "BUY", 1.0

        if extreme_fear_conservative_mode:
            return "HOLD", 0.0

        overweight_checks = sum(
            [
                prediction.up_probability >= min(0.99, self.xgboost_threshold + 0.03),
                final_score >= min(0.99, self.final_score_threshold + 0.04),
                insight.confidence >= 0.55,
                insight.sentiment_score >= 0,
                trend_factor >= 1.0,
                liquidity_ratio >= liquidity_floor + 0.1,
            ]
        )
        if overweight_checks >= 6 and len(insight.risk_warning) <= 1:
            return "OVERWEIGHT", 0.7

        return "OVERWEIGHT", 0.55

    def _liquidity_floor(self, values: dict[str, float]) -> float:
        adaptive_floor = float(
            values.get("adaptive_min_liquidity_ratio", self.min_liquidity_ratio)
            or self.min_liquidity_ratio
        )
        return min(self.min_liquidity_ratio, max(0.0, adaptive_floor))

    def _effective_liquidity_floor(
        self,
        values: dict[str, float],
        *,
        extreme_fear_conservative_mode: bool,
    ) -> float:
        liquidity_floor = self._liquidity_floor(values)
        if not extreme_fear_conservative_mode:
            return liquidity_floor
        return min(
            5.0,
            liquidity_floor + self.extreme_fear_conservative_liquidity_bonus_ratio,
        )

    def _extreme_fear_offensive_setup(self, insight: ResearchInsight) -> bool:
        reasons = {str(reason).strip() for reason in (insight.key_reason or []) if str(reason).strip()}
        return (
            insight.suggested_action == SuggestedAction.OPEN_LONG
            and MarketRegime(insight.market_regime) == MarketRegime.EXTREME_FEAR
            and "extreme_fear_offensive_setup" in reasons
        )

    def _extreme_fear_conservative_mode(
        self,
        insight: ResearchInsight,
        *,
        extreme_fear_offensive_setup: bool,
    ) -> bool:
        if not self.extreme_fear_conservative_enabled:
            return False
        if MarketRegime(insight.market_regime) != MarketRegime.EXTREME_FEAR:
            return False
        if extreme_fear_offensive_setup:
            return False
        reasons = {
            str(reason).strip()
            for reason in (insight.key_reason or [])
            if str(reason).strip()
        }
        return not bool(
            reasons
            & {
                "extreme_fear_quant_override",
                "extreme_fear_quant_override_open",
                "quant_repairing_setup",
                "quant_repairing_setup_open",
            }
        )

    def _effective_xgboost_threshold(
        self,
        extreme_fear_offensive_setup: bool,
        extreme_fear_conservative_mode: bool,
    ) -> float:
        if not extreme_fear_offensive_setup:
            if not extreme_fear_conservative_mode:
                return self.xgboost_threshold
            return min(
                0.99,
                self.xgboost_threshold
                + self.extreme_fear_conservative_xgboost_bonus_pct,
            )
        return max(
            self.EXTREME_FEAR_OFFENSIVE_XGB_FLOOR,
            self.xgboost_threshold - self.EXTREME_FEAR_OFFENSIVE_XGB_DISCOUNT,
        )

    def _effective_final_score_threshold(
        self,
        extreme_fear_offensive_setup: bool,
        extreme_fear_conservative_mode: bool,
    ) -> float:
        if not extreme_fear_offensive_setup:
            if not extreme_fear_conservative_mode:
                return self.final_score_threshold
            return min(
                0.99,
                self.final_score_threshold
                + self.extreme_fear_conservative_final_score_bonus,
            )
        return max(
            self.EXTREME_FEAR_OFFENSIVE_FINAL_FLOOR,
            self.final_score_threshold - self.EXTREME_FEAR_OFFENSIVE_FINAL_DISCOUNT,
        )

    def _tp_level(self, index: int) -> float:
        if not self.take_profit_levels:
            return 0.05
        safe_index = min(index, len(self.take_profit_levels) - 1)
        return self.take_profit_levels[safe_index]
