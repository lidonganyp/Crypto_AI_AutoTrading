"""Open-position management helpers."""
from __future__ import annotations

import json
from datetime import datetime

from config import Settings, get_settings
from core.feature_pipeline import FeatureInput, FeaturePipeline
from core.storage import Storage
from core.strategy_profile import profile_from_trade_metadata


class PositionRuntimeService:
    """Manage exit evaluation and close execution for open positions."""

    RESEARCH_EXIT_WATCH_ONLY_REASON = "research_exit_watch_hold"
    RESEARCH_EXIT_PATIENCE_BLOCKERS = {
        "setup_auto_pause",
        "setup_negative_expectancy",
        "experience_negative_setup",
        "news_event_risk",
        "fallback_research_penalty",
        "bearish_news_exit",
        "cross_validation_exit",
    }
    STRONG_RESEARCH_EXIT_MIN_HOLD_HOURS = 1.5
    STRONG_RESEARCH_EXIT_CONFIRMATION_COUNT = 3
    STRONG_RESEARCH_EXIT_MAX_ADVERSE_PNL_RATIO = -0.03
    STRONG_RESEARCH_EXIT_SEVERE_REVIEW_SCORE = -0.45
    MODERATE_RESEARCH_EXIT_MIN_HOLD_HOURS = 0.75
    MODERATE_RESEARCH_EXIT_CONFIRMATION_COUNT = 3
    MODERATE_RESEARCH_EXIT_MAX_ADVERSE_PNL_RATIO = -0.025
    MODERATE_RESEARCH_EXIT_SEVERE_REVIEW_SCORE = -0.35
    EXECUTION_RESEARCH_EXIT_MIN_HOLD_HOURS = 1.5
    EXECUTION_RESEARCH_EXIT_CONFIRMATION_COUNT = 3
    EXECUTION_RESEARCH_EXIT_MAX_ADVERSE_PNL_RATIO = -0.03
    EXECUTION_RESEARCH_EXIT_SEVERE_REVIEW_SCORE = -0.40
    FAST_ALPHA_RESEARCH_EXIT_MIN_HOLD_HOURS = 0.5
    FAST_ALPHA_RESEARCH_EXIT_CONFIRMATION_COUNT = 2
    FAST_ALPHA_RESEARCH_EXIT_MAX_ADVERSE_PNL_RATIO = -0.015
    FAST_ALPHA_RESEARCH_EXIT_SEVERE_REVIEW_SCORE = -0.25
    FAST_ALPHA_CLOSE_OVERRIDE_RESEARCH_EXIT_SCORE_DELTA = 0.15
    FAST_ALPHA_CLOSE_OVERRIDE_RESEARCH_EXIT_SCORE_FLOOR = -0.55
    FAST_ALPHA_EVIDENCE_SCALE_FLOOR = 0.50
    FAST_ALPHA_STRONG_OPEN_RESEARCH_EXIT_MIN_SCORE = 0.25
    FAST_ALPHA_STRONG_OPEN_RESEARCH_EXIT_MIN_HOLD_HOURS = 1.0
    PAPER_CANARY_PRIMARY_RESEARCH_EXIT_MIN_SCORE = 0.15
    PAPER_CANARY_PRIMARY_RESEARCH_EXIT_MIN_HOLD_HOURS = 1.0

    def __init__(
        self,
        storage: Storage,
        settings: Settings | None = None,
        *,
        market,
        feature_pipeline: FeaturePipeline,
        decision_engine,
        executor,
        notifier,
        prepare_position_review,
        predictor_for_symbol,
        fallback_research,
        record_trade_result,
        reflect_closed_trade_result,
        handle_trade_close_feedback=None,
        exit_policy_getter=None,
        position_review_state_key: str = "position_review_state",
    ):
        self.storage = storage
        self.settings = settings or get_settings()
        self.market = market
        self.feature_pipeline = feature_pipeline
        self.decision_engine = decision_engine
        self.executor = executor
        self.notifier = notifier
        self.prepare_position_review = prepare_position_review
        self.predictor_for_symbol = predictor_for_symbol
        self.fallback_research = fallback_research
        self.record_trade_result = record_trade_result
        self.reflect_closed_trade_result = reflect_closed_trade_result
        self.handle_trade_close_feedback = handle_trade_close_feedback
        self.exit_policy_getter = exit_policy_getter
        self.position_review_state_key = position_review_state_key

    def manage_open_positions(self, now: datetime, positions: list[dict], account) -> int:
        closed_count = 0
        open_trades = {trade["symbol"]: trade for trade in self.storage.get_open_trades()}
        for position in positions:
            market_symbol = self._market_symbol(str(position["symbol"]))
            current_price = self.market.fetch_latest_price(market_symbol)
            if current_price is None:
                continue
            trade = open_trades.get(position["symbol"])
            initial_quantity = float(
                (trade or {}).get("initial_quantity") or position["quantity"]
            )
            current_quantity = float(position["quantity"])
            entry_dt = datetime.fromisoformat(position["entry_time"])
            hours_held = (now - entry_dt).total_seconds() / 3600
            exit_policy = (
                dict(self.exit_policy_getter(position["symbol"]) or {})
                if self.exit_policy_getter is not None
                else {}
            )
            exit_policy = self._merge_pipeline_exit_policy(
                trade=trade,
                exit_policy=exit_policy,
            )
            review_snapshot = self.prepare_position_review(position["symbol"], now)
            review_prediction = (
                review_snapshot["prediction"]
                if review_snapshot and review_snapshot.get("ok")
                else self.predictor_for_symbol(position["symbol"]).predict(
                    self.feature_pipeline.build(
                        FeatureInput(
                            symbol=position["symbol"],
                            candles_1h=self._stored_candles(position["symbol"], "1h", limit=240),
                            candles_4h=self._stored_candles(position["symbol"], "4h", limit=240),
                            candles_1d=self._stored_candles(position["symbol"], "1d", limit=240),
                        )
                    )
                )
            )
            review_insight = (
                review_snapshot["insight"]
                if review_snapshot and review_snapshot.get("ok")
                else self.fallback_research(position["symbol"])
            )
            model_exit_reasons = self.decision_engine.evaluate_exit(
                position=position,
                current_price=current_price,
                prediction=review_prediction,
                insight=review_insight,
                hours_held=hours_held,
            )
            review_exit_reasons = self.position_review_exit_reasons(
                position=position,
                current_price=current_price,
                hours_held=hours_held,
                review_snapshot=review_snapshot,
            )
            review_exit_reasons = self._stabilize_review_exit_reasons(
                trade=trade,
                position=position,
                current_price=current_price,
                hours_held=hours_held,
                review_snapshot=review_snapshot,
                review_exit_reasons=review_exit_reasons,
            )
            adaptive_exit_reasons = self.evidence_exit_reasons(
                position=position,
                current_price=current_price,
                hours_held=hours_held,
                exit_policy=exit_policy,
            )
            protection_reasons = self.position_protection_reasons(
                position,
                current_price=current_price,
                initial_quantity=initial_quantity,
            )
            exit_reasons = protection_reasons + review_exit_reasons + adaptive_exit_reasons + [
                reason
                for reason in model_exit_reasons
                if reason not in {"fixed_stop_loss", "take_profit_1", "take_profit_2"}
                and reason not in protection_reasons
                and reason not in review_exit_reasons
                and reason not in adaptive_exit_reasons
            ]
            watch_only = self.RESEARCH_EXIT_WATCH_ONLY_REASON in exit_reasons
            if watch_only:
                review_state = self._position_review_state().get(
                    self._position_review_trade_key(trade),
                    {},
                )
                self.storage.insert_execution_event(
                    "position_review_watch",
                    position["symbol"],
                    {
                        "trade_id": str((trade or {}).get("id") or ""),
                        "current_price": float(current_price),
                        "hours_held": float(hours_held),
                        "review_exit_reasons": list(review_exit_reasons),
                        "adaptive_exit_reasons": list(adaptive_exit_reasons),
                        "model_exit_reasons": list(model_exit_reasons),
                        "research_exit_count": int(
                            review_state.get("research_exit_count", 0) or 0
                        ),
                        "review_score": float(
                            review_state.get("review_score", 0.0) or 0.0
                        ),
                    },
                )
                exit_reasons = [
                    reason
                    for reason in exit_reasons
                    if reason != self.RESEARCH_EXIT_WATCH_ONLY_REASON
                ]
                if not exit_reasons:
                    continue
            if not exit_reasons:
                self._clear_position_review_state(trade)
                continue

            close_qty = self.review_exit_close_quantity(
                initial_quantity=initial_quantity,
                current_quantity=current_quantity,
                exit_reasons=exit_reasons,
            )

            if (
                "take_profit_2" in exit_reasons
                or (
                    "take_profit_1" in exit_reasons
                    and bool(exit_policy.get("force_full_take_profit", False))
                )
                or "evidence_time_stop" in exit_reasons
            ):
                close_qty = current_quantity
            elif (
                close_qty is None
                and exit_reasons == ["take_profit_1"]
                and current_quantity >= initial_quantity - 1e-8
            ):
                close_qty = current_quantity / 2

            result = self.executor.execute_close(
                position["symbol"],
                current_price,
                reason=",".join(exit_reasons),
                close_qty=close_qty,
            )
            if result:
                if result.get("dry_run"):
                    continue
                if result.get("is_full_close"):
                    closed_count += 1
                    self._clear_position_review_state(trade)
                    self.record_trade_result(result["pnl"])
                    self.reflect_closed_trade_result(position["symbol"], result)
                elif callable(self.handle_trade_close_feedback):
                    self.handle_trade_close_feedback(position["symbol"], result)
                self.notifier.notify_trade_close(
                    position["symbol"],
                    float(result["exit_price"]),
                    result["pnl"],
                    result["pnl_pct"],
                    result["reason"],
                )
                open_trades = {
                    trade["symbol"]: trade for trade in self.storage.get_open_trades()
                }
        return closed_count

    @staticmethod
    def position_review_exit_reasons(
        position: dict,
        current_price: float,
        hours_held: float,
        review_snapshot: dict | None,
    ) -> list[str]:
        if not review_snapshot:
            return []

        if not review_snapshot.get("ok"):
            if review_snapshot.get("reason") == "bearish_news_detected":
                return ["bearish_news_exit"]
            return []

        entry_price = float(position["entry_price"])
        pnl_ratio = (current_price / entry_price) - 1.0 if entry_price > 0 else 0.0
        insight = review_snapshot["insight"]
        validation = review_snapshot["validation"]
        decision = review_snapshot["decision"]
        review = review_snapshot.get("review")
        auto_pause_active = bool(
            review and "setup_auto_pause" in getattr(review, "reasons", [])
        )
        reasons: list[str] = []

        suggested_action = getattr(
            getattr(insight, "suggested_action", None),
            "name",
            getattr(insight, "suggested_action", ""),
        )
        if str(suggested_action) == "CLOSE":
            reasons.append("research_exit")
        if not validation.ok and pnl_ratio >= 0:
            reasons.append("cross_validation_exit")
        if (
            decision.portfolio_rating == "HOLD"
            and pnl_ratio >= 0.01
            and hours_held >= 8
            and not auto_pause_active
        ):
            reasons.append("portfolio_de_risk")
        return reasons

    @staticmethod
    def review_exit_close_quantity(
        initial_quantity: float,
        current_quantity: float,
        exit_reasons: list[str],
    ) -> float | None:
        if "research_exit" in exit_reasons or "bearish_news_exit" in exit_reasons:
            return current_quantity
        if (
            "research_de_risk" in exit_reasons
            or "research_exit_watch" in exit_reasons
            or "cross_validation_exit" in exit_reasons
            or "portfolio_de_risk" in exit_reasons
            or "evidence_de_risk" in exit_reasons
        ):
            if current_quantity >= initial_quantity - 1e-8:
                return current_quantity / 2
            return current_quantity
        return None

    def _stabilize_review_exit_reasons(
        self,
        *,
        trade: dict | None,
        position: dict,
        current_price: float,
        hours_held: float,
        review_snapshot: dict | None,
        review_exit_reasons: list[str],
    ) -> list[str]:
        if not trade:
            return review_exit_reasons
        if not review_snapshot or not review_snapshot.get("ok"):
            return review_exit_reasons
        if "bearish_news_exit" in review_exit_reasons:
            return review_exit_reasons
        if "research_exit" not in review_exit_reasons:
            self._clear_position_review_state(trade)
            return review_exit_reasons

        entry_price = float(position.get("entry_price") or 0.0)
        initial_quantity = float(
            (trade or {}).get("initial_quantity") or position.get("quantity") or 0.0
        )
        current_quantity = float(position.get("quantity") or 0.0)
        pnl_ratio = (current_price / entry_price) - 1.0 if entry_price > 0 else 0.0
        review = review_snapshot.get("review")
        review_score = float(getattr(review, "review_score", 0.0) or 0.0)
        state = self._position_review_state()
        trade_key = self._position_review_trade_key(trade)
        previous = state.get(trade_key, {}) if isinstance(state, dict) else {}
        count = int(previous.get("research_exit_count", 0) or 0) + 1
        patience_policy = self._research_exit_patience_policy(
            trade=trade,
            review_snapshot=review_snapshot,
        )
        if patience_policy:
            self._record_position_review_state(
                trade=trade,
                review_score=review_score,
                count=count,
            )
            min_hold_hours = float(patience_policy["min_hold_hours"])
            should_force_close = (
                pnl_ratio <= float(patience_policy["max_adverse_pnl_ratio"])
                or review_score <= float(patience_policy["severe_review_score"])
                or (
                    hours_held >= min_hold_hours
                    and count >= int(patience_policy["confirmation_count"])
                )
            )
            if should_force_close:
                return review_exit_reasons
            if bool(patience_policy.get("watch_only_until_confirmed", False)):
                return [self.RESEARCH_EXIT_WATCH_ONLY_REASON]
            if current_quantity >= initial_quantity - 1e-8:
                return [
                    "research_de_risk" if reason == "research_exit" else reason
                    for reason in review_exit_reasons
                ] + ["research_exit_watch"]
            return [self.RESEARCH_EXIT_WATCH_ONLY_REASON]
        if hours_held >= 6.0 or pnl_ratio <= -0.015 or review_score <= -0.45:
            self._record_position_review_state(
                trade=trade,
                review_score=review_score,
                count=max(
                    1,
                    int(
                        previous.get("research_exit_count", 0)
                        or 0
                    ),
                ),
            )
            return review_exit_reasons

        self._record_position_review_state(
            trade=trade,
            review_score=review_score,
            count=count,
        )
        if count <= 1:
            return [
                "research_de_risk" if reason == "research_exit" else reason
                for reason in review_exit_reasons
            ] + ["research_exit_watch"]
        return review_exit_reasons

    def _record_position_review_state(
        self,
        *,
        trade: dict,
        review_score: float,
        count: int,
    ) -> dict[str, dict]:
        state = self._position_review_state()
        trade_key = self._position_review_trade_key(trade)
        state[trade_key] = {
            "trade_id": trade.get("id"),
            "symbol": trade.get("symbol"),
            "research_exit_count": max(0, int(count)),
            "review_score": float(review_score),
            "updated_at": datetime.now().isoformat(),
        }
        self.storage.set_json_state(self.position_review_state_key, state)
        return state

    @classmethod
    def _research_exit_patience_policy(
        cls,
        *,
        trade: dict,
        review_snapshot: dict | None,
    ) -> dict | None:
        review = (review_snapshot or {}).get("review")
        if review is None:
            return None
        reasons = {
            str(reason).strip()
            for reason in (getattr(review, "reasons", []) or [])
            if str(reason).strip()
        }
        if any(reason.startswith("validation_") for reason in reasons):
            return None
        if reasons & cls.RESEARCH_EXIT_PATIENCE_BLOCKERS:
            return None
        raw_action = str(getattr(review, "raw_action", "") or "").upper()

        metadata = cls._trade_metadata(trade)
        pipeline_mode = str(metadata.get("pipeline_mode") or "").strip()
        profile = profile_from_trade_metadata(
            metadata,
            current_raw_action=raw_action,
            current_review_reasons=reasons,
        )
        canary_mode = str(profile.canary_mode or "").strip()
        entry_thesis = str(profile.entry_thesis or "").strip()
        entry_thesis_strength = str(profile.entry_thesis_strength or "").strip()
        entry_open_bias = bool(profile.entry_open_bias)
        entry_reviewed_action = str(metadata.get("reviewed_action") or "").upper()
        entry_raw_action = str(metadata.get("raw_action") or "").upper()
        entry_review_score = cls._safe_float(metadata.get("review_score"))
        entry_review_reasons = {
            str(reason).strip()
            for reason in (metadata.get("review_reasons") or [])
            if str(reason).strip()
        }
        is_current_open_bias = bool(profile.is_current_open_bias)
        has_entry_thesis = bool(profile.has_entry_thesis)
        strong_entry_quality = bool(profile.strong_entry_quality)
        if strong_entry_quality and (is_current_open_bias or has_entry_thesis):
            policy = {
                "confirmation_count": cls.STRONG_RESEARCH_EXIT_CONFIRMATION_COUNT,
                "min_hold_hours": cls.STRONG_RESEARCH_EXIT_MIN_HOLD_HOURS,
                "max_adverse_pnl_ratio": cls.STRONG_RESEARCH_EXIT_MAX_ADVERSE_PNL_RATIO,
                "severe_review_score": cls.STRONG_RESEARCH_EXIT_SEVERE_REVIEW_SCORE,
            }
            if (
                pipeline_mode == "fast_alpha"
                and entry_reviewed_action == "OPEN_LONG"
                and entry_review_score >= cls.FAST_ALPHA_STRONG_OPEN_RESEARCH_EXIT_MIN_SCORE
                and (
                    "core_extreme_fear_liquidity_repair" in entry_review_reasons
                    or "xgb_strong" in entry_review_reasons
                )
            ):
                policy["watch_only_until_confirmed"] = True
                policy["min_hold_hours"] = max(
                    float(policy["min_hold_hours"]),
                    cls.FAST_ALPHA_STRONG_OPEN_RESEARCH_EXIT_MIN_HOLD_HOURS,
                )
            elif (
                pipeline_mode == "paper_canary"
                and canary_mode == "primary"
                and entry_reviewed_action == "OPEN_LONG"
                and entry_review_score >= cls.PAPER_CANARY_PRIMARY_RESEARCH_EXIT_MIN_SCORE
            ):
                policy["watch_only_until_confirmed"] = True
                policy["min_hold_hours"] = max(
                    float(policy["min_hold_hours"]),
                    cls.PAPER_CANARY_PRIMARY_RESEARCH_EXIT_MIN_HOLD_HOURS,
                )
            return policy
        if pipeline_mode == "fast_alpha" and (
            entry_reviewed_action in {"OPEN_LONG", "HOLD"}
            or entry_thesis == "fast_alpha_short_horizon"
        ):
            horizon_hours = cls._safe_float(metadata.get("horizon_hours"), 0.0)
            severe_review_score = cls.FAST_ALPHA_RESEARCH_EXIT_SEVERE_REVIEW_SCORE
            if bool(metadata.get("fast_alpha_close_override")):
                severe_review_score = max(
                    cls.FAST_ALPHA_CLOSE_OVERRIDE_RESEARCH_EXIT_SCORE_FLOOR,
                    entry_review_score
                    - cls.FAST_ALPHA_CLOSE_OVERRIDE_RESEARCH_EXIT_SCORE_DELTA,
                )
            min_hold_hours = max(
                cls.FAST_ALPHA_RESEARCH_EXIT_MIN_HOLD_HOURS,
                min(horizon_hours * 0.25 if horizon_hours > 0 else 1.0, 1.0),
            )
            return {
                "confirmation_count": cls.FAST_ALPHA_RESEARCH_EXIT_CONFIRMATION_COUNT,
                "min_hold_hours": float(min_hold_hours),
                "max_adverse_pnl_ratio": cls.FAST_ALPHA_RESEARCH_EXIT_MAX_ADVERSE_PNL_RATIO,
                "severe_review_score": float(severe_review_score),
                "watch_only_until_confirmed": bool(
                    metadata.get("fast_alpha_close_override")
                ),
            }
        if (
            pipeline_mode == "paper_canary"
            and (
                canary_mode == "soft_review"
                or entry_thesis == "paper_canary_soft"
            )
            and (raw_action in {"OPEN_LONG", "HOLD"} or has_entry_thesis)
            and entry_review_score >= -0.05
        ):
            return {
                "confirmation_count": cls.MODERATE_RESEARCH_EXIT_CONFIRMATION_COUNT,
                "min_hold_hours": cls.MODERATE_RESEARCH_EXIT_MIN_HOLD_HOURS,
                "max_adverse_pnl_ratio": cls.MODERATE_RESEARCH_EXIT_MAX_ADVERSE_PNL_RATIO,
                "severe_review_score": cls.MODERATE_RESEARCH_EXIT_SEVERE_REVIEW_SCORE,
            }
        if (
            pipeline_mode != "paper_canary"
            and (
                entry_reviewed_action == "OPEN_LONG"
                or entry_thesis_strength in {"moderate", "strong"}
            )
            and entry_review_score >= 0.15
            and has_entry_thesis
        ):
            return {
                "confirmation_count": cls.EXECUTION_RESEARCH_EXIT_CONFIRMATION_COUNT,
                "min_hold_hours": cls.EXECUTION_RESEARCH_EXIT_MIN_HOLD_HOURS,
                "max_adverse_pnl_ratio": cls.EXECUTION_RESEARCH_EXIT_MAX_ADVERSE_PNL_RATIO,
                "severe_review_score": cls.EXECUTION_RESEARCH_EXIT_SEVERE_REVIEW_SCORE,
            }
        return None

    def _clear_position_review_state(self, trade: dict | None) -> None:
        if not trade:
            return
        state = self._position_review_state()
        trade_key = self._position_review_trade_key(trade)
        if trade_key in state:
            state.pop(trade_key, None)
            self.storage.set_json_state(self.position_review_state_key, state)

    def _position_review_state(self) -> dict[str, dict]:
        state = self.storage.get_json_state(self.position_review_state_key, {})
        return state if isinstance(state, dict) else {}

    @staticmethod
    def _position_review_trade_key(trade: dict) -> str:
        return str(trade.get("id") or trade.get("symbol") or "")

    @staticmethod
    def _trade_metadata(trade: dict | None) -> dict:
        if not trade:
            return {}
        raw = trade.get("metadata_json")
        if isinstance(raw, dict):
            return dict(raw)
        try:
            payload = json.loads(raw or "{}")
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _market_symbol(symbol: str) -> str:
        return symbol if ":USDT" in symbol else f"{symbol}:USDT"

    @staticmethod
    def _storage_symbol_variants(symbol: str) -> list[str]:
        variants = [str(symbol)]
        if ":USDT" in symbol:
            variants.append(symbol.replace(":USDT", ""))
        else:
            variants.append(f"{symbol}:USDT")
        return list(dict.fromkeys(variants))

    def _stored_candles(
        self,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 240,
    ) -> list[dict]:
        for storage_symbol in self._storage_symbol_variants(symbol):
            candles = self.storage.get_ohlcv(storage_symbol, timeframe, limit=limit)
            if candles:
                return candles
        return []

    def _merge_pipeline_exit_policy(
        self,
        *,
        trade: dict | None,
        exit_policy: dict | None,
    ) -> dict:
        policy = dict(exit_policy or {})
        metadata = self._trade_metadata(trade)
        if str(metadata.get("pipeline_mode") or "").strip() != "fast_alpha":
            return policy
        horizon_hours = max(
            1.0,
            self._safe_float(
                metadata.get("horizon_hours"),
                float(self.settings.strategy.fast_alpha_max_hold_hours),
            ),
        )
        de_risk_min_hours = max(
            0.5,
            min(
                horizon_hours,
                self._safe_float(
                    metadata.get("fast_alpha_de_risk_hours"),
                    float(self.settings.strategy.fast_alpha_de_risk_hours),
                ),
            ),
        )
        de_risk_min_pnl_ratio = self._safe_float(
            metadata.get("fast_alpha_de_risk_pnl_ratio"),
            float(self.settings.strategy.fast_alpha_de_risk_pnl_ratio),
        )
        evidence_scale = max(
            self.FAST_ALPHA_EVIDENCE_SCALE_FLOOR,
            min(
                1.0,
                self._safe_float(metadata.get("model_evidence_scale"), 1.0),
            ),
        )
        if evidence_scale < 0.999:
            horizon_hours = max(1.0, horizon_hours * evidence_scale)
            de_risk_min_hours = max(
                0.25,
                min(horizon_hours, de_risk_min_hours * evidence_scale),
            )
            if de_risk_min_pnl_ratio > 0:
                de_risk_min_pnl_ratio = max(
                    0.001,
                    de_risk_min_pnl_ratio * evidence_scale,
                )
        policy.update(
            {
                "adaptive_active": True,
                "source": "fast_alpha_runtime",
                "reason": (
                    "fast_alpha_short_horizon"
                    if evidence_scale >= 0.999
                    else f"fast_alpha_short_horizon|evidence_scale={evidence_scale:.2f}"
                ),
                "time_stop_hours": float(horizon_hours),
                "de_risk_min_hours": float(de_risk_min_hours),
                "de_risk_min_pnl_ratio": float(de_risk_min_pnl_ratio),
                "force_full_take_profit": True,
            }
        )
        return policy

    @staticmethod
    def evidence_exit_reasons(
        position: dict,
        current_price: float,
        hours_held: float,
        exit_policy: dict | None,
    ) -> list[str]:
        policy = dict(exit_policy or {})
        if not bool(policy.get("adaptive_active", False)):
            return []
        entry_price = float(position.get("entry_price") or 0.0)
        pnl_ratio = (current_price / entry_price) - 1.0 if entry_price > 0 else 0.0
        reasons: list[str] = []
        time_stop_hours = float(policy.get("time_stop_hours", 0.0) or 0.0)
        if time_stop_hours > 0 and hours_held >= time_stop_hours:
            reasons.append("evidence_time_stop")
        de_risk_min_hours = float(policy.get("de_risk_min_hours", 0.0) or 0.0)
        de_risk_min_pnl_ratio = float(
            policy.get("de_risk_min_pnl_ratio", 0.0) or 0.0
        )
        if (
            pnl_ratio >= de_risk_min_pnl_ratio > 0
            and hours_held >= de_risk_min_hours > 0
            and "evidence_time_stop" not in reasons
        ):
            reasons.append("evidence_de_risk")
        return reasons

    @staticmethod
    def position_protection_reasons(
        position: dict,
        current_price: float,
        initial_quantity: float,
    ) -> list[str]:
        direction = str(position.get("direction") or "LONG").upper()
        current_quantity = float(position["quantity"])
        stop_loss = position.get("stop_loss")
        take_profit = position.get("take_profit")
        reasons: list[str] = []

        if stop_loss is not None:
            stop_price = float(stop_loss)
            stop_hit = current_price <= stop_price
            if direction == "SHORT":
                stop_hit = current_price >= stop_price
            if stop_hit:
                reasons.append("fixed_stop_loss")

        if take_profit is not None:
            target_price = float(take_profit)
            target_hit = current_price >= target_price
            if direction == "SHORT":
                target_hit = current_price <= target_price
            if target_hit:
                reasons.append(
                    "take_profit_1"
                    if current_quantity >= initial_quantity - 1e-8
                    else "take_profit_2"
                )

        return reasons
