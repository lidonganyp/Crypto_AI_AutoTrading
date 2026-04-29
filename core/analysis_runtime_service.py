"""Active and shadow symbol execution passes for the runtime loop."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from config import Settings, get_settings
from core.models import RiskCheckResult, SignalDirection
from core.storage import Storage
from core.strategy_profile import (
    THESIS_REASON_EXTREME_FEAR,
    THESIS_REASON_HIGH_CONVICTION,
    derive_entry_profile,
)


class AnalysisRuntimeService:
    """Run active-entry and shadow-observation passes outside of engine.run_once()."""

    FAST_ALPHA_REGIME_REVERSAL_CLOSE_OVERRIDE_MIN_REVIEW_SCORE = -0.60
    FAST_ALPHA_EXPERIENCE_HARD_BLOCK_MAX_AVG_OUTCOME_PCT = -0.12

    OFFENSIVE_REVIEW_REASONS = THESIS_REASON_EXTREME_FEAR
    SOFT_REVIEW_SUPPORT_REASONS = {
        "xgb_strong",
        "xgb_pass",
        "liquidity_supportive",
        "trend_supportive",
        "extreme_fear_offensive_setup",
        "extreme_fear_quant_override",
        "extreme_fear_quant_override_open",
    }
    SOFT_REVIEW_EXTREME_FEAR_CONTEXT_REASONS = {
        "regime_extreme_fear",
        "regime_extreme_fear_discounted",
        "fear_greed_extreme_fear",
        "fear_greed_extreme_fear_discounted",
    }
    SOFT_REVIEW_EXTREME_FEAR_ALLOWED_REASONS = {
        "extreme_fear_offensive_setup",
        "extreme_fear_quant_override",
        "extreme_fear_quant_override_open",
        "liquidity_supportive",
        "trend_supportive",
    }
    FAST_ALPHA_SUPPORT_REASONS = {
        "xgb_pass",
        "xgb_strong",
        "liquidity_supportive",
        "trend_supportive",
        *THESIS_REASON_EXTREME_FEAR,
    }
    FAST_ALPHA_SHORT_HORIZON_SPECIAL_SUPPORT_REASONS = {
        "extreme_fear_offensive_setup",
        "extreme_fear_offensive_open",
        "extreme_fear_quant_override",
        "extreme_fear_quant_override_open",
        "quant_repairing_setup",
        "quant_repairing_setup_open",
        "core_extreme_fear_liquidity_repair",
        "core_extreme_fear_liquidity_repair_open",
    }
    FAST_ALPHA_SHORT_HORIZON_NON_OVERRIDABLE_REASONS = {
        "setup_auto_pause",
        "news_event_risk",
        "fallback_research_penalty",
    }

    def __init__(
        self,
        storage: Storage,
        settings: Settings | None = None,
        *,
        decision_engine,
        risk,
        executor,
        notifier,
        prepare_symbol_snapshot,
        detect_abnormal_move,
        evaluate_ab_test,
        persist_analysis,
        record_shadow_trade_if_blocked,
        compose_trade_rationale,
        get_positions,
        account_state,
        get_circuit_breaker_reason,
        performance_getter=None,
        position_value_adjuster=None,
    ):
        self.storage = storage
        self.settings = settings or get_settings()
        self.decision_engine = decision_engine
        self.risk = risk
        self.executor = executor
        self.notifier = notifier
        self.prepare_symbol_snapshot = prepare_symbol_snapshot
        self.detect_abnormal_move = detect_abnormal_move
        self.evaluate_ab_test = evaluate_ab_test
        self.persist_analysis = persist_analysis
        self.record_shadow_trade_if_blocked = record_shadow_trade_if_blocked
        self.compose_trade_rationale = compose_trade_rationale
        self.get_positions = get_positions
        self.account_state = account_state
        self.get_circuit_breaker_reason = get_circuit_breaker_reason
        self.performance_getter = performance_getter
        self.position_value_adjuster = position_value_adjuster
        self._short_horizon_adaptive_profile_cache: dict | None = None

    def _risk_supports_parameter(self, parameter_name: str) -> bool:
        try:
            signature = inspect.signature(self.risk.can_open_position)
        except (TypeError, ValueError):
            return True
        parameters = signature.parameters.values()
        return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters) or (
            parameter_name in signature.parameters
        )

    @staticmethod
    def _storage_symbol_variants(symbol: str) -> list[str]:
        variants = [str(symbol)]
        if ":USDT" in symbol:
            variants.append(symbol.replace(":USDT", ""))
        else:
            variants.append(f"{symbol}:USDT")
        return list(dict.fromkeys(variants))

    def _stored_close_series(
        self,
        symbol: str,
        *,
        timeframe: str = "4h",
        limit: int = 120,
    ) -> list[float]:
        for storage_symbol in self._storage_symbol_variants(symbol):
            candles = self.storage.get_ohlcv(storage_symbol, timeframe, limit=limit)
            if not candles:
                continue
            candles = sorted(candles, key=lambda candle: int(candle["timestamp"]))
            closes = [
                float(candle["close"])
                for candle in candles
                if candle.get("close") is not None
            ]
            if closes:
                return closes
        return []

    def _build_correlation_price_data(
        self,
        *,
        symbol: str,
        positions: list[dict],
    ) -> dict[str, list[float]]:
        price_data: dict[str, list[float]] = {}
        symbols = [symbol, *[str(position.get("symbol") or "") for position in positions]]
        for candidate in dict.fromkeys(symbols):
            if not candidate:
                continue
            closes = self._stored_close_series(candidate)
            if len(closes) >= 10:
                price_data[candidate] = closes
        return price_data

    def _can_open_position(
        self,
        *,
        account,
        positions: list[dict],
        symbol: str,
        atr: float,
        entry_price: float,
        liquidity_ratio: float,
        liquidity_floor_override: float | None = None,
        consecutive_wins: int,
        consecutive_losses: int,
        performance_snapshot,
    ) -> RiskCheckResult:
        kwargs = {
            "account": account,
            "positions": positions,
            "symbol": symbol,
            "atr": atr,
            "entry_price": entry_price,
            "liquidity_ratio": liquidity_ratio,
            "consecutive_wins": consecutive_wins,
            "consecutive_losses": consecutive_losses,
        }
        if self._risk_supports_parameter("liquidity_floor_override"):
            kwargs["liquidity_floor_override"] = liquidity_floor_override
        if self._risk_supports_parameter("performance_snapshot"):
            kwargs["performance_snapshot"] = performance_snapshot
        if self._risk_supports_parameter("correlation_price_data"):
            kwargs["correlation_price_data"] = self._build_correlation_price_data(
                symbol=symbol,
                positions=positions,
            )
        return self.risk.can_open_position(**kwargs)

    @staticmethod
    def _execution_metadata(
        *,
        prediction,
        final_score: float,
        pipeline_mode: str,
        decision_reason: str,
        review=None,
        extra: dict | None = None,
    ) -> dict:
        timestamp = getattr(prediction, "timestamp", None)
        review_reasons = list(getattr(review, "reasons", []) or [])
        review_score = float(getattr(review, "review_score", 0.0) or 0.0)
        reviewed_action = str(getattr(review, "reviewed_action", "") or "")
        raw_action = str(getattr(review, "raw_action", "") or "")
        thesis = derive_entry_profile(
            pipeline_mode=pipeline_mode,
            raw_action=raw_action,
            reviewed_action=reviewed_action,
            review_score=review_score,
            review_reasons=review_reasons,
        )
        return {
            "model_id": str(getattr(prediction, "model_id", "") or ""),
            "model_version": str(getattr(prediction, "model_version", "") or ""),
            "pipeline_mode": pipeline_mode,
            "prediction_timestamp": (
                timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp or "")
            ),
            "up_probability": float(getattr(prediction, "up_probability", 0.0) or 0.0),
            "final_score": float(final_score or 0.0),
            "decision_reason": decision_reason,
            "raw_action": raw_action,
            "reviewed_action": reviewed_action,
            "approval_rating": str(getattr(review, "approval_rating", "") or ""),
            "review_score": review_score,
            "review_reasons": review_reasons,
            "setup_profile": dict(getattr(review, "setup_profile", {}) or {}),
            "setup_performance": dict(getattr(review, "setup_performance", {}) or {}),
            "entry_thesis": thesis.entry_thesis,
            "entry_thesis_strength": thesis.entry_thesis_strength,
            "entry_open_bias": thesis.entry_open_bias,
            **dict(extra or {}),
        }

    @staticmethod
    def _risk_metadata_extra(risk_result) -> dict:
        return {
            "portfolio_heat_factor": float(
                getattr(risk_result, "portfolio_heat_factor", 1.0) or 1.0
            ),
            "effective_max_total_exposure_pct": float(
                getattr(risk_result, "effective_max_total_exposure_pct", 0.0) or 0.0
            ),
            "effective_max_positions": int(
                getattr(risk_result, "effective_max_positions", 0) or 0
            ),
            "correlation_position_factor": float(
                getattr(risk_result, "correlation_position_factor", 1.0) or 1.0
            ),
            "correlation_effective_exposure_pct": float(
                getattr(risk_result, "correlation_effective_exposure_pct", 0.0) or 0.0
            ),
            "correlation_crowded_symbols": list(
                getattr(risk_result, "correlation_crowded_symbols", []) or []
            ),
            "liquidity_floor_used": float(
                getattr(risk_result, "liquidity_floor_used", 0.0) or 0.0
            ),
            "observed_liquidity_ratio": float(
                getattr(risk_result, "observed_liquidity_ratio", 0.0) or 0.0
            ),
        }

    def _adjust_position_value(
        self,
        *,
        symbol: str,
        pipeline_mode: str,
        base_position_value: float,
        now=None,
        prediction=None,
        decision=None,
    ) -> dict:
        position_value = max(0.0, float(base_position_value or 0.0))
        if self.position_value_adjuster is None:
            return {
                "position_value": position_value,
                "scale": 1.0,
                "source": "none",
                "reason": "",
            }
        payload = self.position_value_adjuster(
            symbol=symbol,
            pipeline_mode=pipeline_mode,
            base_position_value=position_value,
            now=now,
            prediction=prediction,
            decision=decision,
        )
        if not isinstance(payload, dict):
            return {
                "position_value": position_value,
                "scale": 1.0,
                "source": "none",
                "reason": "",
            }
        adjusted_position_value = max(
            0.0,
            float(payload.get("position_value", position_value) or 0.0),
        )
        return {
            "position_value": adjusted_position_value,
            "scale": float(payload.get("scale", 1.0) or 1.0),
            "source": str(payload.get("source", "none") or "none"),
            "reason": str(payload.get("reason", "") or ""),
        }

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return str(symbol or "").strip().upper().replace(" ", "")

    def _fast_alpha_enabled_for_symbol(self, symbol: str) -> bool:
        if not bool(self.settings.strategy.fast_alpha_enabled):
            return False
        enabled_symbols = {
            self._normalize_symbol(item)
            for item in (self.settings.strategy.fast_alpha_symbols or [])
            if str(item).strip()
        }
        return self._normalize_symbol(symbol) in enabled_symbols

    def _fast_alpha_min_probability_pct(self, symbol: str) -> float:
        normalized = self._normalize_symbol(symbol)
        if normalized == "ETH/USDT":
            return float(
                getattr(
                    self.settings.strategy,
                    "fast_alpha_eth_min_probability_pct",
                    self.settings.strategy.fast_alpha_min_probability_pct,
                )
                or self.settings.strategy.fast_alpha_min_probability_pct
            )
        return float(self.settings.strategy.fast_alpha_min_probability_pct)

    def _core_extreme_fear_fast_alpha_liquidity_floor(
        self,
        *,
        symbol: str,
        review_reasons: set[str],
        up_probability: float,
        fast_alpha_liquidity_floor: float,
    ) -> float:
        if str(getattr(self.settings.app, "runtime_mode", "") or "").lower() != "paper":
            return fast_alpha_liquidity_floor
        core_symbols = {
            self._normalize_symbol(item)
            for item in (self.settings.exchange.core_symbols or [])
            if str(item).strip()
        }
        if self._normalize_symbol(symbol) not in core_symbols:
            return fast_alpha_liquidity_floor
        if up_probability < max(
            0.70,
            float(self.decision_engine.xgboost_threshold) - 0.03,
        ):
            return fast_alpha_liquidity_floor
        if "trend_supportive" not in review_reasons:
            return fast_alpha_liquidity_floor
        if not (
            review_reasons
            & {
                "regime_extreme_fear",
                "regime_extreme_fear_discounted",
                "regime_extreme_fear_core_repair_discounted",
                "fear_greed_extreme_fear",
                "fear_greed_extreme_fear_discounted",
                "fear_greed_extreme_fear_core_repair_discounted",
                "core_extreme_fear_liquidity_repair",
                "core_extreme_fear_liquidity_repair_open",
            }
        ):
            return fast_alpha_liquidity_floor
        discounted_floor = max(
            0.18,
            float(
                getattr(self.settings.strategy, "adaptive_liquidity_floor_min_ratio", 0.35)
            )
            * 0.55,
        )
        return min(float(fast_alpha_liquidity_floor), discounted_floor)

    def _core_extreme_fear_fast_alpha_min_final_score(
        self,
        *,
        symbol: str,
        review_reasons: set[str],
        up_probability: float,
        min_final_score: float,
    ) -> float:
        if str(getattr(self.settings.app, "runtime_mode", "") or "").lower() != "paper":
            return min_final_score
        core_symbols = {
            self._normalize_symbol(item)
            for item in (self.settings.exchange.core_symbols or [])
            if str(item).strip()
        }
        if self._normalize_symbol(symbol) not in core_symbols:
            return min_final_score
        if up_probability < max(
            0.70,
            float(self.decision_engine.xgboost_threshold) - 0.03,
        ):
            return min_final_score
        if "core_extreme_fear_liquidity_repair" not in review_reasons:
            return min_final_score
        return min(
            float(min_final_score),
            max(0.50, float(min_final_score) - 0.02),
        )

    def _record_fast_alpha_blocked(
        self,
        *,
        symbol: str,
        reason: str,
        prediction=None,
        review=None,
        final_score: float | None = None,
        extra: dict | None = None,
    ) -> None:
        if not self._fast_alpha_enabled_for_symbol(symbol):
            return
        payload = {
            "reason": str(reason or "unknown"),
            "up_probability": float(
                getattr(prediction, "up_probability", 0.0) or 0.0
            ),
            "final_score": float(final_score or 0.0),
            "review_score": float(getattr(review, "review_score", 0.0) or 0.0),
            "reviewed_action": str(getattr(review, "reviewed_action", "") or ""),
            "raw_action": str(getattr(review, "raw_action", "") or ""),
            "review_reasons": list(getattr(review, "reasons", []) or []),
        }
        payload.update(dict(extra or {}))
        self.storage.insert_execution_event(
            "fast_alpha_blocked",
            symbol,
            payload,
        )

    @staticmethod
    def _review_reason_metric_value(
        review_reasons: set[str],
        prefix: str,
    ) -> float | None:
        for reason in review_reasons:
            if not reason.startswith(prefix):
                continue
            try:
                return float(reason.rsplit("_", 1)[-1])
            except ValueError:
                return None
        return None

    @staticmethod
    def _profit_factor(values: list[float]) -> float:
        wins = [value for value in values if value > 0]
        losses = [value for value in values if value < 0]
        if losses and abs(sum(losses)) > 1e-12:
            return sum(wins) / abs(sum(losses))
        return 5.0 if wins else 0.0

    @staticmethod
    def _returns_max_drawdown_pct(returns_pct: list[float]) -> float:
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for return_pct in returns_pct:
            equity *= 1.0 + float(return_pct) / 100.0
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        return max_drawdown * 100.0

    def _pipeline_mode_recent_summary(
        self,
        *,
        pipeline_modes: list[str] | tuple[str, ...],
        limit: int,
    ) -> dict[str, float | int | dict[str, int]]:
        normalized_modes = [
            str(mode or "").strip()
            for mode in (pipeline_modes or [])
            if str(mode or "").strip()
        ]
        if not normalized_modes:
            return {
                "closed_trade_count": 0,
                "win_rate_pct": 0.0,
                "expectancy_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "mode_counts": {},
            }

        placeholders = ",".join("?" for _ in normalized_modes)
        with self.storage._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    t.id,
                    t.entry_price,
                    t.quantity,
                    t.initial_quantity,
                    t.pnl,
                    t.pnl_pct,
                    COALESCE(json_extract(t.metadata_json, '$.pipeline_mode'), 'execution')
                        AS pipeline_mode,
                    COUNT(l.id) AS ledger_count,
                    COALESCE(SUM(l.net_pnl), t.pnl, 0.0) AS net_pnl,
                    COALESCE(t.exit_time, t.entry_time) AS closed_at
                FROM trades t
                LEFT JOIN pnl_ledger l
                    ON l.trade_id = t.id
                WHERE t.status='closed'
                  AND COALESCE(json_extract(t.metadata_json, '$.pipeline_mode'), 'execution')
                      IN ({placeholders})
                GROUP BY
                    t.id,
                    t.entry_price,
                    t.quantity,
                    t.initial_quantity,
                    t.pnl,
                    t.pnl_pct,
                    COALESCE(json_extract(t.metadata_json, '$.pipeline_mode'), 'execution'),
                    COALESCE(t.exit_time, t.entry_time)
                ORDER BY closed_at DESC
                LIMIT ?
                """,
                (*normalized_modes, max(1, int(limit))),
            ).fetchall()

        if not rows:
            return {
                "closed_trade_count": 0,
                "win_rate_pct": 0.0,
                "expectancy_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "mode_counts": {},
            }

        ordered_returns: list[float] = []
        ordered_pnls: list[float] = []
        mode_counts: dict[str, int] = {}
        for row in reversed(rows):
            base_qty = float(row["initial_quantity"] or row["quantity"] or 0.0)
            base_notional = float(row["entry_price"] or 0.0) * base_qty
            ledger_count = int(row["ledger_count"] or 0)
            net_pnl = float(row["net_pnl"] or row["pnl"] or 0.0)
            if ledger_count > 0 and base_notional > 0:
                net_return_pct = net_pnl / base_notional * 100.0
            else:
                net_return_pct = float(row["pnl_pct"] or 0.0)
            ordered_returns.append(float(net_return_pct))
            ordered_pnls.append(net_pnl)
            mode = str(row["pipeline_mode"] or "execution")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        closed_trade_count = len(ordered_returns)
        win_rate_pct = (
            sum(1 for value in ordered_returns if value > 0) / closed_trade_count * 100.0
            if closed_trade_count
            else 0.0
        )
        expectancy_pct = (
            sum(ordered_returns) / closed_trade_count if closed_trade_count else 0.0
        )
        return {
            "closed_trade_count": closed_trade_count,
            "win_rate_pct": win_rate_pct,
            "expectancy_pct": expectancy_pct,
            "profit_factor": self._profit_factor(ordered_pnls),
            "max_drawdown_pct": self._returns_max_drawdown_pct(ordered_returns),
            "mode_counts": mode_counts,
        }

    def _short_horizon_adaptive_profile(self) -> dict[str, float | int | bool | str | dict[str, int]]:
        cached = self._short_horizon_adaptive_profile_cache
        if isinstance(cached, dict):
            return cached

        default_profile = {
            "enabled": bool(self.settings.strategy.short_horizon_adaptive_enabled),
            "positive_edge": False,
            "pause_entries": False,
            "reason": "",
            "closed_trade_count": 0,
            "expectancy_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "probability_discount_pct": 0.0,
            "final_score_discount": 0.0,
            "review_score_discount": 0.0,
            "mode_counts": {},
        }
        if not bool(self.settings.strategy.short_horizon_adaptive_enabled):
            self._short_horizon_adaptive_profile_cache = default_profile
            return default_profile

        lookback = max(
            1,
            int(getattr(self.settings.strategy, "short_horizon_adaptive_lookback_trades", 20) or 20),
        )
        combined_summary: dict[str, float | int | dict[str, int]] = {}
        performance = self.performance_getter() if self.performance_getter else None
        build_summary = getattr(performance, "build_pipeline_mode_summary", None)
        if callable(build_summary):
            try:
                summary = build_summary(
                    pipeline_modes=["fast_alpha", "paper_canary"],
                    limit=lookback,
                )
            except TypeError:
                summary = build_summary(["fast_alpha", "paper_canary"], lookback)
            if isinstance(summary, dict):
                combined_summary = dict(summary.get("_combined") or {})
        if not combined_summary:
            combined_summary = self._pipeline_mode_recent_summary(
                pipeline_modes=("fast_alpha", "paper_canary"),
                limit=lookback,
            )

        closed_trade_count = int(combined_summary.get("closed_trade_count", 0) or 0)
        expectancy_pct = float(combined_summary.get("expectancy_pct", 0.0) or 0.0)
        profit_factor = float(combined_summary.get("profit_factor", 0.0) or 0.0)
        max_drawdown_pct = float(
            combined_summary.get("max_drawdown_pct", 0.0) or 0.0
        )
        win_rate_pct = float(combined_summary.get("win_rate_pct", 0.0) or 0.0)
        min_closed_trades = max(
            1,
            int(
                getattr(
                    self.settings.strategy,
                    "short_horizon_adaptive_min_closed_trades",
                    6,
                )
                or 6
            ),
        )
        positive_edge = False
        pause_entries = False
        reason = ""
        if closed_trade_count < min_closed_trades:
            reason = "insufficient_samples"
        else:
            positive_edge = (
                expectancy_pct
                >= float(
                    getattr(
                        self.settings.strategy,
                        "short_horizon_adaptive_positive_expectancy_pct",
                        0.12,
                    )
                    or 0.12
                )
                and (
                    profit_factor == 0.0
                    or profit_factor
                    >= float(
                        getattr(
                            self.settings.strategy,
                            "short_horizon_adaptive_positive_profit_factor",
                            1.10,
                        )
                        or 1.10
                    )
                )
                and max_drawdown_pct
                <= float(
                    getattr(
                        self.settings.strategy,
                        "short_horizon_adaptive_max_drawdown_pct",
                        4.0,
                    )
                    or 4.0
                )
            )
            pause_entries = (
                expectancy_pct
                <= float(
                    getattr(
                        self.settings.strategy,
                        "short_horizon_adaptive_negative_expectancy_pct",
                        -0.08,
                    )
                    or -0.08
                )
                or (
                    0.0 < profit_factor
                    < float(
                        getattr(
                            self.settings.strategy,
                            "short_horizon_adaptive_negative_profit_factor",
                            0.95,
                        )
                        or 0.95
                    )
                )
                or max_drawdown_pct
                > float(
                    getattr(
                        self.settings.strategy,
                        "short_horizon_adaptive_max_drawdown_pct",
                        4.0,
                    )
                    or 4.0
                )
            )
            if positive_edge:
                pause_entries = False
                reason = "positive_edge_expand"
            elif pause_entries:
                reason = "negative_edge_pause"

        profile = {
            "enabled": True,
            "positive_edge": positive_edge,
            "pause_entries": pause_entries,
            "reason": reason,
            "closed_trade_count": closed_trade_count,
            "expectancy_pct": expectancy_pct,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown_pct,
            "win_rate_pct": win_rate_pct,
            "probability_discount_pct": (
                float(
                    getattr(
                        self.settings.strategy,
                        "short_horizon_adaptive_probability_discount_pct",
                        0.04,
                    )
                    or 0.04
                )
                if positive_edge
                else 0.0
            ),
            "final_score_discount": (
                float(
                    getattr(
                        self.settings.strategy,
                        "short_horizon_adaptive_final_score_discount",
                        0.03,
                    )
                    or 0.03
                )
                if positive_edge
                else 0.0
            ),
            "review_score_discount": (
                float(
                    getattr(
                        self.settings.strategy,
                        "short_horizon_adaptive_review_score_discount",
                        0.05,
                    )
                    or 0.05
                )
                if positive_edge
                else 0.0
            ),
            "mode_counts": dict(combined_summary.get("mode_counts", {}) or {}),
        }
        self._short_horizon_adaptive_profile_cache = profile
        return profile

    def _fast_alpha_short_horizon_trade_feedback(
        self,
        *,
        symbol: str,
    ) -> dict[str, float | int | bool | str]:
        if not bool(
            getattr(
                self.settings.strategy,
                "fast_alpha_short_horizon_trade_feedback_enabled",
                True,
            )
        ):
            return {
                "enabled": False,
                "allow_softening": True,
                "status": "feature_off",
                "reason": "feature_off",
                "closed_trade_count": 0,
                "lookback_trades": 0,
                "expectancy_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
            }

        lookback = max(
            1,
            int(
                getattr(
                    self.settings.strategy,
                    "fast_alpha_short_horizon_trade_feedback_lookback_trades",
                    8,
                )
                or 8
            ),
        )
        min_closed_trades = max(
            1,
            int(
                getattr(
                    self.settings.strategy,
                    "fast_alpha_short_horizon_trade_feedback_min_closed_trades",
                    3,
                )
                or 3
            ),
        )
        with self.storage._conn() as conn:
            rows = conn.execute(
                """
                SELECT pnl, pnl_pct
                FROM trades
                WHERE status='closed'
                  AND symbol = ?
                  AND json_extract(metadata_json, '$.pipeline_mode')='fast_alpha'
                  AND COALESCE(json_extract(metadata_json, '$.fast_alpha_review_policy_reason'), '') != ''
                ORDER BY COALESCE(exit_time, entry_time) DESC
                LIMIT ?
                """,
                (symbol, lookback),
            ).fetchall()

        pnls = [float(row["pnl"] or 0.0) for row in rows]
        returns_pct = [float(row["pnl_pct"] or 0.0) for row in rows]
        closed_trade_count = len(rows)
        if closed_trade_count < min_closed_trades:
            warm_scale = max(
                0.0,
                min(
                    1.0,
                    float(
                        getattr(
                            self.settings.strategy,
                            "fast_alpha_short_horizon_trade_feedback_warming_scale",
                            0.70,
                        )
                        or 0.70
                    ),
                ),
            )
            return {
                "enabled": True,
                "allow_softening": True,
                "status": "insufficient_samples",
                "reason": "insufficient_samples",
                "softening_scale": warm_scale,
                "closed_trade_count": closed_trade_count,
                "lookback_trades": lookback,
                "expectancy_pct": (sum(returns_pct) / closed_trade_count if closed_trade_count else 0.0),
                "profit_factor": self._profit_factor(pnls),
                "max_drawdown_pct": self._returns_max_drawdown_pct(returns_pct),
            }

        expectancy_pct = sum(returns_pct) / closed_trade_count if closed_trade_count else 0.0
        profit_factor = self._profit_factor(pnls)
        max_drawdown_pct = self._returns_max_drawdown_pct(returns_pct)
        negative_edge = (
            expectancy_pct
            <= float(
                getattr(
                    self.settings.strategy,
                    "fast_alpha_short_horizon_trade_feedback_negative_expectancy_pct",
                    -0.05,
                )
                or -0.05
            )
            or (
                0.0 < profit_factor
                < float(
                    getattr(
                        self.settings.strategy,
                        "fast_alpha_short_horizon_trade_feedback_negative_profit_factor",
                        0.95,
                    )
                    or 0.95
                )
            )
            or max_drawdown_pct
            > float(
                getattr(
                    self.settings.strategy,
                    "fast_alpha_short_horizon_trade_feedback_negative_max_drawdown_pct",
                    3.0,
                )
                or 3.0
            )
        )
        healthy_scale = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.settings.strategy,
                        "fast_alpha_short_horizon_trade_feedback_healthy_scale",
                        1.0,
                    )
                    or 1.0
                ),
            ),
        )
        scale = 0.0
        status = "disabled_negative_edge" if negative_edge else "healthy"
        if not negative_edge:
            positive_ratio = min(1.0, max(0.0, expectancy_pct / 0.10))
            pf_ratio = 1.0 if profit_factor <= 0.0 else min(1.0, max(0.0, profit_factor / 1.20))
            drawdown_ratio = max(0.0, min(1.0, 1.0 - (max_drawdown_pct / max(1.0, float(getattr(self.settings.strategy, "fast_alpha_short_horizon_trade_feedback_negative_max_drawdown_pct", 3.0) or 3.0)))))
            scale = healthy_scale * max(0.35, min(1.0, (positive_ratio + pf_ratio + drawdown_ratio) / 3.0))
        return {
            "enabled": True,
            "allow_softening": not negative_edge,
            "status": status,
            "reason": status,
            "softening_scale": scale,
            "closed_trade_count": closed_trade_count,
            "lookback_trades": lookback,
            "expectancy_pct": expectancy_pct,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown_pct,
        }

    def _fast_alpha_short_horizon_review_policy(
        self,
        *,
        symbol: str,
        prediction,
        review,
        validation,
        adaptive_profile: dict[str, float | int | bool | str | dict[str, int]],
    ) -> dict[str, float | bool | str | list[str]]:
        raw_action = str(getattr(review, "raw_action", "") or "").upper()
        reviewed_action = str(getattr(review, "reviewed_action", "") or "").upper()
        review_score = float(getattr(review, "review_score", 0.0) or 0.0)
        review_reasons = {
            str(reason).strip()
            for reason in (getattr(review, "reasons", []) or [])
            if str(reason).strip()
        }
        trade_feedback = self._fast_alpha_short_horizon_trade_feedback(symbol=symbol)
        policy = {
            "enabled": False,
            "effective_review_score": review_score,
            "effective_reviewed_action": reviewed_action,
            "relaxed_reasons": [],
            "reason": "",
            "trade_feedback": trade_feedback,
        }
        if not bool(
            getattr(
                self.settings.strategy,
                "fast_alpha_short_horizon_review_policy_enabled",
                True,
            )
        ):
            return policy
        if bool(adaptive_profile.get("pause_entries", False)):
            return policy
        if not getattr(validation, "ok", True):
            return policy
        if raw_action == "CLOSE":
            return policy
        if any(
            reason in review_reasons
            for reason in self.FAST_ALPHA_SHORT_HORIZON_NON_OVERRIDABLE_REASONS
        ):
            return policy
        if any(
            reason.startswith("realized_setup_negative_expectancy")
            for reason in review_reasons
        ):
            return policy
        if not bool(trade_feedback.get("allow_softening", True)):
            policy["reason"] = (
                "disabled_by_softened_trade_edge"
                f"|expectancy={float(trade_feedback.get('expectancy_pct', 0.0) or 0.0):+.2f}"
                f"|profit_factor={float(trade_feedback.get('profit_factor', 0.0) or 0.0):.2f}"
                f"|max_drawdown={float(trade_feedback.get('max_drawdown_pct', 0.0) or 0.0):.2f}"
                f"|count={int(trade_feedback.get('closed_trade_count', 0) or 0)}"
            )
            return policy

        warming_up = str(adaptive_profile.get("reason", "") or "") == "insufficient_samples"
        if not (bool(adaptive_profile.get("positive_edge", False)) or warming_up):
            return policy

        positive_support_count = len(review_reasons & self.FAST_ALPHA_SUPPORT_REASONS)
        special_support = bool(
            review_reasons & self.FAST_ALPHA_SHORT_HORIZON_SPECIAL_SUPPORT_REASONS
        )
        if positive_support_count < 2 and not special_support:
            return policy

        softening_scale = max(
            0.0,
            min(
                1.0,
                float(trade_feedback.get("softening_scale", 1.0) or 1.0),
            ),
        )
        effective_review_score = review_score
        relaxed_reasons: list[str] = []
        relaxed_reason_notes: list[str] = []

        if "setup_negative_expectancy" in review_reasons:
            setup_avg_outcome = self._review_reason_metric_value(
                review_reasons,
                "setup_avg_outcome_",
            )
            if (
                setup_avg_outcome is not None
                and setup_avg_outcome
                >= float(
                    getattr(
                        self.settings.strategy,
                        "fast_alpha_short_horizon_setup_avg_outcome_floor_pct",
                        -0.18,
                    )
                    or -0.18
                )
            ):
                effective_review_score += float(
                    getattr(
                        self.settings.strategy,
                        "fast_alpha_short_horizon_setup_score_bonus",
                        0.10,
                    )
                    or 0.10
                ) * softening_scale
                relaxed_reasons.append("setup_negative_expectancy")
                relaxed_reason_notes.append(
                    f"setup_avg_outcome={setup_avg_outcome:.2f}"
                )

        if "experience_negative_setup" in review_reasons:
            experience_avg_outcome = self._review_reason_metric_value(
                review_reasons,
                "experience_avg_outcome_",
            )
            if (
                experience_avg_outcome is not None
                and experience_avg_outcome
                >= float(
                    getattr(
                        self.settings.strategy,
                        "fast_alpha_short_horizon_experience_avg_outcome_floor_pct",
                        -0.14,
                    )
                    or -0.14
                )
            ):
                effective_review_score += float(
                    getattr(
                        self.settings.strategy,
                        "fast_alpha_short_horizon_experience_score_bonus",
                        0.08,
                    )
                    or 0.08
                ) * softening_scale
                relaxed_reasons.append("experience_negative_setup")
                relaxed_reason_notes.append(
                    f"experience_avg_outcome={experience_avg_outcome:.2f}"
                )

        if "risk_warning_present" in review_reasons:
            reviewed_insight = getattr(review, "reviewed_insight", None)
            reviewed_warnings = list(
                getattr(reviewed_insight, "risk_warning", []) or []
            )
            warning_count = sum(
                1
                for warning in reviewed_warnings
                if str(warning).strip() != "manager_not_approved"
            )
            max_risk_warnings = max(
                0,
                int(
                    getattr(
                        self.settings.strategy,
                        "fast_alpha_short_horizon_max_risk_warnings",
                        1,
                    )
                    or 1
                ),
            )
            if warning_count <= max_risk_warnings:
                effective_review_score += float(
                    getattr(
                        self.settings.strategy,
                        "fast_alpha_short_horizon_risk_warning_score_bonus",
                        0.05,
                    )
                    or 0.05
                ) * softening_scale
                relaxed_reasons.append("risk_warning_present")
                relaxed_reason_notes.append(f"risk_warnings={warning_count}")

        if not relaxed_reasons:
            return policy

        effective_reviewed_action = reviewed_action
        if reviewed_action == "CLOSE" and effective_review_score >= -0.02:
            effective_reviewed_action = "HOLD"
        elif reviewed_action == "HOLD" and special_support and raw_action == "OPEN_LONG":
            effective_reviewed_action = "HOLD"

        policy.update(
            {
                "enabled": True,
                "effective_review_score": effective_review_score,
                "effective_reviewed_action": effective_reviewed_action,
                "relaxed_reasons": relaxed_reasons,
                "reason": (
                    "positive_edge_soften"
                    if bool(adaptive_profile.get("positive_edge", False))
                    else "warming_up_soften"
                )
                + f"|scale={softening_scale:.2f}"
                + (
                    f"|{','.join(relaxed_reason_notes)}"
                    if relaxed_reason_notes
                    else ""
                ),
            }
        )
        return policy

    @classmethod
    def _fast_alpha_experience_hard_block(cls, review_reasons: set[str]) -> bool:
        if "experience_negative_setup" not in review_reasons:
            return False
        for reason in review_reasons:
            if not reason.startswith("experience_avg_outcome_"):
                continue
            try:
                avg_outcome = float(reason.rsplit("_", 1)[-1])
            except ValueError:
                continue
            return avg_outcome <= cls.FAST_ALPHA_EXPERIENCE_HARD_BLOCK_MAX_AVG_OUTCOME_PCT
        return True

    def _fast_alpha_gate(
        self,
        *,
        symbol: str,
        prediction,
        review,
        validation,
        features,
        risk_result,
        final_score: float,
    ) -> dict | None:
        if not self._fast_alpha_enabled_for_symbol(symbol):
            return None
        adaptive_profile = self._short_horizon_adaptive_profile()
        if bool(adaptive_profile.get("pause_entries", False)):
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="short_horizon_negative_expectancy_pause",
                prediction=prediction,
                review=review,
                final_score=final_score,
                extra={
                    "adaptive_reason": str(adaptive_profile.get("reason", "") or ""),
                    "adaptive_closed_trade_count": int(
                        adaptive_profile.get("closed_trade_count", 0) or 0
                    ),
                    "adaptive_expectancy_pct": float(
                        adaptive_profile.get("expectancy_pct", 0.0) or 0.0
                    ),
                    "adaptive_profit_factor": float(
                        adaptive_profile.get("profit_factor", 0.0) or 0.0
                    ),
                    "adaptive_max_drawdown_pct": float(
                        adaptive_profile.get("max_drawdown_pct", 0.0) or 0.0
                    ),
                },
            )
            return None
        if not getattr(validation, "ok", True):
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="validation_not_ok",
                prediction=prediction,
                review=review,
                final_score=final_score,
                extra={"validation_reason": getattr(validation, "reason", "")},
            )
            return None

        reviewed_action = str(getattr(review, "reviewed_action", "") or "").upper()
        raw_action = str(getattr(review, "raw_action", "") or "").upper()
        if raw_action == "CLOSE":
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="raw_action_close",
                prediction=prediction,
                review=review,
                final_score=final_score,
            )
            return None

        review_score = float(getattr(review, "review_score", 0.0) or 0.0)
        review_reasons = {
            str(reason).strip()
            for reason in (getattr(review, "reasons", []) or [])
            if str(reason).strip()
        }
        validation_reasons = {
            str(reason).strip()
            for reason in (
                ((getattr(validation, "details", {}) or {}).get("reasons", []) or [])
                if validation is not None
                else []
            )
            if str(reason).strip()
        }
        review_policy = self._fast_alpha_short_horizon_review_policy(
            symbol=symbol,
            prediction=prediction,
            review=review,
            validation=validation,
            adaptive_profile=adaptive_profile,
        )
        effective_review_score = float(
            review_policy.get("effective_review_score", review_score) or review_score
        )
        effective_reviewed_action = str(
            review_policy.get("effective_reviewed_action", reviewed_action) or reviewed_action
        ).upper()
        relaxed_review_reasons = {
            str(reason).strip()
            for reason in (review_policy.get("relaxed_reasons", []) or [])
            if str(reason).strip()
        }
        review_policy_guard_blocked = str(
            review_policy.get("reason", "") or ""
        ).startswith("disabled_by_softened_trade_edge")
        review_policy_guard_reasons = {
            "setup_negative_expectancy",
            "experience_negative_setup",
            "risk_warning_present",
        }
        if review_policy_guard_blocked and (review_reasons & review_policy_guard_reasons):
            trade_feedback = dict(review_policy.get("trade_feedback", {}) or {})
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="review_policy_guard_blocked",
                prediction=prediction,
                review=review,
                final_score=final_score,
                extra={
                    "effective_review_score": effective_review_score,
                    "effective_reviewed_action": effective_reviewed_action,
                    "review_policy_reason": str(review_policy.get("reason", "") or ""),
                    "review_policy_feedback_status": str(trade_feedback.get("status", "") or ""),
                    "review_policy_feedback_count": int(trade_feedback.get("closed_trade_count", 0) or 0),
                    "review_policy_feedback_expectancy_pct": float(trade_feedback.get("expectancy_pct", 0.0) or 0.0),
                    "review_policy_feedback_profit_factor": float(trade_feedback.get("profit_factor", 0.0) or 0.0),
                    "review_policy_feedback_max_drawdown_pct": float(trade_feedback.get("max_drawdown_pct", 0.0) or 0.0),
                },
            )
            return None
        if reviewed_action == "CLOSE":
            if effective_reviewed_action == "CLOSE":
                self._record_fast_alpha_blocked(
                    symbol=symbol,
                    reason="reviewed_action_close",
                    prediction=prediction,
                    review=review,
                    final_score=final_score,
                    extra={
                        "effective_review_score": effective_review_score,
                        "effective_reviewed_action": effective_reviewed_action,
                        "review_policy_reason": str(
                            review_policy.get("reason", "") or ""
                        ),
                    },
                )
                return None
        if effective_review_score < 0.0:
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="negative_review_score",
                prediction=prediction,
                review=review,
                final_score=final_score,
                extra={
                    "effective_review_score": effective_review_score,
                    "effective_reviewed_action": effective_reviewed_action,
                    "review_policy_reason": str(
                        review_policy.get("reason", "") or ""
                    ),
                },
            )
            return None
        hard_review_reasons = {
            "setup_auto_pause",
            "news_event_risk",
            "fallback_research_penalty",
        }
        if "setup_negative_expectancy" in review_reasons and (
            "setup_negative_expectancy" not in relaxed_review_reasons
        ):
            hard_review_reasons.add("setup_negative_expectancy")
        if review_reasons & hard_review_reasons:
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="hard_review_blocker",
                prediction=prediction,
                review=review,
                final_score=final_score,
                extra={
                    "effective_review_score": effective_review_score,
                    "effective_reviewed_action": effective_reviewed_action,
                    "review_policy_reason": str(
                        review_policy.get("reason", "") or ""
                    ),
                    "review_policy_relaxed_reasons": sorted(relaxed_review_reasons),
                },
            )
            return None
        if self._fast_alpha_experience_hard_block(review_reasons) and (
            "experience_negative_setup" not in relaxed_review_reasons
        ):
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="hard_review_blocker",
                prediction=prediction,
                review=review,
                final_score=final_score,
                extra={
                    "effective_review_score": effective_review_score,
                    "effective_reviewed_action": effective_reviewed_action,
                    "review_policy_reason": str(
                        review_policy.get("reason", "") or ""
                    ),
                    "review_policy_relaxed_reasons": sorted(relaxed_review_reasons),
                },
            )
            return None
        if any(
            reason.startswith("realized_setup_negative_expectancy")
            for reason in review_reasons
        ):
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="realized_negative_expectancy",
                prediction=prediction,
                review=review,
                final_score=final_score,
            )
            return None
        if any("validation_" in reason for reason in review_reasons):
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="validation_conflict",
                prediction=prediction,
                review=review,
                final_score=final_score,
            )
            return None

        up_probability = float(getattr(prediction, "up_probability", 0.0) or 0.0)
        min_probability_pct = max(
            0.0,
            self._fast_alpha_min_probability_pct(symbol)
            - float(adaptive_profile.get("probability_discount_pct", 0.0) or 0.0),
        )
        if up_probability < min_probability_pct:
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="probability_below_min",
                prediction=prediction,
                review=review,
                final_score=final_score,
                extra={"min_probability_pct": min_probability_pct},
            )
            return None

        final_score = float(final_score or 0.0)
        min_final_score = self._core_extreme_fear_fast_alpha_min_final_score(
            symbol=symbol,
            review_reasons=review_reasons,
            up_probability=up_probability,
            min_final_score=float(self.settings.strategy.fast_alpha_min_final_score),
        )
        min_final_score = max(
            0.0,
            float(min_final_score)
            - float(adaptive_profile.get("final_score_discount", 0.0) or 0.0),
        )
        if final_score < min_final_score:
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="final_score_below_min",
                prediction=prediction,
                review=review,
                final_score=final_score,
                extra={"min_final_score": min_final_score},
            )
            return None

        xgb_gap = max(
            0.0,
            float(self.decision_engine.xgboost_threshold) - up_probability,
        )
        if xgb_gap > float(self.settings.strategy.fast_alpha_max_xgboost_gap_pct):
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="xgboost_gap_too_wide",
                prediction=prediction,
                review=review,
                final_score=final_score,
                extra={"xgb_gap": xgb_gap},
            )
            return None

        score_gap = max(
            0.0,
            float(self.decision_engine.final_score_threshold) - final_score,
        )
        if score_gap > float(self.settings.strategy.fast_alpha_max_final_score_gap_pct):
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="final_score_gap_too_wide",
                prediction=prediction,
                review=review,
                final_score=final_score,
                extra={"score_gap": score_gap},
            )
            return None

        liquidity_ratio = float(features.values.get("volume_ratio_1h", 0.0) or 0.0)
        liquidity_floor = float(
            features.values.get(
                "adaptive_min_liquidity_ratio",
                self.settings.strategy.min_liquidity_ratio,
            )
            or self.settings.strategy.min_liquidity_ratio
        )
        fast_alpha_liquidity_floor = float(
            self.settings.strategy.fast_alpha_liquidity_floor_ratio
            if bool(self.settings.strategy.fast_alpha_liquidity_override_enabled)
            else max(0.35, liquidity_floor * 0.9)
        )
        fast_alpha_liquidity_floor = self._core_extreme_fear_fast_alpha_liquidity_floor(
            symbol=symbol,
            review_reasons=review_reasons,
            up_probability=up_probability,
            fast_alpha_liquidity_floor=fast_alpha_liquidity_floor,
        )
        if liquidity_ratio < fast_alpha_liquidity_floor:
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="liquidity_below_fast_alpha_floor",
                prediction=prediction,
                review=review,
                final_score=final_score,
                extra={
                    "liquidity_ratio": liquidity_ratio,
                    "fast_alpha_liquidity_floor": fast_alpha_liquidity_floor,
                },
            )
            return None

        positive_support = bool(review_reasons & self.FAST_ALPHA_SUPPORT_REASONS)
        min_review_score = max(
            0.0,
            float(self.settings.strategy.fast_alpha_min_review_score)
            - float(adaptive_profile.get("review_score_discount", 0.0) or 0.0),
        )
        if effective_review_score < min_review_score:
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="review_score_below_min",
                prediction=prediction,
                review=review,
                final_score=final_score,
                extra={
                    "min_review_score": min_review_score,
                    "effective_review_score": effective_review_score,
                    "effective_reviewed_action": effective_reviewed_action,
                    "review_policy_reason": str(
                        review_policy.get("reason", "") or ""
                    ),
                },
            )
            return None
        if (
            not positive_support
            and effective_reviewed_action != "OPEN_LONG"
            and raw_action != "OPEN_LONG"
        ):
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="support_missing",
                prediction=prediction,
                review=review,
                final_score=final_score,
            )
            return None

        return {
            "review_score": review_score,
            "effective_review_score": effective_review_score,
            "review_reasons": sorted(review_reasons),
            "up_probability": up_probability,
            "xgb_gap": xgb_gap,
            "score_gap": score_gap,
            "liquidity_ratio": liquidity_ratio,
            "liquidity_floor": fast_alpha_liquidity_floor,
            "min_probability_pct": min_probability_pct,
            "min_final_score": min_final_score,
            "min_review_score": min_review_score,
            "adaptive_profile": adaptive_profile,
            "effective_reviewed_action": effective_reviewed_action,
            "review_policy_reason": str(review_policy.get("reason", "") or ""),
            "review_policy_relaxed_reasons": sorted(relaxed_review_reasons),
            "review_policy_trade_feedback": dict(
                review_policy.get("trade_feedback", {}) or {}
            ),
        }

    def _maybe_open_fast_alpha(
        self,
        *,
        symbol: str,
        positions: list[dict],
        now,
        account,
        model_trading_disabled: bool,
        prediction,
        insight,
        validation,
        review,
        risk_result,
        entry_price: float,
        final_score: float,
        decision,
        features,
    ) -> bool:
        if model_trading_disabled:
            return False
        if any(position["symbol"] == symbol for position in positions):
            return False

        gate = self._fast_alpha_gate(
            symbol=symbol,
            prediction=prediction,
            review=review,
            validation=validation,
            features=features,
            risk_result=risk_result,
            final_score=final_score,
        )
        if gate is None:
            return False
        effective_risk_result = risk_result
        if not bool(getattr(risk_result, "allowed", False)):
            risk_reason = str(getattr(risk_result, "reason", "") or "")
            risk_override = self._fast_alpha_risk_override(
                account=account,
                positions=positions,
                symbol=symbol,
                features=features,
                consecutive_wins=0,
                consecutive_losses=0,
                risk_reason=risk_reason,
                liquidity_floor_override=float(gate.get("liquidity_floor") or 0.0),
            )
            if risk_override is None:
                self._record_fast_alpha_blocked(
                    symbol=symbol,
                    reason="risk_guard_blocked",
                    prediction=prediction,
                    review=review,
                    final_score=final_score,
                    extra={"risk_reason": risk_reason},
                )
                return False
            effective_risk_result = risk_override

        base_position_value = float(
            getattr(effective_risk_result, "allowed_position_value", 0.0) or 0.0
        ) * float(self.settings.strategy.fast_alpha_position_scale)
        if base_position_value <= 0:
            return False
        if not bool(getattr(risk_result, "allowed", False)):
            base_position_value *= float(
                self.settings.strategy.fast_alpha_portfolio_heat_override_scale
            )
        if base_position_value <= 0:
            return False

        position_adjustment = self._adjust_position_value(
            symbol=symbol,
            pipeline_mode="execution",
            base_position_value=base_position_value,
            now=now,
            prediction=prediction,
            decision=decision,
        )
        adjusted_position_value = float(
            position_adjustment.get("position_value", 0.0) or 0.0
        )
        if adjusted_position_value <= 0:
            self._record_fast_alpha_blocked(
                symbol=symbol,
                reason="position_value_non_positive",
                prediction=prediction,
                review=review,
                final_score=final_score,
            )
            return False

        stop_loss_pct = float(self.settings.strategy.fast_alpha_fixed_stop_loss_pct)
        take_profit_levels = list(self.settings.strategy.fast_alpha_take_profit_levels)
        horizon_hours = int(self.settings.strategy.fast_alpha_max_hold_hours)
        rationale = self.compose_trade_rationale(
            (
                f"[fast_alpha] {getattr(decision, 'reason', '')}; "
                f"xgb_gap={gate['xgb_gap']:.2f}; score_gap={gate['score_gap']:.2f}"
            ),
            review,
        )
        if float(position_adjustment.get("scale", 1.0) or 1.0) < 0.999:
            rationale = (
                f"{rationale}; evidence_scale="
                f"{float(position_adjustment.get('scale', 1.0) or 1.0):.2f}"
            )

        result = self.executor.execute_open(
            symbol=symbol,
            direction=SignalDirection.LONG,
            price=entry_price,
            confidence=final_score,
            rationale=rationale,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_levels[0],
            position_value=adjusted_position_value,
            metadata=self._execution_metadata(
                prediction=prediction,
                final_score=final_score,
                pipeline_mode="fast_alpha",
                decision_reason=str(getattr(decision, "reason", "")),
                review=review,
                extra={
                    **self._risk_metadata_extra(risk_result),
                    "fast_alpha_risk_reason": str(
                        getattr(risk_result, "reason", "") or ""
                    ),
                    "fast_alpha_enabled": True,
                    "fast_alpha_position_scale": float(
                        self.settings.strategy.fast_alpha_position_scale
                    ),
                    "fast_alpha_de_risk_hours": float(
                        self.settings.strategy.fast_alpha_de_risk_hours
                    ),
                    "fast_alpha_de_risk_pnl_ratio": float(
                        self.settings.strategy.fast_alpha_de_risk_pnl_ratio
                    ),
                    "horizon_hours": horizon_hours,
                    "fixed_stop_loss_pct": stop_loss_pct,
                    "take_profit_levels": list(take_profit_levels),
                    "model_evidence_scale": float(
                        position_adjustment.get("scale", 1.0) or 1.0
                    ),
                    "model_evidence_source": str(
                        position_adjustment.get("source", "none") or "none"
                    ),
                    "model_evidence_reason": str(
                        position_adjustment.get("reason", "") or ""
                    ),
                    "short_horizon_adaptive_reason": str(
                        (gate.get("adaptive_profile") or {}).get("reason", "") or ""
                    ),
                    "short_horizon_adaptive_positive_edge": bool(
                        (gate.get("adaptive_profile") or {}).get("positive_edge", False)
                    ),
                    "short_horizon_adaptive_closed_trade_count": int(
                        (gate.get("adaptive_profile") or {}).get(
                            "closed_trade_count",
                            0,
                        )
                        or 0
                    ),
                    "short_horizon_adaptive_expectancy_pct": float(
                        (gate.get("adaptive_profile") or {}).get(
                            "expectancy_pct",
                            0.0,
                        )
                        or 0.0
                    ),
                    "short_horizon_adaptive_profit_factor": float(
                        (gate.get("adaptive_profile") or {}).get(
                            "profit_factor",
                            0.0,
                        )
                        or 0.0
                    ),
                    "short_horizon_adaptive_max_drawdown_pct": float(
                        (gate.get("adaptive_profile") or {}).get(
                            "max_drawdown_pct",
                            0.0,
                        )
                        or 0.0
                    ),
                    "fast_alpha_min_probability_pct": float(
                        gate.get("min_probability_pct", 0.0) or 0.0
                    ),
                    "fast_alpha_min_final_score": float(
                        gate.get("min_final_score", 0.0) or 0.0
                    ),
                    "fast_alpha_min_review_score": float(
                        gate.get("min_review_score", 0.0) or 0.0
                    ),
                    "fast_alpha_effective_review_score": float(
                        gate.get("effective_review_score", 0.0) or 0.0
                    ),
                    "fast_alpha_effective_reviewed_action": str(
                        gate.get("effective_reviewed_action", "") or ""
                    ),
                    "fast_alpha_review_policy_reason": str(
                        gate.get("review_policy_reason", "") or ""
                    ),
                    "fast_alpha_review_policy_relaxed_reasons": list(
                        gate.get("review_policy_relaxed_reasons", []) or []
                    ),
                    "fast_alpha_review_policy_trade_feedback_status": str(
                        (gate.get("review_policy_trade_feedback") or {}).get("status", "") or ""
                    ),
                    "fast_alpha_review_policy_trade_feedback_count": int(
                        (gate.get("review_policy_trade_feedback") or {}).get("closed_trade_count", 0) or 0
                    ),
                    "fast_alpha_review_policy_trade_feedback_expectancy_pct": float(
                        (gate.get("review_policy_trade_feedback") or {}).get("expectancy_pct", 0.0) or 0.0
                    ),
                    "fast_alpha_review_policy_trade_feedback_profit_factor": float(
                        (gate.get("review_policy_trade_feedback") or {}).get("profit_factor", 0.0) or 0.0
                    ),
                    "fast_alpha_review_policy_trade_feedback_max_drawdown_pct": float(
                        (gate.get("review_policy_trade_feedback") or {}).get("max_drawdown_pct", 0.0) or 0.0
                    ),
                },
            ),
        )
        if not result or result.get("dry_run"):
            return False

        fast_decision = SimpleNamespace(
            should_execute=True,
            reason=(
                f"fast_alpha_execute: {getattr(decision, 'reason', '')}; "
                f"xgb_gap={gate['xgb_gap']:.2f}; score_gap={gate['score_gap']:.2f}"
            ),
            portfolio_rating="FAST_ALPHA",
            position_scale=float(self.settings.strategy.fast_alpha_position_scale),
            position_value=adjusted_position_value,
            stop_loss_pct=stop_loss_pct,
            take_profit_levels=list(take_profit_levels),
            final_score=float(final_score),
            horizon_hours=horizon_hours,
        )
        self.persist_analysis(
            symbol,
            insight,
            prediction,
            final_score,
            decision=fast_decision,
            analysis_timestamp=features.timestamp,
            review=review,
            validation=validation,
            pipeline_mode="fast_alpha",
        )
        self.storage.insert_execution_event(
            "fast_alpha_open",
            symbol,
            {
                "price": entry_price,
                "final_score": final_score,
                "up_probability": gate["up_probability"],
                "review_score": gate["review_score"],
                "effective_review_score": float(
                    gate.get("effective_review_score", gate["review_score"]) or 0.0
                ),
                "position_value": adjusted_position_value,
                "xgb_gap": gate["xgb_gap"],
                "score_gap": gate["score_gap"],
                "liquidity_ratio": gate["liquidity_ratio"],
                "horizon_hours": horizon_hours,
                "risk_reason": str(getattr(risk_result, "reason", "") or ""),
                "adaptive_reason": str(
                    (gate.get("adaptive_profile") or {}).get("reason", "") or ""
                ),
                "review_policy_reason": str(gate.get("review_policy_reason", "") or ""),
            },
        )
        self.notifier.notify_trade_open(
            symbol,
            SignalDirection.LONG.value,
            float(result["price"]),
            final_score,
            rationale,
        )
        return True

    def _fast_alpha_risk_override(
        self,
        *,
        account,
        positions: list[dict],
        symbol: str,
        features,
        consecutive_wins: int,
        consecutive_losses: int,
        risk_reason: str,
        liquidity_floor_override: float = 0.0,
    ):
        reason = str(risk_reason or "")
        liquidity_ratio = float(features.values.get("volume_ratio_1h", 0.0) or 0.0)
        entry_price = float(features.values.get("close_4h", 0.0) or 0.0)
        atr = float(features.values.get("atr_4h", 0.0) or 0.0)
        if reason == "insufficient liquidity":
            if not bool(self.settings.strategy.fast_alpha_liquidity_override_enabled):
                return None
            return self._can_open_position(
                account=account,
                positions=positions,
                symbol=symbol,
                atr=atr,
                entry_price=entry_price,
                liquidity_ratio=liquidity_ratio,
                liquidity_floor_override=(
                    float(liquidity_floor_override)
                    if float(liquidity_floor_override or 0.0) > 0.0
                    else float(self.settings.strategy.fast_alpha_liquidity_floor_ratio)
                ),
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                performance_snapshot=None,
            )
        if reason.startswith("portfolio heat") or reason == "max positions reached":
            if not bool(self.settings.strategy.fast_alpha_portfolio_heat_override_enabled):
                return None
            return self._can_open_position(
                account=account,
                positions=positions,
                symbol=symbol,
                atr=atr,
                entry_price=entry_price,
                liquidity_ratio=liquidity_ratio,
                liquidity_floor_override=float(
                    features.values.get(
                        "adaptive_min_liquidity_ratio",
                        self.settings.strategy.min_liquidity_ratio,
                    )
                    or self.settings.strategy.min_liquidity_ratio
                ),
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                performance_snapshot=None,
            )
        return None

    def run_active_symbols(
        self,
        *,
        now,
        active_symbols: list[str],
        positions: list[dict],
        account,
        model_trading_disabled: bool,
        consecutive_wins: int,
        consecutive_losses: int,
    ) -> dict:
        opened_positions = 0
        self._short_horizon_adaptive_profile_cache = None
        runtime_performance = (
            self.performance_getter().build() if self.performance_getter else None
        )
        for symbol in active_symbols:
            snapshot = self.prepare_symbol_snapshot(symbol, now, include_blocked=True)
            if snapshot is None:
                if self.get_circuit_breaker_reason() == "api_failure_circuit_breaker":
                    break
                continue

            features, insight, prediction, validation, review = snapshot
            if self.detect_abnormal_move(symbol, now):
                positions = self.get_positions()
                account = self.account_state(now, positions)
                continue
            atr = float(features.values.get("atr_4h", 0.0))
            entry_price = float(features.values.get("close_4h", 0.0))
            liquidity_ratio = float(features.values.get("volume_ratio_1h", 0.0))
            liquidity_floor_override = float(
                features.values.get(
                    "adaptive_min_liquidity_ratio",
                    self.settings.strategy.min_liquidity_ratio,
                )
                or self.settings.strategy.min_liquidity_ratio
            )
            risk_result = self._can_open_position(
                account=account,
                positions=positions,
                symbol=symbol,
                atr=atr,
                entry_price=entry_price,
                liquidity_ratio=liquidity_ratio,
                liquidity_floor_override=liquidity_floor_override,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                performance_snapshot=runtime_performance,
            )
            context, decision = self.decision_engine.evaluate_entry(
                symbol=symbol,
                prediction=prediction,
                insight=insight,
                features=features,
                risk_result=risk_result,
            )
            ab_result = self.evaluate_ab_test(
                now=now,
                symbol=symbol,
                features=features,
                insight=insight,
                risk_result=risk_result,
                champion_prediction=prediction,
                champion_decision=decision,
                account=account,
                positions=positions,
            )
            self.persist_analysis(
                symbol,
                insight,
                prediction,
                context.final_score,
                decision=decision,
                analysis_timestamp=features.timestamp,
                review=review,
                validation=validation,
                pipeline_mode="execution",
            )
            if ab_result and ab_result.get("challenger_prediction") is not None:
                self.persist_analysis(
                    symbol,
                    insight,
                    ab_result["challenger_prediction"],
                    float(ab_result.get("challenger_final_score", 0.0) or 0.0),
                    decision=ab_result.get("challenger_decision"),
                    analysis_timestamp=features.timestamp,
                    review=review,
                    validation=validation,
                    pipeline_mode=str(
                        ab_result.get("analysis_pipeline_mode") or "challenger_shadow"
                    ),
                )
            self.notifier.notify_analysis_result(
                symbol,
                context.direction.value,
                max(prediction.up_probability, insight.confidence),
                decision.reason,
            )
            self.record_shadow_trade_if_blocked(
                symbol=symbol,
                features=features,
                prediction=prediction,
                decision=decision,
                validation=validation,
                review=review,
                risk_result=risk_result,
                entry_price=entry_price,
            )

            if not decision.should_execute:
                if not (ab_result and ab_result.get("execute_live")):
                    soft_execution_result = self._maybe_open_execution_soft_entry(
                        symbol=symbol,
                        positions=positions,
                        now=now,
                        model_trading_disabled=model_trading_disabled,
                        prediction=prediction,
                        insight=insight,
                        validation=validation,
                        review=review,
                        risk_result=risk_result,
                        entry_price=entry_price,
                        final_score=context.final_score,
                        decision=decision,
                        features=features,
                    )
                    if soft_execution_result:
                        opened_positions += 1
                        positions = self.get_positions()
                        account = self.account_state(now, positions)
                        continue
                    fast_alpha_result = self._maybe_open_fast_alpha(
                        symbol=symbol,
                        positions=positions,
                        now=now,
                        account=account,
                        model_trading_disabled=model_trading_disabled,
                        prediction=prediction,
                        insight=insight,
                        validation=validation,
                        review=review,
                        risk_result=risk_result,
                        entry_price=entry_price,
                        final_score=context.final_score,
                        decision=decision,
                        features=features,
                    )
                    if fast_alpha_result:
                        opened_positions += 1
                        positions = self.get_positions()
                        account = self.account_state(now, positions)
                        continue
                    canary_result = self._maybe_open_paper_canary(
                        symbol=symbol,
                        positions=positions,
                        now=now,
                        model_trading_disabled=model_trading_disabled,
                        prediction=prediction,
                        validation=validation,
                        review=review,
                        risk_result=risk_result,
                        entry_price=entry_price,
                        final_score=context.final_score,
                        decision=decision,
                    )
                    if canary_result:
                        opened_positions += 1
                        positions = self.get_positions()
                        account = self.account_state(now, positions)
                        continue
                    continue
            if model_trading_disabled:
                continue
            if any(position["symbol"] == symbol for position in positions):
                continue

            if decision.should_execute:
                if self._block_entry_from_review(review):
                    continue
                position_adjustment = self._adjust_position_value(
                    symbol=symbol,
                    pipeline_mode="execution",
                    base_position_value=decision.position_value,
                    now=now,
                    prediction=prediction,
                    decision=decision,
                )
                adjusted_position_value = float(
                    position_adjustment.get("position_value", 0.0) or 0.0
                )
                if adjusted_position_value <= 0:
                    continue

                rationale = self.compose_trade_rationale(decision.reason, review)
                if float(position_adjustment.get("scale", 1.0) or 1.0) < 0.999:
                    rationale = (
                        f"{rationale}; evidence_scale="
                        f"{float(position_adjustment.get('scale', 1.0) or 1.0):.2f}"
                    )

                result = self.executor.execute_open(
                    symbol=symbol,
                    direction=context.direction,
                    price=entry_price,
                    confidence=context.final_score,
                    rationale=rationale,
                    stop_loss_pct=decision.stop_loss_pct,
                    take_profit_pct=decision.take_profit_levels[0],
                    position_value=adjusted_position_value,
                    metadata=self._execution_metadata(
                        prediction=prediction,
                        final_score=context.final_score,
                        pipeline_mode="execution",
                        decision_reason=decision.reason,
                        review=review,
                        extra={
                            **self._risk_metadata_extra(risk_result),
                            "model_evidence_scale": float(
                                position_adjustment.get("scale", 1.0) or 1.0
                            ),
                            "model_evidence_source": str(
                                position_adjustment.get("source", "none") or "none"
                            ),
                            "model_evidence_reason": str(
                                position_adjustment.get("reason", "") or ""
                            ),
                        },
                    ),
                )
                if result:
                    if result.get("dry_run"):
                        continue
                    opened_positions += 1
                    self.notifier.notify_trade_open(
                        symbol,
                        context.direction.value,
                        float(result["price"]),
                        context.final_score,
                        self.compose_trade_rationale(decision.reason, review),
                    )
                    positions = self.get_positions()
                    account = self.account_state(now, positions)
                    continue

            if ab_result and ab_result.get("execute_live") and not any(
                position["symbol"] == symbol for position in positions
            ):
                if self._block_entry_from_review(review):
                    continue
                adjusted_ab_position_value = float(
                    ab_result.get("position_value", 0.0) or 0.0
                )
                if adjusted_ab_position_value <= 0:
                    continue

                rationale = self.compose_trade_rationale(
                    f"[challenger_ab] {ab_result['reason']}",
                    review,
                )
                if float(ab_result.get("evidence_scale", 1.0) or 1.0) < 0.999:
                    rationale = (
                        f"{rationale}; evidence_scale="
                        f"{float(ab_result.get('evidence_scale', 1.0) or 1.0):.2f}"
                    )

                ab_live = self.executor.execute_open(
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    price=entry_price,
                    confidence=ab_result["final_score"],
                    rationale=rationale,
                    stop_loss_pct=decision.stop_loss_pct,
                    take_profit_pct=decision.take_profit_levels[0],
                    position_value=adjusted_ab_position_value,
                    metadata=self._execution_metadata(
                        prediction=ab_result["challenger_prediction"],
                        final_score=ab_result["final_score"],
                        pipeline_mode=str(
                            ab_result.get("analysis_pipeline_mode") or "challenger_live"
                        ),
                        decision_reason=str(ab_result["reason"]),
                        review=review,
                        extra={
                            **self._risk_metadata_extra(risk_result),
                            "ab_selected_variant": str(
                                ab_result.get("selected_variant") or ""
                            ),
                            "model_evidence_scale": float(
                                ab_result.get("evidence_scale", 1.0) or 1.0
                            ),
                            "model_evidence_source": str(
                                ab_result.get("evidence_source", "none") or "none"
                            ),
                            "model_evidence_reason": str(
                                ab_result.get("evidence_reason", "") or ""
                            ),
                        },
                    ),
                )
                if ab_live:
                    if ab_live.get("dry_run"):
                        continue
                    opened_positions += 1
                    positions = self.get_positions()
                    account = self.account_state(now, positions)

        return {
            "opened_positions": opened_positions,
            "positions": positions,
            "account": account,
        }

    def _maybe_open_execution_soft_entry(
        self,
        *,
        symbol: str,
        positions: list[dict],
        now,
        model_trading_disabled: bool,
        prediction,
        insight,
        validation,
        review,
        risk_result,
        entry_price: float,
        final_score: float,
        decision,
        features,
    ) -> bool:
        if not bool(self.settings.strategy.execution_soft_entry_enabled):
            return False
        if model_trading_disabled:
            return False
        if any(position["symbol"] == symbol for position in positions):
            return False
        if not getattr(validation, "ok", True):
            return False
        if not bool(getattr(risk_result, "allowed", False)):
            return False
        if self._block_entry_from_review(review):
            return False

        review_reasons = {
            str(reason).strip()
            for reason in (getattr(review, "reasons", []) or [])
            if str(reason).strip()
        }
        reviewed_action = str(getattr(review, "reviewed_action", "") or "").upper()
        raw_action = str(getattr(review, "raw_action", "") or "").upper()
        review_score = float(getattr(review, "review_score", 0.0) or 0.0)
        if reviewed_action != "HOLD" or raw_action == "CLOSE":
            return False
        if not self._soft_review_quality_ok(
            prediction=prediction,
            final_score=final_score,
            review_score=review_score,
            review_reasons=review_reasons,
            review_min_score=float(
                self.settings.strategy.execution_soft_entry_review_min_score
            ),
            final_score_gap_cap=float(
                self.settings.strategy.execution_soft_entry_max_final_score_gap_pct
            ),
        ):
            return False

        adaptive_profile = self._short_horizon_adaptive_profile()
        if bool(adaptive_profile.get("pause_entries", False)):
            return False

        base_position_value = (
            float(getattr(risk_result, "allowed_position_value", 0.0) or 0.0)
            * float(self.settings.strategy.execution_soft_entry_position_scale)
        )
        if base_position_value <= 0:
            return False

        position_adjustment = self._adjust_position_value(
            symbol=symbol,
            pipeline_mode="execution",
            base_position_value=base_position_value,
            now=now,
            prediction=prediction,
            decision=decision,
        )
        adjusted_position_value = float(
            position_adjustment.get("position_value", 0.0) or 0.0
        )
        if adjusted_position_value <= 0:
            return False

        rationale = self.compose_trade_rationale(
            f"[execution_soft_entry] {getattr(decision, 'reason', '')}",
            review,
        )
        if float(position_adjustment.get("scale", 1.0) or 1.0) < 0.999:
            rationale = (
                f"{rationale}; evidence_scale="
                f"{float(position_adjustment.get('scale', 1.0) or 1.0):.2f}"
            )

        result = self.executor.execute_open(
            symbol=symbol,
            direction=SignalDirection.LONG,
            price=entry_price,
            confidence=float(final_score or 0.0),
            rationale=rationale,
            stop_loss_pct=getattr(
                decision,
                "stop_loss_pct",
                self.settings.strategy.fixed_stop_loss_pct,
            ),
            take_profit_pct=getattr(
                decision,
                "take_profit_levels",
                [self.settings.strategy.take_profit_levels[0]],
            )[0],
            position_value=adjusted_position_value,
            metadata=self._execution_metadata(
                prediction=prediction,
                final_score=final_score,
                pipeline_mode="execution_soft_entry",
                decision_reason=str(getattr(decision, "reason", "")),
                review=review,
                extra={
                    **self._risk_metadata_extra(risk_result),
                    "execution_soft_entry_enabled": True,
                    "execution_soft_entry_position_scale": float(
                        self.settings.strategy.execution_soft_entry_position_scale
                    ),
                    "execution_soft_entry_review_min_score": float(
                        self.settings.strategy.execution_soft_entry_review_min_score
                    ),
                    "execution_soft_entry_max_final_score_gap_pct": float(
                        self.settings.strategy.execution_soft_entry_max_final_score_gap_pct
                    ),
                    "execution_soft_entry_raw_action": raw_action,
                    "execution_soft_entry_reviewed_action": reviewed_action,
                    "execution_soft_entry_review_score": review_score,
                    "model_evidence_scale": float(
                        position_adjustment.get("scale", 1.0) or 1.0
                    ),
                    "model_evidence_source": str(
                        position_adjustment.get("source", "none") or "none"
                    ),
                    "model_evidence_reason": str(
                        position_adjustment.get("reason", "") or ""
                    ),
                    "short_horizon_adaptive_reason": str(
                        adaptive_profile.get("reason", "") or ""
                    ),
                    "short_horizon_adaptive_positive_edge": bool(
                        adaptive_profile.get("positive_edge", False)
                    ),
                },
            ),
        )
        if not result or result.get("dry_run"):
            return False

        soft_decision = SimpleNamespace(
            should_execute=True,
            reason=f"execution_soft_entry: {getattr(decision, 'reason', '')}",
            portfolio_rating="SOFT_ENTRY",
            position_scale=float(
                self.settings.strategy.execution_soft_entry_position_scale
            ),
            position_value=adjusted_position_value,
            stop_loss_pct=getattr(
                decision,
                "stop_loss_pct",
                self.settings.strategy.fixed_stop_loss_pct,
            ),
            take_profit_levels=list(
                getattr(
                    decision,
                    "take_profit_levels",
                    self.settings.strategy.take_profit_levels,
                )
            ),
            final_score=float(final_score or 0.0),
            horizon_hours=int(self.settings.strategy.max_hold_hours),
        )
        self.persist_analysis(
            symbol,
            insight,
            prediction,
            final_score,
            decision=soft_decision,
            analysis_timestamp=features.timestamp,
            review=review,
            validation=validation,
            pipeline_mode="execution_soft_entry",
        )
        self.storage.insert_execution_event(
            "execution_soft_entry_open",
            symbol,
            {
                "price": entry_price,
                "final_score": float(final_score or 0.0),
                "up_probability": float(getattr(prediction, "up_probability", 0.0) or 0.0),
                "review_score": review_score,
                "position_value": adjusted_position_value,
                "opened_at": now.isoformat(),
                "adaptive_reason": str(adaptive_profile.get("reason", "") or ""),
            },
        )
        return True

    def _maybe_open_paper_canary(
        self,
        *,
        symbol: str,
        positions: list[dict],
        now,
        model_trading_disabled: bool,
        prediction,
        validation,
        review,
        risk_result,
        entry_price: float,
        final_score: float,
        decision,
    ) -> bool:
        if self.settings.app.runtime_mode != "paper":
            return False
        if not self.settings.strategy.paper_canary_enabled:
            return False
        if model_trading_disabled:
            return False
        if any(position["symbol"] == symbol for position in positions):
            return False
        if not getattr(validation, "ok", True):
            return False
        if not bool(getattr(risk_result, "allowed", False)):
            return False
        review_reasons = {
            str(reason).strip()
            for reason in (getattr(review, "reasons", []) or [])
            if str(reason).strip()
        }
        if review_reasons & {
            "setup_auto_pause",
            "setup_negative_expectancy",
            "experience_negative_setup",
        }:
            return False
        reviewed_action = str(getattr(review, "reviewed_action", "") or "")
        raw_action = str(getattr(review, "raw_action", "") or "")
        review_score = float(getattr(review, "review_score", 0.0) or 0.0)
        if review_score < 0.0:
            return False
        primary_review_min_score = float(
            self.settings.strategy.paper_canary_min_review_score
        )
        primary_review_ok = (
            reviewed_action == "OPEN_LONG"
            and review_score >= primary_review_min_score
        )
        offensive_review_ok = (
            not primary_review_ok
            and bool(self.settings.strategy.paper_canary_offensive_enabled)
            and review_score
            >= float(self.settings.strategy.paper_canary_offensive_review_min_score)
            and self._offensive_review_quality_ok(
                prediction=prediction,
                final_score=final_score,
                raw_action=raw_action,
                reviewed_action=reviewed_action,
                review_score=review_score,
                review_reasons=review_reasons,
                primary_review_min_score=primary_review_min_score,
            )
        )
        soft_review_ok = (
            not offensive_review_ok
            and bool(self.settings.strategy.paper_canary_soft_enabled)
            and reviewed_action == "HOLD"
            and raw_action != "CLOSE"
            and review_score
            >= float(self.settings.strategy.paper_canary_soft_review_min_score)
            and self._soft_review_quality_ok(
                prediction=prediction,
                final_score=final_score,
                review_score=review_score,
                review_reasons=review_reasons,
            )
        )
        if not (primary_review_ok or offensive_review_ok or soft_review_ok):
            return False
        adaptive_profile = self._short_horizon_adaptive_profile()
        if bool(adaptive_profile.get("pause_entries", False)) and not primary_review_ok:
            return False

        xgb_gap = max(
            0.0,
            float(self.decision_engine.xgboost_threshold) - float(prediction.up_probability),
        )
        score_gap = max(
            0.0,
            float(self.decision_engine.final_score_threshold) - float(final_score),
        )
        if (
            xgb_gap > float(self.settings.strategy.paper_canary_xgboost_gap_pct)
            and score_gap > float(self.settings.strategy.paper_canary_final_score_gap_pct)
        ):
            return False

        base_position_value = float(getattr(risk_result, "allowed_position_value", 0.0) or 0.0)
        soft_scale = (
            1.0
            if primary_review_ok
            else float(self.settings.strategy.paper_canary_offensive_position_scale)
            if offensive_review_ok
            else float(self.settings.strategy.paper_canary_soft_position_scale)
        )
        canary_position_value = (
            base_position_value * float(self.settings.strategy.paper_canary_position_scale)
            * soft_scale
        )
        if canary_position_value <= 0:
            return False

        canary_mode = (
            "primary_review"
            if primary_review_ok
            else "offensive_review"
            if offensive_review_ok
            else "soft_review"
        )
        rationale = self.compose_trade_rationale(
            (
                f"[paper_canary] {getattr(decision, 'reason', '')}"
                if primary_review_ok
                else f"[paper_canary_offensive] {getattr(decision, 'reason', '')}"
                if offensive_review_ok
                else f"[paper_canary_soft] {getattr(decision, 'reason', '')}"
            ),
            review,
        )
        result = self.executor.execute_open(
            symbol=symbol,
            direction=SignalDirection.LONG,
            price=entry_price,
            confidence=final_score,
            rationale=rationale,
            stop_loss_pct=getattr(decision, "stop_loss_pct", self.settings.strategy.fixed_stop_loss_pct),
            take_profit_pct=getattr(decision, "take_profit_levels", [self.settings.strategy.take_profit_levels[0]])[0],
            position_value=canary_position_value,
            metadata=self._execution_metadata(
                prediction=prediction,
                final_score=final_score,
                pipeline_mode="paper_canary",
                decision_reason=str(getattr(decision, "reason", "")),
                review=review,
                extra={
                    **self._risk_metadata_extra(risk_result),
                    "paper_canary_mode": canary_mode,
                    "paper_canary_soft_scale": soft_scale,
                    "short_horizon_adaptive_reason": str(
                        adaptive_profile.get("reason", "") or ""
                    ),
                    "short_horizon_adaptive_positive_edge": bool(
                        adaptive_profile.get("positive_edge", False)
                    ),
                },
            ),
        )
        if not result or result.get("dry_run"):
            return False
        self.storage.insert_execution_event(
            "paper_canary_open",
            symbol,
            {
                "price": entry_price,
                "final_score": final_score,
                "up_probability": float(prediction.up_probability),
                "review_score": float(getattr(review, "review_score", 0.0) or 0.0),
                "position_value": canary_position_value,
                "canary_mode": canary_mode,
                "portfolio_heat_factor": float(
                    getattr(risk_result, "portfolio_heat_factor", 1.0) or 1.0
                ),
                "effective_max_total_exposure_pct": float(
                    getattr(risk_result, "effective_max_total_exposure_pct", 0.0) or 0.0
                ),
                "effective_max_positions": int(
                    getattr(risk_result, "effective_max_positions", 0) or 0
                ),
                "opened_at": now.isoformat(),
                "adaptive_reason": str(adaptive_profile.get("reason", "") or ""),
            },
        )
        return True

    @staticmethod
    def _setup_performance_blocks_entry(setup_performance: dict | None) -> bool:
        payload = setup_performance if isinstance(setup_performance, dict) else {}
        weighted_count = float(payload.get("weighted_count", 0.0) or 0.0)
        avg_outcome = float(payload.get("avg_outcome_24h", 0.0) or 0.0)
        negative_ratio = float(payload.get("negative_ratio", 0.0) or 0.0)
        return weighted_count >= 2.0 and avg_outcome < 0.0 and negative_ratio >= 0.6

    def _block_entry_from_review(self, review) -> bool:
        reasons = {
            str(reason).strip()
            for reason in (getattr(review, "reasons", []) or [])
            if str(reason).strip()
        }
        if reasons & {
            "setup_auto_pause",
            "setup_negative_expectancy",
            "experience_negative_setup",
        }:
            return True
        return self._setup_performance_blocks_entry(
            getattr(review, "setup_performance", {}) or {}
        )

    def _offensive_review_quality_ok(
        self,
        *,
        prediction,
        final_score: float,
        raw_action: str,
        reviewed_action: str,
        review_score: float,
        review_reasons: set[str],
        primary_review_min_score: float,
    ) -> bool:
        if not (review_reasons & self.OFFENSIVE_REVIEW_REASONS):
            return False
        if raw_action != "OPEN_LONG" and reviewed_action != "OPEN_LONG":
            return False

        up_probability = float(getattr(prediction, "up_probability", 0.0) or 0.0)
        xgb_threshold = float(self.decision_engine.xgboost_threshold)
        final_threshold = float(self.decision_engine.final_score_threshold)
        if up_probability < max(0.55, xgb_threshold - 0.04):
            return False
        if float(final_score or 0.0) < max(0.0, final_threshold - 0.10):
            return False
        if reviewed_action == "OPEN_LONG" and review_score >= primary_review_min_score:
            return False
        return True

    def _soft_review_quality_ok(
        self,
        *,
        prediction,
        final_score: float,
        review_score: float,
        review_reasons: set[str],
        review_min_score: float | None = None,
        final_score_gap_cap: float | None = None,
    ) -> bool:
        up_probability = float(getattr(prediction, "up_probability", 0.0) or 0.0)
        xgb_threshold = float(self.decision_engine.xgboost_threshold)
        final_threshold = float(self.decision_engine.final_score_threshold)
        if up_probability < xgb_threshold:
            return False

        score_gap = max(0.0, final_threshold - float(final_score or 0.0))
        if final_score_gap_cap is None:
            soft_score_gap_cap = min(
                0.06,
                float(self.settings.strategy.paper_canary_final_score_gap_pct) * 0.5,
            )
        else:
            soft_score_gap_cap = max(0.0, float(final_score_gap_cap))
        if score_gap > soft_score_gap_cap:
            return False

        positive_support = review_reasons & self.SOFT_REVIEW_SUPPORT_REASONS
        if not positive_support:
            return False

        if "news_event_risk" in review_reasons:
            return False
        if "liquidity_weak" in review_reasons and (
            "trend_against" in review_reasons
            or "trend_against_discounted" in review_reasons
        ):
            return False

        extreme_fear_context = bool(
            review_reasons & self.SOFT_REVIEW_EXTREME_FEAR_CONTEXT_REASONS
        )
        if extreme_fear_context and not (
            review_reasons & self.SOFT_REVIEW_EXTREME_FEAR_ALLOWED_REASONS
        ):
            return False

        effective_review_min_score = (
            float(self.settings.strategy.paper_canary_soft_review_min_score)
            if review_min_score is None
            else float(review_min_score)
        )
        return review_score >= effective_review_min_score

    def run_shadow_symbols(
        self,
        *,
        now,
        shadow_symbols: list[str],
    ) -> None:
        for symbol in shadow_symbols:
            snapshot = self.prepare_symbol_snapshot(
                symbol,
                now,
                include_blocked=True,
            )
            if snapshot is None:
                continue
            features, insight, prediction, validation, review = snapshot
            shadow_decision = self.decision_engine.evaluate_entry(
                symbol=symbol,
                prediction=prediction,
                insight=insight,
                features=features,
                risk_result=RiskCheckResult(
                    allowed=True,
                    allowed_position_value=0.0,
                    stop_loss_pct=self.settings.strategy.fixed_stop_loss_pct,
                    take_profit_levels=list(self.settings.strategy.take_profit_levels),
                    trailing_stop_drawdown_pct=self.settings.strategy.trailing_stop_drawdown_pct,
                ),
            )[1]
            self.persist_analysis(
                symbol,
                insight,
                prediction,
                shadow_decision.final_score,
                decision=shadow_decision,
                analysis_timestamp=features.timestamp,
                review=review,
                validation=validation,
                pipeline_mode="shadow_observation",
            )
            self.storage.insert_execution_event(
                "shadow_observation",
                symbol,
                {
                    "validation_ok": validation.ok,
                    "reviewed_action": review.reviewed_action,
                    "review_score": review.review_score,
                },
            )
