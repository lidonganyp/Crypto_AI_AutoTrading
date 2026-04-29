"""Account, guard, and live-readiness helpers."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from config import Settings, get_settings
from core.storage import Storage
from loguru import logger


class GuardRuntimeService:
    """Manage runtime account state, guards, and live-readiness checks."""

    def __init__(
        self,
        storage: Storage,
        settings: Settings | None = None,
        *,
        market_getter,
        risk,
        performance_getter,
        health_getter,
        notifier,
        executor_getter,
        train_models_if_due,
        get_active_symbols,
        storage_symbol_variants,
        get_peak_equity,
        set_peak_equity,
        get_cooldown_until,
        set_cooldown_until,
        get_circuit_breaker_reason,
        set_circuit_breaker_reason,
        set_circuit_breaker_active,
        get_model_degradation_status,
        set_model_degradation_status,
        get_model_degradation_reason,
        set_model_degradation_reason,
        set_model_trading_disabled,
        nextgen_live_guard_callback=None,
    ):
        self.storage = storage
        self.settings = settings or get_settings()
        self.market_getter = market_getter
        self.risk = risk
        self.performance_getter = performance_getter
        self.health_getter = health_getter
        self.notifier = notifier
        self.executor_getter = executor_getter
        self.train_models_if_due = train_models_if_due
        self.get_active_symbols = get_active_symbols
        self.storage_symbol_variants = storage_symbol_variants
        self.get_peak_equity = get_peak_equity
        self.set_peak_equity = set_peak_equity
        self.get_cooldown_until = get_cooldown_until
        self.set_cooldown_until = set_cooldown_until
        self.get_circuit_breaker_reason = get_circuit_breaker_reason
        self.set_circuit_breaker_reason = set_circuit_breaker_reason
        self.set_circuit_breaker_active = set_circuit_breaker_active
        self.get_model_degradation_status = get_model_degradation_status
        self.set_model_degradation_status = set_model_degradation_status
        self.get_model_degradation_reason = get_model_degradation_reason
        self.set_model_degradation_reason = set_model_degradation_reason
        self.set_model_trading_disabled = set_model_trading_disabled
        self.nextgen_live_guard_callback = nextgen_live_guard_callback

    def account_state(self, now: datetime, positions: list[dict]):
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        realized_total = self._realized_total_pnl()
        realized_today = self._realized_pnl_since(today_start)
        realized_week = self._realized_pnl_since(week_start)
        priced_positions = self.priced_positions(positions)
        unrealized_total = 0.0
        unrealized_today = 0.0
        unrealized_week = 0.0
        for position in priced_positions:
            current_price = float(position["current_price"])
            quantity = float(position["quantity"])
            entry_price = float(position["entry_price"])
            unrealized_total += (current_price - entry_price) * quantity
            day_baseline = self.period_baseline_price(position, today_start, current_price)
            week_baseline = self.period_baseline_price(position, week_start, current_price)
            unrealized_today += (current_price - day_baseline) * quantity
            unrealized_week += (current_price - week_baseline) * quantity
        equity = self._account_equity(priced_positions, realized_total, unrealized_total)
        peak_equity = max(self.get_peak_equity(), equity)
        self.set_peak_equity(peak_equity)
        account_state = self.risk.build_account_state(
            equity=equity,
            positions=priced_positions,
            realized_pnl_today=realized_today,
            realized_pnl_week=realized_week,
            unrealized_pnl=unrealized_total,
            unrealized_pnl_today=unrealized_today,
            unrealized_pnl_week=unrealized_week,
            peak_equity=peak_equity,
            cooldown_until=self.get_cooldown_until(),
            circuit_breaker_active=bool(self.get_circuit_breaker_reason()),
        )
        cooldown_until = self.risk.apply_account_cooldown(
            account_state,
            current_cooldown_until=self.get_cooldown_until(),
            now=now,
        )
        self.set_cooldown_until(cooldown_until)
        account_state.cooldown_until = cooldown_until
        breaker_reason = self.risk.check_circuit_breaker(account_state)
        self.set_circuit_breaker_reason(breaker_reason)
        self.set_circuit_breaker_active(bool(breaker_reason))
        account_state.circuit_breaker_active = bool(breaker_reason)
        self.storage.insert_account_snapshot(
            {
                "timestamp": now.isoformat(),
                "equity": account_state.equity,
                "realized_pnl": account_state.realized_pnl,
                "unrealized_pnl": account_state.unrealized_pnl,
                "daily_loss_pct": account_state.daily_loss_pct,
                "weekly_loss_pct": account_state.weekly_loss_pct,
                "drawdown_pct": account_state.drawdown_pct,
                "total_exposure_pct": account_state.total_exposure_pct,
                "open_positions": account_state.open_positions,
                "cooldown_until": (
                    account_state.cooldown_until.isoformat()
                    if account_state.cooldown_until
                    else None
                ),
                "circuit_breaker_active": account_state.circuit_breaker_active,
            }
        )
        return account_state

    def priced_positions(self, positions: list[dict]) -> list[dict]:
        priced_positions: list[dict] = []
        fetch_latest_price = getattr(self.market_getter(), "fetch_latest_price", None)
        for position in positions:
            current_price = float(position["entry_price"])
            market_symbol = (
                position["symbol"]
                if ":USDT" in position["symbol"]
                else f"{position['symbol']}:USDT"
            )
            if callable(fetch_latest_price):
                try:
                    latest_price = fetch_latest_price(market_symbol)
                except Exception:
                    latest_price = None
                if latest_price is not None:
                    current_price = float(latest_price)
            priced_positions.append({**position, "current_price": current_price})
        return priced_positions

    def portfolio_equity(self, positions: list[dict]) -> float:
        priced_positions = self.priced_positions(positions)
        realized_total = self._realized_total_pnl()
        unrealized_total = sum(
            (float(position["current_price"]) - float(position["entry_price"]))
            * float(position["quantity"])
            for position in priced_positions
        )
        return self._account_equity(priced_positions, realized_total, unrealized_total)

    def _account_equity(
        self,
        priced_positions: list[dict],
        realized_total: float,
        unrealized_total: float,
    ) -> float:
        live_equity = self._live_account_equity(priced_positions)
        if live_equity is not None:
            return live_equity
        return self.executor_getter().initial_balance + realized_total + unrealized_total

    def _live_account_equity(self, priced_positions: list[dict]) -> float | None:
        if self.settings.app.runtime_mode != "live":
            return None
        exchange = getattr(self.executor_getter(), "exchange", None)
        if exchange is None or not hasattr(exchange, "fetch_total_balance"):
            return None
        quote_asset = self._quote_asset(priced_positions)
        try:
            quote_balance = exchange.fetch_total_balance(quote_asset)
        except Exception as exc:
            logger.warning(f"Unable to fetch live total balance for {quote_asset}: {exc}")
            return None
        if quote_balance is None:
            return None
        managed_position_value = sum(
            float(position["current_price"]) * float(position["quantity"])
            for position in priced_positions
        )
        return float(quote_balance) + managed_position_value

    def _quote_asset(self, priced_positions: list[dict]) -> str:
        symbol = ""
        if priced_positions:
            symbol = str(priced_positions[0].get("symbol") or "")
        elif self.settings.exchange.symbols:
            symbol = str(self.settings.exchange.symbols[0])
        if "/" not in symbol:
            return "USDT"
        quote = symbol.split("/", 1)[1]
        return quote.split(":", 1)[0] or "USDT"

    def _realized_total_pnl(self) -> float:
        ledger_rows = self.storage.get_pnl_ledger(limit=5000)
        if ledger_rows:
            return sum(float(row.get("net_pnl") or 0.0) for row in ledger_rows)
        return sum(
            float(trade.get("pnl") or 0.0)
            for trade in (
                self.storage.get_closed_trades() + self.storage.get_open_trades()
            )
        )

    def _realized_pnl_since(self, since: datetime) -> float:
        ledger_rows = self.storage.get_pnl_ledger(
            limit=5000,
            since=since.isoformat(),
        )
        if ledger_rows:
            return sum(float(row.get("net_pnl") or 0.0) for row in ledger_rows)

        with self.storage._conn() as conn:
            rows = conn.execute(
                """
                SELECT payload_json FROM execution_events
                WHERE event_type IN ('close', 'live_close') AND created_at >= ?
                ORDER BY created_at DESC
                """,
                (since.isoformat(),),
            ).fetchall()
        realized = 0.0
        for row in rows:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except Exception:
                continue
            incremental = payload.get("incremental_pnl")
            if incremental is None:
                incremental = payload.get("pnl", 0.0)
            try:
                realized += float(incremental or 0.0)
            except (TypeError, ValueError):
                continue
        return realized

    def stored_close_at_or_before(self, symbol: str, moment: datetime) -> float | None:
        target_ms = int(moment.timestamp() * 1000)
        variants = self.storage_symbol_variants(symbol)
        with self.storage._conn() as conn:
            for timeframe in ("1h", "4h", "1d"):
                placeholders = ",".join(["?"] * len(variants))
                row = conn.execute(
                    (
                        f"SELECT close FROM ohlcv WHERE symbol IN ({placeholders}) "
                        "AND timeframe = ? AND timestamp <= ? "
                        "ORDER BY timestamp DESC LIMIT 1"
                    ),
                    [*variants, timeframe, target_ms],
                ).fetchone()
                if row:
                    return float(row["close"])
        return None

    def period_baseline_price(
        self,
        position: dict,
        period_start: datetime,
        current_price: float,
    ) -> float:
        entry_time = datetime.fromisoformat(position["entry_time"])
        entry_price = float(position["entry_price"])
        if entry_time >= period_start:
            return entry_price
        stored_price = self.stored_close_at_or_before(position["symbol"], period_start)
        if stored_price is not None:
            return stored_price
        return current_price

    def enforce_accuracy_guard(self, now: datetime):
        cooldown_until_current = self.get_cooldown_until()
        if cooldown_until_current and now < cooldown_until_current:
            last_reason = self.storage.get_state("last_accuracy_guard_reason", "")
            if last_reason and "xgboost_accuracy=" in last_reason:
                return
        snapshot = self.performance_getter().build()
        min_samples = self.settings.risk.model_accuracy_min_samples
        floor_pct = self.settings.risk.model_accuracy_floor_pct
        prediction_eval_count = int(
            getattr(snapshot, "current_prediction_eval_count", 0)
            or snapshot.prediction_eval_count
        )
        xgboost_accuracy_pct = float(
            getattr(snapshot, "current_xgboost_accuracy_pct", 0.0)
            if int(getattr(snapshot, "current_prediction_eval_count", 0) or 0) > 0
            else snapshot.xgboost_accuracy_pct
        )
        if prediction_eval_count < min_samples:
            return
        if xgboost_accuracy_pct >= floor_pct:
            return

        cooldown_until = None
        if self.settings.app.runtime_mode != "paper":
            cooldown_until = now + timedelta(
                hours=self.settings.risk.model_accuracy_cooldown_hours
            )
            current = self.get_cooldown_until()
            self.set_cooldown_until(max(current or cooldown_until, cooldown_until))
        self.storage.set_state("last_accuracy_guard_triggered", now.isoformat())
        self.storage.set_state(
            "last_accuracy_guard_reason",
            f"xgboost_accuracy={xgboost_accuracy_pct:.2f}",
        )
        self.notifier.notify(
            "model_accuracy_guard",
            "模型准确率低于阈值",
            (
                f"xgboost_accuracy={xgboost_accuracy_pct:.2f}% | "
                f"samples={prediction_eval_count} | "
                f"cooldown_until={cooldown_until.isoformat() if cooldown_until else 'paper_skip'}"
            ),
            level="warning",
        )
        try:
            self.train_models_if_due(now, force=False, reason="accuracy_guard")
        except Exception:
            pass

    def apply_model_degradation(self, now: datetime):
        snapshot = self.performance_getter().build()
        new_status = snapshot.degradation_status
        new_reason = snapshot.degradation_reason

        self.storage.set_state("model_degradation_status", new_status)
        self.storage.set_state("model_degradation_reason", new_reason or "")
        self.storage.set_state(
            "model_thresholds_runtime",
            (
                f"xgb={snapshot.recommended_xgboost_threshold:.2f},"
                f"final={snapshot.recommended_final_score_threshold:.2f}"
            ),
        )

        self.set_model_trading_disabled(new_status == "disabled")

        if (
            new_status == self.get_model_degradation_status()
            and new_reason == self.get_model_degradation_reason()
        ):
            return {
                "recommended_xgboost_threshold": snapshot.recommended_xgboost_threshold,
                "recommended_final_score_threshold": snapshot.recommended_final_score_threshold,
                "model_trading_disabled": new_status == "disabled",
            }

        self.set_model_degradation_status(new_status)
        self.set_model_degradation_reason(new_reason)
        if new_status in {"healthy", "warming_up"}:
            return {
                "recommended_xgboost_threshold": snapshot.recommended_xgboost_threshold,
                "recommended_final_score_threshold": snapshot.recommended_final_score_threshold,
                "model_trading_disabled": new_status == "disabled",
            }

        if new_status == "disabled":
            cooldown_until = now + timedelta(
                hours=self.settings.risk.model_accuracy_cooldown_hours
            )
            current = self.get_cooldown_until()
            self.set_cooldown_until(max(current or cooldown_until, cooldown_until))

        self.storage.insert_execution_event(
            "model_degradation",
            "SYSTEM",
            {
                "status": new_status,
                "reason": new_reason,
                "xgboost_accuracy_pct": snapshot.xgboost_accuracy_pct,
                "fusion_accuracy_pct": snapshot.fusion_accuracy_pct,
                "recommended_xgboost_threshold": snapshot.recommended_xgboost_threshold,
                "recommended_final_score_threshold": snapshot.recommended_final_score_threshold,
                "cooldown_until": (
                    self.get_cooldown_until().isoformat()
                    if self.get_cooldown_until()
                    else ""
                ),
            },
        )
        self.notifier.notify(
            "model_degradation",
            "模型衰减状态变化",
            (
                f"status={new_status} | reason={new_reason or 'none'} | "
                f"xgb_acc={snapshot.xgboost_accuracy_pct:.2f}% | "
                f"fusion_acc={snapshot.fusion_accuracy_pct:.2f}%"
            ),
            level="warning" if new_status == "degraded" else "error",
        )
        if new_status == "disabled":
            self._run_nextgen_live_guard_callback(
                trigger="model_degradation_disabled",
                reason=new_reason or new_status,
                details=(
                    f"xgboost_accuracy_pct={snapshot.xgboost_accuracy_pct:.2f};"
                    f"fusion_accuracy_pct={snapshot.fusion_accuracy_pct:.2f}"
                ),
            )
        return {
            "recommended_xgboost_threshold": snapshot.recommended_xgboost_threshold,
            "recommended_final_score_threshold": snapshot.recommended_final_score_threshold,
            "model_trading_disabled": new_status == "disabled",
        }

    def manual_recovery_blocked(self) -> bool:
        required = self.storage.get_state("manual_recovery_required", "false") == "true"
        approved = self.storage.get_state("manual_recovery_approved", "false") == "true"
        if not required or approved:
            return False
        reason = self.storage.get_state("manual_recovery_reason", "unknown")
        self.notifier.notify(
            "manual_recovery_required",
            "等待人工确认恢复",
            f"reason={reason}",
            level="critical",
        )
        return True

    @staticmethod
    def requires_manual_recovery(reason: str) -> bool:
        return reason in {
            "daily_loss_limit",
            "reconciliation_mismatch",
            "api_failure_circuit_breaker",
        }

    def trigger_manual_recovery(self, reason: str, details: str):
        self.storage.set_state("manual_recovery_required", "true")
        self.storage.set_state("manual_recovery_approved", "false")
        self.storage.set_state("manual_recovery_reason", reason)
        self.storage.set_state("manual_recovery_details", details)
        self.storage.insert_execution_event(
            "manual_recovery_required",
            "SYSTEM",
            {
                "reason": reason,
                "details": details,
            },
        )
        self._run_nextgen_live_guard_callback(
            trigger="manual_recovery_required",
            reason=reason,
            details=details,
        )

    def approve_manual_recovery(self) -> dict:
        self.storage.set_state("manual_recovery_required", "false")
        self.storage.set_state("manual_recovery_approved", "true")
        self.storage.set_state("manual_recovery_reason", "")
        self.storage.set_state("manual_recovery_details", "")
        self.set_circuit_breaker_active(False)
        self.set_circuit_breaker_reason("")
        return {"status": "approved"}

    def check_market_latency(self, now: datetime, symbols: list[str] | None = None) -> bool:
        market = self.market_getter()
        if not hasattr(market, "measure_latency"):
            return False
        symbols = symbols or self.get_active_symbols(force_refresh=False, now=now)
        if not symbols:
            return False
        market_symbol = symbols[0]
        market_symbol = f"{market_symbol}:USDT" if ":USDT" not in market_symbol else market_symbol
        result = market.measure_latency(market_symbol)
        latency_seconds = result.get("latency_seconds")
        if latency_seconds is None:
            return False
        self.storage.set_state("latest_market_latency_seconds", f"{latency_seconds:.3f}")
        if latency_seconds >= self.settings.exchange.data_latency_circuit_breaker_seconds:
            self.set_circuit_breaker_active(True)
            self.set_circuit_breaker_reason("market_data_latency")
            self.trigger_manual_recovery(
                "market_data_latency",
                f"latency_seconds={latency_seconds:.3f}",
            )
            self.storage.insert_execution_event(
                "market_latency_circuit_breaker",
                market_symbol,
                {"latency_seconds": latency_seconds},
            )
            self.notifier.notify(
                "market_latency",
                "市场数据延迟熔断",
                f"latency_seconds={latency_seconds:.3f}",
                level="critical",
            )
            return True
        if latency_seconds >= self.settings.exchange.data_latency_warning_seconds:
            self.storage.insert_execution_event(
                "market_latency_warning",
                market_symbol,
                {"latency_seconds": latency_seconds},
            )
            self.notifier.notify(
                "market_latency",
                "市场数据延迟告警",
                f"latency_seconds={latency_seconds:.3f}",
                level="warning",
            )
        return False

    def build_live_readiness(self) -> dict:
        performance = self.performance_getter().build()
        health = self.health_getter().run()
        prediction_eval_count = int(
            getattr(performance, "current_prediction_eval_count", 0)
            or performance.prediction_eval_count
        )
        xgboost_accuracy_pct = float(
            getattr(performance, "current_xgboost_accuracy_pct", 0.0)
            if int(getattr(performance, "current_prediction_eval_count", 0) or 0) > 0
            else performance.xgboost_accuracy_pct
        )
        fusion_accuracy_pct = float(
            getattr(performance, "current_fusion_accuracy_pct", 0.0)
            if int(getattr(performance, "current_prediction_eval_count", 0) or 0) > 0
            else performance.fusion_accuracy_pct
        )
        checks = {
            "llm_runtime_configured": health.llm_runtime_configured,
            "latest_research_mode_llm": health.latest_research_mode != "fallback",
            "research_fallback_ratio_ok": (
                health.recent_research_fallback_ratio_pct
                <= self.settings.risk.live_max_research_fallback_ratio_pct
            ),
            "market_streams_fresh": health.stale_market_streams == 0,
            "market_latency_ok": health.market_latency_status == "ok",
            "closed_trades_ready": (
                performance.total_closed_trades
                >= self.settings.risk.live_min_closed_trades
            ),
            "prediction_eval_count_ready": (
                prediction_eval_count
                >= self.settings.risk.live_min_prediction_eval_count
            ),
            "holdout_accuracy_ready": (
                performance.latest_holdout_accuracy
                >= self.settings.risk.live_min_holdout_accuracy_pct
            ),
            "xgboost_accuracy_ready": (
                xgboost_accuracy_pct
                >= self.settings.risk.live_min_xgboost_accuracy_pct
            ),
            "fusion_accuracy_ready": (
                fusion_accuracy_pct
                >= self.settings.risk.live_min_fusion_accuracy_pct
            ),
            "realized_pnl_ready": (
                performance.total_realized_pnl
                > self.settings.risk.live_min_total_realized_pnl
            ),
            "drawdown_ready": (
                performance.latest_drawdown_pct
                <= self.settings.risk.live_max_drawdown_pct
            ),
            "model_not_degraded": performance.degradation_status == "healthy",
        }
        reasons = []
        if not checks["llm_runtime_configured"]:
            reasons.append("llm_runtime_not_configured")
        if not checks["latest_research_mode_llm"]:
            reasons.append("latest_research_is_fallback")
        if not checks["research_fallback_ratio_ok"]:
            reasons.append("research_fallback_ratio_too_high")
        if not checks["market_streams_fresh"]:
            reasons.append("stale_market_streams_present")
        if not checks["market_latency_ok"]:
            reasons.append("market_latency_not_ok")
        if not checks["closed_trades_ready"]:
            reasons.append("insufficient_closed_trades")
        if not checks["prediction_eval_count_ready"]:
            reasons.append("insufficient_prediction_eval_samples")
        if not checks["holdout_accuracy_ready"]:
            reasons.append("holdout_accuracy_below_floor")
        if not checks["xgboost_accuracy_ready"]:
            reasons.append("xgboost_accuracy_below_floor")
        if not checks["fusion_accuracy_ready"]:
            reasons.append("fusion_accuracy_below_floor")
        if not checks["realized_pnl_ready"]:
            reasons.append("realized_pnl_not_positive")
        if not checks["drawdown_ready"]:
            reasons.append("drawdown_above_limit")
        if not checks["model_not_degraded"]:
            reasons.append(f"model_status={performance.degradation_status}")

        metrics = {
            "runtime_mode": self.settings.app.runtime_mode,
            "allow_live_orders": self.settings.app.allow_live_orders,
            "llm_runtime_configured": health.llm_runtime_configured,
            "latest_research_mode": health.latest_research_mode,
            "recent_research_fallback_ratio_pct": round(
                health.recent_research_fallback_ratio_pct, 2
            ),
            "stale_market_streams": health.stale_market_streams,
            "market_latency_status": health.market_latency_status,
            "closed_trades": performance.total_closed_trades,
            "prediction_eval_count": prediction_eval_count,
            "latest_holdout_accuracy_pct": round(performance.latest_holdout_accuracy, 2),
            "xgboost_accuracy_pct": round(xgboost_accuracy_pct, 2),
            "fusion_accuracy_pct": round(fusion_accuracy_pct, 2),
            "total_realized_pnl": round(performance.total_realized_pnl, 4),
            "latest_drawdown_pct": round(performance.latest_drawdown_pct, 4),
            "degradation_status": performance.degradation_status,
        }
        return {
            "ready": not reasons,
            "reasons": reasons,
            "metrics": metrics,
        }

    @staticmethod
    def render_live_readiness_report(readiness: dict) -> str:
        lines = [
            "# Live Readiness",
            f"- ready: {readiness['ready']}",
            f"- reasons: {', '.join(readiness['reasons']) if readiness['reasons'] else 'none'}",
        ]
        for key, value in readiness["metrics"].items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _run_nextgen_live_guard_callback(
        self,
        *,
        trigger: str,
        reason: str,
        details: str,
    ):
        if self.nextgen_live_guard_callback is None:
            return None
        try:
            return self.nextgen_live_guard_callback(
                trigger=trigger,
                reason=reason,
                details=details,
            )
        except Exception as exc:
            logger.exception("Nextgen autonomy live guard callback failed")
            self.storage.insert_execution_event(
                "nextgen_autonomy_live_guard_callback_failed",
                "SYSTEM",
                {
                    "trigger": trigger,
                    "reason": reason,
                    "details": details,
                    "error": str(exc),
                },
            )
            return None
