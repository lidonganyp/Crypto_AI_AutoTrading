"""Performance aggregation for CryptoAI v3."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from config import Settings, get_settings
from core.i18n import get_default_language, normalize_language, text_for
from core.scoring import objective_score_from_metrics, objective_score_quality
from core.storage import Storage
from core.trade_simulation import effective_trade_horizon_hours, simulate_long_trade


@dataclass
class PerformanceSnapshot:
    runtime_mode: str
    current_execution_symbols: tuple[str, ...]
    total_closed_trades: int
    win_rate_pct: float
    total_realized_pnl: float
    avg_pnl_pct: float
    total_trade_cost: float
    total_slippage_cost: float
    equity_return_pct: float
    recent_closed_trades: int
    recent_expectancy_pct: float
    recent_profit_factor: float
    recent_max_drawdown_pct: float
    recent_sharpe_like: float
    recent_sortino_like: float
    recent_return_volatility_pct: float
    recent_loss_cluster_ratio_pct: float
    recent_drawdown_velocity_pct: float
    avg_holding_hours: float
    latest_equity: float
    latest_drawdown_pct: float
    latest_holdout_accuracy: float
    latest_walkforward_return_pct: float
    prediction_eval_count: int
    prediction_window_size: int
    prediction_accuracy_min_samples: int
    xgboost_accuracy_pct: float
    llm_accuracy_pct: float
    fusion_accuracy_pct: float
    expanded_prediction_eval_count: int
    expanded_prediction_window_size: int
    expanded_xgboost_accuracy_pct: float
    expanded_llm_accuracy_pct: float
    expanded_fusion_accuracy_pct: float
    llm_runtime_configured: bool
    research_fallback_count: int
    research_total_count: int
    research_fallback_ratio_pct: float
    execution_evaluation_count: int
    execution_accuracy_pct: float
    shadow_evaluation_count: int
    shadow_accuracy_pct: float
    current_prediction_eval_count: int
    current_prediction_window_size: int
    current_xgboost_accuracy_pct: float
    current_llm_accuracy_pct: float
    current_fusion_accuracy_pct: float
    current_execution_evaluation_count: int
    current_execution_accuracy_pct: float
    paper_canary_open_count: int
    soft_paper_canary_open_count: int
    paper_exploration_grace_active: bool
    degradation_status: str
    degradation_reason: str
    recommended_xgboost_threshold: float
    recommended_final_score_threshold: float


class PerformanceReporter:
    """Aggregate core metrics from stored artifacts."""

    RECENT_PREDICTION_EVALUATED_TARGET = 50
    EXPANDED_PREDICTION_EVALUATED_TARGET = 200
    PREDICTION_RUN_SCAN_BATCH_SIZE = 500
    MAX_PREDICTION_RUN_SCAN_ROWS = 5000

    def __init__(self, storage: Storage, settings: Settings | None = None):
        self.storage = storage
        self.settings = settings or get_settings()

    def build(self) -> PerformanceSnapshot:
        total_closed = 0
        current_execution_symbols = tuple(
            self._normalize_symbols(self.settings.exchange.symbols)
        )
        win_rate = 0.0
        total_pnl = 0.0
        avg_pnl_pct = 0.0
        total_trade_cost = 0.0
        total_slippage_cost = 0.0
        equity_return_pct = 0.0
        recent_closed_trades = 0
        recent_expectancy_pct = 0.0
        recent_profit_factor = 0.0
        recent_max_drawdown_pct = 0.0
        recent_sharpe_like = 0.0
        recent_sortino_like = 0.0
        recent_return_volatility_pct = 0.0
        recent_loss_cluster_ratio_pct = 0.0
        recent_drawdown_velocity_pct = 0.0
        avg_holding_hours = 0.0
        latest_equity = 0.0
        latest_drawdown = 0.0
        latest_holdout = 0.0
        latest_walkforward = 0.0
        prediction_eval_count = 0
        prediction_window_size = self.RECENT_PREDICTION_EVALUATED_TARGET
        prediction_accuracy_min_samples = int(
            self.settings.risk.model_accuracy_min_samples
        )
        xgboost_accuracy = 0.0
        llm_accuracy = 0.0
        fusion_accuracy = 0.0
        expanded_prediction_eval_count = 0
        expanded_prediction_window_size = self.EXPANDED_PREDICTION_EVALUATED_TARGET
        expanded_xgboost_accuracy = 0.0
        expanded_llm_accuracy = 0.0
        expanded_fusion_accuracy = 0.0
        llm_runtime_configured = bool(
            self.settings.llm.deepseek_api_key.get_secret_value()
            or self.settings.llm.qwen_api_key.get_secret_value()
        )
        research_fallback_count = 0
        research_total_count = 0
        research_fallback_ratio_pct = 0.0
        execution_evaluation_count = 0
        execution_accuracy_pct = 0.0
        shadow_evaluation_count = 0
        shadow_accuracy_pct = 0.0
        current_prediction_eval_count = 0
        current_prediction_window_size = self.RECENT_PREDICTION_EVALUATED_TARGET
        current_xgboost_accuracy_pct = 0.0
        current_llm_accuracy_pct = 0.0
        current_fusion_accuracy_pct = 0.0
        current_execution_evaluation_count = 0
        current_execution_accuracy_pct = 0.0
        paper_canary_open_count = 0
        soft_paper_canary_open_count = 0
        paper_exploration_grace_active = False
        degradation_status = "healthy"
        degradation_reason = ""
        recommended_xgboost_threshold = self.settings.model.xgboost_probability_threshold
        recommended_final_score_threshold = self.settings.model.final_score_threshold

        with self.storage._conn() as conn:
            ledger_summary = self._build_ledger_summary(conn)
            total_closed = int(ledger_summary["total_closed_trades"])
            win_rate = float(ledger_summary["win_rate_pct"])
            total_pnl = float(ledger_summary["total_realized_pnl"])
            avg_pnl_pct = float(ledger_summary["avg_pnl_pct"])
            total_trade_cost = float(ledger_summary["total_trade_cost"])
            total_slippage_cost = float(ledger_summary["total_slippage_cost"])
            recent_closed_trades = int(ledger_summary["recent_closed_trades"])
            recent_expectancy_pct = float(ledger_summary["recent_expectancy_pct"])
            recent_profit_factor = float(ledger_summary["recent_profit_factor"])
            recent_max_drawdown_pct = float(ledger_summary["recent_max_drawdown_pct"])
            recent_sharpe_like = float(ledger_summary["recent_sharpe_like"])
            recent_sortino_like = float(ledger_summary["recent_sortino_like"])
            recent_return_volatility_pct = float(
                ledger_summary["recent_return_volatility_pct"]
            )
            recent_loss_cluster_ratio_pct = float(
                ledger_summary["recent_loss_cluster_ratio_pct"]
            )
            recent_drawdown_velocity_pct = float(
                ledger_summary["recent_drawdown_velocity_pct"]
            )
            avg_holding_hours = float(ledger_summary["avg_holding_hours"])

            account = conn.execute(
                "SELECT equity, drawdown_pct FROM account_snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if account:
                latest_equity = float(account["equity"])
                latest_drawdown = float(account["drawdown_pct"]) * 100
            first_account = conn.execute(
                "SELECT equity FROM account_snapshots ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            if (
                first_account
                and float(first_account["equity"] or 0.0) > 0
                and latest_equity > 0
            ):
                equity_return_pct = (
                    latest_equity / float(first_account["equity"]) - 1.0
                ) * 100

            training = conn.execute(
                "SELECT metadata_json FROM training_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if training:
                latest_holdout = float(
                    json.loads(training["metadata_json"]).get("holdout_accuracy", 0.0)
                ) * 100

            walkforward = conn.execute(
                "SELECT summary_json FROM walkforward_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if walkforward:
                latest_walkforward = float(
                    json.loads(walkforward["summary_json"]).get("total_return_pct", 0.0)
                )

            expanded_rows = self._load_recent_evaluated_prediction_rows(
                conn,
                target_count=self.EXPANDED_PREDICTION_EVALUATED_TARGET,
            )
            rows = expanded_rows[: self.RECENT_PREDICTION_EVALUATED_TARGET]
            metrics = self._compute_prediction_accuracy_from_evaluated_rows(rows)
            prediction_eval_count = metrics["count"]
            xgboost_accuracy = metrics["xgboost_accuracy_pct"]
            llm_accuracy = metrics["llm_accuracy_pct"]
            fusion_accuracy = metrics["fusion_accuracy_pct"]
            research_total_count = len(rows)
            research_fallback_count = sum(
                1
                for row in rows
                if "fallback_research_model"
                in str((row.get("research") or {}).get("key_reason", []))
            )
            research_fallback_ratio_pct = (
                research_fallback_count / research_total_count * 100
                if research_total_count
                else 0.0
            )
            current_rows = self._load_recent_evaluated_prediction_rows(
                conn,
                target_count=self.RECENT_PREDICTION_EVALUATED_TARGET,
                allowed_symbols=current_execution_symbols,
            )
            current_metrics = self._compute_prediction_accuracy_from_evaluated_rows(
                current_rows
            )
            current_prediction_eval_count = int(current_metrics["count"])
            current_xgboost_accuracy_pct = float(
                current_metrics["xgboost_accuracy_pct"]
            )
            current_llm_accuracy_pct = float(current_metrics["llm_accuracy_pct"])
            current_fusion_accuracy_pct = float(
                current_metrics["fusion_accuracy_pct"]
            )

            expanded_prediction_eval_count = prediction_eval_count
            expanded_xgboost_accuracy = xgboost_accuracy
            expanded_llm_accuracy = llm_accuracy
            expanded_fusion_accuracy = fusion_accuracy
            if (
                self.EXPANDED_PREDICTION_EVALUATED_TARGET
                > self.RECENT_PREDICTION_EVALUATED_TARGET
            ):
                expanded_metrics = self._compute_prediction_accuracy_from_evaluated_rows(
                    expanded_rows
                )
                expanded_prediction_eval_count = int(expanded_metrics["count"])
                expanded_xgboost_accuracy = float(
                    expanded_metrics["xgboost_accuracy_pct"]
                )
                expanded_llm_accuracy = float(expanded_metrics["llm_accuracy_pct"])
                expanded_fusion_accuracy = float(
                    expanded_metrics["fusion_accuracy_pct"]
                )

            evaluation_summary = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN evaluation_type='execution' THEN 1 ELSE 0 END) AS execution_count,
                    SUM(CASE WHEN evaluation_type='shadow_observation' THEN 1 ELSE 0 END) AS shadow_count,
                    AVG(CASE WHEN evaluation_type='execution' THEN is_correct END) AS execution_accuracy,
                    AVG(CASE WHEN evaluation_type='shadow_observation' THEN is_correct END) AS shadow_accuracy
                FROM prediction_evaluations
                """
            ).fetchone()
            if evaluation_summary:
                execution_evaluation_count = int(
                    evaluation_summary["execution_count"] or 0
                )
                shadow_evaluation_count = int(
                    evaluation_summary["shadow_count"] or 0
                )
                execution_accuracy = evaluation_summary["execution_accuracy"]
                shadow_accuracy = evaluation_summary["shadow_accuracy"]
                execution_accuracy_pct = (
                    float(execution_accuracy) * 100
                    if execution_accuracy is not None
                    else 0.0
                )
                shadow_accuracy_pct = (
                    float(shadow_accuracy) * 100
                    if shadow_accuracy is not None
                    else 0.0
                )
            current_execution_summary = self._evaluation_accuracy_summary(
                conn,
                evaluation_type="execution",
                allowed_symbols=current_execution_symbols,
            )
            current_execution_evaluation_count = int(
                current_execution_summary["count"] or 0
            )
            current_execution_accuracy_pct = float(
                current_execution_summary["accuracy_pct"] or 0.0
            )

            canary_counts = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_count,
                    SUM(
                        CASE
                            WHEN json_extract(payload_json, '$.canary_mode')='soft_review'
                            THEN 1 ELSE 0
                        END
                    ) AS soft_count
                FROM execution_events
                WHERE event_type='paper_canary_open'
                """
            ).fetchone()
            if canary_counts:
                paper_canary_open_count = int(canary_counts["total_count"] or 0)
                soft_paper_canary_open_count = int(canary_counts["soft_count"] or 0)

            degradation = self._detect_degradation(
                xgboost_accuracy=xgboost_accuracy,
                fusion_accuracy=(
                    fusion_accuracy if metrics["fusion_total"] > 0 else None
                ),
                holdout_accuracy=latest_holdout,
                prediction_eval_count=prediction_eval_count,
                total_closed_trades=total_closed,
                paper_canary_open_count=paper_canary_open_count,
                recent_closed_trades=recent_closed_trades,
                recent_expectancy_pct=recent_expectancy_pct,
                recent_profit_factor=recent_profit_factor,
                recent_max_drawdown_pct=recent_max_drawdown_pct,
            )
            degradation_status = degradation["status"]
            degradation_reason = degradation["reason"]
            recommended_xgboost_threshold = degradation["recommended_xgboost_threshold"]
            recommended_final_score_threshold = degradation["recommended_final_score_threshold"]
            paper_exploration_grace_active = bool(
                degradation.get("paper_exploration_grace_active", False)
            )

        return PerformanceSnapshot(
            runtime_mode=self.settings.app.runtime_mode,
            current_execution_symbols=current_execution_symbols,
            total_closed_trades=total_closed,
            win_rate_pct=win_rate,
            total_realized_pnl=total_pnl,
            avg_pnl_pct=avg_pnl_pct,
            total_trade_cost=total_trade_cost,
            total_slippage_cost=total_slippage_cost,
            equity_return_pct=equity_return_pct,
            recent_closed_trades=recent_closed_trades,
            recent_expectancy_pct=recent_expectancy_pct,
            recent_profit_factor=recent_profit_factor,
            recent_max_drawdown_pct=recent_max_drawdown_pct,
            recent_sharpe_like=recent_sharpe_like,
            recent_sortino_like=recent_sortino_like,
            recent_return_volatility_pct=recent_return_volatility_pct,
            recent_loss_cluster_ratio_pct=recent_loss_cluster_ratio_pct,
            recent_drawdown_velocity_pct=recent_drawdown_velocity_pct,
            avg_holding_hours=avg_holding_hours,
            latest_equity=latest_equity,
            latest_drawdown_pct=latest_drawdown,
            latest_holdout_accuracy=latest_holdout,
            latest_walkforward_return_pct=latest_walkforward,
            prediction_eval_count=prediction_eval_count,
            prediction_window_size=prediction_window_size,
            prediction_accuracy_min_samples=prediction_accuracy_min_samples,
            xgboost_accuracy_pct=xgboost_accuracy,
            llm_accuracy_pct=llm_accuracy,
            fusion_accuracy_pct=fusion_accuracy,
            expanded_prediction_eval_count=expanded_prediction_eval_count,
            expanded_prediction_window_size=expanded_prediction_window_size,
            expanded_xgboost_accuracy_pct=expanded_xgboost_accuracy,
            expanded_llm_accuracy_pct=expanded_llm_accuracy,
            expanded_fusion_accuracy_pct=expanded_fusion_accuracy,
            llm_runtime_configured=llm_runtime_configured,
            research_fallback_count=research_fallback_count,
            research_total_count=research_total_count,
            research_fallback_ratio_pct=research_fallback_ratio_pct,
            execution_evaluation_count=execution_evaluation_count,
            execution_accuracy_pct=execution_accuracy_pct,
            shadow_evaluation_count=shadow_evaluation_count,
            shadow_accuracy_pct=shadow_accuracy_pct,
            current_prediction_eval_count=current_prediction_eval_count,
            current_prediction_window_size=current_prediction_window_size,
            current_xgboost_accuracy_pct=current_xgboost_accuracy_pct,
            current_llm_accuracy_pct=current_llm_accuracy_pct,
            current_fusion_accuracy_pct=current_fusion_accuracy_pct,
            current_execution_evaluation_count=current_execution_evaluation_count,
            current_execution_accuracy_pct=current_execution_accuracy_pct,
            paper_canary_open_count=paper_canary_open_count,
            soft_paper_canary_open_count=soft_paper_canary_open_count,
            paper_exploration_grace_active=paper_exploration_grace_active,
            degradation_status=degradation_status,
            degradation_reason=degradation_reason,
            recommended_xgboost_threshold=recommended_xgboost_threshold,
            recommended_final_score_threshold=recommended_final_score_threshold,
        )

    def _build_ledger_summary(self, conn, recent_limit: int = 20) -> dict[str, float | int]:
        ledger_rows = conn.execute(
            "SELECT trade_id, net_pnl, fee_cost, slippage_cost FROM pnl_ledger"
        ).fetchall()
        if not ledger_rows:
            closed = self.storage.get_closed_trades()
            open_trades = self.storage.get_open_trades()
            total_closed = len(closed)
            wins = [trade for trade in closed if (trade.get("pnl") or 0) > 0]
            return {
                "total_closed_trades": total_closed,
                "win_rate_pct": (len(wins) / total_closed * 100) if total_closed else 0.0,
                "total_realized_pnl": sum(trade.get("pnl") or 0.0 for trade in closed)
                + sum(trade.get("pnl") or 0.0 for trade in open_trades),
                "avg_pnl_pct": (
                    sum(trade.get("pnl_pct") or 0.0 for trade in closed) / total_closed
                    if total_closed
                    else 0.0
                ),
                "total_trade_cost": 0.0,
                "total_slippage_cost": 0.0,
                "recent_closed_trades": total_closed,
                "recent_expectancy_pct": 0.0,
                "recent_profit_factor": 0.0,
                "recent_max_drawdown_pct": 0.0,
                "recent_sharpe_like": 0.0,
                "recent_sortino_like": 0.0,
                "recent_return_volatility_pct": 0.0,
                "recent_loss_cluster_ratio_pct": 0.0,
                "recent_drawdown_velocity_pct": 0.0,
                "avg_holding_hours": 0.0,
            }

        total_realized_pnl = sum(float(row["net_pnl"] or 0.0) for row in ledger_rows)
        total_trade_cost = sum(float(row["fee_cost"] or 0.0) for row in ledger_rows)
        total_slippage_cost = sum(float(row["slippage_cost"] or 0.0) for row in ledger_rows)

        closed_rows = conn.execute(
            "SELECT id, entry_price, quantity, initial_quantity, entry_time, exit_time "
            "FROM trades WHERE status='closed' ORDER BY exit_time DESC"
        ).fetchall()
        by_trade: dict[str, list] = {}
        for row in ledger_rows:
            by_trade.setdefault(str(row["trade_id"]), []).append(row)

        closed_summaries: list[dict[str, float | str]] = []
        for trade in closed_rows:
            trade_id = str(trade["id"])
            trade_ledger = by_trade.get(trade_id, [])
            if not trade_ledger:
                continue
            entry_price = float(trade["entry_price"] or 0.0)
            initial_qty = float(trade["initial_quantity"] or trade["quantity"] or 0.0)
            base_notional = entry_price * initial_qty
            net_pnl = sum(float(item["net_pnl"] or 0.0) for item in trade_ledger)
            fee_cost = sum(float(item["fee_cost"] or 0.0) for item in trade_ledger)
            slippage_cost = sum(
                float(item["slippage_cost"] or 0.0) for item in trade_ledger
            )
            holding_hours = 0.0
            try:
                holding_hours = (
                    datetime.fromisoformat(str(trade["exit_time"]))
                    - datetime.fromisoformat(str(trade["entry_time"]))
                ).total_seconds() / 3600
            except Exception:
                holding_hours = 0.0
            closed_summaries.append(
                {
                    "trade_id": trade_id,
                    "exit_time": str(trade["exit_time"] or ""),
                    "net_pnl": net_pnl,
                    "net_return_pct": (
                        net_pnl / base_notional * 100 if base_notional > 0 else 0.0
                    ),
                    "fee_cost": fee_cost,
                    "slippage_cost": slippage_cost,
                    "holding_hours": holding_hours,
                }
            )

        total_closed = len(closed_summaries)
        wins = [trade for trade in closed_summaries if float(trade["net_pnl"]) > 0]
        avg_pnl_pct = (
            sum(float(trade["net_return_pct"]) for trade in closed_summaries) / total_closed
            if total_closed
            else 0.0
        )
        avg_holding_hours = (
            sum(float(trade["holding_hours"]) for trade in closed_summaries) / total_closed
            if total_closed
            else 0.0
        )

        recent_trades = list(reversed(closed_summaries[:recent_limit]))
        recent_returns = [float(trade["net_return_pct"]) for trade in recent_trades]
        recent_pnls = [float(trade["net_pnl"]) for trade in recent_trades]
        recent_expectancy_pct = (
            sum(recent_returns) / len(recent_returns) if recent_returns else 0.0
        )
        recent_profit_factor = self._profit_factor(recent_pnls)
        recent_max_drawdown_pct = self._returns_max_drawdown_pct(recent_returns)
        recent_sharpe_like = self._sharpe_like(recent_returns)
        recent_sortino_like = self._sortino_like(recent_returns)
        recent_return_volatility_pct = self._return_volatility_pct(recent_returns)
        recent_loss_cluster_ratio_pct = self._loss_cluster_ratio_pct(recent_returns)
        recent_drawdown_velocity_pct = self._drawdown_velocity_pct(
            recent_max_drawdown_pct,
            len(recent_returns),
        )
        return {
            "total_closed_trades": total_closed,
            "win_rate_pct": (len(wins) / total_closed * 100) if total_closed else 0.0,
            "total_realized_pnl": total_realized_pnl,
            "avg_pnl_pct": avg_pnl_pct,
            "total_trade_cost": total_trade_cost,
            "total_slippage_cost": total_slippage_cost,
            "recent_closed_trades": len(recent_trades),
            "recent_expectancy_pct": recent_expectancy_pct,
            "recent_profit_factor": recent_profit_factor,
            "recent_max_drawdown_pct": recent_max_drawdown_pct,
            "recent_sharpe_like": recent_sharpe_like,
            "recent_sortino_like": recent_sortino_like,
            "recent_return_volatility_pct": recent_return_volatility_pct,
            "recent_loss_cluster_ratio_pct": recent_loss_cluster_ratio_pct,
            "recent_drawdown_velocity_pct": recent_drawdown_velocity_pct,
            "avg_holding_hours": avg_holding_hours,
        }

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
        return max_drawdown * 100

    @staticmethod
    def _sharpe_like(values: list[float]) -> float:
        if len(values) < 2:
            return float(values[0]) if values else 0.0
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_dev = math.sqrt(max(variance, 0.0))
        return mean_value / std_dev if std_dev > 1e-12 else (5.0 if mean_value > 0 else 0.0)

    @staticmethod
    def _sortino_like(values: list[float]) -> float:
        if not values:
            return 0.0
        mean_value = sum(values) / len(values)
        downside = [value for value in values if value < 0]
        if not downside:
            return 5.0 if mean_value > 0 else 0.0
        downside_variance = sum(value**2 for value in downside) / len(downside)
        downside_dev = math.sqrt(max(downside_variance, 0.0))
        return mean_value / downside_dev if downside_dev > 1e-12 else 0.0

    @staticmethod
    def _return_volatility_pct(values: list[float]) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return abs(float(values[0]))
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        return math.sqrt(max(variance, 0.0))

    @staticmethod
    def _loss_cluster_ratio_pct(values: list[float]) -> float:
        if not values:
            return 0.0
        clustered_losses = 0
        current_streak = 0
        for value in list(values) + [0.0]:
            if float(value) < 0.0:
                current_streak += 1
                continue
            if current_streak >= 2:
                clustered_losses += current_streak
            current_streak = 0
        return clustered_losses / len(values) * 100.0

    @staticmethod
    def _drawdown_velocity_pct(max_drawdown_pct: float, sample_count: int) -> float:
        if sample_count <= 0:
            return 0.0
        return float(max_drawdown_pct or 0.0) / sample_count

    def build_symbol_accuracy_summary(self, limit: int = 500) -> dict[str, dict[str, float | int]]:
        with self.storage._conn() as conn:
            rows = conn.execute(
                "SELECT id, symbol, timestamp, up_probability, research_json, decision_json "
                "FROM prediction_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            rows = self._dedupe_prediction_rows(rows)

            per_symbol: dict[str, dict[str, float | int]] = {}
            timeframe = self.settings.strategy.primary_timeframe
            for row in rows:
                timestamp = datetime.fromisoformat(row["timestamp"])
                try:
                    decision = json.loads(row["decision_json"])
                except Exception:
                    decision = {}
                outcome = self._simulate_prediction_outcome(
                    conn,
                    symbol=row["symbol"],
                    timestamp=timestamp,
                    timeframe=timeframe,
                    decision=decision,
                )
                if outcome is None:
                    continue
                actual_up = bool(outcome["actual_up"])
                xgboost_threshold = float(
                    decision.get(
                        "xgboost_threshold",
                        self.settings.model.xgboost_probability_threshold,
                    )
                )
                predicted_up = float(row["up_probability"]) >= xgboost_threshold
                bucket = per_symbol.setdefault(
                    str(row["symbol"]),
                    {"count": 0, "correct": 0, "accuracy_pct": 0.0},
                )
                bucket["count"] += 1
                bucket["correct"] += int(predicted_up == actual_up)

            for bucket in per_symbol.values():
                count = int(bucket["count"])
                correct = int(bucket["correct"])
                bucket["accuracy_pct"] = correct / count * 100 if count else 0.0
            return per_symbol

    def build_symbol_edge_summary(
        self,
        limit: int = 500,
        evaluation_type: str = "execution",
    ) -> dict[str, dict[str, float | int]]:
        with self.storage._conn() as conn:
            evaluation_types = [evaluation_type]
            if evaluation_type == "execution":
                evaluation_types.append("paper_canary")
            placeholders = ",".join("?" for _ in evaluation_types)
            rows = conn.execute(
                "SELECT symbol, timestamp, predicted_up, is_correct, entry_close, future_close, metadata_json "
                "FROM prediction_evaluations "
                f"WHERE evaluation_type IN ({placeholders}) "
                "ORDER BY created_at DESC LIMIT ?",
                (*evaluation_types, limit),
            ).fetchall()
            if evaluation_type == "execution":
                rows = list(rows) + self._pending_paper_canary_feedback(
                    conn,
                    limit=limit,
                )

        per_symbol_rows: dict[str, list[dict]] = {}
        for row in rows:
            symbol = str(row["symbol"] or "").strip()
            if not symbol:
                continue
            per_symbol_rows.setdefault(symbol, []).append(dict(row))

        summary: dict[str, dict[str, float | int]] = {}
        for symbol, symbol_rows in per_symbol_rows.items():
            sample_count = len(symbol_rows)
            if sample_count <= 0:
                continue
            accuracy = (
                sum(int(row.get("is_correct") or 0) for row in symbol_rows) / sample_count
            )
            executed_records: list[dict[str, float | str]] = []
            for row in symbol_rows:
                try:
                    metadata = json.loads(row.get("metadata_json") or "{}")
                except Exception:
                    metadata = {}
                predicted_up = int(row.get("predicted_up") or 0) == 1
                trade_taken = bool(metadata.get("trade_taken", predicted_up))
                if not trade_taken:
                    continue
                entry_close = float(row.get("entry_close") or 0.0)
                future_close = float(row.get("future_close") or 0.0)
                fallback_opportunity_return_pct = (
                    ((future_close / entry_close) - 1.0) * 100
                    if entry_close > 0
                    else 0.0
                )
                estimated_cost_pct = float(
                    metadata.get(
                        "estimated_cost_pct",
                        metadata.get("cost_pct", 0.15 if predicted_up else 0.0),
                    )
                    or 0.0
                )
                executed_records.append(
                    {
                        "timestamp": str(row.get("timestamp") or ""),
                        "trade_net_return_pct": float(
                            metadata.get(
                                "trade_net_return_pct",
                                fallback_opportunity_return_pct - estimated_cost_pct,
                            )
                            or 0.0
                        ),
                        "cost_pct": float(
                            metadata.get("cost_pct", estimated_cost_pct) or 0.0
                        ),
                    }
                )
            executed_count = len(executed_records)
            trade_returns_pct = [
                float(item["trade_net_return_pct"] or 0.0) for item in executed_records
            ]
            avg_trade_return_pct = (
                sum(trade_returns_pct) / executed_count if executed_count else 0.0
            )
            expectancy_pct = (
                sum(trade_returns_pct) / sample_count if sample_count else 0.0
            )
            trade_win_rate = (
                sum(1 for value in trade_returns_pct if value > 0) / executed_count
                if executed_count
                else 0.0
            )
            avg_cost_pct = (
                sum(float(item["cost_pct"] or 0.0) for item in executed_records)
                / executed_count
                if executed_count
                else 0.0
            )
            ordered_returns_pct = [
                float(item["trade_net_return_pct"] or 0.0)
                for item in sorted(executed_records, key=lambda item: item["timestamp"])
            ]
            metrics = {
                "sample_count": sample_count,
                "executed_count": executed_count,
                "count": sample_count,
                "accuracy": accuracy,
                "accuracy_pct": accuracy * 100,
                "expectancy_pct": expectancy_pct,
                "avg_trade_return_pct": avg_trade_return_pct,
                "profit_factor": self._profit_factor(trade_returns_pct),
                "max_drawdown_pct": self._returns_max_drawdown_pct(ordered_returns_pct),
                "trade_win_rate": trade_win_rate,
                "avg_cost_pct": avg_cost_pct,
            }
            metrics["objective_score"] = objective_score_from_metrics(metrics)
            metrics["objective_quality"] = objective_score_quality(metrics)
            summary[symbol] = metrics
        return summary

    def build_pipeline_mode_summary(
        self,
        pipeline_modes: list[str] | tuple[str, ...] | None = None,
        limit: int = 20,
    ) -> dict[str, dict[str, float | int | dict[str, int] | str]]:
        normalized_modes = [
            str(mode or "").strip()
            for mode in (pipeline_modes or [])
            if str(mode or "").strip()
        ]
        if not normalized_modes:
            normalized_modes = ["execution", "fast_alpha", "paper_canary"]

        placeholders = ",".join("?" for _ in normalized_modes)
        with self.storage._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    t.id,
                    t.entry_price,
                    t.quantity,
                    t.initial_quantity,
                    t.entry_time,
                    t.exit_time,
                    t.pnl,
                    t.pnl_pct,
                    COALESCE(json_extract(t.metadata_json, '$.pipeline_mode'), 'execution') AS pipeline_mode,
                    COUNT(l.id) AS ledger_count,
                    COALESCE(SUM(l.net_pnl), t.pnl, 0.0) AS net_pnl
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
                    t.entry_time,
                    t.exit_time,
                    t.pnl,
                    t.pnl_pct,
                    COALESCE(json_extract(t.metadata_json, '$.pipeline_mode'), 'execution')
                ORDER BY COALESCE(t.exit_time, t.entry_time) DESC
                LIMIT ?
                """,
                (*normalized_modes, max(1, int(limit))),
            ).fetchall()

        per_mode: dict[str, list[dict[str, float | str]]] = {
            mode: [] for mode in normalized_modes
        }
        combined: list[dict[str, float | str]] = []

        for row in rows:
            pipeline_mode = str(row["pipeline_mode"] or "execution").strip() or "execution"
            base_qty = float(row["initial_quantity"] or row["quantity"] or 0.0)
            base_notional = float(row["entry_price"] or 0.0) * base_qty
            ledger_count = int(row["ledger_count"] or 0)
            if ledger_count > 0 and base_notional > 0:
                net_return_pct = float(row["net_pnl"] or 0.0) / base_notional * 100.0
            else:
                net_return_pct = float(row["pnl_pct"] or 0.0)

            holding_hours = 0.0
            try:
                holding_hours = (
                    datetime.fromisoformat(str(row["exit_time"]))
                    - datetime.fromisoformat(str(row["entry_time"]))
                ).total_seconds() / 3600.0
            except Exception:
                holding_hours = 0.0

            record = {
                "trade_id": str(row["id"] or ""),
                "pipeline_mode": pipeline_mode,
                "net_pnl": float(row["net_pnl"] or row["pnl"] or 0.0),
                "net_return_pct": float(net_return_pct),
                "holding_hours": float(holding_hours),
                "exit_time": str(row["exit_time"] or row["entry_time"] or ""),
            }
            if pipeline_mode in per_mode:
                per_mode[pipeline_mode].append(record)
            combined.append(record)

        def summarize(
            label: str,
            records: list[dict[str, float | str]],
        ) -> dict[str, float | int | dict[str, int] | str]:
            if not records:
                return {
                    "pipeline_mode": label,
                    "closed_trade_count": 0,
                    "win_rate_pct": 0.0,
                    "expectancy_pct": 0.0,
                    "profit_factor": 0.0,
                    "max_drawdown_pct": 0.0,
                    "avg_holding_hours": 0.0,
                    "total_net_pnl": 0.0,
                    "mode_counts": {},
                }

            ordered = sorted(records, key=lambda item: str(item["exit_time"]))
            returns_pct = [float(item["net_return_pct"] or 0.0) for item in ordered]
            pnls = [float(item["net_pnl"] or 0.0) for item in ordered]
            holding_hours = [float(item["holding_hours"] or 0.0) for item in ordered]
            wins = sum(1 for value in returns_pct if value > 0)
            mode_counts: dict[str, int] = {}
            for item in ordered:
                mode = str(item.get("pipeline_mode") or "")
                if not mode:
                    continue
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            return {
                "pipeline_mode": label,
                "closed_trade_count": len(ordered),
                "win_rate_pct": wins / len(ordered) * 100.0,
                "expectancy_pct": sum(returns_pct) / len(ordered),
                "profit_factor": self._profit_factor(pnls),
                "max_drawdown_pct": self._returns_max_drawdown_pct(returns_pct),
                "avg_holding_hours": (
                    sum(holding_hours) / len(holding_hours) if holding_hours else 0.0
                ),
                "total_net_pnl": sum(pnls),
                "mode_counts": mode_counts,
            }

        summary = {
            mode: summarize(mode, records)
            for mode, records in per_mode.items()
        }
        summary["_combined"] = summarize("_combined", combined)
        return summary

    def _pending_paper_canary_feedback(self, conn, limit: int = 100) -> list[dict]:
        rows = conn.execute(
            """
            SELECT id, symbol, entry_time, entry_price, quantity, initial_quantity,
                   pnl_pct, metadata_json, exit_time
            FROM trades
            WHERE json_extract(metadata_json, '$.pipeline_mode')='paper_canary'
              AND (
                    status='closed'
                    OR (status='open' AND pnl IS NOT NULL)
                  )
            ORDER BY exit_time DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        pending: list[dict] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            prediction_timestamp = str(metadata.get("prediction_timestamp") or "")
            if prediction_timestamp:
                exists = conn.execute(
                    """
                    SELECT 1 FROM prediction_evaluations
                    WHERE symbol = ?
                      AND timestamp = ?
                      AND evaluation_type = 'paper_canary'
                    LIMIT 1
                    """,
                    (row["symbol"], prediction_timestamp),
                ).fetchone()
                if exists:
                    continue
            trade_id = str(row["id"] or "")
            ledger_rows = conn.execute(
                """
                SELECT net_pnl, fee_cost, slippage_cost
                FROM pnl_ledger
                WHERE trade_id = ?
                ORDER BY event_time ASC, id ASC
                """,
                (trade_id,),
            ).fetchall()
            base_qty = float(row["initial_quantity"] or row["quantity"] or 0.0)
            base_notional = float(row["entry_price"] or 0.0) * base_qty
            total_net_pnl = sum(float(item["net_pnl"] or 0.0) for item in ledger_rows)
            total_cost = sum(
                float(item["fee_cost"] or 0.0) + float(item["slippage_cost"] or 0.0)
                for item in ledger_rows
            )
            if base_notional > 0:
                trade_net_return_pct = total_net_pnl / base_notional * 100.0
                cost_pct = total_cost / base_notional * 100.0
            else:
                trade_net_return_pct = float(row["pnl_pct"] or 0.0)
                cost_pct = 0.0
            status = str(row["exit_time"] or "").strip()
            pending.append(
                {
                    "symbol": row["symbol"],
                    "timestamp": prediction_timestamp or str(row["entry_time"] or ""),
                    "predicted_up": 1,
                    "is_correct": 1 if trade_net_return_pct > 0 else 0,
                    "entry_close": float(row["entry_price"] or 0.0),
                    "future_close": float(row["entry_price"] or 0.0)
                    * (1.0 + trade_net_return_pct / 100.0),
                    "metadata_json": json.dumps(
                        {
                            "trade_taken": True,
                            "estimated_cost_pct": cost_pct,
                            "cost_pct": cost_pct,
                            "trade_net_return_pct": trade_net_return_pct,
                            "opportunity_return_pct": trade_net_return_pct,
                            "opportunity_net_return_pct": trade_net_return_pct,
                            "source": (
                                "paper_canary_trade_close"
                                if status
                                else "paper_canary_partial_close"
                            ),
                        },
                        default=str,
                    ),
                }
            )
        return pending

    def _detect_degradation(
        self,
        xgboost_accuracy: float,
        fusion_accuracy: float | None,
        holdout_accuracy: float,
        prediction_eval_count: int,
        total_closed_trades: int = 0,
        paper_canary_open_count: int = 0,
        recent_closed_trades: int = 0,
        recent_expectancy_pct: float = 0.0,
        recent_profit_factor: float = 0.0,
        recent_max_drawdown_pct: float = 0.0,
    ) -> dict[str, float | str]:
        base_xgb = self.settings.model.xgboost_probability_threshold
        base_final = self.settings.model.final_score_threshold
        tighten = self.settings.risk.model_threshold_tighten_pct
        paper_grace = self._paper_exploration_grace_active(
            total_closed_trades=total_closed_trades,
            paper_canary_open_count=paper_canary_open_count,
        )
        realized_edge_positive = self._realized_edge_positive(
            recent_closed_trades=recent_closed_trades,
            recent_expectancy_pct=recent_expectancy_pct,
            recent_profit_factor=recent_profit_factor,
            recent_max_drawdown_pct=recent_max_drawdown_pct,
        )
        realized_edge_negative = self._realized_edge_negative(
            recent_closed_trades=recent_closed_trades,
            recent_expectancy_pct=recent_expectancy_pct,
            recent_profit_factor=recent_profit_factor,
            recent_max_drawdown_pct=recent_max_drawdown_pct,
        )
        paper_grace_tighten = min(
            tighten,
            float(self.settings.risk.paper_exploration_grace_threshold_tighten_pct or 0.0),
        )

        if prediction_eval_count < self.settings.risk.model_accuracy_min_samples:
            return {
                "status": "warming_up",
                "reason": "insufficient_samples",
                "recommended_xgboost_threshold": base_xgb,
                "recommended_final_score_threshold": base_final,
                "paper_exploration_grace_active": False,
            }

        accuracy_gap = holdout_accuracy - xgboost_accuracy
        if self.settings.app.runtime_mode == "paper" and realized_edge_positive:
            return {
                "status": "healthy",
                "reason": "realized_edge_override",
                "recommended_xgboost_threshold": base_xgb,
                "recommended_final_score_threshold": base_final,
                "paper_exploration_grace_active": False,
            }
        if realized_edge_negative:
            return {
                "status": "degraded",
                "reason": "realized_edge_negative",
                "recommended_xgboost_threshold": min(0.95, base_xgb + tighten),
                "recommended_final_score_threshold": min(0.99, base_final + tighten),
                "paper_exploration_grace_active": False,
            }
        if (
            xgboost_accuracy < self.settings.risk.model_disable_floor_pct
            or (
                fusion_accuracy is not None
                and fusion_accuracy < self.settings.risk.model_disable_floor_pct
            )
        ):
            if self.settings.app.runtime_mode == "paper":
                if paper_grace:
                    return {
                        "status": "degraded",
                        "reason": "paper_mode_accuracy_guard_exploration_grace",
                        "recommended_xgboost_threshold": min(0.95, base_xgb + paper_grace_tighten),
                        "recommended_final_score_threshold": min(0.99, base_final + paper_grace_tighten),
                        "paper_exploration_grace_active": True,
                    }
                return {
                    "status": "degraded",
                    "reason": "paper_mode_accuracy_guard",
                    "recommended_xgboost_threshold": min(0.95, base_xgb + tighten * 2),
                    "recommended_final_score_threshold": min(0.99, base_final + tighten * 2),
                    "paper_exploration_grace_active": False,
                }
            return {
                "status": "disabled",
                "reason": "live_accuracy_below_disable_floor",
                "recommended_xgboost_threshold": min(0.95, base_xgb + tighten * 2),
                "recommended_final_score_threshold": min(0.99, base_final + tighten * 2),
                "paper_exploration_grace_active": False,
            }

        if (
            xgboost_accuracy < self.settings.risk.model_degrade_floor_pct
            or accuracy_gap > self.settings.risk.model_decay_gap_pct
        ):
            reason = (
                "accuracy_gap_vs_holdout"
                if accuracy_gap > self.settings.risk.model_decay_gap_pct
                else "live_accuracy_below_degrade_floor"
            )
            if self.settings.app.runtime_mode == "paper":
                if paper_grace:
                    return {
                        "status": "degraded",
                        "reason": f"{reason}_exploration_grace",
                        "recommended_xgboost_threshold": min(0.95, base_xgb + paper_grace_tighten),
                        "recommended_final_score_threshold": min(0.99, base_final + paper_grace_tighten),
                        "paper_exploration_grace_active": True,
                    }
                return {
                    "status": "degraded",
                    "reason": reason,
                    "recommended_xgboost_threshold": min(0.95, base_xgb + tighten),
                    "recommended_final_score_threshold": min(0.99, base_final + tighten),
                    "paper_exploration_grace_active": False,
                }
            return {
                "status": "degraded",
                "reason": reason,
                "recommended_xgboost_threshold": min(0.95, base_xgb + tighten),
                "recommended_final_score_threshold": min(0.99, base_final + tighten),
                "paper_exploration_grace_active": False,
            }

        return {
            "status": "healthy",
            "reason": "",
            "recommended_xgboost_threshold": base_xgb,
            "recommended_final_score_threshold": base_final,
            "paper_exploration_grace_active": False,
        }

    def _paper_exploration_grace_active(
        self,
        *,
        total_closed_trades: int,
        paper_canary_open_count: int,
    ) -> bool:
        if self.settings.app.runtime_mode != "paper":
            return False
        if paper_canary_open_count <= 0:
            return False
        return total_closed_trades < int(
            self.settings.risk.paper_exploration_grace_closed_trades
        )

    def _realized_edge_positive(
        self,
        *,
        recent_closed_trades: int,
        recent_expectancy_pct: float,
        recent_profit_factor: float,
        recent_max_drawdown_pct: float,
    ) -> bool:
        min_trades = max(3, int(self.settings.risk.portfolio_heat_min_recent_trades))
        if recent_closed_trades < min_trades:
            return False
        if recent_expectancy_pct <= 0.0:
            return False
        if 0.0 < recent_profit_factor < 1.05:
            return False
        return recent_max_drawdown_pct <= max(5.0, self.settings.risk.live_max_drawdown_pct)

    def _realized_edge_negative(
        self,
        *,
        recent_closed_trades: int,
        recent_expectancy_pct: float,
        recent_profit_factor: float,
        recent_max_drawdown_pct: float,
    ) -> bool:
        min_trades = max(3, int(self.settings.risk.portfolio_heat_min_recent_trades))
        if recent_closed_trades < min_trades:
            return False
        if recent_expectancy_pct < -0.05:
            return True
        if 0.0 < recent_profit_factor < 0.95:
            return True
        return recent_max_drawdown_pct > max(5.0, self.settings.risk.live_max_drawdown_pct)

    def _compute_prediction_accuracy(self, conn, rows) -> dict[str, float]:
        timeframe = self.settings.strategy.primary_timeframe
        xgboost_correct = 0
        llm_correct = 0
        llm_total = 0
        fusion_correct = 0
        fusion_total = 0
        evaluated = 0

        for row in rows:
            timestamp = datetime.fromisoformat(row["timestamp"])
            try:
                decision = json.loads(row["decision_json"])
            except Exception:
                decision = {}
            outcome = self._simulate_prediction_outcome(
                conn,
                symbol=row["symbol"],
                timestamp=timestamp,
                timeframe=timeframe,
                decision=decision,
            )
            if outcome is None:
                continue
            evaluated += 1
            actual_up = bool(outcome["actual_up"])
            xgboost_threshold = float(
                decision.get(
                    "xgboost_threshold",
                    self.settings.model.xgboost_probability_threshold,
                )
            )
            predicted_up = float(row["up_probability"]) >= xgboost_threshold
            if predicted_up == actual_up:
                xgboost_correct += 1

            research = json.loads(row["research_json"])
            llm_action = research.get("suggested_action", "HOLD")
            if llm_action in {"OPEN_LONG", "CLOSE"}:
                llm_total += 1
                llm_up = llm_action == "OPEN_LONG"
                if llm_up == actual_up:
                    llm_correct += 1

            final_score = float(decision.get("final_score", 0.0))
            if final_score > 0:
                fusion_total += 1
                final_threshold = float(
                    decision.get(
                        "final_score_threshold",
                        self.settings.model.final_score_threshold,
                    )
                )
                fusion_up = final_score >= final_threshold
                if fusion_up == actual_up:
                    fusion_correct += 1

        return {
            "count": evaluated,
            "xgboost_accuracy_pct": (xgboost_correct / evaluated * 100) if evaluated else 0.0,
            "llm_accuracy_pct": (llm_correct / llm_total * 100) if llm_total else 0.0,
            "fusion_accuracy_pct": (fusion_correct / fusion_total * 100) if fusion_total else 0.0,
            "fusion_total": fusion_total,
        }

    @staticmethod
    def _compute_prediction_accuracy_from_evaluated_rows(
        rows: list[dict[str, object]]
    ) -> dict[str, float | int]:
        evaluated = len(rows)
        xgboost_correct = sum(1 for row in rows if bool(row.get("xgboost_correct")))
        llm_rows = [row for row in rows if row.get("llm_correct") is not None]
        llm_correct = sum(1 for row in llm_rows if bool(row.get("llm_correct")))
        fusion_rows = [row for row in rows if row.get("fusion_correct") is not None]
        fusion_correct = sum(
            1 for row in fusion_rows if bool(row.get("fusion_correct"))
        )
        return {
            "count": evaluated,
            "xgboost_accuracy_pct": (
                xgboost_correct / evaluated * 100 if evaluated else 0.0
            ),
            "llm_accuracy_pct": (
                llm_correct / len(llm_rows) * 100 if llm_rows else 0.0
            ),
            "fusion_accuracy_pct": (
                fusion_correct / len(fusion_rows) * 100 if fusion_rows else 0.0
            ),
            "fusion_total": len(fusion_rows),
        }

    @staticmethod
    def _load_prediction_rows(conn, limit: int) -> list:
        rows = conn.execute(
            "SELECT id, symbol, timestamp, up_probability, research_json, decision_json "
            "FROM prediction_runs ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return PerformanceReporter._dedupe_prediction_rows(rows)

    def _load_recent_evaluated_prediction_rows(
        self,
        conn,
        *,
        target_count: int,
        allowed_symbols: tuple[str, ...] | list[str] | None = None,
    ) -> list[dict[str, object]]:
        if target_count <= 0:
            return []
        evaluated_rows: list[dict[str, object]] = []
        seen: set[tuple[str, str, str]] = set()
        allowed = {str(symbol) for symbol in (allowed_symbols or []) if str(symbol)}
        offset = 0
        scanned = 0
        timeframe = self.settings.strategy.primary_timeframe
        while (
            len(evaluated_rows) < int(target_count)
            and scanned < self.MAX_PREDICTION_RUN_SCAN_ROWS
        ):
            batch_limit = min(
                self.PREDICTION_RUN_SCAN_BATCH_SIZE,
                self.MAX_PREDICTION_RUN_SCAN_ROWS - scanned,
            )
            rows = conn.execute(
                "SELECT id, symbol, timestamp, up_probability, research_json, decision_json, created_at "
                "FROM prediction_runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (int(batch_limit), int(offset)),
            ).fetchall()
            if not rows:
                break
            offset += len(rows)
            scanned += len(rows)
            for row in rows:
                if allowed and str(row["symbol"]) not in allowed:
                    continue
                key = (
                    str(row["symbol"]),
                    str(row["timestamp"]),
                    self._prediction_pipeline_mode(row),
                )
                if key in seen:
                    continue
                seen.add(key)
                evaluated = self._build_evaluated_prediction_row(
                    conn,
                    row=row,
                    timeframe=timeframe,
                )
                if evaluated is None:
                    continue
                evaluated_rows.append(evaluated)
                if len(evaluated_rows) >= int(target_count):
                    break
        return evaluated_rows

    @staticmethod
    def _normalize_symbols(symbols: list[str] | tuple[str, ...] | None) -> list[str]:
        normalized: list[str] = []
        for raw in list(symbols or []):
            symbol = str(raw).strip().upper().replace(" ", "")
            if not symbol:
                continue
            if "-" in symbol and "/" not in symbol:
                parts = symbol.split("-")
                if len(parts) >= 2:
                    symbol = f"{parts[0]}/{parts[1]}"
            if symbol.endswith("USDT") and "/" not in symbol and len(symbol) > 4:
                symbol = f"{symbol[:-4]}/USDT"
            if symbol not in normalized:
                normalized.append(symbol)
        return normalized

    def _evaluation_accuracy_summary(
        self,
        conn,
        *,
        evaluation_type: str,
        allowed_symbols: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, float | int]:
        allowed = tuple(self._normalize_symbols(allowed_symbols))
        params: list[object] = [evaluation_type]
        query = (
            "SELECT COUNT(*) AS count, AVG(is_correct) AS accuracy "
            "FROM prediction_evaluations WHERE evaluation_type = ?"
        )
        if allowed:
            placeholders = ",".join("?" for _ in allowed)
            query += f" AND symbol IN ({placeholders})"
            params.extend(allowed)
        row = conn.execute(query, params).fetchone()
        count = int((row["count"] or 0) if row else 0)
        accuracy = float(row["accuracy"] or 0.0) if row and row["accuracy"] is not None else 0.0
        return {
            "count": count,
            "accuracy_pct": accuracy * 100 if count > 0 else 0.0,
        }

    def _build_evaluated_prediction_row(
        self,
        conn,
        *,
        row,
        timeframe: str,
    ) -> dict[str, object] | None:
        timestamp = datetime.fromisoformat(row["timestamp"])
        try:
            decision = json.loads(row["decision_json"] or "{}")
        except Exception:
            decision = {}
        try:
            research = json.loads(row["research_json"] or "{}")
        except Exception:
            research = {}
        outcome = self._simulate_prediction_outcome(
            conn,
            symbol=row["symbol"],
            timestamp=timestamp,
            timeframe=timeframe,
            decision=decision,
        )
        if outcome is None:
            return None
        actual_up = bool(outcome["actual_up"])
        xgboost_threshold = float(
            decision.get(
                "xgboost_threshold",
                self.settings.model.xgboost_probability_threshold,
            )
        )
        predicted_up = float(row["up_probability"]) >= xgboost_threshold
        llm_action = research.get("suggested_action", "HOLD")
        llm_correct = None
        if llm_action in {"OPEN_LONG", "CLOSE"}:
            llm_correct = (llm_action == "OPEN_LONG") == actual_up
        final_score = float(decision.get("final_score", 0.0) or 0.0)
        fusion_correct = None
        if final_score > 0:
            final_threshold = float(
                decision.get(
                    "final_score_threshold",
                    self.settings.model.final_score_threshold,
                )
            )
            fusion_correct = (final_score >= final_threshold) == actual_up
        return {
            "symbol": str(row["symbol"]),
            "timestamp": str(row["timestamp"]),
            "created_at": str(row["created_at"]),
            "research": research if isinstance(research, dict) else {},
            "decision": decision if isinstance(decision, dict) else {},
            "actual_up": actual_up,
            "predicted_up": predicted_up,
            "xgboost_correct": predicted_up == actual_up,
            "llm_correct": llm_correct,
            "fusion_correct": fusion_correct,
        }

    @staticmethod
    def _dedupe_prediction_rows(rows) -> list:
        deduped = []
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            key = (
                str(row["symbol"]),
                str(row["timestamp"]),
                PerformanceReporter._prediction_pipeline_mode(row),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    @staticmethod
    def _prediction_pipeline_mode(row) -> str:
        try:
            decision = json.loads(row["decision_json"] or "{}")
        except Exception:
            decision = {}
        return str(decision.get("pipeline_mode") or "execution")

    @staticmethod
    def _symbol_variants(symbol: str) -> list[str]:
        variants = [symbol]
        if ":USDT" in symbol:
            variants.append(symbol.replace(":USDT", ""))
        else:
            variants.append(f"{symbol}:USDT")
        return list(dict.fromkeys(variants))

    def _fetch_close(
        self,
        conn,
        symbol: str,
        timeframe: str,
        before_ms: int | None = None,
        after_ms: int | None = None,
    ) -> float | None:
        variants = self._symbol_variants(symbol)
        placeholders = ",".join(["?"] * len(variants))
        params: list[object] = [*variants, timeframe]
        query = (
            f"SELECT close FROM ohlcv WHERE symbol IN ({placeholders}) "
            "AND timeframe = ?"
        )
        if before_ms is not None:
            query += " AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1"
            params.append(before_ms)
        elif after_ms is not None:
            query += " AND timestamp >= ? ORDER BY timestamp ASC LIMIT 1"
            params.append(after_ms)
        else:
            return None
        row = conn.execute(query, params).fetchone()
        return float(row["close"]) if row else None

    def _fetch_candles(
        self,
        conn,
        *,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
    ) -> list[dict]:
        variants = self._symbol_variants(symbol)
        placeholders = ",".join(["?"] * len(variants))
        rows = conn.execute(
            f"""SELECT timestamp, open, high, low, close, volume
                  FROM ohlcv
                 WHERE symbol IN ({placeholders})
                   AND timeframe = ?
                   AND timestamp > ?
                   AND timestamp <= ?
                 ORDER BY timestamp ASC""",
            [*variants, timeframe, start_ms, end_ms],
        ).fetchall()
        return [dict(row) for row in rows]

    def _simulate_prediction_outcome(
        self,
        conn,
        *,
        symbol: str,
        timestamp: datetime,
        timeframe: str,
        decision: dict,
        estimated_cost_pct: float = 0.15,
    ) -> dict[str, object] | None:
        start_ms = int(timestamp.timestamp() * 1000)
        entry_close = self._fetch_close(
            conn,
            symbol,
            timeframe,
            before_ms=start_ms,
        )
        if entry_close is None:
            return None

        horizon_hours = self._decision_horizon_hours(decision)
        end_at = timestamp + timedelta(hours=horizon_hours)
        future_candles = self._fetch_candles(
            conn,
            symbol=symbol,
            timeframe=timeframe,
            start_ms=start_ms,
            end_ms=int(end_at.timestamp() * 1000),
        )
        stop_loss_pct = float(
            decision.get(
                "stop_loss_pct",
                decision.get(
                    "fixed_stop_loss_pct",
                    self.settings.strategy.fixed_stop_loss_pct,
                ),
            )
            or 0.0
        )
        raw_take_profit_levels = decision.get(
            "take_profit_levels",
            self.settings.strategy.take_profit_levels,
        )
        take_profit_levels = (
            list(raw_take_profit_levels)
            if isinstance(raw_take_profit_levels, list)
            else list(self.settings.strategy.take_profit_levels)
        )
        outcome = simulate_long_trade(
            future_candles=future_candles,
            entry_price=float(entry_close),
            timeframe=timeframe,
            max_hold_hours=horizon_hours,
            stop_loss_pct=stop_loss_pct,
            take_profit_levels=take_profit_levels,
            round_trip_cost_pct=estimated_cost_pct,
        )
        if outcome is None:
            future_close = self._fetch_close(
                conn,
                symbol,
                timeframe,
                after_ms=int(end_at.timestamp() * 1000),
            )
            if future_close is None:
                return None
            gross_return_pct = ((future_close / entry_close) - 1.0) * 100.0
            outcome = {
                "exit_price": float(future_close),
                "exit_reason": "horizon_close",
                "gross_return_pct": gross_return_pct,
                "net_return_pct": gross_return_pct - float(estimated_cost_pct or 0.0),
                "bars_held": 0,
                "holding_hours": float(horizon_hours),
                "favorable_excursion_pct": 0.0,
                "adverse_excursion_pct": 0.0,
            }

        return {
            "entry_close": float(entry_close),
            "future_close": float(outcome["exit_price"]),
            "actual_up": float(outcome["net_return_pct"]) > 0.0,
            "metadata": {
                "estimated_cost_pct": float(estimated_cost_pct or 0.0),
                "cost_pct": float(estimated_cost_pct or 0.0),
                "opportunity_return_pct": float(outcome["gross_return_pct"]),
                "opportunity_net_return_pct": float(outcome["net_return_pct"]),
                "trade_net_return_pct": float(outcome["net_return_pct"]),
                "gross_return_pct": float(outcome["gross_return_pct"]),
                "net_return_pct": float(outcome["net_return_pct"]),
                "exit_reason": str(outcome["exit_reason"]),
                "holding_hours": float(outcome["holding_hours"]),
                "bars_held": int(outcome["bars_held"]),
                "horizon_hours": int(horizon_hours),
                "favorable_excursion_pct": float(outcome["favorable_excursion_pct"]),
                "adverse_excursion_pct": float(outcome["adverse_excursion_pct"]),
            },
        }

    def _decision_horizon_hours(self, decision: dict) -> int:
        explicit = int(decision.get("horizon_hours") or 0)
        if explicit > 0:
            return explicit
        if any(
            key in decision
            for key in ("stop_loss_pct", "fixed_stop_loss_pct", "take_profit_levels")
        ):
            return effective_trade_horizon_hours(self.settings)
        return int(self.settings.training.prediction_horizon_hours or 4)

    @staticmethod
    def render(snapshot: PerformanceSnapshot, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())

        def format_accuracy_pct(value: float, sample_count: int) -> str:
            if sample_count <= 0:
                return "N/A"
            return f"{value:.2f}% ({sample_count})"

        def format_direction_accuracy_pct(
            value: float,
            sample_count: int,
            min_samples: int,
        ) -> str:
            if sample_count < min_samples:
                return "N/A"
            return f"{value:.2f}%"

        return "\n".join(
            [
                text_for(lang, "# 性能报告", "# Performance Report"),
                text_for(lang, f"- 已平仓交易数: {snapshot.total_closed_trades}", f"- Closed Trades: {snapshot.total_closed_trades}"),
                text_for(lang, f"- 胜率: {snapshot.win_rate_pct:.2f}%", f"- Win Rate: {snapshot.win_rate_pct:.2f}%"),
                text_for(lang, f"- 累计已实现盈亏: ${snapshot.total_realized_pnl:,.2f}", f"- Total Realized PnL: ${snapshot.total_realized_pnl:,.2f}"),
                text_for(lang, f"- 平均单笔盈亏: {snapshot.avg_pnl_pct:.2f}%", f"- Average Trade PnL: {snapshot.avg_pnl_pct:.2f}%"),
                text_for(lang, f"- 累计交易成本: ${snapshot.total_trade_cost:,.2f}", f"- Total Trade Cost: ${snapshot.total_trade_cost:,.2f}"),
                text_for(lang, f"- 累计滑点拖累: ${snapshot.total_slippage_cost:,.2f}", f"- Total Slippage Drag: ${snapshot.total_slippage_cost:,.2f}"),
                text_for(lang, f"- 权益累计收益: {snapshot.equity_return_pct:.2f}%", f"- Equity Return: {snapshot.equity_return_pct:.2f}%"),
                text_for(lang, f"- 最近闭环交易数: {snapshot.recent_closed_trades}", f"- Recent Closed Trades: {snapshot.recent_closed_trades}"),
                text_for(lang, f"- 最近净期望收益: {snapshot.recent_expectancy_pct:.2f}%", f"- Recent Net Expectancy: {snapshot.recent_expectancy_pct:.2f}%"),
                text_for(lang, f"- 最近净盈亏比: {snapshot.recent_profit_factor:.2f}", f"- Recent Net Profit Factor: {snapshot.recent_profit_factor:.2f}"),
                text_for(lang, f"- 最近净回撤: {snapshot.recent_max_drawdown_pct:.2f}%", f"- Recent Net Max Drawdown: {snapshot.recent_max_drawdown_pct:.2f}%"),
                text_for(lang, f"- 最近净 Sharpe: {snapshot.recent_sharpe_like:.2f}", f"- Recent Net Sharpe: {snapshot.recent_sharpe_like:.2f}"),
                text_for(lang, f"- 最近净 Sortino: {snapshot.recent_sortino_like:.2f}", f"- Recent Net Sortino: {snapshot.recent_sortino_like:.2f}"),
                text_for(lang, f"- 最近收益波动: {snapshot.recent_return_volatility_pct:.2f}%", f"- Recent Return Volatility: {snapshot.recent_return_volatility_pct:.2f}%"),
                text_for(lang, f"- 最近亏损聚集: {snapshot.recent_loss_cluster_ratio_pct:.2f}%", f"- Recent Loss Clustering: {snapshot.recent_loss_cluster_ratio_pct:.2f}%"),
                text_for(lang, f"- 最近回撤速度: {snapshot.recent_drawdown_velocity_pct:.2f}%/trade", f"- Recent Drawdown Velocity: {snapshot.recent_drawdown_velocity_pct:.2f}%/trade"),
                text_for(lang, f"- 平均持有时长: {snapshot.avg_holding_hours:.2f}h", f"- Average Holding Hours: {snapshot.avg_holding_hours:.2f}h"),
                text_for(lang, f"- 最新权益: ${snapshot.latest_equity:,.2f}", f"- Latest Equity: ${snapshot.latest_equity:,.2f}"),
                text_for(lang, f"- 最新回撤: {snapshot.latest_drawdown_pct:.2f}%", f"- Latest Drawdown: {snapshot.latest_drawdown_pct:.2f}%"),
                text_for(lang, f"- 最近 Holdout 准确率: {snapshot.latest_holdout_accuracy:.2f}%", f"- Latest Holdout Accuracy: {snapshot.latest_holdout_accuracy:.2f}%"),
                text_for(lang, f"- 最近 Walk-Forward 收益: {snapshot.latest_walkforward_return_pct:.2f}%", f"- Latest Walk-Forward Return: {snapshot.latest_walkforward_return_pct:.2f}%"),
                text_for(lang, f"- 已评估预测数: {snapshot.prediction_eval_count}", f"- Evaluated Predictions: {snapshot.prediction_eval_count}"),
                text_for(lang, f"- 最近预测窗口样本: {snapshot.prediction_eval_count}/{snapshot.prediction_window_size}", f"- Recent Prediction Window: {snapshot.prediction_eval_count}/{snapshot.prediction_window_size}"),
                text_for(lang, f"- 扩展预测窗口样本: {snapshot.expanded_prediction_eval_count}/{snapshot.expanded_prediction_window_size}", f"- Expanded Prediction Window: {snapshot.expanded_prediction_eval_count}/{snapshot.expanded_prediction_window_size}"),
                text_for(
                    lang,
                    f"- 当前执行宇宙预测窗口样本: {snapshot.current_prediction_eval_count}/{snapshot.current_prediction_window_size}",
                    f"- Current Execution Universe Window: {snapshot.current_prediction_eval_count}/{snapshot.current_prediction_window_size}",
                ),
                text_for(lang, f"- 执行评估数: {snapshot.execution_evaluation_count}", f"- Execution Evaluations: {snapshot.execution_evaluation_count}"),
                text_for(lang, f"- Shadow 评估数: {snapshot.shadow_evaluation_count}", f"- Shadow Evaluations: {snapshot.shadow_evaluation_count}"),
                text_for(
                    lang,
                    f"- 当前执行宇宙 XGBoost 方向准确率: {format_direction_accuracy_pct(snapshot.current_xgboost_accuracy_pct, snapshot.current_prediction_eval_count, snapshot.prediction_accuracy_min_samples)}",
                    f"- Current Execution Universe XGBoost Direction Accuracy: {format_direction_accuracy_pct(snapshot.current_xgboost_accuracy_pct, snapshot.current_prediction_eval_count, snapshot.prediction_accuracy_min_samples)}",
                ),
                text_for(
                    lang,
                    f"- 当前执行宇宙 LLM 动作准确率: {format_direction_accuracy_pct(snapshot.current_llm_accuracy_pct, snapshot.current_prediction_eval_count, snapshot.prediction_accuracy_min_samples)}",
                    f"- Current Execution Universe LLM Action Accuracy: {format_direction_accuracy_pct(snapshot.current_llm_accuracy_pct, snapshot.current_prediction_eval_count, snapshot.prediction_accuracy_min_samples)}",
                ),
                text_for(
                    lang,
                    f"- 当前执行宇宙 融合信号准确率: {format_direction_accuracy_pct(snapshot.current_fusion_accuracy_pct, snapshot.current_prediction_eval_count, snapshot.prediction_accuracy_min_samples)}",
                    f"- Current Execution Universe Fusion Signal Accuracy: {format_direction_accuracy_pct(snapshot.current_fusion_accuracy_pct, snapshot.current_prediction_eval_count, snapshot.prediction_accuracy_min_samples)}",
                ),
                text_for(lang, f"- XGBoost 方向准确率: {format_direction_accuracy_pct(snapshot.xgboost_accuracy_pct, snapshot.prediction_eval_count, snapshot.prediction_accuracy_min_samples)}", f"- XGBoost Direction Accuracy: {format_direction_accuracy_pct(snapshot.xgboost_accuracy_pct, snapshot.prediction_eval_count, snapshot.prediction_accuracy_min_samples)}"),
                text_for(lang, f"- LLM 动作准确率: {format_direction_accuracy_pct(snapshot.llm_accuracy_pct, snapshot.prediction_eval_count, snapshot.prediction_accuracy_min_samples)}", f"- LLM Action Accuracy: {format_direction_accuracy_pct(snapshot.llm_accuracy_pct, snapshot.prediction_eval_count, snapshot.prediction_accuracy_min_samples)}"),
                text_for(lang, f"- 融合信号准确率: {format_direction_accuracy_pct(snapshot.fusion_accuracy_pct, snapshot.prediction_eval_count, snapshot.prediction_accuracy_min_samples)}", f"- Fusion Signal Accuracy: {format_direction_accuracy_pct(snapshot.fusion_accuracy_pct, snapshot.prediction_eval_count, snapshot.prediction_accuracy_min_samples)}"),
                text_for(lang, f"- 扩展窗口 XGBoost 方向准确率: {format_direction_accuracy_pct(snapshot.expanded_xgboost_accuracy_pct, snapshot.expanded_prediction_eval_count, snapshot.prediction_accuracy_min_samples)}", f"- Expanded XGBoost Direction Accuracy: {format_direction_accuracy_pct(snapshot.expanded_xgboost_accuracy_pct, snapshot.expanded_prediction_eval_count, snapshot.prediction_accuracy_min_samples)}"),
                text_for(lang, f"- 扩展窗口 LLM 动作准确率: {format_direction_accuracy_pct(snapshot.expanded_llm_accuracy_pct, snapshot.expanded_prediction_eval_count, snapshot.prediction_accuracy_min_samples)}", f"- Expanded LLM Action Accuracy: {format_direction_accuracy_pct(snapshot.expanded_llm_accuracy_pct, snapshot.expanded_prediction_eval_count, snapshot.prediction_accuracy_min_samples)}"),
                text_for(lang, f"- 扩展窗口 融合信号准确率: {format_direction_accuracy_pct(snapshot.expanded_fusion_accuracy_pct, snapshot.expanded_prediction_eval_count, snapshot.prediction_accuracy_min_samples)}", f"- Expanded Fusion Signal Accuracy: {format_direction_accuracy_pct(snapshot.expanded_fusion_accuracy_pct, snapshot.expanded_prediction_eval_count, snapshot.prediction_accuracy_min_samples)}"),
                text_for(
                    lang,
                    f"- 当前执行宇宙 执行闭环准确率: {format_accuracy_pct(snapshot.current_execution_accuracy_pct, snapshot.current_execution_evaluation_count)}",
                    f"- Current Execution Universe Execution Accuracy: {format_accuracy_pct(snapshot.current_execution_accuracy_pct, snapshot.current_execution_evaluation_count)}",
                ),
                text_for(lang, f"- 执行闭环准确率: {format_accuracy_pct(snapshot.execution_accuracy_pct, snapshot.execution_evaluation_count)}", f"- Execution Accuracy: {format_accuracy_pct(snapshot.execution_accuracy_pct, snapshot.execution_evaluation_count)}"),
                text_for(lang, f"- Shadow 观察准确率: {format_accuracy_pct(snapshot.shadow_accuracy_pct, snapshot.shadow_evaluation_count)}", f"- Shadow Accuracy: {format_accuracy_pct(snapshot.shadow_accuracy_pct, snapshot.shadow_evaluation_count)}"),
                text_for(lang, f"- LLM 运行时已配置: {snapshot.llm_runtime_configured}", f"- LLM Runtime Configured: {snapshot.llm_runtime_configured}"),
                text_for(lang, f"- 最近研究回退次数: {snapshot.research_fallback_count}/{snapshot.research_total_count}", f"- Recent Research Fallbacks: {snapshot.research_fallback_count}/{snapshot.research_total_count}"),
                text_for(lang, f"- 最近研究回退占比: {snapshot.research_fallback_ratio_pct:.2f}%", f"- Recent Research Fallback Ratio: {snapshot.research_fallback_ratio_pct:.2f}%"),
                text_for(lang, f"- Paper Canary 开仓数: {snapshot.paper_canary_open_count}", f"- Paper Canary Opens: {snapshot.paper_canary_open_count}"),
                text_for(lang, f"- 软审批 Canary 开仓数: {snapshot.soft_paper_canary_open_count}", f"- Soft-Review Canary Opens: {snapshot.soft_paper_canary_open_count}"),
                text_for(lang, f"- 探索期降级缓冲: {snapshot.paper_exploration_grace_active}", f"- Exploration Grace Active: {snapshot.paper_exploration_grace_active}"),
                text_for(lang, f"- 衰减状态: {snapshot.degradation_status}", f"- Degradation Status: {snapshot.degradation_status}"),
                text_for(lang, f"- 衰减原因: {snapshot.degradation_reason or 'none'}", f"- Degradation Reason: {snapshot.degradation_reason or 'none'}"),
                text_for(lang, f"- 建议 XGB 阈值: {snapshot.recommended_xgboost_threshold:.2f}", f"- Recommended XGB Threshold: {snapshot.recommended_xgboost_threshold:.2f}"),
                text_for(lang, f"- 建议最终阈值: {snapshot.recommended_final_score_threshold:.2f}", f"- Recommended Final Threshold: {snapshot.recommended_final_score_threshold:.2f}"),
            ]
        )
