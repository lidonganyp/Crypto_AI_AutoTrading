"""Compact alpha diagnostics report for CryptoAI v3."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from config import Settings, get_settings
from core.i18n import get_default_language, normalize_language, text_for
from core.storage import Storage
from monitor.performance_report import PerformanceReporter
from monitor.pool_attribution_report import PoolAttributionReporter


class AlphaDiagnosticsReporter:
    """Summarize the current alpha quality and main drag sources."""

    EXECUTION_SYMBOLS_STATE_KEY = "execution_symbols"
    LEARNING_DETAILS_STATE_KEY = "runtime_settings_learning_details"

    def __init__(self, storage: Storage, settings: Settings | None = None):
        self.storage = storage
        self.settings = settings or get_settings()
        self.performance = PerformanceReporter(storage, self.settings)
        self.pool_attribution = PoolAttributionReporter(storage)

    def build(self, symbols: list[str] | None = None) -> dict:
        selected_symbols = symbols or self.storage.get_json_state(
            self.EXECUTION_SYMBOLS_STATE_KEY,
            [],
        ) or []
        performance = self.performance.build()
        attribution = self.pool_attribution.build(selected_symbols)
        learning_details = self.storage.get_json_state(self.LEARNING_DETAILS_STATE_KEY, {}) or {}
        blocked_setups = learning_details.get("blocked_setups", []) or []
        learning_stats = learning_details.get("stats", {}) or {}
        recent_canary_counts = self._paper_canary_breakdown(days=1, symbols=selected_symbols)
        total_canary_counts = self._paper_canary_breakdown(days=None, symbols=selected_symbols)
        paper_canary_patience_stats = self._paper_canary_patience_stats(
            symbols=selected_symbols,
        )
        recent_fast_alpha_counts = self._fast_alpha_breakdown(days=1, symbols=selected_symbols)
        total_fast_alpha_counts = self._fast_alpha_breakdown(days=None, symbols=selected_symbols)
        short_exit_stats = self._short_research_exit_stats(
            attribution.get("closed_trades", []) or []
        )
        fast_alpha_trade_stats = self._fast_alpha_trade_stats(symbols=selected_symbols)
        fast_alpha_eval_stats = self._fast_alpha_eval_stats(symbols=selected_symbols)
        fast_alpha_patience_stats = self._fast_alpha_patience_stats(
            symbols=selected_symbols,
        )
        fast_alpha_review_policy_stats = self._fast_alpha_review_policy_stats(
            symbols=selected_symbols,
        )
        short_horizon_status = self._short_horizon_status_summary()
        recent_watch_count = self._recent_position_review_watch_count(
            hours=24,
            symbols=selected_symbols,
        )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "runtime_mode": self.settings.app.runtime_mode,
            "execution_symbols": list(selected_symbols),
            "performance": {
                "degradation_status": performance.degradation_status,
                "degradation_reason": performance.degradation_reason,
                "recent_expectancy_pct": performance.recent_expectancy_pct,
                "recent_profit_factor": performance.recent_profit_factor,
                "avg_holding_hours": performance.avg_holding_hours,
                "paper_canary_open_count": performance.paper_canary_open_count,
                "soft_paper_canary_open_count": performance.soft_paper_canary_open_count,
                "total_realized_pnl": performance.total_realized_pnl,
            },
            "paper_canary_mix": {
                "total": total_canary_counts,
                "recent_24h": recent_canary_counts,
                "soft_share_pct": self._share_pct(
                    total_canary_counts.get("soft_review", 0),
                    total_canary_counts.get("total", 0),
                ),
                "recent_soft_share_pct": self._share_pct(
                    recent_canary_counts.get("soft_review", 0),
                    recent_canary_counts.get("total", 0),
                ),
                "strong_open_trade_count": paper_canary_patience_stats.get(
                    "strong_open_trade_count",
                    0,
                ),
                "early_research_de_risk_trade_count": paper_canary_patience_stats.get(
                    "early_research_de_risk_trade_count",
                    0,
                ),
                "early_research_exit_trade_count": paper_canary_patience_stats.get(
                    "early_research_exit_trade_count",
                    0,
                ),
                "early_research_exit_ratio_pct": paper_canary_patience_stats.get(
                    "early_research_exit_ratio_pct",
                    0.0,
                ),
                "avg_first_research_exit_hours": paper_canary_patience_stats.get(
                    "avg_first_research_exit_hours",
                    0.0,
                ),
            },
            "fast_alpha": {
                "total_opens": total_fast_alpha_counts.get("total", 0),
                "recent_24h_opens": recent_fast_alpha_counts.get("total", 0),
                "closed_trade_count": fast_alpha_trade_stats.get("closed_trade_count", 0),
                "win_rate_pct": fast_alpha_trade_stats.get("win_rate_pct", 0.0),
                "total_net_pnl": fast_alpha_trade_stats.get("total_net_pnl", 0.0),
                "avg_net_return_pct": fast_alpha_trade_stats.get("avg_net_return_pct", 0.0),
                "execution_eval_count": fast_alpha_eval_stats.get("execution_eval_count", 0),
                "execution_accuracy_pct": fast_alpha_eval_stats.get("execution_accuracy_pct", 0.0),
                "strong_open_trade_count": fast_alpha_patience_stats.get(
                    "strong_open_trade_count",
                    0,
                ),
                "early_research_de_risk_trade_count": fast_alpha_patience_stats.get(
                    "early_research_de_risk_trade_count",
                    0,
                ),
                "early_research_exit_trade_count": fast_alpha_patience_stats.get(
                    "early_research_exit_trade_count",
                    0,
                ),
                "early_research_exit_ratio_pct": fast_alpha_patience_stats.get(
                    "early_research_exit_ratio_pct",
                    0.0,
                ),
                "avg_first_research_exit_hours": fast_alpha_patience_stats.get(
                    "avg_first_research_exit_hours",
                    0.0,
                ),
                "short_horizon_status": short_horizon_status,
                "short_horizon_policy": fast_alpha_review_policy_stats,
            },
            "attribution": {
                "closed_trade_count": attribution.get("closed_trade_count", 0),
                "total_net_pnl": attribution.get("total_net_pnl", 0.0),
                "avg_net_return_pct": attribution.get("avg_net_return_pct", 0.0),
                "symbol_summary": attribution.get("symbol_summary", []),
                "short_research_exit": short_exit_stats,
            },
            "learning": {
                "blocked_setup_count": len(blocked_setups),
                "recent_negative_setup_pause_count": int(
                    learning_stats.get("recent_negative_setup_pause_count", 0) or 0
                ),
                "shadow_rehabilitated_setup_count": int(
                    learning_stats.get("shadow_rehabilitated_setup_count", 0) or 0
                ),
                "blocked_setups": blocked_setups[:5],
                "reasons": list(learning_details.get("reasons", []) or []),
            },
            "review_watch": {
                "recent_24h_count": recent_watch_count,
            },
        }

    def _paper_canary_breakdown(
        self,
        days: int | None,
        symbols: list[str] | None = None,
    ) -> dict[str, int]:
        query = (
            "SELECT json_extract(payload_json, '$.canary_mode') AS canary_mode "
            "FROM execution_events WHERE event_type='paper_canary_open'"
        )
        params: list[str] = []
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            query += f" AND symbol IN ({placeholders})"
            params.extend(list(symbols))
        if days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            query += " AND created_at >= ?"
            params.append(cutoff)
        counts = {"total": 0, "primary_review": 0, "offensive_review": 0, "soft_review": 0}
        with self.storage._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        for row in rows:
            counts["total"] += 1
            mode = str(row["canary_mode"] or "")
            if mode in counts:
                counts[mode] += 1
        return counts

    def _fast_alpha_breakdown(
        self,
        days: int | None,
        symbols: list[str] | None = None,
    ) -> dict[str, int]:
        query = "SELECT COUNT(*) AS c FROM execution_events WHERE event_type='fast_alpha_open'"
        params: list[str] = []
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            query += f" AND symbol IN ({placeholders})"
            params.extend(list(symbols))
        if days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            query += " AND created_at >= ?"
            params.append(cutoff)
        with self.storage._conn() as conn:
            row = conn.execute(query, params).fetchone()
        return {"total": int((row["c"] or 0) if row else 0)}

    def _paper_canary_patience_stats(
        self,
        *,
        symbols: list[str] | None = None,
    ) -> dict[str, float | int]:
        open_query = (
            "SELECT trade_id FROM pnl_ledger "
            "WHERE event_type='open' "
            "AND json_extract(metadata_json, '$.pipeline_mode')='paper_canary' "
            "AND upper(json_extract(metadata_json, '$.reviewed_action'))='OPEN_LONG' "
            "AND json_extract(metadata_json, '$.paper_canary_mode') IN ('primary_review','offensive_review') "
            "AND CAST(json_extract(metadata_json, '$.review_score') AS REAL) >= 0.15"
        )
        close_query = (
            "SELECT trade_id, holding_hours, json_extract(metadata_json, '$.reason') AS reason "
            "FROM pnl_ledger "
            "WHERE event_type='close' "
            "AND json_extract(metadata_json, '$.pipeline_mode')='paper_canary'"
        )
        open_params: list[str] = []
        close_params: list[str] = []
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            open_query += f" AND symbol IN ({placeholders})"
            close_query += f" AND symbol IN ({placeholders})"
            open_params.extend(list(symbols))
            close_params.extend(list(symbols))

        with self.storage._conn() as conn:
            open_rows = conn.execute(open_query, open_params).fetchall()
            close_rows = conn.execute(close_query, close_params).fetchall()

        strong_open_trade_ids = {
            str(row["trade_id"] or "").strip()
            for row in open_rows
            if str(row["trade_id"] or "").strip()
        }
        if not strong_open_trade_ids:
            return {
                "strong_open_trade_count": 0,
                "early_research_de_risk_trade_count": 0,
                "early_research_exit_trade_count": 0,
                "early_research_exit_ratio_pct": 0.0,
                "avg_first_research_exit_hours": 0.0,
            }

        early_de_risk_trade_ids: set[str] = set()
        early_exit_trade_ids: set[str] = set()
        first_exit_hours: dict[str, float] = {}
        for row in close_rows:
            trade_id = str(row["trade_id"] or "").strip()
            if trade_id not in strong_open_trade_ids:
                continue
            holding_hours = float(row["holding_hours"] or 0.0)
            reason = str(row["reason"] or "")
            if "research_de_risk" in reason and holding_hours <= 1.0:
                early_de_risk_trade_ids.add(trade_id)
            if "research_exit" in reason and holding_hours <= 1.0:
                early_exit_trade_ids.add(trade_id)
            if "research_de_risk" in reason or "research_exit" in reason:
                previous = first_exit_hours.get(trade_id)
                if previous is None or holding_hours < previous:
                    first_exit_hours[trade_id] = holding_hours

        avg_first_exit_hours = (
            sum(first_exit_hours.values()) / len(first_exit_hours)
            if first_exit_hours
            else 0.0
        )
        return {
            "strong_open_trade_count": len(strong_open_trade_ids),
            "early_research_de_risk_trade_count": len(early_de_risk_trade_ids),
            "early_research_exit_trade_count": len(early_exit_trade_ids),
            "early_research_exit_ratio_pct": self._share_pct(
                len(early_exit_trade_ids),
                len(strong_open_trade_ids),
            ),
            "avg_first_research_exit_hours": avg_first_exit_hours,
        }

    def _fast_alpha_trade_stats(
        self,
        *,
        symbols: list[str] | None = None,
    ) -> dict[str, float | int]:
        query = (
            "SELECT pnl, pnl_pct FROM trades "
            "WHERE status='closed' "
            "AND json_extract(metadata_json, '$.pipeline_mode')='fast_alpha'"
        )
        params: list[str] = []
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            query += f" AND symbol IN ({placeholders})"
            params.extend(list(symbols))
        with self.storage._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        pnls = [float(row["pnl"] or 0.0) for row in rows]
        returns = [float(row["pnl_pct"] or 0.0) for row in rows]
        win_rate_pct = self._share_pct(
            sum(1 for pnl in pnls if pnl > 0),
            len(pnls),
        )
        return {
            "closed_trade_count": len(pnls),
            "total_net_pnl": sum(pnls),
            "avg_net_return_pct": (sum(returns) / len(returns) if returns else 0.0),
            "win_rate_pct": win_rate_pct,
        }

    def _fast_alpha_eval_stats(
        self,
        *,
        symbols: list[str] | None = None,
    ) -> dict[str, float | int]:
        query = (
            "SELECT COUNT(*) AS c, AVG(pe.is_correct) AS accuracy "
            "FROM prediction_evaluations pe "
            "JOIN prediction_runs pr "
            "ON pr.symbol = pe.symbol AND pr.timestamp = pe.timestamp "
            "WHERE pe.evaluation_type='execution' "
            "AND json_extract(pr.decision_json, '$.pipeline_mode')='fast_alpha'"
        )
        params: list[str] = []
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            query += f" AND pe.symbol IN ({placeholders})"
            params.extend(list(symbols))
        with self.storage._conn() as conn:
            row = conn.execute(query, params).fetchone()
        count = int((row["c"] or 0) if row else 0)
        accuracy = float(row["accuracy"] or 0.0) if row and row["accuracy"] is not None else 0.0
        return {
            "execution_eval_count": count,
            "execution_accuracy_pct": accuracy * 100 if count > 0 else 0.0,
        }

    def _fast_alpha_patience_stats(
        self,
        *,
        symbols: list[str] | None = None,
    ) -> dict[str, float | int]:
        open_query = (
            "SELECT trade_id FROM pnl_ledger "
            "WHERE event_type='open' "
            "AND json_extract(metadata_json, '$.pipeline_mode')='fast_alpha' "
            "AND upper(json_extract(metadata_json, '$.reviewed_action'))='OPEN_LONG' "
            "AND CAST(json_extract(metadata_json, '$.review_score') AS REAL) >= 0.25"
        )
        close_query = (
            "SELECT trade_id, holding_hours, json_extract(metadata_json, '$.reason') AS reason "
            "FROM pnl_ledger "
            "WHERE event_type='close' "
            "AND json_extract(metadata_json, '$.pipeline_mode')='fast_alpha'"
        )
        open_params: list[str] = []
        close_params: list[str] = []
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            open_query += f" AND symbol IN ({placeholders})"
            close_query += f" AND symbol IN ({placeholders})"
            open_params.extend(list(symbols))
            close_params.extend(list(symbols))

        with self.storage._conn() as conn:
            open_rows = conn.execute(open_query, open_params).fetchall()
            close_rows = conn.execute(close_query, close_params).fetchall()

        strong_open_trade_ids = {
            str(row["trade_id"] or "").strip()
            for row in open_rows
            if str(row["trade_id"] or "").strip()
        }
        if not strong_open_trade_ids:
            return {
                "strong_open_trade_count": 0,
                "early_research_de_risk_trade_count": 0,
                "early_research_exit_trade_count": 0,
                "early_research_exit_ratio_pct": 0.0,
                "avg_first_research_exit_hours": 0.0,
            }

        early_de_risk_trade_ids: set[str] = set()
        early_exit_trade_ids: set[str] = set()
        first_exit_hours: dict[str, float] = {}
        for row in close_rows:
            trade_id = str(row["trade_id"] or "").strip()
            if trade_id not in strong_open_trade_ids:
                continue
            holding_hours = float(row["holding_hours"] or 0.0)
            reason = str(row["reason"] or "")
            if "research_de_risk" in reason and holding_hours <= 1.0:
                early_de_risk_trade_ids.add(trade_id)
            if "research_exit" in reason and holding_hours <= 1.0:
                early_exit_trade_ids.add(trade_id)
            if "research_de_risk" in reason or "research_exit" in reason:
                previous = first_exit_hours.get(trade_id)
                if previous is None or holding_hours < previous:
                    first_exit_hours[trade_id] = holding_hours

        avg_first_exit_hours = (
            sum(first_exit_hours.values()) / len(first_exit_hours)
            if first_exit_hours
            else 0.0
        )
        return {
            "strong_open_trade_count": len(strong_open_trade_ids),
            "early_research_de_risk_trade_count": len(early_de_risk_trade_ids),
            "early_research_exit_trade_count": len(early_exit_trade_ids),
            "early_research_exit_ratio_pct": self._share_pct(
                len(early_exit_trade_ids),
                len(strong_open_trade_ids),
            ),
            "avg_first_research_exit_hours": avg_first_exit_hours,
        }

    def _fast_alpha_review_policy_stats(
        self,
        *,
        symbols: list[str] | None = None,
    ) -> dict[str, float | int]:
        open_query = (
            "SELECT payload_json, created_at FROM execution_events "
            "WHERE event_type='fast_alpha_open'"
        )
        block_query = (
            "SELECT COUNT(*) AS c FROM execution_events "
            "WHERE event_type='fast_alpha_blocked' "
            "AND json_extract(payload_json, '$.reason')='short_horizon_negative_expectancy_pause'"
        )
        trade_query = (
            "SELECT pnl, pnl_pct FROM trades "
            "WHERE status='closed' "
            "AND json_extract(metadata_json, '$.pipeline_mode')='fast_alpha' "
            "AND COALESCE(json_extract(metadata_json, '$.fast_alpha_review_policy_reason'), '') != ''"
        )
        open_params: list[str] = []
        block_params: list[str] = []
        trade_params: list[str] = []
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            open_query += f" AND symbol IN ({placeholders})"
            block_query += f" AND symbol IN ({placeholders})"
            trade_query += f" AND symbol IN ({placeholders})"
            open_params.extend(list(symbols))
            block_params.extend(list(symbols))
            trade_params.extend(list(symbols))

        recent_cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with self.storage._conn() as conn:
            open_rows = conn.execute(open_query, open_params).fetchall()
            block_total_row = conn.execute(block_query, block_params).fetchone()
            block_recent_row = conn.execute(
                block_query + " AND created_at >= ?",
                [*block_params, recent_cutoff],
            ).fetchone()
            trade_rows = conn.execute(trade_query, trade_params).fetchall()

        softened_total = 0
        softened_recent = 0
        positive_edge_softened = 0
        warming_up_softened = 0
        for row in open_rows:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except Exception:
                payload = {}
            review_policy_reason = str(payload.get("review_policy_reason") or "")
            if not review_policy_reason:
                continue
            softened_total += 1
            if str(row["created_at"] or "") >= recent_cutoff:
                softened_recent += 1
            if review_policy_reason.startswith("positive_edge_soften"):
                positive_edge_softened += 1
            elif review_policy_reason.startswith("warming_up_soften"):
                warming_up_softened += 1

        softened_pnls = [float(row["pnl"] or 0.0) for row in trade_rows]
        softened_returns = [float(row["pnl_pct"] or 0.0) for row in trade_rows]
        softened_closed_trade_count = len(trade_rows)
        softened_avg_net_return_pct = (
            sum(softened_returns) / len(softened_returns)
            if softened_returns
            else 0.0
        )
        softened_profit_factor = self.performance._profit_factor(softened_pnls)
        softened_max_drawdown_pct = self.performance._returns_max_drawdown_pct(
            softened_returns
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
        if softened_closed_trade_count < min_closed_trades:
            softened_status = "insufficient_samples"
        elif (
            softened_avg_net_return_pct
            <= float(
                getattr(
                    self.settings.strategy,
                    "fast_alpha_short_horizon_trade_feedback_negative_expectancy_pct",
                    -0.05,
                )
                or -0.05
            )
            or (
                0.0 < softened_profit_factor
                < float(
                    getattr(
                        self.settings.strategy,
                        "fast_alpha_short_horizon_trade_feedback_negative_profit_factor",
                        0.95,
                    )
                    or 0.95
                )
            )
            or softened_max_drawdown_pct
            > float(
                getattr(
                    self.settings.strategy,
                    "fast_alpha_short_horizon_trade_feedback_negative_max_drawdown_pct",
                    3.0,
                )
                or 3.0
            )
        ):
            softened_status = "disabled_negative_edge"
        else:
            softened_status = "healthy"
        return {
            "softened_open_count": softened_total,
            "recent_24h_softened_open_count": softened_recent,
            "positive_edge_softened_open_count": positive_edge_softened,
            "warming_up_softened_open_count": warming_up_softened,
            "softened_closed_trade_count": softened_closed_trade_count,
            "softened_total_net_pnl": sum(softened_pnls),
            "softened_avg_net_return_pct": softened_avg_net_return_pct,
            "softened_profit_factor": softened_profit_factor,
            "softened_max_drawdown_pct": softened_max_drawdown_pct,
            "softened_status": softened_status,
            "softened_min_closed_trades": min_closed_trades,
            "negative_expectancy_pause_count": int(
                (block_total_row["c"] or 0) if block_total_row else 0
            ),
            "recent_24h_negative_expectancy_pause_count": int(
                (block_recent_row["c"] or 0) if block_recent_row else 0
            ),
        }

    def _short_horizon_status_summary(self) -> dict[str, float | int | str]:
        lookback = max(
            1,
            int(
                getattr(
                    self.settings.strategy,
                    "short_horizon_adaptive_lookback_trades",
                    20,
                )
                or 20
            ),
        )
        summary = self.performance.build_pipeline_mode_summary(
            ["fast_alpha", "paper_canary"],
            limit=lookback,
        ).get("_combined", {})
        closed_trade_count = int(summary.get("closed_trade_count", 0) or 0)
        expectancy_pct = float(summary.get("expectancy_pct", 0.0) or 0.0)
        profit_factor = float(summary.get("profit_factor", 0.0) or 0.0)
        max_drawdown_pct = float(summary.get("max_drawdown_pct", 0.0) or 0.0)
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
        negative_edge = (
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
        if closed_trade_count < min_closed_trades:
            status = "warming_up"
            reason = "insufficient_samples"
        elif positive_edge:
            status = "positive_edge"
            reason = "positive_edge_expand"
        elif negative_edge:
            status = "negative_edge_pause"
            reason = "negative_edge_pause"
        else:
            status = "neutral"
            reason = "mixed_edge"
        return {
            "status": status,
            "reason": reason,
            "closed_trade_count": closed_trade_count,
            "min_closed_trades": min_closed_trades,
            "lookback_trades": lookback,
            "expectancy_pct": expectancy_pct,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown_pct,
        }

    def _recent_position_review_watch_count(
        self,
        hours: int = 24,
        symbols: list[str] | None = None,
    ) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        query = (
            "SELECT COUNT(*) AS c FROM execution_events "
            "WHERE event_type='position_review_watch' AND created_at >= ?"
        )
        params: list[str] = [cutoff]
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            query += f" AND symbol IN ({placeholders})"
            params.extend(list(symbols))
        with self.storage._conn() as conn:
            row = conn.execute(query, params).fetchone()
        return int((row["c"] or 0) if row else 0)

    @staticmethod
    def _short_research_exit_stats(closed_trades: list[dict]) -> dict[str, float | int]:
        short_trades = [
            trade
            for trade in closed_trades
            if float(trade.get("holding_hours") or 0.0) <= 0.5
            and "research_exit" in (trade.get("exit_reasons") or [])
        ]
        loss_trades = [
            trade
            for trade in short_trades
            if float(trade.get("net_return_pct") or 0.0) < 0.0
        ]
        return {
            "count": len(short_trades),
            "loss_count": len(loss_trades),
            "loss_ratio_pct": AlphaDiagnosticsReporter._share_pct(
                len(loss_trades),
                len(short_trades),
            ),
            "avg_net_return_pct": (
                sum(float(trade.get("net_return_pct") or 0.0) for trade in short_trades)
                / len(short_trades)
                if short_trades
                else 0.0
            ),
        }

    @staticmethod
    def _share_pct(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator * 100.0

    @staticmethod
    def render(data: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        perf = data.get("performance", {}) or {}
        mix = data.get("paper_canary_mix", {}) or {}
        fast_alpha = data.get("fast_alpha", {}) or {}
        fast_alpha_status = fast_alpha.get("short_horizon_status", {}) or {}
        fast_alpha_policy = fast_alpha.get("short_horizon_policy", {}) or {}
        attr = data.get("attribution", {}) or {}
        short_exit = attr.get("short_research_exit", {}) or {}
        learning = data.get("learning", {}) or {}
        symbol_summary = attr.get("symbol_summary", []) or []
        lines = [
            text_for(lang, "# Alpha 诊断日报", "# Alpha Diagnostics"),
            text_for(
                lang,
                f"- 执行池: {', '.join(data.get('execution_symbols', []) or []) or '无'}",
                f"- Execution Pool: {', '.join(data.get('execution_symbols', []) or []) or 'none'}",
            ),
            text_for(
                lang,
                f"- 衰减状态: {perf.get('degradation_status', 'unknown')} ({perf.get('degradation_reason', '')})",
                f"- Degradation: {perf.get('degradation_status', 'unknown')} ({perf.get('degradation_reason', '')})",
            ),
            text_for(
                lang,
                f"- 最近净期望: {float(perf.get('recent_expectancy_pct') or 0.0):+.2f}%",
                f"- Recent Expectancy: {float(perf.get('recent_expectancy_pct') or 0.0):+.2f}%",
            ),
            text_for(
                lang,
                f"- 最近净盈亏比: {float(perf.get('recent_profit_factor') or 0.0):.2f}",
                f"- Recent Profit Factor: {float(perf.get('recent_profit_factor') or 0.0):.2f}",
            ),
            text_for(
                lang,
                f"- 平均持有时长: {float(perf.get('avg_holding_hours') or 0.0):.2f}h",
                f"- Avg Holding Time: {float(perf.get('avg_holding_hours') or 0.0):.2f}h",
            ),
            text_for(
                lang,
                f"- 累计已实现盈亏: ${float(perf.get('total_realized_pnl') or 0.0):+.2f}",
                f"- Total Realized PnL: ${float(perf.get('total_realized_pnl') or 0.0):+.2f}",
            ),
            "",
            text_for(lang, "## Canary 结构", "## Canary Mix"),
            text_for(
                lang,
                (
                    f"- 全量: primary={mix.get('total', {}).get('primary_review', 0)}, "
                    f"offensive={mix.get('total', {}).get('offensive_review', 0)}, "
                    f"soft={mix.get('total', {}).get('soft_review', 0)}, "
                    f"soft占比={float(mix.get('soft_share_pct') or 0.0):.2f}%"
                ),
                (
                    f"- Total: primary={mix.get('total', {}).get('primary_review', 0)}, "
                    f"offensive={mix.get('total', {}).get('offensive_review', 0)}, "
                    f"soft={mix.get('total', {}).get('soft_review', 0)}, "
                    f"soft_share={float(mix.get('soft_share_pct') or 0.0):.2f}%"
                ),
            ),
            text_for(
                lang,
                (
                    f"- 最近24h: primary={mix.get('recent_24h', {}).get('primary_review', 0)}, "
                    f"offensive={mix.get('recent_24h', {}).get('offensive_review', 0)}, "
                    f"soft={mix.get('recent_24h', {}).get('soft_review', 0)}, "
                    f"soft占比={float(mix.get('recent_soft_share_pct') or 0.0):.2f}%"
                ),
                (
                    f"- Last 24h: primary={mix.get('recent_24h', {}).get('primary_review', 0)}, "
                    f"offensive={mix.get('recent_24h', {}).get('offensive_review', 0)}, "
                    f"soft={mix.get('recent_24h', {}).get('soft_review', 0)}, "
                    f"soft_share={float(mix.get('recent_soft_share_pct') or 0.0):.2f}%"
                ),
            ),
            text_for(
                lang,
                (
                    f"- Canary 强质量开仓数: "
                    f"{int(mix.get('strong_open_trade_count') or 0)}"
                ),
                (
                    f"- Canary Strong-Open Trades: "
                    f"{int(mix.get('strong_open_trade_count') or 0)}"
                ),
            ),
            text_for(
                lang,
                (
                    f"- Canary 1h内 research_de_risk 交易数: "
                    f"{int(mix.get('early_research_de_risk_trade_count') or 0)}"
                ),
                (
                    f"- Canary <=1h research_de_risk trades: "
                    f"{int(mix.get('early_research_de_risk_trade_count') or 0)}"
                ),
            ),
            text_for(
                lang,
                (
                    f"- Canary 1h内 research_exit 交易占比: "
                    f"{float(mix.get('early_research_exit_ratio_pct') or 0.0):.2f}% "
                    f"({int(mix.get('early_research_exit_trade_count') or 0)} 笔)"
                ),
                (
                    f"- Canary <=1h research_exit trade ratio: "
                    f"{float(mix.get('early_research_exit_ratio_pct') or 0.0):.2f}% "
                    f"({int(mix.get('early_research_exit_trade_count') or 0)} trades)"
                ),
            ),
            text_for(
                lang,
                (
                    f"- Canary 首次 research exit 平均时长: "
                    f"{float(mix.get('avg_first_research_exit_hours') or 0.0):.2f}h"
                ),
                (
                    f"- Canary Avg first research exit time: "
                    f"{float(mix.get('avg_first_research_exit_hours') or 0.0):.2f}h"
                ),
            ),
            "",
            text_for(lang, "## Fast Alpha", "## Fast Alpha"),
            text_for(
                lang,
                f"- Fast Alpha 总开仓数: {int(fast_alpha.get('total_opens') or 0)}",
                f"- Fast Alpha Total Opens: {int(fast_alpha.get('total_opens') or 0)}",
            ),
            text_for(
                lang,
                f"- Fast Alpha 最近24h开仓数: {int(fast_alpha.get('recent_24h_opens') or 0)}",
                f"- Fast Alpha Last 24h Opens: {int(fast_alpha.get('recent_24h_opens') or 0)}",
            ),
            text_for(
                lang,
                f"- Fast Alpha 已平仓数: {int(fast_alpha.get('closed_trade_count') or 0)}",
                f"- Fast Alpha Closed Trades: {int(fast_alpha.get('closed_trade_count') or 0)}",
            ),
            text_for(
                lang,
                f"- Fast Alpha 胜率: {float(fast_alpha.get('win_rate_pct') or 0.0):.2f}%",
                f"- Fast Alpha Win Rate: {float(fast_alpha.get('win_rate_pct') or 0.0):.2f}%",
            ),
            text_for(
                lang,
                f"- Fast Alpha 净收益: ${float(fast_alpha.get('total_net_pnl') or 0.0):+.2f}",
                f"- Fast Alpha Net PnL: ${float(fast_alpha.get('total_net_pnl') or 0.0):+.2f}",
            ),
            text_for(
                lang,
                f"- Fast Alpha 平均净收益: {float(fast_alpha.get('avg_net_return_pct') or 0.0):+.2f}%",
                f"- Fast Alpha Avg Net Return: {float(fast_alpha.get('avg_net_return_pct') or 0.0):+.2f}%",
            ),
            text_for(
                lang,
                (
                    f"- Fast Alpha 执行准确率: "
                    f"{float(fast_alpha.get('execution_accuracy_pct') or 0.0):.2f}% "
                    f"({int(fast_alpha.get('execution_eval_count') or 0)})"
                ),
                (
                    f"- Fast Alpha Execution Accuracy: "
                    f"{float(fast_alpha.get('execution_accuracy_pct') or 0.0):.2f}% "
                    f"({int(fast_alpha.get('execution_eval_count') or 0)})"
                ),
            ),
            text_for(
                lang,
                (
                    f"- Fast Alpha 强质量开仓数: "
                    f"{int(fast_alpha.get('strong_open_trade_count') or 0)}"
                ),
                (
                    f"- Fast Alpha Strong-Open Trades: "
                    f"{int(fast_alpha.get('strong_open_trade_count') or 0)}"
                ),
            ),
            text_for(
                lang,
                (
                    f"- 1h内 research_de_risk 交易数: "
                    f"{int(fast_alpha.get('early_research_de_risk_trade_count') or 0)}"
                ),
                (
                    f"- <=1h research_de_risk trades: "
                    f"{int(fast_alpha.get('early_research_de_risk_trade_count') or 0)}"
                ),
            ),
            text_for(
                lang,
                (
                    f"- 1h内 research_exit 交易占比: "
                    f"{float(fast_alpha.get('early_research_exit_ratio_pct') or 0.0):.2f}% "
                    f"({int(fast_alpha.get('early_research_exit_trade_count') or 0)} 笔)"
                ),
                (
                    f"- <=1h research_exit trade ratio: "
                    f"{float(fast_alpha.get('early_research_exit_ratio_pct') or 0.0):.2f}% "
                    f"({int(fast_alpha.get('early_research_exit_trade_count') or 0)} trades)"
                ),
            ),
            text_for(
                lang,
                (
                    f"- 首次 research exit 平均时长: "
                    f"{float(fast_alpha.get('avg_first_research_exit_hours') or 0.0):.2f}h"
                ),
                (
                    f"- Avg first research exit time: "
                    f"{float(fast_alpha.get('avg_first_research_exit_hours') or 0.0):.2f}h"
                ),
            ),
            text_for(
                lang,
                f"- Short-horizon 当前状态: {fast_alpha_status.get('status', 'unknown')}",
                f"- Short-horizon Status: {fast_alpha_status.get('status', 'unknown')}",
            ),
            text_for(
                lang,
                (
                    f"- Short-horizon 最近样本: "
                    f"{int(fast_alpha_status.get('closed_trade_count') or 0)}/"
                    f"{int(fast_alpha_status.get('min_closed_trades') or 0)} "
                    f"(lookback {int(fast_alpha_status.get('lookback_trades') or 0)})"
                ),
                (
                    f"- Short-horizon Recent Samples: "
                    f"{int(fast_alpha_status.get('closed_trade_count') or 0)}/"
                    f"{int(fast_alpha_status.get('min_closed_trades') or 0)} "
                    f"(lookback {int(fast_alpha_status.get('lookback_trades') or 0)})"
                ),
            ),
            text_for(
                lang,
                f"- Short-horizon 最近净期望: {float(fast_alpha_status.get('expectancy_pct') or 0.0):+.2f}%",
                f"- Short-horizon Recent Expectancy: {float(fast_alpha_status.get('expectancy_pct') or 0.0):+.2f}%",
            ),
            text_for(
                lang,
                f"- Short-horizon 最近净盈亏比: {float(fast_alpha_status.get('profit_factor') or 0.0):.2f}",
                f"- Short-horizon Recent Profit Factor: {float(fast_alpha_status.get('profit_factor') or 0.0):.2f}",
            ),
            text_for(
                lang,
                (
                    f"- Short-horizon 放行开仓: "
                    f"{int(fast_alpha_policy.get('softened_open_count') or 0)} "
                    f"(24h {int(fast_alpha_policy.get('recent_24h_softened_open_count') or 0)})"
                ),
                (
                    f"- Short-horizon softened opens: "
                    f"{int(fast_alpha_policy.get('softened_open_count') or 0)} "
                    f"(24h {int(fast_alpha_policy.get('recent_24h_softened_open_count') or 0)})"
                ),
            ),
            text_for(
                lang,
                (
                    f"- 放行来源: positive_edge="
                    f"{int(fast_alpha_policy.get('positive_edge_softened_open_count') or 0)}, "
                    f"warming_up={int(fast_alpha_policy.get('warming_up_softened_open_count') or 0)}"
                ),
                (
                    f"- Softened source mix: positive_edge="
                    f"{int(fast_alpha_policy.get('positive_edge_softened_open_count') or 0)}, "
                    f"warming_up={int(fast_alpha_policy.get('warming_up_softened_open_count') or 0)}"
                ),
            ),
            text_for(
                lang,
                (
                    f"- 放行后已平仓: {int(fast_alpha_policy.get('softened_closed_trade_count') or 0)} "
                    f"笔, net=${float(fast_alpha_policy.get('softened_total_net_pnl') or 0.0):+.2f}, "
                    f"avg={float(fast_alpha_policy.get('softened_avg_net_return_pct') or 0.0):+.2f}%"
                ),
                (
                    f"- Softened closed trades: {int(fast_alpha_policy.get('softened_closed_trade_count') or 0)} "
                    f"trades, net=${float(fast_alpha_policy.get('softened_total_net_pnl') or 0.0):+.2f}, "
                    f"avg={float(fast_alpha_policy.get('softened_avg_net_return_pct') or 0.0):+.2f}%"
                ),
            ),
            text_for(
                lang,
                (
                    f"- 负期望暂停次数: "
                    f"{int(fast_alpha_policy.get('negative_expectancy_pause_count') or 0)} "
                    f"(24h {int(fast_alpha_policy.get('recent_24h_negative_expectancy_pause_count') or 0)})"
                ),
                (
                    f"- Negative-edge pause count: "
                    f"{int(fast_alpha_policy.get('negative_expectancy_pause_count') or 0)} "
                    f"(24h {int(fast_alpha_policy.get('recent_24h_negative_expectancy_pause_count') or 0)})"
                ),
            ),
            text_for(
                lang,
                f"- Short-horizon 放行后状态: {fast_alpha_policy.get('softened_status', 'unknown')}",
                f"- Short-horizon Softened Status: {fast_alpha_policy.get('softened_status', 'unknown')}",
            ),
            text_for(
                lang,
                f"- Short-horizon 放行后净期望: {float(fast_alpha_policy.get('softened_avg_net_return_pct') or 0.0):+.2f}%",
                f"- Short-horizon Softened Expectancy: {float(fast_alpha_policy.get('softened_avg_net_return_pct') or 0.0):+.2f}%",
            ),
            text_for(
                lang,
                f"- Short-horizon 放行后净盈亏比: {float(fast_alpha_policy.get('softened_profit_factor') or 0.0):.2f}",
                f"- Short-horizon Softened Profit Factor: {float(fast_alpha_policy.get('softened_profit_factor') or 0.0):.2f}",
            ),
            "",
            text_for(lang, "## 退出拖累", "## Exit Drag"),
            text_for(
                lang,
                (
                    f"- 超短 research_exit: {short_exit.get('count', 0)} 笔, "
                    f"亏损占比 {float(short_exit.get('loss_ratio_pct') or 0.0):.2f}%, "
                    f"均值 {float(short_exit.get('avg_net_return_pct') or 0.0):+.2f}%"
                ),
                (
                    f"- Short-hold research_exit: {short_exit.get('count', 0)} trades, "
                    f"loss_ratio {float(short_exit.get('loss_ratio_pct') or 0.0):.2f}%, "
                    f"avg {float(short_exit.get('avg_net_return_pct') or 0.0):+.2f}%"
                ),
            ),
            text_for(
                lang,
                f"- 最近24h runner 观察事件: {int((data.get('review_watch') or {}).get('recent_24h_count', 0) or 0)}",
                f"- Runner watch events (24h): {int((data.get('review_watch') or {}).get('recent_24h_count', 0) or 0)}",
            ),
            "",
            text_for(lang, "## 在线 Pause", "## Online Pause"),
            text_for(
                lang,
                (
                    f"- 当前 blocked setups: {learning.get('blocked_setup_count', 0)} "
                    f"(recent_negative={learning.get('recent_negative_setup_pause_count', 0)}, "
                    f"shadow_rehab={learning.get('shadow_rehabilitated_setup_count', 0)})"
                ),
                (
                    f"- Blocked setups: {learning.get('blocked_setup_count', 0)} "
                    f"(recent_negative={learning.get('recent_negative_setup_pause_count', 0)}, "
                    f"shadow_rehab={learning.get('shadow_rehabilitated_setup_count', 0)})"
                ),
            ),
        ]
        blocked = learning.get("blocked_setups", []) or []
        if blocked:
            lines.append(text_for(lang, "- 最近封禁:", "- Recent pauses:"))
            for entry in blocked[:5]:
                criteria = entry.get("criteria", {}) or {}
                lines.append(
                    text_for(
                        lang,
                        f"  {json.dumps(criteria, ensure_ascii=False, sort_keys=True)} -> {entry.get('reason', '')}",
                        f"  {json.dumps(criteria, ensure_ascii=False, sort_keys=True)} -> {entry.get('reason', '')}",
                    )
                )
        lines.extend(
            [
                "",
                text_for(lang, "## 标的摘要", "## Symbol Snapshot"),
            ]
        )
        if symbol_summary:
            for row in symbol_summary:
                lines.append(
                    text_for(
                        lang,
                        (
                            f"- {row.get('symbol')}: trades={row.get('trade_count', 0)}, "
                            f"avg={float(row.get('avg_net_return_pct') or 0.0):+.2f}%, "
                            f"net=${float(row.get('total_net_pnl') or 0.0):+.2f}, "
                            f"latest_review={row.get('latest_reviewed_action') or '-'}"
                        ),
                        (
                            f"- {row.get('symbol')}: trades={row.get('trade_count', 0)}, "
                            f"avg={float(row.get('avg_net_return_pct') or 0.0):+.2f}%, "
                            f"net=${float(row.get('total_net_pnl') or 0.0):+.2f}, "
                            f"latest_review={row.get('latest_reviewed_action') or '-'}"
                        ),
                    )
                )
        return "\n".join(lines)
