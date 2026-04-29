"""Daily focus report with five operational metrics."""
from __future__ import annotations

from datetime import datetime, timezone

from config import Settings, get_settings
from core.i18n import get_default_language, normalize_language, text_for
from core.storage import Storage
from monitor.performance_report import PerformanceReporter


class DailyFocusReporter:
    """Summarize the five most decision-relevant daily metrics."""

    EXECUTION_SYMBOLS_STATE_KEY = "execution_symbols"

    def __init__(self, storage: Storage, settings: Settings | None = None):
        self.storage = storage
        self.settings = settings or get_settings()
        self.performance = PerformanceReporter(storage, self.settings)

    def build(self, symbols: list[str] | None = None) -> dict:
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        selected_symbols = symbols or self.storage.get_json_state(
            self.EXECUTION_SYMBOLS_STATE_KEY,
            [],
        ) or []

        closed_trade_pnls: list[float] = []
        closed_trade_returns: list[float] = []
        daily_net_pnl = 0.0
        fast_alpha_net_pnl = 0.0
        fast_alpha_short_horizon_net_pnl = 0.0
        fast_alpha_closed_trade_count = 0
        fast_alpha_short_horizon_closed_trade_count = 0
        fast_alpha_open_count = 0
        fast_alpha_short_horizon_open_count = 0
        fast_alpha_negative_expectancy_pause_count = 0
        with self.storage._conn() as conn:
            ledger_query = (
                "SELECT event_time, net_pnl FROM pnl_ledger "
                "WHERE event_time >= ?"
            )
            params: list[str] = [today_start.isoformat()]
            if selected_symbols:
                placeholders = ",".join("?" for _ in selected_symbols)
                ledger_query += f" AND symbol IN ({placeholders})"
                params.extend(list(selected_symbols))
            ledger_rows = conn.execute(ledger_query, params).fetchall()
            daily_net_pnl = sum(float(row["net_pnl"] or 0.0) for row in ledger_rows)
            fast_alpha_ledger_query = (
                "SELECT event_time, net_pnl FROM pnl_ledger "
                "WHERE event_time >= ? "
                "AND json_extract(metadata_json, '$.pipeline_mode')='fast_alpha'"
            )
            fast_alpha_ledger_params: list[str] = [today_start.isoformat()]
            if selected_symbols:
                placeholders = ",".join("?" for _ in selected_symbols)
                fast_alpha_ledger_query += f" AND symbol IN ({placeholders})"
                fast_alpha_ledger_params.extend(list(selected_symbols))
            fast_alpha_ledger_rows = conn.execute(
                fast_alpha_ledger_query,
                fast_alpha_ledger_params,
            ).fetchall()
            fast_alpha_net_pnl = sum(
                float(row["net_pnl"] or 0.0) for row in fast_alpha_ledger_rows
            )
            fast_alpha_short_horizon_ledger_query = (
                "SELECT event_time, net_pnl FROM pnl_ledger "
                "WHERE event_time >= ? "
                "AND json_extract(metadata_json, '$.pipeline_mode')='fast_alpha' "
                "AND COALESCE(json_extract(metadata_json, '$.fast_alpha_review_policy_reason'), '') != ''"
            )
            fast_alpha_short_horizon_ledger_params: list[str] = [today_start.isoformat()]
            if selected_symbols:
                placeholders = ",".join("?" for _ in selected_symbols)
                fast_alpha_short_horizon_ledger_query += f" AND symbol IN ({placeholders})"
                fast_alpha_short_horizon_ledger_params.extend(list(selected_symbols))
            fast_alpha_short_horizon_ledger_rows = conn.execute(
                fast_alpha_short_horizon_ledger_query,
                fast_alpha_short_horizon_ledger_params,
            ).fetchall()
            fast_alpha_short_horizon_net_pnl = sum(
                float(row["net_pnl"] or 0.0)
                for row in fast_alpha_short_horizon_ledger_rows
            )

            closed_query = (
                "SELECT pnl, pnl_pct FROM trades WHERE status='closed' AND exit_time >= ?"
            )
            closed_params: list[str] = [today_start.isoformat()]
            if selected_symbols:
                placeholders = ",".join("?" for _ in selected_symbols)
                closed_query += f" AND symbol IN ({placeholders})"
                closed_params.extend(list(selected_symbols))
            closed_rows = conn.execute(closed_query, closed_params).fetchall()
            closed_trade_pnls = [float(row["pnl"] or 0.0) for row in closed_rows]
            closed_trade_returns = [float(row["pnl_pct"] or 0.0) for row in closed_rows]
            fast_alpha_closed_query = (
                "SELECT COUNT(*) AS c FROM trades "
                "WHERE status='closed' AND exit_time >= ? "
                "AND json_extract(metadata_json, '$.pipeline_mode')='fast_alpha'"
            )
            fast_alpha_closed_params: list[str] = [today_start.isoformat()]
            if selected_symbols:
                placeholders = ",".join("?" for _ in selected_symbols)
                fast_alpha_closed_query += f" AND symbol IN ({placeholders})"
                fast_alpha_closed_params.extend(list(selected_symbols))
            fast_alpha_closed_row = conn.execute(
                fast_alpha_closed_query,
                fast_alpha_closed_params,
            ).fetchone()
            fast_alpha_closed_trade_count = int(
                (fast_alpha_closed_row["c"] or 0) if fast_alpha_closed_row else 0
            )
            fast_alpha_short_horizon_closed_query = (
                "SELECT COUNT(*) AS c FROM trades "
                "WHERE status='closed' AND exit_time >= ? "
                "AND json_extract(metadata_json, '$.pipeline_mode')='fast_alpha' "
                "AND COALESCE(json_extract(metadata_json, '$.fast_alpha_review_policy_reason'), '') != ''"
            )
            fast_alpha_short_horizon_closed_params: list[str] = [today_start.isoformat()]
            if selected_symbols:
                placeholders = ",".join("?" for _ in selected_symbols)
                fast_alpha_short_horizon_closed_query += f" AND symbol IN ({placeholders})"
                fast_alpha_short_horizon_closed_params.extend(list(selected_symbols))
            fast_alpha_short_horizon_closed_row = conn.execute(
                fast_alpha_short_horizon_closed_query,
                fast_alpha_short_horizon_closed_params,
            ).fetchone()
            fast_alpha_short_horizon_closed_trade_count = int(
                (fast_alpha_short_horizon_closed_row["c"] or 0)
                if fast_alpha_short_horizon_closed_row
                else 0
            )
            fast_alpha_open_query = (
                "SELECT COUNT(*) AS c FROM execution_events "
                "WHERE event_type='fast_alpha_open' AND created_at >= ?"
            )
            fast_alpha_open_params: list[str] = [today_start.isoformat()]
            if selected_symbols:
                placeholders = ",".join("?" for _ in selected_symbols)
                fast_alpha_open_query += f" AND symbol IN ({placeholders})"
                fast_alpha_open_params.extend(list(selected_symbols))
            fast_alpha_open_row = conn.execute(
                fast_alpha_open_query,
                fast_alpha_open_params,
            ).fetchone()
            fast_alpha_open_count = int(
                (fast_alpha_open_row["c"] or 0) if fast_alpha_open_row else 0
            )
            fast_alpha_short_horizon_open_query = (
                "SELECT COUNT(*) AS c FROM execution_events "
                "WHERE event_type='fast_alpha_open' AND created_at >= ? "
                "AND COALESCE(json_extract(payload_json, '$.review_policy_reason'), '') != ''"
            )
            fast_alpha_short_horizon_open_params: list[str] = [today_start.isoformat()]
            if selected_symbols:
                placeholders = ",".join("?" for _ in selected_symbols)
                fast_alpha_short_horizon_open_query += f" AND symbol IN ({placeholders})"
                fast_alpha_short_horizon_open_params.extend(list(selected_symbols))
            fast_alpha_short_horizon_open_row = conn.execute(
                fast_alpha_short_horizon_open_query,
                fast_alpha_short_horizon_open_params,
            ).fetchone()
            fast_alpha_short_horizon_open_count = int(
                (fast_alpha_short_horizon_open_row["c"] or 0)
                if fast_alpha_short_horizon_open_row
                else 0
            )
            fast_alpha_negative_pause_query = (
                "SELECT COUNT(*) AS c FROM execution_events "
                "WHERE event_type='fast_alpha_blocked' AND created_at >= ? "
                "AND json_extract(payload_json, '$.reason')='short_horizon_negative_expectancy_pause'"
            )
            fast_alpha_negative_pause_params: list[str] = [today_start.isoformat()]
            if selected_symbols:
                placeholders = ",".join("?" for _ in selected_symbols)
                fast_alpha_negative_pause_query += f" AND symbol IN ({placeholders})"
                fast_alpha_negative_pause_params.extend(list(selected_symbols))
            fast_alpha_negative_pause_row = conn.execute(
                fast_alpha_negative_pause_query,
                fast_alpha_negative_pause_params,
            ).fetchone()
            fast_alpha_negative_expectancy_pause_count = int(
                (fast_alpha_negative_pause_row["c"] or 0)
                if fast_alpha_negative_pause_row
                else 0
            )

            blocked_query = (
                "SELECT pnl_pct FROM shadow_trade_runs "
                "WHERE status='closed' AND evaluated_at >= ?"
            )
            blocked_params: list[str] = [today_start.isoformat()]
            if selected_symbols:
                placeholders = ",".join("?" for _ in selected_symbols)
                blocked_query += f" AND symbol IN ({placeholders})"
                blocked_params.extend(list(selected_symbols))
            blocked_rows = conn.execute(blocked_query, blocked_params).fetchall()

        blocked_avoided_loss_pct = sum(
            abs(float(row["pnl_pct"] or 0.0))
            for row in blocked_rows
            if float(row["pnl_pct"] or 0.0) < 0.0
        )
        blocked_avoided_loss_count = sum(
            1 for row in blocked_rows if float(row["pnl_pct"] or 0.0) < 0.0
        )
        daily_profit_factor = self.performance._profit_factor(closed_trade_pnls)
        daily_max_drawdown_pct = self.performance._returns_max_drawdown_pct(
            closed_trade_returns
        )
        avg_net_pnl_per_trade = (
            sum(closed_trade_pnls) / len(closed_trade_pnls)
            if closed_trade_pnls
            else 0.0
        )
        return {
            "generated_at": now.isoformat(),
            "execution_symbols": list(selected_symbols),
            "daily_net_pnl": daily_net_pnl,
            "daily_profit_factor": daily_profit_factor,
            "daily_max_drawdown_pct": daily_max_drawdown_pct,
            "avg_net_pnl_per_trade": avg_net_pnl_per_trade,
            "blocked_avoided_loss_pct": blocked_avoided_loss_pct,
            "blocked_avoided_loss_count": blocked_avoided_loss_count,
            "closed_trade_count": len(closed_trade_pnls),
            "fast_alpha_net_pnl": fast_alpha_net_pnl,
            "fast_alpha_short_horizon_net_pnl": fast_alpha_short_horizon_net_pnl,
            "fast_alpha_closed_trade_count": fast_alpha_closed_trade_count,
            "fast_alpha_short_horizon_closed_trade_count": fast_alpha_short_horizon_closed_trade_count,
            "fast_alpha_open_count": fast_alpha_open_count,
            "fast_alpha_short_horizon_open_count": fast_alpha_short_horizon_open_count,
            "fast_alpha_negative_expectancy_pause_count": fast_alpha_negative_expectancy_pause_count,
        }

    @staticmethod
    def render(data: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        lines = [
            text_for(lang, "# 每日焦点复盘", "# Daily Focus Review"),
            text_for(
                lang,
                (
                    f"- 执行池: {', '.join(data.get('execution_symbols', []) or []) or '无'}"
                ),
                (
                    f"- Execution Pool: {', '.join(data.get('execution_symbols', []) or []) or 'none'}"
                ),
            ),
            text_for(
                lang,
                f"- 净收益: ${float(data.get('daily_net_pnl') or 0.0):+.2f}",
                f"- Daily Net PnL: ${float(data.get('daily_net_pnl') or 0.0):+.2f}",
            ),
            text_for(
                lang,
                f"- 盈亏因子: {float(data.get('daily_profit_factor') or 0.0):.2f}",
                f"- Daily Profit Factor: {float(data.get('daily_profit_factor') or 0.0):.2f}",
            ),
            text_for(
                lang,
                f"- 最大回撤: {float(data.get('daily_max_drawdown_pct') or 0.0):.2f}%",
                f"- Daily Max Drawdown: {float(data.get('daily_max_drawdown_pct') or 0.0):.2f}%",
            ),
            text_for(
                lang,
                f"- 每笔平均净收益: ${float(data.get('avg_net_pnl_per_trade') or 0.0):+.2f}",
                f"- Average Net PnL Per Trade: ${float(data.get('avg_net_pnl_per_trade') or 0.0):+.2f}",
            ),
            text_for(
                lang,
                f"- 风控避免亏损: {float(data.get('blocked_avoided_loss_pct') or 0.0):.2f}%",
                f"- Risk-Blocked Loss Avoided: {float(data.get('blocked_avoided_loss_pct') or 0.0):.2f}%",
            ),
            text_for(
                lang,
                (
                    f"- 已平仓交易数: {int(data.get('closed_trade_count') or 0)} "
                    f"| 避险样本数: {int(data.get('blocked_avoided_loss_count') or 0)}"
                ),
                (
                    f"- Closed Trades: {int(data.get('closed_trade_count') or 0)} "
                    f"| Avoided-Loss Samples: {int(data.get('blocked_avoided_loss_count') or 0)}"
                ),
            ),
            text_for(
                lang,
                f"- Fast Alpha 净收益: ${float(data.get('fast_alpha_net_pnl') or 0.0):+.2f}",
                f"- Fast Alpha Net PnL: ${float(data.get('fast_alpha_net_pnl') or 0.0):+.2f}",
            ),
            text_for(
                lang,
                f"- Fast Alpha 平仓数: {int(data.get('fast_alpha_closed_trade_count') or 0)}",
                f"- Fast Alpha Closed Trades: {int(data.get('fast_alpha_closed_trade_count') or 0)}",
            ),
            text_for(
                lang,
                f"- Fast Alpha 开仓数: {int(data.get('fast_alpha_open_count') or 0)}",
                f"- Fast Alpha Opens: {int(data.get('fast_alpha_open_count') or 0)}",
            ),
            text_for(
                lang,
                f"- Short-horizon 放行开仓数: {int(data.get('fast_alpha_short_horizon_open_count') or 0)}",
                f"- Short-horizon Softened Opens: {int(data.get('fast_alpha_short_horizon_open_count') or 0)}",
            ),
            text_for(
                lang,
                f"- Short-horizon 放行净收益: ${float(data.get('fast_alpha_short_horizon_net_pnl') or 0.0):+.2f}",
                f"- Short-horizon Softened Net PnL: ${float(data.get('fast_alpha_short_horizon_net_pnl') or 0.0):+.2f}",
            ),
            text_for(
                lang,
                f"- Short-horizon 放行平仓数: {int(data.get('fast_alpha_short_horizon_closed_trade_count') or 0)}",
                f"- Short-horizon Softened Closed Trades: {int(data.get('fast_alpha_short_horizon_closed_trade_count') or 0)}",
            ),
            text_for(
                lang,
                f"- Short-horizon 负期望暂停次数: {int(data.get('fast_alpha_negative_expectancy_pause_count') or 0)}",
                f"- Short-horizon Negative-Edge Pauses: {int(data.get('fast_alpha_negative_expectancy_pause_count') or 0)}",
            ),
        ]
        return "\n".join(lines)
