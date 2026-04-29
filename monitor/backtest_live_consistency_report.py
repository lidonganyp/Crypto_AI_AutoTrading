"""Compare backtest, walk-forward, and live trading quality."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from config import Settings, get_settings
from core.i18n import get_default_language, normalize_language, text_for
from core.storage import Storage
from monitor.performance_report import PerformanceReporter


class BacktestLiveConsistencyReporter:
    """Highlight where offline validation is diverging from live trading."""

    EXECUTION_SYMBOLS_STATE_KEY = "execution_symbols"
    EXTREME_RETURN_THRESHOLD_PCT = 500.0
    EXTREME_RETURN_MIN_SAMPLE_COUNT = 5
    LIVE_DIVERGENCE_MIN_TRADES = 5
    LIVE_DIVERGENCE_WALKFORWARD_MIN_TRADES = 5
    LIVE_DIVERGENCE_WALKFORWARD_RETURN_PCT = 100.0
    LIVE_DIVERGENCE_WALKFORWARD_PROFIT_FACTOR = 1.5

    def __init__(self, storage: Storage, settings: Settings | None = None):
        self.storage = storage
        self.settings = settings or get_settings()
        self.performance = PerformanceReporter(storage, self.settings)

    def build(self, symbols: list[str] | None = None) -> dict:
        selected_symbols = symbols or self.storage.get_json_state(
            self.EXECUTION_SYMBOLS_STATE_KEY,
            [],
        ) or []
        rows = list(self.build_symbol_consistency_rows(selected_symbols).values())
        suspicious_rows = [row for row in rows if row["flags"]]
        performance = self.performance.build()
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "execution_symbols": list(selected_symbols),
            "rows": rows,
            "suspicious_symbol_count": len(suspicious_rows),
            "recent_live_expectancy_pct": performance.recent_expectancy_pct,
            "recent_live_profit_factor": performance.recent_profit_factor,
            "recent_live_max_drawdown_pct": performance.recent_max_drawdown_pct,
        }

    def build_symbol_consistency_rows(
        self,
        symbols: list[str],
        *,
        walkforward_overrides: dict[str, dict[str, float | int | str]] | None = None,
    ) -> dict[str, dict[str, float | int | str | list[str]]]:
        overrides = walkforward_overrides or {}
        return {
            symbol: self.symbol_consistency_row(
                symbol,
                walkforward_override=overrides.get(symbol),
            )
            for symbol in symbols
        }

    def consistency_flags(
        self,
        symbol: str,
        *,
        walkforward_override: dict[str, float | int | str] | None = None,
    ) -> list[str]:
        row = self.symbol_consistency_row(
            symbol,
            walkforward_override=walkforward_override,
        )
        return list(row.get("flags", []) or [])

    def symbol_consistency_row(
        self,
        symbol: str,
        *,
        walkforward_override: dict[str, float | int | str] | None = None,
    ) -> dict[str, float | int | str | list[str]]:
        with self.storage._conn() as conn:
            backtest_row = conn.execute(
                "SELECT summary_json FROM backtest_runs WHERE symbol = ? "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            walkforward_row = conn.execute(
                "SELECT summary_json, created_at FROM walkforward_runs WHERE symbol = ? "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            training_row = conn.execute(
                "SELECT metadata_json, created_at FROM training_runs WHERE symbol = ? "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            closed_rows = conn.execute(
                "SELECT pnl, pnl_pct FROM trades WHERE status='closed' AND symbol = ? "
                "ORDER BY exit_time ASC, id ASC",
                (symbol,),
            ).fetchall()
            cost_row = conn.execute(
                "SELECT SUM(fee_cost) AS fee_cost, SUM(slippage_cost) AS slippage_cost "
                "FROM pnl_ledger WHERE symbol = ?",
                (symbol,),
            ).fetchone()

        backtest = json.loads(backtest_row["summary_json"]) if backtest_row else {}
        latest_training_walkforward = self._latest_training_walkforward_summary(
            training_row
        )
        walkforward = (
            dict(walkforward_override)
            if isinstance(walkforward_override, dict)
            else latest_training_walkforward
            if self._prefer_training_walkforward(training_row, walkforward_row)
            else json.loads(walkforward_row["summary_json"])
            if walkforward_row
            else {}
        )
        live_pnls = [float(row["pnl"] or 0.0) for row in closed_rows]
        live_returns = [float(row["pnl_pct"] or 0.0) for row in closed_rows]
        live_trade_count = len(live_pnls)
        live_total_pnl = sum(live_pnls)
        live_avg_return_pct = (
            sum(live_returns) / live_trade_count if live_trade_count else 0.0
        )
        live_profit_factor = self.performance._profit_factor(live_pnls)
        live_max_drawdown_pct = self.performance._returns_max_drawdown_pct(
            live_returns
        )
        total_cost = float((cost_row["fee_cost"] or 0.0) if cost_row else 0.0) + float(
            (cost_row["slippage_cost"] or 0.0) if cost_row else 0.0
        )

        backtest_total_return_pct = float(backtest.get("total_return_pct", 0.0) or 0.0)
        backtest_trade_count = int(backtest.get("trade_count", 0) or 0)
        backtest_total_splits = int(backtest.get("total_splits", 0) or 0)
        walkforward_total_return_pct = float(
            walkforward.get("total_return_pct", 0.0) or 0.0
        )
        walkforward_profit_factor = float(
            walkforward.get("profit_factor", 0.0) or 0.0
        )
        walkforward_trade_count = int(walkforward.get("trade_count", 0) or 0)
        walkforward_total_splits = int(walkforward.get("total_splits", 0) or 0)
        flags = self._consistency_flags(
            backtest_total_return_pct=backtest_total_return_pct,
            backtest_trade_count=backtest_trade_count,
            backtest_total_splits=backtest_total_splits,
            walkforward_total_return_pct=walkforward_total_return_pct,
            walkforward_profit_factor=walkforward_profit_factor,
            walkforward_trade_count=walkforward_trade_count,
            walkforward_total_splits=walkforward_total_splits,
            live_trade_count=live_trade_count,
            live_total_pnl=live_total_pnl,
            live_profit_factor=live_profit_factor,
        )
        return {
            "symbol": symbol,
            "backtest_total_return_pct": backtest_total_return_pct,
            "backtest_trade_count": backtest_trade_count,
            "walkforward_total_return_pct": walkforward_total_return_pct,
            "walkforward_profit_factor": walkforward_profit_factor,
            "walkforward_trade_count": walkforward_trade_count,
            "live_trade_count": live_trade_count,
            "live_total_pnl": live_total_pnl,
            "live_avg_return_pct": live_avg_return_pct,
            "live_profit_factor": live_profit_factor,
            "live_max_drawdown_pct": live_max_drawdown_pct,
            "live_total_cost": total_cost,
            "flags": flags,
        }

    @staticmethod
    def _parse_created_at(row) -> datetime | None:
        if not row:
            return None
        raw = str(row["created_at"] or "").strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    @staticmethod
    def _latest_training_walkforward_summary(
        training_row,
    ) -> dict[str, float | int | str]:
        if not training_row:
            return {}
        try:
            metadata = json.loads(training_row["metadata_json"] or "{}")
        except Exception:
            return {}
        if not isinstance(metadata, dict):
            return {}
        summary = metadata.get("candidate_walkforward_summary", {}) or {}
        return dict(summary) if isinstance(summary, dict) else {}

    def _prefer_training_walkforward(self, training_row, walkforward_row) -> bool:
        training_summary = self._latest_training_walkforward_summary(training_row)
        if not training_summary:
            return False
        training_created_at = self._parse_created_at(training_row)
        walkforward_created_at = self._parse_created_at(walkforward_row)
        if training_created_at is None:
            return False
        if walkforward_created_at is None:
            return True
        return training_created_at >= walkforward_created_at

    def _consistency_flags(
        self,
        *,
        backtest_total_return_pct: float,
        backtest_trade_count: int,
        backtest_total_splits: int,
        walkforward_total_return_pct: float,
        walkforward_profit_factor: float,
        walkforward_trade_count: int,
        walkforward_total_splits: int,
        live_trade_count: int,
        live_total_pnl: float,
        live_profit_factor: float,
    ) -> list[str]:
        flags: list[str] = []
        backtest_sample_count = (
            backtest_trade_count
            if backtest_trade_count > 0
            else backtest_total_splits
        )
        walkforward_sample_count = (
            walkforward_trade_count
            if walkforward_trade_count > 0
            else walkforward_total_splits
        )
        if abs(backtest_total_return_pct) >= self.EXTREME_RETURN_THRESHOLD_PCT:
            flags.append(
                "backtest_return_extreme"
                if backtest_sample_count >= self.EXTREME_RETURN_MIN_SAMPLE_COUNT
                else "backtest_return_extreme_sparse"
            )
        if abs(walkforward_total_return_pct) >= self.EXTREME_RETURN_THRESHOLD_PCT:
            flags.append(
                "walkforward_return_extreme"
                if walkforward_sample_count >= self.EXTREME_RETURN_MIN_SAMPLE_COUNT
                else "walkforward_return_extreme_sparse"
            )
        if (
            live_trade_count >= self.LIVE_DIVERGENCE_MIN_TRADES
            and walkforward_sample_count >= self.LIVE_DIVERGENCE_WALKFORWARD_MIN_TRADES
            and walkforward_total_return_pct > self.LIVE_DIVERGENCE_WALKFORWARD_RETURN_PCT
            and live_total_pnl <= 0
        ):
            flags.append("walkforward_live_pnl_divergence")
        if (
            live_trade_count >= self.LIVE_DIVERGENCE_MIN_TRADES
            and walkforward_sample_count >= self.LIVE_DIVERGENCE_WALKFORWARD_MIN_TRADES
            and walkforward_profit_factor > self.LIVE_DIVERGENCE_WALKFORWARD_PROFIT_FACTOR
            and live_profit_factor < 1.0
        ):
            flags.append("profit_factor_divergence")
        return flags

    @staticmethod
    def render(data: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        lines = [
            text_for(lang, "# 回测/实盘一致性校验", "# Backtest/Live Consistency Check"),
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
                f"- 可疑标的数: {int(data.get('suspicious_symbol_count') or 0)}",
                f"- Suspicious Symbols: {int(data.get('suspicious_symbol_count') or 0)}",
            ),
            text_for(
                lang,
                f"- 最近实盘净期望: {float(data.get('recent_live_expectancy_pct') or 0.0):+.2f}%",
                f"- Recent Live Expectancy: {float(data.get('recent_live_expectancy_pct') or 0.0):+.2f}%",
            ),
            text_for(
                lang,
                f"- 最近实盘盈亏因子: {float(data.get('recent_live_profit_factor') or 0.0):.2f}",
                f"- Recent Live Profit Factor: {float(data.get('recent_live_profit_factor') or 0.0):.2f}",
            ),
            text_for(
                lang,
                f"- 最近实盘最大回撤: {float(data.get('recent_live_max_drawdown_pct') or 0.0):.2f}%",
                f"- Recent Live Max Drawdown: {float(data.get('recent_live_max_drawdown_pct') or 0.0):.2f}%",
            ),
            "",
            text_for(lang, "## 标的对照", "## Symbol Comparison"),
        ]
        rows = data.get("rows", []) or []
        if not rows:
            lines.append(text_for(lang, "- 无可用样本", "- No symbols available"))
            return "\n".join(lines)
        for row in rows:
            flags = ",".join(row.get("flags", []) or []) or "-"
            lines.append(
                text_for(
                    lang,
                    (
                        f"- {row['symbol']}: bt={float(row['backtest_total_return_pct']):+.2f}% | "
                        f"bt_trades={int(row.get('backtest_trade_count') or 0)} | "
                        f"wf={float(row['walkforward_total_return_pct']):+.2f}% | "
                        f"wf_trades={int(row.get('walkforward_trade_count') or 0)} | "
                        f"live_trades={int(row['live_trade_count'])} | "
                        f"live_avg={float(row['live_avg_return_pct']):+.2f}% | "
                        f"live_pf={float(row['live_profit_factor']):.2f} | "
                        f"cost=${float(row['live_total_cost']):.2f} | "
                        f"flags={flags}"
                    ),
                    (
                        f"- {row['symbol']}: bt={float(row['backtest_total_return_pct']):+.2f}% | "
                        f"bt_trades={int(row.get('backtest_trade_count') or 0)} | "
                        f"wf={float(row['walkforward_total_return_pct']):+.2f}% | "
                        f"wf_trades={int(row.get('walkforward_trade_count') or 0)} | "
                        f"live_trades={int(row['live_trade_count'])} | "
                        f"live_avg={float(row['live_avg_return_pct']):+.2f}% | "
                        f"live_pf={float(row['live_profit_factor']):.2f} | "
                        f"cost=${float(row['live_total_cost']):.2f} | "
                        f"flags={flags}"
                    ),
                )
            )
        return "\n".join(lines)
