"""Execution-pool trade attribution report for CryptoAI v3."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import re

from core.i18n import get_default_language, normalize_language, text_for
from core.storage import Storage


class PoolAttributionReporter:
    """Summarize execution-pool trade outcomes and latest review posture."""

    EXECUTION_SYMBOLS_STATE_KEY = "execution_symbols"

    def __init__(self, storage: Storage):
        self.storage = storage

    def build(self, symbols: list[str] | None = None) -> dict:
        selected_symbols = self._normalize_symbols(
            symbols or self.storage.get_json_state(self.EXECUTION_SYMBOLS_STATE_KEY, [])
        )
        with self.storage._conn() as conn:
            if not selected_symbols:
                trade_rows = conn.execute(
                    "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 100"
                ).fetchall()
                selected_symbols = self._normalize_symbols(
                    [str(row["symbol"]) for row in trade_rows]
                )
            else:
                placeholders = ",".join("?" for _ in selected_symbols)
                trade_rows = conn.execute(
                    f"""
                    SELECT * FROM trades
                    WHERE symbol IN ({placeholders})
                    ORDER BY entry_time DESC
                    """,
                    tuple(selected_symbols),
                ).fetchall()

            ledger_rows = []
            latest_reviews: dict[str, dict] = {}
            if selected_symbols:
                placeholders = ",".join("?" for _ in selected_symbols)
                ledger_rows = conn.execute(
                    f"""
                    SELECT * FROM pnl_ledger
                    WHERE symbol IN ({placeholders})
                    ORDER BY event_time ASC, id ASC
                    """,
                    tuple(selected_symbols),
                ).fetchall()
                review_rows = conn.execute(
                    f"""
                    SELECT symbol, payload_json, created_at
                    FROM execution_events
                    WHERE event_type='research_review'
                      AND symbol IN ({placeholders})
                    ORDER BY created_at DESC, id DESC
                    """,
                    tuple(selected_symbols),
                ).fetchall()
                for row in review_rows:
                    symbol = str(row["symbol"])
                    if symbol in latest_reviews:
                        continue
                    payload = self._load_json(row["payload_json"])
                    latest_reviews[symbol] = {
                        "created_at": str(row["created_at"] or ""),
                        "reviewed_action": str(payload.get("reviewed_action") or ""),
                        "review_score": self._safe_float(payload.get("review_score"), None),
                        "reasons": [
                            str(item)
                            for item in (payload.get("reasons") or [])
                            if str(item).strip()
                        ],
                    }

        ledger_by_trade: dict[str, list[dict]] = defaultdict(list)
        for row in ledger_rows:
            payload = dict(row)
            payload["metadata"] = self._load_json(payload.get("metadata_json"))
            ledger_by_trade[str(payload.get("trade_id") or "")].append(payload)

        symbol_rollups: dict[str, dict] = {
            symbol: {
                "trade_count": 0,
                "win_count": 0,
                "total_net_pnl": 0.0,
                "partial_realized_net_pnl": 0.0,
                "total_net_return_pct": 0.0,
                "total_holding_hours": 0.0,
                "open_trade_count": 0,
                "partially_realized_open_trade_count": 0,
                "reviewed_actions": Counter(),
                "exit_reasons": Counter(),
                "review_score_sum": 0.0,
                "review_score_count": 0,
            }
            for symbol in selected_symbols
        }
        closed_trades: list[dict] = []
        open_trades: list[dict] = []

        for row in trade_rows:
            trade = dict(row)
            symbol = str(trade.get("symbol") or "")
            symbol_rollups.setdefault(
                symbol,
                {
                    "trade_count": 0,
                    "win_count": 0,
                    "total_net_pnl": 0.0,
                    "partial_realized_net_pnl": 0.0,
                    "total_net_return_pct": 0.0,
                    "total_holding_hours": 0.0,
                    "open_trade_count": 0,
                    "partially_realized_open_trade_count": 0,
                    "reviewed_actions": Counter(),
                    "exit_reasons": Counter(),
                    "review_score_sum": 0.0,
                    "review_score_count": 0,
                },
            )
            metadata = self._load_json(trade.get("metadata_json"))
            trade_ledgers = ledger_by_trade.get(str(trade.get("id") or ""), [])
            open_ledgers = [item for item in trade_ledgers if item.get("event_type") == "open"]
            close_ledgers = [item for item in trade_ledgers if item.get("event_type") == "close"]
            entry_notional = sum(
                self._safe_float(item.get("notional_value")) for item in open_ledgers
            )
            if entry_notional <= 0:
                entry_notional = self._safe_float(trade.get("entry_price")) * self._safe_float(
                    trade.get("initial_quantity") or trade.get("quantity")
                )
            total_net_pnl = sum(
                self._safe_float(item.get("net_pnl")) for item in trade_ledgers
            )
            net_return_pct = (
                total_net_pnl / entry_notional * 100.0 if entry_notional > 0 else 0.0
            )
            reviewed_action = str(metadata.get("reviewed_action") or "")
            review_score = self._safe_float(metadata.get("review_score"), None)
            has_realized_close = bool(close_ledgers)
            trade_row = {
                "trade_id": str(trade.get("id") or ""),
                "symbol": symbol,
                "status": str(trade.get("status") or ""),
                "opened_at": str(trade.get("entry_time") or ""),
                "closed_at": str(trade.get("exit_time") or ""),
                "holding_hours": self._holding_hours(trade, close_ledgers),
                "rationale": self._clean_text(trade.get("rationale")),
                "decision_reason": self._clean_text(metadata.get("decision_reason")),
                "reviewed_action": reviewed_action,
                "review_score": review_score,
                "exit_reasons": self._extract_exit_reasons(close_ledgers),
                "net_pnl": total_net_pnl,
                "net_return_pct": net_return_pct,
                "pipeline_mode": str(metadata.get("pipeline_mode") or ""),
                "paper_canary_mode": str(metadata.get("paper_canary_mode") or ""),
                "has_realized_close": has_realized_close,
            }
            if trade_row["status"] == "closed":
                closed_trades.append(trade_row)
                rollup = symbol_rollups[symbol]
                rollup["trade_count"] += 1
                rollup["win_count"] += 1 if total_net_pnl > 0 else 0
                rollup["total_net_pnl"] += total_net_pnl
                rollup["total_net_return_pct"] += net_return_pct
                rollup["total_holding_hours"] += trade_row["holding_hours"]
                if reviewed_action:
                    rollup["reviewed_actions"][reviewed_action] += 1
                if review_score is not None:
                    rollup["review_score_sum"] += float(review_score)
                    rollup["review_score_count"] += 1
                for reason in trade_row["exit_reasons"]:
                    rollup["exit_reasons"][reason] += 1
            elif trade_row["status"] == "open":
                rollup = symbol_rollups[symbol]
                rollup["open_trade_count"] += 1
                if has_realized_close:
                    rollup["partial_realized_net_pnl"] += total_net_pnl
                    rollup["partially_realized_open_trade_count"] += 1
                    for reason in trade_row["exit_reasons"]:
                        rollup["exit_reasons"][reason] += 1
                open_trades.append(trade_row)

        closed_trades.sort(
            key=lambda item: item.get("closed_at") or item.get("opened_at") or "",
            reverse=True,
        )
        open_trades.sort(key=lambda item: item.get("opened_at") or "", reverse=True)

        symbol_summary = []
        for symbol in selected_symbols:
            rollup = symbol_rollups.get(symbol) or {}
            trade_count = int(rollup.get("trade_count") or 0)
            latest_review = latest_reviews.get(symbol) or {}
            symbol_summary.append(
                {
                    "symbol": symbol,
                    "trade_count": trade_count,
                    "open_trade_count": int(rollup.get("open_trade_count") or 0),
                    "partially_realized_open_trade_count": int(
                        rollup.get("partially_realized_open_trade_count") or 0
                    ),
                    "win_rate_pct": (
                        self._safe_float(rollup.get("win_count")) / trade_count * 100.0
                        if trade_count
                        else 0.0
                    ),
                    "avg_net_return_pct": (
                        self._safe_float(rollup.get("total_net_return_pct")) / trade_count
                        if trade_count
                        else 0.0
                    ),
                    "total_net_pnl": self._safe_float(rollup.get("total_net_pnl"))
                    + self._safe_float(rollup.get("partial_realized_net_pnl")),
                    "partial_realized_net_pnl": self._safe_float(
                        rollup.get("partial_realized_net_pnl")
                    ),
                    "avg_holding_hours": (
                        self._safe_float(rollup.get("total_holding_hours")) / trade_count
                        if trade_count
                        else 0.0
                    ),
                    "dominant_reviewed_action": self._counter_top(
                        rollup.get("reviewed_actions")
                    ),
                    "avg_review_score": (
                        self._safe_float(rollup.get("review_score_sum"))
                        / int(rollup.get("review_score_count") or 0)
                        if int(rollup.get("review_score_count") or 0) > 0
                        else None
                    ),
                    "top_exit_reasons": self._format_counter(
                        rollup.get("exit_reasons")
                    ),
                    "latest_reviewed_action": str(
                        latest_review.get("reviewed_action") or ""
                    ),
                    "latest_review_score": latest_review.get("review_score"),
                    "latest_review_reasons": ", ".join(
                        latest_review.get("reasons") or []
                    ),
                }
            )

        closed_trade_count = len(closed_trades)
        partial_realized_net_pnl = sum(
            self._safe_float(row.get("partial_realized_net_pnl"))
            for row in symbol_summary
        )
        total_net_pnl = (
            sum(self._safe_float(item["net_pnl"]) for item in closed_trades)
            + partial_realized_net_pnl
        )
        total_wins = sum(1 for item in closed_trades if self._safe_float(item["net_pnl"]) > 0)
        total_holding_hours = sum(
            self._safe_float(item["holding_hours"]) for item in closed_trades
        )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbols": selected_symbols,
            "closed_trade_count": closed_trade_count,
            "open_trade_count": len(open_trades),
            "partial_realized_net_pnl": partial_realized_net_pnl,
            "win_rate_pct": total_wins / closed_trade_count * 100.0
            if closed_trade_count
            else 0.0,
            "total_net_pnl": total_net_pnl,
            "avg_net_return_pct": (
                sum(self._safe_float(item["net_return_pct"]) for item in closed_trades)
                / closed_trade_count
                if closed_trade_count
                else 0.0
            ),
            "avg_holding_hours": (
                total_holding_hours / closed_trade_count if closed_trade_count else 0.0
            ),
            "symbol_summary": symbol_summary,
            "closed_trades": closed_trades,
            "open_trades": open_trades,
        }

    @staticmethod
    def render(data: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        lines = [
            text_for(lang, "# 交易池归因报告", "# Pool Attribution Report"),
            text_for(
                lang,
                f"- 交易池: {', '.join(data.get('symbols') or []) or '无'}",
                f"- Symbols: {', '.join(data.get('symbols') or []) or 'none'}",
            ),
            text_for(
                lang,
                f"- 已平仓交易: {data.get('closed_trade_count', 0)}",
                f"- Closed Trades: {data.get('closed_trade_count', 0)}",
            ),
            text_for(
                lang,
                f"- 未平仓交易: {data.get('open_trade_count', 0)}",
                f"- Open Trades: {data.get('open_trade_count', 0)}",
            ),
            text_for(
                lang,
                f"- 胜率: {PoolAttributionReporter._safe_float(data.get('win_rate_pct')):.2f}%",
                f"- Win Rate: {PoolAttributionReporter._safe_float(data.get('win_rate_pct')):.2f}%",
            ),
            text_for(
                lang,
                f"- 总净收益: ${PoolAttributionReporter._safe_float(data.get('total_net_pnl')):+.2f}",
                f"- Total Net PnL: ${PoolAttributionReporter._safe_float(data.get('total_net_pnl')):+.2f}",
            ),
            text_for(
                lang,
                f"- 未全平已实现净收益: ${PoolAttributionReporter._safe_float(data.get('partial_realized_net_pnl')):+.2f}",
                f"- Partial Realized Net PnL: ${PoolAttributionReporter._safe_float(data.get('partial_realized_net_pnl')):+.2f}",
            ),
            text_for(
                lang,
                f"- 平均净收益率: {PoolAttributionReporter._safe_float(data.get('avg_net_return_pct')):+.2f}%",
                f"- Avg Net Return: {PoolAttributionReporter._safe_float(data.get('avg_net_return_pct')):+.2f}%",
            ),
            text_for(
                lang,
                f"- 平均持有时长: {PoolAttributionReporter._safe_float(data.get('avg_holding_hours')):.2f}h",
                f"- Avg Holding Time: {PoolAttributionReporter._safe_float(data.get('avg_holding_hours')):.2f}h",
            ),
            "",
            text_for(lang, "## 标的汇总", "## Symbol Summary"),
        ]

        summary_rows = data.get("symbol_summary") or []
        if summary_rows:
            lines.extend(
                PoolAttributionReporter._render_table(
                    headers=[
                        text_for(lang, "标的", "Symbol"),
                        text_for(lang, "交易数", "Trades"),
                        text_for(lang, "开仓中", "Open"),
                        text_for(lang, "胜率", "Win Rate"),
                        text_for(lang, "均值收益", "Avg Return"),
                        text_for(lang, "总净收益", "Net PnL"),
                        text_for(lang, "未全平已实现", "Partial Realized"),
                        text_for(lang, "均值持有h", "Avg Hold h"),
                        text_for(lang, "主审动作", "Review"),
                        text_for(lang, "最新审查", "Latest Review"),
                        text_for(lang, "退出原因", "Exit Reasons"),
                    ],
                    rows=[
                        [
                            row.get("symbol", ""),
                            str(row.get("trade_count", 0)),
                            str(row.get("open_trade_count", 0)),
                            f"{PoolAttributionReporter._safe_float(row.get('win_rate_pct')):.2f}%",
                            f"{PoolAttributionReporter._safe_float(row.get('avg_net_return_pct')):+.2f}%",
                            f"${PoolAttributionReporter._safe_float(row.get('total_net_pnl')):+.2f}",
                            f"${PoolAttributionReporter._safe_float(row.get('partial_realized_net_pnl')):+.2f}",
                            f"{PoolAttributionReporter._safe_float(row.get('avg_holding_hours')):.2f}",
                            row.get("dominant_reviewed_action") or "-",
                            PoolAttributionReporter._latest_review_cell(row),
                            row.get("top_exit_reasons") or "-",
                        ]
                        for row in summary_rows
                    ],
                )
            )
        else:
            lines.append(text_for(lang, "- 无可归因交易", "- No attributable trades"))

        lines.extend(
            [
                "",
                text_for(lang, "## 已平仓交易明细", "## Closed Trades"),
            ]
        )
        closed_trades = data.get("closed_trades") or []
        if closed_trades:
            lines.extend(
                PoolAttributionReporter._render_table(
                    headers=[
                        text_for(lang, "标的", "Symbol"),
                        text_for(lang, "开仓时间", "Opened"),
                        text_for(lang, "平仓时间", "Closed"),
                        text_for(lang, "持有h", "Hold h"),
                        text_for(lang, "审查动作", "Review"),
                        text_for(lang, "分数", "Score"),
                        text_for(lang, "退出原因", "Exit"),
                        text_for(lang, "净收益率", "Net Ret"),
                        text_for(lang, "决策原因", "Decision Reason"),
                    ],
                    rows=[
                        [
                            row.get("symbol", ""),
                            row.get("opened_at", ""),
                            row.get("closed_at", ""),
                            f"{PoolAttributionReporter._safe_float(row.get('holding_hours')):.2f}",
                            row.get("reviewed_action") or "-",
                            PoolAttributionReporter._format_optional_float(
                                row.get("review_score"),
                                signed=False,
                            ),
                            ", ".join(row.get("exit_reasons") or []) or "-",
                            f"{PoolAttributionReporter._safe_float(row.get('net_return_pct')):+.2f}%",
                            row.get("decision_reason") or row.get("rationale") or "-",
                        ]
                        for row in closed_trades
                    ],
                )
            )
        else:
            lines.append(text_for(lang, "- 当前池还没有已平仓交易", "- No closed trades yet"))

        return "\n".join(lines)

    @staticmethod
    def _normalize_symbols(symbols: list[str] | None) -> list[str]:
        normalized = []
        seen = set()
        for item in symbols or []:
            symbol = str(item or "").strip()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(symbol)
        return normalized

    @staticmethod
    def _load_json(value: object) -> dict:
        if isinstance(value, dict):
            return dict(value)
        try:
            payload = json.loads(str(value or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _safe_float(value: object, default: float = 0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_iso(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @classmethod
    def _holding_hours(cls, trade: dict, close_ledgers: list[dict]) -> float:
        exit_time = cls._parse_iso(str(trade.get("exit_time") or ""))
        entry_time = cls._parse_iso(str(trade.get("entry_time") or ""))
        if entry_time and exit_time:
            return max(0.0, (exit_time - entry_time).total_seconds() / 3600.0)
        return max(
            (cls._safe_float(item.get("holding_hours")) for item in close_ledgers),
            default=0.0,
        )

    @classmethod
    def _extract_exit_reasons(cls, close_ledgers: list[dict]) -> list[str]:
        reasons: list[str] = []
        seen = set()
        for item in close_ledgers:
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                continue
            raw_reason = str(metadata.get("reason") or "")
            for part in raw_reason.split(","):
                reason = part.strip()
                if not reason or reason in seen:
                    continue
                seen.add(reason)
                reasons.append(reason)
        return reasons

    @staticmethod
    def _counter_top(counter: Counter | None) -> str:
        if not counter:
            return ""
        return str(counter.most_common(1)[0][0])

    @staticmethod
    def _format_counter(counter: Counter | None) -> str:
        if not counter:
            return ""
        return ", ".join(f"{key} x{count}" for key, count in counter.most_common(3))

    @staticmethod
    def _clean_text(value: object, limit: int = 96) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _render_table(*, headers: list[str], rows: list[list[str]]) -> list[str]:
        lines = [
            "| " + " | ".join(PoolAttributionReporter._table_cell(item) for item in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            lines.append(
                "| "
                + " | ".join(PoolAttributionReporter._table_cell(item) for item in row)
                + " |"
            )
        return lines

    @staticmethod
    def _table_cell(value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text.replace("|", "/") or "-"

    @staticmethod
    def _format_optional_float(value: object, signed: bool = True) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "-"
        return f"{number:+.2f}" if signed else f"{number:.2f}"

    @staticmethod
    def _latest_review_cell(row: dict) -> str:
        action = str(row.get("latest_reviewed_action") or "").strip()
        if not action:
            return "-"
        score = PoolAttributionReporter._format_optional_float(
            row.get("latest_review_score"),
            signed=False,
        )
        reasons = str(row.get("latest_review_reasons") or "").strip()
        if reasons:
            return PoolAttributionReporter._clean_text(f"{action} {score} {reasons}", limit=72)
        return PoolAttributionReporter._clean_text(f"{action} {score}", limit=72)
