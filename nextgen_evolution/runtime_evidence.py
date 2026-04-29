"""Aggregate runtime evidence from actual paper/live execution records."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .config import EvolutionConfig
from .experiment_lab import ExperimentResult
from .models import RuntimeEvidenceSnapshot, RuntimeState


class RuntimeEvidenceCollector:
    """Build runtime evidence from trades, positions, and PnL ledger rows."""

    MANAGED_SOURCE = "nextgen_autonomy"

    def __init__(self, feed, config: EvolutionConfig | None = None):
        self.feed = feed
        self.storage = feed.storage
        self.config = config or EvolutionConfig()

    @staticmethod
    def runtime_id(symbol: str, timeframe: str, strategy_id: str) -> str:
        return f"{symbol}|{timeframe}|{strategy_id}"

    @staticmethod
    def parse_runtime_id(runtime_id: str) -> tuple[str, str, str]:
        parts = str(runtime_id).split("|", 2)
        if len(parts) != 3:
            return "", "", str(runtime_id)
        return parts[0], parts[1], parts[2]

    def collect_for_results(
        self,
        results: list[ExperimentResult],
        *,
        previous_states: list[RuntimeState] | None = None,
    ) -> dict[str, dict]:
        runtime_ids = [
            self.runtime_id(result.symbol, result.timeframe, card.genome.strategy_id)
            for result in results
            for card in result.scorecards
        ]
        snapshots = self.collect(runtime_ids, previous_states=previous_states)
        return {
            key: {
                "current_capital": value.current_capital,
                "realized_pnl": value.realized_pnl,
                "unrealized_pnl": value.unrealized_pnl,
                "current_drawdown_pct": value.current_drawdown_pct,
                "consecutive_losses": value.consecutive_losses,
                "health_status": value.health_status,
                "notes": {
                    **dict(value.notes or {}),
                    "closed_trade_count": value.closed_trade_count,
                    "win_rate": value.win_rate,
                    "max_drawdown_pct": value.max_drawdown_pct,
                    "total_net_pnl": value.total_net_pnl,
                    "holding_minutes": value.notes.get("holding_minutes", 0.0),
                    "timeframe_minutes": value.notes.get("timeframe_minutes", 0.0),
                    "holding_bars": value.notes.get("holding_bars", 0.0),
                    "peak_unrealized_pnl": value.notes.get("peak_unrealized_pnl", 0.0),
                    "peak_unrealized_return_pct": value.notes.get("peak_unrealized_return_pct", 0.0),
                    "profit_retrace_pct": value.notes.get("profit_retrace_pct", 0.0),
                },
            }
            for key, value in snapshots.items()
        }

    def collect(
        self,
        runtime_ids: list[str],
        *,
        previous_states: list[RuntimeState] | None = None,
    ) -> dict[str, RuntimeEvidenceSnapshot]:
        unique_ids = list(dict.fromkeys(str(item) for item in runtime_ids if str(item).strip()))
        state_index = {item.runtime_id: item for item in (previous_states or [])}
        ledger_index: dict[str, list[dict]] = {item: [] for item in unique_ids}
        open_trade_index: dict[str, dict] = {}
        closed_trade_index: dict[str, list[dict]] = {item: [] for item in unique_ids}

        for row in self.storage.get_pnl_ledger(limit=5000):
            metadata = self._row_metadata(row)
            if metadata.get("source") != self.MANAGED_SOURCE:
                continue
            runtime_id = str(metadata.get("runtime_id") or "")
            if runtime_id in ledger_index:
                ledger_index[runtime_id].append(row)

        for row in self.storage.get_open_trades():
            metadata = self._row_metadata(row)
            if metadata.get("source") != self.MANAGED_SOURCE:
                continue
            runtime_id = str(metadata.get("runtime_id") or "")
            if runtime_id in ledger_index:
                open_trade_index[runtime_id] = row

        for row in self.storage.get_closed_trades():
            metadata = self._row_metadata(row)
            if metadata.get("source") != self.MANAGED_SOURCE:
                continue
            runtime_id = str(metadata.get("runtime_id") or "")
            if runtime_id in closed_trade_index:
                closed_trade_index[runtime_id].append(row)

        positions_by_symbol = {
            str(item["symbol"]): item
            for item in self.storage.get_positions()
        }

        evidence: dict[str, RuntimeEvidenceSnapshot] = {}
        for runtime_id in unique_ids:
            symbol, timeframe, strategy_id = self.parse_runtime_id(runtime_id)
            previous = state_index.get(runtime_id)
            rows = sorted(
                ledger_index.get(runtime_id, []),
                key=lambda item: (str(item.get("event_time") or ""), int(item.get("id") or 0)),
            )
            open_trade = open_trade_index.get(runtime_id)
            closed_trades = sorted(
                closed_trade_index.get(runtime_id, []),
                key=lambda item: str(item.get("exit_time") or ""),
                reverse=True,
            )
            position = positions_by_symbol.get(symbol) if open_trade is not None else None
            current_capital = 0.0
            unrealized_pnl = 0.0
            open_position = False
            family = ""
            notes: dict = {}
            if open_trade is not None:
                open_metadata = self._row_metadata(open_trade)
                family = str(open_metadata.get("family") or "")
                price = self._latest_price(symbol, timeframe or str(open_metadata.get("timeframe") or "5m"))
                entry_price = float(open_trade.get("entry_price") or 0.0)
                quantity = float(open_trade.get("quantity") or 0.0)
                entry_time = str(open_trade.get("entry_time") or "")
                timeframe_text = timeframe or str(open_metadata.get("timeframe") or "5m")
                timeframe_minutes = self._timeframe_minutes(timeframe_text)
                holding_minutes = self._holding_minutes(entry_time)
                if position is not None:
                    quantity = float(position.get("quantity") or quantity)
                if price is not None and quantity > 0:
                    current_capital = round(price * quantity, 2)
                    unrealized_pnl = round((price - entry_price) * quantity, 2)
                    open_position = True
                    notes["latest_price"] = price
                    notes["entry_price"] = entry_price
                    notes["open_quantity"] = quantity
                notes["entry_time"] = entry_time
                notes["timeframe_minutes"] = timeframe_minutes
                notes["holding_minutes"] = round(holding_minutes, 4)
                notes["holding_bars"] = (
                    round(holding_minutes / timeframe_minutes, 4)
                    if timeframe_minutes > 0
                    else 0.0
                )
                peak_price = self._peak_price_since_entry(
                    symbol=symbol,
                    timeframe=timeframe_text,
                    entry_time=entry_time,
                )
                entry_notional = max(entry_price * quantity, 0.0)
                peak_unrealized_pnl = (
                    round(max((peak_price - entry_price) * quantity, 0.0), 2)
                    if peak_price is not None and quantity > 0
                    else 0.0
                )
                notes["peak_price"] = float(peak_price or 0.0)
                notes["entry_notional"] = round(entry_notional, 2)
                notes["peak_unrealized_pnl"] = peak_unrealized_pnl
                notes["peak_unrealized_return_pct"] = (
                    round(peak_unrealized_pnl / entry_notional * 100.0, 4)
                    if entry_notional > 0
                    else 0.0
                )
                notes["profit_retrace_pct"] = (
                    round(
                        max(peak_unrealized_pnl - max(unrealized_pnl, 0.0), 0.0)
                        / peak_unrealized_pnl
                        * 100.0,
                        4,
                    )
                    if peak_unrealized_pnl > 0
                    else 0.0
                )
            elif previous is not None:
                family = previous.family

            cumulative = 0.0
            peak = 0.0
            realized_pnl = 0.0
            max_notional = max(
                float(previous.desired_capital) if previous is not None else 0.0,
                float(previous.current_capital) if previous is not None else 0.0,
                current_capital,
            )
            for row in rows:
                metadata = self._row_metadata(row)
                family = family or str(metadata.get("family") or "")
                event_net_pnl = float(row.get("net_pnl") or 0.0)
                cumulative += event_net_pnl
                if str(row.get("event_type") or "") == "close":
                    realized_pnl += event_net_pnl
                peak = max(peak, cumulative)
                max_notional = max(max_notional, float(row.get("notional_value") or 0.0))
            total_net_pnl = round(cumulative + unrealized_pnl, 2)
            denominator = max(max_notional, 1e-9)
            current_drawdown_pct = 0.0
            if total_net_pnl < peak:
                current_drawdown_pct = round((peak - total_net_pnl) / denominator * 100.0, 4)
            max_drawdown_pct = 0.0
            rolling = 0.0
            rolling_peak = 0.0
            for row in rows:
                rolling += float(row.get("net_pnl") or 0.0)
                rolling_peak = max(rolling_peak, rolling)
                max_drawdown_pct = max(
                    max_drawdown_pct,
                    (rolling_peak - rolling) / denominator * 100.0,
                )
            max_drawdown_pct = round(max(max_drawdown_pct, current_drawdown_pct), 4)

            closed_trade_count = len(closed_trades)
            win_count = sum(1 for item in closed_trades if float(item.get("pnl") or 0.0) > 0)
            consecutive_losses = 0
            for item in closed_trades:
                if float(item.get("pnl") or 0.0) < 0:
                    consecutive_losses += 1
                else:
                    break
            win_rate = round(win_count / closed_trade_count, 4) if closed_trade_count else 0.0
            health_status = self._health_status(
                current_drawdown_pct=current_drawdown_pct,
                closed_trade_count=closed_trade_count,
                consecutive_losses=consecutive_losses,
                total_net_pnl=total_net_pnl,
                open_position=open_position,
            )
            if closed_trades:
                notes["last_closed_trade_id"] = str(closed_trades[0].get("id") or "")
                notes["last_closed_pnl"] = float(closed_trades[0].get("pnl") or 0.0)
            evidence[runtime_id] = RuntimeEvidenceSnapshot(
                runtime_id=runtime_id,
                symbol=symbol,
                timeframe=timeframe,
                strategy_id=strategy_id,
                family=family,
                open_position=open_position,
                current_capital=round(current_capital, 2),
                realized_pnl=round(realized_pnl, 2),
                unrealized_pnl=round(unrealized_pnl, 2),
                total_net_pnl=total_net_pnl,
                current_drawdown_pct=current_drawdown_pct,
                max_drawdown_pct=max_drawdown_pct,
                closed_trade_count=closed_trade_count,
                win_rate=win_rate,
                consecutive_losses=consecutive_losses,
                health_status=health_status,
                notes=notes,
            )
        return evidence

    def _health_status(
        self,
        *,
        current_drawdown_pct: float,
        closed_trade_count: int,
        consecutive_losses: int,
        total_net_pnl: float,
        open_position: bool,
    ) -> str:
        if current_drawdown_pct >= self.config.autonomy_repair_drawdown_pct:
            return "failing"
        if consecutive_losses >= 3:
            return "failing"
        if (
            current_drawdown_pct >= self.config.autonomy_repair_drawdown_pct * 0.60
            or consecutive_losses >= 2
            or (closed_trade_count >= 2 and total_net_pnl < 0)
        ):
            return "degraded"
        if open_position or closed_trade_count > 0:
            return "active"
        return "unproven"

    @staticmethod
    def _row_metadata(row: dict | None) -> dict:
        if not row:
            return {}
        raw = row.get("metadata_json")
        if isinstance(raw, dict):
            return dict(raw)
        try:
            payload = json.loads(raw or "{}")
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def _latest_price(self, symbol: str, timeframe: str) -> float | None:
        candles = self.feed.load_candles(symbol, timeframe, limit=1)
        if not candles:
            return None
        return float(candles[-1]["close"])

    def _peak_price_since_entry(
        self,
        *,
        symbol: str,
        timeframe: str,
        entry_time: str,
    ) -> float | None:
        candles = self.feed.load_candles(symbol, timeframe, limit=5000)
        if not candles:
            return None
        entry_ts = self._entry_timestamp_ms(entry_time)
        filtered = candles
        if entry_ts is not None:
            filtered = [
                item
                for item in candles
                if int(item.get("timestamp") or 0) >= entry_ts
            ]
        if not filtered:
            filtered = candles[-1:]
        peak = max(float(item.get("high") or item.get("close") or 0.0) for item in filtered)
        return peak if peak > 0 else None

    @staticmethod
    def _holding_minutes(entry_time: str) -> float:
        text = str(entry_time or "").strip()
        if not text:
            return 0.0
        try:
            opened_at = datetime.fromisoformat(text)
        except ValueError:
            return 0.0
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (now - opened_at.astimezone(timezone.utc)).total_seconds() / 60.0)

    @staticmethod
    def _timeframe_minutes(timeframe: str) -> float:
        text = str(timeframe or "").strip().lower()
        if len(text) < 2:
            return 0.0
        unit = text[-1]
        try:
            value = float(text[:-1])
        except ValueError:
            return 0.0
        if unit == "m":
            return value
        if unit == "h":
            return value * 60.0
        if unit == "d":
            return value * 1440.0
        return 0.0

    @staticmethod
    def _entry_timestamp_ms(entry_time: str) -> int | None:
        text = str(entry_time or "").strip()
        if not text:
            return None
        try:
            opened_at = datetime.fromisoformat(text)
        except ValueError:
            return None
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        return int(opened_at.astimezone(timezone.utc).timestamp() * 1000)
