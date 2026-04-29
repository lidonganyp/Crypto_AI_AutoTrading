"""Mark-to-market updater for portfolio runs."""

from __future__ import annotations

from .data_feed import SQLiteOHLCVFeed
from .models import PortfolioPerformanceSnapshot
from .promotion_registry import PromotionRegistry


class PortfolioTracker:
    """Refresh portfolio run performance from the latest OHLCV closes."""

    def __init__(self, feed: SQLiteOHLCVFeed, registry: PromotionRegistry):
        self.feed = feed
        self.registry = registry

    def latest_close(self, symbol: str, timeframe: str) -> float | None:
        candles = self.feed.load_candles(symbol, timeframe, limit=1)
        if not candles:
            return None
        return float(candles[-1]["close"])

    def refresh(self, portfolio_run_id: int) -> PortfolioPerformanceSnapshot | None:
        run = self.registry.portfolio_run(portfolio_run_id)
        allocations = self.registry.portfolio_allocations(portfolio_run_id)
        if run is None or not allocations:
            return None

        marks: list[dict] = []
        realized_pnl = float(run.get("latest_realized_pnl", 0.0) or 0.0)
        total_unrealized = 0.0
        gross_exposure = 0.0
        net_exposure = 0.0
        positive_positions = 0
        open_positions = 0

        for allocation in allocations:
            entry_price = float(allocation.get("entry_price", 0.0) or 0.0)
            timeframe = str(allocation.get("timeframe") or "5m")
            latest_price = self.latest_close(str(allocation["symbol"]), timeframe)
            if latest_price is None:
                continue

            capital = float(allocation["allocated_capital"])
            unrealized_pnl = 0.0
            if entry_price > 0:
                unrealized_pnl = round(capital * ((latest_price / entry_price) - 1.0), 2)
            marks.append(
                {
                    "portfolio_allocation_id": int(allocation["id"]),
                    "mark_price": latest_price,
                    "realized_pnl": float(allocation.get("realized_pnl", 0.0) or 0.0),
                    "unrealized_pnl": unrealized_pnl,
                }
            )
            total_unrealized += unrealized_pnl
            gross_exposure += abs(capital)
            net_exposure += capital
            open_positions += 1
            if unrealized_pnl > 0:
                positive_positions += 1

        self.registry.update_portfolio_allocation_marks(marks)
        total_capital = float(run["total_capital"])
        equity = round(total_capital + realized_pnl + total_unrealized, 2)
        latest_drawdown = max(
            float(run.get("latest_max_drawdown_pct", 0.0) or 0.0),
            round(max(0.0, (total_capital - equity) / max(total_capital, 1e-9) * 100.0), 4),
        )
        snapshot = PortfolioPerformanceSnapshot(
            realized_pnl=round(realized_pnl, 2),
            unrealized_pnl=round(total_unrealized, 2),
            equity=equity,
            gross_exposure=round(gross_exposure, 2),
            net_exposure=round(net_exposure, 2),
            open_positions=open_positions,
            closed_positions=int(run.get("latest_closed_positions", 0) or 0),
            win_rate=round(positive_positions / open_positions, 4) if open_positions else 0.0,
            max_drawdown_pct=latest_drawdown,
            status="active" if open_positions else "flat",
            notes={"source": "mark_to_market_refresh"},
        )
        self.registry.persist_portfolio_snapshot(portfolio_run_id, snapshot)
        return snapshot
