"""Scheduling-friendly portfolio refresh runner."""

from __future__ import annotations

from datetime import datetime, timezone
import time

from .config import EvolutionConfig
from .models import PortfolioPerformanceSnapshot
from .portfolio_alerts import PortfolioAlertEvaluator
from .portfolio_tracker import PortfolioTracker
from .promotion_registry import PromotionRegistry


class PortfolioMonitor:
    """Run periodic mark-to-market refresh cycles for portfolio runs."""

    def __init__(
        self,
        tracker: PortfolioTracker,
        registry: PromotionRegistry,
        config: EvolutionConfig | None = None,
    ):
        self.tracker = tracker
        self.registry = registry
        self.config = config or EvolutionConfig()
        self.alerts = PortfolioAlertEvaluator(self.config)

    def refresh_latest(self) -> PortfolioPerformanceSnapshot | None:
        latest = self.registry.latest_portfolio_run()
        if latest is None:
            return None
        return self.tracker.refresh(int(latest["id"]))

    def status(
        self,
        *,
        portfolio_run_id: int | None = None,
        stale_after_minutes: int | None = None,
    ) -> dict | None:
        stale_after_minutes = (
            self.config.stale_snapshot_after_minutes
            if stale_after_minutes is None
            else stale_after_minutes
        )
        target_run = None
        if portfolio_run_id is None:
            target_run = self.registry.latest_portfolio_run()
        else:
            target_run = self.registry.portfolio_run(int(portfolio_run_id))
        if target_run is None:
            return None

        run_id = int(target_run["id"])
        allocations = self.registry.portfolio_allocations(run_id)
        snapshots = self.registry.portfolio_snapshots(run_id, limit=1)
        latest_snapshot = snapshots[0] if snapshots else None
        latest_snapshot_at = None
        snapshot_age_seconds = None
        freshness = "missing"
        if latest_snapshot is not None:
            latest_snapshot_at = str(latest_snapshot["created_at"])
            snapshot_dt = datetime.fromisoformat(latest_snapshot_at)
            if snapshot_dt.tzinfo is None:
                snapshot_dt = snapshot_dt.replace(tzinfo=timezone.utc)
            snapshot_age_seconds = max(
                0.0,
                (datetime.now(timezone.utc) - snapshot_dt.astimezone(timezone.utc)).total_seconds(),
            )
            freshness = (
                "fresh"
                if snapshot_age_seconds <= max(1, stale_after_minutes * 60)
                else "stale"
            )

        status = {
            "portfolio_run_id": run_id,
            "status": str(target_run.get("status", "unknown")),
            "freshness": freshness,
            "stale_after_minutes": stale_after_minutes,
            "snapshot_count": len(self.registry.portfolio_snapshots(run_id, limit=1000)),
            "latest_snapshot_at": latest_snapshot_at,
            "snapshot_age_seconds": round(snapshot_age_seconds, 2) if snapshot_age_seconds is not None else None,
            "total_capital": float(target_run["total_capital"]),
            "allocated_capital": float(target_run["allocated_capital"]),
            "reserve_capital": float(target_run["reserve_capital"]),
            "equity": float(target_run.get("latest_equity", 0.0) or 0.0),
            "realized_pnl": float(target_run.get("latest_realized_pnl", 0.0) or 0.0),
            "unrealized_pnl": float(target_run.get("latest_unrealized_pnl", 0.0) or 0.0),
            "gross_exposure": float(target_run.get("latest_gross_exposure", 0.0) or 0.0),
            "net_exposure": float(target_run.get("latest_net_exposure", 0.0) or 0.0),
            "max_drawdown_pct": float(target_run.get("latest_max_drawdown_pct", 0.0) or 0.0),
            "open_positions": int(target_run.get("latest_open_positions", 0) or 0),
            "closed_positions": int(target_run.get("latest_closed_positions", 0) or 0),
            "allocation_count": len(allocations),
            "symbols": sorted({str(item["symbol"]) for item in allocations}),
        }
        alerts = self.alerts.evaluate(status)
        status["health"] = "healthy" if not alerts else "alerting"
        status["alerts"] = [
            {
                "code": alert.code,
                "severity": alert.severity,
                "message": alert.message,
            }
            for alert in alerts
        ]
        return status

    def run_cycles(
        self,
        *,
        portfolio_run_id: int | None = None,
        cycles: int = 1,
        interval_seconds: float = 0.0,
    ) -> list[PortfolioPerformanceSnapshot]:
        snapshots: list[PortfolioPerformanceSnapshot] = []
        target_run_id = portfolio_run_id
        if target_run_id is None:
            latest = self.registry.latest_portfolio_run()
            if latest is None:
                return snapshots
            target_run_id = int(latest["id"])

        for index in range(max(1, cycles)):
            snapshot = self.tracker.refresh(int(target_run_id))
            if snapshot is not None:
                snapshots.append(snapshot)
            if index < cycles - 1 and interval_seconds > 0:
                time.sleep(interval_seconds)
        return snapshots
