"""Portfolio health alert evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from .config import EvolutionConfig


@dataclass(slots=True)
class PortfolioAlert:
    """A concrete health alert for a portfolio run."""

    code: str
    severity: str
    message: str


class PortfolioAlertEvaluator:
    """Apply lightweight operational guardrails to portfolio status snapshots."""

    def __init__(self, config: EvolutionConfig | None = None):
        self.config = config or EvolutionConfig()

    def evaluate(self, status: dict) -> list[PortfolioAlert]:
        alerts: list[PortfolioAlert] = []
        total_capital = float(status.get("total_capital", 0.0) or 0.0)
        equity = float(status.get("equity", 0.0) or 0.0)
        gross_exposure = float(status.get("gross_exposure", 0.0) or 0.0)
        max_drawdown_pct = float(status.get("max_drawdown_pct", 0.0) or 0.0)
        stale_after_minutes = int(
            status.get("stale_after_minutes", self.config.stale_snapshot_after_minutes) or 0
        )

        if status.get("freshness") == "missing":
            alerts.append(
                PortfolioAlert(
                    code="missing_snapshot",
                    severity="critical",
                    message="No portfolio snapshot is available yet.",
                )
            )
        elif status.get("freshness") == "stale":
            alerts.append(
                PortfolioAlert(
                    code="stale_snapshot",
                    severity="warning",
                    message=(
                        "Latest portfolio snapshot is older than "
                        f"{stale_after_minutes} minutes."
                    ),
                )
            )

        if max_drawdown_pct >= self.config.health_max_drawdown_pct:
            alerts.append(
                PortfolioAlert(
                    code="drawdown_limit",
                    severity="critical",
                    message=(
                        f"Portfolio drawdown {round(max_drawdown_pct, 2)}% exceeds "
                        f"limit {self.config.health_max_drawdown_pct}%."
                    ),
                )
            )

        if total_capital > 0:
            minimum_equity = total_capital * self.config.health_min_equity_ratio
            if equity < minimum_equity:
                alerts.append(
                    PortfolioAlert(
                        code="equity_floor",
                        severity="critical",
                        message=(
                            f"Portfolio equity {round(equity, 2)} is below floor "
                            f"{round(minimum_equity, 2)}."
                        ),
                    )
                )

            max_gross_exposure = total_capital * self.config.health_max_gross_exposure_ratio
            if gross_exposure > max_gross_exposure:
                alerts.append(
                    PortfolioAlert(
                        code="gross_exposure_limit",
                        severity="warning",
                        message=(
                            f"Gross exposure {round(gross_exposure, 2)} exceeds "
                            f"limit {round(max_gross_exposure, 2)}."
                        ),
                    )
                )

        return alerts
