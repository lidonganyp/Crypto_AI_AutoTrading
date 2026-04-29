"""Lightweight scheduler for next-generation experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .experiment_lab import ExperimentLab, ExperimentResult


@dataclass(slots=True)
class ScheduledExperiment:
    symbol: str
    timeframe: str
    candle_limit: int = 1200
    total_capital: float = 10000.0


class ExperimentScheduler:
    """Run batches of symbol/timeframe experiments."""

    def __init__(self, lab: ExperimentLab):
        self.lab = lab

    def default_jobs(
        self,
        *,
        timeframe: str = "5m",
        min_rows: int = 200,
        max_symbols: int = 4,
    ) -> list[ScheduledExperiment]:
        symbols = self.lab.feed.list_symbols(timeframe, min_rows=min_rows)[:max_symbols]
        return [ScheduledExperiment(symbol=symbol, timeframe=timeframe) for symbol in symbols]

    def run(self, jobs: list[ScheduledExperiment]) -> list[ExperimentResult]:
        return [
            self.lab.run_for_symbol(
                symbol=job.symbol,
                timeframe=job.timeframe,
                total_capital=job.total_capital,
                candle_limit=job.candle_limit,
            )
            for job in jobs
        ]
