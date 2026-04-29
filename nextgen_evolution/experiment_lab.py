"""Experiment lab that evaluates candidate strategies on real OHLCV windows."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .alpha_factory import StrategyPrimitive
from .data_feed import SQLiteOHLCVFeed
from .engine import NextGenEvolutionEngine
from .feature_miner import FeatureMiner, FeatureMiningResult, FeaturePoint
from .models import CapitalAllocation, ScoreCard, StrategyGenome, ValidationMetrics


@dataclass(slots=True)
class ExperimentResult:
    symbol: str
    timeframe: str
    scorecards: list[ScoreCard]
    promoted: list[ScoreCard]
    allocations: list[CapitalAllocation]
    candle_count: int
    metrics_by_strategy: dict[str, ValidationMetrics] = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    registry_run_id: int | None = None


class ExperimentLab:
    """Run candidate experiments against historical candles from the SQLite store."""

    def __init__(
        self,
        feed: SQLiteOHLCVFeed,
        engine: NextGenEvolutionEngine | None = None,
        feature_miner: FeatureMiner | None = None,
    ):
        self.feed = feed
        self.engine = engine or NextGenEvolutionEngine()
        self.feature_miner = feature_miner or FeatureMiner()

    def run_for_symbol(
        self,
        *,
        symbol: str,
        timeframe: str = "5m",
        total_capital: float = 10000.0,
        candle_limit: int = 1200,
        extra_genomes: list[StrategyGenome] | None = None,
        notes: dict | None = None,
    ) -> ExperimentResult:
        primitives = self.default_primitives()
        population = self._merge_genomes(
            self.engine.propose_population(primitives),
            extra_genomes or [],
        )
        return self.run_candidates_for_symbol(
            symbol=symbol,
            timeframe=timeframe,
            genomes=population,
            total_capital=total_capital,
            candle_limit=candle_limit,
            notes=notes,
        )

    def run_candidates_for_symbol(
        self,
        *,
        symbol: str,
        timeframe: str = "5m",
        genomes: list[StrategyGenome],
        total_capital: float = 10000.0,
        candle_limit: int = 1200,
        notes: dict | None = None,
    ) -> ExperimentResult:
        metrics_by_strategy: dict[str, ValidationMetrics] = {}
        family_notes: dict[str, dict[str, float | int]] = {}
        candles = self.feed.load_candles(symbol, timeframe, limit=candle_limit)
        mined = self.feature_miner.mine(candles)
        for genome in genomes:
            metrics, diagnostics = self._evaluate_candidate(
                genome.family,
                candles,
                mined,
                genome.params,
            )
            if metrics is None:
                continue
            metrics_by_strategy[genome.strategy_id] = metrics
            family_notes[genome.strategy_id] = diagnostics
        merged_notes = {
            "feature_summary": mined.summary,
            "candidate_diagnostics": family_notes,
        }
        merged_notes.update(notes or {})
        scorecards, promoted, allocations = self.engine.build_candidate_bundle(
            genomes=genomes,
            metrics_by_strategy=metrics_by_strategy,
            total_capital=total_capital,
        )
        return ExperimentResult(
            symbol=symbol,
            timeframe=timeframe,
            scorecards=scorecards,
            promoted=promoted,
            allocations=allocations,
            candle_count=len(candles),
            metrics_by_strategy=metrics_by_strategy,
            notes=merged_notes,
        )

    @staticmethod
    def _merge_genomes(
        base_genomes: list[StrategyGenome],
        extra_genomes: list[StrategyGenome],
    ) -> list[StrategyGenome]:
        merged: dict[str, StrategyGenome] = {
            genome.strategy_id: genome
            for genome in base_genomes
        }
        for genome in extra_genomes:
            merged[genome.strategy_id] = genome
        return list(merged.values())

    @staticmethod
    def default_primitives() -> list[StrategyPrimitive]:
        return [
            StrategyPrimitive(
                family="microtrend_breakout",
                base_params={"lookback": 18.0, "breakout_buffer": 0.002, "hold_bars": 6.0},
                tags=("momentum", "breakout"),
            ),
            StrategyPrimitive(
                family="trend_pullback_continuation",
                base_params={
                    "lookback": 18.0,
                    "momentum_floor": 0.0,
                    "breakout_slack": 0.008,
                    "pullback_z": 0.8,
                    "continuation_ceiling": 0.4,
                    "hold_bars": 6.0,
                    "cooldown_bars": 3.0,
                },
                tags=("trend", "pullback"),
            ),
            StrategyPrimitive(
                family="mean_revert_imbalance",
                base_params={"lookback": 24.0, "zscore": 1.5, "hold_bars": 6.0},
                tags=("reversion", "intraday"),
            ),
            StrategyPrimitive(
                family="volatility_reclaim",
                base_params={"lookback": 20.0, "shock": 0.008, "hold_bars": 8.0},
                tags=("reversal", "volatility"),
            ),
        ]

    def _evaluate_candidate(
        self,
        family: str,
        candles: list[dict],
        mined: FeatureMiningResult,
        params: dict[str, float],
    ) -> tuple[ValidationMetrics | None, dict[str, float | int]]:
        diagnostics: dict[str, float | int] = {
            "trades": 0,
            "signals": 0,
            "avg_trade_return_bps": 0.0,
        }
        if len(candles) < 80 or len(mined.points) < 24:
            return None, diagnostics
        returns, signal_count, avg_cost = self._simulate_returns(family, candles, mined, params)
        if len(returns) < 8:
            return None, diagnostics

        split = max(4, int(len(returns) * 0.65))
        backtest_chunk = returns[:split]
        walkforward_chunk = returns[split:]
        full_expectancy = self._expectancy_score(returns)
        walkforward_expectancy = self._blended_walkforward_expectancy(
            full_returns=returns,
            walkforward_returns=walkforward_chunk,
        )
        shadow_expectancy = round(walkforward_expectancy * 0.80, 4)
        live_expectancy = round(walkforward_expectancy * 0.60, 4)
        diagnostics = {
            "trades": len(returns),
            "signals": signal_count,
            "avg_trade_return_bps": round(self._average(returns) * 10000.0, 2),
            "avg_cost_bps": round(avg_cost * 10000.0, 2),
            "win_rate": round(self._win_rate(returns), 4),
            "profit_factor": round(self._profit_factor(returns), 4),
        }
        return ValidationMetrics(
            backtest_expectancy=full_expectancy,
            walkforward_expectancy=walkforward_expectancy,
            shadow_expectancy=shadow_expectancy,
            live_expectancy=live_expectancy,
            max_drawdown_pct=self._max_drawdown_pct(returns),
            trade_count=len(returns),
            cost_drag_pct=round(avg_cost * 100.0, 4),
            latency_ms=35.0 if family != "volatility_reclaim" else 55.0,
            regime_consistency=self._regime_consistency(backtest_chunk, walkforward_chunk),
        ), diagnostics

    def _simulate_returns(
        self,
        family: str,
        candles: list[dict],
        mined: FeatureMiningResult,
        params: dict[str, float],
    ) -> tuple[list[float], int, float]:
        closes = [float(c["close"]) for c in candles]
        lookback = max(5, int(round(params.get("lookback", 20.0))))
        hold_bars = max(1, int(round(params.get("hold_bars", 4.0))))
        point_map = {point.index: point for point in mined.points}
        returns: list[float] = []
        costs: list[float] = []
        signal_count = 0
        next_entry_index = 0
        cooldown_bars = max(1, int(round(params.get("cooldown_bars", 1.0))))

        for point in mined.points:
            idx = point.index
            if idx < lookback or idx + hold_bars >= len(closes) or idx < next_entry_index:
                continue
            entry = point.close
            if entry <= 0:
                continue

            signal = self._should_enter(family, point, params)
            if not signal:
                continue
            signal_count += 1
            gross_return, exit_offset = self._realize_trade_return(
                family=family,
                point=point,
                params=params,
                closes=closes,
                point_map=point_map,
                hold_bars=hold_bars,
            )
            trade_cost = self._estimate_cost(point)
            returns.append(gross_return - trade_cost)
            costs.append(trade_cost)
            next_entry_index = idx + max(exit_offset, cooldown_bars)
        return returns, signal_count, self._average(costs)

    def _should_enter(
        self,
        family: str,
        point: FeaturePoint,
        params: dict[str, float],
    ) -> bool:
        if family == "microtrend_breakout":
            breakout_buffer = params.get("breakout_buffer", 0.002)
            return (
                point.momentum_slope >= max(0.0012, breakout_buffer * 1.2)
                and point.breakout_pressure >= -(breakout_buffer * 0.10)
                and point.volume_impulse >= -0.15
                and point.range_expansion >= -0.25
                and point.range_expansion <= 0.95
                and point.mean_reversion_zscore >= -0.10
                and point.mean_reversion_zscore <= 2.2
                and point.wick_imbalance >= -0.25
                and point.regime_consistency >= 0.50
            )
        if family == "trend_pullback_continuation":
            return (
                point.momentum_slope >= params.get("momentum_floor", 0.0002)
                and point.breakout_pressure >= -params.get("breakout_slack", 0.008)
                and point.mean_reversion_zscore >= -params.get("pullback_z", 0.8)
                and point.mean_reversion_zscore <= params.get("continuation_ceiling", 0.4)
                and point.wick_imbalance >= 0.0
                and point.reclaim_score >= 0.0
                and point.range_expansion >= -0.40
                and point.range_expansion <= 1.0
                and point.volume_impulse >= -0.10
                and point.regime_consistency >= 0.50
            )
        if family == "mean_revert_imbalance":
            zscore = params.get("zscore", 1.6)
            return (
                point.mean_reversion_zscore <= -max(0.8, zscore * 0.55)
                and point.reclaim_score >= 0.05
                and point.wick_imbalance >= 0.08
                and point.range_expansion >= -0.30
                and point.range_expansion <= 0.20
                and point.volume_impulse >= -0.20
                and point.regime_consistency <= 0.90
            )
        if family == "volatility_reclaim":
            shock = params.get("shock", 0.008)
            return (
                point.range_expansion >= max(0.18, shock * 18.0)
                and point.wick_imbalance >= 0.0
                and point.reclaim_score >= 0.15
                and point.bar_return >= -(shock * 2.0)
                and point.volume_impulse >= -0.25
            )
        return False

    def _realize_trade_return(
        self,
        *,
        family: str,
        point: FeaturePoint,
        params: dict[str, float],
        closes: list[float],
        point_map: dict[int, FeaturePoint],
        hold_bars: int,
    ) -> tuple[float, int]:
        entry = point.close
        if entry <= 0:
            return 0.0, 1

        if family == "microtrend_breakout":
            stop = max(0.0025, point.realized_volatility * 1.4)
            target = max(0.0035, stop * 1.6, max(point.breakout_pressure, 0.0) * 0.8)
            for offset in range(1, hold_bars + 1):
                gross_return = (closes[point.index + offset] / entry) - 1.0
                future_point = point_map.get(point.index + offset)
                if gross_return <= -stop or gross_return >= target:
                    return gross_return, offset
                if future_point and (
                    future_point.momentum_slope < 0
                    or future_point.breakout_pressure < 0
                    or future_point.volume_impulse < -0.15
                ):
                    return gross_return, offset
            return (closes[point.index + hold_bars] / entry) - 1.0, hold_bars

        if family == "trend_pullback_continuation":
            stop = max(0.0030, point.realized_volatility * 1.8)
            target = max(
                0.0045,
                max(point.momentum_slope, 0.0) * 3.5 + max(point.breakout_pressure, 0.0) * 0.6,
            )
            for offset in range(1, hold_bars + 1):
                gross_return = (closes[point.index + offset] / entry) - 1.0
                future_point = point_map.get(point.index + offset)
                if gross_return <= -stop or gross_return >= target:
                    return gross_return, offset
                if offset >= 2 and future_point and (
                    future_point.momentum_slope < 0
                    or future_point.reclaim_score < -0.10
                    or future_point.wick_imbalance < -0.20
                ):
                    return gross_return, offset
            return (closes[point.index + hold_bars] / entry) - 1.0, hold_bars

        if family == "mean_revert_imbalance":
            stop = max(0.0035, point.realized_volatility * 2.4)
            target = max(
                0.0040,
                abs(point.mean_reversion_zscore) * 0.0016 + max(point.reclaim_score, 0.0) * 0.003,
            )
            for offset in range(1, hold_bars + 1):
                gross_return = (closes[point.index + offset] / entry) - 1.0
                future_point = point_map.get(point.index + offset)
                if gross_return <= -stop or gross_return >= target:
                    return gross_return, offset
                if offset >= 2 and future_point and future_point.mean_reversion_zscore >= 0.15:
                    return gross_return, offset
                if offset >= 2 and future_point and (
                    future_point.wick_imbalance < -0.20
                    and future_point.bar_return < 0
                ):
                    return gross_return, offset
            return (closes[point.index + hold_bars] / entry) - 1.0, hold_bars

        if family == "volatility_reclaim":
            stop = max(0.0040, point.realized_volatility * 2.0)
            target = max(0.0045, point.range_expansion * 0.003 + max(point.reclaim_score, 0.0) * 0.002)
            for offset in range(1, hold_bars + 1):
                gross_return = (closes[point.index + offset] / entry) - 1.0
                future_point = point_map.get(point.index + offset)
                if gross_return <= -stop or gross_return >= target:
                    return gross_return, offset
                if future_point and future_point.wick_imbalance < -0.15 and future_point.bar_return < 0:
                    return gross_return, offset
            return (closes[point.index + hold_bars] / entry) - 1.0, hold_bars

        return (closes[point.index + hold_bars] / entry) - 1.0, hold_bars

    @staticmethod
    def _estimate_cost(point: FeaturePoint) -> float:
        spread_cost = 0.0008
        volatility_slippage = point.realized_volatility * 0.20
        impulse_slippage = max(0.0, point.range_expansion) * 0.0002
        return spread_cost + volatility_slippage + impulse_slippage

    def _expectancy_score(self, returns: list[float]) -> float:
        if not returns:
            return 0.0
        average_return = self._average(returns)
        win_rate = self._win_rate(returns)
        payoff_ratio = self._payoff_ratio(returns)
        profit_factor = self._profit_factor(returns)
        expectancy = (
            average_return * 120.0
            + (win_rate - 0.5) * 0.20
            + (payoff_ratio - 1.0) * 0.18
            + (profit_factor - 1.0) * 0.12
        )
        return round(max(-1.0, min(1.0, expectancy)), 4)

    def _blended_walkforward_expectancy(
        self,
        *,
        full_returns: list[float],
        walkforward_returns: list[float],
    ) -> float:
        if not walkforward_returns:
            return 0.0
        full_expectancy = self._expectancy_score(full_returns)
        raw_walkforward = self._expectancy_score(walkforward_returns)
        min_confidence_trades = max(8, self.engine.config.min_trade_count)
        walkforward_weight = min(1.0, len(walkforward_returns) / float(min_confidence_trades))
        blended = raw_walkforward * walkforward_weight + full_expectancy * (1.0 - walkforward_weight)
        return round(blended, 4)

    @staticmethod
    def _win_rate(returns: list[float]) -> float:
        if not returns:
            return 0.0
        return sum(1 for item in returns if item > 0) / len(returns)

    @staticmethod
    def _payoff_ratio(returns: list[float]) -> float:
        wins = [item for item in returns if item > 0]
        losses = [abs(item) for item in returns if item < 0]
        if not wins and not losses:
            return 0.0
        if not losses:
            return 2.0 if wins else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses)
        if avg_loss <= 0:
            return 0.0
        return avg_win / avg_loss

    @staticmethod
    def _profit_factor(returns: list[float]) -> float:
        gross_profit = sum(item for item in returns if item > 0)
        gross_loss = abs(sum(item for item in returns if item < 0))
        if gross_loss <= 0:
            return 2.0 if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def _average(values: list[float]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)

    @staticmethod
    def _stddev(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(max(variance, 0.0))

    @staticmethod
    def _max_drawdown_pct(returns: list[float]) -> float:
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for value in returns:
            equity *= 1.0 + value
            peak = max(peak, equity)
            if peak > 0:
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
        return round(max_drawdown * 100.0, 4)

    @staticmethod
    def _regime_consistency(backtest_chunk: list[float], walkforward_chunk: list[float]) -> float:
        if not backtest_chunk or not walkforward_chunk:
            return 0.0
        train = sum(1 for item in backtest_chunk if item > 0) / len(backtest_chunk)
        test = sum(1 for item in walkforward_chunk if item > 0) / len(walkforward_chunk)
        return round(max(0.0, 1.0 - abs(train - test)), 4)
