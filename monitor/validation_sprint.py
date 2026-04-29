"""Accelerated validation workflow for narrowing symbols and thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from config import Settings
from core.i18n import get_default_language, normalize_language, text_for
from core.scoring import objective_score_from_metrics
from core.storage import Storage
from backtest.v2_engine import V2BacktestEngine
from backtest.walkforward import WalkForwardBacktester


@dataclass
class ValidationCandidate:
    symbol: str
    xgboost_threshold: float
    final_score_threshold: float
    min_liquidity_ratio: float
    total_trades: int
    total_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    sharpe_like: float
    score: float


class ValidationSprintService:
    """Run a compact validation cycle that is faster than waiting for live samples."""

    DEFAULT_CANDIDATES = (
        (0.60, 0.44, 0.30),
        (0.64, 0.48, 0.50),
        (0.66, 0.48, 0.50),
        (0.68, 0.52, 0.50),
    )
    DEFAULT_WALKFORWARD_WINDOW_DAYS = 10

    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings

    def run(self, symbols: list[str]) -> dict:
        normalized_symbols = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
        baseline: list[dict] = []
        candidates: list[ValidationCandidate] = []

        for symbol in normalized_symbols:
            baseline_backtest = self._run_backtest(symbol, self.settings)
            baseline_walkforward = self._run_walkforward(symbol, self.settings)
            baseline.append(
                {
                    "symbol": symbol,
                    "backtest": baseline_backtest["summary"],
                    "walkforward": baseline_walkforward["summary"],
                }
            )
            for candidate in self._scan_symbol(symbol):
                candidates.append(candidate)

        top_candidates = sorted(
            candidates,
            key=lambda item: (item.score, item.total_return_pct, item.profit_factor),
            reverse=True,
        )[:6]
        return {
            "symbols": normalized_symbols,
            "baseline": baseline,
            "top_candidates": [candidate.__dict__ for candidate in top_candidates],
            "scan_count": len(candidates),
            "walkforward_window_days": self.DEFAULT_WALKFORWARD_WINDOW_DAYS,
        }

    def render(self, result: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        lines = [
            text_for(lang, "# 快速验证报告", "# Validation Sprint Report"),
            text_for(lang, f"- 验证币种: {', '.join(result['symbols']) or '无'}", f"- Symbols: {', '.join(result['symbols']) or 'none'}"),
            text_for(lang, f"- 扫描组合数: {result['scan_count']}", f"- Candidate Scans: {result['scan_count']}"),
            text_for(
                lang,
                f"- Walk-Forward 窗口: {result['walkforward_window_days']} 天",
                f"- Walk-Forward Window: {result['walkforward_window_days']} days",
            ),
            "",
            text_for(lang, "## 当前基线", "## Current Baseline"),
        ]
        for item in result["baseline"]:
            backtest = item["backtest"]
            walkforward = item["walkforward"]
            lines.extend(
                [
                    text_for(lang, f"### {item['symbol']}", f"### {item['symbol']}"),
                    text_for(
                        lang,
                        (
                            f"- 回测: trades={backtest['total_trades']} "
                            f"return={backtest['total_return_pct']:.2f}% "
                            f"drawdown={backtest['max_drawdown_pct']:.2f}% "
                            f"profit_factor={backtest['profit_factor']:.2f}"
                        ),
                        (
                            f"- Backtest: trades={backtest['total_trades']} "
                            f"return={backtest['total_return_pct']:.2f}% "
                            f"drawdown={backtest['max_drawdown_pct']:.2f}% "
                            f"profit_factor={backtest['profit_factor']:.2f}"
                        ),
                    ),
                    text_for(
                        lang,
                        (
                            f"- 滚动验证: splits={walkforward['total_splits']} "
                            f"return={walkforward['total_return_pct']:.2f}% "
                            f"avg_win_rate={walkforward['avg_win_rate']:.2f}% "
                            f"profit_factor={walkforward['profit_factor']:.2f}"
                        ),
                        (
                            f"- Walk-forward: splits={walkforward['total_splits']} "
                            f"return={walkforward['total_return_pct']:.2f}% "
                            f"avg_win_rate={walkforward['avg_win_rate']:.2f}% "
                            f"profit_factor={walkforward['profit_factor']:.2f}"
                        ),
                    ),
                ]
            )

        lines.extend(["", text_for(lang, "## 阈值候选", "## Threshold Candidates")])
        if not result["top_candidates"]:
            lines.append(text_for(lang, "- 无有效候选", "- No viable candidates"))
            return "\n".join(lines)

        for index, item in enumerate(result["top_candidates"], start=1):
            lines.append(
                text_for(
                    lang,
                    (
                        f"{index}. {item['symbol']} | xgb={item['xgboost_threshold']:.2f} "
                        f"final={item['final_score_threshold']:.2f} "
                        f"liq={item['min_liquidity_ratio']:.2f} "
                        f"trades={item['total_trades']} "
                        f"return={item['total_return_pct']:.2f}% "
                        f"drawdown={item['max_drawdown_pct']:.2f}% "
                        f"pf={item['profit_factor']:.2f} "
                        f"score={item['score']:.2f}"
                    ),
                    (
                        f"{index}. {item['symbol']} | xgb={item['xgboost_threshold']:.2f} "
                        f"final={item['final_score_threshold']:.2f} "
                        f"liq={item['min_liquidity_ratio']:.2f} "
                        f"trades={item['total_trades']} "
                        f"return={item['total_return_pct']:.2f}% "
                        f"drawdown={item['max_drawdown_pct']:.2f}% "
                        f"pf={item['profit_factor']:.2f} "
                        f"score={item['score']:.2f}"
                    ),
                )
            )
        return "\n".join(lines)

    def _scan_symbol(self, symbol: str) -> list[ValidationCandidate]:
        candidates: list[ValidationCandidate] = []
        for xgb_threshold, final_threshold, min_liquidity in self.DEFAULT_CANDIDATES:
            candidate_settings = self.settings.model_copy(deep=True)
            candidate_settings.model.xgboost_probability_threshold = xgb_threshold
            candidate_settings.model.final_score_threshold = final_threshold
            candidate_settings.strategy.min_liquidity_ratio = min_liquidity
            backtest = self._run_backtest(symbol, candidate_settings)
            summary = backtest["summary"]
            score = self._candidate_score(summary)
            candidates.append(
                ValidationCandidate(
                    symbol=symbol,
                    xgboost_threshold=xgb_threshold,
                    final_score_threshold=final_threshold,
                    min_liquidity_ratio=min_liquidity,
                    total_trades=int(summary["total_trades"]),
                    total_return_pct=float(summary["total_return_pct"]),
                    max_drawdown_pct=float(summary["max_drawdown_pct"]),
                    profit_factor=float(summary["profit_factor"]),
                    sharpe_like=float(summary["sharpe_like"]),
                    score=score,
                )
            )
        return candidates

    def _run_backtest(self, symbol: str, settings: Settings) -> dict:
        engine = V2BacktestEngine(self.storage, settings)
        engine.news = SimpleNamespace(
            get_summary=lambda _symbol: SimpleNamespace(summary="validation_sprint_news")
        )
        engine.macro = SimpleNamespace(
            get_summary=lambda fear_greed=None: SimpleNamespace(summary="validation_sprint_macro")
        )
        return engine.run(symbol)

    def _run_walkforward(self, symbol: str, settings: Settings) -> dict:
        sprint_settings = settings.model_copy(deep=True)
        sprint_settings.training.walkforward_window_days = self.DEFAULT_WALKFORWARD_WINDOW_DAYS
        return WalkForwardBacktester(self.storage, sprint_settings).run(symbol)

    @staticmethod
    def _candidate_score(summary: dict) -> float:
        total_trades = int(summary.get("total_trades") or 0)
        if total_trades <= 0:
            return -1000.0
        total_return = float(summary.get("total_return_pct") or 0.0)
        avg_trade_return_pct = float(
            summary.get(
                "avg_trade_return_pct",
                total_return / total_trades if total_trades else 0.0,
            )
            or 0.0
        )
        profit_factor = float(summary.get("profit_factor", 0.0) or 0.0)
        max_drawdown = float(summary.get("max_drawdown_pct", 0.0) or 0.0)
        win_rate = float(summary.get("win_rate", summary.get("avg_win_rate", 0.0)) or 0.0)
        sharpe_like = float(summary.get("sharpe_like") or 0.0)
        metrics = {
            "sample_count": total_trades,
            "executed_count": total_trades,
            "accuracy": win_rate / 100.0,
            "executed_precision": win_rate / 100.0,
            "expectancy_pct": avg_trade_return_pct,
            "avg_trade_return_pct": avg_trade_return_pct,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown,
            "trade_win_rate": win_rate / 100.0,
            "avg_cost_pct": 0.15,
        }
        trade_bonus = min(total_trades, 12) * 0.15
        return (
            objective_score_from_metrics(metrics)
            + min(sharpe_like, 5.0) * 0.75
            + trade_bonus
        )
