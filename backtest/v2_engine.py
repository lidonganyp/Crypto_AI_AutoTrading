"""Backtest engine for CryptoAI v3 decision flow."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from analysis.macro_service import MacroService
from analysis.news_service import NewsService
from analysis.research_llm import ResearchLLMAnalyzer
from config import Settings, resolve_project_path
from core.i18n import get_default_language, normalize_language, text_for
from core.feature_pipeline import FeatureInput, FeaturePipeline
from core.models import MarketRegime, PredictionResult, ResearchInsight, SuggestedAction
from core.storage import Storage
from core.trade_simulation import timeframe_hours
from strategy.decision_engine import DecisionEngine
from strategy.risk_manager import RiskManager
from strategy.model_trainer import model_path_for_symbol
from strategy.xgboost_predictor import XGBoostPredictor


@dataclass
class BacktestTrade:
    entry_timestamp: int
    exit_timestamp: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fee_cost: float
    reason: str


class V2BacktestEngine:
    """Simulate the current v3 strategy on stored OHLCV data."""

    ONE_WAY_FEE_RATE = 0.00075

    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings
        self.pipeline = FeaturePipeline()
        self._predictor_base_path = resolve_project_path(
            self.settings.model.xgboost_model_path,
            self.settings,
        )
        self.news = NewsService()
        self.macro = MacroService()
        self.research = ResearchLLMAnalyzer(self.settings.llm, clients={})
        self.risk = RiskManager(self.settings.risk, self.settings.strategy)
        self.decision = DecisionEngine(
            xgboost_threshold=self.settings.model.xgboost_probability_threshold,
            final_score_threshold=self.settings.model.final_score_threshold,
            sentiment_weight=self.settings.strategy.sentiment_weight,
            min_liquidity_ratio=self.settings.strategy.min_liquidity_ratio,
            trend_reversal_probability=self.settings.strategy.trend_reversal_probability,
            sentiment_exit_threshold=self.settings.strategy.sentiment_exit_threshold,
            fixed_stop_loss_pct=self.settings.strategy.fixed_stop_loss_pct,
            take_profit_levels=self.settings.strategy.take_profit_levels,
            max_hold_hours=self.settings.strategy.max_hold_hours,
        )

    def run(self, symbol: str, initial_balance: float = 10000.0) -> dict:
        predictor = XGBoostPredictor(
            str(model_path_for_symbol(self._predictor_base_path, symbol)),
            enable_fallback=self.settings.model.enable_fallback_predictor,
        )
        market_symbol = f"{symbol}:USDT" if ":USDT" not in symbol else symbol
        candles_1h = self.storage.get_ohlcv(market_symbol, "1h", limit=2000)
        candles_4h = self.storage.get_ohlcv(market_symbol, "4h", limit=1000)
        candles_1d = self.storage.get_ohlcv(market_symbol, "1d", limit=400)
        candles_1h.sort(key=lambda item: item["timestamp"])
        candles_4h.sort(key=lambda item: item["timestamp"])
        candles_1d.sort(key=lambda item: item["timestamp"])

        if len(candles_4h) < 80:
            return {
                "symbol": symbol,
                "summary": {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "total_return_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "profit_factor": 0.0,
                    "sharpe_like": 0.0,
                },
                "trades": [],
            }

        balance = initial_balance
        peak_equity = initial_balance
        max_drawdown = 0.0
        trades: list[BacktestTrade] = []
        open_position: dict | None = None
        macro_summary = self.macro.get_summary(fear_greed=55).summary
        news_summary = self.news.get_summary(symbol).summary
        candle_hours = timeframe_hours("4h")

        for index in range(60, len(candles_4h) - 1):
            current_4h = candles_4h[index]
            next_4h = candles_4h[index + 1]
            one_h_subset = [c for c in candles_1h if c["timestamp"] <= current_4h["timestamp"]][-240:]
            four_h_subset = candles_4h[: index + 1][-240:]
            one_d_subset = [c for c in candles_1d if c["timestamp"] <= current_4h["timestamp"]][-240:]
            snapshot = self.pipeline.build(
                FeatureInput(
                    symbol=symbol,
                    candles_1h=one_h_subset,
                    candles_4h=four_h_subset,
                    candles_1d=one_d_subset,
                    sentiment_value=0.1,
                    market_regime_score=0.4,
                )
            )
            if not snapshot.valid:
                continue

            prediction = predictor.predict(snapshot)
            insight = self._backtest_insight(symbol, macro_summary, news_summary)

            closed_this_bar = False
            if open_position:
                open_position["hours_held"] = float(open_position.get("hours_held", 0.0)) + candle_hours
                exit_signal = self._exit_signal(
                    position=open_position,
                    candle=next_4h,
                    prediction=prediction,
                    insight=insight,
                )
                if exit_signal is not None:
                    exit_price, exit_reason = exit_signal
                    trade, cash_credit = self._close_position(
                        open_position,
                        next_4h,
                        exit_price,
                        exit_reason,
                    )
                    balance += cash_credit
                    trades.append(trade)
                    open_position = None
                    closed_this_bar = True

            if open_position is None and not closed_this_bar:
                account = self.risk.build_account_state(
                    equity=balance,
                    positions=[],
                    peak_equity=peak_equity,
                )
                risk_result = self.risk.can_open_position(
                    account=account,
                    positions=[],
                    symbol=symbol,
                    atr=float(snapshot.values.get("atr_4h", 0.0)),
                    entry_price=float(current_4h["close"]),
                    liquidity_ratio=float(snapshot.values.get("volume_ratio_1h", 0.0)),
                )
                _, execution = self.decision.evaluate_entry(
                    symbol=symbol,
                    prediction=prediction,
                    insight=insight,
                    features=snapshot,
                    risk_result=risk_result,
                )
                if execution.should_execute and execution.position_value > 0:
                    entry_price = float(current_4h["close"])
                    entry_notional = min(
                        float(execution.position_value),
                        balance / (1.0 + self.ONE_WAY_FEE_RATE),
                    )
                    if entry_notional <= 0:
                        continue
                    quantity = entry_notional / entry_price
                    entry_fee = entry_notional * self.ONE_WAY_FEE_RATE
                    balance -= entry_notional + entry_fee
                    open_position = {
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "quantity": quantity,
                        "entry_timestamp": int(current_4h["timestamp"]),
                        "entry_notional": entry_notional,
                        "entry_fee": entry_fee,
                        "hours_held": 0.0,
                    }

            equity = balance
            if open_position:
                equity += open_position["quantity"] * float(next_4h["close"])
            peak_equity = max(peak_equity, equity)
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

        if open_position:
            trade, cash_credit = self._close_position(
                open_position,
                candles_4h[-1],
                float(candles_4h[-1]["close"]),
                "end_of_data",
            )
            balance += cash_credit
            trades.append(trade)

        summary = self._summary(initial_balance, balance, max_drawdown, trades)
        return {
            "symbol": symbol,
            "summary": summary,
            "trades": [trade.__dict__ for trade in trades],
        }

    def _backtest_insight(
        self,
        symbol: str,
        macro_summary: str,
        news_summary: str,
    ) -> ResearchInsight:
        return self.research._fallback(symbol, 55.0).model_copy(
            update={
                "symbol": symbol,
                "key_reason": [news_summary, macro_summary],
                "market_regime": MarketRegime.UPTREND,
                "sentiment_score": 0.15,
                "suggested_action": SuggestedAction.OPEN_LONG,
            }
        )

    def _exit_signal(
        self,
        *,
        position: dict,
        candle: dict,
        prediction: PredictionResult,
        insight: ResearchInsight,
    ) -> tuple[float, str] | None:
        entry_price = float(position["entry_price"])
        low_price = float(candle.get("low") or candle.get("close") or entry_price)
        high_price = float(candle.get("high") or candle.get("close") or entry_price)
        stop_price = entry_price * (1.0 - self.decision.fixed_stop_loss_pct)
        if low_price <= stop_price:
            return stop_price, "fixed_stop_loss"
        for target_index, target_pct in enumerate(self.decision.take_profit_levels, start=1):
            target_price = entry_price * (1.0 + float(target_pct))
            if high_price >= target_price:
                return target_price, f"take_profit_{target_index}"

        exit_reasons = self.decision.evaluate_exit(
            position=position,
            current_price=float(candle["close"]),
            prediction=prediction,
            insight=insight,
            hours_held=float(position.get("hours_held", 0.0)),
        )
        for reason in ("trend_reversal", "time_stop"):
            if reason in exit_reasons:
                return float(candle["close"]), reason
        return None

    @classmethod
    def _close_position(
        cls,
        position: dict,
        candle: dict,
        exit_price: float,
        reason: str,
    ) -> tuple[BacktestTrade, float]:
        entry_price = float(position["entry_price"])
        quantity = float(position["quantity"])
        entry_notional = float(position.get("entry_notional") or (entry_price * quantity))
        entry_fee = float(position.get("entry_fee") or 0.0)
        exit_notional = quantity * float(exit_price)
        exit_fee = exit_notional * cls.ONE_WAY_FEE_RATE
        cash_credit = exit_notional - exit_fee
        pnl = cash_credit - entry_notional - entry_fee
        pnl_pct = (pnl / entry_notional) * 100 if entry_notional > 0 else 0.0
        return BacktestTrade(
            entry_timestamp=position["entry_timestamp"],
            exit_timestamp=int(candle["timestamp"]),
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fee_cost=entry_fee + exit_fee,
            reason=reason,
        ), cash_credit

    @staticmethod
    def render_report(result: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        summary = result["summary"]
        lines = [
            text_for(lang, f"# V3 回测报告: {result['symbol']}", f"# V3 Backtest Report: {result['symbol']}"),
            text_for(lang, f"- 总交易数: {summary['total_trades']}", f"- Total Trades: {summary['total_trades']}"),
            text_for(lang, f"- 胜率: {summary['win_rate']:.2f}%", f"- Win Rate: {summary['win_rate']:.2f}%"),
            text_for(lang, f"- 总收益: {summary['total_return_pct']:.4f}%", f"- Total Return: {summary['total_return_pct']:.4f}%"),
            text_for(lang, f"- 最大回撤: {summary['max_drawdown_pct']:.4f}%", f"- Max Drawdown: {summary['max_drawdown_pct']:.4f}%"),
            text_for(lang, f"- 盈亏因子: {summary['profit_factor']:.4f}", f"- Profit Factor: {summary['profit_factor']:.4f}"),
            text_for(lang, f"- 类夏普值: {summary['sharpe_like']:.4f}", f"- Sharpe-like: {summary['sharpe_like']:.4f}"),
        ]
        return "\n".join(lines)

    @staticmethod
    def _summary(
        initial_balance: float,
        final_balance: float,
        max_drawdown: float,
        trades: list[BacktestTrade],
    ) -> dict:
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "profit_factor": 0.0,
                "sharpe_like": 0.0,
            }

        returns = [trade.pnl_pct / 100.0 for trade in trades]
        wins = [value for value in returns if value > 0]
        losses = [value for value in returns if value <= 0]
        profit_factor = (
            sum(wins) / abs(sum(losses))
            if losses and abs(sum(losses)) > 1e-12
            else 0.0
        )
        avg_return = sum(returns) / len(returns)
        std = np.std(returns) if len(returns) > 1 else 0.0
        sharpe_like = avg_return / std * sqrt(len(returns)) if std > 1e-12 else 0.0
        return {
            "total_trades": len(trades),
            "win_rate": sum(1 for value in returns if value > 0) / len(returns) * 100,
            "total_return_pct": (final_balance / initial_balance - 1.0) * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "profit_factor": profit_factor,
            "sharpe_like": sharpe_like,
        }
