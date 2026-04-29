"""Backtest Engine — 历史回测"""
from __future__ import annotations

from datetime import datetime
from loguru import logger

import pandas as pd
import numpy as np

from core.storage import Storage
from analysis.technical import TechnicalAnalyzer


class BacktestEngine:
    """策略回测引擎"""

    def __init__(self, storage: Storage):
        self.storage = storage
        self.technical = TechnicalAnalyzer()

    def run_backtest(
        self,
        symbol: str,
        timeframe: str = "1h",
        initial_balance: float = 10000.0,
        confidence_threshold: float = 0.65,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        max_position_pct: float = 0.30,
    ) -> dict:
        """运行回测

        简化版回测：基于技术指标信号，模拟历史交易
        """
        logger.info(
            f"Starting backtest: {symbol} {timeframe} "
            f"threshold={confidence_threshold}"
        )

        # 获取历史 K 线数据
        candles = self.storage.get_ohlcv(symbol, timeframe, limit=2000)
        if len(candles) < 100:
            logger.error(f"Not enough data for backtest: {len(candles)} candles")
            return {}

        # 按时间正序排列
        candles.sort(key=lambda c: c["timestamp"])

        # 模拟交易
        balance = initial_balance
        position = None  # {"entry_price", "quantity", "entry_idx"}
        trades = []
        equity_curve = [{"idx": 0, "equity": balance}]

        for i in range(60, len(candles)):
            # 用过去 60 根 K 线计算指标
            window = candles[max(0, i - 60):i]
            tech = self.technical.calculate_all(window)
            signal = self.technical.generate_signal(tech)

            current_close = candles[i]["close"]

            # 如果有持仓，检查止损止盈
            if position:
                entry_price = position["entry_price"]
                pnl_pct = (current_close / entry_price - 1) * 100

                # 止损
                if pnl_pct <= -stop_loss_pct * 100:
                    pnl = position["quantity"] * (current_close - entry_price)
                    balance += position["quantity"] * current_close
                    trades.append({
                        "entry_idx": position["entry_idx"],
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": current_close,
                        "quantity": position["quantity"],
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "reason": "stop_loss",
                        "direction": "LONG",
                    })
                    position = None

                # 止盈
                elif pnl_pct >= take_profit_pct * 100:
                    pnl = position["quantity"] * (current_close - entry_price)
                    balance += position["quantity"] * current_close
                    trades.append({
                        "entry_idx": position["entry_idx"],
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": current_close,
                        "quantity": position["quantity"],
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "reason": "take_profit",
                        "direction": "LONG",
                    })
                    position = None

            # 如果无持仓，检查是否开仓
            if not position and signal["direction"] == "LONG":
                if signal["confidence"] >= confidence_threshold:
                    invest_amount = balance * max_position_pct
                    qty = invest_amount / current_close
                    balance -= invest_amount
                    position = {
                        "entry_price": current_close,
                        "quantity": qty,
                        "entry_idx": i,
                    }

            # 计算当前权益
            current_equity = balance
            if position:
                current_equity += position["quantity"] * current_close
            equity_curve.append({
                "idx": i,
                "equity": current_equity,
                "timestamp": candles[i]["timestamp"],
            })

        # 如果还有未平仓，按最后一根 K 线平仓
        if position and candles:
            last_close = candles[-1]["close"]
            pnl = position["quantity"] * (last_close - position["entry_price"])
            pnl_pct = (last_close / position["entry_price"] - 1) * 100
            balance += position["quantity"] * last_close
            trades.append({
                "entry_idx": position["entry_idx"],
                "exit_idx": len(candles) - 1,
                "entry_price": position["entry_price"],
                "exit_price": last_close,
                "quantity": position["quantity"],
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "reason": "end_of_data",
                "direction": "LONG",
            })

        # 计算统计指标
        stats = self._calculate_stats(trades, equity_curve, initial_balance)

        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "initial_balance": initial_balance,
            "final_balance": balance,
            "total_return_pct": (balance / initial_balance - 1) * 100,
            "trades": trades,
            "equity_curve": equity_curve,
            "stats": stats,
            "data_points": len(candles),
        }

        logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"return={result['total_return_pct']:+.2f}%"
        )

        return result

    def _calculate_stats(self, trades: list, equity_curve: list,
                         initial_balance: float) -> dict:
        """计算回测统计指标"""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_pnl_pct": 0,
                "max_drawdown_pct": 0,
                "sharpe_ratio": 0,
                "profit_factor": 0,
            }

        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # 胜率
        win_rate = len(wins) / len(pnls) if pnls else 0

        # 平均盈亏
        avg_pnl = sum(pnls) / len(pnls)

        # 最大回撤
        equities = [e["equity"] for e in equity_curve]
        max_dd = 0
        peak = equities[0] if equities else initial_balance
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # 盈亏比
        total_profit = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0.001
        profit_factor = total_profit / total_loss if total_loss else 0

        # 简化夏普比率（假设无风险利率为0）
        if len(pnls) > 1:
            sharpe = (avg_pnl / (np.std(pnls) + 0.001)) * np.sqrt(252)
        else:
            sharpe = 0

        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate * 100, 2),
            "wins": len(wins),
            "losses": len(losses),
            "avg_pnl_pct": round(avg_pnl, 2),
            "avg_win_pct": round(sum(wins) / len(wins), 2) if wins else 0,
            "avg_loss_pct": round(sum(losses) / len(losses), 2) if losses else 0,
            "max_drawdown_pct": round(max_dd, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
        }

    def compare_with_buy_and_hold(
        self, backtest_result: dict, symbol: str
    ) -> dict:
        """与买入持有策略对比"""
        candles = self.storage.get_ohlcv(symbol, "1h", limit=2000)
        if not candles:
            return {}

        candles.sort(key=lambda c: c["timestamp"])
        first_price = candles[0]["close"]
        last_price = candles[-1]["close"]
        hold_return = (last_price / first_price - 1) * 100

        return {
            "strategy_return": round(backtest_result["total_return_pct"], 2),
            "buy_and_hold_return": round(hold_return, 2),
            "outperformance": round(
                backtest_result["total_return_pct"] - hold_return, 2
            ),
        }
