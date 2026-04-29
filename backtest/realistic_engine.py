"""Realistic Backtest Engine — 真实成本回测引擎

不包含手续费/滑点/资金费率的回测 = 自我欺骗。
真实回测必须包含:
1. 交易手续费 (OKX: maker 0.02%, taker 0.05%)
2. 滑点模型 (按成交量的 0.05-0.1%)
3. 资金费率成本 (持仓过夜的成本)
4. 流动性限制 (单笔不超过近期成交量的 5%)
5. 分市场状态回测
"""

from __future__ import annotations

from datetime import datetime
from loguru import logger

import pandas as pd
import numpy as np

from core.storage import Storage
from analysis.technical import TechnicalAnalyzer


class RealisticBacktestEngine:
    """真实成本回测引擎"""

    # OKX 真实费率
    MAKER_FEE = 0.0002   # 0.02%
    TAKER_FEE = 0.0005   # 0.05%
    DEFAULT_SLIPPAGE = 0.0005  # 0.05% 默认滑点
    FUNDING_RATE_AVG = 0.0001  # 平均资金费率 0.01%/8h
    MAX_VOLUME_SHARE = 0.05     # 单笔不超过近期成交量的5%

    def __init__(self, storage: Storage):
        self.storage = storage

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
        """运行含真实成本的专业回测"""
        candles = self.storage.get_ohlcv(symbol, timeframe, limit=2000)
        if len(candles) < 100:
            logger.error(f"Not enough data: {len(candles)} candles")
            return {}

        candles.sort(key=lambda c: c["timestamp"])
        balance = initial_balance
        position = None
        trades = []
        equity_curve = [{"idx": 0, "equity": balance, "cost": 0}]
        total_fees = 0
        total_slippage = 0
        total_funding = 0

        # 预计算成交量均值（用于滑点和流动性限制）
        volumes = [c["volume"] for c in candles]
        avg_volume = np.mean(volumes[-100:]) if len(volumes) >= 100 else np.mean(volumes)

        for i in range(60, len(candles)):
            window = candles[max(0, i - 60):i]
            tech = self.technical_calculate_all(window)
            signal = self.generate_signal(tech)
            current = candles[i]
            close = current["close"]
            volume = current["volume"]

            # 有持仓 → 检查止损止盈
            if position:
                pnl_pct = (close / position["entry_price"] - 1) * 100

                # 资金费率成本（每8小时扣除一次）
                hours_held = (i - position["entry_idx"])
                funding_periods = hours_held // 8
                if funding_periods > position.get("funding_paid", 0):
                    funding_cost = position["quantity"] * close * self.FUNDING_RATE_AVG * (funding_periods - position.get("funding_paid", 0))
                    balance -= funding_cost
                    total_funding += funding_cost
                    position["funding_paid"] = funding_periods

                if pnl_pct <= -stop_loss_pct * 100:
                    exit_price, trade = self._execute_close(
                        position, close, volume, avg_volume, "stop_loss", balance, i
                    )
                    balance = trade["balance_after"]
                    total_fees += trade["fee"]
                    total_slippage += trade["slippage"]
                    trades.append(trade)
                    position = None

                elif pnl_pct >= take_profit_pct * 100:
                    exit_price, trade = self._execute_close(
                        position, close, volume, avg_volume, "take_profit", balance, i
                    )
                    balance = trade["balance_after"]
                    total_fees += trade["fee"]
                    total_slippage += trade["slippage"]
                    trades.append(trade)
                    position = None

            # 无持仓 → 检查开仓
            if not position and signal["direction"] == "LONG":
                if signal["confidence"] >= confidence_threshold:
                    # 流动性检查
                    max_order_value = avg_volume * close * self.MAX_VOLUME_SHARE
                    invest_amount = min(balance * max_position_pct, max_order_value)
                    if invest_amount < 10:  # 最小下单金额
                        continue

                    # 计算滑点（小单滑点小，大单滑点大）
                    slippage_rate = self._calc_slippage(invest_amount, avg_volume, close)
                    adjusted_price = close * (1 + slippage_rate)

                    # 手续费 (taker)
                    fee = invest_amount * self.TAKER_FEE

                    qty = invest_amount / adjusted_price
                    balance -= (invest_amount + fee)

                    position = {
                        "entry_price": adjusted_price,
                        "quantity": qty,
                        "entry_idx": i,
                        "funding_paid": 0,
                        "fee_paid": fee,
                    }
                    total_fees += fee

            # 权益计算
            current_equity = balance
            if position:
                current_equity += position["quantity"] * close

            equity_curve.append({
                "idx": i, "equity": current_equity, "cost": total_fees + total_slippage + total_funding,
            })

        # 未平仓的收尾
        if position and candles:
            close = candles[-1]["close"]
            volume = candles[-1]["volume"]
            exit_price, trade = self._execute_close(
                position, close, volume, avg_volume, "end_of_data", balance, len(candles) - 1
            )
            balance = trade["balance_after"]
            total_fees += trade["fee"]
            total_slippage += trade["slippage"]
            trades.append(trade)

        stats = self._calculate_stats(trades, equity_curve, initial_balance)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "initial_balance": initial_balance,
            "final_balance": balance,
            "total_return_pct": (balance / initial_balance - 1) * 100,
            "trades": trades,
            "equity_curve": equity_curve,
            "stats": stats,
            "cost_breakdown": {
                "total_fees": round(total_fees, 2),
                "total_slippage": round(total_slippage, 2),
                "total_funding": round(total_funding, 2),
                "total_cost": round(total_fees + total_slippage + total_funding, 2),
                "cost_pct": round((total_fees + total_slippage + total_funding) / initial_balance * 100, 4),
            },
        }

    def _execute_close(self, position, close, volume, avg_volume,
                        reason, balance, exit_idx):
        """执行平仓（含手续费和滑点）"""
        exit_price = close * (1 - self.DEFAULT_SLIPPAGE)  # 卖出时滑点反向
        pnl = position["quantity"] * (exit_price - position["entry_price"])
        fee = position["quantity"] * exit_price * self.TAKER_FEE
        slippage_cost = abs(position["quantity"] * close * self.DEFAULT_SLIPPAGE)
        balance += position["quantity"] * exit_price - fee
        return exit_price, {
            "entry_idx": position["entry_idx"],
            "exit_idx": exit_idx,
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "quantity": position["quantity"],
            "pnl": pnl,
            "pnl_pct": (exit_price / position["entry_price"] - 1) * 100,
            "reason": reason,
            "fee": fee,
            "slippage": slippage_cost,
            "balance_after": balance,
        }

    @staticmethod
    def _calc_slippage(order_value, avg_volume, price):
        """动态滑点计算"""
        # 基础滑点 0.05%
        base_slip = 0.0005
        # 订单越大滑点越大
        volume_ratio = order_value / (avg_volume * price) if avg_volume * price > 0 else 0
        # 线性增长，最大 0.5%
        return min(0.005, base_slip * (1 + volume_ratio * 10))

    def technical_calculate_all(self, candles):
        """简化版技术指标计算"""
        from analysis.technical import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        df = pd.DataFrame(candles)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return analyzer.calculate_all(df)

    @staticmethod
    def generate_signal(tech):
        """生成信号（简化版）"""
        score = 0
        reasons = []
        rsi = tech.get("RSI_14")
        if rsi is not None and rsi < 30:
            score += 30
        elif rsi is not None and rsi < 45:
            score += 10
        elif rsi is not None and rsi > 70:
            score -= 30

        ema_cross = tech.get("EMA_cross_9_21")
        if ema_cross == "bullish":
            score += 15
        elif ema_cross == "bearish":
            score -= 15

        bb_pos = tech.get("BB_position")
        if bb_pos == "near_lower":
            score += 10

        vol_change = tech.get("volume_vs_avg_20")
        if vol_change is not None and vol_change > 50:
            score += 10

        if score > 30:
            return {"direction": "LONG", "confidence": min(0.7, 0.4 + score / 200), "reasons": reasons}
        return {"direction": "FLAT", "confidence": 0.3, "reasons": reasons}

    def _calculate_stats(self, trades, equity_curve, initial_balance):
        if not trades:
            return {"total_trades": 0, "win_rate": 0, "avg_pnl_pct": 0,
                    "max_drawdown_pct": 0, "sharpe_ratio": 0, "profit_factor": 0}
        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_pnl = np.mean(pnls) if pnls else 0
        max_dd = 0
        peak = initial_balance
        for e in equity_curve:
            if e["equity"] > peak:
                peak = e["equity"]
            dd = (peak - e["equity"]) / peak * 100
            max_dd = max(max_dd, dd)
        total_profit = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0.001
        sharpe = (avg_pnl / (np.std(pnls) + 0.001)) * np.sqrt(252) if len(pnls) > 1 else 0
        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate * 100, 2),
            "wins": len(wins), "losses": len(losses),
            "avg_pnl_pct": round(avg_pnl, 2),
            "avg_win_pct": round(np.mean(wins), 2) if wins else 0,
            "avg_loss_pct": round(np.mean(losses), 2) if losses else 0,
            "max_drawdown_pct": round(max_dd, 2),
            "profit_factor": round(total_profit / total_loss, 2),
            "sharpe_ratio": round(sharpe, 2),
        }
