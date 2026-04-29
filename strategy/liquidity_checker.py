"""Liquidity Checker — 流动性检查

核心理念 (来源: 专业量化):
- "流动性决定你能以什么价格成交，而不仅仅是市价"
- 流动性不足 = 大额滑点 = 实际成交价远差于预期 = 隐性亏损
- 开仓前必须检查: 24h成交量够不够、买卖价差有多大、大单冲击成本

设计逻辑:
1. 24h 成交量检查 — 量太小不做
2. 买卖价差检查 — spread > 1% 危险
3. 深度检查 — 订单簿前5档总挂单量
4. 冲击成本估算 — 预估本次交易会造成多少滑点

输出: LiquidityStatus — 明确告诉你"能不能做，预计滑点多少"
"""

from __future__ import annotations

from dataclasses import dataclass
from loguru import logger

import ccxt


@dataclass
class LiquidityStatus:
    """流动性状态"""
    liquid: bool                      # 是否满足最低流动性要求
    spread_pct: float                 # 买卖价差百分比
    estimated_slippage_pct: float     # 预估滑点百分比
    depth_score: float                # 深度评分 (0-1, 高=好)
    volume_24h: float                 # 24h 成交量 (USDT)
    reason: str                       # 决策理由
    risk_level: str                   # "good" / "acceptable" / "poor" / "dangerous"


class LiquidityChecker:
    """
    流动性检查器

    用法:
        checker = LiquidityChecker()
        status = checker.check("BTC/USDT", amount=1000)
        if not status.liquid:
            logger.warning(f"流动性不足: {status.reason}")
    """

    # 流动性阈值
    MIN_VOLUME_24H = 1_000_000         # 最低 24h 成交量 100万 USDT
    GOOD_VOLUME_24H = 10_000_000       # 良好成交量 1000万 USDT

    MAX_SPREAD_PCT = 0.5               # 最大允许价差 0.5%
    GOOD_SPREAD_PCT = 0.1              # 良好价差 0.1%

    MAX_SLIPPAGE_PCT = 0.3             # 最大允许滑点 0.3%
    GOOD_SLIPPAGE_PCT = 0.05           # 良好滑点 0.05%

    MIN_DEPTH_USDT = 500_000           # 最小深度（前5档）50万 USDT

    def __init__(
        self,
        exchange: ccxt.Exchange | None = None,
        min_volume: float = MIN_VOLUME_24H,
        max_spread: float = MAX_SPREAD_PCT,
        max_slippage: float = MAX_SLIPPAGE_PCT,
    ):
        self.exchange = exchange
        self.min_volume = min_volume
        self.max_spread = max_spread
        self.max_slippage = max_slippage

    def set_exchange(self, exchange: ccxt.Exchange):
        """设置交易所实例"""
        self.exchange = exchange

    def check(
        self,
        symbol: str,
        amount_usdt: float = 1000.0,
        orderbook_depth: int = 5,
    ) -> LiquidityStatus:
        """
        综合检查目标币种的流动性

        Args:
            symbol: 交易对 (如 "BTC/USDT")
            amount_usdt: 计划交易金额 (USDT)
            orderbook_depth: 订单簿检查深度（档位数）

        Returns:
            LiquidityStatus
        """
        if not self.exchange:
            logger.warning("LiquidityChecker: 未设置交易所实例，使用默认评估")
            return LiquidityStatus(
                liquid=True,
                spread_pct=0.05,
                estimated_slippage_pct=0.05,
                depth_score=0.7,
                volume_24h=0,
                reason="未设置交易所，假设流动性可接受（请配置交易所实例）",
                risk_level="acceptable",
            )

        try:
            # 并行获取 ticker 和 orderbook
            ticker = self.exchange.fetch_ticker(symbol)
            orderbook = self.exchange.fetch_order_book(symbol, limit=20)

            return self._analyze(
                symbol=symbol,
                ticker=ticker,
                orderbook=orderbook,
                amount_usdt=amount_usdt,
                depth=orderbook_depth,
            )

        except ccxt.NetworkError as e:
            logger.warning(f"流动性检查网络错误 [{symbol}]: {e}")
            return LiquidityStatus(
                liquid=False,
                spread_pct=0, estimated_slippage_pct=0,
                depth_score=0, volume_24h=0,
                reason=f"网络错误: {e}",
                risk_level="dangerous",
            )
        except ccxt.BadSymbol as e:
            logger.error(f"流动性检查: 无效交易对 [{symbol}]: {e}")
            return LiquidityStatus(
                liquid=False,
                spread_pct=0, estimated_slippage_pct=0,
                depth_score=0, volume_24h=0,
                reason=f"无效交易对: {symbol}",
                risk_level="dangerous",
            )
        except Exception as e:
            logger.error(f"流动性检查异常 [{symbol}]: {e}")
            return LiquidityStatus(
                liquid=False,
                spread_pct=0, estimated_slippage_pct=0,
                depth_score=0, volume_24h=0,
                reason=f"异常: {e}",
                risk_level="dangerous",
            )

    def _analyze(
        self,
        symbol: str,
        ticker: dict,
        orderbook: dict,
        amount_usdt: float,
        depth: int,
    ) -> LiquidityStatus:
        """分析 ticker + orderbook 数据"""
        warnings = []

        # ── 1. 24h 成交量 ──
        volume_24h = ticker.get("quoteVolume", 0) or 0

        if volume_24h < self.min_volume:
            warnings.append(f"24h成交量 ${volume_24h:,.0f} 低于最低要求 ${self.min_volume:,.0f}")
        elif volume_24h < self.GOOD_VOLUME_24H:
            warnings.append(f"24h成交量 ${volume_24h:,.0f} 一般（良好标准: ${self.GOOD_VOLUME_24H:,.0f}）")

        volume_score = min(1.0, volume_24h / self.GOOD_VOLUME_24H)

        # ── 2. 买卖价差 ──
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return LiquidityStatus(
                liquid=False,
                spread_pct=0, estimated_slippage_pct=0,
                depth_score=0, volume_24h=volume_24h,
                reason="订单簿为空，无流动性",
                risk_level="dangerous",
            )

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2

        if mid_price > 0:
            spread_pct = (best_ask - best_bid) / mid_price * 100
        else:
            spread_pct = 100.0

        if spread_pct > self.max_spread:
            warnings.append(f"价差 {spread_pct:.3f}% 超过上限 {self.max_spread:.2f}%")

        spread_score = max(0, 1.0 - spread_pct / self.max_spread)

        # ── 3. 订单簿深度 ──
        # 计算前 N 档的总挂单价值
        total_bid_depth = sum(bid[0] * bid[1] for bid in bids[:depth])
        total_ask_depth = sum(ask[0] * ask[1] for ask in asks[:depth])
        total_depth = (total_bid_depth + total_ask_depth) / 2

        depth_score = min(1.0, total_depth / self.MIN_DEPTH_USDT)

        if total_depth < self.MIN_DEPTH_USDT:
            warnings.append(
                f"前{depth}档深度 ${total_depth:,.0f} 低于最低 ${self.MIN_DEPTH_USDT:,.0f}"
            )

        # ── 4. 滑点估算 ──
        slippage = self._estimate_slippage(
            amount_usdt=amount_usdt,
            bids=bids,
            asks=asks,
            side="buy",
        )

        if slippage > self.max_slippage:
            warnings.append(
                f"预估滑点 {slippage:.3f}% 超过上限 {self.max_slippage:.2f}%"
            )

        # ── 5. 综合评估 ──
        # 加权评分
        overall_score = (
            volume_score * 0.3 +
            spread_score * 0.25 +
            depth_score * 0.2 +
            max(0, 1.0 - slippage / self.max_slippage) * 0.25
        )

        if overall_score > 0.7 and not warnings:
            risk_level = "good"
            liquid = True
            reason = f"流动性良好 (评分={overall_score:.2f})"
        elif overall_score > 0.4:
            risk_level = "acceptable"
            liquid = True
            reason = f"流动性可接受 (评分={overall_score:.2f})" + (
                f"; {'; '.join(warnings)}" if warnings else ""
            )
        elif overall_score > 0.2:
            risk_level = "poor"
            liquid = True
            reason = f"流动性较差 (评分={overall_score:.2f}); {'; '.join(warnings)}"
        else:
            risk_level = "dangerous"
            liquid = False
            reason = f"流动性危险 (评分={overall_score:.2f}); {'; '.join(warnings)}"

        status = LiquidityStatus(
            liquid=liquid,
            spread_pct=round(spread_pct, 4),
            estimated_slippage_pct=round(slippage, 4),
            depth_score=round(depth_score, 3),
            volume_24h=volume_24h,
            reason=reason,
            risk_level=risk_level,
        )

        emoji = {"good": "🟢", "acceptable": "🟡", "poor": "🟠", "dangerous": "🔴"}.get(risk_level, "?")
        logger.info(
            f"{emoji} 流动性 [{symbol}]: {risk_level} | "
            f"spread={spread_pct:.3f}% | slippage={slippage:.3f}% | "
            f"depth=${total_depth:,.0f} | vol24h=${volume_24h:,.0f}"
        )

        return status

    def _estimate_slippage(
        self,
        amount_usdt: float,
        bids: list,
        asks: list,
        side: str = "buy",
    ) -> float:
        """
        估算大单冲击滑点

        模拟逐步吃单，计算平均成交价与最优价的偏差
        """
        if amount_usdt <= 0:
            return 0.0

        levels = asks if side == "buy" else bids
        if not levels:
            return 1.0  # 100% 滑点（完全无法成交）

        best_price = levels[0][0]
        remaining = amount_usdt
        total_qty = 0.0
        total_cost = 0.0

        for price, qty in levels:
            if remaining <= 0:
                break

            level_value = price * qty
            fill_value = min(remaining, level_value)
            fill_qty = fill_value / price

            total_qty += fill_qty
            total_cost += fill_value
            remaining -= fill_value

        if total_qty == 0 or total_cost == 0:
            return 1.0

        avg_price = total_cost / total_qty
        slippage = abs(avg_price - best_price) / best_price * 100

        # 如果订单簿不够深（有剩余），额外惩罚
        if remaining > 0:
            unfilled_pct = remaining / amount_usdt
            slippage += unfilled_pct * 5  # 每未成交 1% 加 5% 滑点

        return min(slippage, 100.0)

    def get_liquidity_tier(self, symbol: str, amount_usdt: float = 1000.0) -> str:
        """快速获取流动性等级（简化接口）"""
        status = self.check(symbol, amount_usdt)
        return status.risk_level
