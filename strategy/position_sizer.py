"""Professional Position Sizing — 专业仓位管理

核心原则：活得久比赚得多重要。

仓位管理不是"用多少钱买"，而是：
1. 这笔交易最多亏多少？（风险预算）
2. 亏了之后还能不能交易？（账户保护）
3. 多个持仓的相关性风险？（组合风险）

四大模块：
1. 波动率自适应仓位 — 波大仓小，波小仓大
2. 凯利公式 — 基于胜率和盈亏比的最优仓位
3. 账户级风控 — 单日/连续亏损/最大回撤熔断
4. 金字塔加仓 — 盈利加仓，亏损不加
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from loguru import logger

import numpy as np


@dataclass
class PositionSize:
    """仓位建议"""
    position_pct: float      # 占总资金的比例 (0.0-1.0)
    quantity: float          # 具体数量
    stop_loss_price: float   # 止损价
    risk_amount: float       # 最大亏损金额
    risk_pct: float          # 最大亏损占总资金比例
    reason: str              # 仓位计算理由
    size_method: str         # 用的哪种方法
    adjusted: bool = False   # 是否被风控调整过


@dataclass
class RiskStatus:
    """当前账户风险状态"""
    available_balance: float
    used_balance: float
    daily_pnl: float
    daily_pnl_pct: float
    consecutive_losses: int
    max_drawdown_pct: float
    is_trading_allowed: bool
    cooldown_until: datetime | None = None
    position_limit: float = 0.30  # 当前允许的最大仓位
    warnings: list[str] = field(default_factory=list)


class PositionSizer:
    """专业仓位管理器"""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_single_risk_pct: float = 0.02,    # 单笔最大风险 2%
        max_daily_loss_pct: float = 0.03,     # 单日最大亏损 3%
        max_drawdown_pct: float = 0.10,       # 最大回撤 10%
        cooldown_hours: int = 72,             # 熔断后冷却时间
        consecutive_loss_limit: int = 3,      # 连续亏损限制
    ):
        self.initial_balance = initial_balance
        self.max_single_risk_pct = max_single_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_hours = cooldown_hours
        self.consecutive_loss_limit = consecutive_loss_limit

        self._daily_loss = 0.0
        self._consecutive_losses = 0
        self._peak_balance = initial_balance
        self._cooldown_until: datetime | None = None

    def calculate_position(
        self,
        entry_price: float,
        stop_loss_price: float | None = None,
        atr: float | None = None,
        win_rate: float = 0.55,
        avg_win_pct: float = 3.0,
        avg_loss_pct: float = 1.5,
        volatility: float = 0.5,
        current_positions: int = 0,
        max_positions: int = 3,
        regime_max_pct: float = 0.30,
    ) -> PositionSize:
        """
        综合计算仓位大小

        Args:
            entry_price: 入场价格
            stop_loss_price: 止损价格（如果 None，用 ATR 计算）
            atr: 平均真实波幅
            win_rate: 历史胜率
            avg_win_pct: 平均盈利百分比
            avg_loss_pct: 平均亏损百分比
            volatility: 当前波动率 (0-1)
            current_positions: 当前持仓数量
            max_positions: 最大持仓数
            regime_max_pct: 当前市场状态允许的最大仓位
        """
        # 1. 先检查风控
        risk = self.get_risk_status(current_positions, regime_max_pct)
        if not risk.is_trading_allowed:
            return PositionSize(
                position_pct=0, quantity=0,
                stop_loss_price=entry_price, risk_amount=0,
                risk_pct=0,
                reason=f"交易被暂停: {'; '.join(risk.warnings)}",
                size_method="blocked", adjusted=True,
            )

        # 2. 计算 ATR 止损
        if stop_loss_price is None and atr is not None:
            stop_loss_price = entry_price - atr * 1.5

        if stop_loss_price is None or stop_loss_price >= entry_price:
            stop_loss_price = entry_price * 0.95  # 默认 5% 止损

        risk_per_unit = entry_price - stop_loss_price
        if risk_per_unit <= 0:
            risk_per_unit = entry_price * 0.05

        # 3. 方法 A: 固定风险比例（最基本）
        risk_budget = self.initial_balance * self.max_single_risk_pct
        # 连续亏损后减仓
        if self._consecutive_losses >= 2:
            risk_budget *= 0.5  # 减半
        # 波动率高时减仓
        risk_budget *= max(0.3, 1.0 - volatility * 0.5)

        quantity_a = risk_budget / risk_per_unit
        position_pct_a = (quantity_a * entry_price) / self.initial_balance

        # 4. 方法 B: 凯利公式（进阶）
        win_loss_ratio = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 2.0
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        kelly = max(0, kelly)  # 凯利为负就不交易
        half_kelly = kelly / 2  # 用半凯利更保守

        # 凯利仓位 = half_kelly * balance
        quantity_b = (half_kelly * self.initial_balance) / entry_price
        position_pct_b = quantity_b * entry_price / self.initial_balance

        # 5. 综合取较小值（保守原则）
        position_pct = min(position_pct_a, position_pct_b)

        # 6. 限制在市场状态允许范围内
        position_pct = min(position_pct, regime_max_pct)

        # 7. 考虑已有持仓
        if current_positions >= max_positions:
            return PositionSize(
                position_pct=0, quantity=0,
                stop_loss_price=stop_loss_price, risk_amount=0,
                risk_pct=0,
                reason=f"已达最大持仓数 ({current_positions}/{max_positions})",
                size_method="limit", adjusted=True,
            )

        # 持仓越多，后续仓位越小
        position_factor = max(0.3, 1.0 - current_positions * 0.2)
        position_pct *= position_factor

        # 最终计算
        quantity = (position_pct * self.initial_balance) / entry_price
        risk_amount = quantity * risk_per_unit
        risk_pct = risk_amount / self.initial_balance

        reasons = []
        reasons.append(f"固定风险: {position_pct_a:.2%}")
        if kelly > 0:
            reasons.append(f"半凯利: {position_pct_b:.2%}")
        if volatility > 0.5:
            reasons.append(f"高波动率减仓 ({volatility:.0%})")
        if self._consecutive_losses >= 2:
            reasons.append(f"连续亏损{self._consecutive_losses}次，仓位减半")
        if current_positions > 0:
            reasons.append(f"已有{current_positions}持仓，仓位系数{position_factor:.0%}")

        return PositionSize(
            position_pct=round(position_pct, 4),
            quantity=round(quantity, 6),
            stop_loss_price=round(stop_loss_price, 2),
            risk_amount=round(risk_amount, 2),
            risk_pct=round(risk_pct, 4),
            reason="；".join(reasons),
            size_method="combined",
        )

    def get_risk_status(
        self, current_positions: int = 0,
        regime_max_pct: float = 0.30,
    ) -> RiskStatus:
        """获取当前账户风险状态"""
        warnings = []
        is_allowed = True

        current_balance = self.initial_balance - self._daily_loss

        # 连续亏损检查
        if self._consecutive_losses >= self.consecutive_loss_limit:
            warnings.append(
                f"连续亏损 {self._consecutive_losses} 次，"
                f"建议冷却 {self.cooldown_hours} 小时"
            )
            if self._consecutive_losses >= self.consecutive_loss_limit + 1:
                is_allowed = False

        # 单日亏损检查
        daily_loss_pct = self._daily_loss / self.initial_balance
        if daily_loss_pct >= self.max_daily_loss_pct:
            warnings.append(f"今日亏损 {daily_loss_pct:.2%}，已达日限额")
            is_allowed = False

        # 最大回撤检查
        drawdown_pct = (self._peak_balance - current_balance) / self._peak_balance
        if drawdown_pct >= self.max_drawdown_pct:
            warnings.append(f"最大回撤 {drawdown_pct:.2%}，已达限额")
            is_allowed = False

        # 冷却期检查
        if self._cooldown_until and datetime.now(timezone.utc) < self._cooldown_until:
            remaining = (self._cooldown_until - datetime.now(timezone.utc)).total_seconds() / 3600
            warnings.append(f"冷却期中，剩余 {remaining:.1f} 小时")
            is_allowed = False

        # 计算可用仓位上限
        position_limit = min(regime_max_pct, 0.30)
        if self._consecutive_losses >= 2:
            position_limit *= 0.5
        if daily_loss_pct >= self.max_daily_loss_pct * 0.7:
            position_limit *= 0.5

        return RiskStatus(
            available_balance=current_balance,
            used_balance=self._daily_loss,
            daily_pnl=self._daily_loss,
            daily_pnl_pct=daily_loss_pct,
            consecutive_losses=self._consecutive_losses,
            max_drawdown_pct=drawdown_pct * 100,
            is_trading_allowed=is_allowed,
            cooldown_until=self._cooldown_until,
            position_limit=position_limit,
            warnings=warnings,
        )

    def record_trade_result(self, pnl: float, is_win: bool):
        """记录交易结果（更新风控状态）"""
        self._daily_loss += (-pnl if pnl < 0 else 0)

        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        # 更新峰值
        current = self.initial_balance - self._daily_loss
        if current > self._peak_balance:
            self._peak_balance = current

        # 检查是否需要触发冷却
        if self._consecutive_losses >= self.consecutive_loss_limit:
            self._cooldown_until = datetime.now(timezone.utc) + timedelta(
                hours=self.cooldown_hours
            )

        status = self.get_risk_status()
        if status.warnings:
            for w in status.warnings:
                logger.warning(f"⚠️ 风控: {w}")

    def reset_daily(self):
        """每日重置"""
        self._daily_loss = 0.0
        logger.info("Daily risk counters reset")
