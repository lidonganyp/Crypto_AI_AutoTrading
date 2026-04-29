"""Patience Gate — "不交易 = 最好的交易" 独立模块

核心理念 (来源: CZ / Mark Douglas):
- "The best trade is the one you don't make" — CZ
- "交易中最重要的是知道什么时候不出手" — Mark Douglas
- 大多数亏损来自"应该空仓的时候强行交易"

多维度评估系统是否应该交易：
1. 市场状态不明确 → 不交易
2. 波动率过低（震荡无聊市） → 不交易
3. 波动率过高（黑天鹅/闪崩） → 不交易
4. 信号置信度不足 → 不交易
5. 临近重大事件 → 不交易（或极小仓位）
6. 流动性不足 → 不交易
7. 冷却期中 → 不交易

输出: PatienceResult — 明确告诉你"能不能交易，为什么，等多久"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from loguru import logger


@dataclass
class PatienceRule:
    """单条耐心规则"""
    name: str               # 规则名称
    enabled: bool = True    # 是否启用
    block: bool = True      # True=阻止交易, False=仅警告
    weight: float = 1.0     # 权重（影响最终评分）
    min_wait_minutes: int = 0  # 建议最少等待时间


@dataclass
class PatienceResult:
    """耐心门控结果"""
    ok_to_trade: bool              # 是否允许交易
    reason: str                    # 决策理由（人可读）
    wait_time_minutes: int         # 建议等待时间（0=可以立即交易）
    priority: str                  # "block" / "caution" / "go"
    patience_score: float          # 耐心评分 0.0(很焦虑)-1.0(很耐心)
    blocked_rules: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 内置规则定义
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_RULES = {
    "market_unclear": PatienceRule(
        name="市场状态不明确",
        block=True, weight=1.5,
        min_wait_minutes=60,
    ),
    "volatility_too_low": PatienceRule(
        name="波动率过低",
        block=True, weight=1.0,
        min_wait_minutes=120,
    ),
    "volatility_too_high": PatienceRule(
        name="波动率过高（黑天鹅）",
        block=True, weight=2.0,
        min_wait_minutes=240,
    ),
    "low_confidence": PatienceRule(
        name="信号置信度不足",
        block=True, weight=1.2,
        min_wait_minutes=30,
    ),
    "near_major_event": PatienceRule(
        name="临近重大事件",
        block=False, weight=1.5,
        min_wait_minutes=60,
    ),
    "low_liquidity": PatienceRule(
        name="流动性不足",
        block=True, weight=1.8,
        min_wait_minutes=60,
    ),
    "cooldown_active": PatienceRule(
        name="冷却期中",
        block=True, weight=2.5,
        min_wait_minutes=60,
    ),
    "psychology_danger": PatienceRule(
        name="心理状态危险",
        block=True, weight=2.0,
        min_wait_minutes=60,
    ),
    "weekend_holiday": PatienceRule(
        name="周末/节假日",
        block=False, weight=0.5,
        min_wait_minutes=0,
    ),
    "recent_big_loss": PatienceRule(
        name="近期大亏",
        block=True, weight=1.5,
        min_wait_minutes=120,
    ),
}


class PatienceGate:
    """
    "不交易 = 最好的交易" 独立门控模块

    用法:
        gate = PatienceGate()
        result = gate.evaluate(
            regime_state="BULL_TREND",
            volatility=0.4,
            signal_confidence=0.75,
            fear_greed=55,
        )
        if not result.ok_to_trade:
            logger.info(f"耐心门控: {result.reason}")
            # 跳过本次交易
    """

    # ── 默认阈值 ──
    VOLATILITY_LOW_THRESHOLD = 0.15    # ATR/价格 < 15% → 波动过低
    VOLATILITY_HIGH_THRESHOLD = 0.80   # ATR/价格 > 80% → 波动过高
    CONFIDENCE_THRESHOLD = 0.55        # 信号置信度低于此值 → 不交易
    FEAR_GREED_EXTREME_LOW = 10        # FGI < 10 → 极端恐慌（谨慎，不等于抄底）
    FEAR_GREED_EXTREME_HIGH = 90       # FGI > 90 → 极端贪婪

    # 允许交易的市场状态（其他状态需要额外检查）
    CLEAR_STATES = {"BULL_TREND", "BEAR_TREND", "EXTREME_FEAR"}
    CAUTION_STATES = {"BULL_CONSOL", "BEAR_RALLY"}

    def __init__(self, rules: dict[str, PatienceRule] | None = None):
        self.rules = rules or BUILTIN_RULES.copy()
        self._last_block_time: dict[str, datetime] = {}

    def evaluate(
        self,
        regime_state: str = "UNKNOWN",
        volatility: float | None = None,
        signal_confidence: float | None = None,
        fear_greed: float | None = None,
        total_position_pct: float = 0.0,
        recent_pnl_pct: float | None = None,
        psychology_risk: str | None = None,
        is_weekend: bool = False,
        liquidity_score: float | None = None,
    ) -> PatienceResult:
        """
        多维度评估是否应该交易

        Args:
            regime_state: 市场状态 (来自 MarketRegimeDetector)
            volatility: 当前波动率 (0-1, ATR/price)
            signal_confidence: 信号置信度 (0-1)
            fear_greed: 恐惧贪婪指数 (0-100)
            total_position_pct: 当前总仓位占比
            recent_pnl_pct: 近期盈亏百分比
            psychology_risk: 心理风险等级 ("safe" / "caution" / "danger")
            is_weekend: 是否周末
            liquidity_score: 流动性评分 (0-1, 高=好)
        """
        blocked = []
        warned = []
        max_wait = 0
        patience_score = 0.5  # 基础耐心分

        # ── 规则 1: 市场状态不明确 ──
        rule = self.rules.get("market_unclear")
        if rule and rule.enabled and regime_state == "UNKNOWN":
            blocked.append(rule.name)
            max_wait = max(max_wait, rule.min_wait_minutes)
            patience_score += rule.weight * 0.15
            logger.info(f"🚫 Patience[{rule.name}]: 市场状态不明确，不建议交易")

        # ── 规则 2: 波动率过低 ──
        rule = self.rules.get("volatility_too_low")
        if rule and rule.enabled and volatility is not None:
            if volatility < self.VOLATILITY_LOW_THRESHOLD:
                blocked.append(rule.name)
                max_wait = max(max_wait, rule.min_wait_minutes)
                patience_score += rule.weight * 0.1
                logger.info(f"🚫 Patience[{rule.name}]: 波动率={volatility:.2%} 过低，无趋势")

        # ── 规则 3: 波动率过高 ──
        rule = self.rules.get("volatility_too_high")
        if rule and rule.enabled and volatility is not None:
            if volatility > self.VOLATILITY_HIGH_THRESHOLD:
                blocked.append(rule.name)
                max_wait = max(max_wait, rule.min_wait_minutes)
                patience_score += rule.weight * 0.2
                logger.warning(f"🚫 Patience[{rule.name}]: 波动率={volatility:.2%} 过高，可能黑天鹅")

        # ── 规则 4: 信号置信度不足 ──
        rule = self.rules.get("low_confidence")
        if rule and rule.enabled and signal_confidence is not None:
            if signal_confidence < self.CONFIDENCE_THRESHOLD:
                blocked.append(rule.name)
                max_wait = max(max_wait, rule.min_wait_minutes)
                patience_score += rule.weight * 0.12
                logger.info(
                    f"🚫 Patience[{rule.name}]: "
                    f"置信度={signal_confidence:.2%} < {self.CONFIDENCE_THRESHOLD:.0%}"
                )

        # ── 规则 5: 临近重大事件 ──
        rule = self.rules.get("near_major_event")
        if rule and rule.enabled and fear_greed is not None:
            if (fear_greed < self.FEAR_GREED_EXTREME_LOW
                    or fear_greed > self.FEAR_GREED_EXTREME_HIGH):
                warned.append(rule.name)
                max_wait = max(max_wait, rule.min_wait_minutes)
                patience_score += rule.weight * 0.08
                logger.info(f"⚠️ Patience[{rule.name}]: FGI={fear_greed} 极端值，谨慎")

        # ── 规则 6: 流动性不足 ──
        rule = self.rules.get("low_liquidity")
        if rule and rule.enabled and liquidity_score is not None:
            if liquidity_score < 0.3:
                blocked.append(rule.name)
                max_wait = max(max_wait, rule.min_wait_minutes)
                patience_score += rule.weight * 0.15
                logger.info(f"🚫 Patience[{rule.name}]: 流动性评分={liquidity_score:.2f} 不足")

        # ── 规则 7: 冷却期 ──
        rule = self.rules.get("cooldown_active")
        if rule and rule.enabled:
            remaining = self._check_cooldown()
            if remaining > 0:
                blocked.append(rule.name)
                max_wait = max(max_wait, int(remaining))
                patience_score += rule.weight * 0.2
                logger.info(f"🚫 Patience[{rule.name}]: 冷却剩余 {remaining:.0f} 分钟")

        # ── 规则 8: 心理状态危险 ──
        rule = self.rules.get("psychology_danger")
        if rule and rule.enabled and psychology_risk == "danger":
            blocked.append(rule.name)
            max_wait = max(max_wait, rule.min_wait_minutes)
            patience_score += rule.weight * 0.18
            logger.info(f"🚫 Patience[{rule.name}]: 心理状态危险，强制休息")

        # ── 规则 9: 周末 ──
        rule = self.rules.get("weekend_holiday")
        if rule and rule.enabled and is_weekend:
            warned.append(rule.name)
            patience_score += rule.weight * 0.05

        # ── 规则 10: 近期大亏 ──
        rule = self.rules.get("recent_big_loss")
        if rule and rule.enabled and recent_pnl_pct is not None:
            if recent_pnl_pct < -5.0:
                blocked.append(rule.name)
                max_wait = max(max_wait, rule.min_wait_minutes)
                patience_score += rule.weight * 0.15
                logger.info(f"🚫 Patience[{rule.name}]: 近期亏损 {recent_pnl_pct:+.2f}%")

        # ── 市场状态加分 ──
        if regime_state in self.CLEAR_STATES:
            patience_score -= 0.1  # 明确趋势，降低耐心（更激进）
        elif regime_state in self.CAUTION_STATES:
            patience_score += 0.05
        elif regime_state == "EXTREME_GREED":
            patience_score += 0.15  # 极端贪婪，高度耐心
            warned.append("极端贪婪期，耐心等待回调")

        # ── 限制范围 ──
        patience_score = max(0.0, min(1.0, patience_score))

        # ── 综合决策 ──
        # 检查是否有 blocking 规则被触发
        blocking_rules = [
            self.rules[name] for name in blocked
            if name in self.rules and self.rules[name].block
        ]

        if blocking_rules:
            ok = False
            priority = "block"
            reasons = [f"❌ {r}" for r in blocked]
            # 去重
            reasons += [f"⚠️ {r}" for r in warned if r not in blocked]
            reason = " | ".join(reasons)
        elif warned:
            ok = True
            priority = "caution"
            reason = "⚠️ " + " | ".join(warned) + " — 可交易但需谨慎"
        else:
            ok = True
            priority = "go"
            reason = "✅ 所有门控通过，可以交易"

        if ok and max_wait > 0:
            # 有警告但不阻止
            max_wait = 0

        result = PatienceResult(
            ok_to_trade=ok,
            reason=reason,
            wait_time_minutes=max_wait,
            priority=priority,
            patience_score=round(patience_score, 3),
            blocked_rules=blocked,
            warnings=warned,
        )

        if not ok:
            logger.warning(
                f"🚫 PatienceGate: BLOCKED — 等待 {max_wait} 分钟 | "
                f"耐心分={patience_score:.2f} | {reason}"
            )
        elif priority == "caution":
            logger.info(f"🟡 PatienceGate: CAUTION — {reason}")
        else:
            logger.info(f"🟢 PatienceGate: GO — {reason}")

        return result

    def set_cooldown(self, minutes: int, reason: str = ""):
        """设置冷却期"""
        self._last_block_time["manual"] = datetime.now(timezone.utc)
        self._cooldown_minutes = minutes
        logger.info(f"🔒 冷却期设置: {minutes} 分钟 | {reason}")

    def _check_cooldown(self) -> float:
        """检查冷却期，返回剩余分钟数"""
        last = self._last_block_time.get("manual")
        if not last:
            return 0
        cooldown = getattr(self, "_cooldown_minutes", 60)
        remaining = cooldown - (datetime.now(timezone.utc) - last).total_seconds() / 60
        return max(0, remaining)

    def enable_rule(self, name: str, enabled: bool = True):
        """启用/禁用规则"""
        if name in self.rules:
            self.rules[name].enabled = enabled
            logger.info(f"Patience rule '{name}': {'enabled' if enabled else 'disabled'}")

    def add_custom_rule(self, rule: PatienceRule):
        """添加自定义规则"""
        self.rules[rule.name] = rule
        logger.info(f"Custom patience rule added: {rule.name}")

    def get_status(self) -> str:
        """获取当前门控状态摘要"""
        enabled = {k: v for k, v in self.rules.items() if v.enabled}
        blocked_names = [k for k, v in enabled.items() if v.block]
        warn_names = [k for k, v in enabled.items() if not v.block]
        cooldown_remaining = self._check_cooldown()

        lines = [
            "## 🚦 Patience Gate 状态",
            f"冷却期剩余: {cooldown_remaining:.0f} 分钟",
            f"启用规则: {len(enabled)}/{len(self.rules)}",
            f"阻止型规则: {', '.join(blocked_names) if blocked_names else '无'}",
            f"警告型规则: {', '.join(warn_names) if warn_names else '无'}",
        ]
        return "\n".join(lines)
