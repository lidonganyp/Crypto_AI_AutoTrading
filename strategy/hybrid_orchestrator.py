"""Hybrid Orchestrator — 混合策略编排器

核心理念 (来源: 行业最佳实践):
- 最好的策略不是单一策略，而是"80% 被动 + 20% 主动"的混合
- CZ: "大多数人最好的策略就是定期定额买入并忘记密码"
- Dalio: "分散化是唯一免费的午餐"
- Douglas: "但要确保你的主动交易不会毁掉被动收益"

设计逻辑:
- 统一编排 DCA 定投 + 主动交易策略
- 资金分配: 默认 80% DCA + 20% 主动，根据市场状态动态调整
- 决策流程（7步漏斗）:
  1. 宏观环境评估 (macro.py)
  2. 市场状态检测 (market_regime.py)
  3. 周期位置判断 (cycle_awareness.py)
  4. 心理状态检测 (psychology_detector.py)
  5. 耐心门控 (patience_gate.py)
  6. 风控检查 (position_sizer.py + correlation_risk.py + liquidity_checker.py)
  7. 仓位计算 (position_sizer.py)

输出: OrchestratedDecision — 包含所有行动、风险评分和总结
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from loguru import logger


@dataclass
class Action:
    """单个行动"""
    action_type: str       # "dca_buy" / "active_buy" / "active_sell" / "hold" / "close_all"
    symbol: str
    amount_usdt: float
    reason: str
    confidence: float = 0.0
    priority: str = "normal"  # "high" / "normal" / "low"


@dataclass
class OrchestratedDecision:
    """编排决策结果"""
    actions: list[Action] = field(default_factory=list)
    total_risk_pct: float = 0.0
    dca_amount: float = 0.0
    active_amount: float = 0.0
    dca_ratio: float = 0.80
    active_ratio: float = 0.20
    summary: str = ""
    risk_level: str = "safe"  # "safe" / "caution" / "danger"
    blocked: bool = False
    block_reason: str = ""
    macro_score: float = 0.0
    cycle_phase: str = ""
    patience_ok: bool = True
    psychology_ok: bool = True
    timestamp: str = ""


class HybridOrchestrator:
    """
    混合策略编排器

    用法:
        orchestrator = HybridOrchestrator(
            dca_engine=dca,
            psychology_detector=psychology,
            position_sizer=sizer,
            macro_analyzer=macro,
            cycle_awareness=cycle,
            patience_gate=gate,
            correlation_manager=corr,
            liquidity_checker=liq,
        )
        decision = orchestrator.decide()
        for action in decision.actions:
            execute(action)
    """

    def __init__(
        self,
        dca_engine=None,
        psychology_detector=None,
        position_sizer=None,
        macro_analyzer=None,
        cycle_awareness=None,
        patience_gate=None,
        correlation_manager=None,
        liquidity_checker=None,
    ):
        self.dca_engine = dca_engine
        self.psychology_detector = psychology_detector
        self.position_sizer = position_sizer
        self.macro_analyzer = macro_analyzer
        self.cycle_awareness = cycle_awareness
        self.patience_gate = patience_gate
        self.correlation_manager = correlation_manager
        self.liquidity_checker = liquidity_checker

        self._base_dca_ratio = 0.80
        self._base_active_ratio = 0.20

    def decide(
        self,
        regime_state: str = "UNKNOWN",
        volatility: float | None = None,
        signal_confidence: float | None = None,
        signal_direction: str = "FLAT",
        fear_greed: float | None = None,
        current_positions: list[dict] | None = None,
        total_balance: float = 10000.0,
        btc_price: float | None = None,
    ) -> OrchestratedDecision:
        """
        执行完整的决策漏斗，输出编排后的决策

        这是系统的核心入口。所有模块的输出在这里汇聚。

        Args:
            regime_state: 当前市场状态
            volatility: 波动率
            signal_confidence: 信号置信度
            signal_direction: 信号方向 (LONG/SHORT/FLAT)
            fear_greed: 恐惧贪婪指数
            current_positions: 当前持仓列表
            total_balance: 总资金
            btc_price: 当前 BTC 价格

        Returns:
            OrchestratedDecision
        """
        now = datetime.now(timezone.utc).isoformat()
        actions = []
        warnings = []
        risk_scores = []

        logger.info("=" * 60)
        logger.info("🎯 开始混合策略编排决策...")
        logger.info("=" * 60)

        # ── Step 1: 宏观环境评估 ──
        macro_score = 0.0
        pos_adj = 1.0
        if self.macro_analyzer:
            try:
                macro_env = self.macro_analyzer.analyze()
                macro_score = macro_env.overall_score
                pos_adj = macro_env.position_adjustment
                logger.info(
                    f"  1️⃣ 宏观环境: {macro_env.overall_score:+.2f} "
                    f"(仓位系数={pos_adj:.1f}x)"
                )
                if not macro_env.bullish and macro_score < -0.3:
                    warnings.append("宏观环境看空，降低整体仓位")
            except Exception as e:
                logger.warning(f"  1️⃣ 宏观分析失败: {e}")

        # ── Step 2: 周期位置判断 ──
        cycle_phase = ""
        aggressiveness = 0.5
        dca_mult = 1.0
        if self.cycle_awareness:
            try:
                cycle = self.cycle_awareness.get_current_phase(btc_price)
                cycle_phase = cycle.phase_cn
                aggressiveness = cycle.aggressiveness
                dca_mult = cycle.dca_multiplier
                logger.info(
                    f"  2️⃣ 周期阶段: {cycle_phase} "
                    f"(激进度={aggressiveness:.2f}, DCA倍数={dca_mult:.1f}x)"
                )
            except Exception as e:
                logger.warning(f"  2️⃣ 周期分析失败: {e}")

        # ── Step 3: 心理状态检测 ──
        psychology_ok = True
        pos_restriction = 1.0
        if self.psychology_detector:
            try:
                psych_state = self.psychology_detector.detect(
                    fear_greed=fear_greed,
                    total_position_pct=sum(
                        p.get("weight", 0) for p in (current_positions or [])
                    ),
                )
                psychology_ok = psych_state.overall_risk != "danger"
                pos_restriction = psych_state.position_restriction
                logger.info(
                    f"  3️⃣ 心理状态: {psych_state.overall_risk} "
                    f"(仓位限制={pos_restriction:.0%})"
                )
                if not psychology_ok:
                    warnings.append("心理状态危险，交易受限")
            except Exception as e:
                logger.warning(f"  3️⃣ 心理检测失败: {e}")

        # ── Step 4: 耐心门控 ──
        patience_ok = True
        if self.patience_gate:
            try:
                patience_result = self.patience_gate.evaluate(
                    regime_state=regime_state,
                    volatility=volatility,
                    signal_confidence=signal_confidence,
                    fear_greed=fear_greed,
                    psychology_risk="danger" if not psychology_ok else "safe",
                )
                patience_ok = patience_result.ok_to_trade
                logger.info(
                    f"  4️⃣ 耐心门控: {patience_result.priority} "
                    f"({patience_result.patience_score:.2f})"
                )
                if not patience_ok:
                    warnings.append(f"耐心门控阻止: {patience_result.reason}")
            except Exception as e:
                logger.warning(f"  4️⃣ 耐心门控失败: {e}")

        # ── Step 5: 风控检查 ──
        if self.position_sizer:
            try:
                risk_status = self.position_sizer.get_risk_status(
                    current_positions=len(current_positions or []),
                )
                if not risk_status.is_trading_allowed:
                    patience_ok = False
                    warnings.append(f"风控熔断: {'; '.join(risk_status.warnings)}")
                    logger.warning(f"  5️⃣ 风控: BLOCKED")
                else:
                    logger.info(f"  5️⃣ 风控: OK")
            except Exception as e:
                logger.warning(f"  5️⃣ 风控检查失败: {e}")

        # ── 检查是否完全阻止 ──
        blocked = not patience_ok
        block_reason = "; ".join(warnings) if blocked else ""

        # ── 计算动态资金分配 ──
        # 基础比例根据周期和宏观调整
        dca_ratio = self._base_dca_ratio
        active_ratio = self._base_active_ratio

        # 牛市阶段适当提高主动比例
        if aggressiveness > 0.7:
            dca_ratio = 0.70
            active_ratio = 0.30
        # 熊市阶段提高 DCA 比例
        elif aggressiveness < 0.3:
            dca_ratio = 0.90
            active_ratio = 0.10

        # 宏观调整
        dca_amount_budget = total_balance * dca_ratio
        active_amount_budget = total_balance * active_ratio

        # ── Step 6: DCA 决策 ──
        if self.dca_engine and not blocked:
            try:
                dca_result = self.dca_engine.calculate_dca_amount(
                    regime_state=regime_state,
                    fear_greed=fear_greed,
                )
                dca_total = dca_result.get("total_usdt", 0)
                # 应用 DCA 倍数和仓位调整
                dca_total *= dca_mult * pos_adj * pos_restriction
                dca_total = min(dca_total, dca_amount_budget)

                if dca_total > 10:  # 最小定投金额
                    for symbol, amount in dca_result.get("allocations", {}).items():
                        adjusted_amount = amount * dca_mult * pos_adj * pos_restriction
                        if adjusted_amount > 1:
                            actions.append(Action(
                                action_type="dca_buy",
                                symbol=symbol,
                                amount_usdt=round(adjusted_amount, 2),
                                reason=f"DCA定投 ({dca_result.get('reason', '')})",
                                confidence=0.9,
                                priority="high",
                            ))
                    logger.info(f"  6️⃣ DCA: ${dca_total:.2f}")
            except Exception as e:
                logger.warning(f"  6️⃣ DCA 计算失败: {e}")

        # ── Step 7: 主动交易决策 ──
        if not blocked and patience_ok and signal_confidence and signal_confidence > 0.55:
            if signal_direction in ("LONG", "SHORT") and signal_confidence > 0.65:
                # 相关性检查
                corr_ok = True
                if self.correlation_manager and current_positions:
                    try:
                        corr_report = self.correlation_manager.assess(current_positions)
                        if corr_report.risk_level == "danger":
                            corr_ok = False
                            warnings.append("相关性风险过高，暂停主动交易")
                            logger.warning(f"  7️⃣ 相关性: DANGER")
                    except Exception:
                        pass

                if corr_ok:
                    # 主动交易金额
                    active_amount = active_amount_budget * signal_confidence * pos_adj * pos_restriction

                    actions.append(Action(
                        action_type=f"active_{signal_direction.lower()}",
                        symbol="BTC/USDT",  # 默认 BTC
                        amount_usdt=round(active_amount, 2),
                        reason=f"主动交易信号 (方向={signal_direction}, 置信度={signal_confidence:.2f})",
                        confidence=signal_confidence,
                        priority="normal",
                    ))
                    logger.info(f"  7️⃣ 主动交易: {signal_direction} ${active_amount:.2f}")

        # 如果没有任何行动，添加 hold
        if not actions:
            if blocked:
                actions.append(Action(
                    action_type="hold",
                    symbol="ALL",
                    amount_usdt=0,
                    reason=f"交易被阻止: {block_reason}",
                    priority="high",
                ))
            else:
                actions.append(Action(
                    action_type="hold",
                    symbol="ALL",
                    amount_usdt=0,
                    reason="无明确信号，保持当前仓位",
                    priority="low",
                ))

        # ── 综合风险评估 ──
        num_warnings = len(warnings)
        if blocked or num_warnings >= 3:
            risk_level = "danger"
        elif num_warnings >= 1:
            risk_level = "caution"
        else:
            risk_level = "safe"

        # 总风险敞口
        total_action_amount = sum(a.amount_usdt for a in actions)
        total_risk_pct = total_action_amount / total_balance if total_balance > 0 else 0

        # ── 生成总结 ──
        summary_lines = [
            f"{'🔴 阻止' if blocked else '🟡 谨慎' if risk_level == 'caution' else '🟢 正常'}",
            f"周期: {cycle_phase or 'N/A'} | 宏观: {macro_score:+.2f}",
            f"DCA比例: {dca_ratio:.0%} | 主动比例: {active_ratio:.0%}",
            f"行动: {len(actions)} 个",
        ]
        if warnings:
            summary_lines.append(f"警告: {'; '.join(warnings[:3])}")

        decision = OrchestratedDecision(
            actions=actions,
            total_risk_pct=round(total_risk_pct, 4),
            dca_amount=round(
                sum(a.amount_usdt for a in actions if a.action_type == "dca_buy"), 2
            ),
            active_amount=round(
                sum(a.amount_usdt for a in actions if "active" in a.action_type), 2
            ),
            dca_ratio=dca_ratio,
            active_ratio=active_ratio,
            summary="\n".join(summary_lines),
            risk_level=risk_level,
            blocked=blocked,
            block_reason=block_reason,
            macro_score=macro_score,
            cycle_phase=cycle_phase,
            patience_ok=patience_ok,
            psychology_ok=psychology_ok,
            timestamp=now,
        )

        logger.info("=" * 60)
        logger.info(f"🎯 编排决策完成: {risk_level.upper()}")
        for a in actions:
            logger.info(f"  → {a.action_type} {a.symbol} ${a.amount_usdt:.2f} | {a.reason}")
        logger.info("=" * 60)

        return decision

    def _log_step(self, step: int, name: str, status: str, detail: str = ""):
        """记录步骤日志"""
        msg = f"  {step}️⃣ {name}: {status}"
        if detail:
            msg += f" ({detail})"
        logger.info(msg)
