"""Signal fusion — multi-model voting"""
from __future__ import annotations

from loguru import logger

from core.models import TradeSignal, FusedSignal, SignalDirection, RiskLevel


class SignalFusion:
    """多信号融合器：LLM 信号 + 技术信号 → 最终决策"""

    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold

    def fuse(self, signals: list[TradeSignal]) -> FusedSignal:
        """融合多个信号，输出最终交易决策

        融合策略：
        - 加权投票：DeepSeek 权重 0.4, Qwen 权重 0.3, Technical 权重 0.3
        - 置信度加权平均
        - 所有模型方向一致时，提升 10% 置信度
        - 任一模型 HIGH risk 时，降低 15% 置信度
        """
        if not signals:
            return FusedSignal(
                direction=SignalDirection.FLAT,
                confidence=0.0,
                signals=[],
                rationale="无可用信号",
            )

        weights = {
            "deepseek": 0.4,
            "qwen": 0.3,
            "technical": 0.3,
        }

        long_score = 0.0
        flat_score = 0.0
        short_score = 0.0
        total_weight = 0.0
        all_rationales = []
        high_risk = False

        for sig in signals:
            w = weights.get(sig.source, 0.2)
            total_weight += w

            if sig.direction == SignalDirection.LONG:
                long_score += sig.confidence * w
            elif sig.direction == SignalDirection.FLAT:
                flat_score += sig.confidence * w
            elif sig.direction == SignalDirection.SHORT:
                short_score += sig.confidence * w

            if sig.rationale:
                all_rationales.append(f"[{sig.source}] {sig.rationale}")

            if sig.risk_level == RiskLevel.HIGH:
                high_risk = True

        # 归一化
        if total_weight > 0:
            long_score /= total_weight
            flat_score /= total_weight
            short_score /= total_weight

        # 决策
        scores = {
            SignalDirection.LONG: long_score,
            SignalDirection.FLAT: flat_score,
            SignalDirection.SHORT: short_score,
        }
        best_direction = max(scores, key=scores.get)
        final_confidence = scores[best_direction]

        # 一致性加成
        directions = [s.direction for s in signals]
        if len(set(directions)) == 1:
            final_confidence *= 1.1
            logger.info("All models agree — confidence boosted")

        # 高风险惩罚
        if high_risk:
            final_confidence *= 0.85
            logger.info("HIGH risk detected — confidence reduced")

        # 限制范围
        final_confidence = min(1.0, max(0.0, final_confidence))

        # 构建理由
        rationale = f"信号融合: LONG={long_score:.2f} FLAT={flat_score:.2f} SHORT={short_score:.2f}\n"
        for r in all_rationales:
            rationale += f"  {r}\n"
        if high_risk:
            rationale += "⚠️ 检测到高风险因素，置信度已降低"

        fused = FusedSignal(
            direction=best_direction,
            confidence=round(final_confidence, 3),
            signals=signals,
            rationale=rationale.strip(),
            risk_level=RiskLevel.HIGH if high_risk else RiskLevel.MEDIUM,
        )

        logger.info(
            f"Fused signal: {best_direction.value} "
            f"confidence={final_confidence:.3f} "
            f"(threshold={self.confidence_threshold})"
        )

        # 判断是否触发交易
        if final_confidence >= self.confidence_threshold and best_direction != SignalDirection.FLAT:
            logger.info(f"✅ Signal passes threshold — TRADE SIGNAL!")
        else:
            logger.info(f"❌ Signal below threshold — NO TRADE")

        return fused
