"""Trading Psychology Detector — 交易心理检测器

Mark Douglas 核心教导:
"交易失败 80% 是心理问题，不是策略问题"

4 种危险心理状态:
1. 报复性交易 (Revenge Trading) — 连亏后加大仓位想翻本
2. FOMO 追涨 (Fear of Missing Out) — 极度贪婪时强行开仓
3. 恐慌抛售 (Panic Selling) — 极度恐惧时割肉离场
4. 过度自信 (Overconfidence) — 连赚后放松风控

核心功能:
- 检测系统自身或用户的危险心理状态
- 触发心理保护机制（强制冷却/降低仓位/暂停交易）
- 生成心理状态报告
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from loguru import logger

from core.storage import Storage


@dataclass
class PsychologyState:
    """心理状态快照"""
    overall_risk: str         # "safe", "caution", "danger"
    revenge_trading: bool
    fomo_level: float         # 0.0-1.0
    panic_level: float        # 0.0-1.0
    overconfidence: float     # 0.0-1.0
    consecutive_losses: int
    consecutive_wins: int
    total_trades_today: int
    daily_pnl_pct: float
    fear_greed: float | None = None  # 恐惧贪婪指数快照
    warnings: list[str] = field(default_factory=list)
    cooldown_minutes: int = 0   # 建议冷却时间
    position_restriction: float = 1.0  # 仓位限制系数 (0.0-1.0)


class PsychologyDetector:
    """交易心理检测器"""

    # 心理阈值
    REVENGE_THRESHOLD = 3       # 连亏3次触发报复性交易检测
    FOMO_THRESHOLD = 85         # FGI > 85 触发 FOMO
    PANIC_THRESHOLD = 15        # FGI < 15 触发恐慌
    OVERCONFIDENCE_THRESHOLD = 5 # 连赢5次触发过度自信
    DAILY_TRADE_LIMIT = 10       # 单日交易次数上限
    COOLDOWN_AFTER_REVENGE = 120  # 报复性交易后冷却 2 小时
    COOLDOWN_AFTER_PANIC = 240    # 恐慌后冷却 4 小时

    def __init__(self, storage: Storage):
        self.storage = storage

    def detect(
        self,
        consecutive_losses: int = 0,
        consecutive_wins: int = 0,
        daily_pnl_pct: float = 0,
        daily_trade_count: int = 0,
        fear_greed: float | None = None,
        total_position_pct: float = 0,
    ) -> PsychologyState:
        """
        检测当前心理状态

        Args:
            consecutive_losses: 连续亏损次数
            consecutive_wins: 连续盈利次数
            daily_pnl_pct: 今日盈亏百分比
            daily_trade_count: 今日交易次数
            fear_greed: 恐惧贪婪指数
            total_position_pct: 当前总仓位占比
        """
        warnings = []
        revenge = False
        fomo = 0.0
        panic = 0.0
        overconf = 0.0
        cooldown = 0
        pos_restriction = 1.0

        # ── 1. 报复性交易检测 ──
        if consecutive_losses >= self.REVENGE_THRESHOLD:
            revenge = True
            cooldown = max(cooldown, self.COOLDOWN_AFTER_REVENGE)
            pos_restriction *= 0.3  # 仓位降到 30%
            warnings.append(
                f"⚠️ 报复性交易风险！连亏 {consecutive_losses} 次，"
                f"建议冷却 {cooldown} 分钟，仓位降至 30%"
            )

        # ── 2. FOMO 检测 ──
        if fear_greed is not None:
            if fear_greed >= self.FOMO_THRESHOLD:
                fomo = min(1.0, (fear_greed - 80) / 20)
                cooldown = max(cooldown, 60)
                pos_restriction *= 0.5
                warnings.append(
                    f"⚠️ FOMO 警告！FGI={fear_greed}（极度贪婪），"
                    f"此时开仓容易被套在顶部"
                )
            elif fear_greed >= 75:
                fomo = min(0.5, (fear_greed - 70) / 20)
                pos_restriction *= 0.8

        # ── 3. 恐慌检测 ──
        if fear_greed is not None:
            if fear_greed <= self.PANIC_THRESHOLD:
                panic = min(1.0, (20 - fear_greed) / 20)
                cooldown = max(cooldown, self.COOLDOWN_AFTER_PANIC)
                # 注意: 恐慌时是定投的最佳时机，但不宜主动交易
                warnings.append(
                    f"⚠️ 恐慌情绪！FGI={fear_greed}（极度恐慌），"
                    f"不建议在此情绪下平仓。冷静 {cooldown} 分钟。"
                )
            elif fear_greed <= 25:
                panic = min(0.4, (30 - fear_greed) / 20)

        # ── 4. 过度自信检测 ──
        if consecutive_wins >= self.OVERCONFIDENCE_THRESHOLD:
            overconf = min(1.0, consecutive_wins / 8)
            pos_restriction *= 0.7
            warnings.append(
                f"⚠️ 过度自信！连赢 {consecutive_wins} 次，"
                f"放松风控是大亏的前兆。建议降低仓位。"
            )

        # ── 5. 过度交易检测 ──
        if daily_trade_count >= self.DAILY_TRADE_LIMIT:
            cooldown = max(cooldown, 180)
            warnings.append(
                f"⚠️ 过度交易！今日已交易 {daily_trade_count} 次，"
                f"超过上限 {self.DAILY_TRADE_LIMIT}。停止交易。"
            )

        # ── 6. 大仓位检测 ──
        if total_position_pct > 0.7:
            warnings.append(
                f"⚠️ 仓位过重 ({total_position_pct:.0%})！"
                f"留给意外情况的空间不足。"
            )

        # ── 7. 单日大亏检测 ──
        if daily_pnl_pct < -2.0:
            cooldown = max(cooldown, 90)
            pos_restriction *= 0.5
            warnings.append(
                f"⚠️ 今日亏损 {daily_pnl_pct:.2f}%，"
                f"情绪可能影响判断，建议冷静 {cooldown} 分钟"
            )

        # ── 7. 单日大赚也要小心 ──
        elif daily_pnl_pct > 5.0:
            pos_restriction *= 0.8
            warnings.append(
                f"💰 今日大赚 {daily_pnl_pct:.2f}%！"
                f"大赚后容易放松警惕，注意保持纪律"
            )

        # ── 综合评估 ──
        risk_scores = [
            (1.0 if revenge else 0, "revenge"),
            (fomo, "fomo"),
            (panic, "panic"),
            (overconf, "overconf"),
        ]
        max_risk = max(score for score, _ in risk_scores)

        if max_risk >= 0.7 or cooldown >= 120:
            overall = "danger"
        elif max_risk >= 0.3 or cooldown >= 60:
            overall = "caution"
        else:
            overall = "safe"

        # 检查冷却期
        last_warning = self.storage.get_state("psychology_cooldown_until")
        if last_warning:
            try:
                cooldown_until = datetime.fromisoformat(last_warning)
                remaining = (cooldown_until - datetime.now(timezone.utc)).total_seconds() / 60
                if remaining > 0:
                    cooldown = max(cooldown, int(remaining))
            except:
                pass

        if cooldown > 0:
            cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=cooldown)
            self.storage.set_state("psychology_cooldown_until", cooldown_until.isoformat())
        else:
            self.storage.set_state("psychology_cooldown_until", "")

        state = PsychologyState(
            overall_risk=overall,
            revenge_trading=revenge,
            fomo_level=round(fomo, 2),
            panic_level=round(panic, 2),
            overconfidence=round(overconf, 2),
            consecutive_losses=consecutive_losses,
            consecutive_wins=consecutive_wins,
            total_trades_today=daily_trade_count,
            daily_pnl_pct=daily_pnl_pct,
            warnings=warnings,
            cooldown_minutes=cooldown,
            position_restriction=round(pos_restriction, 2),
        )

        if warnings:
            for w in warnings:
                logger.warning(f"🧠 Psychology: {w}")

        if overall == "danger":
            logger.warning("🚫 Psychology: DANGER state! Trading restricted.")
        elif overall == "caution":
            logger.info("🟡 Psychology: CAUTION — trade carefully.")

        return state

    def is_allowed_to_trade(self) -> tuple[bool, str]:
        """快速检查是否允许交易（考虑冷却期）"""
        cooldown_until = self.storage.get_state("psychology_cooldown_until")
        if not cooldown_until:
            return True, ""

        try:
            until = datetime.fromisoformat(cooldown_until)
            remaining = (until - datetime.now(timezone.utc)).total_seconds() / 60
            if remaining > 0:
                return False, f"心理冷却期中，剩余 {remaining:.0f} 分钟"
        except:
            pass

        return True, ""

    def generate_report(self) -> str:
        """生成心理状态报告"""
        state = self._get_latest_state()
        if not state:
            return "暂无心理状态数据。系统会在交易过程中自动检测。"

        report = f"""## 🧠 交易心理状态报告

**综合评估**: {"🟢 安全" if state.overall_risk == "safe" else "🟡 注意" if state.overall_risk == "caution" else "🔴 危险"}
**连亏**: {state.consecutive_losses} 次
**连赢**: {state.consecutive_wins} 次
**今日交易**: {state.total_trades_today} 次
**今日盈亏**: {state.daily_pnl_pct:+.2f}%

### 心理指标
- 报复性交易: {"🔴 检测到" if state.revenge_trading else "🟢 正常"}
- FOMO 指数: {"🔴" if state.fomo_level > 0.7 else "🟡" if state.fomo_level > 0.3 else "🟢"} {state.fomo_level:.0%}
- 恐慌指数: {"🔴" if state.panic_level > 0.7 else "🟡" if state.panic_level > 0.3 else "🟢"} {state.panic_level:.0%}
- 过度自信: {"🔴" if state.overconfidence > 0.7 else "🟡" if state.overconfidence > 0.3 else "🟢"} {state.overconfidence:.0%}
- 仓位限制: {state.position_restriction:.0%}
- 冷却时间: {state.cooldown_minutes} 分钟
"""
        if state.warnings:
            report += "\n### ⚠️ 警告\n"
            for w in state.warnings:
                report += f"- {w}\n"

        return report

    def _get_latest_state(self) -> PsychologyState | None:
        """获取最近的心理状态"""
        cl = self.storage.get_state("consecutive_losses")
        cw = self.storage.get_state("consecutive_wins")
        dp = self.storage.get_state("daily_pnl_pct")
        dc = self.storage.get_state("daily_trade_count")
        fg = self.storage.get_state("latest_fear_greed")

        if cl is None and cw is None:
            return None

        return PsychologyState(
            overall_risk="safe",
            revenge_trading=False,
            fomo_level=0.0,
            panic_level=0.0,
            overconfidence=0.0,
            consecutive_losses=int(cl) if cl else 0,
            consecutive_wins=int(cw) if cw else 0,
            daily_pnl_pct=float(dp) if dp else 0,
            total_trades_today=int(dc) if dc else 0,
            fear_greed=float(fg) if fg else None,
        )

    def record_trade_result(self, pnl_pct: float):
        """记录交易结果（更新心理状态计数器）"""
        # 更新连亏/连赢
        cl = int(self.storage.get_state("consecutive_losses") or 0)
        cw = int(self.storage.get_state("consecutive_wins") or 0)

        if pnl_pct > 0:
            cw += 1
            cl = 0
        else:
            cl += 1
            cw = 0

        self.storage.set_state("consecutive_losses", str(cl))
        self.storage.set_state("consecutive_wins", str(cw))

        # 更新今日盈亏
        dp = float(self.storage.get_state("daily_pnl_pct") or 0)
        dp += pnl_pct
        self.storage.set_state("daily_pnl_pct", str(dp))

        # 更新今日交易次数
        dc = int(self.storage.get_state("daily_trade_count") or 0)
        dc += 1
        self.storage.set_state("daily_trade_count", str(dc))

    def reset_daily(self):
        """每日重置"""
        self.storage.set_state("daily_pnl_pct", "0")
        self.storage.set_state("daily_trade_count", "0")
        logger.info("Psychology daily counters reset")
