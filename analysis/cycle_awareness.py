"""Cycle Awareness — 周期感知（4年减半周期）

核心理念 (来源: 行业共识 / PlanB / Willy Woo):
- 比特币约每 4 年减半一次，历史形成清晰的 4 年周期
- 周期阶段:
  1. 积累期 (底部) — 减半后 12-18 个月，BTC 在底部震荡
  2. 上升期 (牛市) — 减半后 12-24 个月，BTC 开始主升浪
  3. 狂热期 (顶部) — 减半后 18-30 个月，市场极度 FOMO
  4. 分发期 (下跌) — 减半后 30-48 个月，价格持续下跌

历史数据:
- 2012 减半: $12 → 2013顶 $1,100 (91x) → 2015底 $200
- 2016 减半: $650 → 2017顶 $19,800 (30x) → 2018底 $3,200
- 2020 减半: $8,700 → 2021顶 $69,000 (8x) → 2022底 $15,500
- 2024 减半: ~$64,000 → 预计 2025 顶 $150,000-300,000 → 2026底 ~$40,000-60,000

关键洞察:
- 每个周期的涨幅在递减（91x → 30x → 8x → 预计 2-4x）
- 周期不是精确的，会偏移
- 周期感知用于调整系统整体激进程度，不是精确择时

输出: CyclePhase — 当前在周期中的位置和建议激进程度
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from loguru import logger


@dataclass
class CyclePhase:
    """周期阶段"""
    phase: str                    # "accumulation" / "markup" / "distribution" / "markdown"
    phase_cn: str                 # 中文描述
    months_into_cycle: int        # 距上次减半的月数
    cycle_length_months: int      # 预计周期总长度（月）
    progress_pct: float           # 周期进度 (0-100%)
    expected_peak_month: int      # 预计顶部月份（距减半）
    aggressiveness: float         # 激进程度 (0.0=极度保守, 1.0=极度激进)
    position_multiplier: float    # 仓位倍数 (0.3-1.5)
    dca_multiplier: float         # DCA 倍数 (0.3-2.0)
    summary: str                  # 人可读总结


# ── 历史减半数据 ──
HALVING_HISTORY = [
    {"date": "2012-11-28", "block": 210000, "price": 12.0},
    {"date": "2016-07-09", "block": 420000, "price": 650.0},
    {"date": "2020-05-11", "block": 630000, "price": 8700.0},
    {"date": "2024-04-20", "block": 840000, "price": 64000.0},
]

# 预计下次减半
NEXT_HALVING = {"date": "2028-04-01", "block": 1050000}


class CycleAwareness:
    """
    周期感知器

    用法:
        cycle = CycleAwareness()
        phase = cycle.get_current_phase()
        logger.info(f"当前周期阶段: {phase.phase_cn}, 激进度={phase.aggressiveness:.2f}")
    """

    # 周期参数（基于历史数据统计）
    CYCLE_LENGTH_MONTHS = 48          # 约 48 个月一个完整周期
    ACCUMULATION_END_MONTH = 12       # 积累期约 12 个月
    MARKUP_START_MONTH = 12           # 上升期开始
    MARKUP_PEAK_MONTH = 24            # 历史顶部约在减半后 18-24 个月
    DISTRIBUTION_END_MONTH = 30       # 分发期约到 30 个月
    MARKDOWN_END_MONTH = 48           # 下跌期到下次减半

    def __init__(self, halvings: list[dict] | None = None):
        self.halvings = halvings or HALVING_HISTORY.copy()
        self.last_halving = self._parse_date(self.halvings[-1]["date"])
        self.next_halving = self._parse_date(NEXT_HALVING["date"])

    def get_current_phase(self, current_btc_price: float | None = None) -> CyclePhase:
        """
        获取当前周期阶段

        Args:
            current_btc_price: 当前 BTC 价格（可选，用于增强判断）

        Returns:
            CyclePhase
        """
        now = datetime.now(timezone.utc)
        months_since_halving = self._months_between(self.last_halving, now)
        cycle_length = self._months_between(self.last_halving, self.next_halving)
        progress = min(1.0, months_since_halving / cycle_length)

        # ── 判断周期阶段 ──
        phase, phase_cn = self._determine_phase(months_since_halving)

        # ── 计算激进程度 ──
        aggressiveness = self._calculate_aggressiveness(
            months_since_halving, phase, current_btc_price
        )

        # ── 仓位倍数 ──
        position_mult = 0.3 + aggressiveness * 1.2  # 0.3 - 1.5
        dca_mult = self._calculate_dca_multiplier(months_since_halving, phase)

        # ── 预计顶部月份 ──
        peak_month = self.MARKUP_PEAK_MONTH

        # ── 生成总结 ──
        summary = self._generate_summary(
            phase_cn, months_since_halving, cycle_length,
            aggressiveness, current_btc_price,
        )

        result = CyclePhase(
            phase=phase,
            phase_cn=phase_cn,
            months_into_cycle=months_since_halving,
            cycle_length_months=cycle_length,
            progress_pct=round(progress * 100, 1),
            expected_peak_month=peak_month,
            aggressiveness=round(aggressiveness, 3),
            position_multiplier=round(position_mult, 2),
            dca_multiplier=round(dca_mult, 2),
            summary=summary,
        )

        logger.info(
            f"🕐 周期: {phase_cn} | {months_since_halving}个月/共{cycle_length}个月 | "
            f"进度={progress:.0%} | 激进度={aggressiveness:.2f}"
        )

        return result

    def _determine_phase(self, months: int) -> tuple[str, str]:
        """根据月份判断周期阶段"""
        if months <= self.ACCUMULATION_END_MONTH:
            return "accumulation", "积累期（底部震荡）"
        elif months <= self.MARKUP_PEAK_MONTH:
            return "markup", "上升期（牛市主升浪）"
        elif months <= self.DISTRIBUTION_END_MONTH:
            return "distribution", "分发期（高位震荡）"
        else:
            return "markdown", "下跌期（熊市回调）"

    def _calculate_aggressiveness(
        self,
        months: int,
        phase: str,
        btc_price: float | None = None,
    ) -> float:
        """
        计算激进程度

        基于周期位置和（可选的）价格位置
        """
        base = 0.5  # 基础激进度

        if phase == "accumulation":
            # 积累期：适度保守，但可以开始建仓
            base = 0.4
            # 越接近上升期，越激进
            progress = months / self.ACCUMULATION_END_MONTH
            base += progress * 0.2

        elif phase == "markup":
            # 上升期：应该较激进
            progress = (months - self.MARKUP_START_MONTH) / (
                self.MARKUP_PEAK_MONTH - self.MARKUP_START_MONTH
            )
            # 前半段更激进，后半段逐步减仓
            if progress < 0.5:
                base = 0.8
            elif progress < 0.8:
                base = 0.6
            else:
                base = 0.3  # 接近顶部，大幅减仓

        elif phase == "distribution":
            # 分发期：保守，逐步退出
            progress = (months - self.MARKUP_PEAK_MONTH) / (
                self.DISTRIBUTION_END_MONTH - self.MARKUP_PEAK_MONTH
            )
            base = 0.3 - progress * 0.2  # 0.3 → 0.1

        elif phase == "markdown":
            # 下跌期：极度保守
            progress = (months - self.DISTRIBUTION_END_MONTH) / (
                self.MARKDOWN_END_MONTH - self.DISTRIBUTION_END_MONTH
            )
            if progress < 0.5:
                base = 0.1  # 刚开始跌，等待
            else:
                base = 0.2  # 接近底部，开始小仓定投

        # 价格辅助判断（如果有当前价格）
        if btc_price is not None and self.halvings:
            halving_price = self.halvings[-1]["price"]
            if halving_price > 0:
                gain_from_halving = (btc_price / halving_price - 1) * 100

                # 涨幅过大时降低激进程度
                if gain_from_halving > 300:
                    base *= 0.5
                    logger.info(f"周期校准: 从减半价涨幅={gain_from_halving:.0f}% 过大，降低激进程度")
                elif gain_from_halving > 200:
                    base *= 0.7
                elif gain_from_halving < -30:
                    # 从减半价跌了 30%，可能接近底部
                    base = max(base, 0.3)
                    logger.info(f"周期校准: 从减半价跌幅={gain_from_halving:.0f}%，可能接近底部")

        return max(0.0, min(1.0, base))

    def _calculate_dca_multiplier(self, months: int, phase: str) -> float:
        """计算 DCA 倍数"""
        if phase == "accumulation":
            # 积累期是最好的定投时机
            return 1.5
        elif phase == "markup":
            # 上升期维持定投，但不要加太多
            return 1.0
        elif phase == "distribution":
            # 分发期减少定投
            return 0.5
        else:
            # 下跌期加大定投（逢低买入）
            return 2.0

    def _generate_summary(
        self,
        phase_cn: str,
        months: int,
        cycle_length: int,
        aggressiveness: float,
        btc_price: float | None,
    ) -> str:
        """生成人可读的周期总结"""
        lines = [
            f"## 📅 比特币周期分析",
            f"**当前阶段**: {phase_cn}",
            f"**周期进度**: {months} 个月 / 共 {cycle_length} 个月",
        ]

        if btc_price:
            lines.append(f"**BTC 当前价格**: ${btc_price:,.0f}")

        halving_str = self.last_halving.strftime("%Y-%m-%d")
        next_str = self.next_halving.strftime("%Y-%m-%d")
        lines.append(f"**上次减半**: {halving_str}")
        lines.append(f"**下次减半**: {next_str}")

        # 周期特征描述
        phase_descriptions = {
            "accumulation": (
                "📈 积累期特征：市场低迷，机构默默建仓，波动率低。\n"
                "策略：这是最好的定投和分批建仓时机。不要试图精确抄底。"
            ),
            "markup": (
                "🚀 上升期特征：市场热度升温，资金持续流入，新高不断。\n"
                "策略：持有为主，可适当加仓。注意不要 FOMO 追涨。"
            ),
            "distribution": (
                "⚠️ 分发期特征：市场极度亢奋，大户开始出货，波动加剧。\n"
                "策略：逐步减仓锁定利润。不要贪心。这是最危险的阶段。"
            ),
            "markdown": (
                "📉 下跌期特征：恐慌蔓延，持续阴跌，市场情绪极度悲观。\n"
                "策略：空仓等待，或小额定投。最忌讳抄底抄在半山腰。"
            ),
        }

        phase = self._determine_phase(months)[0]
        lines.append(f"\n{phase_descriptions.get(phase, '')}")

        return "\n".join(lines)

    def _parse_date(self, date_str: str) -> datetime:
        """解析日期字符串"""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return datetime(2024, 4, 20, tzinfo=timezone.utc)

    def _months_between(self, start: datetime, end: datetime) -> int:
        """计算两个日期之间的月数"""
        return (end.year - start.year) * 12 + (end.month - start.month)

    def get_historical_performance(self) -> str:
        """获取历史周期表现"""
        lines = ["## 📊 历史周期表现"]

        for i, h in enumerate(self.halvings):
            halving_date = self._parse_date(h["date"])
            if i < len(self.halvings) - 1:
                next_h = self.halvings[i + 1]
                # 找到这个周期内的最高价（简化：使用已知数据）
                next_halving_date = self._parse_date(next_h["date"])

                # 历史峰值（简化数据）
                peaks = {
                    0: {"date": "2013-11", "price": 1100, "gain": "91x"},
                    1: {"date": "2017-12", "price": 19800, "gain": "30x"},
                    2: {"date": "2021-11", "price": 69000, "gain": "8x"},
                }

                if i in peaks:
                    p = peaks[i]
                    lines.append(
                        f"- **{halving_date.strftime('%Y-%m')}减半** "
                        f"(${h['price']:,.0f}) → "
                        f"**{p['date']}顶** "
                        f"(${p['price']:,.0f}, {p['gain']})"
                    )

        # 当前周期
        latest = self.halvings[-1]
        now = datetime.now(timezone.utc)
        months = self._months_between(self._parse_date(latest["date"]), now)
        lines.append(
            f"- **{latest['date']}减半** "
            f"(${latest['price']:,.0f}) → "
            f"当前进行中 ({months} 个月)"
        )

        return "\n".join(lines)
