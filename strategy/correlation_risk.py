"""Correlation Risk — 相关性风控

核心理念 (来源: Ray Dalio / Bridgewater):
- "不要把所有鸡蛋放在一个篮子里" 是废话
- 真正重要的是 "不要把所有看起来不同的篮子放在同一辆卡车上"
- BTC 和 ETH 相关性常 >0.8，持两个不叫分散，叫双倍押注

设计逻辑:
1. 计算币种间的滚动相关性矩阵（30/60/90日窗口）
2. 高相关币种视为同一风险暴露
3. 相关币种的合计仓位不超过单资产上限
4. 定期更新相关性矩阵

输出: CorrelationRiskReport — 告诉你实际的风险暴露，而非表面的持仓分布
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from loguru import logger

import numpy as np
import pandas as pd


@dataclass
class CorrelationPair:
    """币种对的相关性"""
    symbol_a: str
    symbol_b: str
    correlation_30d: float
    correlation_60d: float
    correlation_90d: float
    avg_correlation: float
    risk_level: str  # "low" / "medium" / "high"


@dataclass
class EffectiveExposure:
    """有效风险暴露"""
    group_name: str              # 风险组名称（如 "主流币"）
    symbols: list[str]           # 包含的币种
    total_weight: float          # 合计权重
    max_allowed_weight: float    # 最大允许权重
    over_exposed: bool           # 是否过度暴露


@dataclass
class CorrelationRiskReport:
    """相关性风控报告"""
    total_exposure: float            # 总有效暴露
    max_allowed_exposure: float      # 最大允许暴露
    risk_level: str                  # "safe" / "caution" / "danger"
    pairs: list[CorrelationPair] = field(default_factory=list)
    exposures: list[EffectiveExposure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    matrix: dict = field(default_factory=dict)  # 原始相关性矩阵


class CorrelationRiskManager:
    """
    相关性风控管理器

    用法:
        manager = CorrelationRiskManager()
        report = manager.assess(positions, ohlcv_data)
        if report.risk_level == "danger":
            logger.warning(f"相关性风险过高: {report.warnings}")
    """

    # 相关性阈值
    HIGH_CORRELATION = 0.75     # > 0.75 视为高相关
    MEDIUM_CORRELATION = 0.50   # > 0.50 视为中等相关
    MAX_GROUP_EXPOSURE = 0.40   # 单风险组最大暴露 40%
    MAX_TOTAL_EXPOSURE = 0.80   # 总最大暴露 80%

    # 默认风险分组（基于行业知识）
    DEFAULT_GROUPS = {
        "主流币": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        "DeFi": ["UNI/USDT", "AAVE/USDT", "MKR/USDT", "LINK/USDT"],
        "L1公链": ["SOL/USDT", "AVAX/USDT", "NEAR/USDT"],
        "Meme": ["DOGE/USDT", "SHIB/USDT", "PEPE/USDT"],
    }

    def __init__(
        self,
        high_threshold: float = 0.75,
        medium_threshold: float = 0.50,
        max_group_exposure: float = 0.40,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.max_group_exposure = max_group_exposure
        self._correlation_cache: dict[tuple, float] = {}

    def calculate_correlation(
        self,
        series_a: list[float],
        series_b: list[float],
    ) -> float:
        """
        计算两个价格序列的相关系数

        Args:
            series_a: 币种A的收盘价序列
            series_b: 币种B的收盘价序列

        Returns:
            相关系数 (-1.0 到 1.0)
        """
        if len(series_a) < 10 or len(series_b) < 10:
            return 0.0

        # 取相同长度
        n = min(len(series_a), len(series_b))
        a = np.array(series_a[-n:])
        b = np.array(series_b[-n:])

        # 计算日收益率（消除趋势影响）
        returns_a = np.diff(a) / a[:-1]
        returns_b = np.diff(b) / b[:-1]

        if len(returns_a) < 5:
            return 0.0

        # 计算相关系数
        try:
            corr = np.corrcoef(returns_a, returns_b)[0, 1]
            if np.isnan(corr):
                return 0.0
            return float(corr)
        except Exception:
            return 0.0

    def build_correlation_matrix(
        self,
        price_data: dict[str, list[float]],
        windows: list[int] = None,
    ) -> dict:
        """
        构建多币种相关性矩阵

        Args:
            price_data: {symbol: [close_prices]}
            windows: 计算窗口（天数），默认 [30, 60, 90]

        Returns:
            {symbol_a: {symbol_b: {window: correlation}}}
        """
        windows = windows or [30, 60, 90]
        symbols = list(price_data.keys())
        matrix = {}

        for i, sym_a in enumerate(symbols):
            matrix[sym_a] = {}
            for j, sym_b in enumerate(symbols):
                if i == j:
                    matrix[sym_a][sym_b] = {w: 1.0 for w in windows}
                    continue

                # 避免重复计算
                cache_key = tuple(sorted([sym_a, sym_b]))
                if cache_key in self._correlation_cache:
                    matrix[sym_a][sym_b] = self._correlation_cache[cache_key]
                    continue

                window_corrs = {}
                for w in windows:
                    corr = self.calculate_correlation(
                        price_data[sym_a][-w:] if len(price_data[sym_a]) >= w else price_data[sym_a],
                        price_data[sym_b][-w:] if len(price_data[sym_b]) >= w else price_data[sym_b],
                    )
                    window_corrs[w] = round(corr, 4)

                matrix[sym_a][sym_b] = window_corrs
                self._correlation_cache[cache_key] = window_corrs

        logger.info(f"🔗 相关性矩阵已构建: {len(symbols)} 个币种, {len(windows)} 个窗口")
        return matrix

    def assess(
        self,
        positions: list[dict],
        price_data: dict[str, list[float]] | None = None,
        correlation_matrix: dict | None = None,
    ) -> CorrelationRiskReport:
        """
        评估当前持仓的相关性风险

        Args:
            positions: 持仓列表 [{"symbol": "BTC/USDT", "weight": 0.3}, ...]
                       weight 为占总资金的比例
            price_data: 可选，价格数据用于计算实时相关性
            correlation_matrix: 可选，预计算的相关性矩阵

        Returns:
            CorrelationRiskReport
        """
        if not positions:
            return CorrelationRiskReport(
                total_exposure=0, max_allowed_exposure=self.max_group_exposure,
                risk_level="safe",
            )

        warnings = []
        suggestions = []
        pairs_list = []
        exposures_list = []

        held_symbols = [p["symbol"] for p in positions]

        # ── 1. 构建相关性矩阵（如果有价格数据） ──
        if correlation_matrix is None and price_data:
            held_price_data = {
                s: p for s, p in price_data.items() if s in held_symbols
            }
            if len(held_price_data) >= 2:
                correlation_matrix = self.build_correlation_matrix(held_price_data)

        # ── 2. 分析币种对相关性 ──
        if correlation_matrix:
            for i, p1 in enumerate(positions):
                for j, p2 in enumerate(positions):
                    if i >= j:
                        continue

                    sym_a, sym_b = p1["symbol"], p2["symbol"]
                    corrs = correlation_matrix.get(sym_a, {}).get(sym_b, {})

                    if not corrs:
                        continue

                    corr_30 = corrs.get(30, corrs.get(60, 0))
                    corr_60 = corrs.get(60, corrs.get(30, 0))
                    corr_90 = corrs.get(90, corr_60)
                    avg_c = np.mean([c for c in [corr_30, corr_60, corr_90] if c != 0])

                    risk = "low"
                    if avg_c >= self.high_threshold:
                        risk = "high"
                        combined_weight = p1.get("weight", 0) + p2.get("weight", 0)
                        warnings.append(
                            f"⚠️ {sym_a} ↔ {sym_b} 高度相关 (r={avg_c:.2f})，"
                            f"合计仓位 {combined_weight:.0%} 视为单一风险"
                        )
                    elif avg_c >= self.medium_threshold:
                        risk = "medium"

                    pairs_list.append(CorrelationPair(
                        symbol_a=sym_a, symbol_b=sym_b,
                        correlation_30d=round(corr_30, 3),
                        correlation_60d=round(corr_60, 3),
                        correlation_90d=round(corr_90, 3),
                        avg_correlation=round(avg_c, 3),
                        risk_level=risk,
                    ))

        # ── 3. 识别风险组暴露 ──
        position_map = {p["symbol"]: p.get("weight", 0) for p in positions}

        for group_name, group_symbols in self.DEFAULT_GROUPS.items():
            held_in_group = [s for s in group_symbols if s in position_map]
            if len(held_in_group) < 2:
                continue

            total_weight = sum(position_map[s] for s in held_in_group)
            over = total_weight > self.max_group_exposure

            if over:
                warnings.append(
                    f"⚠️ {group_name}风险组过度暴露: "
                    f"{', '.join(held_in_group)} 合计 {total_weight:.0%} "
                    f"> 上限 {self.max_group_exposure:.0%}"
                )
                suggestions.append(
                    f"建议减少 {group_name} 组仓位至 {self.max_group_exposure:.0%} 以下，"
                    f"当前合计 {total_weight:.0%}"
                )

            exposures_list.append(EffectiveExposure(
                group_name=group_name,
                symbols=held_in_group,
                total_weight=round(total_weight, 4),
                max_allowed_weight=self.max_group_exposure,
                over_exposed=over,
            ))

        # ── 4. 计算总有效暴露 ──
        total_weight = sum(p.get("weight", 0) for p in positions)
        total_effective = total_weight  # 简化版，实际应考虑相关性调整

        # ── 5. 综合评估 ──
        high_risk_pairs = sum(1 for p in pairs_list if p.risk_level == "high")
        over_groups = sum(1 for e in exposures_list if e.over_exposed)

        if high_risk_pairs >= 2 or over_groups >= 2 or total_effective > self.MAX_TOTAL_EXPOSURE:
            risk_level = "danger"
        elif high_risk_pairs >= 1 or over_groups >= 1:
            risk_level = "caution"
        else:
            risk_level = "safe"

        # 默认建议
        if not suggestions and risk_level != "safe":
            suggestions.append("考虑增加低相关性资产以分散风险")

        report = CorrelationRiskReport(
            total_exposure=round(total_effective, 4),
            max_allowed_exposure=self.MAX_TOTAL_EXPOSURE,
            risk_level=risk_level,
            pairs=pairs_list,
            exposures=exposures_list,
            warnings=warnings,
            suggestions=suggestions,
            matrix=correlation_matrix or {},
        )

        if risk_level == "danger":
            logger.warning(f"🔗 相关性风险: DANGER | {len(warnings)} 个警告")
        elif risk_level == "caution":
            logger.info(f"🔗 相关性风险: CAUTION | {len(warnings)} 个警告")
        else:
            logger.info(f"🔗 相关性风险: SAFE")

        return report

    def suggest_diversification(
        self,
        current_positions: list[dict],
        all_symbols: list[str],
        price_data: dict[str, list[float]],
    ) -> list[str]:
        """
        建议分散化的币种

        Args:
            current_positions: 当前持仓
            all_symbols: 所有可选币种
            price_data: 价格数据

        Returns:
            建议添加的低相关性币种列表
        """
        held = [p["symbol"] for p in current_positions]
        candidates = [s for s in all_symbols if s not in held]

        if not candidates or not held:
            return []

        suggestions = []
        for candidate in candidates:
            if candidate not in price_data:
                continue

            # 计算与所有持仓的平均相关性
            corrs = []
            for held_sym in held:
                if held_sym in price_data:
                    c = self.calculate_correlation(
                        price_data[candidate], price_data[held_sym]
                    )
                    corrs.append(abs(c))

            if corrs:
                avg_abs_corr = np.mean(corrs)
                if avg_abs_corr < 0.3:
                    suggestions.append(
                        f"{candidate} (平均相关性={avg_abs_corr:.2f})"
                    )

        suggestions.sort(key=lambda x: float(x.split("=")[1].rstrip(")")))
        return suggestions
