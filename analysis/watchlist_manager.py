"""Watchlist Manager — 动态交易池管理

核心理念:
- "不要同时追踪太多品种" — Mark Douglas
- "分散化的关键是选不相关的资产" — Ray Dalio
- "先做好 BTC/ETH，再扩展到其他" — CZ

设计逻辑:
1. 候选池 (Candidate Pool): 30-50 个经过筛选的优质币
2. 活跃交易池 (Active Pool): 从候选中选出 Top N（默认10个）进行交易
3. 核心持仓 (Core): BTC + ETH 始终在活跃池中，占比最大
4. 卫星币 (Satellite): 其余 8 个按赛道轮换，每周重新评分

选币评分维度:
- 市值排名 (30%) — 越大越安全
- 流动性 (25%) — 日成交额越高越好
- 赛道分散度 (20%) — 与现有持仓相关性越低越好
- 动量表现 (15%) — 近 30 日 vs BTC 的相对表现
- 新闻/热度 (10%) — 社区热度、趋势排名

赛道分类:
- L1 公链: SOL, AVAX, NEAR, SUI, APT, ALGO, etc.
- L2 扩容: MATIC(POL), ARB, OP, STRK, etc.
- DeFi: UNI, AAVE, LINK, MKR, CRV, etc.
- AI 代理: FET, RNDR, ARKM, etc.
- Meme: DOGE, SHIB, PEPE, WIF, etc.
- RWA: ONDO, LINK, etc.
- 存储: FIL, AR, etc.
- GameFi: AXS, GALA, IMX, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from loguru import logger

import json
from typing import Any


# ── 赛道定义 ──

SECTORS = {
    "L1_公链": {
        "description": "底层公链，智能合约平台",
        "max_allocation_pct": 0.25,  # 整个赛道最大仓位
        "coins": ["SOL/USDT", "AVAX/USDT", "NEAR/USDT", "SUI/USDT", "APT/USDT", "ALGO/USDT", "FTM/USDT"],
    },
    "L2_扩容": {
        "description": "以太坊二层网络",
        "max_allocation_pct": 0.15,
        "coins": ["POL/USDT", "ARB/USDT", "OP/USDT", "MANTA/USDT"],
    },
    "DeFi": {
        "description": "去中心化金融协议",
        "max_allocation_pct": 0.15,
        "coins": ["UNI/USDT", "AAVE/USDT", "LINK/USDT", "MKR/USDT", "CRV/USDT", "PENDLE/USDT"],
    },
    "AI_人工智能": {
        "description": "AI 相关加密项目",
        "max_allocation_pct": 0.15,
        "coins": ["WLD/USDT", "RENDER/USDT", "ARKM/USDT", "TAO/USDT"],
        # 注意: FET/USDT 已更名为 ASI/USDT，部分交易所不再支持旧交易对
    },
    "Meme": {
        "description": "迷因币，高风险高波动",
        "max_allocation_pct": 0.10,  # Meme 赛道限制最小仓位
        "coins": ["DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "WIF/USDT", "BONK/USDT"],
    },
    "RWA": {
        "description": "真实世界资产代币化",
        "max_allocation_pct": 0.10,
        "coins": ["ONDO/USDT", "FDUSD/USDT", "PYUSD/USDT"],
    },
    "存储_Web3": {
        "description": "去中心化存储和网络",
        "max_allocation_pct": 0.10,
        "coins": ["FIL/USDT", "AR/USDT"],
    },
    "GameFi": {
        "description": "区块链游戏",
        "max_allocation_pct": 0.10,
        "coins": ["AXS/USDT", "GALA/USDT", "IMX/USDT"],
    },
    "基础设施": {
        "description": "跨链、预言机、数据等",
        "max_allocation_pct": 0.10,
        "coins": ["DOT/USDT", "ATOM/USDT", "INJ/USDT", "TIA/USDT"],
    },
}

# 核心币（始终在活跃池中）
CORE_COINS = ["BTC/USDT", "ETH/USDT"]
CORE_WEIGHT = 0.55  # 核心币占总交易池的 55%

# 默认最大活跃币种数
DEFAULT_MAX_ACTIVE = 10


@dataclass
class CoinProfile:
    """单个币种档案"""
    symbol: str
    name: str = ""
    sector: str = ""
    market_cap: float = 0           # 市值 (USDT)
    volume_24h: float = 0           # 24h 成交量
    price: float = 0
    change_24h: float = 0           # 24h 涨跌幅
    change_7d: float = 0            # 7d 涨跌幅
    change_30d_vs_btc: float = 0    # 30 日 vs BTC 相对表现
    btc_correlation: float = 0.5    # 与 BTC 相关性 (估计值)
    momentum_score: float = 0.5     # 动量评分
    sentiment_score: float = 0.5    # 情绪/热度评分
    composite_score: float = 0.0    # 综合评分
    in_active_pool: bool = False
    weight: float = 0.0             # 建议权重
    rank: int = 0                   # 排名
    tier: str = "satellite"         # "core" / "satellite"


@dataclass
class WatchlistSnapshot:
    """交易池快照"""
    active_pool: list[CoinProfile] = field(default_factory=list)
    removed_coins: list[str] = field(default_factory=list)
    added_coins: list[str] = field(default_factory=list)
    rotation_reason: str = ""
    sector_distribution: dict[str, int] = field(default_factory=dict)
    total_weight: float = 0.0
    timestamp: str = ""


class WatchlistManager:
    """
    动态交易池管理器

    用法:
        manager = WatchlistManager()
        # 初始化候选池
        manager.refresh_candidate_pool()
        # 评分并选出 Top 10
        snapshot = manager.select_active_pool()
        # 获取活跃交易列表
        for coin in snapshot.active_pool:
            logger.info(f"  {coin.symbol} (评分={coin.composite_score:.2f})")
    """

    # 评分权重
    WEIGHT_MARKET_CAP = 0.25
    WEIGHT_LIQUIDITY = 0.25
    WEIGHT_DIVERSIFICATION = 0.20
    WEIGHT_MOMENTUM = 0.15
    WEIGHT_SENTIMENT = 0.15

    # 流动性门槛
    MIN_VOLUME_24H = 5_000_000      # 最低日成交额 500 万 USDT
    GOOD_VOLUME_24H = 50_000_000    # 良好日成交额 5000 万 USDT

    def __init__(
        self,
        max_active: int = DEFAULT_MAX_ACTIVE,
        core_weight: float = CORE_WEIGHT,
        rotation_interval_days: int = 7,
        storage=None,
    ):
        self.max_active = max_active
        self.core_weight = core_weight
        self.rotation_interval_days = rotation_interval_days
        self.storage = storage

        self._candidate_pool: dict[str, CoinProfile] = {}
        self._active_pool: list[CoinProfile] = []
        self._last_rotation: datetime | None = None
        self._trending_symbols: list[str] = []

    def refresh_candidate_pool(self, market_data: dict[str, dict] | None = None):
        """
        刷新候选池

        Args:
            market_data: 可选，从交易所获取的市场数据
                {symbol: {"price": float, "volume24h": float, "change24h": float, ...}}
        """
        logger.info("🔄 刷新候选池...")

        # 从赛道定义中构建候选池
        for sector_name, sector_info in SECTORS.items():
            for symbol in sector_info["coins"]:
                if symbol not in self._candidate_pool:
                    self._candidate_pool[symbol] = CoinProfile(
                        symbol=symbol,
                        sector=sector_name,
                        tier="satellite",
                    )

        # 添加核心币
        for symbol in CORE_COINS:
            if symbol not in self._candidate_pool:
                self._candidate_pool[symbol] = CoinProfile(
                    symbol=symbol,
                    name="Bitcoin" if "BTC" in symbol else "Ethereum",
                    sector="核心",
                    tier="core",
                    btc_correlation=1.0 if "BTC" in symbol else 0.85,
                )

        # 用市场数据更新档案
        if market_data:
            for symbol, data in market_data.items():
                if symbol in self._candidate_pool:
                    profile = self._candidate_pool[symbol]
                    profile.price = data.get("price", profile.price)
                    profile.volume_24h = data.get("volume24h", profile.volume_24h)
                    profile.market_cap = data.get("marketCap", profile.market_cap)
                    profile.change_24h = data.get("change24h", profile.change_24h)
                    profile.change_7d = data.get("change7d", profile.change_7d)

        # 添加热门/趋势币
        for symbol in self._trending_symbols:
            if symbol not in self._candidate_pool:
                self._candidate_pool[symbol] = CoinProfile(
                    symbol=symbol,
                    sector="热点追踪",
                    tier="satellite",
                    sentiment_score=0.8,  # 趋势币默认高热度
                )

        logger.info(f"候选池: {len(self._candidate_pool)} 个币种")

    def select_active_pool(
        self,
        current_holdings: list[dict] | None = None,
        price_data: dict[str, list[float]] | None = None,
        fear_greed: float | None = None,
    ) -> WatchlistSnapshot:
        """
        评分并选出活跃交易池

        Args:
            current_holdings: 当前持仓 [{"symbol": "BTC/USDT", "weight": 0.3}]
            price_data: 用于计算相关性的价格数据
            fear_greed: 恐惧贪婪指数

        Returns:
            WatchlistSnapshot
        """
        logger.info(f"🎯 评选活跃交易池 (Top {self.max_active})...")

        # ── 1. 过滤流动性不达标的 ──
        qualified = []
        filtered_out = []
        has_real_data = any(p.volume_24h > 0 for p in self._candidate_pool.values())

        for symbol, profile in self._candidate_pool.items():
            if has_real_data:
                # 有真实数据时，严格过滤
                if profile.volume_24h >= self.MIN_VOLUME_24H or profile.tier == "core":
                    qualified.append(profile)
                else:
                    filtered_out.append(symbol)
            else:
                # 无真实数据时（交易所未连接），全部候选参与评分
                # 给非核心币一个合理的默认成交量，避免被过滤
                if profile.tier == "core":
                    profile.volume_24h = 1_000_000_000
                else:
                    profile.volume_24h = 50_000_000  # 假设达标
                qualified.append(profile)

        if filtered_out:
            logger.info(f"  过滤掉 {len(filtered_out)} 个流动性不足的币种")

        # ── 2. 计算每个币的评分 ──
        held_sectors = set()
        if current_holdings:
            for h in current_holdings:
                sym = h.get("symbol", "")
                if sym in self._candidate_pool:
                    held_sectors.add(self._candidate_pool[sym].sector)

        for profile in qualified:
            score = self._calculate_composite_score(
                profile=profile,
                held_sectors=held_sectors,
                fear_greed=fear_greed,
            )
            profile.composite_score = score

        # ── 3. 排序 ──
        # 核心币始终排在最前
        qualified.sort(key=lambda p: (
            0 if p.tier == "core" else 1,
            -p.composite_score,
        ))

        # ── 4. 选取，考虑赛道约束 ──
        selected = []
        sector_count: dict[str, int] = {}
        max_per_sector = 2  # 每个赛道最多 2 个

        for profile in qualified:
            # 核心币直接入选
            if profile.tier == "core":
                selected.append(profile)
                sector_count[profile.sector] = sector_count.get(profile.sector, 0) + 1
                continue

            # 检查数量限制
            if len(selected) >= self.max_active:
                break

            # 检查赛道限制
            sc = sector_count.get(profile.sector, 0)
            if sc >= max_per_sector and profile.sector != "核心":
                continue

            # 检查赛道总仓位限制
            sector_alloc = sum(
                p.weight for p in selected if p.sector == profile.sector
            )
            sector_max = SECTORS.get(profile.sector, {}).get(
                "max_allocation_pct", 0.10
            )
            if sector_alloc >= sector_max:
                continue

            selected.append(profile)
            sector_count[profile.sector] = sc + 1

        # ── 5. 分配权重 ──
        self._assign_weights(selected)

        # ── 6. 对比上次，记录变动 ──
        prev_symbols = set(p.symbol for p in self._active_pool)
        new_symbols = set(p.symbol for p in selected)
        added = new_symbols - prev_symbols
        removed = prev_symbols - new_symbols

        # ── 7. 记录排名 ──
        for i, p in enumerate(selected):
            p.rank = i + 1
            p.in_active_pool = True

        # 构建快照
        sector_dist = {}
        for p in selected:
            sector_dist[p.sector] = sector_dist.get(p.sector, 0) + 1

        snapshot = WatchlistSnapshot(
            active_pool=selected,
            removed_coins=list(removed),
            added_coins=list(added),
            rotation_reason=self._generate_rotation_reason(added, removed),
            sector_distribution=sector_dist,
            total_weight=sum(p.weight for p in selected),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._active_pool = selected
        self._last_rotation = datetime.now(timezone.utc)

        # ── 日志 ──
        logger.info(f"  活跃交易池 ({len(selected)} 个):")
        for p in selected:
            tier_mark = "⭐" if p.tier == "core" else "🛰️"
            logger.info(
                f"    {p.rank}. {tier_mark} {p.symbol} "
                f"(评分={p.composite_score:.2f}, 权重={p.weight:.0%}, "
                f"赛道={p.sector})"
            )

        if added:
            logger.info(f"  ➕ 新增: {', '.join(added)}")
        if removed:
            logger.info(f"  ➖ 移除: {', '.join(removed)}")

        return snapshot

    def _calculate_composite_score(
        self,
        profile: CoinProfile,
        held_sectors: set[str],
        fear_greed: float | None = None,
    ) -> float:
        """计算综合评分"""
        scores = []

        # 1. 市值评分 (0-1)
        mc_score = min(1.0, profile.market_cap / 100_000_000_000)  # 1000亿=满分
        scores.append(("市值", mc_score, self.WEIGHT_MARKET_CAP))

        # 2. 流动性评分 (0-1)
        liq_score = min(1.0, profile.volume_24h / self.GOOD_VOLUME_24H)
        scores.append(("流动性", liq_score, self.WEIGHT_LIQUIDITY))

        # 3. 分散化评分 (0-1) — 已有该赛道的币则降分
        if profile.sector in held_sectors:
            div_score = 0.3  # 赛道已有持仓，分散度低
        else:
            div_score = 0.8  # 新赛道，分散度高

        # BTC 相关性低加分
        if profile.btc_correlation < 0.5:
            div_score *= 1.2

        div_score = min(1.0, div_score)
        scores.append(("分散化", div_score, self.WEIGHT_DIVERSIFICATION))

        # 4. 动量评分 (0-1)
        # 30 日 vs BTC 相对表现
        momentum = profile.momentum_score
        if profile.change_30d_vs_btc != 0:
            momentum = 0.5 + (profile.change_30d_vs_btc / 100)  # 归一化到 0-1
        momentum = max(0, min(1, momentum))

        # FGI 调整：高贪婪时降低动量权重，恐慌时提高
        if fear_greed is not None:
            if fear_greed > 80:
                momentum *= 0.7  # 贪婪时不要太追
            elif fear_greed < 20:
                momentum *= 1.3  # 恐慌时强者恒强

        scores.append(("动量", momentum, self.WEIGHT_MOMENTUM))

        # 5. 情绪/热度评分 (0-1)
        sent_score = profile.sentiment_score
        scores.append(("热度", sent_score, self.WEIGHT_SENTIMENT))

        # 加权求和
        total = sum(s[1] * s[2] for s in scores)

        # Meme 赛道惩罚（高风险）
        if profile.sector == "Meme":
            total *= 0.85

        # 核心币加分（BTC/ETH 永远是最安全的）
        if profile.tier == "core":
            total = max(total, 0.9)

        return round(total, 4)

    def _assign_weights(self, pool: list[CoinProfile]):
        """分配权重"""
        # 核心币权重
        core_count = sum(1 for p in pool if p.tier == "core")
        satellite_count = len(pool) - core_count

        if core_count == 0:
            core_count = 1  # 保底

        # BTC : ETH = 60% : 40% 核心比例
        core_weights = [0.60, 0.40]
        for i, p in enumerate(pool):
            if p.tier == "core":
                idx = min(i, len(core_weights) - 1)
                raw_weight = core_weights[idx] * self.core_weight
                p.weight = round(raw_weight, 4)

        # 卫星币按评分分配剩余权重
        satellite_remaining = 1.0 - self.core_weight
        satellites = [p for p in pool if p.tier != "core"]

        if satellites:
            total_score = sum(p.composite_score for p in satellites)
            if total_score > 0:
                for p in satellites:
                    score_ratio = p.composite_score / total_score
                    p.weight = round(score_ratio * satellite_remaining, 4)

    def add_trending_coin(self, symbol: str, sector: str = "热点追踪"):
        """添加趋势/热点币到候选池"""
        if symbol not in self._candidate_pool:
            self._candidate_pool[symbol] = CoinProfile(
                symbol=symbol,
                sector=sector,
                sentiment_score=0.85,
            )
            self._trending_symbols.append(symbol)
            logger.info(f"🔥 趋势币添加: {symbol} ({sector})")

    def remove_from_pool(self, symbol: str):
        """从候选池移除币种"""
        if symbol in CORE_COINS:
            logger.warning(f"不能移除核心币: {symbol}")
            return
        self._candidate_pool.pop(symbol, None)
        self._trending_symbols = [s for s in self._trending_symbols if s != symbol]
        logger.info(f"已移除: {symbol}")

    def validate_against_exchange(
        self,
        available_symbols: set[str],
    ) -> list[str]:
        """
        校验活跃池中的交易对是否在交易所可用。

        Args:
            available_symbols: 交易所支持的交易对集合，如 {"BTC/USDT:USDT", "ETH/USDT:USDT"}

        Returns:
            被移除的不支持币种列表
        """
        removed = []
        for symbol in list(self._candidate_pool.keys()):
            # 构造可能的交易所格式: BTC/USDT → BTC/USDT:USDT 或 BTC-USDT-SWAP
            okx_format = symbol.replace("/", "") + "USDT"
            usdt_format = symbol + ":USDT"
            if not any(
                fmt in available_symbols
                for fmt in [symbol, usdt_format, okx_format]
            ):
                if symbol not in CORE_COINS:  # 核心币不移除
                    self._candidate_pool.pop(symbol, None)
                    removed.append(symbol)
                    logger.warning(
                        f"⚠️ {symbol} 不在交易所支持列表中，已从候选池移除"
                    )

        # 同步清理活跃池
        unsupported_active = [
            p for p in self._active_pool
            if p.symbol in removed
        ]
        if unsupported_active:
            self._active_pool = [
                p for p in self._active_pool if p.symbol not in removed
            ]

        if removed:
            logger.info(f"  校验移除 {len(removed)} 个不支持的币种: {removed}")
        return removed

    def get_active_symbols(self) -> list[str]:
        """获取当前活跃交易池的币种列表"""
        return [p.symbol for p in self._active_pool]

    def get_candidate_pool_summary(self) -> str:
        """获取候选池摘要"""
        lines = [f"## 📋 候选池概览 ({len(self._candidate_pool)} 个币)"]
        for sector, info in SECTORS.items():
            coins = info["coins"]
            lines.append(f"\n### {sector} ({info['description']})")
            lines.append(f"  最大仓位: {info['max_allocation_pct']:.0%}")
            lines.append(f"  候选: {', '.join(c.replace('/USDT', '') for c in coins)}")

        active = self.get_active_symbols()
        if active:
            lines.append(f"\n### 当前活跃池 ({len(active)} 个)")
            lines.append(f"  {', '.join(s.replace('/USDT', '') for s in active)}")

        return "\n".join(lines)

    def needs_rotation(self) -> bool:
        """检查是否需要轮换"""
        if not self._last_rotation:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_rotation).total_seconds() / 86400
        return elapsed >= self.rotation_interval_days

    def _generate_rotation_reason(self, added: set, removed: set) -> str:
        """生成轮换理由"""
        if not added and not removed:
            return "无变动"
        parts = []
        if added:
            parts.append(f"新增 {len(added)} 个")
        if removed:
            parts.append(f"移除 {len(removed)} 个")
        return ", ".join(parts)

    def get_rotation_report(self) -> str:
        """生成轮换报告（人可读）"""
        lines = [
            "## 🔄 交易池轮换报告",
            f"**更新时间**: {self._last_rotation.strftime('%Y-%m-%d %H:%M UTC') if self._last_rotation else 'N/A'}",
            f"**候选池**: {len(self._candidate_pool)} 个",
            f"**活跃池**: {len(self._active_pool)} 个 (上限 {self.max_active})",
            "",
        ]

        if self._active_pool:
            lines.append("### 📊 活跃交易池排名")
            lines.append("| 排名 | 币种 | 赛道 | 评分 | 权重 | 类型 |")
            lines.append("|------|------|------|------|------|------|")
            for p in self._active_pool:
                tier = "⭐核心" if p.tier == "core" else "🛰️卫星"
                lines.append(
                    f"| {p.rank} | {p.symbol} | {p.sector} | "
                    f"{p.composite_score:.2f} | {p.weight:.0%} | {tier} |"
                )

            lines.append("")
            lines.append("### 🏷️ 赛道分布")
            for sector, count in sorted(
                {k: v for k, v in
                 {p.sector: 0 for p in self._active_pool}.items()
                }.items()
            ):
                sector_coins = [p.symbol.replace('/USDT', '') for p in self._active_pool if p.sector == sector]
                lines.append(f"- **{sector}**: {', '.join(sector_coins)}")

        return "\n".join(lines)
