"""Shared bearish news and event-risk helpers."""
from __future__ import annotations

import re


EVENT_RISK_KEYWORDS = (
    "hack",
    "exploit",
    "lawsuit",
    "liquidation",
    "bankruptcy",
    "regulation",
    "sanction",
    "suspended",
    "delist",
    "stolen",
    "黑客",
    "监管",
    "冻结",
    "下架",
    "暴雷",
    "清算",
    "被盗",
)

MARKET_WIDE_CONTEXT_KEYWORDS = (
    "crypto market",
    "market-wide",
    "entire market",
    "all crypto",
    "macro",
    "federal reserve",
    "fed",
    "cpi",
    "sec",
    "stablecoin",
    "usdt",
    "usdc",
    "bitcoin etf",
    "spot etf",
    "全市场",
    "市场整体",
    "宏观",
    "美联储",
    "稳定币",
    "全行业",
)

NON_BEARISH_EVENT_PATTERNS = (
    "liquidation targets bears",
    "liquidation targets shorts",
    "targets bears",
    "targets shorts",
    "bears liquidated",
    "shorts liquidated",
    "bear liquidation",
    "short liquidation",
    "short squeeze",
    "空头被清算",
    "空头清算",
)

ASSET_ALIAS_MAP = {
    "BTC": {"BITCOIN", "比特币"},
    "ETH": {"ETHEREUM", "以太坊"},
    "SOL": {"SOLANA"},
    "DOGE": {"DOGECOIN"},
    "LINK": {"CHAINLINK"},
    "FIL": {"FILECOIN"},
    "AAVE": {"AAVE"},
    "ARB": {"ARBITRUM"},
    "OP": {"OPTIMISM"},
    "WLD": {"WORLDCOIN"},
    "SUI": {"SUI"},
    "AVAX": {"AVALANCHE"},
    "UNI": {"UNISWAP"},
    "NEAR": {"NEAR"},
    "INJ": {"INJECTIVE"},
    "ATOM": {"COSMOS"},
    "TIA": {"CELESTIA"},
    "ONDO": {"ONDO"},
    "RENDER": {"RNDR"},
    "POL": {"POLYGON", "MATIC"},
}


def asset_aliases(symbol: str) -> set[str]:
    base = symbol.split("/", 1)[0].upper()
    aliases = {base, symbol.upper()}
    aliases.update(ASSET_ALIAS_MAP.get(base, set()))
    return {alias.upper() for alias in aliases}


def text_mentions_alias(text: str, alias: str) -> bool:
    alias = alias.lower()
    if not alias:
        return False
    if any("\u4e00" <= char <= "\u9fff" for char in alias):
        return alias in text
    return re.search(
        rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
        text,
    ) is not None


def contains_event_risk_keywords(summary: str | None) -> bool:
    text = (summary or "").lower()
    return any(keyword in text for keyword in EVENT_RISK_KEYWORDS)


def split_news_fragments(summary: str | None) -> list[str]:
    text = str(summary or "").strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split("|") if part.strip()]
    return parts or [text]


def is_non_bearish_event_context(summary: str | None) -> bool:
    text = (summary or "").lower()
    if "liquidation" in text and any(
        pattern in text for pattern in NON_BEARISH_EVENT_PATTERNS
    ):
        return True
    return False


def contains_bearish_news_risk(symbol: str, summary: str | None) -> bool:
    aliases = {alias.lower() for alias in asset_aliases(symbol)}
    for fragment in split_news_fragments(summary):
        text = fragment.lower()
        if not contains_event_risk_keywords(fragment):
            continue
        if is_non_bearish_event_context(fragment):
            continue
        if any(text_mentions_alias(text, alias) for alias in aliases):
            return True

        if any(keyword in text for keyword in MARKET_WIDE_CONTEXT_KEYWORDS):
            return True

    return False
