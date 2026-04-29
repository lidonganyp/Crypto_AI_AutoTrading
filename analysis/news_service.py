"""News service for CryptoAI v3."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from config import NewsSettings


@dataclass
class NewsSummary:
    symbol: str
    headline_count: int
    summary: str
    trending_symbols: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    coverage_score: float = 0.0
    service_health_score: float = 0.0
    source_status: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""


class NewsService:
    """Fetch crypto news context from multiple lightweight sources."""

    ASSET_ALIASES = {
        "BTC": ("BTC", "BITCOIN", "比特币"),
        "ETH": ("ETH", "ETHEREUM", "以太坊"),
        "SOL": ("SOL", "SOLANA"),
        "AVAX": ("AVAX", "AVALANCHE"),
        "ARB": ("ARB", "ARBITRUM"),
        "OP": ("OP", "OPTIMISM"),
        "UNI": ("UNI", "UNISWAP"),
        "AAVE": ("AAVE",),
        "WLD": ("WLD", "WORLDCOIN"),
        "RENDER": ("RENDER", "RNDR"),
        "LINK": ("LINK", "CHAINLINK"),
        "NEAR": ("NEAR",),
        "INJ": ("INJ", "INJECTIVE"),
        "ATOM": ("ATOM", "COSMOS"),
        "FIL": ("FIL", "FILECOIN"),
        "TIA": ("TIA", "CELESTIA"),
        "ONDO": ("ONDO",),
        "SUI": ("SUI",),
        "POL": ("POL", "POLYGON", "MATIC"),
        "DOGE": ("DOGE", "DOGECOIN"),
        "SHIB": ("SHIB", "SHIBA"),
    }

    def __init__(self, settings: NewsSettings | None = None):
        self.settings = settings or NewsSettings()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, application/rss+xml, application/xml",
                "User-Agent": "CryptoAI-v2",
            }
        )

    def get_summary(self, symbol: str) -> NewsSummary:
        lines: list[str] = []
        sources: list[str] = []
        headlines: list[str] = []
        trending_symbols: list[str] = []
        source_status: dict[str, str] = {}

        coindesk_headlines, coindesk_status = self._fetch_rss_result(
            self.settings.coindesk_rss_url,
            symbol,
        )
        source_status["CoinDesk"] = coindesk_status
        if coindesk_headlines:
            sources.append("CoinDesk")
            headlines.extend(coindesk_headlines)
            lines.append(
                "CoinDesk headlines: " + " | ".join(coindesk_headlines[:3])
            )

        cointelegraph_headlines, cointelegraph_status = self._fetch_rss_result(
            self.settings.cointelegraph_rss_url,
            symbol,
        )
        source_status["Cointelegraph"] = cointelegraph_status
        if cointelegraph_headlines:
            sources.append("Cointelegraph")
            headlines.extend(cointelegraph_headlines)
            lines.append(
                "Cointelegraph headlines: " + " | ".join(cointelegraph_headlines[:3])
            )

        jin10_headlines, jin10_status = self._fetch_jin10_result(symbol)
        source_status["Jin10"] = jin10_status
        if jin10_headlines:
            sources.append("Jin10")
            headlines.extend(jin10_headlines)
            lines.append(
                "Jin10 headlines: " + " | ".join(jin10_headlines[:3])
            )

        cryptopanic_items, cryptopanic_status = self._fetch_cryptopanic_result(symbol)
        source_status["CryptoPanic"] = cryptopanic_status
        if cryptopanic_items:
            sources.append("CryptoPanic")
            headlines.extend(cryptopanic_items)
            lines.append(
                "CryptoPanic headlines: " + " | ".join(cryptopanic_items[:3])
            )

        if headlines:
            trending_symbols.append(symbol)

        if not lines:
            lines.append("No external news feed available, using neutral news context.")

        active_sources = [
            name for name, status in source_status.items()
            if status != "disabled"
        ]
        healthy_sources = [
            name for name, status in source_status.items()
            if status in {"matched", "healthy_no_match"}
        ]
        matched_sources = [
            name for name, status in source_status.items()
            if status == "matched"
        ]
        service_health_score = (
            len(healthy_sources) / len(active_sources)
            if active_sources
            else 0.0
        )
        source_coverage = (
            len(matched_sources) / max(1, min(2, len(active_sources)))
            if active_sources
            else 0.0
        )
        headline_coverage = min(1.0, len(headlines) / 4.0)
        coverage_score = min(
            1.0,
            source_coverage * 0.7 + headline_coverage * 0.3,
        )
        summary = "\n".join(lines)
        return NewsSummary(
            symbol=symbol,
            headline_count=len(headlines),
            summary=summary,
            trending_symbols=trending_symbols,
            sources=sources,
            coverage_score=round(coverage_score, 4),
            service_health_score=round(service_health_score, 4),
            source_status=source_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _fetch_coindesk_headlines(self, symbol: str) -> list[str]:
        headlines, _ = self._fetch_rss_result(
            self.settings.coindesk_rss_url,
            symbol,
        )
        return headlines

    def _fetch_cointelegraph_headlines(self, symbol: str) -> list[str]:
        headlines, _ = self._fetch_rss_result(
            self.settings.cointelegraph_rss_url,
            symbol,
        )
        return headlines

    def _fetch_rss_headlines(self, feed_url: str, symbol: str) -> list[str]:
        headlines, _ = self._fetch_rss_result(feed_url, symbol)
        return headlines

    def _fetch_rss_result(self, feed_url: str, symbol: str) -> tuple[list[str], str]:
        asset = symbol.split("/", 1)[0].upper()
        try:
            response = self.session.get(feed_url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            headlines: list[str] = []
            for item in root.findall(".//item")[:20]:
                title = (item.findtext("title") or "").strip()
                if not title:
                    continue
                if self._title_mentions_asset(title, asset):
                    headlines.append(title)
            if headlines:
                return headlines[:5], "matched"
            return [], "healthy_no_match"
        except Exception:
            return [], "unavailable"

    @classmethod
    def _title_mentions_asset(cls, title: str, asset: str) -> bool:
        title_upper = title.upper()
        aliases = cls.ASSET_ALIASES.get(asset, (asset,))
        for alias in aliases:
            if cls._text_mentions_alias(title_upper, alias.upper()):
                return True
        return False

    @staticmethod
    def _text_mentions_alias(text: str, alias: str) -> bool:
        if any("\u4e00" <= char <= "\u9fff" for char in alias):
            return alias in text
        return re.search(rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])", text) is not None

    def _fetch_jin10_headlines(self, symbol: str) -> list[str]:
        headlines, _ = self._fetch_jin10_result(symbol)
        return headlines

    def _fetch_jin10_result(self, symbol: str) -> tuple[list[str], str]:
        asset = symbol.split("/", 1)[0].upper()
        try:
            response = self.session.get(self.settings.jin10_url, timeout=10)
            response.raise_for_status()
            html = response.text
            candidates = re.findall(r">([^<>]{8,160})<", html)
            cleaned = []
            seen = set()
            aliases = tuple(alias.upper() for alias in self.ASSET_ALIASES.get(asset, (asset,)))
            for raw in candidates:
                text = " ".join(raw.split())
                if len(text) < 8:
                    continue
                text_upper = text.upper()
                if not any(self._text_mentions_alias(text_upper, alias) for alias in aliases):
                    continue
                if text in seen:
                    continue
                seen.add(text)
                cleaned.append(text)
                if len(cleaned) >= 5:
                    break
            if cleaned:
                return cleaned, "matched"
            return [], "healthy_no_match"
        except Exception:
            return [], "unavailable"

    def _fetch_cryptopanic(self, symbol: str) -> list[str]:
        headlines, _ = self._fetch_cryptopanic_result(symbol)
        return headlines

    def _fetch_cryptopanic_result(self, symbol: str) -> tuple[list[str], str]:
        api_key = self.settings.cryptopanic_api_key.get_secret_value()
        if not api_key:
            return [], "disabled"
        asset = symbol.split("/", 1)[0].upper()
        try:
            response = self.session.get(
                self.settings.cryptopanic_api_base,
                params={
                    "auth_token": api_key,
                    "public": "true",
                    "currencies": asset,
                    "kind": "news",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            headlines: list[str] = []
            for item in data.get("results", [])[:10]:
                title = (item.get("title") or "").strip()
                if title:
                    headlines.append(title)
            if headlines:
                return headlines[:5], "matched"
            return [], "healthy_no_match"
        except Exception:
            return [], "unavailable"
