"""Sentiment data collector for Fear & Greed and LunarCrush."""
from __future__ import annotations

from loguru import logger
import requests

from config import SentimentSettings
from core.storage import Storage


class SentimentCollector:
    """Collect and merge sentiment sources for the engine."""

    def __init__(
        self,
        storage: Storage,
        settings: SentimentSettings | None = None,
        session: requests.Session | None = None,
    ):
        self.storage = storage
        self.settings = settings or SentimentSettings()
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "CryptoAI-v2",
            }
        )

    def fetch_fear_greed_index(self) -> dict | None:
        try:
            resp = self.session.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()["data"][0]
            result = {
                "source": "fear_greed",
                "value": int(data["value"]),
                "label": data["value_classification"],
                "summary": f"Fear & Greed: {data['value']} ({data['value_classification']})",
            }
            self.storage.insert_sentiment(result)
            self.storage.set_state("latest_fear_greed", str(result["value"]))
            self.storage.set_state("latest_sentiment", result["summary"])
            logger.info(f"Fear & Greed Index: {data['value']} ({data['value_classification']})")
            return result
        except Exception as exc:
            logger.error(f"Failed to fetch Fear & Greed Index: {exc}")
            return None

    def fetch_lunarcrush_sentiment(self, symbol: str) -> dict | None:
        api_key = self.settings.lunarcrush_api_key.get_secret_value()
        if not api_key:
            return None

        asset = symbol.split("/", 1)[0].upper()
        try:
            response = self.session.get(
                f"{self.settings.lunarcrush_api_base}/public/topic/{asset}/v1",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") or payload
            if isinstance(data, list):
                data = data[0] if data else {}
            sentiment_score = float(
                data.get("sentiment")
                or data.get("sentiment_score")
                or data.get("types_sentiment")
                or 0.0
            )
            social_volume = float(
                data.get("interactions_24h")
                or data.get("social_volume_24h")
                or data.get("contributors_active")
                or 0.0
            )
            normalized = max(-1.0, min(1.0, sentiment_score))
            result = {
                "source": "lunarcrush",
                "value": normalized,
                "label": "social_sentiment",
                "summary": (
                    f"LunarCrush {asset}: sentiment={normalized:.2f}, "
                    f"social_volume={social_volume:.0f}"
                ),
                "social_volume": social_volume,
            }
            self.storage.insert_sentiment(result)
            self.storage.set_state("latest_lunarcrush_sentiment", f"{normalized:.4f}")
            self.storage.set_state("latest_lunarcrush_summary", result["summary"])
            return result
        except Exception as exc:
            logger.error(f"Failed to fetch LunarCrush sentiment for {asset}: {exc}")
            return None

    def fetch_crypto_news(self, limit: int = 10) -> list[dict]:
        logger.debug("Crypto news fetch not yet implemented, returning empty")
        return []

    def get_latest_sentiment(self, symbol: str = "BTC/USDT") -> dict | None:
        fear_greed = self.fetch_fear_greed_index()
        lunarcrush = self.fetch_lunarcrush_sentiment(symbol)
        if not fear_greed and not lunarcrush:
            return None

        merged = dict(fear_greed or {})
        if lunarcrush:
            merged["lunarcrush_sentiment"] = lunarcrush["value"]
            merged["lunarcrush_summary"] = lunarcrush["summary"]
            merged["social_volume"] = lunarcrush.get("social_volume", 0.0)
            base_summary = merged.get("summary", "")
            merged["summary"] = (
                f"{base_summary} | {lunarcrush['summary']}"
                if base_summary
                else lunarcrush["summary"]
            )
        return merged
