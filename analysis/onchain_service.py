"""On-chain summary service for CryptoAI v3."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import requests
from requests import HTTPError

from config import OnchainSettings


@dataclass
class OnchainSummary:
    symbol: str
    summary: str
    netflow_score: float = 0.0
    whale_score: float = 0.0
    source: str = "fallback"
    timestamp: str = ""


class OnchainService:
    """Provide a structured on-chain context layer with graceful fallback."""

    COMMUNITY_COINMETRICS_API_BASE = "https://community-api.coinmetrics.io/v4"
    COMMUNITY_METRIC_CANDIDATES = [
        ("TxCnt,AdrActCnt", "TxCnt", "AdrActCnt"),
        ("TxCnt", "TxCnt", ""),
    ]

    def __init__(
        self,
        settings: OnchainSettings | None = None,
        glassnode_fetcher: Callable[[str], dict] | None = None,
        coinmetrics_fetcher: Callable[[str], dict] | None = None,
        session: requests.Session | None = None,
    ):
        self.settings = settings or OnchainSettings()
        self.glassnode_fetcher = glassnode_fetcher
        self.coinmetrics_fetcher = coinmetrics_fetcher
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "CryptoAI-v2",
            }
        )

    def get_summary(self, symbol: str) -> OnchainSummary:
        if self.glassnode_fetcher is not None:
            payload = self._safe_fetch(self.glassnode_fetcher, symbol)
            if payload:
                return self._build_summary(symbol, payload, "glassnode")

        glassnode_key = self.settings.glassnode_api_key.get_secret_value()
        if glassnode_key:
            payload = self._fetch_glassnode(symbol)
            if payload:
                return self._build_summary(symbol, payload, "glassnode")

        if self.coinmetrics_fetcher is not None:
            payload = self._safe_fetch(self.coinmetrics_fetcher, symbol)
            if payload:
                return self._build_summary(symbol, payload, "coinmetrics")

        payload = self._fetch_coinmetrics(symbol)
        if payload:
            return self._build_summary(symbol, payload, "coinmetrics")

        return OnchainSummary(
            symbol=symbol,
            summary="On-chain data unavailable, using neutral on-chain context.",
            netflow_score=0.0,
            whale_score=0.0,
            source="fallback",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _safe_fetch(fetcher: Callable[[str], dict], symbol: str) -> dict | None:
        try:
            return fetcher(symbol) or None
        except Exception:
            return None

    def _fetch_glassnode(self, symbol: str) -> dict | None:
        asset = symbol.split("/", 1)[0].lower()
        params = {
            "a": asset,
            "i": self.settings.glassnode_interval,
            "api_key": self.settings.glassnode_api_key.get_secret_value(),
            "e": self.settings.glassnode_exchange,
        }
        absolute_series = self._glassnode_series(
            self.settings.glassnode_exchange_balance_path,
            params,
        )
        relative_series = self._glassnode_series(
            self.settings.glassnode_exchange_balance_relative_path,
            params,
        )
        if not absolute_series and not relative_series:
            return None

        latest_absolute = self._last_value(absolute_series)
        latest_relative = self._last_value(relative_series)
        previous_relative = self._prev_value(relative_series)
        relative_change = (
            latest_relative - previous_relative if previous_relative is not None else 0.0
        )
        return {
            "summary": (
                f"Glassnode exchange balance={latest_absolute:.4f}, "
                f"relative_balance={latest_relative:.4f}, "
                f"relative_change={relative_change:+.4f}"
            ),
            "netflow_score": relative_change,
            "whale_score": latest_relative,
        }

    def _glassnode_series(self, path: str, params: dict) -> list[dict]:
        response = self.session.get(
            f"{self.settings.glassnode_api_base}{path}",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []

    def _fetch_coinmetrics(self, symbol: str) -> dict | None:
        asset = symbol.split("/", 1)[0].lower()
        api_key = self.settings.coinmetrics_api_key.get_secret_value()
        for metrics_param, primary_metric, secondary_metric in self.COMMUNITY_METRIC_CANDIDATES:
            try:
                asset_metric = self._coinmetrics_series(
                    "/timeseries/asset-metrics",
                    {
                        "assets": asset,
                        "metrics": metrics_param,
                        "page_size": 2,
                        **({"api_key": api_key} if api_key else {}),
                    },
                    api_key=api_key,
                )
            except Exception:
                continue
            if not asset_metric:
                continue

            latest_primary = self._last_metric_value(asset_metric, primary_metric)
            previous_primary = self._prev_metric_value(asset_metric, primary_metric)
            secondary_value = (
                self._last_metric_value(asset_metric, secondary_metric)
                if secondary_metric
                else 0.0
            )
            activity_change = (
                latest_primary - previous_primary if previous_primary is not None else 0.0
            )
            summary = (
                f"CoinMetrics {primary_metric}={latest_primary:.4f}, "
                f"activity_change={activity_change:+.4f}"
            )
            if secondary_metric:
                summary = (
                    f"CoinMetrics {primary_metric}={latest_primary:.4f}, "
                    f"{secondary_metric}={secondary_value:.4f}, "
                    f"activity_change={activity_change:+.4f}"
                )
            return {
                "summary": summary,
                "netflow_score": activity_change,
                "whale_score": secondary_value if secondary_metric else latest_primary,
            }
        return None

    def _coinmetrics_series(self, path: str, params: dict, api_key: str = "") -> list[dict]:
        base_url = (
            self.settings.coinmetrics_api_base
            if api_key
            else self.COMMUNITY_COINMETRICS_API_BASE
        )
        response = self.session.get(
            f"{base_url}{path}",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", []) if isinstance(data, dict) else []

    def _build_summary(self, symbol: str, payload: dict, source: str) -> OnchainSummary:
        return OnchainSummary(
            symbol=symbol,
            summary=payload.get("summary", f"{source} summary available"),
            netflow_score=float(payload.get("netflow_score", 0.0)),
            whale_score=float(payload.get("whale_score", 0.0)),
            source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _last_value(series: list[dict]) -> float:
        if not series:
            return 0.0
        return float(series[-1].get("v") or 0.0)

    @staticmethod
    def _prev_value(series: list[dict]) -> float | None:
        if len(series) < 2:
            return None
        return float(series[-2].get("v") or 0.0)

    @staticmethod
    def _last_metric_value(series: list[dict], key: str) -> float:
        if not series:
            return 0.0
        return float(series[-1].get(key) or 0.0)

    @staticmethod
    def _prev_metric_value(series: list[dict], key: str) -> float | None:
        if len(series) < 2:
            return None
        return float(series[-2].get(key) or 0.0)
