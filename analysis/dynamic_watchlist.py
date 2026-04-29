"""Dynamic watchlist management for the active v2 engine path."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from math import tanh
from typing import Any

from config import Settings
from core.storage import Storage


@dataclass
class WatchlistCandidate:
    symbol: str
    score: float
    sector: str = "other"
    quote_volume_24h: float = 0.0
    change_pct_24h: float = 0.0
    is_core: bool = False
    source: str = "market"
    notes: list[str] = field(default_factory=list)


@dataclass
class WatchlistSnapshot:
    active_symbols: list[str]
    added_symbols: list[str] = field(default_factory=list)
    removed_symbols: list[str] = field(default_factory=list)
    candidates: list[WatchlistCandidate] = field(default_factory=list)
    whitelist: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)
    refreshed_at: str = ""
    refresh_reason: str = ""


class DynamicWatchlistService:
    """Score and maintain a dynamic active symbol universe."""

    ACTIVE_SYMBOLS_STATE_KEY = "active_symbols"
    SNAPSHOT_STATE_KEY = "active_watchlist_snapshot"
    REFRESHED_AT_STATE_KEY = "active_watchlist_refreshed_at"
    WHITELIST_STATE_KEY = "watchlist_whitelist"
    BLACKLIST_STATE_KEY = "watchlist_blacklist"
    SYMBOL_SECTORS = {
        "BTC/USDT": "core",
        "ETH/USDT": "core",
        "SOL/USDT": "l1",
        "AVAX/USDT": "l1",
        "SUI/USDT": "l1",
        "APT/USDT": "l1",
        "NEAR/USDT": "l1",
        "POL/USDT": "l2",
        "ARB/USDT": "l2",
        "OP/USDT": "l2",
        "UNI/USDT": "defi",
        "AAVE/USDT": "defi",
        "LINK/USDT": "defi",
        "WLD/USDT": "ai",
        "RENDER/USDT": "ai",
        "TAO/USDT": "ai",
        "DOGE/USDT": "meme",
        "SHIB/USDT": "meme",
        "ONDO/USDT": "rwa",
        "FIL/USDT": "storage",
        "ATOM/USDT": "infrastructure",
        "TIA/USDT": "infrastructure",
        "INJ/USDT": "infrastructure",
    }

    def __init__(self, storage: Storage, settings: Settings, market):
        self.storage = storage
        self.settings = settings
        self.market = market

    def get_active_symbols(self) -> list[str]:
        cached = self.storage.get_json_state(self.ACTIVE_SYMBOLS_STATE_KEY, None)
        if isinstance(cached, list) and cached:
            return self._filter_allowed_symbols([str(symbol) for symbol in cached])
        return self._default_symbols()

    def get_manual_lists(self) -> tuple[list[str], list[str]]:
        whitelist = self.storage.get_json_state(self.WHITELIST_STATE_KEY, []) or []
        blacklist = self.storage.get_json_state(self.BLACKLIST_STATE_KEY, []) or []
        whitelist = self._filter_allowed_symbols([str(symbol) for symbol in whitelist])
        blacklist = [str(symbol) for symbol in blacklist]
        return whitelist, blacklist

    def set_manual_lists(self, whitelist: list[str], blacklist: list[str]) -> None:
        normalized_whitelist = self._filter_allowed_symbols(
            self._normalize_symbol_list(whitelist)
        )
        normalized_blacklist = self._normalize_symbol_list(blacklist)
        self.storage.set_json_state(self.WHITELIST_STATE_KEY, normalized_whitelist)
        self.storage.set_json_state(self.BLACKLIST_STATE_KEY, normalized_blacklist)

    def refresh(self, force: bool = False, now: datetime | None = None) -> WatchlistSnapshot:
        now = now or datetime.now(timezone.utc)
        if not self.settings.exchange.dynamic_watchlist_enabled:
            return self._persist_snapshot(
                WatchlistSnapshot(
                    active_symbols=self._default_symbols(),
                    refreshed_at=now.isoformat(),
                    refresh_reason="dynamic_watchlist_disabled",
                )
            )

        if not force and not self._refresh_due(now):
            snapshot = self.storage.get_json_state(self.SNAPSHOT_STATE_KEY, {})
            if snapshot:
                return self._persist_snapshot(
                    self._sanitize_snapshot(self._snapshot_from_state(snapshot))
                )

        previous = self.get_active_symbols()
        available = set(self._available_symbols())
        whitelist, blacklist = self.get_manual_lists()
        candidates = []
        for symbol in self._candidate_universe(whitelist):
            if symbol in blacklist:
                continue
            if self._is_disallowed(symbol):
                continue
            if available and symbol not in available:
                continue
            candidates.append(self._score_symbol(symbol))

        if not candidates:
            snapshot = WatchlistSnapshot(
                active_symbols=self._default_symbols(),
                refreshed_at=now.isoformat(),
                refresh_reason="candidate_scoring_unavailable",
            )
            return self._persist_snapshot(snapshot)

        selected = self._select_active_symbols(candidates)
        added = [symbol for symbol in selected if symbol not in previous]
        removed = [symbol for symbol in previous if symbol not in selected]
        snapshot = WatchlistSnapshot(
            active_symbols=selected,
            added_symbols=added,
            removed_symbols=removed,
            candidates=candidates,
            whitelist=whitelist,
            blacklist=blacklist,
            refreshed_at=now.isoformat(),
            refresh_reason="manual_refresh" if force else "scheduled_refresh",
        )
        return self._persist_snapshot(self._sanitize_snapshot(snapshot))

    def _score_symbol(self, symbol: str) -> WatchlistCandidate:
        ticker = self._fetch_ticker(symbol)
        is_core = symbol in self.settings.exchange.core_symbols
        volume = float(ticker.get("quoteVolume") or 0.0)
        if volume <= 0:
            base_volume = float(ticker.get("baseVolume") or 0.0)
            last_price = float(ticker.get("last") or 0.0)
            volume = base_volume * last_price
        change_pct = float(ticker.get("percentage") or 0.0)

        volume_score = min(1.0, volume / 100_000_000.0)
        momentum_score = 0.5 + tanh(change_pct / 12.0) * 0.5
        score = volume_score * 0.7 + momentum_score * 0.3
        notes = [
            f"volume_24h={volume:,.0f}",
            f"change_24h={change_pct:+.2f}%",
        ]
        if is_core:
            score = max(score, 0.95)
            notes.append("core_symbol")
        return WatchlistCandidate(
            symbol=symbol,
            score=round(score, 4),
            sector=self._sector(symbol),
            quote_volume_24h=volume,
            change_pct_24h=change_pct,
            is_core=is_core,
            notes=notes,
        )

    def _select_active_symbols(self, candidates: list[WatchlistCandidate]) -> list[str]:
        max_active = max(1, self.settings.exchange.max_active_symbols)
        whitelist, blacklist = self.get_manual_lists()
        core_symbols = [
            symbol
            for symbol in self.settings.exchange.core_symbols
            if symbol not in blacklist and any(c.symbol == symbol for c in candidates)
        ]
        satellites = sorted(
            [candidate for candidate in candidates if candidate.symbol not in core_symbols],
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        selected = list(core_symbols)
        sector_counts: dict[str, int] = {}

        for symbol in selected:
            sector = self._sector(symbol)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        forced_symbols = [
            symbol
            for symbol in whitelist
            if symbol not in selected and symbol not in blacklist and any(c.symbol == symbol for c in candidates)
        ]
        for symbol in forced_symbols:
            if len(selected) >= max_active:
                break
            selected.append(symbol)
            sector = self._sector(symbol)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        for candidate in satellites:
            if len(selected) >= max_active:
                break
            sector = self._sector(candidate.symbol)
            if sector_counts.get(sector, 0) >= self.settings.exchange.max_symbols_per_sector:
                continue
            selected.append(candidate.symbol)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) < max_active:
            for symbol in self._default_symbols():
                if symbol in blacklist:
                    continue
                if symbol not in selected:
                    selected.append(symbol)
                    sector = self._sector(symbol)
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                if len(selected) >= max_active:
                    break
        return selected[:max_active]

    def _fetch_ticker(self, symbol: str) -> dict[str, Any]:
        exchange = getattr(self.market, "exchange", None)
        market_symbol = f"{symbol}:USDT" if ":USDT" not in symbol else symbol
        if exchange is None or not hasattr(exchange, "fetch_ticker"):
            return {}
        try:
            return exchange.fetch_ticker(market_symbol) or {}
        except Exception:
            try:
                return exchange.fetch_ticker(symbol) or {}
            except Exception:
                return {}

    def _available_symbols(self) -> list[str]:
        if hasattr(self.market, "fetch_available_instruments"):
            try:
                raw = self.market.fetch_available_instruments()
                normalized = []
                for symbol in raw:
                    if ":USDT" in symbol:
                        normalized.append(symbol.replace(":USDT", ""))
                    else:
                        normalized.append(symbol)
                return list(dict.fromkeys(normalized))
            except Exception:
                return []
        return []

    def _refresh_due(self, now: datetime) -> bool:
        raw = self.storage.get_state(self.REFRESHED_AT_STATE_KEY)
        if not raw:
            return True
        try:
            last = datetime.fromisoformat(raw)
        except ValueError:
            return True
        return now - last >= timedelta(hours=self.settings.exchange.dynamic_watchlist_refresh_hours)

    def _default_symbols(self) -> list[str]:
        defaults = self._filter_allowed_symbols(list(self.settings.exchange.symbols))
        return defaults[: self.settings.exchange.max_active_symbols]

    def _persist_snapshot(self, snapshot: WatchlistSnapshot) -> WatchlistSnapshot:
        snapshot = self._sanitize_snapshot(snapshot)
        self.storage.set_json_state(self.ACTIVE_SYMBOLS_STATE_KEY, snapshot.active_symbols)
        self.storage.set_json_state(
            self.SNAPSHOT_STATE_KEY,
            {
                "active_symbols": snapshot.active_symbols,
                "added_symbols": snapshot.added_symbols,
                "removed_symbols": snapshot.removed_symbols,
                "candidates": [asdict(candidate) for candidate in snapshot.candidates],
                "whitelist": snapshot.whitelist,
                "blacklist": snapshot.blacklist,
                "refreshed_at": snapshot.refreshed_at,
                "refresh_reason": snapshot.refresh_reason,
            },
        )
        self.storage.set_state(self.REFRESHED_AT_STATE_KEY, snapshot.refreshed_at)
        return snapshot

    @staticmethod
    def _snapshot_from_state(payload: dict[str, Any]) -> WatchlistSnapshot:
        candidates = [
            WatchlistCandidate(**item)
            for item in payload.get("candidates", [])
        ]
        return WatchlistSnapshot(
            active_symbols=list(payload.get("active_symbols", [])),
            added_symbols=list(payload.get("added_symbols", [])),
            removed_symbols=list(payload.get("removed_symbols", [])),
            candidates=candidates,
            whitelist=list(payload.get("whitelist", [])),
            blacklist=list(payload.get("blacklist", [])),
            refreshed_at=str(payload.get("refreshed_at", "")),
            refresh_reason=str(payload.get("refresh_reason", "")),
        )

    def _candidate_universe(self, whitelist: list[str]) -> list[str]:
        ordered = []
        for symbol in self.settings.exchange.core_symbols + self.settings.exchange.candidate_symbols + whitelist:
            normalized = self._normalize_symbol(symbol)
            if (
                normalized
                and normalized not in ordered
                and not self._is_disallowed(normalized)
                and self._is_allowlisted(normalized)
            ):
                ordered.append(normalized)
        return ordered

    def _allowlisted_symbols(self) -> set[str]:
        return set(self._normalize_symbol_list(list(self.settings.exchange.symbols)))

    def _is_allowlisted(self, symbol: str) -> bool:
        allowed = self._allowlisted_symbols()
        if not allowed:
            return True
        return self._normalize_symbol(symbol) in allowed

    @staticmethod
    def _normalize_symbol(symbol: object) -> str:
        normalized_list = DynamicWatchlistService._normalize_symbol_list([symbol])
        return normalized_list[0] if normalized_list else ""

    def _candidate_allowed(self, symbol: str) -> bool:
        normalized = self._normalize_symbol(symbol)
        return bool(
            normalized
            and not self._is_disallowed(normalized)
            and self._is_allowlisted(normalized)
        )

    def _filter_allowed_symbols(self, symbols: list[str]) -> list[str]:
        filtered = []
        for symbol in symbols:
            normalized = self._normalize_symbol(symbol)
            if not normalized or not self._candidate_allowed(normalized):
                continue
            if normalized not in filtered:
                filtered.append(normalized)
        return filtered

    def _sanitize_snapshot(self, snapshot: WatchlistSnapshot) -> WatchlistSnapshot:
        allowed_candidate_symbols = set(
            self._filter_allowed_symbols([candidate.symbol for candidate in snapshot.candidates])
        )
        candidates = [
            candidate
            for candidate in snapshot.candidates
            if self._normalize_symbol(candidate.symbol) in allowed_candidate_symbols
        ]
        allowed_symbols = self._filter_allowed_symbols(snapshot.active_symbols)
        whitelist = self._filter_allowed_symbols(snapshot.whitelist)
        blacklist = list(dict.fromkeys(str(symbol).strip().upper() for symbol in snapshot.blacklist if str(symbol).strip()))
        added = [symbol for symbol in self._filter_allowed_symbols(snapshot.added_symbols) if symbol in allowed_symbols]
        removed = self._filter_allowed_symbols(snapshot.removed_symbols)
        return WatchlistSnapshot(
            active_symbols=allowed_symbols,
            added_symbols=added,
            removed_symbols=removed,
            candidates=candidates,
            whitelist=whitelist,
            blacklist=blacklist,
            refreshed_at=snapshot.refreshed_at,
            refresh_reason=snapshot.refresh_reason,
        )

    def _is_disallowed(self, symbol: str) -> bool:
        normalized = str(symbol).strip().upper()
        if normalized in {
            entry.strip().upper()
            for entry in self.settings.exchange.disallowed_symbols
        }:
            return True
        return self._sector(normalized) in {
            entry.strip().lower()
            for entry in self.settings.exchange.disallowed_sectors
        }

    @classmethod
    def _sector(cls, symbol: str) -> str:
        return cls.SYMBOL_SECTORS.get(symbol, "other")

    @staticmethod
    def _normalize_symbol_list(symbols: list[str]) -> list[str]:
        normalized = []
        for symbol in symbols:
            value = str(symbol).strip().upper().replace(" ", "")
            if not value:
                continue
            if "-" in value and "/" not in value:
                parts = value.split("-")
                if len(parts) >= 2:
                    value = f"{parts[0]}/{parts[1]}"
            if value.endswith("USDT") and "/" not in value and len(value) > 4:
                value = value[:-4] + "/USDT"
            if value not in normalized:
                normalized.append(value)
        return normalized
