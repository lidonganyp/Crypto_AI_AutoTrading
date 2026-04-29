"""Data adapters for the next-generation evolution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.storage import Storage


@dataclass(slots=True)
class SQLiteOHLCVFeed:
    """Read OHLCV history from the existing CryptoAI SQLite store."""

    db_path: str
    storage: Storage = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.storage = Storage(str(self.db_path))

    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 1000,
        since: int | None = None,
    ) -> list[dict]:
        for variant in self._symbol_variants(symbol):
            candles = self.storage.get_ohlcv(
                variant,
                timeframe,
                since=since,
                limit=limit,
            )
            if candles:
                return list(reversed(candles))
        return []

    def list_symbols(self, timeframe: str, *, min_rows: int = 200) -> list[str]:
        with self.storage._conn() as conn:
            rows = conn.execute(
                """SELECT symbol, COUNT(*) AS row_count
                   FROM ohlcv
                   WHERE timeframe = ?
                   GROUP BY symbol
                   HAVING COUNT(*) >= ?
                   ORDER BY row_count DESC, symbol ASC""",
                (timeframe, min_rows),
            ).fetchall()
        return [str(row["symbol"]) for row in rows]

    @staticmethod
    def _symbol_variants(symbol: str) -> tuple[str, ...]:
        text = str(symbol or "").strip()
        if not text:
            return ()
        variants = [text]
        if ":USDT" not in text:
            variants.append(f"{text}:USDT")
        return tuple(dict.fromkeys(variants))

    @classmethod
    def from_project_default(cls, project_root: str | Path) -> "SQLiteOHLCVFeed":
        return cls(str(Path(project_root) / "data" / "cryptoai.db"))
