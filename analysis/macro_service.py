"""Macro summary service for CryptoAI v3."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class MacroSummary:
    score: float
    position_adjustment: float
    summary: str
    timestamp: str


class MacroService:
    """Provide a lightweight macro summary with deterministic fallback."""

    def get_summary(self, fear_greed: float | None = None) -> MacroSummary:
        score = 0.0
        position_adjustment = 1.0
        summary = "Macro context neutral."

        if fear_greed is not None:
            if fear_greed >= 75:
                score = 0.15
                position_adjustment = 0.9
                summary = "Risk appetite elevated; avoid aggressive sizing."
            elif fear_greed <= 25:
                score = -0.15
                position_adjustment = 0.8
                summary = "Market stress elevated; keep size defensive."

        return MacroSummary(
            score=score,
            position_adjustment=position_adjustment,
            summary=summary,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
