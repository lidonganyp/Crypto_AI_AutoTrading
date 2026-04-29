"""Promotion engine for stage transitions."""

from __future__ import annotations

from .models import PromotionStage, ScoreCard


class PromotionEngine:
    """Filter and sort candidates ready for deployment."""

    def shortlist(self, scorecards: list[ScoreCard]) -> list[ScoreCard]:
        stage_rank = {
            PromotionStage.LIVE: 3,
            PromotionStage.PAPER: 2,
            PromotionStage.SHADOW: 1,
            PromotionStage.REJECT: 0,
        }
        promoted = [
            card
            for card in scorecards
            if card.stage in {PromotionStage.SHADOW, PromotionStage.PAPER, PromotionStage.LIVE}
        ]
        promoted.sort(
            key=lambda card: (
                stage_rank[card.stage],
                card.total_score,
                card.deployment_score,
            ),
            reverse=True,
        )
        return promoted
