"""Capital allocation inspired by portfolio-model separation."""

from __future__ import annotations

from collections import defaultdict

from .config import EvolutionConfig
from .models import CapitalAllocation, PromotionStage, ScoreCard


class CapitalAllocator:
    """Allocate capital only to promoted candidates."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def allocate(
        self,
        scorecards: list[ScoreCard],
        total_capital: float,
    ) -> list[CapitalAllocation]:
        eligible = [
            card for card in scorecards if card.stage in {PromotionStage.PAPER, PromotionStage.LIVE}
        ]
        if not eligible or total_capital <= 0:
            return []
        diversified = self._diversify(eligible)
        if not diversified:
            return []

        raw_scores = [max(0.0, card.total_score) for card in diversified]
        raw_total = sum(raw_scores)
        if raw_total <= 0:
            return []

        allocations: list[CapitalAllocation] = []
        for card, raw_score in zip(diversified, raw_scores):
            weight = raw_score / raw_total
            cap = (
                self.config.max_live_weight
                if card.stage == PromotionStage.LIVE
                else self.config.max_paper_weight
            )
            bounded_weight = min(weight, cap)
            allocations.append(
                CapitalAllocation(
                    strategy_id=card.genome.strategy_id,
                    stage=card.stage,
                    allocated_capital=round(total_capital * bounded_weight, 2),
                    weight=round(bounded_weight, 4),
                    reasons=list(card.reasons) + ["lineage_diversified"],
                )
            )
        return allocations

    def _diversify(self, eligible: list[ScoreCard]) -> list[ScoreCard]:
        stage_rank = {
            PromotionStage.LIVE: 2,
            PromotionStage.PAPER: 1,
        }
        ordered = sorted(
            eligible,
            key=lambda card: (
                stage_rank.get(card.stage, 0),
                card.total_score,
                card.deployment_score,
            ),
            reverse=True,
        )
        lineage_counts: dict[str, int] = defaultdict(int)
        diversified: list[ScoreCard] = []
        for card in ordered:
            lineage = self._lineage_key(card)
            if lineage_counts[lineage] >= self.config.max_allocations_per_lineage:
                continue
            lineage_counts[lineage] += 1
            diversified.append(card)
        return diversified

    @staticmethod
    def _lineage_key(card: ScoreCard) -> str:
        return card.genome.mutation_of or f"{card.genome.family}:seed"
