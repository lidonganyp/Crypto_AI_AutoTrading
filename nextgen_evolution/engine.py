"""Fused orchestration layer for the next-generation scaffold."""

from __future__ import annotations

from .allocator import CapitalAllocator
from .alpha_factory import AlphaFactory, StrategyPrimitive
from .config import EvolutionConfig
from .models import CapitalAllocation, ScoreCard, StrategyGenome, ValidationMetrics
from .promotion import PromotionEngine
from .validation import ValidationPipeline


class NextGenEvolutionEngine:
    """Research-to-deployment loop inspired by top open-source trading frameworks."""

    def __init__(self, config: EvolutionConfig | None = None):
        self.config = config or EvolutionConfig()
        self.factory = AlphaFactory(self.config)
        self.validation = ValidationPipeline(self.config)
        self.promotion = PromotionEngine()
        self.allocator = CapitalAllocator(self.config)

    def propose_population(
        self,
        primitives: list[StrategyPrimitive],
    ):
        return self.factory.generate(primitives)

    def evaluate_population(
        self,
        metrics_by_strategy: dict[str, ValidationMetrics],
        primitives: list[StrategyPrimitive],
    ) -> list[ScoreCard]:
        return self.evaluate_candidates(
            genomes=self.propose_population(primitives),
            metrics_by_strategy=metrics_by_strategy,
        )

    def evaluate_candidates(
        self,
        genomes: list[StrategyGenome],
        metrics_by_strategy: dict[str, ValidationMetrics],
    ) -> list[ScoreCard]:
        scorecards: list[ScoreCard] = []
        for genome in genomes:
            metrics = metrics_by_strategy.get(genome.strategy_id)
            if metrics is None:
                continue
            scorecards.append(self.validation.score(genome, metrics))
        return scorecards

    def build_deployment_plan(
        self,
        metrics_by_strategy: dict[str, ValidationMetrics],
        primitives: list[StrategyPrimitive],
        total_capital: float,
    ) -> tuple[list[ScoreCard], list[CapitalAllocation]]:
        scorecards, promoted, allocations = self.build_deployment_bundle(
            metrics_by_strategy=metrics_by_strategy,
            primitives=primitives,
            total_capital=total_capital,
        )
        return promoted, allocations

    def build_deployment_bundle(
        self,
        metrics_by_strategy: dict[str, ValidationMetrics],
        primitives: list[StrategyPrimitive],
        total_capital: float,
    ) -> tuple[list[ScoreCard], list[ScoreCard], list[CapitalAllocation]]:
        return self.build_candidate_bundle(
            genomes=self.propose_population(primitives),
            metrics_by_strategy=metrics_by_strategy,
            total_capital=total_capital,
        )

    def build_candidate_bundle(
        self,
        genomes: list[StrategyGenome],
        metrics_by_strategy: dict[str, ValidationMetrics],
        total_capital: float,
    ) -> tuple[list[ScoreCard], list[ScoreCard], list[CapitalAllocation]]:
        scorecards = self.evaluate_candidates(
            genomes=genomes,
            metrics_by_strategy=metrics_by_strategy,
        )
        promoted = self.promotion.shortlist(scorecards)
        allocations = self.allocator.allocate(promoted, total_capital)
        return scorecards, promoted, allocations
