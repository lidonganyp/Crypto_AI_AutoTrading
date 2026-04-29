"""Strategy generation inspired by large-scale research engines."""

from __future__ import annotations

from dataclasses import dataclass
import random

from .config import EvolutionConfig
from .models import StrategyGenome


@dataclass(slots=True)
class StrategyPrimitive:
    """Seed family used to generate mutated strategy candidates."""

    family: str
    base_params: dict[str, float]
    tags: tuple[str, ...] = ()


class AlphaFactory:
    """Generate a population of mutated candidate strategies."""

    def __init__(self, config: EvolutionConfig, seed: int = 7):
        self.config = config
        self._rng = random.Random(seed)

    def generate(self, primitives: list[StrategyPrimitive]) -> list[StrategyGenome]:
        population: list[StrategyGenome] = []
        for primitive in primitives:
            root_id = f"{primitive.family}:seed"
            population.append(
                StrategyGenome(
                    strategy_id=root_id,
                    family=primitive.family,
                    params=dict(primitive.base_params),
                    tags=primitive.tags,
                )
            )
            for index in range(self.config.mutation_per_seed):
                population.append(self._mutate(root_id, primitive, index))
            if len(population) >= self.config.experiment_budget:
                break
        return population[: self.config.experiment_budget]

    def _mutate(
        self,
        parent_id: str,
        primitive: StrategyPrimitive,
        index: int,
    ) -> StrategyGenome:
        params: dict[str, float] = {}
        for key, value in primitive.base_params.items():
            delta = self._rng.uniform(
                -self.config.mutation_scale,
                self.config.mutation_scale,
            )
            params[key] = round(max(0.0, value * (1.0 + delta)), 6)
        return StrategyGenome(
            strategy_id=f"{primitive.family}:mut{index + 1}",
            family=primitive.family,
            params=params,
            mutation_of=parent_id,
            tags=primitive.tags,
        )
