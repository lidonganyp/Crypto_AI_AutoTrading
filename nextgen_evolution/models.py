"""Core models for the next-generation evolution scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PromotionStage(str, Enum):
    REJECT = "reject"
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


class ExecutionAction(str, Enum):
    OBSERVE = "observe"
    PROMOTE_TO_SHADOW = "promote_to_shadow"
    PROMOTE_TO_PAPER = "promote_to_paper"
    PROMOTE_TO_LIVE = "promote_to_live"
    KEEP = "keep"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    PAUSE_NEW = "pause_new"
    EXIT = "exit"


class RepairActionType(str, Enum):
    TIGHTEN_RISK = "tighten_risk"
    RAISE_SELECTIVITY = "raise_selectivity"
    MUTATE_AND_REVALIDATE = "mutate_and_revalidate"
    QUARANTINE = "quarantine"
    RETIRE = "retire"
    REBUILD_LINEAGE = "rebuild_lineage"


class RuntimeLifecycleState(str, Enum):
    IDLE = "idle"
    SHADOW = "shadow"
    PAPER = "paper"
    LIMITED_LIVE = "limited_live"
    LIVE = "live"
    ROLLBACK = "rollback"


class ExecutionIntentAction(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    REDUCE = "reduce"
    HOLD = "hold"
    ROTATE = "rotate"
    SKIP = "skip"


@dataclass(slots=True)
class StrategyGenome:
    """A candidate trading strategy generated from a family and parameter set."""

    strategy_id: str
    family: str
    params: dict[str, float]
    mutation_of: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class ValidationMetrics:
    """Unified candidate metrics across the full validation ladder."""

    backtest_expectancy: float
    walkforward_expectancy: float
    shadow_expectancy: float
    live_expectancy: float
    max_drawdown_pct: float
    trade_count: int
    cost_drag_pct: float
    latency_ms: float
    regime_consistency: float


@dataclass(slots=True)
class ScoreCard:
    """Decision-ready candidate score."""

    genome: StrategyGenome
    stage: PromotionStage
    edge_score: float
    robustness_score: float
    deployment_score: float
    total_score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CapitalAllocation:
    """Capital assigned to a promoted strategy."""

    strategy_id: str
    stage: PromotionStage
    allocated_capital: float
    weight: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PortfolioAllocation:
    """Capital assigned across symbols from the unified portfolio layer."""

    symbol: str
    strategy_id: str
    family: str
    stage: PromotionStage
    allocated_capital: float
    weight: float
    score: float
    timeframe: str = ""
    entry_price: float = 0.0
    mark_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PortfolioPerformanceSnapshot:
    """Observed post-allocation portfolio performance snapshot."""

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    equity: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    open_positions: int = 0
    closed_positions: int = 0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    status: str = "active"
    notes: dict = field(default_factory=dict)


@dataclass(slots=True)
class StrategyRuntimeSnapshot:
    """Observed runtime state for one strategy candidate."""

    scorecard: ScoreCard
    metrics: ValidationMetrics
    runtime_id: str = ""
    allocated_capital: float = 0.0
    current_weight: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    current_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    health_status: str = "active"
    notes: dict = field(default_factory=dict)

    @property
    def strategy_id(self) -> str:
        return self.scorecard.genome.strategy_id

    @property
    def stage(self) -> PromotionStage:
        return self.scorecard.stage

    @property
    def directive_id(self) -> str:
        return self.runtime_id or self.scorecard.genome.strategy_id


@dataclass(slots=True)
class RuntimeEvidenceSnapshot:
    """Observed runtime evidence aggregated from actual execution records."""

    runtime_id: str
    symbol: str
    timeframe: str
    strategy_id: str
    family: str = ""
    open_position: bool = False
    current_capital: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_net_pnl: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    closed_trade_count: int = 0
    win_rate: float = 0.0
    consecutive_losses: int = 0
    health_status: str = "unproven"
    notes: dict = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionDirective:
    """Actionable deployment decision for one runtime strategy."""

    strategy_id: str
    action: ExecutionAction
    from_stage: PromotionStage
    target_stage: PromotionStage
    capital_multiplier: float = 1.0
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RepairPlan:
    """Repair or rollback action proposed by the autonomy layer."""

    strategy_id: str
    action: RepairActionType
    priority: int
    candidate_genome: StrategyGenome | None = None
    validation_stage: PromotionStage = PromotionStage.SHADOW
    capital_multiplier: float = 1.0
    runtime_overrides: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AutonomyDirective:
    """Combined execution and repair plan for one autonomy cycle."""

    execution: list[ExecutionDirective] = field(default_factory=list)
    repairs: list[RepairPlan] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeState:
    """Persistent deployment state for one runtime strategy instance."""

    runtime_id: str
    symbol: str
    timeframe: str
    strategy_id: str
    family: str
    lifecycle_state: RuntimeLifecycleState
    promotion_stage: PromotionStage
    target_stage: PromotionStage
    last_directive_action: ExecutionAction
    score: float = 0.0
    allocated_capital: float = 0.0
    desired_capital: float = 0.0
    current_capital: float = 0.0
    current_weight: float = 0.0
    capital_multiplier: float = 1.0
    limited_live_cycles: int = 0
    notes: dict = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionIntent:
    """Concrete execution intent derived from rollout state and current positions."""

    runtime_id: str
    symbol: str
    timeframe: str
    strategy_id: str
    family: str
    lifecycle_state: RuntimeLifecycleState
    action: ExecutionIntentAction
    desired_capital: float = 0.0
    current_capital: float = 0.0
    price: float = 0.0
    quantity: float = 0.0
    close_quantity: float = 0.0
    status: str = "planned"
    reasons: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)
