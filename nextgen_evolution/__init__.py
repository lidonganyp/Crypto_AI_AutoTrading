"""Next-generation self-evolving trading architecture scaffold."""

from .director import AutonomousDirector
from .allocator import CapitalAllocator
from .alpha_factory import AlphaFactory, StrategyPrimitive
from .execution_bridge import AutonomyPaperBridge
from .live_cycle import AutonomyLiveCycleRunner
from .live_bridge import AutonomyLiveBridge
from .live_runtime import AutonomyLiveRuntime, AutonomyLiveStatus
from .lineage_rebuild import LineageRebuildPlanner
from .planner import AutonomyPlanner
from .config import EvolutionConfig
from .data_feed import SQLiteOHLCVFeed
from .engine import NextGenEvolutionEngine
from .experiment_lab import ExperimentLab
from .feature_miner import FeatureMiner
from .portfolio_allocator import PortfolioAllocator
from .portfolio_alerts import PortfolioAlert, PortfolioAlertEvaluator
from .portfolio_monitor import PortfolioMonitor
from .portfolio_tracker import PortfolioTracker
from .promotion import PromotionEngine
from .promotion_registry import PromotionRegistry
from .repair import StrategyRepairEngine
from .repair_cycle import RepairCycleRunner, RepairExecutionResult
from .repair_feedback import RepairFeedbackEngine, RepairFeedbackSummary
from .repair_reentry import RepairReentryPlanner
from .rollout import RolloutStateMachine
from .runtime_evidence import RuntimeEvidenceCollector
from .scheduler import ExperimentScheduler
from .validation import ValidationPipeline
from .models import (
    AutonomyDirective,
    CapitalAllocation,
    ExecutionAction,
    ExecutionDirective,
    ExecutionIntent,
    ExecutionIntentAction,
    PortfolioPerformanceSnapshot,
    PromotionStage,
    RepairActionType,
    RepairPlan,
    RuntimeEvidenceSnapshot,
    RuntimeLifecycleState,
    RuntimeState,
    ScoreCard,
    StrategyGenome,
    StrategyRuntimeSnapshot,
    ValidationMetrics,
)

__all__ = [
    "AutonomyDirective",
    "AutonomousDirector",
    "AutonomyPlanner",
    "AutonomyPaperBridge",
    "AutonomyLiveCycleRunner",
    "AutonomyLiveBridge",
    "AutonomyLiveRuntime",
    "AutonomyLiveStatus",
    "LineageRebuildPlanner",
    "AlphaFactory",
    "CapitalAllocation",
    "CapitalAllocator",
    "EvolutionConfig",
    "ExecutionAction",
    "ExecutionDirective",
    "ExecutionIntent",
    "ExecutionIntentAction",
    "ExperimentLab",
    "FeatureMiner",
    "NextGenEvolutionEngine",
    "PortfolioAllocator",
    "PortfolioAlert",
    "PortfolioAlertEvaluator",
    "PortfolioMonitor",
    "PortfolioPerformanceSnapshot",
    "PortfolioTracker",
    "PromotionStage",
    "PromotionEngine",
    "PromotionRegistry",
    "RepairActionType",
    "RepairCycleRunner",
    "RepairExecutionResult",
    "RepairFeedbackEngine",
    "RepairFeedbackSummary",
    "RepairReentryPlanner",
    "RepairPlan",
    "RuntimeEvidenceCollector",
    "RuntimeEvidenceSnapshot",
    "RolloutStateMachine",
    "ScoreCard",
    "StrategyGenome",
    "RuntimeLifecycleState",
    "RuntimeState",
    "ExperimentScheduler",
    "SQLiteOHLCVFeed",
    "StrategyRepairEngine",
    "StrategyPrimitive",
    "StrategyRuntimeSnapshot",
    "ValidationMetrics",
    "ValidationPipeline",
]
