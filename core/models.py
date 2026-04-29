"""Core models for CryptoAI v3."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class MarketRegime(str, Enum):
    UPTREND = "UPTREND"
    RANGE = "RANGE"
    DOWNTREND = "DOWNTREND"
    EXTREME_FEAR = "EXTREME_FEAR"
    EXTREME_GREED = "EXTREME_GREED"
    UNKNOWN = "UNKNOWN"


class SuggestedAction(str, Enum):
    OPEN_LONG = "OPEN_LONG"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OHLCV(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class FeatureSnapshot(BaseModel):
    symbol: str
    timeframe: str
    timestamp: datetime = Field(default_factory=utc_now)
    values: dict[str, float] = Field(default_factory=dict)
    valid: bool = True


class PredictionResult(BaseModel):
    symbol: str
    up_probability: float = Field(ge=0.0, le=1.0)
    feature_count: int = 0
    model_version: str = "fallback"
    model_id: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class ResearchInsight(BaseModel):
    symbol: str
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_warning: list[str] = Field(default_factory=list)
    key_reason: list[str] = Field(default_factory=list)
    suggested_action: SuggestedAction = SuggestedAction.HOLD
    raw_content: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class DecisionContext(BaseModel):
    symbol: str
    prediction: PredictionResult
    insight: ResearchInsight
    features: FeatureSnapshot
    trend_filter: bool
    volatility_factor: float
    liquidity_ratio: float
    final_score: float
    portfolio_rating: str = "HOLD"
    position_scale: float = 0.0
    direction: SignalDirection
    should_open: bool
    exit_reasons: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utc_now)


class AccountState(BaseModel):
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    daily_loss_pct: float
    weekly_loss_pct: float
    drawdown_pct: float
    total_exposure_pct: float
    open_positions: int
    cooldown_until: datetime | None = None
    circuit_breaker_active: bool = False


class RiskCheckResult(BaseModel):
    allowed: bool
    reason: str = ""
    allowed_risk_amount: float = 0.0
    allowed_position_value: float = 0.0
    liquidity_floor_used: float = 0.0
    observed_liquidity_ratio: float = 0.0
    dynamic_position_factor: float = 1.0
    portfolio_heat_factor: float = 1.0
    effective_max_total_exposure_pct: float = 0.0
    effective_max_positions: int = 0
    correlation_position_factor: float = 1.0
    correlation_effective_exposure_pct: float = 0.0
    correlation_crowded_symbols: list[str] = Field(default_factory=list)
    stop_loss_pct: float = 0.0
    take_profit_levels: list[float] = Field(default_factory=list)
    trailing_stop_drawdown_pct: float = 0.0


class ExecutionDecision(BaseModel):
    symbol: str
    direction: SignalDirection
    should_execute: bool
    portfolio_rating: str = "HOLD"
    position_scale: float = 0.0
    reason: str = ""
    quantity: float = 0.0
    position_value: float = 0.0
    stop_loss_pct: float = 0.0
    take_profit_levels: list[float] = Field(default_factory=list)
    trailing_stop_drawdown_pct: float = 0.0
    final_score: float = 0.0
    timestamp: datetime = Field(default_factory=utc_now)


class OrderRecord(BaseModel):
    order_id: str
    symbol: str
    side: str
    order_type: str
    status: OrderStatus
    price: float
    quantity: float
    reason: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class TradeSignal(BaseModel):
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    key_factors: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    suggested_stop_pct: float = 0.05
    suggested_target_pct: float = 0.10
    source: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class FusedSignal(BaseModel):
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[TradeSignal] = Field(default_factory=list)
    rationale: str
    risk_level: RiskLevel = RiskLevel.MEDIUM


class Position(BaseModel):
    symbol: str
    direction: SignalDirection
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    current_price: float | None = None
    unrealized_pnl_pct: float | None = None


class Trade(BaseModel):
    id: str = ""
    symbol: str
    direction: SignalDirection
    entry_price: float
    exit_price: float | None = None
    quantity: float
    entry_time: datetime
    exit_time: datetime | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    rationale: str = ""
    confidence: float = 0.0


class TradeReflection(BaseModel):
    trade_id: str
    symbol: str
    direction: SignalDirection | str
    confidence: float
    rationale: str
    source: str = ""
    experience_weight: float = 0.0
    realized_return_pct: float | None = None
    outcome_24h: float | None = None
    outcome_7d: float | None = None
    correct_signals: list[str] = Field(default_factory=list)
    wrong_signals: list[str] = Field(default_factory=list)
    lesson: str = ""
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    timestamp: datetime = Field(default_factory=utc_now)


class SentimentData(BaseModel):
    source: str
    value: float | None = None
    label: str = ""
    summary: str = ""
    timestamp: datetime = Field(default_factory=utc_now)
