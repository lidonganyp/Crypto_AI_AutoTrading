"""Configuration for CryptoAI v3."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_ENV_PATH = Path(
    os.getenv(
        "CRYPTOAI_SHARED_ENV_FILE",
        str(Path.home() / ".config" / "cryptoai" / "llm.env"),
    )
)
load_dotenv(SHARED_ENV_PATH, override=True)
load_dotenv(PROJECT_ROOT / ".env", override=False)


class AppSettings(BaseSettings):
    env: str = Field(
        default="development",
        validation_alias=AliasChoices("APP_ENV", "ENV"),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("APP_LOG_LEVEL", "LOG_LEVEL"),
    )
    db_path: str = Field(
        default="data/cryptoai.db",
        validation_alias=AliasChoices("DB_PATH", "APP_DB_PATH"),
    )
    project_root: Path = PROJECT_ROOT
    runtime_mode: str = Field(
        default="paper",
        validation_alias=AliasChoices("APP_RUNTIME_MODE", "RUNTIME_MODE"),
    )
    allow_live_orders: bool = Field(
        default=False,
        validation_alias=AliasChoices("APP_ALLOW_LIVE_ORDERS", "ALLOW_LIVE_ORDERS"),
    )
    low_resource_mode: bool = Field(
        default=False,
        validation_alias=AliasChoices("APP_LOW_RESOURCE_MODE", "LOW_RESOURCE_MODE"),
    )
    language: str = Field(
        default="zh",
        validation_alias=AliasChoices("APP_LANGUAGE", "LANGUAGE"),
    )


class ExchangeSettings(BaseSettings):
    provider: str = Field(
        default="okx",
        validation_alias=AliasChoices("EXCHANGE_PROVIDER", "PROVIDER"),
    )
    proxy_url: str = Field(
        default="",
        validation_alias=AliasChoices("EXCHANGE_PROXY_URL", "PROXY_URL"),
    )
    symbols: list[str] = Field(
        default=[
            "BTC/USDT",
            "ETH/USDT",
        ]
    )
    candidate_symbols: list[str] = Field(
        default=[
            "BTC/USDT",
            "ETH/USDT",
        ]
    )
    core_symbols: list[str] = Field(default=["BTC/USDT", "ETH/USDT"])
    disallowed_symbols: list[str] = Field(
        default=[
            "DOGE/USDT",
            "SHIB/USDT",
            "PEPE/USDT",
            "WIF/USDT",
            "BONK/USDT",
        ]
    )
    disallowed_sectors: list[str] = Field(default=["meme"])
    max_active_symbols: int = Field(
        default=2,
        validation_alias=AliasChoices("EXCHANGE_MAX_ACTIVE_SYMBOLS"),
    )
    max_symbols_per_sector: int = Field(default=2)
    dynamic_watchlist_enabled: bool = Field(default=True)
    dynamic_watchlist_refresh_hours: int = Field(default=24)
    timeframes: list[str] = Field(default=["1h", "4h", "1d"])
    data_delay_seconds: int = Field(default=5)
    data_latency_warning_seconds: int = Field(default=5)
    data_latency_circuit_breaker_seconds: int = Field(default=10)
    data_quality_max_missing_ratio: float = Field(default=0.001)
    market_data_failover_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("EXCHANGE_MARKET_DATA_FAILOVER_ENABLED"),
    )
    market_data_retry_count: int = Field(
        default=2,
        validation_alias=AliasChoices("EXCHANGE_MARKET_DATA_RETRY_COUNT"),
    )
    market_data_retry_delay_seconds: float = Field(
        default=0.5,
        validation_alias=AliasChoices("EXCHANGE_MARKET_DATA_RETRY_DELAY_SECONDS"),
    )
    api_key: SecretStr = Field(
        default="",
        validation_alias=AliasChoices(
            "EXCHANGE_API_KEY",
            "BINANCE_API_KEY",
            "OKX_API_KEY",
        ),
    )
    api_secret: SecretStr = Field(
        default="",
        validation_alias=AliasChoices(
            "EXCHANGE_API_SECRET",
            "BINANCE_API_SECRET",
            "OKX_API_SECRET",
        ),
    )
    api_passphrase: SecretStr = Field(
        default="",
        validation_alias=AliasChoices(
            "OKX_API_PASSPHRASE",
            "EXCHANGE_API_PASSPHRASE",
        ),
    )


class LLMSettings(BaseSettings):
    deepseek_api_key: SecretStr = Field(default="")
    deepseek_api_base: str = Field(default="https://api.deepseek.com/v1")
    deepseek_model: str = Field(default="deepseek-chat")

    qwen_api_key: SecretStr = Field(default="")
    qwen_api_base: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    qwen_model: str = Field(default="qwen-max")
    request_timeout_seconds: float = Field(default=18.0)
    connect_timeout_seconds: float = Field(default=5.0)
    runtime_failure_backoff_seconds: int = Field(default=90)
    research_interval_seconds: int = Field(default=4 * 3600)


class NewsSettings(BaseSettings):
    coindesk_rss_url: str = Field(default="https://www.coindesk.com/arc/outboundfeeds/rss/")
    cointelegraph_rss_url: str = Field(default="https://cointelegraph.com/rss")
    jin10_url: str = Field(default="https://www.jin10.com/")
    cryptopanic_api_key: SecretStr = Field(
        default="",
        validation_alias=AliasChoices("CRYPTOPANIC_API_KEY"),
    )
    cryptopanic_api_base: str = Field(default="https://cryptopanic.com/api/v1/posts/")


class OnchainSettings(BaseSettings):
    glassnode_api_base: str = Field(default="https://api.glassnode.com/v1/metrics")
    glassnode_api_key: SecretStr = Field(
        default="",
        validation_alias=AliasChoices("GLASSNODE_API_KEY"),
    )
    glassnode_exchange_balance_path: str = Field(default="/distribution/balance_exchanges")
    glassnode_exchange_balance_relative_path: str = Field(
        default="/distribution/balance_exchanges_relative"
    )
    glassnode_interval: str = Field(default="24h")
    glassnode_exchange: str = Field(default="aggregated")
    coinmetrics_api_base: str = Field(default="https://api.coinmetrics.io/v4")
    coinmetrics_api_key: SecretStr = Field(
        default="",
        validation_alias=AliasChoices("COINMETRICS_API_KEY"),
    )
    coinmetrics_exchange: str = Field(default="coinbase")
    coinmetrics_exchange_asset_metric: str = Field(default="volume_reported_spot_usd_1d")
    coinmetrics_asset_metric: str = Field(default="TxCnt")


class NotificationSettings(BaseSettings):
    webhook_url: str = Field(default="")
    webhook_secret: SecretStr = Field(default="")
    critical_webhook_url: str = Field(default="")
    critical_webhook_secret: SecretStr = Field(default="")
    feishu_webhook_url: str = Field(default="")
    feishu_webhook_secret: SecretStr = Field(default="")
    feishu_critical_webhook_url: str = Field(default="")
    feishu_critical_webhook_secret: SecretStr = Field(default="")


class SentimentSettings(BaseSettings):
    lunarcrush_api_base: str = Field(default="https://lunarcrush.com/api4")
    lunarcrush_api_key: SecretStr = Field(
        default="",
        validation_alias=AliasChoices("LUNARCRUSH_API_KEY"),
    )


class ABTestingSettings(BaseSettings):
    enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("AB_TESTING_ENABLED"),
    )
    challenger_model_path: str = Field(
        default="data/models/xgboost_challenger.json",
        validation_alias=AliasChoices("AB_TESTING_CHALLENGER_MODEL_PATH"),
    )
    challenger_allocation_pct: float = Field(
        default=0.10,
        validation_alias=AliasChoices("AB_TESTING_CHALLENGER_ALLOCATION_PCT"),
    )
    execute_challenger_live: bool = Field(
        default=False,
        validation_alias=AliasChoices("AB_TESTING_EXECUTE_CHALLENGER_LIVE"),
    )


class ModelSettings(BaseSettings):
    xgboost_model_path: str = Field(default="data/models/xgboost_v2.json")
    xgboost_probability_threshold: float = Field(default=0.70)
    final_score_threshold: float = Field(default=0.80)
    enable_fallback_predictor: bool = Field(default=True)
    xgboost_num_boost_round: int = Field(default=200)
    xgboost_nthread: int = Field(default=0)


class StrategySettings(BaseSettings):
    primary_timeframe: str = Field(default="4h")
    lower_timeframe: str = Field(default="1h")
    higher_timeframe: str = Field(default="1d")
    analysis_interval_seconds: int = Field(default=4 * 3600)
    position_guard_interval_seconds: int = Field(default=5 * 60)
    entry_scan_interval_seconds: int = Field(default=10 * 60)
    fast_alpha_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("FAST_ALPHA_ENABLED"),
    )
    fast_alpha_entry_scan_interval_seconds: int = Field(
        default=5 * 60,
        validation_alias=AliasChoices("FAST_ALPHA_ENTRY_SCAN_INTERVAL_SECONDS"),
    )
    fast_alpha_symbols: list[str] = Field(default=["BTC/USDT", "ETH/USDT"])
    fast_alpha_min_probability_pct: float = Field(
        default=0.66,
        validation_alias=AliasChoices("FAST_ALPHA_MIN_PROBABILITY_PCT"),
    )
    fast_alpha_eth_min_probability_pct: float = Field(
        default=0.54,
        validation_alias=AliasChoices("FAST_ALPHA_ETH_MIN_PROBABILITY_PCT"),
    )
    fast_alpha_min_final_score: float = Field(
        default=0.54,
        validation_alias=AliasChoices("FAST_ALPHA_MIN_FINAL_SCORE"),
    )
    fast_alpha_min_review_score: float = Field(
        default=0.12,
        validation_alias=AliasChoices("FAST_ALPHA_MIN_REVIEW_SCORE"),
    )
    fast_alpha_close_override_min_review_score: float = Field(
        default=-0.25,
        validation_alias=AliasChoices("FAST_ALPHA_CLOSE_OVERRIDE_MIN_REVIEW_SCORE"),
    )
    fast_alpha_max_xgboost_gap_pct: float = Field(
        default=0.20,
        validation_alias=AliasChoices("FAST_ALPHA_MAX_XGBOOST_GAP_PCT"),
    )
    fast_alpha_max_final_score_gap_pct: float = Field(
        default=0.10,
        validation_alias=AliasChoices("FAST_ALPHA_MAX_FINAL_SCORE_GAP_PCT"),
    )
    fast_alpha_position_scale: float = Field(
        default=0.35,
        validation_alias=AliasChoices("FAST_ALPHA_POSITION_SCALE"),
    )
    fast_alpha_fixed_stop_loss_pct: float = Field(
        default=0.018,
        validation_alias=AliasChoices("FAST_ALPHA_FIXED_STOP_LOSS_PCT"),
    )
    fast_alpha_take_profit_levels: list[float] = Field(default=[0.012, 0.025])
    fast_alpha_max_hold_hours: int = Field(
        default=6,
        validation_alias=AliasChoices("FAST_ALPHA_MAX_HOLD_HOURS"),
    )
    fast_alpha_de_risk_hours: float = Field(
        default=1.0,
        validation_alias=AliasChoices("FAST_ALPHA_DE_RISK_HOURS"),
    )
    fast_alpha_de_risk_pnl_ratio: float = Field(
        default=0.004,
        validation_alias=AliasChoices("FAST_ALPHA_DE_RISK_PNL_RATIO"),
    )
    fast_alpha_portfolio_heat_override_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("FAST_ALPHA_PORTFOLIO_HEAT_OVERRIDE_ENABLED"),
    )
    fast_alpha_portfolio_heat_override_scale: float = Field(
        default=0.50,
        validation_alias=AliasChoices("FAST_ALPHA_PORTFOLIO_HEAT_OVERRIDE_SCALE"),
    )
    fast_alpha_liquidity_override_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("FAST_ALPHA_LIQUIDITY_OVERRIDE_ENABLED"),
    )
    fast_alpha_liquidity_floor_ratio: float = Field(
        default=0.30,
        validation_alias=AliasChoices("FAST_ALPHA_LIQUIDITY_FLOOR_RATIO"),
    )
    short_horizon_adaptive_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("SHORT_HORIZON_ADAPTIVE_ENABLED"),
    )
    short_horizon_adaptive_lookback_trades: int = Field(
        default=20,
        validation_alias=AliasChoices("SHORT_HORIZON_ADAPTIVE_LOOKBACK_TRADES"),
    )
    short_horizon_adaptive_min_closed_trades: int = Field(
        default=6,
        validation_alias=AliasChoices("SHORT_HORIZON_ADAPTIVE_MIN_CLOSED_TRADES"),
    )
    short_horizon_adaptive_positive_expectancy_pct: float = Field(
        default=0.12,
        validation_alias=AliasChoices("SHORT_HORIZON_ADAPTIVE_POSITIVE_EXPECTANCY_PCT"),
    )
    short_horizon_adaptive_positive_profit_factor: float = Field(
        default=1.10,
        validation_alias=AliasChoices("SHORT_HORIZON_ADAPTIVE_POSITIVE_PROFIT_FACTOR"),
    )
    short_horizon_adaptive_negative_expectancy_pct: float = Field(
        default=-0.08,
        validation_alias=AliasChoices("SHORT_HORIZON_ADAPTIVE_NEGATIVE_EXPECTANCY_PCT"),
    )
    short_horizon_adaptive_negative_profit_factor: float = Field(
        default=0.95,
        validation_alias=AliasChoices("SHORT_HORIZON_ADAPTIVE_NEGATIVE_PROFIT_FACTOR"),
    )
    short_horizon_adaptive_max_drawdown_pct: float = Field(
        default=4.0,
        validation_alias=AliasChoices("SHORT_HORIZON_ADAPTIVE_MAX_DRAWDOWN_PCT"),
    )
    short_horizon_adaptive_probability_discount_pct: float = Field(
        default=0.04,
        validation_alias=AliasChoices("SHORT_HORIZON_ADAPTIVE_PROBABILITY_DISCOUNT_PCT"),
    )
    short_horizon_adaptive_final_score_discount: float = Field(
        default=0.03,
        validation_alias=AliasChoices("SHORT_HORIZON_ADAPTIVE_FINAL_SCORE_DISCOUNT"),
    )
    short_horizon_adaptive_review_score_discount: float = Field(
        default=0.05,
        validation_alias=AliasChoices("SHORT_HORIZON_ADAPTIVE_REVIEW_SCORE_DISCOUNT"),
    )
    fast_alpha_short_horizon_review_policy_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_REVIEW_POLICY_ENABLED"),
    )
    fast_alpha_short_horizon_setup_avg_outcome_floor_pct: float = Field(
        default=-0.18,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_SETUP_AVG_OUTCOME_FLOOR_PCT"),
    )
    fast_alpha_short_horizon_experience_avg_outcome_floor_pct: float = Field(
        default=-0.14,
        validation_alias=AliasChoices(
            "FAST_ALPHA_SHORT_HORIZON_EXPERIENCE_AVG_OUTCOME_FLOOR_PCT"
        ),
    )
    fast_alpha_short_horizon_setup_score_bonus: float = Field(
        default=0.10,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_SETUP_SCORE_BONUS"),
    )
    fast_alpha_short_horizon_experience_score_bonus: float = Field(
        default=0.08,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_EXPERIENCE_SCORE_BONUS"),
    )
    fast_alpha_short_horizon_risk_warning_score_bonus: float = Field(
        default=0.05,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_RISK_WARNING_SCORE_BONUS"),
    )
    fast_alpha_short_horizon_max_risk_warnings: int = Field(
        default=1,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_MAX_RISK_WARNINGS"),
    )
    fast_alpha_short_horizon_trade_feedback_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_TRADE_FEEDBACK_ENABLED"),
    )
    fast_alpha_short_horizon_trade_feedback_lookback_trades: int = Field(
        default=8,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_TRADE_FEEDBACK_LOOKBACK_TRADES"),
    )
    fast_alpha_short_horizon_trade_feedback_min_closed_trades: int = Field(
        default=3,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_TRADE_FEEDBACK_MIN_CLOSED_TRADES"),
    )
    fast_alpha_short_horizon_trade_feedback_negative_expectancy_pct: float = Field(
        default=-0.05,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_TRADE_FEEDBACK_NEGATIVE_EXPECTANCY_PCT"),
    )
    fast_alpha_short_horizon_trade_feedback_negative_profit_factor: float = Field(
        default=0.95,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_TRADE_FEEDBACK_NEGATIVE_PROFIT_FACTOR"),
    )
    fast_alpha_short_horizon_trade_feedback_negative_max_drawdown_pct: float = Field(
        default=3.0,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_TRADE_FEEDBACK_NEGATIVE_MAX_DRAWDOWN_PCT"),
    )
    fast_alpha_short_horizon_trade_feedback_warming_scale: float = Field(
        default=0.70,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_TRADE_FEEDBACK_WARMING_SCALE"),
    )
    fast_alpha_short_horizon_trade_feedback_healthy_scale: float = Field(
        default=1.0,
        validation_alias=AliasChoices("FAST_ALPHA_SHORT_HORIZON_TRADE_FEEDBACK_HEALTHY_SCALE"),
    )
    sentiment_weight: float = Field(default=0.20)
    min_liquidity_ratio: float = Field(default=0.80)
    max_hold_hours: int = Field(default=48)
    fixed_stop_loss_pct: float = Field(default=0.05)
    take_profit_levels: list[float] = Field(default=[0.05, 0.08])
    trailing_stop_drawdown_pct: float = Field(default=0.30)
    trend_reversal_probability: float = Field(default=0.40)
    sentiment_exit_threshold: float = Field(default=-0.30)
    near_miss_shadow_enabled: bool = Field(default=True)
    near_miss_xgboost_gap_pct: float = Field(default=0.20)
    near_miss_final_score_gap_pct: float = Field(default=0.25)
    paper_canary_enabled: bool = Field(default=True)
    paper_canary_position_scale: float = Field(default=0.35)
    paper_canary_xgboost_gap_pct: float = Field(default=0.12)
    paper_canary_final_score_gap_pct: float = Field(default=0.15)
    paper_canary_min_review_score: float = Field(default=0.15)
    paper_canary_soft_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("PAPER_CANARY_SOFT_ENABLED"),
    )
    paper_canary_offensive_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("PAPER_CANARY_OFFENSIVE_ENABLED"),
    )
    paper_canary_soft_review_min_score: float = Field(default=-0.05)
    paper_canary_soft_position_scale: float = Field(default=0.35)
    paper_canary_offensive_review_min_score: float = Field(default=0.0)
    paper_canary_offensive_position_scale: float = Field(default=0.75)
    execution_soft_entry_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("EXECUTION_SOFT_ENTRY_ENABLED"),
    )
    execution_soft_entry_review_min_score: float = Field(
        default=0.08,
        validation_alias=AliasChoices("EXECUTION_SOFT_ENTRY_REVIEW_MIN_SCORE"),
    )
    execution_soft_entry_position_scale: float = Field(
        default=0.50,
        validation_alias=AliasChoices("EXECUTION_SOFT_ENTRY_POSITION_SCALE"),
    )
    execution_soft_entry_max_final_score_gap_pct: float = Field(
        default=0.03,
        validation_alias=AliasChoices("EXECUTION_SOFT_ENTRY_MAX_FINAL_SCORE_GAP_PCT"),
    )
    extreme_fear_conservative_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("EXTREME_FEAR_CONSERVATIVE_ENABLED"),
    )
    extreme_fear_conservative_xgboost_bonus_pct: float = Field(
        default=0.08,
        validation_alias=AliasChoices("EXTREME_FEAR_CONSERVATIVE_XGBOOST_BONUS_PCT"),
    )
    extreme_fear_conservative_final_score_bonus: float = Field(
        default=0.10,
        validation_alias=AliasChoices("EXTREME_FEAR_CONSERVATIVE_FINAL_SCORE_BONUS"),
    )
    extreme_fear_conservative_liquidity_bonus_ratio: float = Field(
        default=0.15,
        validation_alias=AliasChoices("EXTREME_FEAR_CONSERVATIVE_LIQUIDITY_BONUS_RATIO"),
    )
    extreme_fear_conservative_position_scale: float = Field(
        default=0.25,
        validation_alias=AliasChoices("EXTREME_FEAR_CONSERVATIVE_POSITION_SCALE"),
    )
    extreme_fear_quant_override_score_bonus: float = Field(default=0.55)
    extreme_fear_quant_override_min_probability_gap_pct: float = Field(default=0.05)
    extreme_fear_quant_override_min_probability_pct: float = Field(default=0.78)
    extreme_fear_quant_override_liquidity_floor_ratio: float = Field(default=0.90)
    adaptive_liquidity_enabled: bool = Field(default=True)
    adaptive_liquidity_floor_min_ratio: float = Field(default=0.35)
    adaptive_liquidity_percentile: float = Field(default=0.35)
    adaptive_liquidity_lookback_snapshots: int = Field(default=120)
    adaptive_liquidity_same_hour_min_samples: int = Field(default=6)
    adaptive_liquidity_hour_window_hours: int = Field(default=1)
    liquidity_microstructure_support_spread_pct: float = Field(default=0.0015)
    liquidity_microstructure_support_depth_usd: float = Field(default=20000.0)


class RiskSettings(BaseSettings):
    single_trade_risk_pct: float = Field(default=0.005)
    max_symbol_exposure_pct: float = Field(default=0.20)
    max_total_exposure_pct: float = Field(default=0.50)
    correlation_high_threshold: float = Field(default=0.75)
    correlation_medium_threshold: float = Field(default=0.50)
    correlation_soft_exposure_pct: float = Field(default=0.30)
    correlation_hard_exposure_pct: float = Field(default=0.45)
    correlation_position_floor: float = Field(default=0.35)
    portfolio_heat_min_recent_trades: int = Field(default=5)
    portfolio_heat_exposure_floor_pct: float = Field(default=0.40)
    portfolio_heat_soft_drawdown_velocity_pct: float = Field(default=0.60)
    portfolio_heat_hard_drawdown_velocity_pct: float = Field(default=1.50)
    portfolio_heat_soft_return_volatility_pct: float = Field(default=1.50)
    portfolio_heat_hard_return_volatility_pct: float = Field(default=3.00)
    portfolio_heat_soft_loss_cluster_pct: float = Field(default=40.0)
    portfolio_heat_hard_loss_cluster_pct: float = Field(default=70.0)
    max_positions: int = Field(default=3)
    daily_loss_limit_pct: float = Field(default=0.02)
    weekly_loss_limit_pct: float = Field(default=0.05)
    max_drawdown_pct: float = Field(default=0.12)
    consecutive_loss_cooldown_hours: int = Field(default=24)
    consecutive_loss_limit: int = Field(default=3)
    abnormal_move_pct_5m: float = Field(default=0.05)
    abnormal_move_cooldown_minutes: int = Field(default=30)
    api_failure_circuit_breaker_count: int = Field(default=3)
    weekly_loss_cooldown_days: int = Field(default=3)
    drawdown_pause_pct: float = Field(default=0.10)
    drawdown_cooldown_days: int = Field(default=7)
    model_accuracy_floor_pct: float = Field(default=55.0)
    model_accuracy_min_samples: int = Field(default=5)
    model_accuracy_cooldown_hours: int = Field(default=24)
    model_degrade_floor_pct: float = Field(default=58.0)
    model_disable_floor_pct: float = Field(default=50.0)
    model_decay_gap_pct: float = Field(default=15.0)
    model_threshold_tighten_pct: float = Field(default=0.05)
    consecutive_win_position_boost: float = Field(default=1.2)
    consecutive_win_threshold: int = Field(default=2)
    consecutive_loss_position_cut: float = Field(default=0.5)
    consecutive_loss_cut_threshold: int = Field(default=2)
    daily_loss_position_cut_threshold_pct: float = Field(default=0.015)
    daily_loss_position_cut_factor: float = Field(default=0.5)
    drawdown_position_cut_threshold_pct: float = Field(default=0.08)
    drawdown_position_cut_factor: float = Field(default=0.3)
    reconciliation_alert_threshold_pct: float = Field(default=0.001)
    news_risk_cooldown_hours: int = Field(default=4)
    max_slippage_pct: float = Field(default=0.001)
    order_timeout_seconds: int = Field(default=30)
    limit_order_timeout_seconds: int = Field(default=300)
    limit_order_retry_count: int = Field(default=1)
    order_poll_interval_seconds: int = Field(default=2)
    execution_symbol_min_samples: int = Field(default=8)
    execution_symbol_accuracy_floor_pct: float = Field(default=45.0)
    execution_pool_target_size: int = Field(
        default=2,
        validation_alias=AliasChoices("EXECUTION_POOL_TARGET_SIZE"),
    )
    execution_pool_rebuild_interval_hours: int = Field(default=24)
    live_min_closed_trades: int = Field(default=50)
    live_min_prediction_eval_count: int = Field(default=100)
    live_min_xgboost_accuracy_pct: float = Field(default=55.0)
    live_min_fusion_accuracy_pct: float = Field(default=52.0)
    live_min_holdout_accuracy_pct: float = Field(default=55.0)
    live_min_total_realized_pnl: float = Field(default=0.0)
    live_max_drawdown_pct: float = Field(default=5.0)
    live_max_research_fallback_ratio_pct: float = Field(default=10.0)
    paper_exploration_grace_closed_trades: int = Field(default=3)
    paper_exploration_grace_threshold_tighten_pct: float = Field(default=0.02)


class TrainingSettings(BaseSettings):
    prediction_horizon_hours: int = Field(default=4)
    fast_alpha_training_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("FAST_ALPHA_TRAINING_ENABLED"),
    )
    fast_alpha_training_horizon_hours: int = Field(
        default=6,
        validation_alias=AliasChoices("FAST_ALPHA_TRAINING_HORIZON_HOURS"),
    )
    fast_alpha_label_min_abs_net_return_pct: float = Field(
        default=0.20,
        validation_alias=AliasChoices("FAST_ALPHA_LABEL_MIN_ABS_NET_RETURN_PCT"),
    )
    minimum_training_rows: int = Field(default=300)
    walkforward_window_days: int = Field(default=30)
    retrain_interval_days: int = Field(default=30)
    dataset_limit_1h: int = Field(
        default=2000,
        validation_alias=AliasChoices("TRAINING_DATASET_LIMIT_1H", "DATASET_LIMIT_1H"),
    )
    dataset_limit_4h: int = Field(
        default=1000,
        validation_alias=AliasChoices("TRAINING_DATASET_LIMIT_4H", "DATASET_LIMIT_4H"),
    )
    dataset_limit_1d: int = Field(
        default=400,
        validation_alias=AliasChoices("TRAINING_DATASET_LIMIT_1D", "DATASET_LIMIT_1D"),
    )


class SchedulerSettings(BaseSettings):
    analysis_cron_minutes: int = Field(default=240)
    training_cron_hours: int = Field(default=24)
    walkforward_cron_hours: int = Field(default=24)
    report_cron_hours: int = Field(default=24)
    health_cron_minutes: int = Field(default=60)
    guard_cron_hours: int = Field(default=24)
    ops_cron_minutes: int = Field(default=60)
    reconcile_cron_hours: int = Field(default=24)
    maintenance_cron_hours: int = Field(default=24)
    failure_cron_hours: int = Field(default=24)
    incident_cron_hours: int = Field(default=24)


class MaintenanceSettings(BaseSettings):
    retain_feature_days: int = Field(default=30)
    retain_prediction_days: int = Field(default=30)
    retain_account_days: int = Field(default=90)
    retain_execution_days: int = Field(default=90)
    retain_scheduler_days: int = Field(default=30)
    retain_report_days: int = Field(default=30)


class Settings(BaseSettings):
    app: AppSettings = Field(default_factory=AppSettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    news: NewsSettings = Field(default_factory=NewsSettings)
    onchain: OnchainSettings = Field(default_factory=OnchainSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    sentiment: SentimentSettings = Field(default_factory=SentimentSettings)
    ab_testing: ABTestingSettings = Field(default_factory=ABTestingSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    maintenance: MaintenanceSettings = Field(default_factory=MaintenanceSettings)

    def model_post_init(self, __context) -> None:
        if not self.app.low_resource_mode:
            return
        if self.model.xgboost_nthread <= 0:
            self.model.xgboost_nthread = 1
        self.model.xgboost_num_boost_round = min(
            self.model.xgboost_num_boost_round,
            120,
        )
        self.training.dataset_limit_1h = min(self.training.dataset_limit_1h, 1200)
        self.training.dataset_limit_4h = min(self.training.dataset_limit_4h, 720)
        self.training.dataset_limit_1d = min(self.training.dataset_limit_1d, 240)
        self.scheduler.walkforward_cron_hours = 0


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_project_path(path: str | Path, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(settings.app.project_root) / candidate
