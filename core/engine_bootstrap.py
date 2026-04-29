"""Bootstrap helpers for engine construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.failover_market_data import FailoverMarketDataCollector


@dataclass
class CoreServiceBundle:
    sentiment: object
    news: object
    macro: object
    onchain: object
    watchlist: object
    cross_validation: object
    feature_pipeline: object
    regime_detector: object
    research: object
    research_manager: object
    _predictor_factory: object
    _predictor_base_path: Path
    _challenger_predictor_base_path: Path
    trainer: object
    risk: object
    walkforward: object
    backtester: object
    decision_engine: object
    _base_runtime_settings: dict[str, float | list[float]]


@dataclass
class OperationsServiceBundle:
    health: object
    performance: object
    cycle_runtime: object
    shadow_runtime: object
    reconciler: object
    reflector: object
    strategy_evolver: object
    maintenance: object
    system_data: object
    failures: object
    ab_tests: object
    drift: object
    incidents: object
    guards: object
    ops: object
    validation: object
    notifier: object


def configure_engine_paths(
    *,
    settings,
    resolve_project_path_fn,
    logger_obj,
):
    db_path = resolve_project_path_fn(settings.app.db_path, settings)
    log_path = db_path.parent / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    logger_obj.add(
        str(log_path / "cryptoai_v3_{time:YYYY-MM-DD}.log"),
        rotation="1 day",
        retention="30 days",
        level=settings.app.log_level,
    )
    report_dir = db_path.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return db_path, report_dir


def build_core_services(
    *,
    settings,
    storage,
    market,
    resolve_project_path_fn,
    sentiment_collector_cls,
    news_service_cls,
    macro_service_cls,
    onchain_service_cls,
    dynamic_watchlist_service_cls,
    cross_validation_service_cls,
    feature_pipeline_cls,
    market_regime_detector_cls,
    research_llm_analyzer_cls,
    research_manager_cls,
    predictor_factory_cls,
    model_trainer_cls,
    risk_manager_cls,
    walkforward_backtester_cls,
    backtest_engine_cls,
    decision_engine_cls,
):
    feature_pipeline = feature_pipeline_cls()
    decision_engine = decision_engine_cls(
        xgboost_threshold=settings.model.xgboost_probability_threshold,
        final_score_threshold=settings.model.final_score_threshold,
        sentiment_weight=settings.strategy.sentiment_weight,
        min_liquidity_ratio=settings.strategy.min_liquidity_ratio,
        trend_reversal_probability=settings.strategy.trend_reversal_probability,
        sentiment_exit_threshold=settings.strategy.sentiment_exit_threshold,
        fixed_stop_loss_pct=settings.strategy.fixed_stop_loss_pct,
        take_profit_levels=settings.strategy.take_profit_levels,
        max_hold_hours=settings.strategy.max_hold_hours,
        extreme_fear_conservative_enabled=settings.strategy.extreme_fear_conservative_enabled,
        extreme_fear_conservative_xgboost_bonus_pct=settings.strategy.extreme_fear_conservative_xgboost_bonus_pct,
        extreme_fear_conservative_final_score_bonus=settings.strategy.extreme_fear_conservative_final_score_bonus,
        extreme_fear_conservative_liquidity_bonus_ratio=settings.strategy.extreme_fear_conservative_liquidity_bonus_ratio,
        extreme_fear_conservative_position_scale=settings.strategy.extreme_fear_conservative_position_scale,
    )
    return CoreServiceBundle(
        sentiment=sentiment_collector_cls(storage, settings.sentiment),
        news=news_service_cls(settings.news),
        macro=macro_service_cls(),
        onchain=onchain_service_cls(settings.onchain),
        watchlist=dynamic_watchlist_service_cls(storage, settings, market),
        cross_validation=cross_validation_service_cls(),
        feature_pipeline=feature_pipeline,
        regime_detector=market_regime_detector_cls(),
        research=research_llm_analyzer_cls(settings.llm),
        research_manager=research_manager_cls(settings, storage=storage),
        _predictor_factory=predictor_factory_cls,
        _predictor_base_path=resolve_project_path_fn(
            settings.model.xgboost_model_path,
            settings,
        ),
        _challenger_predictor_base_path=resolve_project_path_fn(
            settings.ab_testing.challenger_model_path,
            settings,
        ),
        trainer=model_trainer_cls(
            storage=storage,
            settings=settings,
            pipeline=feature_pipeline,
        ),
        risk=risk_manager_cls(settings.risk, settings.strategy),
        walkforward=walkforward_backtester_cls(storage, settings),
        backtester=backtest_engine_cls(storage, settings),
        decision_engine=decision_engine,
        _base_runtime_settings={
            "xgboost_probability_threshold": settings.model.xgboost_probability_threshold,
            "final_score_threshold": settings.model.final_score_threshold,
            "min_liquidity_ratio": settings.strategy.min_liquidity_ratio,
            "sentiment_weight": settings.strategy.sentiment_weight,
            "fixed_stop_loss_pct": settings.strategy.fixed_stop_loss_pct,
            "take_profit_levels": list(settings.strategy.take_profit_levels),
        },
    )


def build_market_and_live_exchange(
    *,
    settings,
    storage,
    okx_market_collector_cls,
    binance_market_collector_cls,
    okx_exchange_adapter_cls,
    binance_exchange_adapter_cls,
):
    provider = settings.exchange.provider.lower()
    if provider == "binance":
        primary_market = binance_market_collector_cls(
            storage,
            proxy=settings.exchange.proxy_url,
        )
        live_exchange = binance_exchange_adapter_cls(
            proxy_url=settings.exchange.proxy_url,
            api_key=settings.exchange.api_key.get_secret_value(),
            api_secret=settings.exchange.api_secret.get_secret_value(),
        )
        secondary_provider = "okx"
        secondary_factory = lambda: okx_market_collector_cls(
            storage,
            proxy=settings.exchange.proxy_url,
        )
    else:
        primary_market = okx_market_collector_cls(
            storage,
            proxy=settings.exchange.proxy_url,
        )
        live_exchange = okx_exchange_adapter_cls(
            proxy_url=settings.exchange.proxy_url,
            api_key=settings.exchange.api_key.get_secret_value(),
            api_secret=settings.exchange.api_secret.get_secret_value(),
            api_passphrase=settings.exchange.api_passphrase.get_secret_value(),
        )
        secondary_provider = "binance"
        secondary_factory = lambda: binance_market_collector_cls(
            storage,
            proxy=settings.exchange.proxy_url,
        )
    if bool(getattr(settings.exchange, "market_data_failover_enabled", True)):
        market = FailoverMarketDataCollector(
            storage,
            primary_provider=provider,
            primary_collector=primary_market,
            secondary_provider=secondary_provider,
            secondary_factory=secondary_factory,
        )
    else:
        market = primary_market
    return market, live_exchange


def build_executor(
    *,
    settings,
    storage,
    live_exchange,
    live_trader_cls,
    paper_trader_cls,
    slippage_guard_cls,
):
    if settings.app.runtime_mode == "live":
        return live_trader_cls(
            storage,
            exchange=live_exchange,
            enabled=settings.app.allow_live_orders,
            slippage_guard=slippage_guard_cls(settings.risk.max_slippage_pct),
            order_timeout_seconds=settings.risk.order_timeout_seconds,
            limit_order_timeout_seconds=settings.risk.limit_order_timeout_seconds,
            limit_order_retry_count=settings.risk.limit_order_retry_count,
            order_poll_interval_seconds=settings.risk.order_poll_interval_seconds,
        )
    return paper_trader_cls(storage)


def build_notifier(
    *,
    storage,
    settings,
    notifier_cls,
    console_channel_cls,
    file_channel_cls,
    webhook_channel_cls,
    critical_webhook_channel_cls,
    feishu_webhook_channel_cls,
    critical_feishu_webhook_channel_cls,
):
    notifier = notifier_cls(storage)
    notifier.add_channel(console_channel_cls())
    notifier.add_channel(file_channel_cls())
    if settings.notifications.webhook_url:
        notifier.add_channel(
            webhook_channel_cls(
                settings.notifications.webhook_url,
                settings.notifications.webhook_secret.get_secret_value(),
            )
        )
    if settings.notifications.critical_webhook_url:
        notifier.add_channel(
            critical_webhook_channel_cls(
                settings.notifications.critical_webhook_url,
                settings.notifications.critical_webhook_secret.get_secret_value(),
            )
        )
    if settings.notifications.feishu_webhook_url:
        notifier.add_channel(
            feishu_webhook_channel_cls(
                settings.notifications.feishu_webhook_url,
                settings.notifications.feishu_webhook_secret.get_secret_value(),
            )
        )
    if settings.notifications.feishu_critical_webhook_url:
        notifier.add_channel(
            critical_feishu_webhook_channel_cls(
                settings.notifications.feishu_critical_webhook_url,
                settings.notifications.feishu_critical_webhook_secret.get_secret_value(),
            )
        )
    return notifier


def build_operations_services(
    *,
    storage,
    settings,
    executor,
    build_notifier_fn,
    health_checker_cls,
    performance_reporter_cls,
    cycle_runtime_service_cls,
    shadow_runtime_service_cls,
    reconciler_cls,
    trade_reflector_cls,
    strategy_evolver_cls,
    maintenance_service_cls,
    system_data_service_cls,
    failure_reporter_cls,
    ab_test_reporter_cls,
    drift_reporter_cls,
    incident_reporter_cls,
    guard_reporter_cls,
    ops_overview_service_cls,
    validation_sprint_service_cls,
    notifier_cls,
    console_channel_cls,
    file_channel_cls,
    webhook_channel_cls,
    critical_webhook_channel_cls,
    feishu_webhook_channel_cls,
    critical_feishu_webhook_channel_cls,
):
    notifier = build_notifier_fn(
        storage=storage,
        settings=settings,
        notifier_cls=notifier_cls,
        console_channel_cls=console_channel_cls,
        file_channel_cls=file_channel_cls,
        webhook_channel_cls=webhook_channel_cls,
        critical_webhook_channel_cls=critical_webhook_channel_cls,
        feishu_webhook_channel_cls=feishu_webhook_channel_cls,
        critical_feishu_webhook_channel_cls=critical_feishu_webhook_channel_cls,
    )
    return OperationsServiceBundle(
        health=health_checker_cls(storage, settings),
        performance=performance_reporter_cls(storage, settings),
        cycle_runtime=cycle_runtime_service_cls(storage, settings),
        shadow_runtime=shadow_runtime_service_cls(storage, settings),
        reconciler=reconciler_cls(
            storage,
            exchange=getattr(executor, "exchange", None),
        ),
        reflector=trade_reflector_cls(storage),
        strategy_evolver=strategy_evolver_cls(storage, settings),
        maintenance=maintenance_service_cls(storage, settings.maintenance),
        system_data=system_data_service_cls(storage, settings),
        failures=failure_reporter_cls(storage),
        ab_tests=ab_test_reporter_cls(storage),
        drift=drift_reporter_cls(storage),
        incidents=incident_reporter_cls(storage),
        guards=guard_reporter_cls(storage),
        ops=ops_overview_service_cls(storage),
        validation=validation_sprint_service_cls(storage, settings),
        notifier=notifier,
    )
