"""Runtime service builders for engine construction."""
from __future__ import annotations


def build_report_runtime_service(*, engine, report_runtime_service_cls):
    return report_runtime_service_cls(
        engine.storage,
        engine.report_dir,
        current_language=engine._current_language,
        reflector=engine.reflector,
        health=engine.health,
        performance=engine.performance,
        maintenance=engine.maintenance,
        guard_runtime=engine.guard_runtime,
        guards=engine.guards,
        ab_tests=engine.ab_tests,
        drift=engine.drift,
        failures=engine.failures,
        incidents=engine.incidents,
        ops=engine.ops,
        pool_attribution=engine.pool_attribution,
        alpha_diagnostics=engine.alpha_diagnostics,
        validation=engine.validation,
        daily_focus=engine.daily_focus,
        backtest_live_consistency=engine.backtest_live_consistency,
        evaluate_live_readiness=engine.evaluate_live_readiness,
        get_execution_symbols=engine.get_execution_symbols,
    )


def build_model_lifecycle_runtime_service(*, engine, model_lifecycle_runtime_service_cls):
    return model_lifecycle_runtime_service_cls(
        engine.storage,
        engine.settings,
        trainer=engine.trainer,
        walkforward=engine.walkforward,
        report_runtime=engine.report_runtime,
        notifier=engine.notifier,
        predictor_base_path=engine._predictor_base_path,
        challenger_predictor_base_path=engine._challenger_predictor_base_path,
        get_model_promotion_candidates=engine._model_promotion_candidates,
        set_model_promotion_candidates=engine._set_model_promotion_candidates,
        get_model_promotion_observations=engine._model_promotion_observations,
        set_model_promotion_observations=engine._set_model_promotion_observations,
        resolved_model_id=engine._resolved_model_id,
        model_file_signature=engine._model_file_signature,
        training_objective_baseline=engine._training_objective_baseline,
        promotion_adaptive_requirements=engine._promotion_adaptive_requirements,
        promotion_live_allocation_pct=engine._promotion_live_allocation_pct,
        upsert_model_registry_entry=engine._upsert_model_registry_entry,
        candidate_active_model_id=engine._candidate_active_model_id,
        candidate_challenger_model_id=engine._candidate_challenger_model_id,
        observation_active_model_id=engine._observation_active_model_id,
        observation_backup_model_id=engine._observation_backup_model_id,
        clear_symbol_models=engine.clear_symbol_models,
        read_model_metadata=engine._read_model_metadata,
        record_model_scorecard=engine._record_model_scorecard,
        build_model_scorecard=engine._build_model_scorecard,
        build_model_live_pnl_summary=engine._build_model_live_pnl_summary,
        objective_score_from_metrics=engine._objective_score_from_metrics,
        objective_score_quality=engine._objective_score_quality,
        accuracy_safety_floor=engine._accuracy_safety_floor,
    )


def build_runtime_settings_service(*, engine, runtime_settings_service_cls):
    return runtime_settings_service_cls(
        engine.storage,
        engine.settings,
        decision_engine=engine.decision_engine,
        strategy_evolver=engine.strategy_evolver,
        base_runtime_settings=engine._base_runtime_settings,
        runtime_override_state_key=engine.RUNTIME_OVERRIDE_STATE_KEY,
        runtime_locked_fields_state_key=engine.RUNTIME_LOCKED_FIELDS_STATE_KEY,
        runtime_learning_override_state_key=engine.RUNTIME_LEARNING_OVERRIDE_STATE_KEY,
        runtime_override_conflict_state_key=engine.RUNTIME_OVERRIDE_CONFLICT_STATE_KEY,
        runtime_learning_details_state_key=engine.RUNTIME_LEARNING_DETAILS_STATE_KEY,
        runtime_effective_state_key=engine.RUNTIME_EFFECTIVE_STATE_KEY,
    )


def build_runtime_coordination_service(*, engine, runtime_coordination_service_cls):
    return runtime_coordination_service_cls(
        observe_promoted_models=lambda now: engine._observe_promoted_models(now),
        refresh_learning_runtime_overrides=lambda now: engine._refresh_learning_runtime_overrides(now),
        apply_runtime_overrides=lambda: engine._apply_runtime_overrides(),
        persist_runtime_settings_effective=lambda: engine._persist_runtime_settings_effective(),
        refresh_runtime_learning_feedback=lambda now, reason: engine._refresh_runtime_learning_feedback(
            now,
            reason=reason,
        ),
        rebuild_execution_symbols=lambda **kwargs: engine.rebuild_execution_symbols(
            **kwargs
        ),
    )


def build_snapshot_runtime_service(*, engine, snapshot_runtime_service_cls):
    return snapshot_runtime_service_cls(
        engine.storage,
        engine.settings,
        market=engine.market,
        executor=engine.executor,
        sentiment=engine.sentiment,
        news=engine.news,
        macro=engine.macro,
        onchain=engine.onchain,
        regime_detector=engine.regime_detector,
        research=engine.research,
        feature_pipeline=engine.feature_pipeline,
        cross_validation=engine.cross_validation,
        research_manager=engine.research_manager,
        decision_engine=engine.decision_engine,
        notifier=engine.notifier,
        predictor_for_symbol=lambda symbol: engine._predictor_for_symbol(symbol),
        handle_model_unavailable=lambda **kwargs: engine._handle_model_unavailable(**kwargs),
        learning_details_state_key=engine.RUNTIME_LEARNING_DETAILS_STATE_KEY,
        register_market_data_failure=lambda reason: engine._register_market_data_failure(reason),
        reset_market_data_failures=lambda: setattr(
            engine,
            "_consecutive_market_data_failures",
            0,
        ),
        extend_cooldown_until=engine._extend_cooldown_until,
    )


def build_position_runtime_service(*, engine, position_runtime_service_cls):
    return position_runtime_service_cls(
        engine.storage,
        engine.settings,
        market=engine.market,
        feature_pipeline=engine.feature_pipeline,
        decision_engine=engine.decision_engine,
        executor=engine.executor,
        notifier=engine.notifier,
        prepare_position_review=lambda symbol, now: engine._prepare_position_review(symbol, now),
        predictor_for_symbol=lambda symbol: engine._predictor_for_symbol(symbol),
        fallback_research=lambda symbol: engine.research._fallback(symbol, None),
        record_trade_result=lambda pnl: engine._record_trade_result(pnl),
        reflect_closed_trade_result=lambda symbol, result: engine._reflect_closed_trade_result(symbol, result),
        handle_trade_close_feedback=lambda symbol, result: engine._handle_trade_close_feedback(symbol, result),
        exit_policy_getter=lambda symbol: engine._active_model_exit_policy(symbol),
        position_review_state_key=engine.POSITION_REVIEW_STATE_KEY,
    )


def build_execution_pool_runtime_service(*, engine, execution_pool_runtime_service_cls):
    return execution_pool_runtime_service_cls(
        engine.storage,
        engine.settings,
        performance_getter=lambda: engine.performance,
        shadow_feedback_getter=lambda: engine.shadow_runtime,
        consistency_getter=lambda: engine.backtest_live_consistency,
        watchlist_getter=lambda: engine.watchlist,
        market_getter=lambda: engine.market,
        trainer_getter=lambda: engine.trainer,
        notifier=engine.notifier,
        current_language=engine._current_language,
        handle_training_summary=lambda **kwargs: engine._process_training_summary(
            **kwargs
        ),
        clear_symbol_models=engine.clear_symbol_models,
        clear_broken_model_symbol=engine._clear_broken_model_symbol,
        runtime_model_path_for_symbol=engine._runtime_model_path_for_symbol,
        broken_model_symbols_state_key=engine.BROKEN_MODEL_SYMBOLS_STATE_KEY,
        execution_symbols_state_key=engine.EXECUTION_SYMBOLS_STATE_KEY,
        execution_pool_last_rebuild_at_state_key=engine.EXECUTION_POOL_LAST_REBUILD_AT_STATE_KEY,
        parse_iso_datetime=engine._parse_iso_datetime,
        rebuild_execution_symbols_callback=lambda **kwargs: engine.rebuild_execution_symbols(**kwargs),
    )


def build_analysis_runtime_service(*, engine, analysis_runtime_service_cls):
    return analysis_runtime_service_cls(
        engine.storage,
        engine.settings,
        decision_engine=engine.decision_engine,
        risk=engine.risk,
        executor=engine.executor,
        notifier=engine.notifier,
        prepare_symbol_snapshot=lambda symbol, now, include_blocked=True: engine._prepare_symbol_snapshot(
            symbol,
            now,
            include_blocked=include_blocked,
        ),
        detect_abnormal_move=lambda symbol, now: engine._detect_abnormal_move(symbol, now),
        evaluate_ab_test=lambda **kwargs: engine._evaluate_ab_test(**kwargs),
        persist_analysis=lambda *args, **kwargs: engine._persist_analysis(*args, **kwargs),
        record_shadow_trade_if_blocked=lambda **kwargs: engine._record_shadow_trade_if_blocked(**kwargs),
        compose_trade_rationale=engine._compose_trade_rationale,
        get_positions=engine.storage.get_positions,
        account_state=lambda now, positions: engine._account_state(now, positions),
        get_circuit_breaker_reason=lambda: engine._circuit_breaker_reason,
        performance_getter=lambda: engine.performance,
        position_value_adjuster=lambda **kwargs: engine._adjust_position_value_for_model_evidence(
            **kwargs
        ),
    )


def build_preflight_runtime_service(*, engine, preflight_runtime_service_cls):
    return preflight_runtime_service_cls(
        engine.storage,
        engine.settings,
        cycle_runtime=engine.cycle_runtime,
        notifier=engine.notifier,
        reconciler=engine.reconciler,
        runtime_coordination=engine.runtime_coordination_runtime,
        observe_promoted_models=lambda now: engine._observe_promoted_models(now),
        refresh_learning_runtime_overrides=lambda now: engine._refresh_learning_runtime_overrides(now),
        apply_runtime_overrides=lambda: engine._apply_runtime_overrides(),
        get_active_symbols=lambda force_refresh=False, now=None: engine.get_active_symbols(
            force_refresh=force_refresh,
            now=now,
        ),
        get_shadow_observation_symbols=lambda force_refresh=False, now=None: engine.get_shadow_observation_symbols(
            force_refresh=force_refresh,
            now=now,
        ),
        maybe_rebuild_execution_pool=lambda now, active_symbols: engine._maybe_rebuild_execution_pool(
            now,
            active_symbols,
        ),
        check_market_latency=lambda now, symbols=None: engine._check_market_latency(now, symbols),
        get_positions=engine.storage.get_positions,
        account_state=lambda now, positions: engine._account_state(now, positions),
        apply_model_degradation=lambda now: engine._apply_model_degradation(now),
        persist_runtime_settings_effective=lambda: engine._persist_runtime_settings_effective(),
        enforce_accuracy_guard=lambda now: engine._enforce_accuracy_guard(now),
        manual_recovery_blocked=lambda: engine._manual_recovery_blocked(),
        trigger_manual_recovery=lambda reason, details: engine._trigger_manual_recovery(reason, details),
        requires_manual_recovery=lambda reason: engine._requires_manual_recovery(reason),
        get_circuit_breaker_active=lambda: engine._circuit_breaker_active,
        get_circuit_breaker_reason=lambda: engine._circuit_breaker_reason,
        get_cooldown_until=lambda: engine._cooldown_until,
        set_circuit_breaker_state=lambda active, reason: (
            setattr(engine, "_circuit_breaker_active", active),
            setattr(engine, "_circuit_breaker_reason", reason),
        ),
    )


def build_guard_runtime_service(*, engine, guard_runtime_service_cls):
    return guard_runtime_service_cls(
        engine.storage,
        engine.settings,
        market_getter=lambda: engine.market,
        risk=engine.risk,
        performance_getter=lambda: engine.performance,
        health_getter=lambda: engine.health,
        notifier=engine.notifier,
        executor_getter=lambda: engine.executor,
        train_models_if_due=lambda now, force=False, reason="manual": engine._train_models_if_due(
            now,
            force,
            reason,
        ),
        get_active_symbols=lambda force_refresh=False, now=None: engine.get_active_symbols(
            force_refresh=force_refresh,
            now=now,
        ),
        storage_symbol_variants=engine._storage_symbol_variants,
        get_peak_equity=lambda: engine._peak_equity,
        set_peak_equity=lambda value: setattr(engine, "_peak_equity", value),
        get_cooldown_until=lambda: engine._cooldown_until,
        set_cooldown_until=lambda value: setattr(engine, "_cooldown_until", value),
        get_circuit_breaker_reason=lambda: engine._circuit_breaker_reason,
        set_circuit_breaker_reason=lambda value: setattr(
            engine,
            "_circuit_breaker_reason",
            value or "",
        ),
        set_circuit_breaker_active=lambda value: setattr(
            engine,
            "_circuit_breaker_active",
            bool(value),
        ),
        get_model_degradation_status=lambda: engine._model_degradation_status,
        set_model_degradation_status=lambda value: setattr(
            engine,
            "_model_degradation_status",
            value,
        ),
        get_model_degradation_reason=lambda: engine._model_degradation_reason,
        set_model_degradation_reason=lambda value: setattr(
            engine,
            "_model_degradation_reason",
            value or "",
        ),
        set_model_trading_disabled=lambda value: setattr(
            engine,
            "_model_trading_disabled",
            bool(value),
        ),
        nextgen_live_guard_callback=lambda trigger, reason, details: engine.run_nextgen_autonomy_live(
            requested_live=False,
            trigger=trigger,
            trigger_reason=reason,
            trigger_details=details,
        ),
    )
