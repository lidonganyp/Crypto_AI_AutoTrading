"""Dashboard page view-model builders."""
from __future__ import annotations


def build_overview_view_model(
    *,
    latest_account: dict | None,
    latest_reconciliation: dict | None,
    latest_ops: dict | None,
    latest_guard: dict | None,
    latest_health: dict | None,
    latest_incident: dict | None,
    latest_failure: dict | None,
    latest_cycle: dict | None,
    latest_scheduler: dict | None,
    latest_model_event: dict | None,
    model_degradation_status: dict | None,
    model_degradation_reason: dict | None,
    last_accuracy_guard_triggered: dict | None,
    last_accuracy_guard_reason: dict | None,
    perf_metrics: dict,
    daily_focus_metrics: dict,
    alpha_metrics: dict,
    consistency_metrics: dict,
    lifecycle_rows: list[dict],
    lifecycle_summary: str,
) -> dict:
    return {
        "lifecycle_rows": list(lifecycle_rows or []),
        "lifecycle_summary": lifecycle_summary,
        "primary_metrics": {
            "equity": latest_account.get("equity") if latest_account else None,
            "open_positions": latest_account.get("open_positions") if latest_account else 0,
            "drawdown_pct": latest_account.get("drawdown_pct") if latest_account else None,
            "circuit_breaker_active": bool(
                latest_account.get("circuit_breaker_active")
            ) if latest_account else False,
        },
        "secondary_metrics": {
            "reconciliation_status": (latest_reconciliation or {}).get("status"),
            "expanded_xgb_accuracy": perf_metrics.get(
                "Expanded XGBoost Direction Accuracy",
                perf_metrics.get("XGBoost Direction Accuracy"),
            ),
            "ops_report_ready": bool(latest_ops),
            "guard_report_ready": bool(latest_guard),
            "net_expectancy": perf_metrics.get("Recent Net Expectancy"),
            "net_profit_factor": perf_metrics.get("Recent Net Profit Factor"),
            "equity_return": perf_metrics.get("Equity Return"),
            "total_trade_cost": perf_metrics.get("Total Trade Cost"),
        },
        "daily_focus_metrics": dict(daily_focus_metrics or {}),
        "alpha_metrics": dict(alpha_metrics or {}),
        "consistency_metrics": dict(consistency_metrics or {}),
        "captions": {
            "latest_cycle": latest_cycle or {},
            "latest_scheduler": latest_scheduler or {},
            "latest_reports": {
                "health": bool(latest_health),
                "incident": bool(latest_incident),
                "ops": bool(latest_ops),
                "guard": bool(latest_guard),
                "failure": bool(latest_failure),
            },
            "model_degradation_status": model_degradation_status or {},
            "model_degradation_reason": model_degradation_reason or {},
            "last_accuracy_guard_triggered": last_accuracy_guard_triggered or {},
            "last_accuracy_guard_reason": last_accuracy_guard_reason or {},
            "latest_model_event": latest_model_event or {},
        },
    }


def build_ops_view_model(
    *,
    shadow_symbols: list[str],
    paper_exploration_symbols: list[str],
    shadow_trade_summary: dict,
    latest_prediction_eval: dict | None,
    paper_canary_summary: dict,
    candidate_rows: list[dict],
    model_observations: dict,
    funnel_rows: list[dict],
    latest_rollback: dict,
    market_data_route: dict | None,
    market_data_failover_stats: dict | None,
    fast_alpha_policy_summary: dict | None = None,
) -> dict:
    shadow_candidate_count = sum(
        1 for row in candidate_rows if str(row.get("status") or "") == "shadow"
    )
    live_candidate_count = sum(
        1 for row in candidate_rows if str(row.get("status") or "") == "live"
    )
    funnel_summary = {
        "shadow_observation": 0,
        "shadow_candidate": 0,
        "live_canary": 0,
        "promoted_observing": 0,
        "paper_canary_opened": 0,
        "promoted": 0,
        "accepted": 0,
        "rolled_back": 0,
    }
    for row in funnel_rows:
        stage = str(row.get("stage") or "")
        if stage in funnel_summary:
            funnel_summary[stage] += 1
    market_data_route = dict(market_data_route or {})
    failover_stats = dict(market_data_failover_stats or {})
    fast_alpha_policy_summary = dict(fast_alpha_policy_summary or {})
    fallback_count = 0
    primary_failures = 0
    secondary_failures = 0
    for payload in failover_stats.values():
        payload = payload if isinstance(payload, dict) else {}
        fallback_count += int(payload.get("fallback_used", 0) or 0)
        primary_failures += int(payload.get("primary_failures", 0) or 0)
        secondary_failures += int(payload.get("secondary_failures", 0) or 0)
    return {
        "shadow_symbols": list(shadow_symbols or []),
        "paper_exploration_symbols": list(paper_exploration_symbols or []),
        "candidate_rows": list(candidate_rows or []),
        "model_observation_count": len(model_observations or {}),
        "funnel_rows": list(funnel_rows or []),
        "funnel_summary": funnel_summary,
        "market_data_route": market_data_route,
        "market_data_failover_stats": failover_stats,
        "metrics": {
            "open_shadow_trade_count": int(shadow_trade_summary.get("open_count") or 0),
            "evaluated_shadow_trade_count": int(
                shadow_trade_summary.get("evaluated_count") or 0
            ),
            "latest_prediction_eval_at": (
                latest_prediction_eval.get("created_at")
                if latest_prediction_eval
                else None
            ),
            "paper_canary_open_count": int(paper_canary_summary.get("total_count") or 0),
            "soft_canary_open_count": int(paper_canary_summary.get("soft_count") or 0),
            "shadow_candidate_count": shadow_candidate_count,
            "live_candidate_count": live_candidate_count,
            "latest_rollback_symbol": (latest_rollback or {}).get("symbol"),
            "latest_market_data_provider": market_data_route.get("selected_provider"),
            "latest_market_data_operation": market_data_route.get("operation"),
            "market_data_fallback_active": bool(market_data_route.get("fallback_used")),
            "market_data_fallback_count": fallback_count,
            "market_data_primary_failures": primary_failures,
            "market_data_secondary_failures": secondary_failures,
            "market_data_updated_at": market_data_route.get("updated_at"),
            "fast_alpha_short_horizon_softened_open_count": int(
                fast_alpha_policy_summary.get("softened_open_count") or 0
            ),
            "fast_alpha_negative_expectancy_pause_count": int(
                fast_alpha_policy_summary.get("negative_pause_count") or 0
            ),
        },
    }


def build_runtime_settings_view_model(
    *,
    defaults: dict[str, float | list[float]],
    overrides: dict,
    locked_fields: list[str],
    learning_overrides: dict,
    learning_details: dict,
    override_conflicts: dict,
    effective: dict,
    override_status: dict | None,
) -> dict:
    return {
        "defaults": defaults,
        "overrides": overrides,
        "locked_fields": locked_fields,
        "learning_overrides": learning_overrides,
        "learning_details": learning_details,
        "override_conflicts": override_conflicts,
        "effective": effective,
        "override_status": override_status,
        "metrics": {
            "override_status": effective.get("effective_mode")
            or (override_status or {}).get("value")
            or "default",
            "learning_override_count": len(learning_overrides or {}),
            "manual_override_count": len(overrides or {}),
            "active_automation_rule_count": len(
                (learning_details or {}).get("blocked_setups", []) or []
            ),
        },
        "automation_summary": {
            "effective_runtime": effective if effective else defaults,
            "learning_reasons": (learning_details or {}).get("reasons", []),
            "learning_stats": (learning_details or {}).get("stats", {}),
            "blocked_setups": (learning_details or {}).get("blocked_setups", []),
            "override_conflicts": override_conflicts,
        },
        "current_values": {
            "xgboost_probability_threshold": overrides.get(
                "xgboost_probability_threshold",
                defaults["xgboost_probability_threshold"],
            ),
            "final_score_threshold": overrides.get(
                "final_score_threshold",
                defaults["final_score_threshold"],
            ),
            "min_liquidity_ratio": overrides.get(
                "min_liquidity_ratio",
                defaults["min_liquidity_ratio"],
            ),
            "sentiment_weight": overrides.get(
                "sentiment_weight",
                defaults["sentiment_weight"],
            ),
            "fixed_stop_loss_pct": overrides.get(
                "fixed_stop_loss_pct",
                defaults["fixed_stop_loss_pct"],
            ),
            "take_profit_levels": overrides.get(
                "take_profit_levels",
                defaults["take_profit_levels"],
            ),
        },
    }


def build_predictions_summary_view_model(
    *,
    shadow_symbols: list[str],
    paper_exploration_symbols: list[str],
    evaluation_summary: dict,
    shadow_trade_summary: dict,
    paper_canary_summary: dict,
) -> dict:
    return {
        "shadow_symbols": list(shadow_symbols or []),
        "paper_exploration_symbols": list(paper_exploration_symbols or []),
        "metrics": {
            "observation_pool_count": len(shadow_symbols or []),
            "execution_evaluation_count": int(
                evaluation_summary.get("execution_count") or 0
            ),
            "shadow_evaluation_count": int(
                evaluation_summary.get("shadow_count") or 0
            ),
            "open_shadow_trade_count": int(
                shadow_trade_summary.get("open_count") or 0
            ),
            "paper_canary_open_count": int(
                paper_canary_summary.get("total_count") or 0
            ),
            "soft_canary_open_count": int(
                paper_canary_summary.get("soft_count") or 0
            ),
            "exploration_slot_count": len(paper_exploration_symbols or []),
            "execution_accuracy": evaluation_summary.get("execution_accuracy"),
            "shadow_accuracy": evaluation_summary.get("shadow_accuracy"),
            "execution_expectancy_pct": evaluation_summary.get("execution_expectancy_pct"),
            "shadow_expectancy_pct": evaluation_summary.get("shadow_expectancy_pct"),
            "shadow_avg_pnl_pct": shadow_trade_summary.get("avg_pnl_pct"),
        },
    }


def build_watchlist_view_model(
    *,
    snapshot: dict,
    active_symbols: list[str],
    execution_symbols: list[str],
    model_ready_symbols: list[str],
    consistency_blocked_symbols: list[str],
    consistency_blocked_details: dict[str, list[str]],
    whitelist: list[str],
    blacklist: list[str],
) -> dict:
    payload = dict(snapshot or {})
    return {
        "snapshot": payload,
        "active_symbols": list(active_symbols or []),
        "raw_active_symbols": list(payload.get("raw_active_symbols") or []),
        "execution_symbols": list(execution_symbols or []),
        "model_ready_symbols": list(model_ready_symbols or []),
        "consistency_blocked_symbols": list(consistency_blocked_symbols or []),
        "consistency_blocked_details": dict(consistency_blocked_details or {}),
        "added_symbols": list(payload.get("added_symbols") or []),
        "removed_symbols": list(payload.get("removed_symbols") or []),
        "candidates": payload.get("candidates") or [],
        "refreshed_at": str(payload.get("refreshed_at") or ""),
        "refresh_reason": str(payload.get("refresh_reason") or ""),
        "whitelist": list(whitelist or []),
        "blacklist": list(blacklist or []),
    }
