"""Standalone dashboard page renderers with injected helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from core.dashboard_view_models import (
    build_ops_view_model,
    build_overview_view_model,
    build_predictions_summary_view_model,
    build_runtime_settings_view_model,
    build_watchlist_view_model,
)
from monitor.nextgen_live_summary import build_recent_nextgen_live_queue_runs


def _ctx_value(page_ctx, name: str, explicit):
    if explicit is not None:
        return explicit
    return getattr(page_ctx, name)


def _model_path_label(path_str: str) -> str:
    text = str(path_str or "").strip()
    return Path(text).name if text else "N/A"


def _coerce_selectbox_value(options, current, *, fallback=None):
    items = list(options or [])
    if not items:
        return fallback
    if current in items:
        return current
    if fallback in items:
        return fallback
    return items[0]


def _to_float(value, digits: int = 4):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _metric_value(metrics: dict, primary_key: str, fallback_key: str | None = None) -> str:
    value = metrics.get(primary_key)
    if value in (None, "") and fallback_key:
        value = metrics.get(fallback_key)
    return value if value not in (None, "") else "N/A"


def _json_list_text(raw, load_json) -> str:
    payload = load_json(raw, []) or []
    if isinstance(payload, list):
        return ", ".join(str(item) for item in payload if str(item).strip())
    if isinstance(payload, dict):
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return str(payload or "")


def _json_object_text(raw, load_json) -> str:
    payload = load_json(raw, {}) or {}
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return str(payload or "")


def _build_nextgen_selected_queue_context(queue_row: dict | None) -> dict:
    payload = dict(queue_row or {})
    return {
        "autonomy_cycle_id": int(payload.get("autonomy_cycle_id") or 0),
        "created_at": str(payload.get("created_at") or ""),
        "trigger": str(payload.get("trigger") or ""),
        "reason": str(payload.get("reason") or ""),
        "latest_issue_event_type": str(payload.get("latest_issue_event_type") or ""),
        "latest_issue_reason": str(payload.get("latest_issue_reason") or ""),
    }


def _limit_nextgen_focus_rows(
    frame: pd.DataFrame,
    *,
    limit: int = 5,
) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame, 0
    max_rows = max(1, int(limit))
    hidden = max(0, len(frame.index) - max_rows)
    return frame.head(max_rows).copy(), hidden


def _filter_nextgen_focus_view(
    *,
    mode: str,
    execution_df: pd.DataFrame,
    repair_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = str(mode or "").strip().lower()
    if normalized == "execution":
        return execution_df.copy(), pd.DataFrame()
    if normalized == "repair":
        return pd.DataFrame(), repair_df.copy()
    return execution_df.copy(), repair_df.copy()


def _filter_nextgen_cycle_detail_tables(
    *,
    query: str,
    execution_df: pd.DataFrame,
    repair_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    text = str(query or "").strip().lower()
    if not text:
        return execution_df.copy(), repair_df.copy()
    execution_columns = (
        "strategy_id",
        "action",
        "from_stage",
        "target_stage",
        "reasons",
        "queue_action",
    )
    repair_columns = (
        "strategy_id",
        "action",
        "validation_stage",
        "candidate_strategy_id",
        "candidate_family",
        "runtime_overrides",
        "reasons",
        "queue_action",
    )
    return (
        _filter_frame_by_query(execution_df, text, execution_columns),
        _filter_frame_by_query(repair_df, text, repair_columns),
    )


def _filter_nextgen_queue_rows(
    *,
    query: str,
    rows: list[dict],
) -> list[dict]:
    text = str(query or "").strip().lower()
    if not text:
        return [dict(item) for item in list(rows or [])]
    filtered: list[dict] = []
    for row in list(rows or []):
        payload = dict(row or {})
        haystack = " ".join(
            str(payload.get(key) or "")
            for key in (
                "autonomy_cycle_id",
                "status",
                "trigger",
                "reason",
                "latest_issue_event_type",
                "latest_issue_reason",
            )
        ).lower()
        if text in haystack:
            filtered.append(payload)
    return filtered


def _default_nextgen_focus_mode(
    *,
    execution_df: pd.DataFrame,
    repair_df: pd.DataFrame,
) -> str:
    if not repair_df.empty:
        return "repair"
    if not execution_df.empty:
        return "execution"
    return "all"


def _filter_frame_by_query(
    frame: pd.DataFrame,
    query: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = pd.Series(False, index=frame.index)
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].fillna("").astype(str).str.lower()
        mask = mask | values.str.contains(query, regex=False)
    return frame[mask].copy()


def _short_horizon_overview_notice(
    alpha_metrics: dict,
    *,
    current_language: str = "en",
) -> dict[str, str] | None:
    metrics = dict(alpha_metrics or {})
    status = str(metrics.get("Short-Horizon Status") or "").strip().lower()
    softened_status = str(
        metrics.get("Short-Horizon Softened Status") or ""
    ).strip().lower()
    if (not status or status == "n/a") and (
        not softened_status or softened_status == "n/a"
    ):
        return None
    samples = str(metrics.get("Short-Horizon Recent Samples") or "N/A")
    expectancy = str(metrics.get("Short-Horizon Recent Expectancy") or "N/A")
    profit_factor = str(metrics.get("Short-Horizon Recent Profit Factor") or "N/A")
    softened_expectancy = str(
        metrics.get("Short-Horizon Softened Expectancy") or "N/A"
    )
    softened_profit_factor = str(
        metrics.get("Short-Horizon Softened Profit Factor") or "N/A"
    )
    zh = str(current_language or "").strip().lower() == "zh"
    if softened_status == "disabled_negative_edge":
        return {
            "level": "error",
            "message": (
                f"Short-Horizon 放行子集已转负，必须保持放行收紧。总体状态={status or 'n/a'} | 样本={samples} | 放行后净期望={softened_expectancy} | 放行后净盈亏比={softened_profit_factor}"
                if zh
                else f"Short-horizon softened entries are underperforming. Keep softening disabled. overall={status or 'n/a'} | samples={samples} | softened_expectancy={softened_expectancy} | softened_profit_factor={softened_profit_factor}"
            ),
        }
    if status == "positive_edge":
        return {
            "level": "success",
            "message": (
                f"Short-Horizon 进入正边际扩张区间，可放宽提频。样本={samples} | 净期望={expectancy} | 净盈亏比={profit_factor}"
                if zh
                else f"Short-horizon is in positive-edge expansion mode. Frequency can expand. samples={samples} | expectancy={expectancy} | profit_factor={profit_factor}"
            ),
        }
    if status == "warming_up":
        return {
            "level": "info",
            "message": (
                f"Short-Horizon 仍在预热，先观察新增样本，不要手动继续放宽。样本={samples} | 净期望={expectancy} | 净盈亏比={profit_factor}"
                if zh
                else f"Short-horizon is still warming up. Observe new samples before loosening further. samples={samples} | expectancy={expectancy} | profit_factor={profit_factor}"
            ),
        }
    if status == "negative_edge_pause":
        return {
            "level": "error",
            "message": (
                f"Short-Horizon 已进入负边际暂停状态，应优先刹车而不是提频。样本={samples} | 净期望={expectancy} | 净盈亏比={profit_factor}"
                if zh
                else f"Short-horizon is paused on negative edge. De-risk instead of increasing frequency. samples={samples} | expectancy={expectancy} | profit_factor={profit_factor}"
            ),
        }
    if status == "neutral":
        return {
            "level": "warning",
            "message": (
                f"Short-Horizon 处于中性混合区间，暂不建议继续扩频。样本={samples} | 净期望={expectancy} | 净盈亏比={profit_factor}"
                if zh
                else f"Short-horizon is in a neutral mixed regime. Avoid pushing frequency higher yet. samples={samples} | expectancy={expectancy} | profit_factor={profit_factor}"
            ),
        }
    return {
        "level": "info",
        "message": (
            f"Short-Horizon 状态={status}。样本={samples} | 净期望={expectancy} | 净盈亏比={profit_factor}"
            if zh
            else f"Short-horizon status={status}. samples={samples} | expectancy={expectancy} | profit_factor={profit_factor}"
        ),
    }


def _build_nextgen_cycle_focus_view(
    *,
    cycle_notes: dict | None,
    execution_df: pd.DataFrame,
    repair_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    notes = dict(cycle_notes or {})
    runtime_ids = [
        str(item).strip()
        for item in list(notes.get("repair_queue_runtime_ids") or [])
        if str(item).strip()
    ]
    actions = [
        str(item).strip()
        for item in list(notes.get("repair_queue_actions") or [])
        if str(item).strip()
    ]
    priorities = [
        int(item)
        for item in list(notes.get("repair_queue_priorities") or [])
        if str(item).strip()
    ]
    queue_rows: list[dict] = []
    for index, runtime_id in enumerate(runtime_ids):
        queue_rows.append(
            {
                "strategy_id": runtime_id,
                "queue_order": index,
                "queue_action": actions[index] if index < len(actions) else "",
                "queue_priority": priorities[index] if index < len(priorities) else 0,
            }
        )
    queue_df = pd.DataFrame(queue_rows)
    focused_execution_df = pd.DataFrame()
    if runtime_ids and not execution_df.empty:
        focused_execution_df = execution_df[
            execution_df["strategy_id"].isin(runtime_ids)
        ].copy()
        if not queue_df.empty:
            focused_execution_df = focused_execution_df.merge(
                queue_df,
                on="strategy_id",
                how="left",
            )
            focused_execution_df = focused_execution_df.sort_values(
                by=["queue_order", "created_at"],
                ascending=[True, False],
            )
            focused_execution_df = focused_execution_df[
                [
                    "queue_priority",
                    "queue_action",
                    "strategy_id",
                    "action",
                    "from_stage",
                    "target_stage",
                    "capital_multiplier",
                    "reasons",
                    "created_at",
                ]
            ]
    focused_repair_df = pd.DataFrame()
    if runtime_ids and not repair_df.empty:
        focused_repair_df = repair_df[
            repair_df["strategy_id"].isin(runtime_ids)
        ].copy()
        if not queue_df.empty:
            focused_repair_df = focused_repair_df.merge(
                queue_df,
                on="strategy_id",
                how="left",
            )
            focused_repair_df = focused_repair_df.sort_values(
                by=["queue_order", "priority", "created_at"],
                ascending=[True, False, False],
            )
            focused_repair_df = focused_repair_df[
                [
                    "queue_priority",
                    "queue_action",
                    "strategy_id",
                    "action",
                    "priority",
                    "validation_stage",
                    "capital_multiplier",
                    "candidate_strategy_id",
                    "candidate_family",
                    "runtime_overrides",
                    "reasons",
                    "created_at",
                ]
            ]
    return (
        {
            "runtime_ids": runtime_ids,
            "actions_text": ", ".join(actions),
            "priorities_text": ", ".join(str(item) for item in priorities),
        },
        focused_execution_df,
        focused_repair_df,
    )


def _build_nextgen_cycle_detail_view(
    *,
    cycle_row: dict | None,
    execution_rows: pd.DataFrame,
    repair_rows: pd.DataFrame,
    load_json,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    cycle = dict(cycle_row or {})
    cycle_notes = load_json(cycle.get("notes_json"), {}) or {}
    cycle_meta = {
        "id": int(cycle.get("id") or 0),
        "created_at": str(cycle.get("created_at") or ""),
        "strategy_count": int(cycle.get("strategy_count") or 0),
        "execution_count": int(cycle.get("execution_count") or 0),
        "repair_count": int(cycle.get("repair_count") or 0),
        "quarantine_count": int(cycle.get("quarantine_count") or 0),
        "retire_count": int(cycle.get("retire_count") or 0),
    }
    execution_df = pd.DataFrame()
    if not execution_rows.empty:
        execution_df = execution_rows.copy()
        execution_df["reasons"] = execution_df["reasons_json"].apply(
            lambda value: _json_list_text(value, load_json)
        )
        execution_df = execution_df[
            [
                "strategy_id",
                "action",
                "from_stage",
                "target_stage",
                "capital_multiplier",
                "reasons",
                "created_at",
            ]
        ]
    repair_df = pd.DataFrame()
    if not repair_rows.empty:
        repair_df = repair_rows.copy()
        repair_df["runtime_overrides"] = repair_df["runtime_overrides_json"].apply(
            lambda value: _json_object_text(value, load_json)
        )
        repair_df["reasons"] = repair_df["reasons_json"].apply(
            lambda value: _json_list_text(value, load_json)
        )
        repair_df = repair_df[
            [
                "strategy_id",
                "action",
                "priority",
                "validation_stage",
                "capital_multiplier",
                "candidate_strategy_id",
                "candidate_family",
                "runtime_overrides",
                "reasons",
                "created_at",
            ]
        ]
    return cycle_meta, cycle_notes, execution_df, repair_df


def _latest_training_by_symbol(latest_training_rows: pd.DataFrame, load_json) -> dict[str, dict]:
    latest_training: dict[str, dict] = {}
    if latest_training_rows.empty:
        return latest_training
    for _, row in latest_training_rows.iterrows():
        symbol = str(row["symbol"])
        if symbol in latest_training:
            continue
        latest_training[symbol] = load_json(row.get("metadata_json"), {}) or {}
    return latest_training


def _build_model_lifecycle_rows(
    *,
    latest_training_rows: pd.DataFrame,
    model_observations,
    promotion_candidates,
    load_json,
) -> list[dict]:
    latest_training = _latest_training_by_symbol(latest_training_rows, load_json)
    observation_map = model_observations if isinstance(model_observations, dict) else {}
    candidate_map = promotion_candidates if isinstance(promotion_candidates, dict) else {}
    ordered_symbols = (
        list(candidate_map.keys())
        + [symbol for symbol in observation_map.keys() if symbol not in candidate_map]
        + [
            symbol
            for symbol in latest_training.keys()
            if symbol not in candidate_map and symbol not in observation_map
        ]
    )
    lifecycle_rows: list[dict] = []
    for symbol in ordered_symbols:
        candidate = candidate_map.get(symbol, {}) or {}
        observation = observation_map.get(symbol, {}) or {}
        metadata = dict(latest_training.get(symbol, {}) or {})
        candidate_training_metadata = candidate.get("training_metadata")
        if isinstance(candidate_training_metadata, dict):
            metadata.update(candidate_training_metadata)
        training_metadata = observation.get("training_metadata")
        if isinstance(training_metadata, dict):
            metadata.update(training_metadata)
        lifecycle_rows.append(
            {
                "symbol": symbol,
                "active_model_path": (
                    candidate.get("active_model_path")
                    or observation.get("active_model_path")
                    or metadata.get("active_model_path")
                    or metadata.get("model_path")
                    or ""
                ),
                "challenger_model_path": (
                    candidate.get("challenger_model_path")
                    or observation.get("challenger_model_path")
                    or metadata.get("challenger_model_path")
                    or ""
                ),
                "active_model": _model_path_label(
                    candidate.get("active_model_path")
                    or observation.get("active_model_path")
                    or metadata.get("active_model_path")
                    or metadata.get("model_path")
                ),
                "challenger_model": _model_path_label(
                    candidate.get("challenger_model_path")
                    or observation.get("challenger_model_path")
                    or metadata.get("challenger_model_path")
                ),
                "promotion_status": metadata.get("promotion_status", ""),
                "promotion_reason": metadata.get("promotion_reason", ""),
                "observation_status": (
                    candidate.get("status")
                    or observation.get("status")
                    or "idle"
                ),
                "promoted_at": observation.get("promoted_at", ""),
                "holdout_accuracy": metadata.get("holdout_accuracy"),
                "candidate_holdout_accuracy": metadata.get("candidate_holdout_accuracy"),
                "wf_return_pct": (
                    (metadata.get("candidate_walkforward_summary") or {}).get("total_return_pct")
                    if isinstance(metadata.get("candidate_walkforward_summary"), dict)
                    else None
                ),
            }
        )
    return lifecycle_rows


def _build_promotion_candidate_rows(promotion_candidates) -> list[dict]:
    candidate_map = promotion_candidates if isinstance(promotion_candidates, dict) else {}
    rows: list[dict] = []
    for symbol, payload in candidate_map.items():
        payload = payload if isinstance(payload, dict) else {}
        rows.append(
            {
                "symbol": symbol,
                "status": payload.get("status", "shadow"),
                "registered_at": payload.get("registered_at", ""),
                "live_started_at": payload.get("live_started_at", ""),
                "min_shadow_evaluations": payload.get("min_shadow_evaluations", ""),
                "min_live_evaluations": payload.get("min_live_evaluations", ""),
                "max_shadow_age_hours": payload.get("max_shadow_age_hours", ""),
                "max_live_age_hours": payload.get("max_live_age_hours", ""),
                "live_allocation_pct": _to_float(payload.get("live_allocation_pct"), 4),
                "active_model": _model_path_label(payload.get("active_model_path")),
                "challenger_model": _model_path_label(payload.get("challenger_model_path")),
                "shadow_eval_count": int(payload.get("shadow_eval_count", 0) or 0),
                "shadow_accuracy": _to_float(payload.get("shadow_accuracy"), 4),
                "shadow_expectancy_pct": _to_float(
                    payload.get("shadow_expectancy_pct"),
                    4,
                ),
                "shadow_profit_factor": _to_float(
                    payload.get("shadow_profit_factor"),
                    4,
                ),
                "shadow_max_drawdown_pct": _to_float(
                    payload.get("shadow_max_drawdown_pct"),
                    4,
                ),
                "shadow_avg_trade_return_pct": _to_float(
                    payload.get("shadow_avg_trade_return_pct"),
                    4,
                ),
                "shadow_champion_accuracy": _to_float(
                    payload.get("shadow_champion_accuracy"),
                    4,
                ),
            }
        )
    return rows


def _build_promotion_observation_rows(model_observations) -> list[dict]:
    observation_map = model_observations if isinstance(model_observations, dict) else {}
    rows: list[dict] = []
    for symbol, payload in observation_map.items():
        payload = payload if isinstance(payload, dict) else {}
        recent_wf = payload.get("recent_walkforward_baseline_summary", {}) or {}
        rows.append(
            {
                "symbol": symbol,
                "status": payload.get("status", "observing"),
                "promoted_at": payload.get("promoted_at", ""),
                "min_evaluations": payload.get("min_evaluations", ""),
                "max_observation_age_hours": payload.get(
                    "max_observation_age_hours",
                    "",
                ),
                "active_model": _model_path_label(payload.get("active_model_path")),
                "backup_model": _model_path_label(payload.get("backup_model_path")),
                "baseline_holdout_accuracy": _to_float(
                    payload.get("baseline_holdout_accuracy"),
                    4,
                ),
                "baseline_expectancy_pct": _to_float(
                    payload.get("baseline_expectancy_pct"),
                    4,
                ),
                "baseline_profit_factor": _to_float(
                    payload.get("baseline_profit_factor"),
                    4,
                ),
                "baseline_max_drawdown_pct": _to_float(
                    payload.get("baseline_max_drawdown_pct"),
                    4,
                ),
                "recent_wf_history_count": int(recent_wf.get("history_count", 0) or 0),
                "recent_wf_avg_expectancy_pct": _to_float(
                    recent_wf.get("avg_expectancy_pct"),
                    4,
                ),
                "recent_wf_avg_profit_factor": _to_float(
                    recent_wf.get("avg_profit_factor"),
                    4,
                ),
                "recent_wf_avg_max_drawdown_pct": _to_float(
                    recent_wf.get("avg_max_drawdown_pct"),
                    4,
                ),
            }
        )
    return rows


def _event_reason(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("reason") or payload.get("promotion_reason") or "")


FOCUS_EXIT_REASONS = (
    "research_exit",
    "research_de_risk",
    "evidence_de_risk",
)


def _reason_tokens(reason_text: str) -> list[str]:
    tokens = [part.strip() for part in str(reason_text or "").split(",")]
    return [token for token in tokens if token]


def _focus_setup_label(metadata: dict) -> str:
    payload = metadata if isinstance(metadata, dict) else {}
    setup_profile = payload.get("setup_profile", {}) or {}
    if not isinstance(setup_profile, dict):
        setup_profile = {}
    regime = str(setup_profile.get("regime") or "unknown").strip()
    liquidity_bucket = str(setup_profile.get("liquidity_bucket") or "unknown").strip()
    thesis = str(payload.get("entry_thesis") or payload.get("pipeline_mode") or "unknown").strip()
    return f"{thesis} | {regime} | {liquidity_bucket}"


def _build_exit_reason_breakdowns(
    close_rows: pd.DataFrame,
    *,
    load_json,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if close_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    symbol_stats: dict[str, dict[str, float | int | str]] = {}
    setup_stats: dict[tuple[str, str], dict[str, float | int | str]] = {}
    for _, row in close_rows.iterrows():
        metadata = load_json(row.get("metadata_json"), {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        reason_tokens = set(_reason_tokens(metadata.get("reason", "")))
        focus_tokens = [reason for reason in FOCUS_EXIT_REASONS if reason in reason_tokens]
        if not focus_tokens:
            continue
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue
        net_pnl = float(row.get("net_pnl") or 0.0)
        net_return_pct = float(row.get("net_return_pct") or 0.0)
        symbol_bucket = symbol_stats.setdefault(
            symbol,
            {
                "symbol": symbol,
                "focus_close_count": 0,
                "research_exit_count": 0,
                "research_de_risk_count": 0,
                "evidence_de_risk_count": 0,
                "_total_net_pnl": 0.0,
                "_total_net_return_pct": 0.0,
            },
        )
        symbol_bucket["focus_close_count"] = int(symbol_bucket["focus_close_count"]) + 1
        symbol_bucket["_total_net_pnl"] = float(symbol_bucket["_total_net_pnl"]) + net_pnl
        symbol_bucket["_total_net_return_pct"] = (
            float(symbol_bucket["_total_net_return_pct"]) + net_return_pct
        )
        if "research_exit" in focus_tokens:
            symbol_bucket["research_exit_count"] = int(symbol_bucket["research_exit_count"]) + 1
        if "research_de_risk" in focus_tokens:
            symbol_bucket["research_de_risk_count"] = int(symbol_bucket["research_de_risk_count"]) + 1
        if "evidence_de_risk" in focus_tokens:
            symbol_bucket["evidence_de_risk_count"] = int(symbol_bucket["evidence_de_risk_count"]) + 1

        setup_label = _focus_setup_label(metadata)
        setup_key = (symbol, setup_label)
        setup_bucket = setup_stats.setdefault(
            setup_key,
            {
                "symbol": symbol,
                "setup": setup_label,
                "focus_close_count": 0,
                "research_exit_count": 0,
                "research_de_risk_count": 0,
                "evidence_de_risk_count": 0,
                "_total_net_pnl": 0.0,
                "_total_net_return_pct": 0.0,
            },
        )
        setup_bucket["focus_close_count"] = int(setup_bucket["focus_close_count"]) + 1
        setup_bucket["_total_net_pnl"] = float(setup_bucket["_total_net_pnl"]) + net_pnl
        setup_bucket["_total_net_return_pct"] = (
            float(setup_bucket["_total_net_return_pct"]) + net_return_pct
        )
        if "research_exit" in focus_tokens:
            setup_bucket["research_exit_count"] = int(setup_bucket["research_exit_count"]) + 1
        if "research_de_risk" in focus_tokens:
            setup_bucket["research_de_risk_count"] = int(setup_bucket["research_de_risk_count"]) + 1
        if "evidence_de_risk" in focus_tokens:
            setup_bucket["evidence_de_risk_count"] = int(setup_bucket["evidence_de_risk_count"]) + 1

    def finalize(rows: dict) -> pd.DataFrame:
        records: list[dict[str, float | int | str]] = []
        for payload in rows.values():
            count = max(1, int(payload["focus_close_count"]))
            records.append(
                {
                    **{
                        key: value
                        for key, value in payload.items()
                        if not str(key).startswith("_")
                    },
                    "avg_net_return_pct": round(
                        float(payload["_total_net_return_pct"]) / count,
                        4,
                    ),
                    "total_net_pnl": round(float(payload["_total_net_pnl"]), 4),
                }
            )
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        return df.sort_values(
            [
                "research_exit_count",
                "focus_close_count",
                "avg_net_return_pct",
            ],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    return finalize(symbol_stats), finalize(setup_stats)


def _build_promotion_funnel_rows(
    *,
    shadow_symbols,
    promotion_candidates,
    model_observations,
    latest_model_events: pd.DataFrame,
    load_json,
) -> list[dict]:
    candidate_map = promotion_candidates if isinstance(promotion_candidates, dict) else {}
    observation_map = model_observations if isinstance(model_observations, dict) else {}
    shadow_set = {
        str(symbol)
        for symbol in (shadow_symbols or [])
        if str(symbol).strip()
    }
    event_map: dict[str, dict] = {}
    if not latest_model_events.empty:
        for _, row in latest_model_events.iterrows():
            symbol = str(row.get("symbol") or "").strip()
            if not symbol or symbol in event_map:
                continue
            payload = load_json(row.get("payload_json"), {}) or {}
            event_map[symbol] = {
                "event_type": str(row.get("event_type") or ""),
                "created_at": str(row.get("created_at") or ""),
                "reason": _event_reason(payload),
            }

    ordered_symbols = list(candidate_map.keys())
    ordered_symbols.extend(
        symbol for symbol in observation_map.keys() if symbol not in ordered_symbols
    )
    ordered_symbols.extend(
        symbol for symbol in shadow_set if symbol not in ordered_symbols
    )
    ordered_symbols.extend(
        symbol for symbol in event_map.keys() if symbol not in ordered_symbols
    )

    rows: list[dict] = []
    for symbol in ordered_symbols:
        candidate = candidate_map.get(symbol, {}) or {}
        observation = observation_map.get(symbol, {}) or {}
        latest_event = event_map.get(symbol, {}) or {}
        candidate_status = str(candidate.get("status") or "")
        if observation:
            funnel_stage = "promoted_observing"
        elif candidate_status == "live":
            funnel_stage = "live_canary"
        elif candidate_status == "shadow":
            funnel_stage = "shadow_candidate"
        elif symbol in shadow_set:
            funnel_stage = "shadow_observation"
        else:
            event_type = str(latest_event.get("event_type") or "")
            stage_by_event = {
                "paper_canary_open": "paper_canary_opened",
                "model_promotion_candidate_rejected": "candidate_rejected",
                "model_promotion_promoted": "promoted",
                "model_observation_accepted": "accepted",
                "model_rollback": "rolled_back",
                "model_promotion_candidate_started": "candidate_registered",
                "model_promotion_live_started": "live_canary_started",
                "model_observation_started": "observation_started",
            }
            funnel_stage = stage_by_event.get(event_type, "idle")

        rows.append(
            {
                "symbol": symbol,
                "funnel_stage": funnel_stage,
                "shadow_observation": symbol in shadow_set,
                "candidate_status": candidate_status or "",
                "observation_status": str(observation.get("status") or ""),
                "active_model": _model_path_label(
                    candidate.get("active_model_path")
                    or observation.get("active_model_path")
                ),
                "challenger_model": _model_path_label(
                    candidate.get("challenger_model_path")
                ),
                "last_event_type": str(latest_event.get("event_type") or ""),
                "last_event_at": str(latest_event.get("created_at") or ""),
                "last_reason": str(latest_event.get("reason") or ""),
            }
        )
    return rows


def _build_promotion_funnel_summary(rows: list[dict]) -> dict[str, int]:
    summary = {
        "shadow_observation": 0,
        "shadow_candidate": 0,
        "live_canary": 0,
        "promoted_observing": 0,
        "paper_canary_opened": 0,
        "promoted": 0,
        "accepted": 0,
        "candidate_rejected": 0,
        "rolled_back": 0,
    }
    for row in rows:
        stage = str(row.get("funnel_stage") or "")
        if stage in summary:
            summary[stage] += 1
    return summary


def _parse_report_field(content: str, field: str) -> str:
    prefix = f"- {field}:"
    for line in str(content or "").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def _latest_rollback_snapshot(query_one, load_json) -> dict[str, str]:
    rollback_event = query_one(
        "SELECT symbol, payload_json, created_at "
        "FROM execution_events "
        "WHERE event_type='model_rollback' "
        "ORDER BY created_at DESC LIMIT 1"
    )
    if rollback_event:
        payload = load_json(rollback_event.get("payload_json"), {}) or {}
        return {
            "symbol": str(rollback_event.get("symbol") or "SYSTEM"),
            "reason": str(payload.get("reason") or ""),
            "created_at": str(rollback_event.get("created_at") or ""),
        }
    rollback_report = query_one(
        "SELECT symbol, content, created_at "
        "FROM report_artifacts "
        "WHERE report_type='model_rollback' "
        "ORDER BY created_at DESC LIMIT 1"
    )
    if not rollback_report:
        return {}
    return {
        "symbol": str(rollback_report.get("symbol") or "SYSTEM"),
        "reason": _parse_report_field(str(rollback_report.get("content") or ""), "reason"),
        "created_at": str(rollback_report.get("created_at") or ""),
    }


def _event_payload_value(payload: dict, *keys: str):
    if not isinstance(payload, dict):
        return ""
    for key in keys:
        value = payload.get(key)
        if value not in ("", None):
            return value
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for key in keys:
            value = metrics.get(key)
            if value not in ("", None):
                return value
    return ""


def _build_model_lifecycle_summary(
    *,
    t,
    lifecycle_rows: list[dict],
    latest_rollback: dict[str, str],
    max_symbols: int = 4,
) -> str:
    if not lifecycle_rows and not any(latest_rollback.values()):
        return ""
    parts: list[str] = []
    for row in lifecycle_rows[:max_symbols]:
        parts.append(
            f"{row['symbol']} "
            f"{t('Active Model')}={row['active_model']} | "
            f"{t('Challenger Model')}={row['challenger_model']} | "
            f"{t('Observation State')}={t(str(row.get('observation_status') or 'idle'))}"
        )
    rollback_symbol = t(str(latest_rollback.get("symbol") or "SYSTEM"))
    rollback_reason = t(str(latest_rollback.get("reason") or "none"))
    parts.append(f"{t('Latest Rollback')}={rollback_symbol} | {rollback_reason}")
    if not parts:
        return ""
    return f"{t('Model Lifecycle Summary')}: " + " ; ".join(parts)


def render_overview_page(
    *,
    page_ctx=None,
    st=None,
    t=None,
    current_language=None,
    query_df=None,
    query_one=None,
    get_state_json=None,
    load_json=None,
    parse_markdown_metrics=None,
    parse_report_history=None,
    to_numeric_percent=None,
    display_df=None,
    display_value=None,
    runtime_state_snapshot=None,
) -> None:
    st = _ctx_value(page_ctx, "st", st)
    t = _ctx_value(page_ctx, "t", t)
    current_language = _ctx_value(page_ctx, "current_language", current_language)
    query_df = _ctx_value(page_ctx, "query_df", query_df)
    query_one = _ctx_value(page_ctx, "query_one", query_one)
    get_state_json = _ctx_value(page_ctx, "get_state_json", get_state_json)
    load_json = _ctx_value(page_ctx, "load_json", load_json)
    parse_markdown_metrics = _ctx_value(page_ctx, "parse_markdown_metrics", parse_markdown_metrics)
    parse_report_history = _ctx_value(page_ctx, "parse_report_history", parse_report_history)
    to_numeric_percent = _ctx_value(page_ctx, "to_numeric_percent", to_numeric_percent)
    display_df = _ctx_value(page_ctx, "display_df", display_df)
    runtime_state_snapshot = _ctx_value(page_ctx, "runtime_state_snapshot", runtime_state_snapshot)
    display_value = display_value or (lambda value: value)
    runtime_state = runtime_state_snapshot() if callable(runtime_state_snapshot) else None
    st.title(t("CryptoAI v3 Overview"))

    latest_account = query_one(
        "SELECT * FROM account_snapshots ORDER BY created_at DESC LIMIT 1"
    )
    account_history = query_df(
        "SELECT created_at, equity, realized_pnl, unrealized_pnl, drawdown_pct "
        "FROM account_snapshots ORDER BY created_at DESC LIMIT 50"
    )
    positions = query_df("SELECT * FROM positions ORDER BY updated_at DESC")
    closed_trades = query_df(
        "SELECT * FROM trades WHERE status='closed' ORDER BY exit_time DESC LIMIT 20"
    )
    focus_close_rows = query_df(
        "SELECT symbol, trade_id, event_time, net_pnl, net_return_pct, holding_hours, metadata_json "
        "FROM pnl_ledger WHERE event_type='close' ORDER BY event_time DESC, id DESC LIMIT 500"
    )
    latest_walkforward = query_one(
        "SELECT * FROM walkforward_runs ORDER BY created_at DESC LIMIT 1"
    )
    latest_health = query_one(
        "SELECT * FROM report_artifacts WHERE report_type='health' ORDER BY created_at DESC LIMIT 1"
    )
    latest_cycle = query_one(
        "SELECT * FROM cycle_runs ORDER BY created_at DESC LIMIT 1"
    )
    latest_scheduler = query_one(
        "SELECT * FROM scheduler_runs ORDER BY created_at DESC LIMIT 1"
    )
    latest_incident = query_one(
        "SELECT * FROM report_artifacts WHERE report_type='incident' ORDER BY created_at DESC LIMIT 1"
    )
    latest_ops = query_one(
        "SELECT * FROM report_artifacts WHERE report_type='ops_overview' ORDER BY created_at DESC LIMIT 1"
    )
    latest_failure = query_one(
        "SELECT * FROM report_artifacts WHERE report_type='failure' ORDER BY created_at DESC LIMIT 1"
    )
    latest_reconciliation = query_one(
        "SELECT * FROM reconciliation_runs ORDER BY created_at DESC LIMIT 1"
    )
    latest_guard = query_one(
        "SELECT * FROM report_artifacts WHERE report_type='guard' ORDER BY created_at DESC LIMIT 1"
    )
    latest_performance = query_one(
        "SELECT * FROM report_artifacts WHERE report_type='performance' ORDER BY created_at DESC LIMIT 1"
    )
    latest_daily_focus = query_one(
        "SELECT * FROM report_artifacts WHERE report_type='daily_focus' ORDER BY created_at DESC LIMIT 1"
    )
    latest_alpha = query_one(
        "SELECT * FROM report_artifacts WHERE report_type='alpha_diagnostics' ORDER BY created_at DESC LIMIT 1"
    )
    latest_consistency = query_one(
        "SELECT * FROM report_artifacts WHERE report_type='backtest_live_consistency' ORDER BY created_at DESC LIMIT 1"
    )
    model_degradation_status = query_one(
        "SELECT value, updated_at FROM system_state WHERE key='model_degradation_status'"
    )
    model_degradation_reason = query_one(
        "SELECT value, updated_at FROM system_state WHERE key='model_degradation_reason'"
    )
    performance_history = parse_report_history("performance", limit=20)
    perf_metrics = (
        parse_markdown_metrics(latest_performance["content"])
        if latest_performance
        else {}
    )
    daily_focus_metrics = (
        parse_markdown_metrics(latest_daily_focus["content"])
        if latest_daily_focus
        else {}
    )
    alpha_metrics = (
        parse_markdown_metrics(latest_alpha["content"])
        if latest_alpha
        else {}
    )
    consistency_metrics = (
        parse_markdown_metrics(latest_consistency["content"])
        if latest_consistency
        else {}
    )
    last_accuracy_guard_triggered = query_one(
        "SELECT value, updated_at FROM system_state WHERE key='last_accuracy_guard_triggered'"
    )
    last_accuracy_guard_reason = query_one(
        "SELECT value, updated_at FROM system_state WHERE key='last_accuracy_guard_reason'"
    )
    latest_model_event = query_one(
        "SELECT event_type, symbol, payload_json, created_at "
        "FROM execution_events "
        "WHERE event_type IN ("
        "'model_promotion_candidate_started',"
        "'model_promotion_live_started',"
        "'model_promotion_candidate_rejected',"
        "'model_promotion_promoted',"
        "'model_observation_started',"
        "'model_observation_accepted',"
        "'model_rollback'"
        ") "
        "ORDER BY created_at DESC LIMIT 1"
    )
    latest_training_rows = query_df(
        "SELECT symbol, metadata_json, created_at FROM training_runs ORDER BY created_at DESC LIMIT 200"
    )
    exit_reason_by_symbol, exit_reason_by_setup = _build_exit_reason_breakdowns(
        focus_close_rows,
        load_json=load_json,
    )
    promotion_candidates = (
        getattr(runtime_state, "model_promotion_candidates", {})
        if runtime_state is not None
        else get_state_json("model_promotion_candidates", {}) or {}
    )
    model_observations = (
        getattr(runtime_state, "model_promotion_observations", {})
        if runtime_state is not None
        else get_state_json("model_promotion_observations", {}) or {}
    )
    lifecycle_rows = _build_model_lifecycle_rows(
        latest_training_rows=latest_training_rows,
        model_observations=model_observations,
        promotion_candidates=promotion_candidates,
        load_json=load_json,
    )
    lifecycle_summary = _build_model_lifecycle_summary(
        t=t,
        lifecycle_rows=lifecycle_rows,
        latest_rollback=_latest_rollback_snapshot(query_one, load_json),
    )
    view_model = build_overview_view_model(
        latest_account=latest_account,
        latest_reconciliation=latest_reconciliation,
        latest_ops=latest_ops,
        latest_guard=latest_guard,
        latest_health=latest_health,
        latest_incident=latest_incident,
        latest_failure=latest_failure,
        latest_cycle=latest_cycle,
        latest_scheduler=latest_scheduler,
        latest_model_event=latest_model_event,
        model_degradation_status=model_degradation_status,
        model_degradation_reason=model_degradation_reason,
        last_accuracy_guard_triggered=last_accuracy_guard_triggered,
        last_accuracy_guard_reason=last_accuracy_guard_reason,
        perf_metrics=perf_metrics,
        daily_focus_metrics=daily_focus_metrics,
        alpha_metrics=alpha_metrics,
        consistency_metrics=consistency_metrics,
        lifecycle_rows=lifecycle_rows,
        lifecycle_summary=lifecycle_summary,
    )

    if view_model["lifecycle_summary"]:
        st.caption(view_model["lifecycle_summary"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        t("Equity"),
        (
            f"${float(view_model['primary_metrics']['equity']):,.2f}"
            if view_model["primary_metrics"]["equity"] is not None
            else display_value("N/A")
        ),
    )
    c2.metric(
        t("Open Positions"),
        str(view_model["primary_metrics"]["open_positions"]),
    )
    c3.metric(
        t("Drawdown"),
        (
            f"{float(view_model['primary_metrics']['drawdown_pct']):.2%}"
            if view_model["primary_metrics"]["drawdown_pct"] is not None
            else display_value("N/A")
        ),
    )
    c4.metric(
        t("Circuit Breaker"),
        t("ACTIVE")
        if view_model["primary_metrics"]["circuit_breaker_active"]
        else t("OFF"),
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        t("Reconciliation"),
        display_value(view_model["secondary_metrics"]["reconciliation_status"] or "N/A"),
    )
    c6.metric(
        t("XGB Accuracy"),
        display_value(view_model["secondary_metrics"]["expanded_xgb_accuracy"]),
    )
    c7.metric(
        t("Ops Report"),
        t("READY") if view_model["secondary_metrics"]["ops_report_ready"] else t("MISSING"),
    )
    c8.metric(
        t("Guard Report"),
        t("READY") if view_model["secondary_metrics"]["guard_report_ready"] else t("MISSING"),
    )
    c9, c10, c11, c12 = st.columns(4)
    c9.metric(
        t("Net Expectancy"),
        display_value(view_model["secondary_metrics"]["net_expectancy"] or "N/A"),
    )
    c10.metric(
        t("Net Profit Factor"),
        display_value(view_model["secondary_metrics"]["net_profit_factor"] or "N/A"),
    )
    c11.metric(
        t("Equity Return"),
        display_value(view_model["secondary_metrics"]["equity_return"] or "N/A"),
    )
    c12.metric(
        t("Total Trade Cost"),
        display_value(view_model["secondary_metrics"]["total_trade_cost"] or "N/A"),
    )
    if daily_focus_metrics:
        st.subheader(t("Daily Focus"))
        c13, c14, c15, c16, c17 = st.columns(5)
        c13.metric(
            t("Net PnL"),
            display_value(daily_focus_metrics.get("Daily Net PnL", "N/A")),
        )
        c14.metric(
            t("Profit Factor"),
            display_value(daily_focus_metrics.get("Daily Profit Factor", "N/A")),
        )
        c15.metric(
            t("Max Drawdown"),
            display_value(daily_focus_metrics.get("Daily Max Drawdown", "N/A")),
        )
        c16.metric(
            t("Avg Net PnL/Trade"),
            display_value(
                daily_focus_metrics.get("Average Net PnL Per Trade", "N/A")
            ),
        )
        c17.metric(
            t("Risk-Blocked Loss Avoided"),
            display_value(
                daily_focus_metrics.get("Risk-Blocked Loss Avoided", "N/A")
            ),
        )
        c18, c19, c20, c21 = st.columns(4)
        c18.metric(
            t("Short-Horizon Opens"),
            display_value(
                daily_focus_metrics.get("Short-horizon Softened Opens", "N/A")
            ),
        )
        c19.metric(
            t("Short-Horizon Net PnL"),
            display_value(
                daily_focus_metrics.get("Short-horizon Softened Net PnL", "N/A")
            ),
        )
        c20.metric(
            t("Short-Horizon Closed Trades"),
            display_value(
                daily_focus_metrics.get("Short-horizon Softened Closed Trades", "N/A")
            ),
        )
        c21.metric(
            t("Short-Horizon Pauses"),
            display_value(
                daily_focus_metrics.get("Short-horizon Negative-Edge Pauses", "N/A")
            ),
        )
        if alpha_metrics:
            c22, c23, c24, c25 = st.columns(4)
            c22.metric(
                t("Short-Horizon Status"),
                display_value(alpha_metrics.get("Short-Horizon Status", "N/A")),
            )
            c23.metric(
                t("Short-Horizon Samples"),
                display_value(alpha_metrics.get("Short-Horizon Recent Samples", "N/A")),
            )
            c24.metric(
                t("Short-Horizon Expectancy"),
                display_value(alpha_metrics.get("Short-Horizon Recent Expectancy", "N/A")),
            )
            c25.metric(
                t("Short-Horizon Profit Factor"),
                display_value(alpha_metrics.get("Short-Horizon Recent Profit Factor", "N/A")),
            )
            short_horizon_notice = _short_horizon_overview_notice(
                alpha_metrics,
                current_language=current_language(),
            )
            if short_horizon_notice:
                getattr(st, short_horizon_notice["level"], st.info)(
                    short_horizon_notice["message"]
                )
            if current_language() == "zh":
                st.caption(
                    "FastAlpha: "
                    f"开仓={display_value(daily_focus_metrics.get('Fast Alpha Opens', 'N/A'))} | "
                    f"平仓={display_value(daily_focus_metrics.get('Fast Alpha Closed Trades', 'N/A'))} | "
                    f"净收益={display_value(daily_focus_metrics.get('Fast Alpha Net PnL', 'N/A'))} | "
                    f"Short-Horizon放行={display_value(daily_focus_metrics.get('Short-horizon Softened Opens', 'N/A'))} | "
                    f"Short-Horizon暂停={display_value(daily_focus_metrics.get('Short-horizon Negative-Edge Pauses', 'N/A'))} | "
                    f"执行准确率={display_value(alpha_metrics.get('execution_accuracy', alpha_metrics.get('execution_accuracy_pct', 'N/A')))}"
                )
            else:
                st.caption(
                    "Fast Alpha: "
                    f"opens={display_value(daily_focus_metrics.get('Fast Alpha Opens', 'N/A'))} | "
                    f"closed={display_value(daily_focus_metrics.get('Fast Alpha Closed Trades', 'N/A'))} | "
                    f"net_pnl={display_value(daily_focus_metrics.get('Fast Alpha Net PnL', 'N/A'))} | "
                    f"softened={display_value(daily_focus_metrics.get('Short-horizon Softened Opens', 'N/A'))} | "
                    f"pauses={display_value(daily_focus_metrics.get('Short-horizon Negative-Edge Pauses', 'N/A'))}"
                )
    if consistency_metrics:
        st.caption(
            f"{t('Consistency Risk')}: "
            f"{display_value(consistency_metrics.get('Suspicious Symbols', 'N/A'))} | "
            f"{t('Recent Live Expectancy')}="
            f"{display_value(consistency_metrics.get('Recent Live Expectancy', 'N/A'))} | "
            f"{t('Recent Live Profit Factor')}="
            f"{display_value(consistency_metrics.get('Recent Live Profit Factor', 'N/A'))}"
        )
    if perf_metrics:
        if current_language() == "zh":
            st.caption(
                "最近XGB="
                f"{display_value(_metric_value(perf_metrics, 'Current Execution Universe XGBoost Direction Accuracy', 'XGBoost Direction Accuracy'))} | "
                "扩展XGB="
                f"{display_value(_metric_value(perf_metrics, 'Expanded XGBoost Direction Accuracy', 'XGBoost Direction Accuracy'))} | "
                "执行准确率="
                f"{display_value(_metric_value(perf_metrics, 'Current Execution Universe Execution Accuracy', 'Execution Accuracy'))} | "
                "预测窗口="
                f"{display_value(_metric_value(perf_metrics, 'Current Execution Universe Window', 'Recent Prediction Window'))} | "
                "Shadow准确率="
                f"{display_value(_metric_value(perf_metrics, 'Shadow Accuracy'))}"
            )
        else:
            st.caption(
                "Recent XGB="
                f"{display_value(_metric_value(perf_metrics, 'Current Execution Universe XGBoost Direction Accuracy', 'XGBoost Direction Accuracy'))} | "
                "Expanded XGB="
                f"{display_value(_metric_value(perf_metrics, 'Expanded XGBoost Direction Accuracy', 'XGBoost Direction Accuracy'))} | "
                "Execution Accuracy="
                f"{display_value(_metric_value(perf_metrics, 'Current Execution Universe Execution Accuracy', 'Execution Accuracy'))} | "
                "Prediction Window="
                f"{display_value(_metric_value(perf_metrics, 'Current Execution Universe Window', 'Recent Prediction Window'))} | "
                "Shadow Accuracy="
                f"{display_value(_metric_value(perf_metrics, 'Shadow Accuracy'))}"
            )
    if latest_cycle:
        if current_language() == "zh":
            st.caption(
                f"{t('Last Cycle')}: {display_value(latest_cycle['status'])} | "
                f"开仓={latest_cycle['opened_positions']} | "
                f"平仓={latest_cycle['closed_positions']}"
            )
        else:
            st.caption(
                f"{t('Last Cycle')}: {display_value(latest_cycle['status'])} | "
                f"opened={latest_cycle['opened_positions']} | "
                f"closed={latest_cycle['closed_positions']}"
            )
    if latest_scheduler:
        st.caption(
            f"{t('Last Scheduler Job')}: {display_value(latest_scheduler['job_name'])} | "
            f"{display_value(latest_scheduler['status'])}"
        )
    if latest_health or latest_incident or latest_ops:
        if current_language() == "zh":
            st.caption(
                f"{t('Latest Reports')}: "
                f"健康={'是' if latest_health else '否'} | "
                f"事故={'是' if latest_incident else '否'} | "
                f"运维={'是' if latest_ops else '否'} | "
                f"风控={'是' if latest_guard else '否'}"
            )
        else:
            st.caption(
                f"{t('Latest Reports')}: "
                f"health={'yes' if latest_health else 'no'} | "
                f"incident={'yes' if latest_incident else 'no'} | "
                f"ops={'yes' if latest_ops else 'no'} | "
                f"guard={'yes' if latest_guard else 'no'}"
            )
    if latest_failure:
        st.caption(t("Failure report available."))
    if model_degradation_status:
        st.caption(
            f"{t('Model Degradation')}: "
            f"{display_value(model_degradation_status['value'])} | "
            f"{display_value(model_degradation_reason['value']) if model_degradation_reason else ''}"
        )
    if last_accuracy_guard_triggered:
        st.caption(
            f"{t('Last Accuracy Guard')}: "
            f"{last_accuracy_guard_triggered['value']} | "
            f"{display_value(last_accuracy_guard_reason['value']) if last_accuracy_guard_reason else ''}"
        )
    if latest_model_event:
        st.caption(
            f"{t('Latest Model Event')}: "
            f"{display_value(latest_model_event['event_type'])} | "
            f"{display_value(latest_model_event['symbol'] or 'SYSTEM')} | "
            f"{latest_model_event['created_at']}"
        )

    if not performance_history.empty:
        st.subheader(t("Accuracy Trend"))
        chart_df = performance_history[["created_at"]].copy()
        if "Expanded XGBoost Direction Accuracy" in performance_history:
            chart_df["XGB"] = to_numeric_percent(
                performance_history["Expanded XGBoost Direction Accuracy"]
            )
        elif "XGBoost Direction Accuracy" in performance_history:
            chart_df["XGB"] = to_numeric_percent(
                performance_history["XGBoost Direction Accuracy"]
            )
        if "LLM Action Accuracy" in performance_history:
            chart_df["LLM"] = to_numeric_percent(
                performance_history["LLM Action Accuracy"]
            )
        if "Fusion Signal Accuracy" in performance_history:
            chart_df["Fusion"] = to_numeric_percent(
                performance_history["Fusion Signal Accuracy"]
            )
        chart_df = chart_df.set_index("created_at").dropna(how="all")
        if not chart_df.empty:
            st.line_chart(chart_df, use_container_width=True)

    if not account_history.empty:
        st.subheader(t("Equity Curve"))
        curve_df = (
            account_history.sort_values("created_at")
            .set_index("created_at")[["equity"]]
        )
        if not curve_df.empty:
            st.line_chart(curve_df, use_container_width=True)

    st.subheader(t("Open Positions"))
    if positions.empty:
        st.info(t("No open positions."))
    else:
        st.dataframe(display_df(positions), use_container_width=True)

    st.subheader(t("Recent Closed Trades"))
    if closed_trades.empty:
        st.info(t("No closed trades."))
    else:
        st.dataframe(
            display_df(
                closed_trades[
                    [
                        "symbol",
                        "entry_price",
                        "exit_price",
                        "quantity",
                        "pnl",
                        "pnl_pct",
                        "exit_time",
                    ]
                ]
            ),
            use_container_width=True,
        )
    st.subheader(t("Exit Reason Drag By Symbol"))
    if exit_reason_by_symbol.empty:
        st.info(t("No focused exit-reason samples."))
    else:
        st.dataframe(
            display_df(exit_reason_by_symbol),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(t("Exit Reason Drag By Setup"))
    if exit_reason_by_setup.empty:
        st.info(t("No focused setup-level exit samples."))
    else:
        st.dataframe(
            display_df(exit_reason_by_setup.head(20)),
            use_container_width=True,
            hide_index=True,
        )


def render_settings_page(
    *,
    page_ctx=None,
    st=None,
    t=None,
    get_state_json=None,
    get_state_row=None,
    set_state_json=None,
    display_json=None,
    display_value=None,
    runtime_setting_defaults=None,
    build_runtime_override_payload=None,
    runtime_state_snapshot=None,
) -> None:
    st = _ctx_value(page_ctx, "st", st)
    t = _ctx_value(page_ctx, "t", t)
    get_state_json = _ctx_value(page_ctx, "get_state_json", get_state_json)
    get_state_row = _ctx_value(page_ctx, "get_state_row", get_state_row)
    set_state_json = _ctx_value(page_ctx, "set_state_json", set_state_json)
    display_json = _ctx_value(page_ctx, "display_json", display_json)
    runtime_setting_defaults = _ctx_value(page_ctx, "runtime_setting_defaults", runtime_setting_defaults)
    build_runtime_override_payload = _ctx_value(page_ctx, "build_runtime_override_payload", build_runtime_override_payload)
    runtime_state_snapshot = _ctx_value(page_ctx, "runtime_state_snapshot", runtime_state_snapshot)
    display_value = display_value or (lambda value: value)
    st.title(t("Runtime Settings"))
    defaults = runtime_setting_defaults()
    runtime_state = runtime_state_snapshot() if callable(runtime_state_snapshot) else None
    overrides = (
        getattr(runtime_state, "runtime_settings_overrides", {}) or {}
        if runtime_state is not None
        else get_state_json("runtime_settings_overrides", {}) or {}
    )
    locked_fields = (
        getattr(runtime_state, "runtime_settings_locked_fields", []) or []
        if runtime_state is not None
        else get_state_json("runtime_settings_locked_fields", []) or []
    )
    learning_overrides = (
        getattr(runtime_state, "runtime_settings_learning_overrides", {}) or {}
        if runtime_state is not None
        else get_state_json("runtime_settings_learning_overrides", {}) or {}
    )
    learning_details = (
        getattr(runtime_state, "runtime_settings_learning_details", {}) or {}
        if runtime_state is not None
        else get_state_json("runtime_settings_learning_details", {}) or {}
    )
    override_conflicts = (
        getattr(runtime_state, "runtime_settings_override_conflicts", {}) or {}
        if runtime_state is not None
        else get_state_json("runtime_settings_override_conflicts", {}) or {}
    )
    effective = (
        getattr(runtime_state, "runtime_settings_effective", {}) or {}
        if runtime_state is not None
        else get_state_json("runtime_settings_effective", {}) or {}
    )
    override_status = get_state_row("runtime_settings_override_status")
    view_model = build_runtime_settings_view_model(
        defaults=defaults,
        overrides=overrides,
        locked_fields=locked_fields,
        learning_overrides=learning_overrides,
        learning_details=learning_details,
        override_conflicts=override_conflicts,
        effective=effective,
        override_status=override_status,
    )

    st.caption(
        t("These overrides apply on the next engine cycle without editing .env. Model degradation logic may still tighten thresholds at runtime.")
    )
    if override_status:
        st.caption(
            f"{t('Override Status')}: {override_status['value']} | {t('updated_at')}={override_status['updated_at']}"
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        t("Override Status"),
        display_value(view_model["metrics"]["override_status"]),
    )
    c2.metric(t("Learning Overrides"), str(view_model["metrics"]["learning_override_count"]))
    c3.metric(t("Manual Overrides"), str(view_model["metrics"]["manual_override_count"]))
    c4.metric(
        t("Active Automation Rules"),
        str(view_model["metrics"]["active_automation_rule_count"]),
    )

    st.subheader(t("Automation Summary"))
    st.json(display_json(view_model["automation_summary"]))

    current_values = view_model["current_values"]

    with st.form("runtime_settings_form"):
        c1, c2 = st.columns(2)
        with c1:
            xgb_threshold = st.number_input(
                t("XGBoost Probability Threshold"),
                min_value=0.0,
                max_value=1.0,
                value=float(current_values["xgboost_probability_threshold"]),
                step=0.01,
                format="%.2f",
            )
            min_liquidity_ratio = st.number_input(
                t("Min Liquidity Ratio"),
                min_value=0.0,
                max_value=5.0,
                value=float(current_values["min_liquidity_ratio"]),
                step=0.05,
                format="%.2f",
            )
            fixed_stop_loss_pct = st.number_input(
                t("Fixed Stop Loss Pct"),
                min_value=0.0,
                max_value=0.2,
                value=float(current_values["fixed_stop_loss_pct"]),
                step=0.001,
                format="%.3f",
            )
        with c2:
            final_score_threshold = st.number_input(
                t("Final Score Threshold"),
                min_value=0.0,
                max_value=1.0,
                value=float(current_values["final_score_threshold"]),
                step=0.01,
                format="%.2f",
            )
            sentiment_weight = st.number_input(
                t("Sentiment Weight"),
                min_value=-1.0,
                max_value=1.0,
                value=float(current_values["sentiment_weight"]),
                step=0.05,
                format="%.2f",
            )
            take_profit_levels = st.text_input(
                t("Take Profit Levels"),
                value=", ".join(
                    f"{level:.3f}".rstrip("0").rstrip(".")
                    for level in current_values["take_profit_levels"]
                ),
                help=t("Comma-separated values, for example: 0.05, 0.08"),
            )
        if hasattr(st, "multiselect"):
            locked_fields_input = st.multiselect(
                t("Locked Fields"),
                options=list(defaults.keys()),
                default=[field for field in locked_fields if field in defaults],
            )
        else:
            locked_fields_input = [field for field in locked_fields if field in defaults]

        save = st.form_submit_button(
            t("Save Runtime Overrides"),
            type="primary",
            use_container_width=True,
        )
        reset = st.form_submit_button(
            t("Reset To Defaults"),
            use_container_width=True,
        )

    if save:
        try:
            payload = build_runtime_override_payload(
                {
                    "xgboost_probability_threshold": xgb_threshold,
                    "final_score_threshold": final_score_threshold,
                    "min_liquidity_ratio": min_liquidity_ratio,
                    "sentiment_weight": sentiment_weight,
                    "fixed_stop_loss_pct": fixed_stop_loss_pct,
                    "take_profit_levels": take_profit_levels,
                }
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            set_state_json("runtime_settings_overrides", payload)
            set_state_json("runtime_settings_locked_fields", locked_fields_input)
            st.success(t("Runtime overrides saved."))
            st.rerun()

    if reset:
        set_state_json("runtime_settings_overrides", {})
        set_state_json("runtime_settings_locked_fields", [])
        st.success(t("Runtime overrides reset to defaults."))
        st.rerun()


def render_ops_page(
    *,
    page_ctx=None,
    st=None,
    t=None,
    run_command=None,
    query_df=None,
    query_one=None,
    get_state_json=None,
    load_json=None,
    display_df=None,
    display_json=None,
    display_value=None,
    localize_report_text=None,
    runtime_state_snapshot=None,
) -> None:
    st = _ctx_value(page_ctx, "st", st)
    t = _ctx_value(page_ctx, "t", t)
    run_command = _ctx_value(page_ctx, "run_command", run_command)
    query_df = _ctx_value(page_ctx, "query_df", query_df)
    query_one = _ctx_value(page_ctx, "query_one", query_one)
    get_state_json = _ctx_value(page_ctx, "get_state_json", get_state_json)
    load_json = _ctx_value(page_ctx, "load_json", load_json)
    display_df = _ctx_value(page_ctx, "display_df", display_df)
    display_json = _ctx_value(page_ctx, "display_json", display_json)
    localize_report_text = _ctx_value(page_ctx, "localize_report_text", localize_report_text)
    runtime_state_snapshot = _ctx_value(page_ctx, "runtime_state_snapshot", runtime_state_snapshot)
    display_value = display_value or (lambda value: value)
    localize_report_text = localize_report_text or (lambda text: text)
    runtime_state = runtime_state_snapshot() if callable(runtime_state_snapshot) else None
    st.title(t("Ops Overview"))
    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("Generate Ops Overview"), type="primary", use_container_width=True):
            ok, output = run_command("ops")
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Ops overview generated."))
            else:
                st.error(t("Ops overview failed."))
    with col2:
        if st.button(t("Run Reconciliation"), use_container_width=True):
            ok, output = run_command("reconcile")
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Reconciliation finished."))
            else:
                st.error(t("Reconciliation failed."))

    latest_ops = query_df(
        "SELECT * FROM report_artifacts WHERE report_type='ops_overview' ORDER BY created_at DESC LIMIT 10"
    )
    recent_nextgen_live_runs = query_df(
        """
        SELECT created_at, payload_json
        FROM execution_events
        WHERE event_type='nextgen_autonomy_live_run'
        ORDER BY created_at DESC, id DESC
        LIMIT 10
        """
    )
    recent_nextgen_live_issues = query_df(
        """
        SELECT created_at, event_type, payload_json
        FROM execution_events
        WHERE event_type IN (
            'nextgen_autonomy_live_run_failed',
            'nextgen_autonomy_live_guard_callback_failed'
        )
        ORDER BY created_at DESC, id DESC
        LIMIT 20
        """
    )
    recent_scheduler = query_df(
        "SELECT * FROM scheduler_runs ORDER BY created_at DESC LIMIT 10"
    )
    recent_reconciliation = query_df(
        "SELECT * FROM reconciliation_runs ORDER BY created_at DESC LIMIT 10"
    )
    shadow_symbols = (
        getattr(runtime_state, "shadow_observation_symbols", [])
        if runtime_state is not None
        else get_state_json("shadow_observation_symbols", []) or []
    )
    paper_exploration_symbols = (
        getattr(runtime_state, "paper_exploration_active_symbols", [])
        if runtime_state is not None
        else get_state_json("paper_exploration_active_symbols", []) or []
    )
    shadow_trade_summary = query_one(
        """
        SELECT
            SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS open_count,
            SUM(CASE WHEN status='evaluated' THEN 1 ELSE 0 END) AS evaluated_count
        FROM shadow_trade_runs
        """
    ) or {}
    latest_prediction_eval = query_one(
        "SELECT created_at FROM prediction_evaluations ORDER BY created_at DESC LIMIT 1"
    )
    paper_canary_summary = query_one(
        """
        SELECT
            COUNT(*) AS total_count,
            SUM(
                CASE
                    WHEN json_extract(payload_json, '$.canary_mode')='soft_review'
                    THEN 1 ELSE 0
                END
            ) AS soft_count
        FROM execution_events
        WHERE event_type='paper_canary_open'
        """
    ) or {}
    fast_alpha_policy_summary = query_one(
        """
        SELECT
            SUM(
                CASE
                    WHEN event_type='fast_alpha_open'
                     AND COALESCE(json_extract(payload_json, '$.review_policy_reason'), '') != ''
                    THEN 1 ELSE 0
                END
            ) AS softened_open_count,
            SUM(
                CASE
                    WHEN event_type='fast_alpha_blocked'
                     AND json_extract(payload_json, '$.reason')='short_horizon_negative_expectancy_pause'
                    THEN 1 ELSE 0
                END
            ) AS negative_pause_count
        FROM execution_events
        WHERE event_type IN ('fast_alpha_open', 'fast_alpha_blocked')
        """
    ) or {}
    latest_training_rows = query_df(
        "SELECT symbol, metadata_json, created_at FROM training_runs ORDER BY created_at DESC LIMIT 200"
    )
    promotion_candidates = (
        getattr(runtime_state, "model_promotion_candidates", {})
        if runtime_state is not None
        else get_state_json("model_promotion_candidates", {}) or {}
    )
    model_observations = (
        getattr(runtime_state, "model_promotion_observations", {})
        if runtime_state is not None
        else get_state_json("model_promotion_observations", {}) or {}
    )
    candidate_rows = _build_promotion_candidate_rows(promotion_candidates)
    latest_model_events = query_df(
        "SELECT event_type, symbol, payload_json, created_at "
        "FROM execution_events "
        "WHERE event_type IN ("
        "'paper_canary_open',"
        "'model_promotion_candidate_started',"
        "'model_promotion_live_started',"
        "'model_promotion_candidate_rejected',"
        "'model_promotion_promoted',"
        "'model_observation_started',"
        "'model_observation_accepted',"
        "'model_rollback'"
        ") "
        "ORDER BY created_at DESC LIMIT 20"
    )
    latest_rollback_reports = query_df(
        "SELECT * FROM report_artifacts WHERE report_type='model_rollback' ORDER BY created_at DESC LIMIT 10"
    )
    funnel_rows = _build_promotion_funnel_rows(
        shadow_symbols=shadow_symbols,
        promotion_candidates=promotion_candidates,
        model_observations=model_observations,
        latest_model_events=latest_model_events,
        load_json=load_json,
    )
    latest_rollback = _latest_rollback_snapshot(query_one, load_json)
    market_data_route = get_state_json("market_data_last_route", {}) or {}
    market_data_failover_stats = get_state_json("market_data_failover_stats", {}) or {}
    view_model = build_ops_view_model(
        shadow_symbols=shadow_symbols,
        paper_exploration_symbols=paper_exploration_symbols,
        shadow_trade_summary=shadow_trade_summary,
        latest_prediction_eval=latest_prediction_eval,
        paper_canary_summary=paper_canary_summary,
        candidate_rows=candidate_rows,
        model_observations=model_observations,
        funnel_rows=funnel_rows,
        latest_rollback=latest_rollback,
        market_data_route=market_data_route,
        market_data_failover_stats=market_data_failover_stats,
        fast_alpha_policy_summary=fast_alpha_policy_summary,
    )
    model_registry_rows = query_df(
        "SELECT symbol, model_id, model_version, role, stage, active, updated_at "
        "FROM model_registry ORDER BY updated_at DESC LIMIT 20"
    )
    model_scorecards = query_df(
        "SELECT symbol, model_id, model_version, stage, evaluation_type, "
        "sample_count, executed_count, accuracy, executed_precision, "
        "avg_trade_return_pct, total_trade_return_pct, expectancy_pct, "
        "profit_factor, max_drawdown_pct, trade_win_rate, avg_cost_pct, "
        "avg_favorable_excursion_pct, avg_adverse_excursion_pct, "
        "objective_score, created_at "
        "FROM model_scorecards ORDER BY created_at DESC LIMIT 20"
    )
    pnl_ledger = query_df(
        "SELECT symbol, trade_id, event_type, event_time, quantity, "
        "notional_value, reference_price, fill_price, gross_pnl, fee_cost, "
        "slippage_cost, net_pnl, net_return_pct, holding_hours, model_id, created_at "
        "FROM pnl_ledger ORDER BY event_time DESC, id DESC LIMIT 20"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("Observation Pool"), str(len(view_model["shadow_symbols"])))
    c2.metric(
        t("Open Shadow Trades"),
        str(view_model["metrics"]["open_shadow_trade_count"]),
    )
    c3.metric(
        t("Evaluated Shadow Trades"),
        str(view_model["metrics"]["evaluated_shadow_trade_count"]),
    )
    c4.metric(
        t("Prediction Evaluations"),
        view_model["metrics"]["latest_prediction_eval_at"] or display_value("N/A"),
    )
    c5, c6, c7, c8 = st.columns(4)
    c5.metric(t("Pending Shadow Candidates"), str(view_model["metrics"]["shadow_candidate_count"]))
    c6.metric(t("Live Canary Candidates"), str(view_model["metrics"]["live_candidate_count"]))
    c7.metric(
        t("Promoted Models Under Observation"),
        str(view_model["model_observation_count"]),
    )
    c8.metric(
        t("Latest Rollback"),
        display_value(view_model["metrics"]["latest_rollback_symbol"] or t("none")),
    )
    c9, c10, c11, c12 = st.columns(4)
    c9.metric(
        t("Paper Canary Opens"),
        str(view_model["metrics"]["paper_canary_open_count"]),
    )
    c10.metric(
        t("Soft Canary Opens"),
        str(view_model["metrics"]["soft_canary_open_count"]),
    )
    c11.metric(
        t("Exploration Slots"),
        str(len(view_model["paper_exploration_symbols"])),
    )
    c12.metric(
        t("Latest Prediction Eval"),
        view_model["metrics"]["latest_prediction_eval_at"] or display_value("N/A"),
    )
    sh1, sh2 = st.columns(2)
    sh1.metric(
        t("Short-Horizon Opens"),
        str(view_model["metrics"]["fast_alpha_short_horizon_softened_open_count"]),
    )
    sh2.metric(
        t("Short-Horizon Pauses"),
        str(view_model["metrics"]["fast_alpha_negative_expectancy_pause_count"]),
    )
    md1, md2, md3, md4 = st.columns(4)
    md1.metric(
        t("Market Data Provider"),
        display_value(view_model["metrics"]["latest_market_data_provider"] or "N/A"),
    )
    md2.metric(
        t("Market Data Operation"),
        display_value(view_model["metrics"]["latest_market_data_operation"] or "N/A"),
    )
    md3.metric(
        t("Market Data Failover Count"),
        str(view_model["metrics"]["market_data_fallback_count"]),
    )
    md4.metric(
        t("Primary Market Data Failures"),
        str(view_model["metrics"]["market_data_primary_failures"]),
    )
    st.caption(
        f"{t('Current Observation Pool')}: "
        + (", ".join(view_model["shadow_symbols"]) if view_model["shadow_symbols"] else t("none"))
    )
    if view_model["paper_exploration_symbols"]:
        st.caption(
            f"{t('Paper Exploration Slot')}: " + ", ".join(view_model["paper_exploration_symbols"])
        )
    st.caption(
        f"{t('Market Data Failover')}: "
        f"{t('ACTIVE') if view_model['metrics']['market_data_fallback_active'] else t('OFF')} | "
        f"{t('Secondary Market Data Failures')}="
        f"{view_model['metrics']['market_data_secondary_failures']} | "
        f"{t('Last Refresh')}="
        f"{display_value(view_model['metrics']['market_data_updated_at'] or 'N/A')}"
    )

    st.subheader(t("Market Data Failover"))
    if view_model["market_data_route"]:
        st.json(display_json(view_model["market_data_route"]))
    else:
        st.info(t("No market data failover route recorded."))

    st.subheader(t("Model Lifecycle"))
    lifecycle_rows = _build_model_lifecycle_rows(
        latest_training_rows=latest_training_rows,
        model_observations=model_observations,
        promotion_candidates=promotion_candidates,
        load_json=load_json,
    )
    lifecycle_df = pd.DataFrame(lifecycle_rows)
    if lifecycle_df.empty:
        st.info(t("No model lifecycle data."))
    else:
        st.dataframe(display_df(lifecycle_df), use_container_width=True)

    st.subheader(t("Promotion Funnel"))
    f1, f2, f3, f4 = st.columns(4)
    f1.metric(
        t("Shadow Observation"),
        str(int(view_model["funnel_summary"].get("shadow_observation", 0))),
    )
    f2.metric(
        t("Shadow Candidates"),
        str(int(view_model["funnel_summary"].get("shadow_candidate", 0))),
    )
    f3.metric(
        t("Live Canary"),
        str(int(view_model["funnel_summary"].get("live_canary", 0))),
    )
    f4.metric(
        t("Promoted Observing"),
        str(int(view_model["funnel_summary"].get("promoted_observing", 0))),
    )
    f5, f6, f7, f8 = st.columns(4)
    f5.metric(
        t("Paper Canary Opened"),
        str(int(view_model["funnel_summary"].get("paper_canary_opened", 0))),
    )
    f6.metric(t("Promoted"), str(int(view_model["funnel_summary"].get("promoted", 0))))
    f7.metric(t("Accepted"), str(int(view_model["funnel_summary"].get("accepted", 0))))
    f8.metric(t("Rolled Back"), str(int(view_model["funnel_summary"].get("rolled_back", 0))))
    if view_model["funnel_rows"]:
        st.dataframe(display_df(pd.DataFrame(view_model["funnel_rows"])), use_container_width=True)
    else:
        st.info(t("No promotion funnel data."))

    st.subheader(t("Current Promotion Candidates"))
    if view_model["candidate_rows"]:
        st.dataframe(
            display_df(pd.DataFrame(view_model["candidate_rows"])),
            use_container_width=True,
        )
    else:
        st.info(t("No promotion candidates pending."))

    st.subheader(t("Current Promotion Observations"))
    observation_rows = _build_promotion_observation_rows(model_observations)
    if observation_rows:
        st.dataframe(
            display_df(pd.DataFrame(observation_rows)),
            use_container_width=True,
        )
    else:
        st.info(t("No promoted models under observation."))

    st.subheader(t("Recent Model Events"))
    if latest_model_events.empty:
        st.info(t("No model lifecycle events recorded."))
    else:
        payloads = latest_model_events["payload_json"].apply(
            lambda value: load_json(value, {}) or {}
        )
        latest_model_events["reason"] = payloads.apply(
            lambda payload: payload.get("reason", "") or payload.get("promotion_reason", "")
        )
        latest_model_events["accuracy"] = payloads.apply(
            lambda payload: _event_payload_value(
                payload,
                "accuracy",
                "live_accuracy",
                "shadow_accuracy",
            )
        )
        latest_model_events["expectancy_pct"] = payloads.apply(
            lambda payload: _event_payload_value(
                payload,
                "expectancy_pct",
                "live_expectancy_pct",
                "shadow_expectancy_pct",
            )
        )
        latest_model_events["profit_factor"] = payloads.apply(
            lambda payload: _event_payload_value(
                payload,
                "profit_factor",
                "live_profit_factor",
                "shadow_profit_factor",
            )
        )
        latest_model_events["max_drawdown_pct"] = payloads.apply(
            lambda payload: _event_payload_value(
                payload,
                "max_drawdown_pct",
                "live_max_drawdown_pct",
                "shadow_max_drawdown_pct",
            )
        )
        latest_model_events["active_model"] = payloads.apply(
            lambda payload: _model_path_label(payload.get("active_model_path"))
        )
        latest_model_events["restored_from"] = payloads.apply(
            lambda payload: _model_path_label(payload.get("restored_from"))
        )
        st.dataframe(
            display_df(
                latest_model_events[
                    [
                        "created_at",
                        "event_type",
                        "symbol",
                        "reason",
                        "accuracy",
                        "expectancy_pct",
                        "profit_factor",
                        "max_drawdown_pct",
                        "active_model",
                        "restored_from",
                    ]
                ]
            ),
            use_container_width=True,
        )

    st.subheader(t("Latest Model Rollback Report"))
    if latest_rollback_reports.empty:
        st.info(t("No rollback reports."))
    else:
        st.code(localize_report_text(latest_rollback_reports.iloc[0]["content"]))

    st.subheader(t("Model Registry"))
    if model_registry_rows.empty:
        st.info(t("No model registry records."))
    else:
        st.dataframe(display_df(model_registry_rows), use_container_width=True)

    st.subheader(t("Recent Model Scorecards"))
    if model_scorecards.empty:
        st.info(t("No model scorecards."))
    else:
        st.dataframe(display_df(model_scorecards), use_container_width=True)

    st.subheader(t("Recent PnL Ledger"))
    if pnl_ledger.empty:
        st.info(t("No PnL ledger entries."))
    else:
        st.dataframe(display_df(pnl_ledger), use_container_width=True)

    if latest_ops.empty:
        st.info(t("No ops overview reports."))
    else:
        st.dataframe(display_df(latest_ops[["report_type", "created_at"]]), use_container_width=True)
        st.code(localize_report_text(latest_ops.iloc[0]["content"]))
    st.subheader(t("Nextgen Live Queue"))
    if recent_nextgen_live_runs.empty:
        st.info(t("No nextgen live queue history."))
    else:
        queue_rows = [
            {
                "created_at": item.get("created_at"),
                "autonomy_cycle_id": int(item.get("autonomy_cycle_id") or 0),
                "status": str(item.get("status") or ""),
                "trigger": str(item.get("trigger") or ""),
                "reason": str(item.get("reason") or ""),
                "latest_issue_event_type": str(item.get("latest_issue_event_type") or ""),
                "latest_issue_reason": str(item.get("latest_issue_reason") or ""),
                "hold_repair_count": int(item.get("hold_priority_count") or 0),
                "postponed_rebuild_count": int(item.get("postponed_rebuild_count") or 0),
                "reprioritized_count": int(item.get("reprioritized_count") or 0),
            }
            for item in build_recent_nextgen_live_queue_runs(
                recent_nextgen_live_runs.to_dict("records"),
                recent_nextgen_live_issues.to_dict("records"),
            )
        ]
        queue_filter_key = "ops_nextgen_queue_filter"
        queue_filter_query = st.text_input(
            t("Queue Filter"),
            value=(
                session_state.get(queue_filter_key)
                if getattr(st, "session_state", None) is not None
                else ""
            )
            or "",
            key=queue_filter_key,
        )
        filtered_queue_rows = _filter_nextgen_queue_rows(
            query=queue_filter_query,
            rows=queue_rows,
        )
        if filtered_queue_rows:
            st.dataframe(
                display_df(pd.DataFrame(filtered_queue_rows)),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(t("No nextgen live queue rows match filter."))
        cycle_options = [
            int(item["autonomy_cycle_id"])
            for item in filtered_queue_rows
            if int(item.get("autonomy_cycle_id") or 0) > 0
        ]
        if cycle_options:
            queue_index = {
                int(item["autonomy_cycle_id"]): item
                for item in filtered_queue_rows
                if int(item.get("autonomy_cycle_id") or 0) > 0
            }
            session_state = getattr(st, "session_state", None)
            cycle_key = "ops_nextgen_cycle_id"
            selected_cycle_value = _coerce_selectbox_value(
                cycle_options,
                session_state.get(cycle_key) if session_state is not None else None,
                fallback=cycle_options[0],
            )
            if session_state is not None and session_state.get(cycle_key) != selected_cycle_value:
                session_state[cycle_key] = selected_cycle_value
            st.subheader(t("Nextgen Autonomy Cycle Details"))
            selected_cycle_id = st.selectbox(
                t("Inspect Nextgen Autonomy Cycle"),
                options=cycle_options,
                index=cycle_options.index(selected_cycle_value),
                format_func=lambda value: (
                    f"cycle={value} | "
                    f"{queue_index.get(int(value), {}).get('created_at', '')}"
                ),
                key=cycle_key,
            )
            cycle_row = query_one(
                "SELECT * FROM nextgen_autonomy_cycles WHERE id = ?",
                (int(selected_cycle_id),),
            )
            execution_rows = query_df(
                "SELECT * FROM nextgen_execution_directives "
                "WHERE autonomy_cycle_id = ? "
                "ORDER BY id DESC",
                (int(selected_cycle_id),),
            )
            repair_rows = query_df(
                "SELECT * FROM nextgen_repair_plans "
                "WHERE autonomy_cycle_id = ? "
                "ORDER BY priority DESC, id DESC",
                (int(selected_cycle_id),),
            )
            cycle_meta, cycle_notes, execution_df, repair_df = _build_nextgen_cycle_detail_view(
                cycle_row=cycle_row,
                execution_rows=execution_rows,
                repair_rows=repair_rows,
                load_json=load_json,
            )
            focus_meta, focused_execution_df, focused_repair_df = _build_nextgen_cycle_focus_view(
                cycle_notes=cycle_notes,
                execution_df=execution_df,
                repair_df=repair_df,
            )
            selected_queue = _build_nextgen_selected_queue_context(
                queue_index.get(int(selected_cycle_id), {})
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(t("Cycle ID"), str(cycle_meta["id"] or int(selected_cycle_id)))
            c2.metric(t("Execution Count"), str(cycle_meta["execution_count"]))
            c3.metric(t("Repair Count"), str(cycle_meta["repair_count"]))
            c4.metric(t("Strategy Count"), str(cycle_meta["strategy_count"]))
            st.caption(
                f"{t('Selected Queue Trigger')}: "
                f"{selected_queue.get('trigger') or t('none')} | "
                f"{t('Selected Queue Reason')}: "
                f"{selected_queue.get('reason') or t('none')}"
            )
            st.caption(
                f"{t('Selected Queue Latest Issue')}: "
                f"{selected_queue.get('latest_issue_event_type') or t('none')} | "
                f"{t('Selected Queue Latest Issue Reason')}: "
                f"{selected_queue.get('latest_issue_reason') or t('none')}"
            )
            cycle_filter_key = f"ops_nextgen_cycle_filter_{int(selected_cycle_id)}"
            cycle_filter_query = st.text_input(
                t("Cycle Runtime Filter"),
                value=(
                    session_state.get(cycle_filter_key)
                    if session_state is not None
                    else ""
                )
                or "",
                key=cycle_filter_key,
            )
            filtered_cycle_execution_df, filtered_cycle_repair_df = _filter_nextgen_cycle_detail_tables(
                query=cycle_filter_query,
                execution_df=execution_df,
                repair_df=repair_df,
            )
            st.subheader(t("Cycle Notes"))
            st.json(display_json(cycle_notes))
            focused_tab, full_tab = st.tabs([t("Focused"), t("Full")])
            with focused_tab:
                if focus_meta["runtime_ids"]:
                    st.subheader(t("Queue Focus"))
                    st.caption(
                        f"{t('Queue Focus Runtimes')}: "
                        f"{', '.join(focus_meta['runtime_ids'])}"
                    )
                    st.caption(
                        f"{t('Queue Focus Actions')}: "
                        f"{focus_meta['actions_text'] or t('none')} | "
                        f"{t('Queue Focus Priorities')}: "
                        f"{focus_meta['priorities_text'] or t('none')}"
                    )
                    focus_options = ("all", "execution", "repair")
                    focus_mode_key = f"ops_nextgen_focus_mode_{int(selected_cycle_id)}"
                    default_focus_mode = _default_nextgen_focus_mode(
                        execution_df=focused_execution_df,
                        repair_df=focused_repair_df,
                    )
                    selected_focus_mode = _coerce_selectbox_value(
                        focus_options,
                        session_state.get(focus_mode_key) if session_state is not None else None,
                        fallback=default_focus_mode,
                    )
                    if session_state is not None and session_state.get(focus_mode_key) != selected_focus_mode:
                        session_state[focus_mode_key] = selected_focus_mode
                    focus_mode = st.selectbox(
                        t("Focused View Mode"),
                        options=focus_options,
                        index=focus_options.index(selected_focus_mode),
                        format_func=lambda value: (
                            t("All Focus Rows")
                            if value == "all"
                            else (
                                t("Execution Only")
                                if value == "execution"
                                else t("Repair Only")
                            )
                        ),
                        key=f"ops_nextgen_focus_mode_{int(selected_cycle_id)}",
                    )
                    filtered_execution_df, filtered_repair_df = _filter_nextgen_focus_view(
                        mode=focus_mode,
                        execution_df=focused_execution_df,
                        repair_df=focused_repair_df,
                    )
                    filtered_execution_df, filtered_repair_df = _filter_nextgen_cycle_detail_tables(
                        query=cycle_filter_query,
                        execution_df=filtered_execution_df,
                        repair_df=filtered_repair_df,
                    )
                    focused_execution_view, hidden_execution_count = _limit_nextgen_focus_rows(
                        filtered_execution_df
                    )
                    focused_repair_view, hidden_repair_count = _limit_nextgen_focus_rows(
                        filtered_repair_df
                    )
                    st.subheader(t("Focused Execution Directives"))
                    if focused_execution_view.empty:
                        st.info(t("No focused execution directives for this cycle."))
                    else:
                        st.dataframe(display_df(focused_execution_view), use_container_width=True, hide_index=True)
                        if hidden_execution_count > 0:
                            st.caption(
                                f"{t('Additional Focused Execution Hidden')}: {hidden_execution_count}"
                            )
                    st.subheader(t("Focused Repair Plans"))
                    if focused_repair_view.empty:
                        st.info(t("No focused repair plans for this cycle."))
                    else:
                        st.dataframe(display_df(focused_repair_view), use_container_width=True, hide_index=True)
                        if hidden_repair_count > 0:
                            st.caption(
                                f"{t('Additional Focused Repair Hidden')}: {hidden_repair_count}"
                            )
                else:
                    st.info(t("No queue-focused rows for this cycle."))
            with full_tab:
                st.subheader(t("Execution Directives"))
                if filtered_cycle_execution_df.empty:
                    st.info(t("No execution directives for this cycle."))
                else:
                    st.dataframe(display_df(filtered_cycle_execution_df), use_container_width=True, hide_index=True)
                st.subheader(t("Repair Plans"))
                if filtered_cycle_repair_df.empty:
                    st.info(t("No repair plans for this cycle."))
                else:
                    st.dataframe(display_df(filtered_cycle_repair_df), use_container_width=True, hide_index=True)
        else:
            st.info(t("No nextgen autonomy cycle details."))
    st.subheader(t("Recent Scheduler Runs"))
    if recent_scheduler.empty:
        st.info(t("No scheduler runs."))
    else:
        st.dataframe(
            display_df(recent_scheduler[["job_name", "status", "started_at", "completed_at"]]),
            use_container_width=True,
        )
    st.subheader(t("Recent Reconciliation Runs"))
    if recent_reconciliation.empty:
        st.info(t("No reconciliation runs."))
    else:
        st.dataframe(
            display_df(recent_reconciliation[["status", "mismatch_count", "created_at"]]),
            use_container_width=True,
        )
        selected_run = st.selectbox(
            t("Inspect Reconciliation Run"),
            options=recent_reconciliation.index.tolist(),
            format_func=lambda idx: (
                f"{recent_reconciliation.loc[idx, 'created_at']} | "
                f"{display_value(recent_reconciliation.loc[idx, 'status'])} | "
                f"{t('Mismatch Count')}={recent_reconciliation.loc[idx, 'mismatch_count']}"
            ),
        )
        selected_row = recent_reconciliation.loc[selected_run].to_dict()
        details = load_json(selected_row.get("details_json"), {}) or {}
        c1, c2, c3 = st.columns(3)
        c1.metric(t("Status"), display_value(selected_row.get("status", "N/A")))
        c2.metric(t("Mismatch Count"), str(selected_row.get("mismatch_count", 0)))
        c3.metric(
            t("Mismatch Ratio"),
            f"{float(details.get('mismatch_ratio_pct', 0.0)):.2%}",
        )
        st.subheader(t("Reconciliation Details"))
        if details.get("missing_positions"):
            st.dataframe(
                display_df(pd.DataFrame({"missing_positions": details["missing_positions"]})),
                use_container_width=True,
                hide_index=True,
            )
        if details.get("missing_trades"):
            st.dataframe(
                display_df(pd.DataFrame({"missing_trades": details["missing_trades"]})),
                use_container_width=True,
                hide_index=True,
            )
        if details.get("quantity_mismatches"):
            st.dataframe(
                display_df(pd.DataFrame(details["quantity_mismatches"])),
                use_container_width=True,
                hide_index=True,
            )
        st.json(display_json(details))


def render_predictions_page(
    *,
    page_ctx=None,
    st=None,
    t=None,
    query_df=None,
    query_one=None,
    get_state_json=None,
    load_json=None,
    display_df=None,
    display_kv_rows=None,
    display_value=None,
    display_research_text=None,
    detect_summary_source=None,
    current_language=None,
    runtime_state_snapshot=None,
) -> None:
    st = _ctx_value(page_ctx, "st", st)
    t = _ctx_value(page_ctx, "t", t)
    query_df = _ctx_value(page_ctx, "query_df", query_df)
    query_one = _ctx_value(page_ctx, "query_one", query_one)
    get_state_json = _ctx_value(page_ctx, "get_state_json", get_state_json)
    load_json = _ctx_value(page_ctx, "load_json", load_json)
    display_df = _ctx_value(page_ctx, "display_df", display_df)
    display_kv_rows = _ctx_value(page_ctx, "display_kv_rows", display_kv_rows)
    display_research_text = _ctx_value(page_ctx, "display_research_text", display_research_text)
    detect_summary_source = _ctx_value(page_ctx, "detect_summary_source", detect_summary_source)
    current_language = _ctx_value(page_ctx, "current_language", current_language)
    runtime_state_snapshot = _ctx_value(page_ctx, "runtime_state_snapshot", runtime_state_snapshot)
    display_value = display_value or (lambda value: value)
    runtime_state = runtime_state_snapshot() if callable(runtime_state_snapshot) else None
    st.title(t("Predictions & Features"))

    predictions = query_df(
        "SELECT * FROM prediction_runs ORDER BY created_at DESC LIMIT 30"
    )
    prediction_evaluations = query_df(
        "SELECT symbol, timestamp, evaluation_type, is_correct, entry_close, future_close, "
        "metadata_json, created_at "
        "FROM prediction_evaluations ORDER BY created_at DESC LIMIT 20"
    )
    shadow_trades = query_df(
        "SELECT symbol, timestamp, block_reason, entry_price, exit_price, pnl_pct, status, created_at "
        "FROM shadow_trade_runs ORDER BY created_at DESC LIMIT 20"
    )
    features = query_df(
        "SELECT * FROM feature_snapshots ORDER BY created_at DESC LIMIT 10"
    )
    research_inputs = query_df(
        "SELECT * FROM research_inputs ORDER BY created_at DESC LIMIT 20"
    )
    shadow_symbols = (
        getattr(runtime_state, "shadow_observation_symbols", [])
        if runtime_state is not None
        else get_state_json("shadow_observation_symbols", []) or []
    )
    paper_exploration_symbols = (
        getattr(runtime_state, "paper_exploration_active_symbols", [])
        if runtime_state is not None
        else get_state_json("paper_exploration_active_symbols", []) or []
    )
    evaluation_summary = query_one(
        """
        SELECT
            SUM(CASE WHEN evaluation_type='execution' THEN 1 ELSE 0 END) AS execution_count,
            SUM(CASE WHEN evaluation_type='shadow_observation' THEN 1 ELSE 0 END) AS shadow_count,
            AVG(CASE WHEN evaluation_type='execution' THEN is_correct END) AS execution_accuracy,
            AVG(CASE WHEN evaluation_type='shadow_observation' THEN is_correct END) AS shadow_accuracy,
            AVG(
                CASE WHEN evaluation_type='execution'
                    THEN CAST(json_extract(metadata_json, '$.trade_net_return_pct') AS REAL)
                END
            ) AS execution_expectancy_pct,
            AVG(
                CASE WHEN evaluation_type='shadow_observation'
                    THEN CAST(json_extract(metadata_json, '$.trade_net_return_pct') AS REAL)
                END
            ) AS shadow_expectancy_pct
        FROM prediction_evaluations
        """
    ) or {}
    shadow_trade_summary = query_one(
        """
        SELECT
            SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS open_count,
            SUM(CASE WHEN status='evaluated' THEN 1 ELSE 0 END) AS evaluated_count,
            AVG(CASE WHEN status='evaluated' THEN pnl_pct END) AS avg_pnl_pct
        FROM shadow_trade_runs
        """
    ) or {}
    paper_canary_summary = query_one(
        """
        SELECT
            COUNT(*) AS total_count,
            SUM(
                CASE
                    WHEN json_extract(payload_json, '$.canary_mode')='soft_review'
                    THEN 1 ELSE 0
                END
            ) AS soft_count
        FROM execution_events
        WHERE event_type='paper_canary_open'
        """
    ) or {}
    prediction_summary = build_predictions_summary_view_model(
        shadow_symbols=shadow_symbols,
        paper_exploration_symbols=paper_exploration_symbols,
        evaluation_summary=evaluation_summary,
        shadow_trade_summary=shadow_trade_summary,
        paper_canary_summary=paper_canary_summary,
    )
    latest_feature = None
    latest_feature_values = {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        t("Observation Pool"),
        str(prediction_summary["metrics"]["observation_pool_count"]),
    )
    c2.metric(
        t("Execution Evaluations"),
        str(prediction_summary["metrics"]["execution_evaluation_count"]),
    )
    c3.metric(
        t("Shadow Evaluations"),
        str(prediction_summary["metrics"]["shadow_evaluation_count"]),
    )
    c4.metric(
        t("Open Shadow Trades"),
        str(prediction_summary["metrics"]["open_shadow_trade_count"]),
    )
    c10, c11, c12 = st.columns(3)
    c10.metric(
        t("Paper Canary Opens"),
        str(prediction_summary["metrics"]["paper_canary_open_count"]),
    )
    c11.metric(
        t("Soft Canary Opens"),
        str(prediction_summary["metrics"]["soft_canary_open_count"]),
    )
    c12.metric(
        t("Exploration Slots"),
        str(prediction_summary["metrics"]["exploration_slot_count"]),
    )
    c5, c6, c7, c8, c9 = st.columns(5)
    c5.metric(
        t("Execution Accuracy"),
        (
            f"{float(evaluation_summary['execution_accuracy']) * 100:.2f}%"
            if prediction_summary["metrics"]["execution_accuracy"] is not None
            else display_value("N/A")
        ),
    )
    c6.metric(
        t("Shadow Accuracy"),
        (
            f"{float(prediction_summary['metrics']['shadow_accuracy']) * 100:.2f}%"
            if prediction_summary["metrics"]["shadow_accuracy"] is not None
            else display_value("N/A")
        ),
    )
    c7.metric(
        t("Execution Expectancy"),
        (
            f"{float(prediction_summary['metrics']['execution_expectancy_pct']):.2f}%"
            if prediction_summary["metrics"]["execution_expectancy_pct"] is not None
            else display_value("N/A")
        ),
    )
    c8.metric(
        t("Shadow Expectancy"),
        (
            f"{float(prediction_summary['metrics']['shadow_expectancy_pct']):.2f}%"
            if prediction_summary["metrics"]["shadow_expectancy_pct"] is not None
            else display_value("N/A")
        ),
    )
    c9.metric(
        t("Shadow Avg PnL"),
        (
            f"{float(prediction_summary['metrics']['shadow_avg_pnl_pct']):.2f}%"
            if prediction_summary["metrics"]["shadow_avg_pnl_pct"] is not None
            else display_value("N/A")
        ),
    )
    st.caption(
        f"{t('Current Observation Pool')}: "
        + (", ".join(shadow_symbols) if shadow_symbols else t("none"))
    )
    if paper_exploration_symbols:
        st.caption(
            f"{t('Paper Exploration Slot')}: " + ", ".join(paper_exploration_symbols)
        )

    st.subheader(t("Recent Prediction Runs"))
    if predictions.empty:
        st.info(t("No prediction runs recorded."))
    else:
        decision_payloads = predictions["decision_json"].apply(
            lambda value: load_json(value, {}) or {}
        )
        prediction_overview = predictions[
            [
                "id",
                "symbol",
                "timestamp",
                "model_version",
                "up_probability",
                "feature_count",
                "created_at",
            ]
        ].copy()
        prediction_overview["pipeline_mode"] = decision_payloads.apply(
            lambda payload: payload.get("pipeline_mode", "execution")
        )
        st.dataframe(display_df(prediction_overview), use_container_width=True)
        latest_symbol_predictions = (
            predictions.sort_values(["symbol", "created_at"], ascending=[True, False])
            .drop_duplicates(subset=["symbol"], keep="first")
            .sort_values("created_at", ascending=False)
        )
        selected_prediction = st.selectbox(
            t("Inspect Prediction Run"),
            options=latest_symbol_predictions.index.tolist(),
            format_func=lambda idx: (
                f"{latest_symbol_predictions.loc[idx, 'symbol']} | "
                f"{latest_symbol_predictions.loc[idx, 'timestamp']} | "
                f"p={float(latest_symbol_predictions.loc[idx, 'up_probability']):.2%}"
            ),
        )
        selected_row = predictions.loc[selected_prediction].to_dict()
        research = load_json(selected_row.get("research_json"), {}) or {}
        decision = load_json(selected_row.get("decision_json"), {}) or {}
        matching_feature = (
            features[features["symbol"] == selected_row.get("symbol", "")]
            if not features.empty
            else pd.DataFrame()
        )
        latest_feature = (
            matching_feature.iloc[0].to_dict()
            if not matching_feature.empty
            else (features.iloc[0].to_dict() if not features.empty else None)
        )
        latest_feature_values = (
            json.loads(latest_feature["features_json"]) if latest_feature else {}
        )
        latest_research = {
            "symbol": selected_row.get("symbol", ""),
            "timestamp": research.get("timestamp", selected_row.get("timestamp", "")),
            "news_summary": research.get("news_summary", ""),
            "macro_summary": research.get("macro_summary", ""),
            "onchain_summary": research.get("onchain_summary", ""),
            "fear_greed": display_value(research.get("fear_greed", "N/A")),
        }
        if not latest_research["news_summary"] and not research_inputs.empty:
            matching_research = research_inputs[
                research_inputs["symbol"] == selected_row.get("symbol", "")
            ]
            if not matching_research.empty:
                latest_research = matching_research.iloc[0].to_dict()

        st.subheader(t("LLM Decision Chain"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("Symbol"), display_value(selected_row.get("symbol", "N/A")))
        c2.metric(
            t("Up Probability"),
            f"{float(selected_row.get('up_probability', 0.0)):.2%}",
        )
        c3.metric(t("Suggested Action"), display_value(research.get("suggested_action", "HOLD")))
        c4.metric(t("Final Score"), f"{float(decision.get('final_score', 0.0)):.2f}")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric(
            t("LLM Confidence"),
            f"{float(research.get('confidence', 0.0)):.2f}",
        )
        c6.metric(t("Market Regime"), display_value(research.get("market_regime", "UNKNOWN")))
        c7.metric(
            t("XGB Threshold"),
            f"{float(decision.get('xgboost_threshold', 0.0)):.2f}",
        )
        c8.metric(
            t("Final Threshold"),
            f"{float(decision.get('final_score_threshold', 0.0)):.2f}",
        )

        left, right = st.columns(2)
        with left:
            st.write(t("Key Reasons"))
            reasons = research.get("key_reason") or []
            if reasons:
                for reason in reasons:
                    st.write(f"- {display_value(reason)}")
            else:
                st.info(t("No key reasons recorded."))
            st.write(t("Risk Warnings"))
            warnings = research.get("risk_warning") or []
            if warnings:
                for warning in warnings:
                    st.write(f"- {display_research_text(str(display_value(warning)))}")
            else:
                st.info(t("No risk warnings recorded."))
        with right:
            st.write(t("Decision Runtime Parameters"))
            decision_rows = display_kv_rows(
                decision,
                preferred_keys=[
                    "final_score",
                    "regime",
                    "suggested_action",
                    "xgboost_threshold",
                    "final_score_threshold",
                    "min_liquidity_ratio",
                    "sentiment_weight",
                    "fixed_stop_loss_pct",
                    "take_profit_levels",
                ],
            )
            if decision_rows.empty:
                st.info(t("No data."))
            else:
                st.dataframe(decision_rows, use_container_width=True, hide_index=True)
            raw_content = research.get("raw_content", "")
            if raw_content:
                st.text_area(t("LLM Raw Content"), value=raw_content, height=180)

    if latest_feature_values:
        st.subheader(t("Latest Intelligence Factors"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            t("LLM Sentiment"),
            f"{latest_feature_values.get('llm_sentiment_score', 0.0):.2f}",
        )
        c2.metric(
            t("Onchain Netflow"),
            f"{latest_feature_values.get('onchain_netflow_score', 0.0):.4f}",
        )
        c3.metric(
            t("Onchain Whale"),
            f"{latest_feature_values.get('onchain_whale_score', 0.0):.4f}",
        )
        c4.metric(
            t("Black Swan Index"),
            f"{latest_feature_values.get('black_swan_index', 0.0):.2f}",
        )
        c5, c6, c7 = st.columns(3)
        c5.metric(
            t("Trend Strength"),
            f"{latest_feature_values.get('trend_strength_4h', 0.0):.2f}",
        )
        c6.metric(
            t("ATR Percentile"),
            f"{latest_feature_values.get('atr_percentile_4h', 0.0):.2f}",
        )
        c7.metric(
            t("Price Structure"),
            f"{latest_feature_values.get('price_structure_score', 0.0):.2f}",
        )

        if latest_research:
            st.subheader(t("Latest Research Digest"))
            news_source, news_fallback = detect_summary_source(
                latest_research.get("news_summary", ""),
                "news",
            )
            onchain_source, onchain_fallback = detect_summary_source(
                latest_research.get("onchain_summary", ""),
                "onchain",
            )
            left, right = st.columns(2)
            research_stamp = str(latest_research.get("timestamp", "latest"))
            with left:
                st.caption(
                    (f"{t('Source')}: " if current_language() == "zh" else "Source: ")
                    + str(display_value(news_source))
                    + (
                        " | 回退摘要"
                        if current_language() == "zh" and news_fallback
                        else " | fallback"
                        if news_fallback
                        else ""
                    )
                )
                st.text_area(
                    t("News Summary"),
                    value=display_research_text(latest_research.get("news_summary", "")),
                    height=140,
                    key=f"latest_news_summary_{research_stamp}",
                    disabled=True,
                )
                st.caption(
                    (f"{t('Source')}: " if current_language() == "zh" else "Source: ")
                    + t("Macro")
                )
                st.text_area(
                    t("Macro Summary"),
                    value=display_research_text(latest_research.get("macro_summary", "")),
                    height=140,
                    key=f"latest_macro_summary_{research_stamp}",
                    disabled=True,
                )
            with right:
                st.caption(
                    (f"{t('Source')}: " if current_language() == "zh" else "Source: ")
                    + str(display_value(onchain_source))
                    + (
                        " | 回退摘要"
                        if current_language() == "zh" and onchain_fallback
                        else " | fallback"
                        if onchain_fallback
                        else ""
                    )
                )
                st.text_area(
                    t("Onchain Summary"),
                    value=display_research_text(latest_research.get("onchain_summary", "")),
                    height=140,
                    key=f"latest_onchain_summary_{research_stamp}",
                    disabled=True,
                )
                st.metric(
                    t("Fear & Greed"),
                    str(display_value(latest_research.get("fear_greed", "N/A"))),
                )

    review_events = query_df(
        "SELECT symbol, payload_json, created_at FROM execution_events "
        "WHERE event_type='research_review' ORDER BY created_at DESC LIMIT 12"
    )
    if not review_events.empty:
        payloads = review_events["payload_json"].apply(lambda value: load_json(value, {}) or {})
        review_events["reviewed_action"] = payloads.apply(lambda payload: payload.get("reviewed_action", ""))
        review_events["review_score"] = payloads.apply(lambda payload: payload.get("review_score", ""))
        review_events["reasons"] = payloads.apply(
            lambda payload: ", ".join(payload.get("reasons", [])[:4])
        )
        st.subheader(t("Recent Research Reviews"))
        st.dataframe(
            display_df(review_events[["symbol", "reviewed_action", "review_score", "reasons", "created_at"]]),
            use_container_width=True,
        )

    st.subheader(t("Recent Prediction Evaluations"))
    if prediction_evaluations.empty:
        st.info(t("No prediction evaluations recorded."))
    else:
        evaluation_payloads = prediction_evaluations["metadata_json"].apply(
            lambda value: load_json(value, {}) or {}
        )
        prediction_evaluations["trade_net_return_pct"] = evaluation_payloads.apply(
            lambda payload: payload.get("trade_net_return_pct", "")
        )
        prediction_evaluations["opportunity_return_pct"] = evaluation_payloads.apply(
            lambda payload: payload.get("opportunity_return_pct", "")
        )
        prediction_evaluations["favorable_excursion_pct"] = evaluation_payloads.apply(
            lambda payload: payload.get("favorable_excursion_pct", "")
        )
        prediction_evaluations["adverse_excursion_pct"] = evaluation_payloads.apply(
            lambda payload: payload.get("adverse_excursion_pct", "")
        )
        prediction_evaluations["is_correct"] = prediction_evaluations["is_correct"].apply(
            lambda value: "yes" if int(value or 0) else "no"
        )
        st.dataframe(
            display_df(
                prediction_evaluations[
                    [
                        "symbol",
                        "timestamp",
                        "evaluation_type",
                        "is_correct",
                        "entry_close",
                        "future_close",
                        "trade_net_return_pct",
                        "opportunity_return_pct",
                        "favorable_excursion_pct",
                        "adverse_excursion_pct",
                        "created_at",
                    ]
                ]
            ),
            use_container_width=True,
        )

    st.subheader(t("Blocked Shadow Trades"))
    if shadow_trades.empty:
        st.info(t("No blocked shadow trades recorded."))
    else:
        st.dataframe(
            display_df(
                shadow_trades[
                    [
                        "symbol",
                        "timestamp",
                        "block_reason",
                        "entry_price",
                        "exit_price",
                        "pnl_pct",
                        "status",
                        "created_at",
                    ]
                ]
            ),
            use_container_width=True,
        )


def render_watchlist_page(
    *,
    page_ctx=None,
    st=None,
    t=None,
    get_state_json=None,
    set_state_json=None,
    parse_symbol_text=None,
    run_command=None,
    query_df=None,
    load_json=None,
    parse_event_payloads,
    display_df=None,
    display_value=None,
    get_settings_fn=None,
    watchlist_snapshot_key,
    execution_symbols_key,
    watchlist_whitelist_key,
    watchlist_blacklist_key,
    runtime_state_snapshot=None,
) -> None:
    st = _ctx_value(page_ctx, "st", st)
    t = _ctx_value(page_ctx, "t", t)
    get_state_json = _ctx_value(page_ctx, "get_state_json", get_state_json)
    set_state_json = _ctx_value(page_ctx, "set_state_json", set_state_json)
    parse_symbol_text = _ctx_value(page_ctx, "parse_symbol_text", parse_symbol_text)
    run_command = _ctx_value(page_ctx, "run_command", run_command)
    query_df = _ctx_value(page_ctx, "query_df", query_df)
    load_json = _ctx_value(page_ctx, "load_json", load_json)
    display_df = _ctx_value(page_ctx, "display_df", display_df)
    get_settings_fn = _ctx_value(page_ctx, "get_settings_fn", get_settings_fn)
    runtime_state_snapshot = _ctx_value(page_ctx, "runtime_state_snapshot", runtime_state_snapshot)
    display_value = display_value or (lambda value: value)
    st.title(t("Dynamic Watchlist"))

    runtime_state = runtime_state_snapshot() if callable(runtime_state_snapshot) else None
    snapshot = get_state_json(watchlist_snapshot_key, {}) or {}
    view_model = build_watchlist_view_model(
        snapshot=snapshot,
        active_symbols=(
        snapshot.get("active_symbols")
        or (
            getattr(runtime_state, "active_symbols", [])
            if runtime_state is not None
            else []
        )
        or get_state_json("active_symbols", [])
        or []
        ),
        execution_symbols=(
        snapshot.get("execution_symbols")
        or (
            getattr(runtime_state, "execution_symbols", [])
            if runtime_state is not None
            else []
        )
        or get_state_json(execution_symbols_key, [])
        or []
        ),
        model_ready_symbols=(
        snapshot.get("model_ready_symbols")
        or (
            getattr(runtime_state, "model_ready_symbols", [])
            if runtime_state is not None
            else []
        )
        or get_state_json("model_ready_symbols", [])
        or []
        ),
        consistency_blocked_symbols=(
        getattr(runtime_state, "consistency_blocked_symbols", [])
        if runtime_state is not None
        else get_state_json("consistency_blocked_symbols", []) or []
        ),
        consistency_blocked_details=(
        getattr(runtime_state, "consistency_blocked_details", {})
        if runtime_state is not None
        else get_state_json("consistency_blocked_details", {}) or {}
        ),
        whitelist=(
        snapshot.get("whitelist")
        or (
            getattr(runtime_state, "watchlist_whitelist", [])
            if runtime_state is not None
            else []
        )
        or get_state_json(watchlist_whitelist_key, [])
        or []
        ),
        blacklist=(
        snapshot.get("blacklist")
        or (
            getattr(runtime_state, "watchlist_blacklist", [])
            if runtime_state is not None
            else []
        )
        or get_state_json(watchlist_blacklist_key, [])
        or []
        ),
    )
    active_symbols = view_model["active_symbols"]
    raw_active_symbols = view_model["raw_active_symbols"]
    execution_symbols = view_model["execution_symbols"]
    model_ready_symbols = view_model["model_ready_symbols"]
    consistency_blocked_symbols = view_model["consistency_blocked_symbols"]
    consistency_blocked_details = view_model["consistency_blocked_details"]
    added_symbols = view_model["added_symbols"]
    removed_symbols = view_model["removed_symbols"]
    candidates = pd.DataFrame(view_model["candidates"])
    refreshed_at = view_model["refreshed_at"]
    refresh_reason = view_model["refresh_reason"]
    whitelist = view_model["whitelist"]
    blacklist = view_model["blacklist"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("Active Symbols"), str(len(active_symbols)))
    c2.metric(t("Execution Pool"), str(len(execution_symbols)))
    c3.metric(t("Model-Ready Pool"), str(len(model_ready_symbols)))
    c4.metric(t("Last Refresh"), refreshed_at or display_value("N/A"))

    if refresh_reason:
        st.caption(f"{t('Refresh Reason')}: {display_value(refresh_reason)}")

    st.subheader(t("Current Execution Pool"))
    if execution_symbols:
        st.write(", ".join(execution_symbols))
    else:
        st.info(t("No active watchlist is available."))

    st.subheader(t("Current Model-Ready Pool"))
    if active_symbols:
        st.write(", ".join(active_symbols))
    else:
        st.info(t("No active watchlist is available."))

    st.subheader(t("Consistency-Blocked Symbols"))
    if consistency_blocked_symbols:
        blocked_rows = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "consistency_flags": ", ".join(
                        list((consistency_blocked_details.get(symbol) or []))
                    ),
                }
                for symbol in consistency_blocked_symbols
            ]
        )
        st.dataframe(display_df(blocked_rows), use_container_width=True, hide_index=True)
    else:
        st.info(t("No consistency-blocked symbols."))

    st.subheader(t("Current Observation Pool"))
    if raw_active_symbols:
        st.write(", ".join(raw_active_symbols))
    else:
        st.info(t("No active watchlist is available."))

    left, right = st.columns(2)
    with left:
        st.subheader(t("Added Symbols"))
        st.write(", ".join(added_symbols) if added_symbols else t("none"))
    with right:
        st.subheader(t("Removed Symbols"))
        st.write(", ".join(removed_symbols) if removed_symbols else t("none"))

    st.subheader(t("Execution Pool"))
    with st.form("execution_pool_form"):
        execution_text = st.text_area(
            t("Execution Pool"),
            value=", ".join(execution_symbols),
            help=t("Execution symbols, comma-separated."),
        )
        execution_set = st.form_submit_button(t("Apply Execution Pool"), type="primary")

    if execution_set:
        symbols = parse_symbol_text(execution_text)
        ok, output = run_command("execution-set", ",".join(symbols))
        st.code(output or t("(no output)"))
        if ok:
            st.success(t("Execution pool updated."))
            st.rerun()
        else:
            st.error(t("Execution pool update failed."))

    add_col, remove_col = st.columns(2)
    with add_col:
        with st.form("execution_add_form"):
            add_text = st.text_area(
                t("Add To Execution Pool"),
                value="",
                help=t("Symbols to add into the execution pool, comma-separated."),
            )
            add_submit = st.form_submit_button(t("Add To Execution Pool"))
        if add_submit:
            symbols = parse_symbol_text(add_text)
            ok, output = run_command("execution-add", ",".join(symbols))
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Execution pool addition finished."))
                st.rerun()
            else:
                st.error(t("Execution pool addition failed."))
    with remove_col:
        with st.form("execution_remove_form"):
            remove_text = st.text_area(
                t("Remove From Execution Pool"),
                value="",
                help=t("Symbols to remove from the execution pool, comma-separated."),
            )
            remove_submit = st.form_submit_button(t("Remove From Execution Pool"))
        if remove_submit:
            symbols = parse_symbol_text(remove_text)
            ok, output = run_command("execution-remove", ",".join(symbols))
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Execution pool removal finished."))
                st.rerun()
            else:
                st.error(t("Execution pool removal failed."))

    st.subheader(t("Manual Overrides"))
    with st.form("watchlist_manual_form"):
        whitelist_text = st.text_area(
            t("Whitelist"),
            value=", ".join(whitelist),
            help=t("Force include symbols, comma-separated."),
        )
        blacklist_text = st.text_area(
            t("Blacklist"),
            value=", ".join(blacklist),
            help=t("Force exclude symbols, comma-separated."),
        )
        save_lists = st.form_submit_button(t("Save Lists"), type="primary")

    if save_lists:
        set_state_json(watchlist_whitelist_key, parse_symbol_text(whitelist_text))
        set_state_json(watchlist_blacklist_key, parse_symbol_text(blacklist_text))
        st.success(t("Whitelist/blacklist saved."))
        st.rerun()

    if st.button(t("Refresh Watchlist Now"), use_container_width=True):
        ok, output = run_command("watchlist-refresh")
        st.code(output or t("(no output)"))
        if ok:
            st.success(t("Watchlist refreshed."))
            st.rerun()
        else:
            st.error(t("Watchlist refresh failed."))

    st.subheader(t("Execution Readiness"))
    readiness_rows = []
    training_runs = query_df(
        "SELECT symbol, metadata_json, created_at FROM training_runs ORDER BY created_at DESC LIMIT 200"
    )
    latest_training_by_symbol = {}
    if not training_runs.empty:
        for _, row in training_runs.iterrows():
            symbol = row["symbol"]
            if symbol in latest_training_by_symbol:
                continue
            latest_training_by_symbol[symbol] = load_json(row["metadata_json"], {}) or {}
    observation_symbols = list(dict.fromkeys((raw_active_symbols or []) + (execution_symbols or [])))
    for symbol in observation_symbols:
        metadata = latest_training_by_symbol.get(symbol, {})
        rows = int(metadata.get("rows") or 0)
        model_path = metadata.get("active_model_path") or metadata.get("model_path") or ""
        model_exists = bool(model_path) and Path(model_path).exists()
        in_execution_pool = symbol in execution_symbols
        model_ready = symbol in model_ready_symbols
        consistency_flags = list((consistency_blocked_details.get(symbol) or []))
        if model_ready:
            status_reason = t("ready")
        elif consistency_flags:
            status_reason = t("consistency_blocked")
        elif not in_execution_pool:
            status_reason = t("not_in_execution_pool")
        elif not metadata:
            status_reason = t("not_trained")
        elif not metadata.get("trained_with_xgboost"):
            status_reason = t("training_failed_or_disabled")
        elif rows < get_settings_fn().training.minimum_training_rows:
            status_reason = t("insufficient_rows")
        elif not model_exists:
            status_reason = t("model_file_missing")
        else:
            status_reason = t("training_failed_or_disabled")
        readiness_rows.append(
            {
                "symbol": symbol,
                "in_execution_pool": in_execution_pool,
                "model_ready": model_ready,
                "training_reason": metadata.get("reason", ""),
                "training_rows": rows,
                "holdout_accuracy": metadata.get("holdout_accuracy"),
                "consistency_flags": ", ".join(consistency_flags),
                "status_reason": status_reason,
            }
        )
    readiness_df = pd.DataFrame(readiness_rows)
    if readiness_df.empty:
        st.info(t("No execution readiness data."))
    else:
        st.dataframe(display_df(readiness_df), use_container_width=True)

    st.subheader(t("Execution Pool Changes"))
    pool_events = query_df(
        "SELECT event_type, symbol, payload_json, created_at FROM execution_events "
        "WHERE event_type='execution_pool_update' ORDER BY created_at DESC LIMIT 20"
    )
    pool_events = parse_event_payloads(pool_events)
    if pool_events.empty:
        st.info(t("No execution pool changes recorded."))
    else:
        payloads = pool_events["payload_json"].apply(lambda value: load_json(value, {}) or {})
        pool_events["action"] = payloads.apply(lambda payload: payload.get("action", ""))
        pool_events["added_symbols"] = payloads.apply(
            lambda payload: ", ".join(payload.get("added_symbols", []))
        )
        pool_events["removed_symbols"] = payloads.apply(
            lambda payload: ", ".join(payload.get("removed_symbols", []))
        )
        pool_events["execution_symbols"] = payloads.apply(
            lambda payload: ", ".join(payload.get("execution_symbols", []))
        )
        st.dataframe(
            display_df(
                pool_events[
                    [
                        "created_at",
                        "action",
                        "added_symbols",
                        "removed_symbols",
                        "execution_symbols",
                    ]
                ]
            ),
            use_container_width=True,
        )

    st.subheader(t("Candidate Scores"))
    if candidates.empty:
        st.info(t("No candidate scoring available. Refresh the watchlist first."))
    else:
        if "notes" in candidates.columns:
            candidates["notes"] = candidates["notes"].apply(
                lambda value: "; ".join(value) if isinstance(value, list) else value
            )
        preferred = [
            "symbol",
            "sector",
            "score",
            "quote_volume_24h",
            "change_pct_24h",
            "is_core",
            "source",
            "notes",
        ]
        visible_columns = [column for column in preferred if column in candidates.columns]
        st.dataframe(display_df(candidates[visible_columns]), use_container_width=True)


def render_training_page(
    *,
    st,
    t,
    is_low_resource_mode,
    query_df,
    run_command,
    display_df,
    display_json,
    get_settings_fn,
) -> None:
    st.title(t("Training"))
    low_resource_mode = is_low_resource_mode()

    training_runs = query_df(
        "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT 20"
    )

    col_run, col_symbol = st.columns([1, 2])
    with col_symbol:
        st.selectbox(
            t("Symbol"),
            options=get_settings_fn().exchange.symbols,
            index=0,
        )
    with col_run:
        if st.button(
            t("Run Training"),
            type="primary",
            use_container_width=True,
            disabled=low_resource_mode,
        ):
            ok, output = run_command("train")
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Training finished."))
            else:
                st.error(t("Training failed."))
    if low_resource_mode:
        st.warning(t("Low resource mode is enabled. Training is disabled in the dashboard on this node."))

    if not training_runs.empty:
        st.subheader(t("Recent Training Runs"))
        st.dataframe(display_df(training_runs), use_container_width=True)

        latest = training_runs.iloc[0].to_dict()
        st.subheader(t("Latest Training Metadata"))
        st.json(display_json(json.loads(latest["metadata_json"])))
    else:
        st.info(t("No training runs recorded."))


def render_walkforward_page(
    *,
    st,
    t,
    is_low_resource_mode,
    run_command,
    query_df,
    display_df,
    get_settings_fn,
) -> None:
    st.title(t("Walk-Forward Evaluation"))
    low_resource_mode = is_low_resource_mode()

    symbol = st.selectbox(
        t("Symbol"),
        options=get_settings_fn().exchange.symbols,
        index=0,
        key="wf_symbol",
    )
    if st.button(
        t("Run Walk-Forward"),
        type="primary",
        use_container_width=True,
        disabled=low_resource_mode,
    ):
        ok, output = run_command("walkforward", symbol)
        st.code(output or t("(no output)"))
        if ok:
            st.success(t("Walk-forward finished."))
        else:
            st.error(t("Walk-forward failed."))
    if low_resource_mode:
        st.warning(t("Low resource mode is enabled. Walk-forward is disabled in the dashboard on this node."))

    runs = query_df(
        "SELECT * FROM walkforward_runs ORDER BY created_at DESC LIMIT 20"
    )
    if runs.empty:
        st.info(t("No walk-forward runs recorded."))
    else:
        st.subheader(t("Recent Walk-Forward Runs"))
        st.dataframe(display_df(runs), use_container_width=True)


def render_backtest_page(
    *,
    st,
    t,
    is_low_resource_mode,
    run_command,
    query_df,
    display_df,
    get_settings_fn,
) -> None:
    st.title(t("V3 Backtest"))
    low_resource_mode = is_low_resource_mode()
    symbol = st.selectbox(
        t("Backtest Symbol"),
        options=get_settings_fn().exchange.symbols,
        index=0,
        key="bt_symbol",
    )
    if st.button(
        t("Run Backtest"),
        type="primary",
        use_container_width=True,
        disabled=low_resource_mode,
    ):
        ok, output = run_command("backtest", symbol)
        st.code(output or t("(no output)"))
        if ok:
            st.success(t("Backtest finished."))
        else:
            st.error(t("Backtest failed."))
    if low_resource_mode:
        st.warning(t("Low resource mode is enabled. Backtest is disabled in the dashboard on this node."))

    runs = query_df(
        "SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT 20"
    )
    if runs.empty:
        st.info(t("No backtest runs recorded."))
    else:
        st.dataframe(display_df(runs), use_container_width=True)


def render_reports_page(
    *,
    st,
    t,
    run_command,
    query_df,
    query_one,
    display_df,
    display_json,
    load_json,
    localize_report_text=None,
) -> None:
    localize_report_text = localize_report_text or (lambda text: text)
    st.title(t("Reports"))
    if st.button(t("Generate Reports"), type="primary", use_container_width=True):
        ok, output = run_command("report")
        st.code(output or t("(no output)"))
        if not ok:
            st.error(t("Report generation failed."))

    latest_training = query_one(
        "SELECT metadata_json FROM training_runs ORDER BY created_at DESC LIMIT 1"
    )
    latest_walkforward = query_one(
        "SELECT summary_json, splits_json FROM walkforward_runs ORDER BY created_at DESC LIMIT 1"
    )
    latest_execution = query_df(
        "SELECT * FROM execution_events ORDER BY created_at DESC LIMIT 20"
    )
    latest_drift = query_one(
        "SELECT * FROM report_artifacts WHERE report_type='drift' ORDER BY created_at DESC LIMIT 1"
    )
    latest_ab_test = query_one(
        "SELECT * FROM report_artifacts WHERE report_type='ab_test' ORDER BY created_at DESC LIMIT 1"
    )
    report_artifacts = query_df(
        "SELECT * FROM report_artifacts ORDER BY created_at DESC LIMIT 20"
    )

    left, right = st.columns(2)
    with left:
        st.subheader(t("Latest Training Metadata"))
        if latest_training:
            st.json(display_json(load_json(latest_training["metadata_json"], {})))
        else:
            st.info(t("No training metadata."))

    with right:
        st.subheader(t("Latest Walk-Forward Summary"))
        if latest_walkforward:
            st.json(display_json(load_json(latest_walkforward["summary_json"], {})))
        else:
            st.info(t("No walk-forward summary."))

    if latest_drift:
        st.subheader(t("Latest Drift Report"))
        st.code(localize_report_text(latest_drift["content"]))
    if latest_ab_test:
        st.subheader(t("Latest AB Test Report"))
        st.code(localize_report_text(latest_ab_test["content"]))

    st.subheader(t("Latest Execution Events"))
    if latest_execution.empty:
        st.info(t("No execution events."))
    else:
        st.dataframe(display_df(latest_execution), use_container_width=True)

    st.subheader(t("Report Artifacts"))
    if report_artifacts.empty:
        st.info(t("No report artifacts."))
    else:
        st.dataframe(
            display_df(report_artifacts[["report_type", "symbol", "created_at"]]),
            use_container_width=True,
        )
        selected = st.selectbox(
            t("View Artifact"),
            options=report_artifacts.index.tolist(),
            format_func=lambda idx: (
                f"{display_df(report_artifacts.loc[[idx], ['report_type']]).iloc[0, 0]} | "
                f"{report_artifacts.loc[idx, 'symbol'] or t('ALL')} | "
                f"{report_artifacts.loc[idx, 'created_at']}"
            ),
        )
        st.code(localize_report_text(report_artifacts.loc[selected, "content"]))
