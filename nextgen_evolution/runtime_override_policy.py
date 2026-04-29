"""Shared runtime deployment policy helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .config import EvolutionConfig
from .models import ExecutionAction, ExecutionDirective, RuntimeEvidenceSnapshot, RuntimeState


_LEGACY_POLICY_KEYS = {
    "repair_reentry",
    "runtime_overrides",
    "runtime_override_state",
    "staged_exit_state",
    "reentry_state",
}


def repair_runtime_overrides_from_notes(payload: dict) -> dict[str, float]:
    raw = dict(payload or {}).get("runtime_overrides") or {}
    overrides: dict[str, float] = {}
    for key, value in dict(raw).items():
        text = str(key).strip()
        if not text:
            continue
        try:
            overrides[text] = float(value)
        except (TypeError, ValueError):
            continue
    return overrides


def decay_runtime_overrides(
    runtime_overrides: dict[str, float],
    *,
    config: EvolutionConfig,
) -> dict[str, float]:
    if not runtime_overrides:
        return {}
    rate = min(max(float(config.autonomy_runtime_override_decay_rate), 0.0), 1.0)
    cooldown_step = max(float(config.autonomy_runtime_override_cooldown_decay_step), 0.0)
    decayed: dict[str, float] = {}
    for key, value in runtime_overrides.items():
        name = str(key).strip()
        current = float(value)
        if name == "max_weight_multiplier":
            updated = current + (1.0 - current) * rate if current < 1.0 else current
            if updated < 0.999:
                decayed[name] = round(updated, 4)
        elif name == "take_profit_bias":
            updated = current + (1.0 - current) * rate
            if abs(updated - 1.0) > 1e-3:
                decayed[name] = round(updated, 4)
        elif name == "entry_cooldown_bars_multiplier":
            updated = max(0.0, current - cooldown_step)
            if updated > 1e-6:
                decayed[name] = round(updated, 4)
        else:
            decayed[name] = current
    return decayed


def runtime_override_recovery_policy(
    *,
    config: EvolutionConfig,
    runtime_evidence: RuntimeEvidenceSnapshot | None,
) -> tuple[str, int]:
    if runtime_evidence is None:
        return "neutral", 1
    health_status = str(runtime_evidence.health_status or "").strip().lower()
    total_net_pnl = float(runtime_evidence.total_net_pnl or 0.0)
    current_drawdown_pct = float(runtime_evidence.current_drawdown_pct or 0.0)
    closed_trade_count = int(runtime_evidence.closed_trade_count or 0)
    win_rate = float(runtime_evidence.win_rate or 0.0)
    consecutive_losses = int(runtime_evidence.consecutive_losses or 0)
    unrealized_pnl = float(runtime_evidence.unrealized_pnl or 0.0)
    if (
        health_status in {"degraded", "failing"}
        or total_net_pnl < 0.0
        or consecutive_losses >= max(1, int(config.autonomy_repair_retire_after_failures))
        or current_drawdown_pct >= config.autonomy_repair_drawdown_pct * 0.60
    ):
        return "hold", 0
    confirmed_successes = max(1, int(config.autonomy_repair_promote_after_successes))
    if (
        health_status == "active"
        and total_net_pnl >= 0.0
        and current_drawdown_pct <= config.autonomy_repair_drawdown_pct * 0.25
        and closed_trade_count >= confirmed_successes
        and win_rate >= 0.50
        and consecutive_losses == 0
    ):
        return "release", 0
    if (
        health_status == "active"
        and total_net_pnl >= 0.0
        and current_drawdown_pct <= config.autonomy_repair_drawdown_pct * 0.50
        and consecutive_losses == 0
        and (closed_trade_count > 0 or unrealized_pnl > 0.0)
    ):
        return "accelerate", 2
    return "neutral", 1


def runtime_override_performance_snapshot(
    runtime_evidence: RuntimeEvidenceSnapshot | None,
) -> dict:
    if runtime_evidence is None:
        return {}
    return {
        "open_position": bool(runtime_evidence.open_position),
        "realized_pnl": round(float(runtime_evidence.realized_pnl or 0.0), 4),
        "unrealized_pnl": round(float(runtime_evidence.unrealized_pnl or 0.0), 4),
        "total_net_pnl": round(float(runtime_evidence.total_net_pnl or 0.0), 4),
        "current_drawdown_pct": round(float(runtime_evidence.current_drawdown_pct or 0.0), 4),
        "closed_trade_count": int(runtime_evidence.closed_trade_count or 0),
        "win_rate": round(float(runtime_evidence.win_rate or 0.0), 4),
        "consecutive_losses": int(runtime_evidence.consecutive_losses or 0),
        "health_status": str(runtime_evidence.health_status or ""),
        "last_closed_pnl": round(
            float((runtime_evidence.notes or {}).get("last_closed_pnl") or 0.0),
            4,
        ),
    }


def merged_runtime_overrides(
    *,
    config: EvolutionConfig,
    previous: RuntimeState | None,
    repair_reentry_notes: dict,
    runtime_evidence: RuntimeEvidenceSnapshot | None = None,
) -> tuple[dict[str, float], dict]:
    previous_overrides = (
        lifecycle_policy_runtime_overrides(previous.notes)
        if previous
        else {}
    )
    current_overrides = repair_runtime_overrides_from_notes(repair_reentry_notes)
    previous_state = lifecycle_policy_runtime_override_state(previous.notes) if previous else {}
    performance_snapshot = runtime_override_performance_snapshot(runtime_evidence)
    if current_overrides:
        merged = {str(key): float(value) for key, value in previous_overrides.items()}
        merged.update(current_overrides)
        return merged, {
            "cycles_since_refresh": 0,
            "fresh_refresh": True,
            "recovery_mode": "refresh",
            "decay_steps_applied": 0,
            "performance_snapshot": performance_snapshot,
        }
    recovery_mode, decay_steps = runtime_override_recovery_policy(
        config=config,
        runtime_evidence=runtime_evidence,
    )
    cycles_since_refresh = (
        int(previous_state.get("cycles_since_refresh") or 0) + 1
        if previous_overrides
        else 0
    )
    if recovery_mode == "release":
        return {}, {
            "cycles_since_refresh": cycles_since_refresh,
            "fresh_refresh": False,
            "recovery_mode": recovery_mode,
            "decay_steps_applied": 0,
            "performance_snapshot": performance_snapshot,
        }
    if recovery_mode == "hold":
        return previous_overrides, {
            "cycles_since_refresh": cycles_since_refresh,
            "fresh_refresh": False,
            "recovery_mode": recovery_mode,
            "decay_steps_applied": 0,
            "performance_snapshot": performance_snapshot,
        }
    decayed = dict(previous_overrides)
    for _ in range(max(decay_steps, 1)):
        if not decayed:
            break
        decayed = decay_runtime_overrides(decayed, config=config)
    return decayed, {
        "cycles_since_refresh": cycles_since_refresh,
        "fresh_refresh": False,
        "recovery_mode": recovery_mode,
        "decay_steps_applied": max(decay_steps, 1) if previous_overrides else 0,
        "performance_snapshot": performance_snapshot,
    }


def staged_exit_active(state: dict | None) -> bool:
    payload = dict(state or {})
    return (
        str(payload.get("mode") or "") == "profit_lock"
        and str(payload.get("phase") or "") != "exit"
        and float(payload.get("target_multiplier") or 1.0) < 0.999
    )


def merged_staged_exit_state(
    *,
    config: EvolutionConfig,
    previous: RuntimeState | None,
    directive: ExecutionDirective | None = None,
    runtime_evidence: RuntimeEvidenceSnapshot | None = None,
) -> dict:
    previous_state = lifecycle_policy_staged_exit_state(previous.notes) if previous else {}
    previous_active = staged_exit_active(previous_state)
    reasons = [
        str(item).strip()
        for item in list((directive.reasons if directive is not None else []) or [])
        if str(item).strip()
    ]
    action = directive.action if directive is not None else None
    capital_multiplier = float(directive.capital_multiplier or 1.0) if directive is not None else 1.0
    if action == ExecutionAction.EXIT and "profit_lock_exit" in reasons:
        return {
            "mode": "profit_lock",
            "phase": "exit",
            "target_multiplier": 0.0,
            "trigger_count": int(previous_state.get("trigger_count") or 0) + 1,
            "recovery_count": 0,
            "last_reason": "profit_lock_exit",
        }
    if action == ExecutionAction.SCALE_DOWN and "profit_lock_harvest" in reasons:
        requested_multiplier = min(1.0, max(0.0, capital_multiplier))
        previous_multiplier = (
            float(previous_state.get("target_multiplier") or 1.0)
            if previous_active
            else 1.0
        )
        if previous_active and requested_multiplier >= previous_multiplier - 1e-6:
            target_multiplier = min(
                previous_multiplier,
                float(config.autonomy_profit_lock_deep_scale_down_factor),
            )
        else:
            target_multiplier = min(requested_multiplier, previous_multiplier)
        target_multiplier = round(max(0.0, min(1.0, target_multiplier)), 4)
        phase = (
            "deep_harvest"
            if target_multiplier
            <= float(config.autonomy_profit_lock_deep_scale_down_factor) + 1e-6
            else "harvest"
        )
        return {
            "mode": "profit_lock",
            "phase": phase,
            "target_multiplier": target_multiplier,
            "trigger_count": int(previous_state.get("trigger_count") or 0) + 1,
            "recovery_count": 0,
            "last_reason": "profit_lock_harvest",
        }
    if not previous_active:
        return {}
    recovery_mode, _ = runtime_override_recovery_policy(
        config=config,
        runtime_evidence=runtime_evidence,
    )
    previous_multiplier = float(previous_state.get("target_multiplier") or 1.0)
    if action == ExecutionAction.EXIT:
        return {}
    if recovery_mode == "release":
        return {}
    if recovery_mode == "accelerate":
        target_multiplier = previous_multiplier + (1.0 - previous_multiplier) * 0.50
        recovery_count = int(previous_state.get("recovery_count") or 0) + 1
        if target_multiplier >= 0.999:
            return {}
        return {
            **previous_state,
            "mode": "profit_lock",
            "phase": "reentry",
            "target_multiplier": round(target_multiplier, 4),
            "recovery_count": recovery_count,
            "last_reason": "profit_lock_reentry",
        }
    if (
        recovery_mode == "neutral"
        and str(previous_state.get("phase") or "") == "reentry"
    ):
        target_multiplier = previous_multiplier + (1.0 - previous_multiplier) * max(
            0.25,
            float(config.autonomy_runtime_override_decay_rate),
        )
        recovery_count = int(previous_state.get("recovery_count") or 0) + 1
        if target_multiplier >= 0.999:
            return {}
        return {
            **previous_state,
            "mode": "profit_lock",
            "phase": "reentry",
            "target_multiplier": round(target_multiplier, 4),
            "recovery_count": recovery_count,
            "last_reason": "profit_lock_reentry",
        }
    return previous_state


def apply_staged_exit_to_capital(
    *,
    desired_capital: float,
    allocated_capital: float,
    staged_exit_state: dict,
) -> float:
    if not staged_exit_state:
        return round(max(0.0, desired_capital), 2)
    target_multiplier = float(staged_exit_state.get("target_multiplier") or 1.0)
    if target_multiplier >= 0.999:
        return round(max(0.0, desired_capital), 2)
    stage_capital = max(0.0, float(allocated_capital) * max(0.0, target_multiplier))
    return round(min(max(0.0, desired_capital), stage_capital), 2)


def merged_repair_reentry_notes(
    *,
    previous: RuntimeState | None,
    current_notes: dict | None,
) -> dict:
    previous_notes = lifecycle_policy_repair_reentry_notes(previous.notes) if previous else {}
    current_notes = dict(current_notes or {})
    merged = {**previous_notes, **current_notes}
    if "runtime_overrides" in previous_notes or "runtime_overrides" in current_notes:
        merged["runtime_overrides"] = {
            **dict(previous_notes.get("runtime_overrides") or {}),
            **dict(current_notes.get("runtime_overrides") or {}),
        }
    return merged


def build_repair_reentry_notes(
    *,
    source_runtime_id: str = "",
    source_strategy_id: str = "",
    raw_stage: str = "",
    effective_target_stage: str = "",
    requested_validation_stage: str = "",
    runtime_overrides: dict | None = None,
) -> dict:
    notes = {
        "source_runtime_id": str(source_runtime_id or ""),
        "source_strategy_id": str(source_strategy_id or ""),
        "raw_stage": str(raw_stage or ""),
        "effective_target_stage": str(effective_target_stage or ""),
        "requested_validation_stage": str(requested_validation_stage or ""),
    }
    overrides = repair_runtime_overrides_from_notes(
        {"runtime_overrides": dict(runtime_overrides or {})}
    )
    if overrides:
        notes["runtime_overrides"] = overrides
    return notes


def reentry_state_active(state: dict | None) -> bool:
    payload = dict(state or {})
    return str(payload.get("mode") or "") == "repair_reentry"


def timeframe_minutes(timeframe: str) -> float:
    text = str(timeframe or "").strip().lower()
    if not text:
        return 0.0
    try:
        if text.endswith("m"):
            return float(text[:-1] or 0.0)
        if text.endswith("h"):
            return float(text[:-1] or 0.0) * 60.0
        if text.endswith("d"):
            return float(text[:-1] or 0.0) * 1440.0
    except ValueError:
        return 0.0
    return 0.0


def latest_managed_close_time_index(
    storage,
    *,
    managed_source: str,
    limit: int = 5000,
) -> dict[str, datetime]:
    latest: dict[str, datetime] = {}
    for row in storage.get_pnl_ledger(limit=limit, event_type="close"):
        metadata = _row_metadata(row)
        if metadata.get("source") != managed_source:
            continue
        runtime_id = str(metadata.get("runtime_id") or "").strip()
        if not runtime_id:
            continue
        event_time = str(row.get("event_time") or "").strip()
        if not event_time:
            continue
        try:
            parsed = datetime.fromisoformat(event_time)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        existing = latest.get(runtime_id)
        if existing is None or parsed > existing:
            latest[runtime_id] = parsed
    return latest


def reentry_cooldown_active(
    *,
    runtime_id: str,
    timeframe: str,
    runtime_overrides: dict[str, float],
    repair_reentry_notes: dict,
    latest_close_time_by_runtime: dict[str, datetime] | None,
    now: datetime | None = None,
) -> bool:
    multiplier = float(runtime_overrides.get("entry_cooldown_bars_multiplier") or 0.0)
    if multiplier <= 0:
        return False
    bar_minutes = timeframe_minutes(timeframe)
    if bar_minutes <= 0:
        return False
    source_runtime_id = str(repair_reentry_notes.get("source_runtime_id") or "").strip()
    latest_close = None
    if latest_close_time_by_runtime:
        latest_close = (
            latest_close_time_by_runtime.get(source_runtime_id)
            or latest_close_time_by_runtime.get(runtime_id)
        )
    if latest_close is None:
        return False
    reference_now = now or datetime.now(timezone.utc)
    elapsed_minutes = (
        reference_now.astimezone(timezone.utc) - latest_close.astimezone(timezone.utc)
    ).total_seconds() / 60.0
    return elapsed_minutes < bar_minutes * multiplier


def merged_reentry_state(
    *,
    config: EvolutionConfig,
    previous: RuntimeState | None,
    repair_reentry_notes: dict,
    runtime_overrides: dict[str, float],
    runtime_override_state: dict,
    latest_close_time_by_runtime: dict[str, datetime] | None,
    runtime_id: str,
    timeframe: str,
) -> dict:
    previous_state = lifecycle_policy_reentry_state(previous.notes) if previous else {}
    notes = dict(repair_reentry_notes or {})
    if not notes and previous_state:
        notes = {
            "source_runtime_id": str(previous_state.get("source_runtime_id") or ""),
            "source_strategy_id": str(previous_state.get("source_strategy_id") or ""),
            "effective_target_stage": str(previous_state.get("effective_target_stage") or ""),
            "requested_validation_stage": str(previous_state.get("requested_validation_stage") or ""),
            "raw_stage": str(previous_state.get("raw_stage") or ""),
        }
    source_runtime_id = str(notes.get("source_runtime_id") or "").strip()
    source_strategy_id = str(notes.get("source_strategy_id") or "").strip()
    if not source_runtime_id and not previous_state:
        return {}
    cooldown_active = reentry_cooldown_active(
        runtime_id=runtime_id,
        timeframe=timeframe,
        runtime_overrides=runtime_overrides,
        repair_reentry_notes=notes,
        latest_close_time_by_runtime=latest_close_time_by_runtime,
    )
    recovery_mode = str(runtime_override_state.get("recovery_mode") or "").strip().lower()
    active_overrides = sorted(
        str(key).strip()
        for key in runtime_overrides.keys()
        if str(key).strip()
    )
    if cooldown_active:
        phase = "cooldown"
    elif not active_overrides and recovery_mode in {"release", ""}:
        return {}
    elif recovery_mode == "accelerate":
        phase = "recovery"
    else:
        phase = "probation"
    return {
        "mode": "repair_reentry",
        "phase": phase,
        "source_runtime_id": source_runtime_id or str(previous_state.get("source_runtime_id") or ""),
        "source_strategy_id": source_strategy_id or str(previous_state.get("source_strategy_id") or ""),
        "effective_target_stage": str(
            notes.get("effective_target_stage")
            or previous_state.get("effective_target_stage")
            or ""
        ),
        "requested_validation_stage": str(
            notes.get("requested_validation_stage")
            or previous_state.get("requested_validation_stage")
            or ""
        ),
        "raw_stage": str(notes.get("raw_stage") or previous_state.get("raw_stage") or ""),
        "fresh_refresh": bool(runtime_override_state.get("fresh_refresh")),
        "recovery_mode": recovery_mode or str(previous_state.get("recovery_mode") or ""),
        "cycles_since_refresh": int(runtime_override_state.get("cycles_since_refresh") or 0),
        "cooldown_active": bool(cooldown_active),
        "active_overrides": active_overrides,
        "release_ready": (
            not cooldown_active
            and not active_overrides
            and recovery_mode == "release"
        ),
    }


def build_runtime_lifecycle_policy(
    *,
    runtime_override_state: dict | None,
    runtime_overrides: dict | None,
    staged_exit_state: dict | None,
    reentry_state: dict | None,
) -> dict:
    override_state = dict(runtime_override_state or {})
    overrides = dict(runtime_overrides or {})
    staged_exit = dict(staged_exit_state or {})
    reentry = dict(reentry_state or {})
    return {
        "policy_version": 1,
        "runtime_override": {
            "active": bool(overrides),
            "recovery_mode": str(override_state.get("recovery_mode") or ""),
            "fresh_refresh": bool(override_state.get("fresh_refresh")),
            "cycles_since_refresh": int(override_state.get("cycles_since_refresh") or 0),
            "decay_steps_applied": int(override_state.get("decay_steps_applied") or 0),
            "performance_snapshot": dict(override_state.get("performance_snapshot") or {}),
            "values": dict(overrides),
            "active_keys": sorted(
                str(key).strip()
                for key in overrides.keys()
                if str(key).strip()
            ),
        },
        "staged_exit": {
            "active": staged_exit_active(staged_exit),
            "mode": str(staged_exit.get("mode") or ""),
            "phase": str(staged_exit.get("phase") or ""),
            "target_multiplier": float(staged_exit.get("target_multiplier") or 1.0),
            "last_reason": str(staged_exit.get("last_reason") or ""),
            "recovery_count": int(staged_exit.get("recovery_count") or 0),
        },
        "repair_reentry": {
            "active": reentry_state_active(reentry),
            "phase": str(reentry.get("phase") or ""),
            "recovery_mode": str(reentry.get("recovery_mode") or ""),
            "cooldown_active": bool(reentry.get("cooldown_active")),
            "release_ready": bool(reentry.get("release_ready")),
            "active_overrides": list(reentry.get("active_overrides") or []),
            "source_runtime_id": str(reentry.get("source_runtime_id") or ""),
            "source_strategy_id": str(reentry.get("source_strategy_id") or ""),
            "effective_target_stage": str(reentry.get("effective_target_stage") or ""),
            "requested_validation_stage": str(reentry.get("requested_validation_stage") or ""),
            "raw_stage": str(reentry.get("raw_stage") or ""),
        },
    }


def runtime_lifecycle_policy(notes: dict | None) -> dict:
    payload = dict(notes or {})
    policy = dict(payload.get("runtime_lifecycle_policy") or {})
    return policy if isinstance(policy, dict) else {}


def lifecycle_policy_runtime_overrides(notes: dict | None) -> dict[str, float]:
    policy = runtime_lifecycle_policy(notes)
    override = dict(policy.get("runtime_override") or {})
    raw = dict(override.get("values") or {})
    if raw:
        values: dict[str, float] = {}
        for key, value in raw.items():
            text = str(key).strip()
            if not text:
                continue
            try:
                values[text] = float(value)
            except (TypeError, ValueError):
                continue
        return values
    return {
        str(key): float(value)
        for key, value in dict((notes or {}).get("runtime_overrides") or {}).items()
        if str(key).strip()
    }


def lifecycle_policy_runtime_override_state(notes: dict | None) -> dict:
    policy = runtime_lifecycle_policy(notes)
    override = dict(policy.get("runtime_override") or {})
    if override:
        return {
            "recovery_mode": str(override.get("recovery_mode") or ""),
            "fresh_refresh": bool(override.get("fresh_refresh")),
            "cycles_since_refresh": int(override.get("cycles_since_refresh") or 0),
            "decay_steps_applied": int(override.get("decay_steps_applied") or 0),
            "performance_snapshot": dict(override.get("performance_snapshot") or {}),
        }
    return dict((notes or {}).get("runtime_override_state") or {})


def lifecycle_policy_staged_exit_state(notes: dict | None) -> dict:
    policy = runtime_lifecycle_policy(notes)
    staged = dict(policy.get("staged_exit") or {})
    if staged:
        return {
            "mode": str(staged.get("mode") or ""),
            "phase": str(staged.get("phase") or ""),
            "target_multiplier": float(staged.get("target_multiplier") or 1.0),
            "last_reason": str(staged.get("last_reason") or ""),
            "recovery_count": int(staged.get("recovery_count") or 0),
            "active": bool(staged.get("active")),
        }
    return dict((notes or {}).get("staged_exit_state") or {})


def lifecycle_policy_reentry_state(notes: dict | None) -> dict:
    policy = runtime_lifecycle_policy(notes)
    reentry = dict(policy.get("repair_reentry") or {})
    if reentry:
        return {
            "mode": "repair_reentry" if bool(reentry.get("active")) else "",
            "phase": str(reentry.get("phase") or ""),
            "recovery_mode": str(reentry.get("recovery_mode") or ""),
            "cooldown_active": bool(reentry.get("cooldown_active")),
            "release_ready": bool(reentry.get("release_ready")),
            "active_overrides": list(reentry.get("active_overrides") or []),
            "source_runtime_id": str(reentry.get("source_runtime_id") or ""),
            "source_strategy_id": str(reentry.get("source_strategy_id") or ""),
            "effective_target_stage": str(reentry.get("effective_target_stage") or ""),
            "requested_validation_stage": str(reentry.get("requested_validation_stage") or ""),
            "raw_stage": str(reentry.get("raw_stage") or ""),
        }
    return dict((notes or {}).get("reentry_state") or {})


def lifecycle_policy_repair_reentry_notes(notes: dict | None) -> dict:
    reentry = lifecycle_policy_reentry_state(notes)
    if reentry:
        payload = {
            "source_runtime_id": str(reentry.get("source_runtime_id") or ""),
            "source_strategy_id": str(reentry.get("source_strategy_id") or ""),
            "effective_target_stage": str(reentry.get("effective_target_stage") or ""),
            "requested_validation_stage": str(reentry.get("requested_validation_stage") or ""),
            "raw_stage": str(reentry.get("raw_stage") or ""),
        }
        runtime_overrides = lifecycle_policy_runtime_overrides(notes)
        if runtime_overrides:
            payload["runtime_overrides"] = runtime_overrides
        return payload
    return dict((notes or {}).get("repair_reentry") or {})


def compose_runtime_policy_notes(
    *,
    base_notes: dict | None = None,
    repair_reentry_notes: dict | None = None,
    runtime_overrides: dict | None = None,
    runtime_override_state: dict | None = None,
    staged_exit_state: dict | None = None,
    reentry_state: dict | None = None,
) -> dict:
    notes = dict(base_notes or {})
    repair_notes = dict(repair_reentry_notes or {})
    repair_note_overrides = repair_runtime_overrides_from_notes(repair_notes)
    if runtime_overrides is None:
        effective_runtime_overrides = dict(repair_note_overrides)
    else:
        effective_runtime_overrides = repair_runtime_overrides_from_notes(
            {"runtime_overrides": dict(runtime_overrides or {})}
        )
        if effective_runtime_overrides and repair_note_overrides:
            effective_runtime_overrides = {
                **repair_note_overrides,
                **effective_runtime_overrides,
            }
    effective_reentry_state = dict(reentry_state or {})
    if repair_notes:
        for key in (
            "source_runtime_id",
            "source_strategy_id",
            "effective_target_stage",
            "requested_validation_stage",
            "raw_stage",
        ):
            value = str(repair_notes.get(key) or "").strip()
            if value and not str(effective_reentry_state.get(key) or "").strip():
                effective_reentry_state[key] = value
        if effective_runtime_overrides and not list(effective_reentry_state.get("active_overrides") or []):
            effective_reentry_state["active_overrides"] = sorted(
                str(key).strip()
                for key in effective_runtime_overrides.keys()
                if str(key).strip()
            )
    policy = build_runtime_lifecycle_policy(
        runtime_override_state=runtime_override_state,
        runtime_overrides=effective_runtime_overrides,
        staged_exit_state=staged_exit_state,
        reentry_state=effective_reentry_state,
    )
    notes["runtime_lifecycle_policy"] = policy
    return notes


def compose_runtime_policy_notes_legacy(
    *,
    base_notes: dict | None = None,
    repair_reentry_notes: dict | None = None,
    runtime_overrides: dict | None = None,
    runtime_override_state: dict | None = None,
    staged_exit_state: dict | None = None,
    reentry_state: dict | None = None,
) -> dict:
    notes = compose_runtime_policy_notes(
        base_notes=base_notes,
        repair_reentry_notes=repair_reentry_notes,
        runtime_overrides=runtime_overrides,
        runtime_override_state=runtime_override_state,
        staged_exit_state=staged_exit_state,
        reentry_state=reentry_state,
    )
    notes["repair_reentry"] = dict(repair_reentry_notes or {})
    notes["runtime_overrides"] = dict(runtime_overrides or {})
    notes["runtime_override_state"] = dict(runtime_override_state or {})
    notes["staged_exit_state"] = dict(staged_exit_state or {})
    notes["reentry_state"] = dict(reentry_state or {})
    return notes


def strip_legacy_runtime_policy_notes(notes: dict | None) -> dict:
    payload = dict(notes or {})
    if "runtime_lifecycle_policy" not in payload:
        return payload
    for key in _LEGACY_POLICY_KEYS:
        payload.pop(key, None)
    return payload


def hydrate_runtime_policy_notes(notes: dict | None) -> dict:
    payload = dict(notes or {})
    policy = runtime_lifecycle_policy(payload)
    if not policy:
        return payload
    base_notes = {
        key: value
        for key, value in payload.items()
        if key not in _LEGACY_POLICY_KEYS and key != "runtime_lifecycle_policy"
    }
    return compose_runtime_policy_notes_legacy(
        base_notes=base_notes,
        repair_reentry_notes=lifecycle_policy_repair_reentry_notes(payload),
        runtime_overrides=lifecycle_policy_runtime_overrides(payload),
        runtime_override_state=lifecycle_policy_runtime_override_state(payload),
        staged_exit_state=lifecycle_policy_staged_exit_state(payload),
        reentry_state=lifecycle_policy_reentry_state(payload),
    )


def _row_metadata(row: dict | None) -> dict:
    if not row:
        return {}
    raw = row.get("metadata_json")
    if isinstance(raw, dict):
        return dict(raw)
    try:
        payload = json.loads(raw or "{}")
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}
