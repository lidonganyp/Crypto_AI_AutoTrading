"""Runtime override and learning-driven settings orchestration."""
from __future__ import annotations

from datetime import datetime, timezone

from config import Settings, get_settings
from core.storage import Storage


class RuntimeSettingsService:
    """Apply default/manual/learning runtime settings consistently."""

    def __init__(
        self,
        storage: Storage,
        settings: Settings | None = None,
        *,
        decision_engine,
        strategy_evolver,
        base_runtime_settings: dict[str, float | list[float]],
        runtime_override_state_key: str,
        runtime_locked_fields_state_key: str,
        runtime_learning_override_state_key: str,
        runtime_override_conflict_state_key: str,
        runtime_learning_details_state_key: str,
        runtime_effective_state_key: str,
    ):
        self.storage = storage
        self.settings = settings or get_settings()
        self.decision_engine = decision_engine
        self.strategy_evolver = strategy_evolver
        self.base_runtime_settings = dict(base_runtime_settings)
        self.runtime_override_state_key = runtime_override_state_key
        self.runtime_locked_fields_state_key = runtime_locked_fields_state_key
        self.runtime_learning_override_state_key = runtime_learning_override_state_key
        self.runtime_override_conflict_state_key = runtime_override_conflict_state_key
        self.runtime_learning_details_state_key = runtime_learning_details_state_key
        self.runtime_effective_state_key = runtime_effective_state_key

    def reset_runtime_settings_to_base(self) -> None:
        self.settings.model.xgboost_probability_threshold = float(
            self.base_runtime_settings["xgboost_probability_threshold"]
        )
        self.settings.model.final_score_threshold = float(
            self.base_runtime_settings["final_score_threshold"]
        )
        self.settings.strategy.min_liquidity_ratio = float(
            self.base_runtime_settings["min_liquidity_ratio"]
        )
        self.settings.strategy.sentiment_weight = float(
            self.base_runtime_settings["sentiment_weight"]
        )
        self.settings.strategy.fixed_stop_loss_pct = float(
            self.base_runtime_settings["fixed_stop_loss_pct"]
        )
        self.settings.strategy.take_profit_levels = list(
            self.base_runtime_settings["take_profit_levels"]
        )

    def sync_runtime_components(self) -> None:
        self.decision_engine.xgboost_threshold = (
            self.settings.model.xgboost_probability_threshold
        )
        self.decision_engine.final_score_threshold = (
            self.settings.model.final_score_threshold
        )
        self.decision_engine.sentiment_weight = self.settings.strategy.sentiment_weight
        self.decision_engine.min_liquidity_ratio = (
            self.settings.strategy.min_liquidity_ratio
        )
        self.decision_engine.fixed_stop_loss_pct = (
            self.settings.strategy.fixed_stop_loss_pct
        )
        self.decision_engine.take_profit_levels = list(
            self.settings.strategy.take_profit_levels
        )
        self.decision_engine.max_hold_hours = self.settings.strategy.max_hold_hours

    def persist_runtime_settings_effective(
        self,
        runtime_settings_overrides: dict[str, float | list[float]],
    ) -> dict[str, float | list[float] | dict[str, float | list[float]] | str]:
        conflict_payload = self.storage.get_json_state(
            self.runtime_override_conflict_state_key,
            {},
        ) or {}
        effective = {
            "xgboost_probability_threshold": float(self.decision_engine.xgboost_threshold),
            "final_score_threshold": float(self.decision_engine.final_score_threshold),
            "min_liquidity_ratio": float(self.decision_engine.min_liquidity_ratio),
            "sentiment_weight": float(self.decision_engine.sentiment_weight),
            "fixed_stop_loss_pct": float(self.settings.strategy.fixed_stop_loss_pct),
            "take_profit_levels": list(self.settings.strategy.take_profit_levels),
            "overrides": dict(runtime_settings_overrides),
            "manual_overrides": self.storage.get_json_state(
                self.runtime_override_state_key, {}
            )
            or {},
            "learning_overrides": self.storage.get_json_state(
                self.runtime_learning_override_state_key, {}
            )
            or {},
            "blocked_learning_overrides": conflict_payload.get(
                "blocked_learning_overrides", {}
            ),
            "auto_superseded_manual_overrides": conflict_payload.get(
                "auto_superseded_manual_overrides", {}
            ),
            "locked_fields": conflict_payload.get("locked_fields", []),
            "override_sources": conflict_payload.get("override_sources")
            or {key: "default" for key in self.base_runtime_settings},
            "effective_mode": conflict_payload.get("effective_mode", "default"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.storage.set_json_state(self.runtime_effective_state_key, effective)
        return effective

    def apply_runtime_overrides(
        self,
    ) -> tuple[
        dict[str, float | list[float]],
        dict[str, float | list[float] | dict[str, float | list[float]] | str],
    ]:
        self.reset_runtime_settings_to_base()
        manual_raw = self.storage.get_json_state(self.runtime_override_state_key, {}) or {}
        learning_raw = (
            self.storage.get_json_state(self.runtime_learning_override_state_key, {}) or {}
        )
        locked_raw = (
            self.storage.get_json_state(self.runtime_locked_fields_state_key, []) or []
        )
        if not isinstance(manual_raw, dict):
            manual_raw = {}
        if not isinstance(learning_raw, dict):
            learning_raw = {}
        if not isinstance(locked_raw, list):
            locked_raw = []

        learning_applied = self.sanitize_runtime_override_payload(learning_raw)
        manual_applied = self.sanitize_runtime_override_payload(manual_raw)
        locked_fields = sorted(
            key
            for key in {str(item) for item in locked_raw}
            if key in self.base_runtime_settings
        )
        applied: dict[str, float | list[float]] = {}
        auto_superseded_manual_overrides: dict[str, float | list[float]] = {}
        for key in self.base_runtime_settings:
            manual_present = key in manual_applied
            learning_present = key in learning_applied
            locked = key in locked_fields
            if locked and manual_present:
                applied[key] = manual_applied[key]
                continue
            if learning_present:
                applied[key] = learning_applied[key]
                if manual_present:
                    auto_superseded_manual_overrides[key] = manual_applied[key]
                continue
            if manual_present:
                applied[key] = manual_applied[key]

        conflict_keys = sorted(
            key
            for key in set(manual_applied) & set(learning_applied)
            if key in locked_fields
        )
        blocked_learning_overrides = {key: learning_applied[key] for key in conflict_keys}
        override_sources = {
            key: (
                "manual_locked"
                if key in manual_applied and key in locked_fields
                else "learning"
                if key in learning_applied and key not in locked_fields
                else "manual"
                if key in manual_applied
                else "default"
            )
            for key in self.base_runtime_settings
        }
        effective_mode = (
            "locked_manual"
            if conflict_keys
            else "automatic"
            if learning_applied
            else "manual"
            if manual_applied
            else "default"
        )

        if "xgboost_probability_threshold" in applied:
            self.settings.model.xgboost_probability_threshold = float(
                applied["xgboost_probability_threshold"]
            )
        if "final_score_threshold" in applied:
            self.settings.model.final_score_threshold = float(
                applied["final_score_threshold"]
            )
        if "min_liquidity_ratio" in applied:
            self.settings.strategy.min_liquidity_ratio = float(
                applied["min_liquidity_ratio"]
            )
        if "sentiment_weight" in applied:
            self.settings.strategy.sentiment_weight = float(applied["sentiment_weight"])
        if "fixed_stop_loss_pct" in applied:
            self.settings.strategy.fixed_stop_loss_pct = float(
                applied["fixed_stop_loss_pct"]
            )
        if "take_profit_levels" in applied:
            self.settings.strategy.take_profit_levels = list(applied["take_profit_levels"])

        self.sync_runtime_components()
        if manual_applied != manual_raw:
            self.storage.set_json_state(self.runtime_override_state_key, manual_applied)
        if learning_applied != learning_raw:
            self.storage.set_json_state(
                self.runtime_learning_override_state_key,
                learning_applied,
            )
        normalized_locked = sorted(
            {
                str(item)
                for item in locked_raw
                if str(item) in self.base_runtime_settings
            }
        )
        if locked_fields != normalized_locked:
            self.storage.set_json_state(
                self.runtime_locked_fields_state_key,
                locked_fields,
            )
        self.storage.set_json_state(
            self.runtime_override_conflict_state_key,
            {
                "conflict_keys": conflict_keys,
                "blocked_learning_overrides": blocked_learning_overrides,
                "manual_only_keys": sorted(set(manual_applied) - set(learning_applied)),
                "learning_only_keys": sorted(set(learning_applied) - set(manual_applied)),
                "auto_superseded_manual_overrides": auto_superseded_manual_overrides,
                "locked_fields": locked_fields,
                "override_sources": override_sources,
                "effective_mode": effective_mode,
            },
        )
        self.storage.set_state("runtime_settings_override_status", effective_mode)
        effective = self.persist_runtime_settings_effective(applied)
        return applied, effective

    def refresh_learning_runtime_overrides(self, now: datetime) -> dict:
        suggestion = self.strategy_evolver.suggest_runtime_overrides(
            self.base_runtime_settings,
        )
        overrides = suggestion.get("runtime_overrides", {}) or {}
        if not isinstance(overrides, dict):
            overrides = {}
        previous = (
            self.storage.get_json_state(self.runtime_learning_override_state_key, {}) or {}
        )
        previous_details = (
            self.storage.get_json_state(self.runtime_learning_details_state_key, {}) or {}
        )
        if overrides != previous:
            self.storage.set_json_state(self.runtime_learning_override_state_key, overrides)
        if suggestion != previous_details:
            self.storage.set_json_state(self.runtime_learning_details_state_key, suggestion)
        if overrides != previous or suggestion != previous_details:
            self.storage.insert_execution_event(
                "learning_runtime_override",
                "SYSTEM",
                {
                    "runtime_overrides": overrides,
                    "reasons": suggestion.get("reasons", []),
                    "stats": suggestion.get("stats", {}),
                    "updated_at": now.isoformat(),
                },
            )
        return suggestion

    def sanitize_runtime_override_payload(
        self,
        overrides: dict,
    ) -> dict[str, float | list[float]]:
        applied: dict[str, float | list[float]] = {}
        for key, raw_value in overrides.items():
            if key not in self.base_runtime_settings:
                continue
            value = self.sanitize_runtime_override_value(key, raw_value)
            if value is None:
                continue
            applied[key] = value
        return applied

    @staticmethod
    def sanitize_runtime_override_value(
        key: str,
        raw_value,
    ) -> float | list[float] | None:
        float_ranges = {
            "xgboost_probability_threshold": (0.0, 1.0),
            "final_score_threshold": (0.0, 1.0),
            "min_liquidity_ratio": (0.0, 5.0),
            "sentiment_weight": (-1.0, 1.0),
            "fixed_stop_loss_pct": (0.0, 0.2),
        }
        if key == "take_profit_levels":
            if isinstance(raw_value, str):
                parts = [part.strip() for part in raw_value.split(",")]
            elif isinstance(raw_value, list):
                parts = raw_value
            else:
                return None
            try:
                levels = [float(part) for part in parts if str(part).strip()]
            except (TypeError, ValueError):
                return None
            levels = sorted(level for level in levels if level > 0)
            return levels or None

        bounds = float_ranges.get(key)
        if bounds is None:
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None
        lower, upper = bounds
        if value < lower or value > upper:
            return None
        return value
