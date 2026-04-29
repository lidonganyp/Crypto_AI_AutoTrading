"""Unified runtime/system state access helpers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeStateSnapshot:
    execution_symbols: list[str]
    active_symbols: list[str]
    model_ready_symbols: list[str]
    consistency_blocked_symbols: list[str]
    consistency_blocked_details: dict[str, list[str]]
    fast_alpha_active_symbols: list[str]
    paper_exploration_active_symbols: list[str]
    shadow_observation_symbols: list[str]
    watchlist_whitelist: list[str]
    watchlist_blacklist: list[str]
    runtime_settings_overrides: dict
    runtime_settings_locked_fields: list[str]
    runtime_settings_learning_overrides: dict
    runtime_settings_learning_details: dict
    runtime_settings_effective: dict
    runtime_settings_override_conflicts: dict
    model_promotion_candidates: dict
    model_promotion_observations: dict


class RuntimeStateView:
    """Centralize normalized runtime state reads from system_state."""

    def __init__(self, storage):
        self.storage = storage

    def _json(self, key: str, default):
        value = self.storage.get_json_state(key, default)
        return default if value is None else value

    @staticmethod
    def _as_str_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    @staticmethod
    def _as_str_dict_list(value) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return {}
        payload: dict[str, list[str]] = {}
        for key, items in value.items():
            normalized = RuntimeStateView._as_str_list(items)
            if normalized:
                payload[str(key)] = normalized
        return payload

    @staticmethod
    def _as_dict(value) -> dict:
        return dict(value) if isinstance(value, dict) else {}

    def snapshot(self) -> RuntimeStateSnapshot:
        return RuntimeStateSnapshot(
            execution_symbols=self._as_str_list(
                self._json("execution_symbols", [])
            ),
            active_symbols=self._as_str_list(
                self._json("active_symbols", [])
            ),
            model_ready_symbols=self._as_str_list(
                self._json("model_ready_symbols", [])
            ),
            consistency_blocked_symbols=self._as_str_list(
                self._json("consistency_blocked_symbols", [])
            ),
            consistency_blocked_details=self._as_str_dict_list(
                self._json("consistency_blocked_details", {})
            ),
            fast_alpha_active_symbols=self._as_str_list(
                self._json("fast_alpha_active_symbols", [])
            ),
            paper_exploration_active_symbols=self._as_str_list(
                self._json("paper_exploration_active_symbols", [])
            ),
            shadow_observation_symbols=self._as_str_list(
                self._json("shadow_observation_symbols", [])
            ),
            watchlist_whitelist=self._as_str_list(
                self._json("watchlist_whitelist", [])
            ),
            watchlist_blacklist=self._as_str_list(
                self._json("watchlist_blacklist", [])
            ),
            runtime_settings_overrides=self._as_dict(
                self._json("runtime_settings_overrides", {})
            ),
            runtime_settings_locked_fields=self._as_str_list(
                self._json("runtime_settings_locked_fields", [])
            ),
            runtime_settings_learning_overrides=self._as_dict(
                self._json("runtime_settings_learning_overrides", {})
            ),
            runtime_settings_learning_details=self._as_dict(
                self._json("runtime_settings_learning_details", {})
            ),
            runtime_settings_effective=self._as_dict(
                self._json("runtime_settings_effective", {})
            ),
            runtime_settings_override_conflicts=self._as_dict(
                self._json("runtime_settings_override_conflicts", {})
            ),
            model_promotion_candidates=self._as_dict(
                self._json("model_promotion_candidates", {})
            ),
            model_promotion_observations=self._as_dict(
                self._json("model_promotion_observations", {})
            ),
        )
