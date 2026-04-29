"""Execution pool and active-symbol management helpers."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

from analysis.dynamic_watchlist import DynamicWatchlistService
from config import Settings, get_settings
from core.storage import Storage


class ExecutionPoolRuntimeService:
    """Manage execution-symbol normalization, readiness, and auto-rebuild flow."""

    NON_BLOCKING_CONSISTENCY_FLAGS = {
        "backtest_return_extreme_sparse",
        "walkforward_return_extreme_sparse",
    }

    def __init__(
        self,
        storage: Storage,
        settings: Settings | None = None,
        *,
        performance_getter,
        shadow_feedback_getter=None,
        consistency_getter=None,
        watchlist_getter,
        market_getter,
        trainer_getter,
        notifier,
        current_language,
        handle_training_summary=None,
        clear_symbol_models,
        clear_broken_model_symbol,
        runtime_model_path_for_symbol,
        broken_model_symbols_state_key: str,
        execution_symbols_state_key: str,
        execution_pool_last_rebuild_at_state_key: str,
        parse_iso_datetime,
        rebuild_execution_symbols_callback,
    ):
        self.storage = storage
        self.settings = settings or get_settings()
        self.performance_getter = performance_getter
        self.shadow_feedback_getter = shadow_feedback_getter
        self.consistency_getter = consistency_getter
        self.watchlist_getter = watchlist_getter
        self.market_getter = market_getter
        self.trainer_getter = trainer_getter
        self.notifier = notifier
        self.current_language = current_language
        self.handle_training_summary = handle_training_summary
        self.clear_symbol_models = clear_symbol_models
        self.clear_broken_model_symbol = clear_broken_model_symbol
        self.runtime_model_path_for_symbol = runtime_model_path_for_symbol
        self.broken_model_symbols_state_key = broken_model_symbols_state_key
        self.execution_symbols_state_key = execution_symbols_state_key
        self.execution_pool_last_rebuild_at_state_key = (
            execution_pool_last_rebuild_at_state_key
        )
        self.parse_iso_datetime = parse_iso_datetime
        self.rebuild_execution_symbols_callback = rebuild_execution_symbols_callback

    @staticmethod
    def _shadow_trade_sample_floor(min_samples: int) -> int:
        return max(3, min_samples // 2)

    @staticmethod
    def _normalize_symbol(raw: object) -> str:
        symbol = str(raw).strip().upper().replace(" ", "")
        if not symbol:
            return ""
        if "-" in symbol and "/" not in symbol:
            parts = symbol.split("-")
            if len(parts) >= 2:
                symbol = f"{parts[0]}/{parts[1]}"
        if symbol.endswith("USDT") and "/" not in symbol and len(symbol) > 4:
            symbol = f"{symbol[:-4]}/USDT"
        return symbol

    @classmethod
    def _blocking_consistency_flags(
        cls,
        flags: list[str] | tuple[str, ...] | set[str] | None,
    ) -> list[str]:
        return [
            str(flag).strip()
            for flag in (flags or [])
            if str(flag).strip()
            and str(flag).strip() not in cls.NON_BLOCKING_CONSISTENCY_FLAGS
        ]

    def _allowed_symbol_set(self) -> set[str]:
        allowed: set[str] = set()
        for raw in self.settings.exchange.symbols:
            symbol = self._normalize_symbol(raw)
            if symbol:
                allowed.add(symbol)
        return allowed

    @staticmethod
    def _promotion_candidates_state(state: object) -> dict[str, dict]:
        return state if isinstance(state, dict) else {}

    def _promotion_candidate_stage(self, symbol: str) -> str:
        state = self._promotion_candidates_state(
            self.storage.get_json_state("model_promotion_candidates", {})
        )
        candidate = state.get(symbol)
        if not isinstance(candidate, dict):
            return ""
        return str(candidate.get("status") or "").strip().lower()

    def shadow_feedback_summary(
        self,
        limit: int = 500,
    ) -> dict[str, dict[str, float | int]]:
        getter = self.shadow_feedback_getter
        if not callable(getter):
            return {}
        service = getter()
        build_summary = getattr(service, "build_observation_feedback", None)
        if not callable(build_summary):
            return {}
        try:
            return build_summary(limit=limit)
        except TypeError:
            return build_summary()

    def _shadow_signal(
        self,
        metrics: dict[str, float | int] | None,
        *,
        min_samples: int,
        floor_pct: float,
    ) -> tuple[bool, bool]:
        if not isinstance(metrics, dict):
            return False, False
        eval_count = int(metrics.get("shadow_eval_count", 0) or 0)
        accuracy_pct = float(metrics.get("shadow_accuracy_pct", 0.0) or 0.0)
        trade_count = int(metrics.get("shadow_trade_count", 0) or 0)
        positive_ratio = float(metrics.get("shadow_positive_ratio", 0.0) or 0.0)
        avg_pnl_pct = float(metrics.get("shadow_avg_pnl_pct", 0.0) or 0.0)
        trade_floor = self._shadow_trade_sample_floor(min_samples)
        positive = (
            (eval_count >= min_samples and accuracy_pct >= floor_pct)
            or (
                trade_count >= trade_floor
                and positive_ratio >= 0.6
                and avg_pnl_pct > 0.0
            )
        )
        negative = (
            (eval_count >= min_samples and accuracy_pct < floor_pct * 0.85)
            or (
                trade_count >= trade_floor
                and positive_ratio < 0.4
                and avg_pnl_pct < 0.0
            )
        )
        return positive, negative

    def normalize_execution_symbols(self, symbols: list[str]) -> list[str]:
        disallowed_symbols = {
            entry.strip().upper() for entry in self.settings.exchange.disallowed_symbols
        }
        disallowed_sectors = {
            entry.strip().lower() for entry in self.settings.exchange.disallowed_sectors
        }
        allowed_symbols = self._allowed_symbol_set()
        normalized: list[str] = []
        for raw in symbols:
            symbol = self._normalize_symbol(raw)
            if not symbol:
                continue
            if allowed_symbols and symbol not in allowed_symbols:
                continue
            if symbol in disallowed_symbols:
                continue
            sector = DynamicWatchlistService.SYMBOL_SECTORS.get(symbol, "other")
            if sector in disallowed_sectors:
                continue
            if symbol not in normalized:
                normalized.append(symbol)
        return normalized

    def _latest_training_rows_by_symbol(self, symbols: list[str]) -> dict[str, object]:
        requested = self.normalize_execution_symbols(symbols)
        if not requested:
            return {}
        placeholders = ",".join("?" for _ in requested)
        query = (
            "SELECT id, symbol, trained_with_xgboost, metadata_json FROM training_runs "
            f"WHERE symbol IN ({placeholders}) "
            "ORDER BY symbol ASC, created_at DESC, id DESC"
        )
        latest: dict[str, object] = {}
        with self.storage._conn() as conn:
            rows = conn.execute(query, tuple(requested)).fetchall()
        for row in rows:
            symbol = str(row["symbol"])
            if symbol not in latest:
                latest[symbol] = row
        return latest

    def _latest_training_metadata_by_symbol(self, symbols: list[str]) -> dict[str, dict]:
        rows = self._latest_training_rows_by_symbol(symbols)
        metadata_by_symbol: dict[str, dict] = {}
        for symbol, row in rows.items():
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = {}
            metadata_by_symbol[str(symbol)] = metadata if isinstance(metadata, dict) else {}
        return metadata_by_symbol

    def _latest_walkforward_summary_by_symbol(self, symbols: list[str]) -> dict[str, dict]:
        requested = self.normalize_execution_symbols(symbols)
        if not requested:
            return {}
        placeholders = ",".join("?" for _ in requested)
        query = (
            "SELECT id, symbol, summary_json FROM walkforward_runs "
            f"WHERE symbol IN ({placeholders}) "
            "ORDER BY symbol ASC, created_at DESC, id DESC"
        )
        latest: dict[str, dict] = {}
        with self.storage._conn() as conn:
            rows = conn.execute(query, tuple(requested)).fetchall()
        for row in rows:
            symbol = str(row["symbol"])
            if symbol in latest:
                continue
            try:
                summary = json.loads(row["summary_json"] or "{}")
            except Exception:
                summary = {}
            latest[symbol] = summary if isinstance(summary, dict) else {}
        return latest

    @staticmethod
    def _recent_training_signal(
        training_metadata: dict | None,
        walkforward_summary: dict | None,
    ) -> dict[str, float | bool | str]:
        metadata = dict(training_metadata or {})
        candidate_wf = dict(metadata.get("candidate_walkforward_summary", {}) or {})
        latest_wf = dict(walkforward_summary or {})
        wf = candidate_wf or latest_wf
        promotion_status = str(metadata.get("promotion_status", "") or "").strip().lower()
        promotion_reason = str(metadata.get("promotion_reason", "") or "").strip().lower()
        wf_return = float(wf.get("total_return_pct", 0.0) or 0.0)
        wf_profit_factor = float(wf.get("profit_factor", 0.0) or 0.0)
        wf_win_rate = float(wf.get("avg_win_rate", 0.0) or 0.0)
        holdout_accuracy = float(
            metadata.get(
                "candidate_holdout_accuracy",
                metadata.get("holdout_accuracy", 0.0),
            )
            or 0.0
        )
        positive = (
            promotion_status in {"canary_pending", "promoted"}
            or "candidate_higher_walkforward" in promotion_reason
            or (
                wf_return > 0.0
                and (
                    wf_profit_factor >= 1.0
                    or wf_profit_factor == 0.0
                    or wf_win_rate >= 55.0
                )
            )
        )
        negative = (
            promotion_status == "rejected"
            and promotion_reason
            in {
                "candidate_negative_walkforward_quality",
                "candidate_below_recent_walkforward_baseline",
            }
        ) or (
            wf_return < 0.0
            or (0.0 < wf_profit_factor < 0.95)
            or (wf_win_rate > 0.0 and wf_win_rate < 45.0)
        )
        return {
            "promotion_status": promotion_status,
            "promotion_reason": promotion_reason,
            "walkforward_return_pct": round(wf_return, 4),
            "walkforward_profit_factor": round(wf_profit_factor, 4),
            "walkforward_win_rate": round(wf_win_rate, 4),
            "holdout_accuracy": round(holdout_accuracy, 4),
            "positive": bool(positive),
            "negative": bool(negative),
        }

    @staticmethod
    def _path_signature(path: Path) -> list[str | int | None]:
        try:
            stat = path.stat()
        except OSError:
            return [str(path), None, None]
        return [str(path), int(stat.st_mtime_ns), int(stat.st_size)]

    def model_ready_symbols(self, symbols: list[str]) -> list[str]:
        ready: list[str] = []
        latest_rows = self._latest_training_rows_by_symbol(symbols)
        broken_state = self.storage.get_json_state(
            self.broken_model_symbols_state_key,
            {},
        )
        broken_state = broken_state if isinstance(broken_state, dict) else {}
        broken_state_changed = False
        for symbol in symbols:
            row = latest_rows.get(symbol)
            if not row:
                continue
            if not bool(row["trained_with_xgboost"]):
                continue
            try:
                metadata = json.loads(row["metadata_json"])
            except Exception:
                metadata = {}
            rows = int(metadata.get("rows") or 0)
            if rows < self.settings.training.minimum_training_rows:
                continue
            path = Path(self.runtime_model_path_for_symbol(symbol))
            if not path.exists():
                continue
            try:
                if path.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            broken_entry = broken_state.get(symbol)
            if isinstance(broken_entry, dict):
                current_signature = self._path_signature(path)
                if (
                    broken_entry.get("model_path") == str(path)
                    and broken_entry.get("signature") == current_signature
                ):
                    continue
                broken_state.pop(symbol, None)
                broken_state_changed = True
            ready.append(symbol)
        if broken_state_changed:
            self.storage.set_json_state(
                self.broken_model_symbols_state_key,
                broken_state,
            )
        return ready

    def filter_symbols_by_recent_edge(self, symbols: list[str]) -> list[str]:
        if not symbols:
            return []
        summary = self.symbol_edge_summary()
        shadow_summary = self.shadow_feedback_summary()
        min_samples = self.settings.risk.execution_symbol_min_samples
        floor_pct = self.settings.risk.execution_symbol_accuracy_floor_pct
        filtered: list[str] = []
        removed: list[dict[str, float | int | str]] = []
        for symbol in symbols:
            candidate_stage = self._promotion_candidate_stage(symbol)
            if candidate_stage in {"shadow", "live"}:
                filtered.append(symbol)
                continue
            metrics = summary.get(symbol)
            shadow_metrics = shadow_summary.get(symbol) or {}
            if not metrics and not shadow_metrics:
                filtered.append(symbol)
                continue
            count = int(
                (metrics or {}).get("sample_count", (metrics or {}).get("count", 0))
                or 0
            )
            accuracy = float((metrics or {}).get("accuracy_pct", 0.0) or 0.0)
            objective_score = float((metrics or {}).get("objective_score", 0.0) or 0.0)
            expectancy_pct = float((metrics or {}).get("expectancy_pct", 0.0) or 0.0)
            profit_factor = float((metrics or {}).get("profit_factor", 0.0) or 0.0)
            max_drawdown_pct = float(
                (metrics or {}).get("max_drawdown_pct", 0.0) or 0.0
            )
            shadow_positive, shadow_negative = self._shadow_signal(
                shadow_metrics,
                min_samples=min_samples,
                floor_pct=floor_pct,
            )
            edge_negative = (
                objective_score < 0.0
                or expectancy_pct < -0.05
                or (count >= min_samples and 0.0 < profit_factor < 0.9)
                or max_drawdown_pct > 8.0
            )
            if count >= min_samples and edge_negative and not shadow_positive:
                removed.append(
                    {
                        "symbol": symbol,
                        "count": count,
                        "accuracy_pct": round(accuracy, 2),
                        "objective_score": round(objective_score, 4),
                        "expectancy_pct": round(expectancy_pct, 4),
                        "profit_factor": round(profit_factor, 4),
                        "max_drawdown_pct": round(max_drawdown_pct, 4),
                        "shadow_eval_count": int(
                            shadow_metrics.get("shadow_eval_count", 0) or 0
                        ),
                        "shadow_accuracy_pct": round(
                            float(
                                shadow_metrics.get("shadow_accuracy_pct", 0.0)
                                or 0.0
                            ),
                            2,
                        ),
                        "shadow_trade_count": int(
                            shadow_metrics.get("shadow_trade_count", 0) or 0
                        ),
                        "shadow_avg_pnl_pct": round(
                            float(
                                shadow_metrics.get("shadow_avg_pnl_pct", 0.0)
                                or 0.0
                            ),
                            4,
                        ),
                    }
                )
                continue
            if count < min_samples and shadow_negative and not shadow_positive:
                removed.append(
                    {
                        "symbol": symbol,
                        "count": count,
                        "accuracy_pct": round(accuracy, 2),
                        "objective_score": round(objective_score, 4),
                        "expectancy_pct": round(expectancy_pct, 4),
                        "profit_factor": round(profit_factor, 4),
                        "max_drawdown_pct": round(max_drawdown_pct, 4),
                        "shadow_eval_count": int(
                            shadow_metrics.get("shadow_eval_count", 0) or 0
                        ),
                        "shadow_accuracy_pct": round(
                            float(
                                shadow_metrics.get("shadow_accuracy_pct", 0.0)
                                or 0.0
                            ),
                            2,
                        ),
                        "shadow_trade_count": int(
                            shadow_metrics.get("shadow_trade_count", 0) or 0
                        ),
                        "shadow_avg_pnl_pct": round(
                            float(
                                shadow_metrics.get("shadow_avg_pnl_pct", 0.0)
                                or 0.0
                            ),
                            4,
                        ),
                    }
                )
                continue
            filtered.append(symbol)
        if removed:
            self.storage.insert_execution_event(
                "edge_filter",
                "SYSTEM",
                {
                    "removed_symbols": removed,
                    "floor_pct": floor_pct,
                    "min_samples": min_samples,
                },
            )
        return filtered

    def filter_active_symbols_by_model_readiness(self, symbols: list[str]) -> list[str]:
        ready = self.model_ready_symbols(symbols)
        self.storage.set_json_state("model_ready_symbols", ready)
        consistency_summary = self.consistency_summary(ready)
        blocked_symbols = sorted(
            symbol
            for symbol, payload in consistency_summary.items()
            if self._blocking_consistency_flags(payload.get("flags", []) or [])
        )
        if blocked_symbols:
            self.storage.insert_execution_event(
                "consistency_filter",
                "SYSTEM",
                {
                    "blocked_symbols": blocked_symbols,
                    "details": {
                        symbol: self._blocking_consistency_flags(
                            (consistency_summary.get(symbol) or {}).get("flags", []) or []
                        )
                        for symbol in blocked_symbols
                    },
                },
            )
        self.storage.set_json_state(
            "consistency_blocked_details",
            {
                symbol: self._blocking_consistency_flags(
                    (consistency_summary.get(symbol) or {}).get("flags", []) or []
                )
                for symbol in blocked_symbols
            },
        )
        filtered_ready = [
            symbol for symbol in ready if symbol not in set(blocked_symbols)
        ]
        self.storage.set_json_state("consistency_blocked_symbols", blocked_symbols)
        edge_filtered = self.filter_symbols_by_recent_edge(filtered_ready)
        self.storage.set_json_state("edge_qualified_symbols", edge_filtered)
        filtered = self._fast_alpha_core_active_symbols(filtered_ready, edge_filtered)
        filtered = self._paper_exploration_active_symbols(filtered_ready, filtered)
        self.storage.set_json_state("active_symbols", filtered)
        return filtered

    def _fast_alpha_core_active_symbols(
        self,
        ready: list[str],
        filtered: list[str],
    ) -> list[str]:
        if self.settings.app.runtime_mode != "paper":
            self.storage.set_json_state("fast_alpha_active_symbols", [])
            return filtered
        if not bool(getattr(self.settings.strategy, "fast_alpha_enabled", False)):
            self.storage.set_json_state("fast_alpha_active_symbols", [])
            return filtered
        fast_alpha_symbols = set(
            self.normalize_execution_symbols(
                list(getattr(self.settings.strategy, "fast_alpha_symbols", []) or [])
            )
        )
        if not fast_alpha_symbols:
            self.storage.set_json_state("fast_alpha_active_symbols", [])
            return filtered
        max_active = max(1, int(self.settings.exchange.max_active_symbols))
        selected = list(filtered[:max_active])
        active_fast_alpha = [
            symbol for symbol in selected if symbol in fast_alpha_symbols
        ]
        self.storage.set_json_state("fast_alpha_active_symbols", active_fast_alpha)
        return selected

    def _paper_exploration_active_symbols(
        self,
        ready: list[str],
        filtered: list[str],
    ) -> list[str]:
        if self.settings.app.runtime_mode != "paper":
            self.storage.set_json_state("paper_exploration_active_symbols", [])
            return filtered
        if not ready:
            self.storage.set_json_state("paper_exploration_active_symbols", [])
            return filtered
        performance = self.performance_getter()
        build = getattr(performance, "build", None)
        if not callable(build):
            self.storage.set_json_state("paper_exploration_active_symbols", [])
            return filtered
        try:
            snapshot = build()
        except Exception:
            self.storage.set_json_state("paper_exploration_active_symbols", [])
            return filtered
        total_closed = int(getattr(snapshot, "total_closed_trades", 0) or 0)
        if total_closed >= 3:
            self.storage.set_json_state("paper_exploration_active_symbols", [])
            return filtered

        core_symbols = set(self.normalize_execution_symbols(self.settings.exchange.core_symbols))
        has_core = any(symbol in core_symbols for symbol in filtered)
        if has_core and filtered:
            self.storage.set_json_state("paper_exploration_active_symbols", [])
            return filtered

        shadow_summary = self.shadow_feedback_summary(limit=500)
        exploration_candidates = [
            symbol for symbol in ready if symbol in core_symbols
        ] or list(ready)
        selected = list(filtered)
        exploration_symbols: list[str] = []
        for symbol in exploration_candidates:
            if symbol in selected:
                continue
            shadow_positive, shadow_negative = self._shadow_signal(
                shadow_summary.get(symbol) or {},
                min_samples=self.settings.risk.execution_symbol_min_samples,
                floor_pct=self.settings.risk.execution_symbol_accuracy_floor_pct,
            )
            if shadow_negative and not shadow_positive:
                continue
            selected.append(symbol)
            exploration_symbols.append(symbol)
            break
        self.storage.set_json_state("paper_exploration_active_symbols", exploration_symbols)
        return selected

    def symbol_accuracy_summary(
        self,
        limit: int = 500,
    ) -> dict[str, dict[str, float | int]]:
        performance = self.performance_getter()
        build_summary = getattr(performance, "build_symbol_accuracy_summary", None)
        if not callable(build_summary):
            return {}
        try:
            return build_summary(limit=limit)
        except TypeError:
            return build_summary()

    def symbol_edge_summary(
        self,
        limit: int = 500,
    ) -> dict[str, dict[str, float | int]]:
        performance = self.performance_getter()
        build_summary = getattr(performance, "build_symbol_edge_summary", None)
        if callable(build_summary):
            try:
                return build_summary(limit=limit)
            except TypeError:
                return build_summary()
        fallback = self.symbol_accuracy_summary(limit=limit)
        converted: dict[str, dict[str, float | int]] = {}
        for symbol, metrics in fallback.items():
            count = int(metrics.get("count", 0) or 0)
            accuracy_pct = float(metrics.get("accuracy_pct", 0.0) or 0.0)
            accuracy = accuracy_pct / 100.0 if accuracy_pct else 0.0
            sample_factor = min(count, 8) / 16 if count else 0.0
            objective_score = (accuracy - 0.5) * min(count, 8)
            converted[symbol] = {
                "count": count,
                "sample_count": count,
                "executed_count": 0,
                "accuracy": accuracy,
                "accuracy_pct": accuracy_pct,
                "expectancy_pct": 0.0,
                "avg_trade_return_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "trade_win_rate": accuracy,
                "avg_cost_pct": 0.0,
                "objective_score": objective_score,
                "objective_quality": (
                    objective_score / sample_factor if sample_factor > 0 else 0.0
                ),
            }
        return converted

    def execution_pool_target_size(self) -> int:
        configured = max(1, int(self.settings.risk.execution_pool_target_size))
        max_active = max(1, int(self.settings.exchange.max_active_symbols))
        return min(configured, max_active)

    def consistency_summary(
        self,
        symbols: list[str],
    ) -> dict[str, dict[str, float | int | str | list[str]]]:
        getter = self.consistency_getter
        if not callable(getter):
            return {}
        service = getter()
        build_rows = getattr(service, "build_symbol_consistency_rows", None)
        if not callable(build_rows):
            return {}
        try:
            return build_rows(symbols)
        except TypeError:
            return build_rows(symbols=symbols)

    def execution_pool_candidate_universe(
        self,
        current_symbols: list[str],
        summary: dict[str, dict[str, float | int]],
    ) -> list[str]:
        combined = list(current_symbols)
        combined.extend(str(symbol) for symbol in summary.keys())
        combined.extend(
            str(symbol)
            for symbol in self._promotion_candidates_state(
                self.storage.get_json_state("model_promotion_candidates", {})
            ).keys()
        )
        combined.extend(self.settings.exchange.core_symbols)
        combined.extend(self.settings.exchange.symbols)
        combined.extend(self.settings.exchange.candidate_symbols)
        return self.normalize_execution_symbols(list(dict.fromkeys(combined)))

    def rank_execution_pool_candidates(
        self,
        symbols: list[str],
        current_symbols: list[str],
        summary: dict[str, dict[str, float | int]],
    ) -> list[dict[str, float | int | str | bool]]:
        if not symbols:
            return []
        min_samples = self.settings.risk.execution_symbol_min_samples
        floor_pct = self.settings.risk.execution_symbol_accuracy_floor_pct
        core_symbols = set(self.normalize_execution_symbols(self.settings.exchange.core_symbols))
        current_set = set(current_symbols)
        ready_symbols = set(self.model_ready_symbols(symbols))
        shadow_summary = self.shadow_feedback_summary(limit=1000)
        consistency_summary = self.consistency_summary(symbols)
        training_metadata = self._latest_training_metadata_by_symbol(symbols)
        walkforward_summary = self._latest_walkforward_summary_by_symbol(symbols)
        ranked: list[dict[str, float | int | str | bool]] = []

        for symbol in symbols:
            metrics = summary.get(symbol) or {}
            count = int(metrics.get("sample_count", metrics.get("count", 0)) or 0)
            accuracy = round(float(metrics.get("accuracy_pct", 0.0)), 2) if metrics else 0.0
            objective_score = round(float(metrics.get("objective_score", 0.0) or 0.0), 4)
            expectancy_pct = round(float(metrics.get("expectancy_pct", 0.0) or 0.0), 4)
            profit_factor = round(float(metrics.get("profit_factor", 0.0) or 0.0), 4)
            max_drawdown_pct = round(
                float(metrics.get("max_drawdown_pct", 0.0) or 0.0),
                4,
            )
            shadow_metrics = shadow_summary.get(symbol) or {}
            shadow_eval_count = int(shadow_metrics.get("shadow_eval_count", 0) or 0)
            shadow_accuracy = round(
                float(shadow_metrics.get("shadow_accuracy_pct", 0.0) or 0.0),
                2,
            )
            shadow_trade_count = int(shadow_metrics.get("shadow_trade_count", 0) or 0)
            shadow_avg_pnl_pct = round(
                float(shadow_metrics.get("shadow_avg_pnl_pct", 0.0) or 0.0),
                4,
            )
            training_signal = self._recent_training_signal(
                training_metadata.get(symbol) or {},
                walkforward_summary.get(symbol) or {},
            )
            shadow_positive, shadow_negative = self._shadow_signal(
                shadow_metrics,
                min_samples=min_samples,
                floor_pct=floor_pct,
            )
            candidate_stage = self._promotion_candidate_stage(symbol)
            is_core = symbol in core_symbols
            has_model = symbol in ready_symbols
            is_current = symbol in current_set
            consistency = consistency_summary.get(symbol) or {}
            consistency_flags = list(consistency.get("flags", []) or [])
            blocking_consistency_flags = self._blocking_consistency_flags(
                consistency_flags
            )
            consistency_blocked = bool(blocking_consistency_flags)
            edge_positive = (
                objective_score > 0.0
                or (expectancy_pct > 0.0 and (profit_factor >= 1.0 or profit_factor == 0.0))
            )
            edge_negative = (
                objective_score < 0.0
                or expectancy_pct < -0.05
                or (count >= min_samples and 0.0 < profit_factor < 0.9)
                or max_drawdown_pct > 8.0
            )

            if candidate_stage == "live":
                status = "candidate_live"
            elif candidate_stage == "shadow":
                status = "candidate_shadow"
            elif consistency_blocked:
                status = "consistency_blocked"
            elif bool(training_signal["positive"]) and has_model:
                status = "training_qualified"
            elif count >= min_samples:
                status = "qualified" if edge_positive else "disqualified"
                if status == "disqualified" and shadow_positive:
                    status = "shadow_qualified"
            elif shadow_positive:
                status = "shadow_qualified"
            elif bool(training_signal["negative"]):
                status = "training_negative"
            elif count > 0:
                status = "provisional" if not edge_negative else "provisional_negative"
            elif shadow_negative:
                status = "shadow_negative"
            else:
                status = "unseen"

            if (
                candidate_stage not in {"live", "shadow"}
                and bool(training_signal["negative"])
                and not bool(training_signal["positive"])
                and status in {"qualified", "shadow_qualified", "provisional", "unseen"}
            ):
                status = "training_negative"

            status_priority = {
                "candidate_live": 0,
                "candidate_shadow": 1,
                "qualified": 2,
                "training_qualified": 3,
                "shadow_qualified": 4,
                "provisional": 5,
                "unseen": 6,
                "shadow_negative": 7,
                "consistency_blocked": 8,
                "training_negative": 9,
                "provisional_negative": 10,
                "disqualified": 11,
            }[status]
            sort_key = (
                status_priority,
                -shadow_eval_count if status in {"candidate_live", "candidate_shadow", "shadow_qualified"} else 0,
                -float(training_signal["walkforward_return_pct"]),
                -float(training_signal["walkforward_profit_factor"]),
                -float(training_signal["walkforward_win_rate"]),
                -float(training_signal["holdout_accuracy"]),
                -objective_score,
                -expectancy_pct,
                -profit_factor,
                max_drawdown_pct,
                -count if status == "qualified" else 0,
                -shadow_accuracy,
                -accuracy,
                -shadow_trade_count,
                -shadow_avg_pnl_pct,
                -count,
                0 if has_model else 1,
                0 if is_core else 1,
                0 if is_current else 1,
                symbol,
            )
            ranked.append(
                {
                    "symbol": symbol,
                    "status": status,
                    "candidate_stage": candidate_stage,
                    "count": count,
                    "accuracy_pct": accuracy,
                    "objective_score": objective_score,
                    "expectancy_pct": expectancy_pct,
                    "profit_factor": profit_factor,
                    "max_drawdown_pct": max_drawdown_pct,
                    "shadow_eval_count": shadow_eval_count,
                    "shadow_accuracy_pct": shadow_accuracy,
                    "shadow_trade_count": shadow_trade_count,
                    "shadow_avg_pnl_pct": shadow_avg_pnl_pct,
                    "recent_training_status": str(training_signal["promotion_status"]),
                    "recent_training_reason": str(training_signal["promotion_reason"]),
                    "recent_walkforward_return_pct": float(
                        training_signal["walkforward_return_pct"]
                    ),
                    "recent_walkforward_profit_factor": float(
                        training_signal["walkforward_profit_factor"]
                    ),
                    "recent_walkforward_win_rate": float(
                        training_signal["walkforward_win_rate"]
                    ),
                    "recent_holdout_accuracy": float(
                        training_signal["holdout_accuracy"]
                    ),
                    "consistency_flags": consistency_flags,
                    "blocking_consistency_flags": blocking_consistency_flags,
                    "is_core": is_core,
                    "has_model": has_model,
                    "is_current": is_current,
                    "sort_key": sort_key,
                }
            )
        ranked.sort(key=lambda item: item["sort_key"])
        for item in ranked:
            item.pop("sort_key", None)
        return ranked

    @staticmethod
    def select_rebuilt_execution_symbols(
        ranked_candidates: list[dict[str, float | int | str | bool]],
        target_size: int,
    ) -> list[str]:
        selected: list[str] = []
        for status in (
            "candidate_live",
            "candidate_shadow",
            "qualified",
            "training_qualified",
            "shadow_qualified",
            "provisional",
        ):
            for candidate in ranked_candidates:
                if candidate["status"] != status:
                    continue
                symbol = str(candidate["symbol"])
                if symbol not in selected:
                    selected.append(symbol)
                if len(selected) >= target_size:
                    return selected

        if selected:
            return selected

        for status in ("unseen", "shadow_negative", "provisional_negative", "disqualified"):
            for candidate in ranked_candidates:
                if candidate["status"] != status:
                    continue
                symbol = str(candidate["symbol"])
                if symbol not in selected:
                    selected.append(symbol)
                    return selected
        return selected

    def get_execution_symbols(self) -> list[str]:
        stored = self.storage.get_json_state(self.execution_symbols_state_key, None)
        if isinstance(stored, list) and stored:
            symbols = self.normalize_execution_symbols([str(symbol) for symbol in stored])
        else:
            migrated = self.storage.get_json_state("active_symbols", None)
            if isinstance(migrated, list) and migrated:
                symbols = self.normalize_execution_symbols([str(symbol) for symbol in migrated])
            else:
                symbols = self.normalize_execution_symbols(
                    list(self.settings.exchange.symbols)
                )
        self.storage.set_json_state(self.execution_symbols_state_key, symbols)
        return symbols

    def _fast_alpha_core_execution_symbols(
        self,
        *,
        ranked_candidates: list[dict[str, float | int | str | bool]],
        selected_symbols: list[str],
        target_size: int,
    ) -> list[str]:
        if self.settings.app.runtime_mode != "paper":
            return list(selected_symbols)
        if not bool(getattr(self.settings.strategy, "fast_alpha_enabled", False)):
            return list(selected_symbols)
        fast_alpha_symbols = self.normalize_execution_symbols(
            list(getattr(self.settings.strategy, "fast_alpha_symbols", []) or [])
        )
        if not fast_alpha_symbols:
            return list(selected_symbols)
        ranked_by_symbol = {
            str(row["symbol"]): row
            for row in ranked_candidates
            if isinstance(row, dict)
        }
        eligible_statuses = {
            "candidate_live",
            "candidate_shadow",
            "qualified",
            "training_qualified",
            "shadow_qualified",
            "provisional",
        }
        merged = list(selected_symbols)
        for symbol in fast_alpha_symbols:
            row = ranked_by_symbol.get(symbol) or {}
            if not row:
                continue
            if not bool(row.get("has_model")):
                continue
            if str(row.get("status") or "") not in eligible_statuses:
                continue
            if list(row.get("consistency_flags", []) or []):
                continue
            if symbol not in merged:
                merged.append(symbol)
            if len(merged) >= target_size:
                break
        return merged[:target_size]

    def backfill_symbols(
        self,
        symbols: list[str],
        days: int = 180,
    ) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        since_ms = now_ms - days * 24 * 60 * 60 * 1000
        market = self.market_getter()
        for symbol in symbols:
            market_symbol = f"{symbol}:USDT" if ":USDT" not in symbol else symbol
            summary[symbol] = {}
            for timeframe in self.settings.exchange.timeframes:
                candles = market.fetch_historical_ohlcv(
                    market_symbol,
                    timeframe,
                    since=since_ms,
                    limit=500,
                )
                summary[symbol][timeframe] = len(candles)
        return summary

    def set_execution_symbols(
        self,
        symbols: list[str],
        backfill_days: int = 180,
        train_models: bool = True,
        action: str = "set",
    ) -> dict:
        now = datetime.now(timezone.utc)
        normalized = self.normalize_execution_symbols(symbols)
        previous = self.get_execution_symbols()
        added = [symbol for symbol in normalized if symbol not in previous]
        removed = [symbol for symbol in previous if symbol not in normalized]
        for symbol in removed:
            self.clear_symbol_models(symbol)
            self.clear_broken_model_symbol(symbol)
        self.storage.set_json_state(self.execution_symbols_state_key, normalized)

        backfill_summary = self.backfill_symbols(added, backfill_days) if added else {}
        training_summaries = []
        trainer = self.trainer_getter()
        if train_models:
            for symbol in added:
                summary = trainer.train_symbol(symbol)
                self.clear_symbol_models(symbol)
                payload = summary.__dict__.copy()
                payload["report"] = trainer.render_report(
                    summary,
                    lang=self.current_language(),
                )
                self.storage.insert_training_run(payload)
                if callable(self.handle_training_summary):
                    self.handle_training_summary(
                        symbol=symbol,
                        summary=summary,
                        now=now,
                        payload=payload,
                    )
                training_summaries.append(payload)

        active = self.filter_active_symbols_by_model_readiness(normalized)
        result = {
            "execution_symbols": normalized,
            "model_ready_symbols": active,
            "added_symbols": added,
            "removed_symbols": removed,
            "backfill_summary": backfill_summary,
            "training_summaries": training_summaries,
        }
        self.storage.insert_execution_event(
            "execution_pool_update",
            "SYSTEM",
            {
                "action": action,
                "execution_symbols": normalized,
                "model_ready_symbols": active,
                "added_symbols": added,
                "removed_symbols": removed,
            },
        )
        self.notifier.notify(
            "execution_pool_update",
            "执行池已更新",
            (
                f"action={action}\n"
                f"execution_symbols={', '.join(normalized) or 'none'}\n"
                f"model_ready_symbols={', '.join(active) or 'none'}\n"
                f"added={', '.join(added) or 'none'}\n"
                f"removed={', '.join(removed) or 'none'}"
            ),
            level="info",
        )
        return result

    def add_execution_symbols(self, symbols: list[str], backfill_days: int = 180) -> dict:
        current = self.get_execution_symbols()
        merged = list(dict.fromkeys(current + self.normalize_execution_symbols(symbols)))
        return self.set_execution_symbols(
            merged,
            backfill_days=backfill_days,
            train_models=True,
            action="add",
        )

    def remove_execution_symbols(self, symbols: list[str]) -> dict:
        current = self.get_execution_symbols()
        removal = set(self.normalize_execution_symbols(symbols))
        remaining = [symbol for symbol in current if symbol not in removal]
        return self.set_execution_symbols(
            remaining,
            backfill_days=0,
            train_models=False,
            action="remove",
        )

    def get_active_symbols(
        self,
        force_refresh: bool = False,
        now: datetime | None = None,
    ) -> list[str]:
        if force_refresh:
            self.watchlist_getter().refresh(force=True, now=now)
        return self.filter_active_symbols_by_model_readiness(self.get_execution_symbols())

    def get_watchlist_snapshot(
        self,
        force_refresh: bool = False,
        now: datetime | None = None,
    ) -> dict:
        snapshot = self.watchlist_getter().refresh(force=force_refresh, now=now)
        execution_symbols = self.get_execution_symbols()
        filtered_active = self.filter_active_symbols_by_model_readiness(execution_symbols)
        return {
            "active_symbols": filtered_active,
            "raw_active_symbols": snapshot.active_symbols,
            "execution_symbols": execution_symbols,
            "model_ready_symbols": self.storage.get_json_state("model_ready_symbols", []),
            "added_symbols": snapshot.added_symbols,
            "removed_symbols": snapshot.removed_symbols,
            "whitelist": snapshot.whitelist,
            "blacklist": snapshot.blacklist,
            "candidates": [candidate.__dict__ for candidate in snapshot.candidates],
            "refreshed_at": snapshot.refreshed_at,
            "refresh_reason": snapshot.refresh_reason,
        }

    def run_watchlist_refresh(self) -> dict:
        return self.get_watchlist_snapshot(force_refresh=True)

    def rebuild_execution_symbols(
        self,
        force: bool = False,
        now: datetime | None = None,
        reason: str = "manual",
    ) -> dict:
        now = now or datetime.now(timezone.utc)
        current_symbols = self.get_execution_symbols()
        interval_hours = int(self.settings.risk.execution_pool_rebuild_interval_hours)
        if not force and interval_hours <= 0:
            return {
                "status": "skipped",
                "reason": "execution_pool_rebuild_disabled",
                "execution_symbols": current_symbols,
                "changed": False,
            }

        summary = self.symbol_edge_summary(limit=1000)
        universe = self.execution_pool_candidate_universe(current_symbols, summary)
        target_size = self.execution_pool_target_size()
        ranked_candidates = self.rank_execution_pool_candidates(
            universe,
            current_symbols,
            summary,
        )
        selected_symbols = self.select_rebuilt_execution_symbols(
            ranked_candidates,
            target_size,
        )
        selected_symbols = self._fast_alpha_core_execution_symbols(
            ranked_candidates=ranked_candidates,
            selected_symbols=selected_symbols,
            target_size=target_size,
        )
        if not selected_symbols:
            fallback = self.normalize_execution_symbols(self.settings.exchange.core_symbols)
            selected_symbols = fallback[:1] or current_symbols[:1]
        changed = selected_symbols != current_symbols
        plan = {
            "reason": reason,
            "target_size": target_size,
            "current_symbols": current_symbols,
            "selected_symbols": selected_symbols,
            "ranked_candidates": ranked_candidates,
            "changed": changed,
            "rebuilt_at": now.isoformat(),
        }
        self.storage.set_state(
            self.execution_pool_last_rebuild_at_state_key,
            now.isoformat(),
        )

        if not changed:
            self.storage.insert_execution_event(
                "execution_pool_rebuild",
                "SYSTEM",
                plan,
            )
            return {"status": "ok", **plan, "execution_symbols": current_symbols}

        train_models = any(symbol not in current_symbols for symbol in selected_symbols)
        result = self.set_execution_symbols(
            selected_symbols,
            backfill_days=180 if train_models else 0,
            train_models=train_models,
            action="auto_rebuild",
        )
        self.storage.insert_execution_event(
            "execution_pool_rebuild",
            "SYSTEM",
            {
                **plan,
                "added_symbols": result["added_symbols"],
                "removed_symbols": result["removed_symbols"],
                "model_ready_symbols": result["model_ready_symbols"],
            },
        )
        return {"status": "ok", **plan, **result}

    def maybe_rebuild_execution_pool(
        self,
        now: datetime,
        active_symbols: list[str],
    ) -> dict | None:
        if not active_symbols:
            try:
                return self.rebuild_execution_symbols_callback(
                    force=True,
                    now=now,
                    reason="no_active_symbols",
                )
            except Exception as exc:
                logger.exception(f"Execution pool rebuild failed: {exc}")
                return {
                    "status": "failed",
                    "reason": "no_active_symbols",
                    "changed": False,
                }

        interval_hours = int(self.settings.risk.execution_pool_rebuild_interval_hours)
        if interval_hours <= 0:
            return None

        last_rebuild_at = self.parse_iso_datetime(
            self.storage.get_state(self.execution_pool_last_rebuild_at_state_key, "")
        )
        if last_rebuild_at and now - last_rebuild_at < timedelta(hours=interval_hours):
            return None

        try:
            return self.rebuild_execution_symbols_callback(
                force=False,
                now=now,
                reason="scheduled_rebuild",
            )
        except Exception as exc:
            logger.exception(f"Execution pool rebuild failed: {exc}")
            return {
                "status": "failed",
                "reason": "scheduled_rebuild",
                "changed": False,
            }
