"""Shadow observation and delayed-evaluation runtime helpers."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from config import Settings, get_settings
from core.models import MarketRegime, TradeReflection
from core.storage import Storage
from learning.experience_store import ExperienceStore
from monitor.performance_report import PerformanceReporter


class ShadowRuntimeService:
    """Manage shadow observation ranking, evaluation, and blocked-trade bookkeeping."""

    ESTIMATED_TRADE_COST_PCT = 0.15

    def __init__(self, storage: Storage, settings: Settings | None = None):
        self.storage = storage
        self.settings = settings or get_settings()

    def build_observation_feedback(
        self,
        limit: int = 500,
    ) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        with self.storage._conn() as conn:
            evaluation_rows = conn.execute(
                """SELECT symbol, is_correct
                   FROM prediction_evaluations
                   WHERE evaluation_type = 'shadow_observation'
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            for row in evaluation_rows:
                symbol = str(row["symbol"])
                bucket = summary.setdefault(
                    symbol,
                    {
                        "shadow_eval_count": 0,
                        "shadow_eval_correct": 0,
                        "shadow_accuracy_pct": 0.0,
                        "shadow_trade_count": 0,
                        "shadow_trade_positive": 0,
                        "shadow_positive_ratio": 0.0,
                        "shadow_avg_pnl_pct": 0.0,
                    },
                )
                bucket["shadow_eval_count"] += 1
                bucket["shadow_eval_correct"] += int(row["is_correct"] or 0)

            trade_rows = conn.execute(
                """SELECT symbol, pnl_pct
                   FROM shadow_trade_runs
                   WHERE status = 'evaluated'
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            for row in trade_rows:
                symbol = str(row["symbol"])
                bucket = summary.setdefault(
                    symbol,
                    {
                        "shadow_eval_count": 0,
                        "shadow_eval_correct": 0,
                        "shadow_accuracy_pct": 0.0,
                        "shadow_trade_count": 0,
                        "shadow_trade_positive": 0,
                        "shadow_positive_ratio": 0.0,
                        "shadow_avg_pnl_pct": 0.0,
                        "_shadow_pnl_total": 0.0,
                    },
                )
                bucket.setdefault("_shadow_pnl_total", 0.0)
                bucket["shadow_trade_count"] += 1
                pnl_pct = float(row["pnl_pct"] or 0.0)
                bucket["_shadow_pnl_total"] += pnl_pct
                bucket["shadow_trade_positive"] += int(pnl_pct > 0)

        for bucket in summary.values():
            eval_count = int(bucket.get("shadow_eval_count", 0))
            eval_correct = int(bucket.pop("shadow_eval_correct", 0))
            if eval_count:
                bucket["shadow_accuracy_pct"] = round(
                    eval_correct / eval_count * 100,
                    2,
                )
            trade_count = int(bucket.get("shadow_trade_count", 0))
            positive_count = int(bucket.get("shadow_trade_positive", 0))
            pnl_total = float(bucket.pop("_shadow_pnl_total", 0.0))
            if trade_count:
                bucket["shadow_positive_ratio"] = round(
                    positive_count / trade_count,
                    4,
                )
                bucket["shadow_avg_pnl_pct"] = round(
                    pnl_total / trade_count,
                    4,
                )
        return summary

    def prioritize_observation_candidates(
        self,
        ranked_candidates: list[dict[str, float | int | str | bool]],
        shadow_feedback: dict[str, dict[str, float | int]],
        floor_pct: float,
    ) -> list[dict[str, float | int | str | bool]]:
        if not ranked_candidates:
            return []
        prioritized: list[dict[str, float | int | str | bool]] = []
        for index, candidate in enumerate(ranked_candidates):
            item = dict(candidate)
            symbol = str(item["symbol"])
            feedback = shadow_feedback.get(symbol, {})
            shadow_eval_count = int(feedback.get("shadow_eval_count", 0))
            shadow_accuracy_pct = float(feedback.get("shadow_accuracy_pct", 0.0))
            shadow_trade_count = int(feedback.get("shadow_trade_count", 0))
            shadow_positive_ratio = float(feedback.get("shadow_positive_ratio", 0.0))
            shadow_avg_pnl_pct = float(feedback.get("shadow_avg_pnl_pct", 0.0))
            positive_signal = (
                (shadow_eval_count >= 3 and shadow_accuracy_pct >= floor_pct)
                or (
                    shadow_trade_count >= 3
                    and shadow_positive_ratio >= 0.6
                    and shadow_avg_pnl_pct > 0
                )
            )
            negative_signal = (
                shadow_eval_count >= 3 and shadow_accuracy_pct < floor_pct * 0.8
            ) or (
                shadow_trade_count >= 3
                and shadow_positive_ratio < 0.4
                and shadow_avg_pnl_pct < 0
            )
            item["shadow_eval_count"] = shadow_eval_count
            item["shadow_accuracy_pct"] = shadow_accuracy_pct
            item["shadow_trade_count"] = shadow_trade_count
            item["shadow_avg_pnl_pct"] = shadow_avg_pnl_pct
            item["_shadow_sort_key"] = (
                0 if positive_signal else 2 if negative_signal else 1,
                -shadow_eval_count if positive_signal else 0,
                -shadow_accuracy_pct,
                -shadow_trade_count,
                -shadow_avg_pnl_pct,
                index,
            )
            prioritized.append(item)
        prioritized.sort(key=lambda item: item["_shadow_sort_key"])
        for item in prioritized:
            item.pop("_shadow_sort_key", None)
        return prioritized

    def evaluate_matured_predictions(
        self,
        now: datetime,
        limit: int = 1000,
    ) -> dict:
        reporter = PerformanceReporter(self.storage, self.settings)
        timeframe = self.settings.strategy.primary_timeframe
        evaluated = 0
        with self.storage._conn() as conn:
            rows = conn.execute(
                "SELECT pr.id, pr.symbol, pr.timestamp, pr.up_probability, "
                "pr.research_json, pr.decision_json, pr.created_at, "
                "COALESCE(json_extract(pr.decision_json, '$.pipeline_mode'), 'execution') "
                "AS evaluation_type "
                "FROM prediction_runs pr "
                "LEFT JOIN prediction_evaluations pe "
                "ON pe.symbol = pr.symbol "
                "AND pe.timestamp = pr.timestamp "
                "AND pe.evaluation_type = "
                "COALESCE(json_extract(pr.decision_json, '$.pipeline_mode'), 'execution') "
                "WHERE pe.id IS NULL "
                "ORDER BY pr.created_at ASC, pr.id ASC LIMIT ?",
                (limit,),
            ).fetchall()
            rows = self._dedupe_pending_prediction_rows(rows)
            for row in rows:
                decision = json.loads(row["decision_json"])
                evaluation_type = self._prediction_evaluation_type(row)
                timestamp = datetime.fromisoformat(row["timestamp"])
                horizon_hours = reporter._decision_horizon_hours(decision)
                horizon = timedelta(hours=horizon_hours)
                if timestamp + horizon > now:
                    continue
                outcome = reporter._simulate_prediction_outcome(
                    conn,
                    symbol=row["symbol"],
                    timestamp=timestamp,
                    timeframe=timeframe,
                    decision=decision,
                    estimated_cost_pct=self.ESTIMATED_TRADE_COST_PCT,
                )
                if outcome is None:
                    continue
                threshold = float(
                    decision.get(
                        "xgboost_threshold",
                        self.settings.model.xgboost_probability_threshold,
                    )
                )
                predicted_up = float(row["up_probability"]) >= threshold
                actual_up = bool(outcome["actual_up"])
                metadata = dict(outcome["metadata"])
                trade_taken = bool(decision.get("should_execute", predicted_up))
                metadata["trade_taken"] = trade_taken
                if not trade_taken:
                    metadata["cost_pct"] = 0.0
                    metadata["trade_net_return_pct"] = 0.0
                self.storage.insert_prediction_evaluation(
                    {
                        "symbol": row["symbol"],
                        "timestamp": row["timestamp"],
                        "evaluation_type": evaluation_type,
                        "actual_up": actual_up,
                        "predicted_up": predicted_up,
                        "is_correct": predicted_up == actual_up,
                        "entry_close": float(outcome["entry_close"]),
                        "future_close": float(outcome["future_close"]),
                        "metadata": {
                            "regime": decision.get("regime"),
                            "validation_reason": decision.get("validation_reason", "ok"),
                            "setup_profile": decision.get("setup_profile", {}),
                            "final_score": decision.get("final_score"),
                            **metadata,
                        },
                    }
                )
                evaluated += 1
        if evaluated:
            self.storage.insert_execution_event(
                "prediction_evaluation",
                "SYSTEM",
                {
                    "evaluated_count": evaluated,
                    "evaluated_at": now.isoformat(),
                },
            )
        return {"evaluated_count": evaluated}

    @staticmethod
    def _dedupe_pending_prediction_rows(rows) -> list:
        ordered_keys: list[tuple[str, str]] = []
        latest_by_key: dict[tuple[str, str], object] = {}
        for row in rows:
            key = (
                str(row["symbol"]),
                str(row["timestamp"]),
                ShadowRuntimeService._prediction_evaluation_type(row),
            )
            if key not in latest_by_key:
                ordered_keys.append(key)
            latest_by_key[key] = row
        return [latest_by_key[key] for key in ordered_keys]

    @staticmethod
    def _prediction_evaluation_type(row) -> str:
        keys = row.keys() if hasattr(row, "keys") else ()
        if "evaluation_type" in keys and row["evaluation_type"]:
            return str(row["evaluation_type"])
        try:
            decision = json.loads(row["decision_json"] or "{}")
        except Exception:
            decision = {}
        return str(decision.get("pipeline_mode") or "execution")

    def record_blocked_shadow_trade(
        self,
        symbol: str,
        features,
        prediction,
        decision,
        validation,
        review,
        risk_result,
        entry_price: float,
        xgboost_threshold: float,
        final_score_threshold: float,
    ) -> None:
        block_reason = ""
        if "setup_auto_pause" in getattr(review, "reasons", []):
            block_reason = "setup_auto_pause"
        elif not validation.ok and (
            prediction.up_probability >= xgboost_threshold
            or review.raw_action == "OPEN_LONG"
        ):
            block_reason = f"cross_validation_block:{validation.reason}"
        elif not risk_result.allowed and review.raw_action == "OPEN_LONG":
            block_reason = f"risk_block:{risk_result.reason}"
        elif self.settings.strategy.near_miss_shadow_enabled:
            block_reason = self._near_miss_block_reason(
                prediction=prediction,
                decision=decision,
                review=review,
                validation=validation,
                risk_result=risk_result,
                xgboost_threshold=xgboost_threshold,
                final_score_threshold=final_score_threshold,
            )
        if not block_reason:
            return
        self.storage.insert_shadow_trade_run(
            {
                "symbol": symbol,
                "timestamp": features.timestamp.isoformat(),
                "block_reason": block_reason,
                "direction": "LONG",
                "entry_price": entry_price,
                "horizon_hours": int(
                    self.settings.strategy.max_hold_hours
                    or self.settings.training.prediction_horizon_hours
                ),
                "status": "open",
                "setup_profile": getattr(review, "setup_profile", {}),
                "metadata": {
                    "prediction_up_probability": prediction.up_probability,
                    "final_score": getattr(decision, "final_score", 0.0),
                    "review_score": getattr(review, "review_score", 0.0),
                    "validation_reason": getattr(validation, "reason", "ok"),
                    "reviewed_action": getattr(review, "reviewed_action", ""),
                    "portfolio_rating": getattr(decision, "portfolio_rating", ""),
                    "stop_loss_pct": getattr(decision, "stop_loss_pct", 0.0),
                    "take_profit_levels": list(
                        getattr(decision, "take_profit_levels", []) or []
                    ),
                },
            }
        )

    def _near_miss_block_reason(
        self,
        *,
        prediction,
        decision,
        review,
        validation,
        risk_result,
        xgboost_threshold: float,
        final_score_threshold: float,
    ) -> str:
        if not getattr(validation, "ok", True):
            return ""
        raw_action = str(getattr(review, "raw_action", "") or "")
        reviewed_action = str(getattr(review, "reviewed_action", "") or "")

        up_probability = float(getattr(prediction, "up_probability", 0.0) or 0.0)
        final_score = float(getattr(decision, "final_score", 0.0) or 0.0)
        xgb_gap = max(0.0, float(xgboost_threshold) - up_probability)
        score_gap = max(0.0, float(final_score_threshold) - final_score)

        near_xgb = xgb_gap <= float(self.settings.strategy.near_miss_xgboost_gap_pct)
        near_score = score_gap <= float(self.settings.strategy.near_miss_final_score_gap_pct)
        if not near_xgb and not near_score:
            return ""

        if not bool(getattr(risk_result, "allowed", False)):
            risk_reason = str(getattr(risk_result, "reason", "") or "risk_block")
            return f"near_miss:risk_guard:{risk_reason}"

        if raw_action == "OPEN_LONG" and reviewed_action != "OPEN_LONG":
            return "near_miss:research_review_hold"

        if reviewed_action != "OPEN_LONG":
            return "near_miss:review_hold"

        if near_xgb and not near_score:
            return "near_miss:xgboost_threshold"
        if near_score:
            return "near_miss:final_score_threshold"
        return ""

    def evaluate_shadow_trades(
        self,
        now: datetime,
        limit: int = 500,
    ) -> dict:
        reporter = PerformanceReporter(self.storage, self.settings)
        timeframe = self.settings.strategy.primary_timeframe
        evaluated = 0
        for run in self.storage.get_open_shadow_trade_runs(limit=limit):
            timestamp = datetime.fromisoformat(run["timestamp"])
            horizon = timedelta(hours=int(run["horizon_hours"]))
            if timestamp + horizon > now:
                continue
            entry_price = float(run["entry_price"])
            metadata = json.loads(run["metadata_json"] or "{}")
            future_close = None
            pnl_pct = 0.0
            path_metrics: dict[str, float | int | str] = {}
            if "stop_loss_pct" in metadata or "take_profit_levels" in metadata:
                with self.storage._conn() as conn:
                    outcome = reporter._simulate_prediction_outcome(
                        conn,
                        symbol=run["symbol"],
                        timestamp=timestamp,
                        timeframe=timeframe,
                        decision={
                            "horizon_hours": int(run["horizon_hours"]),
                            "stop_loss_pct": metadata.get("stop_loss_pct"),
                            "take_profit_levels": metadata.get("take_profit_levels", []),
                        },
                        estimated_cost_pct=self.ESTIMATED_TRADE_COST_PCT,
                    )
                if outcome is None:
                    continue
                future_close = float(outcome["future_close"])
                pnl_pct = float(outcome["metadata"]["gross_return_pct"])
                path_metrics = dict(outcome["metadata"])
            else:
                with self.storage._conn() as conn:
                    future_close = reporter._fetch_close(
                        conn,
                        run["symbol"],
                        timeframe,
                        after_ms=int((timestamp + horizon).timestamp() * 1000),
                    )
                    if future_close is None:
                        continue
                    pnl_pct = (
                        (future_close / entry_price - 1.0) * 100
                        if entry_price > 0
                        else 0.0
                    )
                    path_metrics = self._trade_path_metrics(
                        conn=conn,
                        symbol=run["symbol"],
                        timeframe=timeframe,
                        start_at=timestamp,
                        end_at=timestamp + horizon,
                        entry_price=entry_price,
                    )
            metadata["evaluated_at"] = now.isoformat()
            metadata["estimated_cost_pct"] = self.ESTIMATED_TRADE_COST_PCT
            metadata["gross_return_pct"] = pnl_pct
            metadata["net_return_pct"] = pnl_pct - self.ESTIMATED_TRADE_COST_PCT
            metadata.update(path_metrics)
            self.storage.update_shadow_trade_run(
                int(run["id"]),
                {
                    "status": "evaluated",
                    "exit_price": future_close,
                    "pnl_pct": pnl_pct,
                    "metadata": metadata,
                    "evaluated_at": now.isoformat(),
                },
            )
            evaluated += 1
        if evaluated:
            self.storage.insert_execution_event(
                "shadow_trade_evaluation",
                "SYSTEM",
                {
                    "evaluated_count": evaluated,
                    "evaluated_at": now.isoformat(),
                },
            )
        reflected = self._backfill_shadow_reflections(limit=limit)
        return {"evaluated_count": evaluated, "reflected_count": reflected}

    def _backfill_shadow_reflections(self, limit: int = 500) -> int:
        with self.storage._conn() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """SELECT * FROM shadow_trade_runs
                       WHERE status = 'evaluated'
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (limit,),
                ).fetchall()
            ]
            if not rows:
                return 0
            trade_ids = [f"shadow:{int(row['id'])}" for row in rows]
            placeholders = ",".join("?" for _ in trade_ids)
            existing_rows = conn.execute(
                f"SELECT trade_id FROM reflections WHERE trade_id IN ({placeholders})",
                tuple(trade_ids),
            ).fetchall()
            existing_trade_ids = {str(row["trade_id"]) for row in existing_rows}

        reflected = 0
        for row in rows:
            trade_id = f"shadow:{int(row['id'])}"
            if trade_id in existing_trade_ids:
                continue
            self.storage._insert_reflection(self._shadow_reflection(row))
            reflected += 1
        return reflected

    def _shadow_reflection(self, row: dict) -> TradeReflection:
        setup_profile = json.loads(row.get("setup_profile_json") or "{}")
        metadata = json.loads(row.get("metadata_json") or "{}")
        normalized_profile = {
            **(
                setup_profile
                if isinstance(setup_profile, dict)
                else {}
            )
        }
        normalized_profile.setdefault("symbol", str(row.get("symbol") or ""))
        encoded_profile = ExperienceStore.encode_setup_profile(normalized_profile)
        block_reason = str(row.get("block_reason") or "shadow_block")
        reviewed_action = str(metadata.get("reviewed_action") or "")
        validation_reason = str(metadata.get("validation_reason") or "ok")
        rationale_parts = [
            f"shadow_block_reason={block_reason}",
            f"validation={validation_reason}",
        ]
        if reviewed_action:
            rationale_parts.append(f"reviewed_action={reviewed_action}")
        if encoded_profile:
            rationale_parts.append(encoded_profile)
        rationale = "; ".join(rationale_parts)

        pnl_pct = float(row.get("pnl_pct") or 0.0)
        confidence = self._shadow_confidence(metadata)
        correct_signals: list[str] = []
        wrong_signals: list[str] = []
        if pnl_pct > 0:
            wrong_signals = [
                "blocked_profitable_setup",
                f"block_reason={block_reason}",
            ]
            lesson = f"Blocked setup would have returned {pnl_pct:+.2f}% over shadow horizon."
        else:
            correct_signals = [
                "blocked_losing_setup",
                f"block_reason={block_reason}",
            ]
            lesson = f"Blocked setup avoided {abs(pnl_pct):.2f}% loss over shadow horizon."

        return TradeReflection(
            trade_id=f"shadow:{int(row['id'])}",
            symbol=str(row.get("symbol") or ""),
            direction=str(row.get("direction") or "LONG"),
            confidence=confidence,
            rationale=rationale,
            source="shadow_observation",
            experience_weight=ExperienceStore.SHADOW_REFLECTION_WEIGHT,
            realized_return_pct=pnl_pct,
            outcome_24h=pnl_pct,
            correct_signals=correct_signals,
            wrong_signals=wrong_signals,
            lesson=lesson,
            market_regime=self._shadow_market_regime(normalized_profile),
        )

    @staticmethod
    def _shadow_confidence(metadata: dict) -> float:
        probability = metadata.get("prediction_up_probability")
        if probability is not None:
            try:
                return max(0.0, min(1.0, float(probability)))
            except (TypeError, ValueError):
                pass
        review_score = metadata.get("review_score")
        if review_score is not None:
            try:
                return max(0.0, min(1.0, 0.5 + float(review_score) * 0.5))
            except (TypeError, ValueError):
                pass
        return 0.5

    @staticmethod
    def _shadow_market_regime(setup_profile: dict) -> MarketRegime:
        raw_regime = str(setup_profile.get("regime") or "UNKNOWN").upper()
        try:
            return MarketRegime(raw_regime)
        except ValueError:
            return MarketRegime.UNKNOWN

    @staticmethod
    def _storage_symbol_variants(symbol: str) -> list[str]:
        variants = [str(symbol)]
        if ":USDT" in symbol:
            variants.append(symbol.replace(":USDT", ""))
        else:
            variants.append(f"{symbol}:USDT")
        return list(dict.fromkeys(variants))

    def _trade_path_metrics(
        self,
        *,
        conn,
        symbol: str,
        timeframe: str,
        start_at: datetime,
        end_at: datetime,
        entry_price: float,
    ) -> dict[str, float]:
        if entry_price <= 0:
            return {
                "favorable_excursion_pct": 0.0,
                "adverse_excursion_pct": 0.0,
            }
        placeholders = ",".join("?" for _ in self._storage_symbol_variants(symbol))
        params = [
            *self._storage_symbol_variants(symbol),
            timeframe,
            int(start_at.timestamp() * 1000),
            int(end_at.timestamp() * 1000),
        ]
        row = conn.execute(
            f"""SELECT MAX(high) AS max_high, MIN(low) AS min_low
                  FROM ohlcv
                 WHERE symbol IN ({placeholders})
                   AND timeframe = ?
                   AND timestamp >= ?
                   AND timestamp <= ?""",
            params,
        ).fetchone()
        max_high = float(row["max_high"] or entry_price) if row is not None else entry_price
        min_low = float(row["min_low"] or entry_price) if row is not None else entry_price
        favorable_excursion_pct = max(0.0, (max_high / entry_price - 1.0) * 100)
        adverse_excursion_pct = max(0.0, (1.0 - min_low / entry_price) * 100)
        return {
            "favorable_excursion_pct": favorable_excursion_pct,
            "adverse_excursion_pct": adverse_excursion_pct,
        }
