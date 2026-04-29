"""Strategy Evolver — 策略参数自动进化"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from loguru import logger

from config import Settings, get_settings
from core.storage import Storage
from learning.experience_store import ExperienceStore
from monitor.performance_report import PerformanceReporter


class StrategyEvolver:
    """基于历史经验微调策略参数"""

    RECENT_SETUP_PAUSE_WINDOW_HOURS = 36
    RECENT_SETUP_PAUSE_MIN_TRADES = 2
    RECENT_SETUP_PAUSE_MIN_NEGATIVE_RATIO = 0.75
    RECENT_SETUP_PAUSE_MAX_AVG_OUTCOME_PCT = -0.20
    REALIZED_SETUP_BLOCK_MIN_NEGATIVE_RATIO = 0.75
    REALIZED_SETUP_BLOCK_MAX_AVG_OUTCOME_PCT = -0.25

    def __init__(self, storage: Storage, settings: Settings | None = None):
        self.storage = storage
        self.settings = settings or get_settings()

    def suggest_adjustments(self) -> dict:
        """根据历史交易数据建议策略参数调整"""
        adjustments = {}

        with self.storage._conn() as conn:
            # 分析: 高置信度交易的胜率
            rows = conn.execute("""
                SELECT COALESCE(r.realized_return_pct, r.outcome_24h) AS realized_return_pct,
                       r.confidence, r.market_regime,
                       COUNT(*) as cnt
                FROM reflections r
                GROUP BY
                    CASE
                        WHEN r.confidence >= 0.8 THEN 'high'
                        WHEN r.confidence >= 0.65 THEN 'mid'
                        ELSE 'low'
                    END,
                    CASE WHEN COALESCE(r.realized_return_pct, r.outcome_24h) > 0 THEN 'win' ELSE 'loss' END
                ORDER BY r.confidence DESC
            """).fetchall()

            if rows:
                # 计算不同置信度区间的胜率
                high_total = sum(r["cnt"] for r in rows if r["confidence"] >= 0.8)
                high_wins = sum(r["cnt"] for r in rows
                                if r["confidence"] >= 0.8 and r["realized_return_pct"] > 0)

                if high_total >= 5:
                    high_win_rate = high_wins / high_total
                    if high_win_rate > 0.7:
                        adjustments["confidence_threshold"] = {
                            "current": 0.65,
                            "suggested": 0.70,
                            "reason": f"高置信度交易胜率 {high_win_rate:.0%}，可适当提高阈值减少交易频次",
                        }
                    elif high_win_rate < 0.4:
                        adjustments["confidence_threshold"] = {
                            "current": 0.65,
                            "suggested": 0.60,
                            "reason": f"高置信度交易胜率仅 {high_win_rate:.0%}，建议降低阈值",
                        }

        # 分析市场状态分布
        with self.storage._conn() as conn:
            regime_rows = conn.execute("""
                SELECT market_regime,
                       AVG(COALESCE(realized_return_pct, outcome_24h)) as avg_pnl,
                       COUNT(*) as cnt
                FROM reflections
                GROUP BY market_regime
            """).fetchall()

            if regime_rows:
                worst_regime = min(
                    [r for r in regime_rows if r["cnt"] >= 2],
                    key=lambda r: r["avg_pnl"],
                    default=None,
                )
                if worst_regime and worst_regime["avg_pnl"] < -2:
                    adjustments["regime_warning"] = {
                        "regime": worst_regime["market_regime"],
                        "avg_pnl": worst_regime["avg_pnl"],
                        "suggestion": f"在 {worst_regime['market_regime']} 市场状态下平均亏损 {worst_regime['avg_pnl']:.2f}%，建议该状态下提高阈值或暂停交易",
                    }

        if adjustments:
            logger.info(f"Strategy adjustments suggested: {adjustments}")

        return adjustments

    def suggest_runtime_overrides(
        self,
        base_runtime_settings: dict[str, float | list[float]],
        limit: int = 400,
        min_samples: int = 2,
    ) -> dict:
        with self.storage._conn() as conn:
            rows = conn.execute(
                """SELECT trade_id, source, experience_weight,
                          COALESCE(realized_return_pct, outcome_24h) AS realized_return_pct,
                          market_regime, rationale, created_at
                   FROM reflections
                   WHERE COALESCE(realized_return_pct, outcome_24h) IS NOT NULL
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        regime_stats: dict[str, list[tuple[float, float]]] = {}
        weak_liquidity_outcomes: list[tuple[float, float]] = []
        thin_news_outcomes: list[tuple[float, float]] = []
        setup_outcomes: dict[str, list[tuple[float, float]]] = {}
        setup_profiles: dict[str, dict[str, str]] = {}

        for row in rows:
            outcome = float(row["realized_return_pct"] or 0.0)
            weight = ExperienceStore.reflection_weight(
                row["trade_id"],
                row["source"],
                row["experience_weight"],
            )
            regime = str(row["market_regime"] or "UNKNOWN").upper()
            regime_stats.setdefault(regime, []).append((outcome, weight))

            profile = ExperienceStore.parse_setup_profile(row["rationale"])
            if profile:
                encoded = ExperienceStore.encode_setup_profile(profile)
                setup_profiles[encoded] = profile
                setup_outcomes.setdefault(encoded, []).append((outcome, weight))
                if profile.get("liquidity_bucket") == "weak":
                    weak_liquidity_outcomes.append((outcome, weight))
                if profile.get("news_bucket") == "thin":
                    thin_news_outcomes.append((outcome, weight))

        overrides: dict[str, float] = {}
        reasons: list[str] = []
        stats: dict[str, float | int] = {}
        blocked_setups: list[dict] = []

        def weighted_count(items: list[tuple[float, float]]) -> float:
            return sum(weight for _, weight in items)

        def maybe_avg(items: list[tuple[float, float]]) -> float:
            total_weight = weighted_count(items)
            return (
                sum(outcome * weight for outcome, weight in items) / total_weight
                if total_weight
                else 0.0
            )

        def negative_ratio(items: list[tuple[float, float]]) -> float:
            total_weight = weighted_count(items)
            return (
                sum(weight for outcome, weight in items if outcome < 0) / total_weight
                if total_weight
                else 0.0
            )

        base_xgb = float(base_runtime_settings["xgboost_probability_threshold"])
        base_final = float(base_runtime_settings["final_score_threshold"])
        base_liquidity = float(base_runtime_settings["min_liquidity_ratio"])
        base_sentiment = float(base_runtime_settings["sentiment_weight"])
        total_reflection_count = sum(weighted_count(items) for items in regime_stats.values())
        dominant_reflection_regime = ""
        dominant_reflection_share = 0.0
        if total_reflection_count:
            dominant_reflection_regime, dominant_reflection_items = max(
                regime_stats.items(),
                key=lambda item: weighted_count(item[1]),
            )
            dominant_reflection_share = weighted_count(dominant_reflection_items) / total_reflection_count
        stats["reflection_dominant_regime"] = dominant_reflection_regime or "UNKNOWN"
        stats["reflection_dominant_regime_share"] = round(dominant_reflection_share, 4)

        extreme_fear_outcomes = regime_stats.get("EXTREME_FEAR", [])
        has_reflection_extreme_fear = weighted_count(extreme_fear_outcomes) >= min_samples
        if weighted_count(extreme_fear_outcomes) >= min_samples:
            avg_outcome = maybe_avg(extreme_fear_outcomes)
            stats["extreme_fear_count"] = round(weighted_count(extreme_fear_outcomes), 4)
            stats["extreme_fear_avg_outcome_24h"] = round(avg_outcome, 4)
            if (
                avg_outcome < 0
                and dominant_reflection_regime == "EXTREME_FEAR"
                and dominant_reflection_share >= 0.8
            ):
                overrides["xgboost_probability_threshold"] = round(
                    min(0.95, base_xgb + 0.04),
                    4,
                )
                overrides["final_score_threshold"] = round(
                    min(0.99, base_final + 0.03),
                    4,
                )
                reasons.append(f"extreme_fear_negative_expectancy_{avg_outcome:.2f}")

        if weighted_count(weak_liquidity_outcomes) >= min_samples:
            avg_outcome = maybe_avg(weak_liquidity_outcomes)
            negative_ratio_value = negative_ratio(weak_liquidity_outcomes)
            stats["weak_liquidity_count"] = round(weighted_count(weak_liquidity_outcomes), 4)
            stats["weak_liquidity_avg_outcome_24h"] = round(avg_outcome, 4)
            stats["weak_liquidity_negative_ratio"] = round(negative_ratio_value, 4)
            if avg_outcome < 0 and negative_ratio_value >= 0.6:
                overrides["min_liquidity_ratio"] = round(
                    min(5.0, base_liquidity + 0.10),
                    4,
                )
                reasons.append(f"weak_liquidity_negative_expectancy_{avg_outcome:.2f}")
        has_reflection_weak_liquidity = weighted_count(weak_liquidity_outcomes) >= min_samples

        if weighted_count(thin_news_outcomes) >= min_samples:
            avg_outcome = maybe_avg(thin_news_outcomes)
            negative_ratio_value = negative_ratio(thin_news_outcomes)
            stats["thin_news_count"] = round(weighted_count(thin_news_outcomes), 4)
            stats["thin_news_avg_outcome_24h"] = round(avg_outcome, 4)
            stats["thin_news_negative_ratio"] = round(negative_ratio_value, 4)
            if avg_outcome < 0 and negative_ratio_value >= 0.6:
                overrides["final_score_threshold"] = round(
                    min(0.99, max(overrides.get("final_score_threshold", base_final), base_final + 0.02)),
                    4,
                )
                reasons.append(f"thin_news_negative_expectancy_{avg_outcome:.2f}")
        has_reflection_thin_news = weighted_count(thin_news_outcomes) >= min_samples

        for encoded, outcomes in setup_outcomes.items():
            if weighted_count(outcomes) < min_samples:
                continue
            avg_outcome = maybe_avg(outcomes)
            negative_ratio_value = negative_ratio(outcomes)
            profile = setup_profiles[encoded]
            if (
                avg_outcome <= self.REALIZED_SETUP_BLOCK_MAX_AVG_OUTCOME_PCT
                and negative_ratio_value >= self.REALIZED_SETUP_BLOCK_MIN_NEGATIVE_RATIO
            ):
                if profile.get("liquidity_bucket") == "weak" or profile.get("news_bucket") == "thin":
                    blocked_setups.append(
                        {
                            "criteria": {
                                key: value
                                for key, value in profile.items()
                                if key in {"regime", "liquidity_bucket", "news_bucket", "validation"}
                            },
                            "reason": f"realized_setup_negative_expectancy_{avg_outcome:.2f}",
                            "count": round(weighted_count(outcomes), 4),
                            "mode": "pause_open",
                        }
                    )

        bootstrap = self._prediction_setup_accuracy(
            limit=max(limit, 500),
            min_liquidity_ratio=base_liquidity,
        )
        bootstrap_stats = bootstrap["stats"]
        stats.update(bootstrap_stats)
        bootstrap_reasons = bootstrap["reasons"]
        dominant_prediction_regime = str(
            bootstrap_stats.get("prediction_dominant_regime", "UNKNOWN")
        )
        dominant_prediction_share = float(
            bootstrap_stats.get("prediction_dominant_regime_share", 0.0)
        )

        extreme_fear_prediction_count = int(
            bootstrap_stats.get("prediction_extreme_fear_count", 0)
        )
        extreme_fear_prediction_accuracy = float(
            bootstrap_stats.get("prediction_extreme_fear_accuracy", 1.0)
        )
        if (
            extreme_fear_prediction_count >= 20
            and extreme_fear_prediction_accuracy < 0.45
            and not has_reflection_extreme_fear
            and dominant_prediction_regime == "EXTREME_FEAR"
            and dominant_prediction_share >= 0.8
        ):
            overrides["xgboost_probability_threshold"] = round(
                min(0.95, max(overrides.get("xgboost_probability_threshold", base_xgb), base_xgb + 0.04)),
                4,
            )
            overrides["final_score_threshold"] = round(
                min(0.99, max(overrides.get("final_score_threshold", base_final), base_final + 0.03)),
                4,
            )
            overrides["sentiment_weight"] = round(
                max(-1.0, min(base_sentiment, 0.0)),
                4,
            )
            reasons.append(
                f"prediction_extreme_fear_accuracy_{extreme_fear_prediction_accuracy:.2f}"
            )

        weak_liquidity_prediction_count = int(
            bootstrap_stats.get("prediction_liquidity_weak_count", 0)
        )
        weak_liquidity_prediction_accuracy = float(
            bootstrap_stats.get("prediction_liquidity_weak_accuracy", 1.0)
        )
        if (
            weak_liquidity_prediction_count >= 8
            and weak_liquidity_prediction_accuracy < 0.45
            and not has_reflection_weak_liquidity
        ):
            overrides["min_liquidity_ratio"] = round(
                min(5.0, max(overrides.get("min_liquidity_ratio", base_liquidity), base_liquidity + 0.10)),
                4,
            )
            reasons.append(
                f"prediction_liquidity_weak_accuracy_{weak_liquidity_prediction_accuracy:.2f}"
            )
        thin_news_prediction_count = int(
            bootstrap_stats.get("prediction_news_thin_count", 0)
        )
        thin_news_prediction_accuracy = float(
            bootstrap_stats.get("prediction_news_thin_accuracy", 1.0)
        )
        if (
            thin_news_prediction_count >= 8
            and thin_news_prediction_accuracy < 0.45
            and not has_reflection_thin_news
        ):
            overrides["final_score_threshold"] = round(
                min(0.99, max(overrides.get("final_score_threshold", base_final), base_final + 0.02)),
                4,
            )
            reasons.append(
                f"prediction_news_thin_accuracy_{thin_news_prediction_accuracy:.2f}"
            )
        reasons.extend(reason for reason in bootstrap_reasons if reason not in reasons)

        recent_setup_pause_feedback = self._recent_realized_setup_pause_feedback()
        blocked_setups.extend(
            recent_setup_pause_feedback.get("blocked_setups", []) or []
        )
        stats.update(recent_setup_pause_feedback.get("stats", {}) or {})
        reasons.extend(
            reason
            for reason in recent_setup_pause_feedback.get("reasons", []) or []
            if reason not in reasons
        )

        shadow_feedback = self._shadow_runtime_feedback(
            base_runtime_settings=base_runtime_settings,
            limit=max(limit, 500),
            min_samples=max(3, min_samples),
        )
        for key, value in (shadow_feedback.get("runtime_overrides", {}) or {}).items():
            if key not in overrides:
                overrides[key] = value
                continue
            if key == "sentiment_weight":
                overrides[key] = min(float(overrides[key]), float(value))
            else:
                overrides[key] = max(float(overrides[key]), float(value))
        stats.update(shadow_feedback.get("stats", {}) or {})
        reasons.extend(
            reason
            for reason in shadow_feedback.get("reasons", []) or []
            if reason not in reasons
        )
        rehabilitated_signatures = {
            self._criteria_signature(entry.get("criteria", {}))
            for entry in shadow_feedback.get("rehabilitated_setups", []) or []
        }
        rehabilitated_signature_without_symbol = {
            self._criteria_signature_without_symbol(entry.get("criteria", {}))
            for entry in shadow_feedback.get("rehabilitated_setups", []) or []
        }
        if rehabilitated_signatures:
            blocked_setups = [
                entry
                for entry in blocked_setups
                if (
                    (
                        "symbol" in (entry.get("criteria", {}) or {})
                        and self._criteria_signature(entry.get("criteria", {}))
                        not in rehabilitated_signatures
                    )
                    or (
                        "symbol" not in (entry.get("criteria", {}) or {})
                        and self._criteria_signature_without_symbol(
                            entry.get("criteria", {})
                        )
                        not in rehabilitated_signature_without_symbol
                    )
                )
            ]
            stats["shadow_rehabilitated_setup_count"] = int(
                shadow_feedback.get("stats", {}).get(
                    "shadow_positive_blocked_setup_count",
                    len(rehabilitated_signatures),
                )
                or 0
            )

        blocked_setups = list(
            {
                json.dumps(entry, sort_keys=True): entry
                for entry in blocked_setups
            }.values()
        )

        if overrides:
            logger.info(
                f"Strategy evolver runtime overrides: overrides={overrides}, reasons={reasons}"
            )

        return {
            "runtime_overrides": overrides,
            "reasons": reasons,
            "stats": stats,
            "blocked_setups": blocked_setups,
        }

    @staticmethod
    def _setup_learning_criteria(profile: dict | None) -> dict[str, str]:
        criteria: dict[str, str] = {}
        if not isinstance(profile, dict):
            return criteria
        for key in ("regime", "liquidity_bucket", "news_bucket", "validation"):
            value = profile.get(key)
            if value is None or str(value).strip() == "":
                continue
            criteria[key] = str(value)
        return criteria

    @staticmethod
    def _criteria_signature(criteria: dict | None) -> str:
        if not isinstance(criteria, dict):
            return "{}"
        return json.dumps(
            {
                str(key): str(value)
                for key, value in sorted(criteria.items())
                if str(value).strip()
            },
            sort_keys=True,
        )

    @classmethod
    def _criteria_signature_without_symbol(cls, criteria: dict | None) -> str:
        if not isinstance(criteria, dict):
            return "{}"
        return cls._criteria_signature(
            {
                key: value
                for key, value in criteria.items()
                if str(key) != "symbol"
            }
        )

    def _shadow_runtime_feedback(
        self,
        base_runtime_settings: dict[str, float | list[float]],
        limit: int = 500,
        min_samples: int = 3,
    ) -> dict:
        with self.storage._conn() as conn:
            evaluation_rows = conn.execute(
                """SELECT symbol, is_correct, metadata_json
                   FROM prediction_evaluations
                   WHERE evaluation_type = 'shadow_observation'
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            trade_rows = conn.execute(
                """SELECT symbol, block_reason, pnl_pct, setup_profile_json
                   FROM shadow_trade_runs
                   WHERE status = 'evaluated'
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        base_xgb = float(base_runtime_settings["xgboost_probability_threshold"])
        base_final = float(base_runtime_settings["final_score_threshold"])
        base_liquidity = float(base_runtime_settings["min_liquidity_ratio"])
        base_sentiment = float(base_runtime_settings["sentiment_weight"])
        stats: dict[str, float | int] = {}
        reasons: list[str] = []
        overrides: dict[str, float] = {}
        rehabilitated_setups: list[dict] = []

        regime_results: dict[str, list[int]] = {}
        weak_liquidity_results: list[int] = []
        thin_news_results: list[int] = []

        for row in evaluation_rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            correct = int(row["is_correct"] or 0)
            regime = str(metadata.get("regime") or "UNKNOWN").upper()
            regime_results.setdefault(regime, []).append(correct)
            setup_profile = metadata.get("setup_profile") or {}
            if str(setup_profile.get("liquidity_bucket") or "") == "weak":
                weak_liquidity_results.append(correct)
            if str(setup_profile.get("news_bucket") or "") == "thin":
                thin_news_results.append(correct)

        def accuracy(values: list[int]) -> float:
            return sum(values) / len(values) if values else 0.0

        shadow_eval_count = sum(len(items) for items in regime_results.values())
        stats["shadow_prediction_eval_count"] = shadow_eval_count
        if shadow_eval_count:
            stats["shadow_prediction_accuracy"] = round(
                accuracy(
                    [item for items in regime_results.values() for item in items]
                ),
                4,
            )
        extreme_fear_results = regime_results.get("EXTREME_FEAR", [])
        if len(extreme_fear_results) >= min_samples:
            extreme_fear_accuracy = accuracy(extreme_fear_results)
            stats["shadow_extreme_fear_count"] = len(extreme_fear_results)
            stats["shadow_extreme_fear_accuracy"] = round(extreme_fear_accuracy, 4)
            if extreme_fear_accuracy < 0.45:
                overrides["xgboost_probability_threshold"] = round(
                    min(0.95, base_xgb + 0.03),
                    4,
                )
                overrides["final_score_threshold"] = round(
                    min(0.99, base_final + 0.02),
                    4,
                )
                overrides["sentiment_weight"] = round(
                    max(-1.0, min(base_sentiment, 0.0)),
                    4,
                )
                reasons.append(
                    f"shadow_extreme_fear_accuracy_{extreme_fear_accuracy:.2f}"
                )
        if len(weak_liquidity_results) >= min_samples:
            weak_liquidity_accuracy = accuracy(weak_liquidity_results)
            stats["shadow_liquidity_weak_count"] = len(weak_liquidity_results)
            stats["shadow_liquidity_weak_accuracy"] = round(
                weak_liquidity_accuracy,
                4,
            )
            if weak_liquidity_accuracy < 0.45:
                overrides["min_liquidity_ratio"] = round(
                    min(5.0, base_liquidity + 0.10),
                    4,
                )
                reasons.append(
                    f"shadow_liquidity_weak_accuracy_{weak_liquidity_accuracy:.2f}"
                )
        if len(thin_news_results) >= min_samples:
            thin_news_accuracy = accuracy(thin_news_results)
            stats["shadow_news_thin_count"] = len(thin_news_results)
            stats["shadow_news_thin_accuracy"] = round(thin_news_accuracy, 4)
            if thin_news_accuracy < 0.45:
                overrides["final_score_threshold"] = round(
                    min(0.99, max(overrides.get("final_score_threshold", base_final), base_final + 0.02)),
                    4,
                )
                reasons.append(
                    f"shadow_news_thin_accuracy_{thin_news_accuracy:.2f}"
                )

        setup_outcomes: dict[str, dict[str, object]] = {}
        positive_blocked_count = 0
        for row in trade_rows:
            criteria = self._setup_learning_criteria(
                json.loads(row["setup_profile_json"] or "{}")
            )
            if not criteria:
                continue
            signature = self._criteria_signature(criteria)
            bucket = setup_outcomes.setdefault(
                signature,
                {"criteria": criteria, "pnls": [], "symbols": set()},
            )
            pnl_pct = float(row["pnl_pct"] or 0.0)
            cast_pnls = bucket["pnls"]
            if isinstance(cast_pnls, list):
                cast_pnls.append(pnl_pct)
            cast_symbols = bucket["symbols"]
            if isinstance(cast_symbols, set):
                cast_symbols.add(str(row["symbol"]))

        for bucket in setup_outcomes.values():
            pnls = bucket["pnls"]
            if not isinstance(pnls, list) or len(pnls) < min_samples:
                continue
            avg_pnl = sum(pnls) / len(pnls)
            positive_ratio = sum(1 for value in pnls if value > 0) / len(pnls)
            if avg_pnl <= 0 or positive_ratio < 0.6:
                continue
            positive_blocked_count += 1
            rehabilitated_setups.append(
                {
                    "criteria": bucket["criteria"],
                    "reason": f"shadow_blocked_positive_expectancy_{avg_pnl:.2f}",
                    "count": len(pnls),
                    "mode": "resume_open",
                }
            )
            symbols = bucket.get("symbols")
            if isinstance(symbols, set):
                for symbol in sorted(str(item) for item in symbols if str(item).strip()):
                    rehabilitated_setups.append(
                        {
                            "criteria": {
                                **bucket["criteria"],
                                "symbol": symbol,
                            },
                            "reason": f"shadow_blocked_positive_expectancy_{avg_pnl:.2f}",
                            "count": len(pnls),
                            "mode": "resume_open",
                        }
                    )
        stats["shadow_positive_blocked_setup_count"] = positive_blocked_count
        if positive_blocked_count:
            reasons.append(f"shadow_positive_blocked_setups_{positive_blocked_count}")

        return {
            "runtime_overrides": overrides,
            "reasons": reasons,
            "stats": stats,
            "rehabilitated_setups": rehabilitated_setups,
        }

    def _recent_realized_setup_pause_feedback(self) -> dict:
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(hours=self.RECENT_SETUP_PAUSE_WINDOW_HOURS)
        ).isoformat()
        with self.storage._conn() as conn:
            ledger_rows = conn.execute(
                """SELECT l.id AS ledger_id, l.trade_id, l.symbol, l.net_return_pct,
                          l.event_time, t.rationale
                   FROM pnl_ledger l
                   JOIN trades t ON t.id = l.trade_id
                   WHERE l.event_type='close'
                     AND l.event_time >= ?
                   ORDER BY l.event_time DESC, l.id DESC""",
                (cutoff,),
            ).fetchall()
            reflection_rows = conn.execute(
                """SELECT symbol, source, rationale, created_at,
                          trade_id,
                          COALESCE(realized_return_pct, outcome_24h) AS realized_return_pct
                   FROM reflections
                   WHERE COALESCE(realized_return_pct, outcome_24h) IS NOT NULL
                     AND created_at >= ?
                     AND source != 'shadow_observation'
                   ORDER BY created_at DESC""",
                (cutoff,),
            ).fetchall()

        buckets: dict[str, dict[str, object]] = {}
        ledger_trade_ids = {
            str(row["trade_id"] or "")
            for row in ledger_rows
            if str(row["trade_id"] or "").strip()
        }

        for row in ledger_rows:
            profile = ExperienceStore.parse_setup_profile(row["rationale"])
            if not profile:
                continue
            criteria = {
                key: str(value)
                for key, value in profile.items()
                if key in {"symbol", "regime", "validation", "liquidity_bucket", "news_bucket"}
                and str(value).strip()
            }
            if not criteria:
                continue
            signature = self._criteria_signature(criteria)
            bucket = buckets.setdefault(
                signature,
                {
                    "criteria": criteria,
                    "outcomes": [],
                    "latest_created_at": "",
                },
            )
            outcomes = bucket["outcomes"]
            if isinstance(outcomes, list):
                outcomes.append(float(row["net_return_pct"] or 0.0))
            latest_created_at = str(bucket.get("latest_created_at") or "")
            row_created_at = str(row["event_time"] or "")
            if row_created_at > latest_created_at:
                bucket["latest_created_at"] = row_created_at

        for row in reflection_rows:
            if str(row["trade_id"] or "") in ledger_trade_ids:
                continue
            profile = ExperienceStore.parse_setup_profile(row["rationale"])
            if not profile:
                continue
            criteria = {
                key: str(value)
                for key, value in profile.items()
                if key in {"symbol", "regime", "validation", "liquidity_bucket", "news_bucket"}
                and str(value).strip()
            }
            if not criteria:
                continue
            signature = self._criteria_signature(criteria)
            bucket = buckets.setdefault(
                signature,
                {
                    "criteria": criteria,
                    "outcomes": [],
                    "latest_created_at": "",
                },
            )
            outcomes = bucket["outcomes"]
            if isinstance(outcomes, list):
                outcomes.append(float(row["realized_return_pct"] or 0.0))
            latest_created_at = str(bucket.get("latest_created_at") or "")
            row_created_at = str(row["created_at"] or "")
            if row_created_at > latest_created_at:
                bucket["latest_created_at"] = row_created_at

        blocked_setups: list[dict] = []
        stats: dict[str, float | int] = {}
        reasons: list[str] = []
        pause_count = 0

        for bucket in buckets.values():
            outcomes = bucket.get("outcomes")
            if not isinstance(outcomes, list) or len(outcomes) < self.RECENT_SETUP_PAUSE_MIN_TRADES:
                continue
            avg_outcome = sum(outcomes) / len(outcomes)
            negative_ratio = sum(1 for value in outcomes if value < 0) / len(outcomes)
            if avg_outcome > self.RECENT_SETUP_PAUSE_MAX_AVG_OUTCOME_PCT:
                continue
            if negative_ratio < self.RECENT_SETUP_PAUSE_MIN_NEGATIVE_RATIO:
                continue
            pause_count += 1
            blocked_setups.append(
                {
                    "criteria": dict(bucket["criteria"]),
                    "reason": f"recent_realized_setup_negative_expectancy_{avg_outcome:.2f}",
                    "count": len(outcomes),
                    "negative_ratio": round(negative_ratio, 4),
                    "window_hours": self.RECENT_SETUP_PAUSE_WINDOW_HOURS,
                    "latest_created_at": str(bucket.get("latest_created_at") or ""),
                    "mode": "pause_open",
                }
            )

        stats["recent_negative_setup_pause_count"] = pause_count
        if pause_count:
            reasons.append(f"recent_negative_setup_pauses_{pause_count}")
        return {
            "blocked_setups": blocked_setups,
            "stats": stats,
            "reasons": reasons,
        }

    def _prediction_setup_accuracy(
        self,
        limit: int = 500,
        min_liquidity_ratio: float | None = None,
    ) -> dict:
        reporter = PerformanceReporter(self.storage, self.settings)
        timeframe = self.settings.strategy.primary_timeframe
        liquidity_floor = float(
            min_liquidity_ratio
            if min_liquidity_ratio is not None
            else self.settings.strategy.min_liquidity_ratio
        )
        with self.storage._conn() as conn:
            rows = conn.execute(
                "SELECT id, symbol, timestamp, up_probability, research_json, decision_json "
                "FROM prediction_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            rows = reporter._dedupe_prediction_rows(rows)

            regime_buckets: dict[str, list[int]] = {}
            liquidity_weak_results: list[int] = []
            news_thin_results: list[int] = []

            for row in rows:
                timestamp = datetime.fromisoformat(row["timestamp"])
                decision = json.loads(row["decision_json"])
                outcome = reporter._simulate_prediction_outcome(
                    conn,
                    symbol=row["symbol"],
                    timestamp=timestamp,
                    timeframe=timeframe,
                    decision=decision,
                )
                if outcome is None:
                    continue
                actual_up = bool(outcome["actual_up"])
                research = json.loads(row["research_json"])
                threshold = float(
                    decision.get(
                        "xgboost_threshold",
                        self.settings.model.xgboost_probability_threshold,
                    )
                )
                predicted_up = float(row["up_probability"]) >= threshold
                correct = int(predicted_up == actual_up)
                regime = str(
                    decision.get("regime")
                    or research.get("market_regime")
                    or "UNKNOWN"
                ).upper()
                regime_buckets.setdefault(regime, []).append(correct)
                setup_profile = decision.get("setup_profile") or {}
                key_reasons = {
                    str(item).strip()
                    for item in research.get("key_reason", [])
                    if str(item).strip()
                }
                liquidity_bucket = str(setup_profile.get("liquidity_bucket") or "")
                if liquidity_bucket == "weak":
                    liquidity_weak_results.append(correct)
                else:
                    feature_row = conn.execute(
                        """SELECT features_json FROM feature_snapshots
                           WHERE symbol = ? AND timestamp = ?
                           ORDER BY created_at DESC LIMIT 1""",
                        (row["symbol"], row["timestamp"]),
                    ).fetchone()
                    features = (
                        json.loads(feature_row["features_json"])
                        if feature_row and feature_row["features_json"]
                        else {}
                    )
                    liquidity_ratio = float(features.get("volume_ratio_1h", 0.0))
                    if liquidity_ratio and liquidity_ratio < liquidity_floor:
                        liquidity_weak_results.append(correct)
                    elif "liquidity_weak" in key_reasons:
                        liquidity_weak_results.append(correct)
                news_bucket = str(setup_profile.get("news_bucket") or "")
                if news_bucket == "thin" or "news_coverage_thin" in key_reasons:
                    news_thin_results.append(correct)

        def accuracy(results: list[int]) -> float:
            return sum(results) / len(results) if results else 0.0

        stats: dict[str, float | int] = {}
        reasons: list[str] = []
        total_predictions = sum(len(items) for items in regime_buckets.values())
        if total_predictions:
            dominant_regime, dominant_items = max(
                regime_buckets.items(),
                key=lambda item: len(item[1]),
            )
            stats["prediction_dominant_regime"] = dominant_regime
            stats["prediction_dominant_regime_share"] = round(
                len(dominant_items) / total_predictions,
                4,
            )
        if regime_buckets.get("EXTREME_FEAR"):
            stats["prediction_extreme_fear_count"] = len(regime_buckets["EXTREME_FEAR"])
            stats["prediction_extreme_fear_accuracy"] = round(
                accuracy(regime_buckets["EXTREME_FEAR"]),
                4,
            )
        if liquidity_weak_results:
            stats["prediction_liquidity_weak_count"] = len(liquidity_weak_results)
            stats["prediction_liquidity_weak_accuracy"] = round(
                accuracy(liquidity_weak_results),
                4,
            )
        if news_thin_results:
            stats["prediction_news_thin_count"] = len(news_thin_results)
            stats["prediction_news_thin_accuracy"] = round(
                accuracy(news_thin_results),
                4,
            )
        return {
            "stats": stats,
            "reasons": reasons,
        }
