"""Model evidence policy helpers for lifecycle runtime."""
from __future__ import annotations


class ModelEvidencePolicy:
    """Compute evidence quality and adaptive exit policy for promoted/candidate models."""

    def __init__(self, runtime):
        self.runtime = runtime

    def evidence_scale_from_metrics(
        self,
        payload: dict[str, float | int],
        *,
        source: str,
    ) -> dict[str, float | int | str]:
        payload = dict(payload or {})
        if "objective_score" not in payload:
            payload["objective_score"] = self.runtime.objective_score_from_metrics(
                payload
            )
        if "objective_quality" not in payload:
            payload["objective_quality"] = self.runtime.objective_score_quality(
                payload
            )
        sample_count = int(payload.get("sample_count", 0) or 0)
        executed_count = int(payload.get("executed_count", 0) or 0)
        sample_factor = min(sample_count, max(executed_count, 1), 8) / 8.0
        expectancy_pct = float(
            payload.get(
                "expectancy_pct",
                payload.get("avg_trade_return_pct", 0.0),
            )
            or 0.0
        )
        profit_factor = float(payload.get("profit_factor", 0.0) or 0.0)
        max_drawdown_pct = float(payload.get("max_drawdown_pct", 0.0) or 0.0)
        objective_quality = float(payload.get("objective_quality", 0.0) or 0.0)
        objective_score = float(payload.get("objective_score", 0.0) or 0.0)

        constraints = [("full", 1.0)]
        if sample_factor < 0.30:
            constraints.append(("thin_sample", 0.55))
        elif sample_factor < 0.50:
            constraints.append(("limited_sample", 0.70))
        elif sample_factor < 0.75:
            constraints.append(("still_maturing", 0.85))

        if objective_score < -0.05 or objective_quality < 0.0:
            constraints.append(("negative_objective", 0.35))
        elif objective_quality < 0.40:
            constraints.append(("weak_objective_quality", 0.55))
        elif objective_quality < 0.90:
            constraints.append(("subscale_objective_quality", 0.75))

        if expectancy_pct < 0.0:
            constraints.append(("negative_expectancy", 0.35))
        elif expectancy_pct < 0.10:
            constraints.append(("low_expectancy", 0.55))
        elif expectancy_pct < 0.25:
            constraints.append(("soft_expectancy", 0.75))

        if profit_factor < 0.90:
            constraints.append(("profit_factor_below_one", 0.40))
        elif profit_factor < 1.00:
            constraints.append(("subscale_profit_factor", 0.60))
        elif profit_factor < 1.15:
            constraints.append(("modest_profit_factor", 0.80))

        if max_drawdown_pct > 3.0:
            constraints.append(("elevated_drawdown", 0.45))
        elif max_drawdown_pct > 2.0:
            constraints.append(("high_drawdown", 0.65))
        elif max_drawdown_pct > 1.20:
            constraints.append(("moderate_drawdown", 0.85))

        driver, scale = min(constraints, key=lambda item: float(item[1]))
        return {
            "scale": float(scale),
            "source": source,
            "reason": (
                f"{source}:{driver}"
                f"|quality={objective_quality:.2f}"
                f"|expectancy={expectancy_pct:.2f}"
                f"|pf={profit_factor:.2f}"
                f"|drawdown={max_drawdown_pct:.2f}"
                f"|sample_factor={sample_factor:.2f}"
            ),
            "objective_quality": objective_quality,
            "objective_score": objective_score,
            "expectancy_pct": expectancy_pct,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown_pct,
            "sample_factor": sample_factor,
        }

    def active_model_evidence_scale(
        self, symbol: str
    ) -> dict[str, float | int | str]:
        candidates: list[dict[str, float | int | str]] = []
        active_registry = self.runtime.active_model_registry_entry(symbol)
        if isinstance(active_registry, dict) and active_registry:
            active_model_id = str(active_registry.get("model_id") or "").strip()
            active_started_at = str(
                active_registry.get("created_at")
                or active_registry.get("updated_at")
                or ""
            ).strip()
            if active_model_id and active_started_at:
                live_pnl_summary = self.runtime.build_model_live_pnl_summary(
                    symbol=symbol,
                    model_id=active_model_id,
                    started_at=active_started_at,
                )
                if int(live_pnl_summary.get("closed_trade_count", 0) or 0) > 0:
                    metrics = dict(live_pnl_summary)
                    metrics["objective_score"] = (
                        self.runtime.objective_score_from_metrics(metrics)
                    )
                    metrics["objective_quality"] = self.runtime.objective_score_quality(
                        metrics
                    )
                    candidates.append(
                        self.evidence_scale_from_metrics(
                            metrics,
                            source="active_runtime_realized",
                        )
                    )
                active_scorecard = self.runtime.build_model_scorecard(
                    symbol=symbol,
                    model_id=active_model_id,
                    evaluation_type="execution",
                    started_at=active_started_at,
                )
                if int(active_scorecard.get("sample_count", 0) or 0) > 0:
                    candidates.append(
                        self.evidence_scale_from_metrics(
                            {
                                "sample_count": int(
                                    active_scorecard.get("sample_count", 0) or 0
                                ),
                                "executed_count": int(
                                    active_scorecard.get("executed_count", 0) or 0
                                ),
                                "accuracy": float(
                                    active_scorecard.get("accuracy", 0.0) or 0.0
                                ),
                                "expectancy_pct": float(
                                    active_scorecard.get("expectancy_pct", 0.0) or 0.0
                                ),
                                "profit_factor": float(
                                    active_scorecard.get("profit_factor", 0.0) or 0.0
                                ),
                                "max_drawdown_pct": float(
                                    active_scorecard.get("max_drawdown_pct", 0.0) or 0.0
                                ),
                                "avg_trade_return_pct": float(
                                    active_scorecard.get("avg_trade_return_pct", 0.0)
                                    or 0.0
                                ),
                                "objective_score": float(
                                    active_scorecard.get("objective_score", 0.0) or 0.0
                                ),
                                "objective_quality": self.runtime.objective_score_quality(
                                    active_scorecard
                                ),
                            },
                            source="active_runtime_prediction",
                        )
                    )
        observation = self.runtime.get_model_promotion_observations().get(symbol, {})
        if isinstance(observation, dict) and observation:
            promoted_at = str(observation.get("promoted_at") or "")
            active_model_id = self.runtime.observation_active_model_id(observation)
            if promoted_at and active_model_id:
                scorecard = self.runtime.build_model_scorecard(
                    symbol=symbol,
                    model_id=active_model_id,
                    evaluation_type="execution",
                    started_at=promoted_at,
                )
                eval_count = int(scorecard.get("sample_count", 0) or 0)
                if eval_count > 0:
                    metrics = {
                        "sample_count": eval_count,
                        "executed_count": int(
                            scorecard.get("executed_count", 0) or 0
                        ),
                        "accuracy": float(scorecard.get("accuracy", 0.0) or 0.0),
                        "expectancy_pct": float(
                            scorecard.get("expectancy_pct", 0.0) or 0.0
                        ),
                        "profit_factor": float(
                            scorecard.get("profit_factor", 0.0) or 0.0
                        ),
                        "max_drawdown_pct": float(
                            scorecard.get("max_drawdown_pct", 0.0) or 0.0
                        ),
                        "avg_trade_return_pct": float(
                            scorecard.get("avg_trade_return_pct", 0.0) or 0.0
                        ),
                        "objective_score": float(
                            scorecard.get("objective_score", 0.0) or 0.0
                        ),
                        "objective_quality": self.runtime.objective_score_quality(
                            scorecard
                        ),
                    }
                    candidates.append(
                        self.evidence_scale_from_metrics(
                            metrics,
                            source="post_promotion_observation",
                        )
                    )

        model_metadata = self.runtime.read_model_metadata(
            self.runtime.runtime_model_path_for_symbol(symbol)
        )
        if model_metadata:
            accepted_eval_count = int(
                model_metadata.get("post_promotion_accept_eval_count", 0) or 0
            )
            canary_realized_trade_count = int(
                model_metadata.get("canary_live_realized_trade_count", 0) or 0
            )
            canary_eval_count = int(
                model_metadata.get("canary_live_eval_count", 0) or 0
            )
            sample_count = max(
                accepted_eval_count,
                canary_realized_trade_count,
                canary_eval_count,
            )
            if sample_count > 0:
                metrics = {
                    "sample_count": sample_count,
                    "executed_count": max(
                        accepted_eval_count, canary_realized_trade_count
                    ),
                    "accuracy": float(
                        model_metadata.get(
                            "post_promotion_accuracy",
                            model_metadata.get("canary_live_accuracy", 0.0),
                        )
                        or 0.0
                    ),
                    "expectancy_pct": float(
                        model_metadata.get(
                            "post_promotion_expectancy_pct",
                            model_metadata.get(
                                "canary_live_net_expectancy_pct",
                                model_metadata.get(
                                    "canary_live_expectancy_pct", 0.0
                                ),
                            ),
                        )
                        or 0.0
                    ),
                    "profit_factor": float(
                        model_metadata.get(
                            "post_promotion_profit_factor",
                            model_metadata.get(
                                "canary_live_net_profit_factor",
                                model_metadata.get(
                                    "canary_live_profit_factor", 0.0
                                ),
                            ),
                        )
                        or 0.0
                    ),
                    "max_drawdown_pct": float(
                        model_metadata.get(
                            "post_promotion_max_drawdown_pct",
                            model_metadata.get(
                                "canary_live_net_max_drawdown_pct",
                                model_metadata.get(
                                    "canary_live_max_drawdown_pct", 0.0
                                ),
                            ),
                        )
                        or 0.0
                    ),
                    "avg_trade_return_pct": float(
                        model_metadata.get(
                            "post_promotion_avg_trade_return_pct",
                            model_metadata.get(
                                "canary_live_net_avg_trade_return_pct",
                                0.0,
                            ),
                        )
                        or 0.0
                    ),
                    "objective_score": float(
                        model_metadata.get(
                            "post_promotion_objective_score",
                            model_metadata.get(
                                "canary_live_net_objective_score", 0.0
                            ),
                        )
                        or 0.0
                    ),
                    "objective_quality": float(
                        model_metadata.get(
                            "post_promotion_objective_quality",
                            model_metadata.get(
                                "canary_live_net_objective_quality", 0.0
                            ),
                        )
                        or 0.0
                    ),
                }
                candidates.append(
                    self.evidence_scale_from_metrics(
                        metrics,
                        source=(
                            "accepted_model_runtime"
                            if accepted_eval_count > 0
                            else "promoted_canary_runtime"
                        ),
                    )
                )

        if not candidates:
            return {
                "scale": 1.0,
                "source": "none",
                "reason": "no_model_evidence",
            }
        return min(
            candidates, key=lambda item: float(item.get("scale", 1.0) or 1.0)
        )

    def active_model_exit_policy(
        self, symbol: str
    ) -> dict[str, float | int | str | bool]:
        evidence = self.active_model_evidence_scale(symbol)
        scale = max(0.0, min(float(evidence.get("scale", 1.0) or 1.0), 1.0))
        base_max_hold_hours = max(
            4.0, float(self.runtime.settings.strategy.max_hold_hours or 0)
        )
        if (
            str(evidence.get("source", "none") or "none") == "none"
            or scale >= 0.999
        ):
            return {
                "adaptive_active": False,
                "scale": scale,
                "source": str(evidence.get("source", "none") or "none"),
                "reason": str(evidence.get("reason", "") or ""),
                "time_stop_hours": base_max_hold_hours,
                "de_risk_min_hours": 8.0,
                "de_risk_min_pnl_ratio": 0.01,
                "force_full_take_profit": False,
            }
        if scale <= 0.35:
            time_stop_hours = max(4.0, round(base_max_hold_hours * 0.25))
            de_risk_min_hours = 1.5
            de_risk_min_pnl_ratio = 0.003
            force_full_take_profit = True
        elif scale <= 0.55:
            time_stop_hours = max(4.0, round(base_max_hold_hours * 0.40))
            de_risk_min_hours = 2.0
            de_risk_min_pnl_ratio = 0.004
            force_full_take_profit = True
        elif scale <= 0.75:
            time_stop_hours = max(6.0, round(base_max_hold_hours * 0.60))
            de_risk_min_hours = 4.0
            de_risk_min_pnl_ratio = 0.006
            force_full_take_profit = False
        elif scale <= 0.85:
            time_stop_hours = max(8.0, round(base_max_hold_hours * 0.75))
            de_risk_min_hours = 6.0
            de_risk_min_pnl_ratio = 0.008
            force_full_take_profit = False
        else:
            time_stop_hours = base_max_hold_hours
            de_risk_min_hours = 8.0
            de_risk_min_pnl_ratio = 0.01
            force_full_take_profit = False
        return {
            "adaptive_active": True,
            "scale": scale,
            "source": str(evidence.get("source", "none") or "none"),
            "reason": str(evidence.get("reason", "") or ""),
            "time_stop_hours": min(base_max_hold_hours, float(time_stop_hours)),
            "de_risk_min_hours": float(de_risk_min_hours),
            "de_risk_min_pnl_ratio": float(de_risk_min_pnl_ratio),
            "force_full_take_profit": bool(force_full_take_profit),
        }

    def candidate_live_evidence_scale(
        self,
        symbol: str,
        candidate: dict,
    ) -> dict[str, float | int | str]:
        if str(candidate.get("status") or "") != "live":
            return {
                "scale": 1.0,
                "source": "none",
                "reason": "candidate_not_live",
            }
        started_at = str(
            candidate.get("live_started_at") or candidate.get("registered_at") or ""
        )
        challenger_model_id = self.runtime.candidate_challenger_model_id(candidate)
        if started_at and challenger_model_id:
            live_pnl_summary = self.runtime.build_model_live_pnl_summary(
                symbol=symbol,
                model_id=challenger_model_id,
                started_at=started_at,
            )
            if int(live_pnl_summary.get("closed_trade_count", 0) or 0) > 0:
                metrics = dict(live_pnl_summary)
                metrics["objective_score"] = (
                    self.runtime.objective_score_from_metrics(metrics)
                )
                metrics["objective_quality"] = self.runtime.objective_score_quality(
                    metrics
                )
                return self.evidence_scale_from_metrics(
                    metrics,
                    source="candidate_live_realized",
                )
            challenger_scorecard = self.runtime.build_model_scorecard(
                symbol=symbol,
                model_id=challenger_model_id,
                evaluation_type="challenger_live",
                started_at=started_at,
            )
            if int(challenger_scorecard.get("sample_count", 0) or 0) > 0:
                return self.evidence_scale_from_metrics(
                    {
                        "sample_count": int(
                            challenger_scorecard.get("sample_count", 0) or 0
                        ),
                        "executed_count": int(
                            challenger_scorecard.get("executed_count", 0) or 0
                        ),
                        "accuracy": float(
                            challenger_scorecard.get("accuracy", 0.0) or 0.0
                        ),
                        "expectancy_pct": float(
                            challenger_scorecard.get("expectancy_pct", 0.0) or 0.0
                        ),
                        "profit_factor": float(
                            challenger_scorecard.get("profit_factor", 0.0) or 0.0
                        ),
                        "max_drawdown_pct": float(
                            challenger_scorecard.get("max_drawdown_pct", 0.0) or 0.0
                        ),
                        "avg_trade_return_pct": float(
                            challenger_scorecard.get("avg_trade_return_pct", 0.0)
                            or 0.0
                        ),
                        "objective_score": float(
                            challenger_scorecard.get("objective_score", 0.0) or 0.0
                        ),
                        "objective_quality": self.runtime.objective_score_quality(
                            challenger_scorecard
                        ),
                    },
                    source="candidate_live_prediction",
                )
        shadow_eval_count = int(candidate.get("shadow_eval_count", 0) or 0)
        if shadow_eval_count > 0:
            return self.evidence_scale_from_metrics(
                {
                    "sample_count": shadow_eval_count,
                    "executed_count": int(
                        candidate.get("shadow_executed_count", 0) or 0
                    ),
                    "accuracy": float(candidate.get("shadow_accuracy", 0.0) or 0.0),
                    "expectancy_pct": float(
                        candidate.get("shadow_expectancy_pct", 0.0) or 0.0
                    ),
                    "profit_factor": float(
                        candidate.get("shadow_profit_factor", 0.0) or 0.0
                    ),
                    "max_drawdown_pct": float(
                        candidate.get("shadow_max_drawdown_pct", 0.0) or 0.0
                    ),
                    "avg_trade_return_pct": float(
                        candidate.get("shadow_avg_trade_return_pct", 0.0) or 0.0
                    ),
                    "objective_score": float(
                        candidate.get("shadow_objective_score", 0.0) or 0.0
                    ),
                    "objective_quality": float(
                        candidate.get("shadow_objective_quality", 0.0) or 0.0
                    ),
                },
                source="candidate_shadow",
            )
        return {
            "scale": 1.0,
            "source": "none",
            "reason": "candidate_no_evidence",
        }
