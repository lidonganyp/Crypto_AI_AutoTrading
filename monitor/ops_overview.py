"""Operational overview aggregation for CryptoAI v3."""
from __future__ import annotations

import json

from core.i18n import get_default_language, normalize_language, text_for
from core.runtime_state_view import RuntimeStateView
from core.storage import Storage
from monitor.nextgen_live_summary import build_nextgen_live_summary


class OpsOverviewService:
    """Aggregate the most recent operational state into one compact report."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self.runtime_state = RuntimeStateView(storage)

    def build(self) -> dict:
        runtime_state = self.runtime_state.snapshot()
        market_data_route = self.storage.get_json_state("market_data_last_route", {}) or {}
        market_data_stats = self.storage.get_json_state("market_data_failover_stats", {}) or {}
        with self.storage._conn() as conn:
            latest_account = conn.execute(
                "SELECT * FROM account_snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            latest_cycle = conn.execute(
                "SELECT * FROM cycle_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            latest_scheduler = conn.execute(
                "SELECT * FROM scheduler_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            latest_reconciliation = conn.execute(
                "SELECT * FROM reconciliation_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            latest_health = conn.execute(
                "SELECT * FROM report_artifacts WHERE report_type='health' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            latest_incident = conn.execute(
                "SELECT * FROM report_artifacts WHERE report_type='incident' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            latest_performance = conn.execute(
                "SELECT * FROM report_artifacts WHERE report_type='performance' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            runtime_override_status = conn.execute(
                "SELECT value, updated_at FROM system_state WHERE key='runtime_settings_override_status'"
            ).fetchone()
            latest_canary = conn.execute(
                "SELECT symbol, created_at FROM execution_events "
                "WHERE event_type='paper_canary_open' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            canary_counts = conn.execute(
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
            ).fetchone()
            fast_alpha_policy_counts = conn.execute(
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
            ).fetchone()
            prediction_eval_counts = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN evaluation_type='execution' THEN 1 ELSE 0 END) AS execution_count,
                    SUM(CASE WHEN evaluation_type='shadow_observation' THEN 1 ELSE 0 END) AS shadow_count,
                    MAX(created_at) AS latest_created_at
                FROM prediction_evaluations
                """
            ).fetchone()
            latest_model_lifecycle = conn.execute(
                "SELECT event_type, symbol, payload_json, created_at FROM execution_events "
                "WHERE event_type IN ("
                "'paper_canary_open',"
                "'model_promotion_candidate_started',"
                "'model_promotion_live_started',"
                "'model_promotion_candidate_rejected',"
                "'model_promotion_promoted',"
                "'model_observation_started',"
                "'model_observation_accepted',"
                "'model_rollback'"
                ") ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

        shadow_candidate_count = 0
        live_candidate_count = 0
        if isinstance(runtime_state.model_promotion_candidates, dict):
            for payload in runtime_state.model_promotion_candidates.values():
                payload = payload if isinstance(payload, dict) else {}
                status = str(payload.get("status") or "")
                if status == "shadow":
                    shadow_candidate_count += 1
                elif status == "live":
                    live_candidate_count += 1
        failover_summary = self._market_data_failover_summary(
            market_data_route,
            market_data_stats,
        )
        nextgen_live = build_nextgen_live_summary(self.storage)

        return {
            "latest_account": dict(latest_account) if latest_account else None,
            "latest_cycle": dict(latest_cycle) if latest_cycle else None,
            "latest_scheduler": dict(latest_scheduler) if latest_scheduler else None,
            "latest_reconciliation": (
                dict(latest_reconciliation) if latest_reconciliation else None
            ),
            "latest_health_report": dict(latest_health) if latest_health else None,
            "latest_incident_report": dict(latest_incident) if latest_incident else None,
            "latest_performance_report": (
                dict(latest_performance) if latest_performance else None
            ),
            "runtime_override_status": (
                dict(runtime_override_status) if runtime_override_status else None
            ),
            "runtime_effective": runtime_state.runtime_settings_effective,
            "runtime_conflicts": runtime_state.runtime_settings_override_conflicts,
            "shadow_observation_symbols": runtime_state.shadow_observation_symbols,
            "paper_exploration_active_symbols": (
                runtime_state.paper_exploration_active_symbols
            ),
            "shadow_candidate_count": shadow_candidate_count,
            "live_candidate_count": live_candidate_count,
            "promotion_observation_count": (
                len(runtime_state.model_promotion_observations)
                if isinstance(runtime_state.model_promotion_observations, dict)
                else 0
            ),
            "latest_canary": dict(latest_canary) if latest_canary else None,
            "paper_canary_open_count": int((canary_counts["total_count"] or 0) if canary_counts else 0),
            "soft_paper_canary_open_count": int((canary_counts["soft_count"] or 0) if canary_counts else 0),
            "fast_alpha_short_horizon_softened_open_count": int((fast_alpha_policy_counts["softened_open_count"] or 0) if fast_alpha_policy_counts else 0),
            "fast_alpha_negative_expectancy_pause_count": int((fast_alpha_policy_counts["negative_pause_count"] or 0) if fast_alpha_policy_counts else 0),
            "execution_evaluation_count": int((prediction_eval_counts["execution_count"] or 0) if prediction_eval_counts else 0),
            "shadow_evaluation_count": int((prediction_eval_counts["shadow_count"] or 0) if prediction_eval_counts else 0),
            "latest_prediction_evaluation_at": (
                str(prediction_eval_counts["latest_created_at"] or "")
                if prediction_eval_counts
                else ""
            ),
            "latest_model_lifecycle_event": (
                dict(latest_model_lifecycle) if latest_model_lifecycle else None
            ),
            "market_data_route": market_data_route,
            "market_data_failover_stats": market_data_stats,
            "market_data_failover_summary": failover_summary,
            "nextgen_live": nextgen_live,
        }

    @staticmethod
    def render(data: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        account = data.get("latest_account") or {}
        cycle = data.get("latest_cycle") or {}
        scheduler = data.get("latest_scheduler") or {}
        reconciliation = data.get("latest_reconciliation") or {}
        runtime_status = data.get("runtime_override_status") or {}
        runtime_effective = data.get("runtime_effective") or {}
        runtime_conflicts = data.get("runtime_conflicts") or {}
        latest_canary = data.get("latest_canary") or {}
        latest_lifecycle = data.get("latest_model_lifecycle_event") or {}
        failover = data.get("market_data_failover_summary") or {}
        nextgen_live = data.get("nextgen_live") or {}
        lines = [
            text_for(lang, "# 运维总览", "# Ops Overview"),
            text_for(lang, f"- 最近权益: {account.get('equity', 'N/A')}", f"- Latest Equity: {account.get('equity', 'N/A')}"),
            text_for(lang, f"- 最近回撤: {account.get('drawdown_pct', 'N/A')}", f"- Latest Drawdown: {account.get('drawdown_pct', 'N/A')}"),
            text_for(lang, f"- 最近周期状态: {cycle.get('status', 'N/A')}", f"- Latest Cycle Status: {cycle.get('status', 'N/A')}"),
            text_for(lang, f"- 最近周期开/平仓: {cycle.get('opened_positions', 'N/A')}/{cycle.get('closed_positions', 'N/A')}", f"- Latest Cycle Opened/Closed: {cycle.get('opened_positions', 'N/A')}/{cycle.get('closed_positions', 'N/A')}"),
            text_for(lang, f"- 最近调度任务: {scheduler.get('job_name', 'N/A')}", f"- Latest Scheduler Job: {scheduler.get('job_name', 'N/A')}"),
            text_for(lang, f"- 最近调度状态: {scheduler.get('status', 'N/A')}", f"- Latest Scheduler Status: {scheduler.get('status', 'N/A')}"),
            text_for(lang, f"- 最近对账状态: {reconciliation.get('status', 'N/A')}", f"- Latest Reconciliation: {reconciliation.get('status', 'N/A')}"),
            text_for(lang, f"- Runtime 覆盖状态: {runtime_status.get('value', 'N/A')}", f"- Runtime Override Status: {runtime_status.get('value', 'N/A')}"),
            text_for(lang, f"- 手动覆盖字段数: {len(runtime_effective.get('manual_overrides', {}) or {})}", f"- Manual Override Fields: {len(runtime_effective.get('manual_overrides', {}) or {})}"),
            text_for(lang, f"- 学习覆盖字段数: {len(runtime_effective.get('learning_overrides', {}) or {})}", f"- Learning Override Fields: {len(runtime_effective.get('learning_overrides', {}) or {})}"),
            text_for(lang, f"- 被拦截的学习字段: {len(runtime_conflicts.get('conflict_keys', []) or [])}", f"- Blocked Learning Fields: {len(runtime_conflicts.get('conflict_keys', []) or [])}"),
            text_for(lang, f"- Shadow 观察池数量: {len(data.get('shadow_observation_symbols', []) or [])}", f"- Shadow Observation Pool: {len(data.get('shadow_observation_symbols', []) or [])}"),
            text_for(lang, f"- Paper 探索槽位: {', '.join(data.get('paper_exploration_active_symbols', []) or []) or '无'}", f"- Paper Exploration Slot: {', '.join(data.get('paper_exploration_active_symbols', []) or []) or 'none'}"),
            text_for(lang, f"- Shadow 候选模型数: {data.get('shadow_candidate_count', 0)}", f"- Shadow Candidates: {data.get('shadow_candidate_count', 0)}"),
            text_for(lang, f"- Live Canary 模型数: {data.get('live_candidate_count', 0)}", f"- Live Canary Candidates: {data.get('live_candidate_count', 0)}"),
            text_for(lang, f"- Promotion 观察中模型数: {data.get('promotion_observation_count', 0)}", f"- Promoted Models Under Observation: {data.get('promotion_observation_count', 0)}"),
            text_for(lang, f"- 最近 Paper Canary: {latest_canary.get('symbol', 'N/A')}", f"- Latest Paper Canary: {latest_canary.get('symbol', 'N/A')}"),
            text_for(lang, f"- Paper Canary 累计开仓: {data.get('paper_canary_open_count', 0)}", f"- Paper Canary Opens: {data.get('paper_canary_open_count', 0)}"),
            text_for(lang, f"- 软审批 Canary 累计开仓: {data.get('soft_paper_canary_open_count', 0)}", f"- Soft-Review Canary Opens: {data.get('soft_paper_canary_open_count', 0)}"),
            text_for(lang, f"- Short-horizon 放行累计开仓: {data.get('fast_alpha_short_horizon_softened_open_count', 0)}", f"- Short-horizon Softened Opens: {data.get('fast_alpha_short_horizon_softened_open_count', 0)}"),
            text_for(lang, f"- Short-horizon 负期望暂停累计: {data.get('fast_alpha_negative_expectancy_pause_count', 0)}", f"- Short-horizon Negative-Edge Pauses: {data.get('fast_alpha_negative_expectancy_pause_count', 0)}"),
            text_for(lang, f"- 执行评估累计: {data.get('execution_evaluation_count', 0)}", f"- Execution Evaluations: {data.get('execution_evaluation_count', 0)}"),
            text_for(lang, f"- Shadow 评估累计: {data.get('shadow_evaluation_count', 0)}", f"- Shadow Evaluations: {data.get('shadow_evaluation_count', 0)}"),
            text_for(lang, f"- 最近市场数据提供方: {failover.get('latest_provider', 'unknown')}", f"- Latest Market Data Provider: {failover.get('latest_provider', 'unknown')}"),
            text_for(lang, f"- 最近市场数据操作: {failover.get('latest_operation', 'unknown')}", f"- Latest Market Data Operation: {failover.get('latest_operation', 'unknown')}"),
            text_for(lang, f"- 市场数据 Failover 次数: {failover.get('fallback_count', 0)}", f"- Market Data Failover Count: {failover.get('fallback_count', 0)}"),
            text_for(lang, f"- 主市场数据失败次数: {failover.get('primary_failures', 0)}", f"- Primary Market Data Failures: {failover.get('primary_failures', 0)}"),
            text_for(lang, f"- Nextgen Live 请求启用: {nextgen_live.get('requested_live', False)}", f"- Nextgen Live Requested: {nextgen_live.get('requested_live', False)}"),
            text_for(lang, f"- Nextgen Live 实际生效: {nextgen_live.get('effective_live', False)}", f"- Nextgen Live Effective: {nextgen_live.get('effective_live', False)}"),
            text_for(lang, f"- Nextgen Live 强制平仓: {nextgen_live.get('force_flatten', False)}", f"- Nextgen Live Force Flatten: {nextgen_live.get('force_flatten', False)}"),
            text_for(lang, f"- Nextgen Live Kill Switch: {nextgen_live.get('kill_switch_active', False)}", f"- Nextgen Live Kill Switch: {nextgen_live.get('kill_switch_active', False)}"),
            text_for(lang, f"- Nextgen Live 白名单数: {len(nextgen_live.get('whitelist', []) or [])}", f"- Nextgen Live Whitelist Count: {len(nextgen_live.get('whitelist', []) or [])}"),
            text_for(lang, f"- Nextgen Live 活跃上限: {nextgen_live.get('max_active_runtimes', 'N/A')}", f"- Nextgen Live Max Active Runtimes: {nextgen_live.get('max_active_runtimes', 'N/A')}"),
            text_for(lang, f"- Nextgen Live 最近运行: {nextgen_live.get('last_run_status') or 'none'}", f"- Nextgen Live Latest Run: {nextgen_live.get('last_run_status') or 'none'}"),
            text_for(lang, f"- Nextgen Live 最近触发源: {nextgen_live.get('last_run_trigger') or 'none'}", f"- Nextgen Live Latest Trigger: {nextgen_live.get('last_run_trigger') or 'none'}"),
            text_for(lang, f"- Nextgen Live 请求 Repair 数: {nextgen_live.get('repair_queue_requested_size', 0)}", f"- Nextgen Live Requested Repair Count: {nextgen_live.get('repair_queue_requested_size', 0)}"),
            text_for(lang, f"- Nextgen Live 被丢弃 Repair 数: {nextgen_live.get('repair_queue_dropped_count', 0)}", f"- Nextgen Live Dropped Repair Count: {nextgen_live.get('repair_queue_dropped_count', 0)}"),
            text_for(lang, f"- Nextgen Live Hold Repair 数: {nextgen_live.get('repair_queue_hold_priority_count', 0)}", f"- Nextgen Live Hold Repair Count: {nextgen_live.get('repair_queue_hold_priority_count', 0)}"),
            text_for(lang, f"- Nextgen Live 延后 Rebuild 数: {nextgen_live.get('repair_queue_postponed_rebuild_count', 0)}", f"- Nextgen Live Postponed Rebuild Count: {nextgen_live.get('repair_queue_postponed_rebuild_count', 0)}"),
            text_for(lang, f"- Nextgen Live 重排 Repair 数: {nextgen_live.get('repair_queue_reprioritized_count', 0)}", f"- Nextgen Live Reprioritized Repair Count: {nextgen_live.get('repair_queue_reprioritized_count', 0)}"),
            text_for(lang, f"- Nextgen Live 最近队列趋势: {nextgen_live.get('recent_repair_queue_summary') or '无'}", f"- Nextgen Live Recent Queue Trend: {nextgen_live.get('recent_repair_queue_summary') or 'none'}"),
            text_for(lang, f"- 最近模型生命周期事件: {latest_lifecycle.get('event_type', 'N/A')}", f"- Latest Model Lifecycle Event: {latest_lifecycle.get('event_type', 'N/A')}"),
            text_for(lang, f"- 健康报告存在: {bool(data.get('latest_health_report'))}", f"- Health Report Present: {bool(data.get('latest_health_report'))}"),
            text_for(lang, f"- 事故报告存在: {bool(data.get('latest_incident_report'))}", f"- Incident Report Present: {bool(data.get('latest_incident_report'))}"),
            text_for(lang, f"- 性能报告存在: {bool(data.get('latest_performance_report'))}", f"- Performance Report Present: {bool(data.get('latest_performance_report'))}"),
        ]
        return "\n".join(lines)

    @staticmethod
    def _market_data_failover_summary(route: dict, stats: dict) -> dict[str, str | bool | int]:
        fallback_count = 0
        primary_failures = 0
        secondary_failures = 0
        if isinstance(stats, dict):
            for payload in stats.values():
                payload = payload if isinstance(payload, dict) else {}
                fallback_count += int(payload.get("fallback_used", 0) or 0)
                primary_failures += int(payload.get("primary_failures", 0) or 0)
                secondary_failures += int(payload.get("secondary_failures", 0) or 0)
        route = route if isinstance(route, dict) else {}
        return {
            "latest_provider": str(route.get("selected_provider") or "unknown"),
            "latest_operation": str(route.get("operation") or "unknown"),
            "fallback_active": bool(route.get("fallback_used")),
            "fallback_count": fallback_count,
            "primary_failures": primary_failures,
            "secondary_failures": secondary_failures,
            "updated_at": str(route.get("updated_at") or ""),
        }
