import json
from datetime import datetime, timedelta, timezone

import dashboard as dashboard_module
import pandas as pd
from config import get_settings
from dashboard import build_runtime_override_payload, normalize_take_profit_levels
from core.models import SignalDirection
from core.dashboard_page_renderers import (
    _build_exit_reason_breakdowns,
    _build_promotion_funnel_rows,
    _build_promotion_funnel_summary,
    _build_promotion_candidate_rows,
    _build_promotion_observation_rows,
    _short_horizon_overview_notice,
)
from execution.paper_trader import PaperTrader
from monitor.ab_test_report import ABTestReporter
from monitor.drift_report import DriftReporter
from monitor.health_check import HealthChecker
from monitor.alpha_diagnostics_report import AlphaDiagnosticsReporter
from monitor.backtest_live_consistency_report import BacktestLiveConsistencyReporter
from monitor.daily_focus_report import DailyFocusReporter
from monitor.ops_overview import OpsOverviewService
from monitor.performance_report import PerformanceReporter
from monitor.pool_attribution_report import PoolAttributionReporter
from tests.v2_architecture_support import V2ArchitectureTestCase


class V2MonitorReportingTests(V2ArchitectureTestCase):
    def test_short_horizon_overview_notice_maps_status_to_alert_level(self):
        softened_override = _short_horizon_overview_notice(
            {
                "Short-Horizon Status": "positive_edge",
                "Short-Horizon Recent Samples": "8/6 (lookback 20)",
                "Short-Horizon Recent Expectancy": "+0.24%",
                "Short-Horizon Recent Profit Factor": "1.42",
                "Short-Horizon Softened Status": "disabled_negative_edge",
                "Short-Horizon Softened Expectancy": "-0.18%",
                "Short-Horizon Softened Profit Factor": "0.71",
            },
            current_language="zh",
        )
        positive = _short_horizon_overview_notice(
            {
                "Short-Horizon Status": "positive_edge",
                "Short-Horizon Recent Samples": "8/6 (lookback 20)",
                "Short-Horizon Recent Expectancy": "+0.24%",
                "Short-Horizon Recent Profit Factor": "1.42",
            },
            current_language="en",
        )
        warming = _short_horizon_overview_notice(
            {
                "Short-Horizon Status": "warming_up",
                "Short-Horizon Recent Samples": "2/6 (lookback 20)",
                "Short-Horizon Recent Expectancy": "+0.02%",
                "Short-Horizon Recent Profit Factor": "1.01",
            },
            current_language="zh",
        )
        negative = _short_horizon_overview_notice(
            {
                "Short-Horizon Status": "negative_edge_pause",
                "Short-Horizon Recent Samples": "9/6 (lookback 20)",
                "Short-Horizon Recent Expectancy": "-0.35%",
                "Short-Horizon Recent Profit Factor": "0.62",
            },
            current_language="en",
        )
        neutral = _short_horizon_overview_notice(
            {
                "Short-Horizon Status": "neutral",
                "Short-Horizon Recent Samples": "7/6 (lookback 20)",
                "Short-Horizon Recent Expectancy": "+0.01%",
                "Short-Horizon Recent Profit Factor": "0.99",
            },
            current_language="zh",
        )

        self.assertEqual(softened_override["level"], "error")
        self.assertIn("放行子集已转负", softened_override["message"])
        self.assertEqual(positive["level"], "success")
        self.assertIn("positive-edge", positive["message"])
        self.assertEqual(warming["level"], "info")
        self.assertIn("预热", warming["message"])
        self.assertEqual(negative["level"], "error")
        self.assertIn("De-risk", negative["message"])
        self.assertEqual(neutral["level"], "warning")
        self.assertIn("中性混合区间", neutral["message"])

    def test_strategy_settings_enable_paper_canary_by_default(self):
        settings = get_settings().model_copy(deep=True)
        self.assertTrue(settings.strategy.paper_canary_enabled)

    def test_dashboard_lifecycle_helpers_include_candidate_and_observation_thresholds(self):
        candidate_rows = _build_promotion_candidate_rows(
            {
                "BTC/USDT": {
                    "status": "live",
                    "registered_at": "2026-04-01T00:00:00+00:00",
                    "live_started_at": "2026-04-01T04:00:00+00:00",
                    "min_shadow_evaluations": 8,
                    "min_live_evaluations": 4,
                    "max_shadow_age_hours": 72,
                    "max_live_age_hours": 72,
                    "live_allocation_pct": 0.03,
                    "active_model_path": "data/models/xgboost_v2_BTC_USDT.json",
                    "challenger_model_path": "data/models/xgboost_challenger_BTC_USDT.json",
                    "shadow_eval_count": 10,
                    "shadow_accuracy": 0.63,
                    "shadow_expectancy_pct": 0.22,
                }
            }
        )
        observation_rows = _build_promotion_observation_rows(
            {
                "BTC/USDT": {
                    "status": "observing",
                    "promoted_at": "2026-04-01T08:00:00+00:00",
                    "min_evaluations": 8,
                    "max_observation_age_hours": 72,
                    "active_model_path": "data/models/xgboost_v2_BTC_USDT.json",
                    "backup_model_path": "data/models/backups/xgboost_v2_BTC_USDT.json",
                    "baseline_holdout_accuracy": 0.61,
                    "baseline_expectancy_pct": 0.18,
                    "baseline_profit_factor": 1.12,
                    "baseline_max_drawdown_pct": 0.45,
                    "recent_walkforward_baseline_summary": {
                        "history_count": 3,
                        "avg_expectancy_pct": 0.16,
                        "avg_profit_factor": 1.08,
                        "avg_max_drawdown_pct": 0.41,
                    },
                }
            }
        )

        self.assertEqual(len(candidate_rows), 1)
        self.assertEqual(candidate_rows[0]["status"], "live")
        self.assertEqual(candidate_rows[0]["min_shadow_evaluations"], 8)
        self.assertAlmostEqual(candidate_rows[0]["live_allocation_pct"], 0.03, places=6)
        self.assertEqual(candidate_rows[0]["challenger_model"], "xgboost_challenger_BTC_USDT.json")

        self.assertEqual(len(observation_rows), 1)
        self.assertEqual(observation_rows[0]["min_evaluations"], 8)
        self.assertEqual(observation_rows[0]["recent_wf_history_count"], 3)
        self.assertAlmostEqual(
            observation_rows[0]["baseline_profit_factor"],
            1.12,
            places=6,
        )

    def test_dashboard_promotion_funnel_helpers_build_stage_summary(self):
        latest_model_events = pd.DataFrame(
            [
                {
                    "event_type": "model_rollback",
                    "symbol": "AVAX/USDT",
                    "payload_json": '{"reason":"post_promotion_expectancy"}',
                    "created_at": "2026-04-01T10:00:00+00:00",
                },
                {
                    "event_type": "paper_canary_open",
                    "symbol": "AAVE/USDT",
                    "payload_json": '{"final_score":0.62}',
                    "created_at": "2026-04-01T09:00:00+00:00",
                },
            ]
        )
        rows = _build_promotion_funnel_rows(
            shadow_symbols=["BTC/USDT"],
            promotion_candidates={
                "ETH/USDT": {"status": "shadow"},
                "SOL/USDT": {"status": "live"},
            },
            model_observations={
                "FIL/USDT": {"status": "observing"},
            },
            latest_model_events=latest_model_events,
            load_json=lambda value, default=None: json.loads(value)
            if value
            else (default or {}),
        )
        summary = _build_promotion_funnel_summary(rows)

        stage_map = {row["symbol"]: row["funnel_stage"] for row in rows}
        self.assertEqual(stage_map["BTC/USDT"], "shadow_observation")
        self.assertEqual(stage_map["ETH/USDT"], "shadow_candidate")
        self.assertEqual(stage_map["SOL/USDT"], "live_canary")
        self.assertEqual(stage_map["FIL/USDT"], "promoted_observing")
        self.assertEqual(stage_map["AAVE/USDT"], "paper_canary_opened")
        self.assertEqual(stage_map["AVAX/USDT"], "rolled_back")
        self.assertEqual(summary["shadow_observation"], 1)
        self.assertEqual(summary["shadow_candidate"], 1)
        self.assertEqual(summary["live_canary"], 1)
        self.assertEqual(summary["promoted_observing"], 1)
        self.assertEqual(summary["paper_canary_opened"], 1)
        self.assertEqual(summary["rolled_back"], 1)

    def test_ops_overview_reports_promotion_funnel_counts(self):
        self.storage.set_json_state("shadow_observation_symbols", ["BTC/USDT", "ETH/USDT"])
        self.storage.set_json_state("paper_exploration_active_symbols", ["BTC/USDT"])
        self.storage.set_json_state(
            "model_promotion_candidates",
            {
                "BTC/USDT": {"status": "shadow"},
                "SOL/USDT": {"status": "live"},
            },
        )
        self.storage.set_json_state(
            "model_promotion_observations",
            {
                "FIL/USDT": {"status": "observing"},
            },
        )
        self.storage.insert_execution_event(
            "paper_canary_open",
            "SOL/USDT",
            {"final_score": 0.63, "canary_mode": "soft_review"},
        )
        self.storage.insert_execution_event(
            "fast_alpha_open",
            "BTC/USDT",
            {"review_policy_reason": "warming_up_soften|setup_avg_outcome=-0.14"},
        )
        self.storage.insert_execution_event(
            "fast_alpha_blocked",
            "BTC/USDT",
            {"reason": "short_horizon_negative_expectancy_pause"},
        )
        self.storage.insert_prediction_evaluation(
            {
                "symbol": "SOL/USDT",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "evaluation_type": "execution",
                "actual_up": True,
                "predicted_up": True,
                "is_correct": True,
                "entry_close": 100.0,
                "future_close": 101.0,
                "metadata": {},
            }
        )
        report_data = OpsOverviewService(self.storage).build()
        report_text = OpsOverviewService.render(report_data)

        self.assertEqual(report_data["shadow_candidate_count"], 1)
        self.assertEqual(report_data["live_candidate_count"], 1)
        self.assertEqual(report_data["promotion_observation_count"], 1)
        self.assertEqual(report_data["latest_canary"]["symbol"], "SOL/USDT")
        self.assertEqual(report_data["paper_canary_open_count"], 1)
        self.assertEqual(report_data["soft_paper_canary_open_count"], 1)
        self.assertEqual(report_data["fast_alpha_short_horizon_softened_open_count"], 1)
        self.assertEqual(report_data["fast_alpha_negative_expectancy_pause_count"], 1)
        self.assertEqual(report_data["execution_evaluation_count"], 1)
        self.assertIn("Shadow 候选模型数: 1", report_text)
        self.assertIn("Live Canary 模型数: 1", report_text)
        self.assertIn("Promotion 观察中模型数: 1", report_text)
        self.assertIn("Short-horizon 放行累计开仓: 1", report_text)
        self.assertIn("Short-horizon 负期望暂停累计: 1", report_text)
        self.assertIn("Paper 探索槽位: BTC/USDT", report_text)

    def test_ops_overview_service_includes_market_data_failover_summary(self):
        self.storage.set_json_state(
            "market_data_last_route",
            {
                "operation": "fetch_historical_ohlcv",
                "selected_provider": "binance",
                "fallback_used": True,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.storage.set_json_state(
            "market_data_failover_stats",
            {
                "fetch_historical_ohlcv": {
                    "fallback_used": 2,
                    "primary_failures": 2,
                    "secondary_failures": 0,
                },
                "measure_latency": {
                    "fallback_used": 1,
                    "primary_failures": 1,
                    "secondary_failures": 0,
                },
            },
        )

        report_data = OpsOverviewService(self.storage).build()
        report_text = OpsOverviewService.render(report_data)

        self.assertEqual(
            report_data["market_data_failover_summary"]["latest_provider"],
            "binance",
        )
        self.assertEqual(
            report_data["market_data_failover_summary"]["fallback_count"],
            3,
        )
        self.assertEqual(
            report_data["market_data_failover_summary"]["primary_failures"],
            3,
        )
        self.assertIn("最近市场数据提供方: binance", report_text)
        self.assertIn("市场数据 Failover 次数: 3", report_text)

    def test_ops_overview_service_includes_nextgen_live_queue_summary(self):
        self.storage.set_json_state(
            "nextgen_autonomy_live_operator_request",
            {
                "requested_live": True,
                "whitelist": ["BTC/USDT:USDT"],
                "max_active_runtimes": 1,
                "reason": "cli",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.storage.set_json_state(
            "nextgen_autonomy_live_status",
            {
                "requested_live": True,
                "effective_live": False,
                "allow_entries": False,
                "allow_managed_closes": True,
                "force_flatten": False,
                "kill_switch_active": False,
                "reasons": [],
            },
        )
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_run",
            "SYSTEM",
            {
                "requested_live": True,
                "trigger": "scheduler",
                "status": "ok",
                "repair_queue_hold_priority_count": 3,
                "repair_queue_postponed_rebuild_count": 2,
                "repair_queue_reprioritized_count": 4,
                "repair_queue_hold_priority_active": True,
                "repair_queue_postponed_rebuild_active": True,
                "repair_queue_reprioritized_active": True,
            },
        )
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_run",
            "SYSTEM",
            {
                "requested_live": True,
                "trigger": "manual_recovery_required",
                "status": "ok",
                "repair_queue_hold_priority_count": 1,
                "repair_queue_postponed_rebuild_count": 0,
                "repair_queue_reprioritized_count": 1,
            },
        )

        report_data = OpsOverviewService(self.storage).build()
        report_text = OpsOverviewService.render(report_data)

        self.assertEqual(
            report_data["nextgen_live"]["repair_queue_hold_priority_count"],
            1,
        )
        self.assertEqual(
            report_data["nextgen_live"]["repair_queue_postponed_rebuild_count"],
            0,
        )
        self.assertEqual(
            report_data["nextgen_live"]["repair_queue_reprioritized_count"],
            1,
        )
        self.assertEqual(len(report_data["nextgen_live"]["recent_repair_queue_runs"]), 2)
        self.assertTrue(report_data["nextgen_live"]["recent_repair_queue_summary"])
        self.assertIn("Nextgen Live Hold Repair 数: 1", report_text)
        self.assertIn("Nextgen Live 延后 Rebuild 数: 0", report_text)
        self.assertIn("Nextgen Live 重排 Repair 数: 1", report_text)
        self.assertIn("Nextgen Live 最近队列趋势:", report_text)

    def test_health_checker_reports_basic_status(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.symbols = ["BTC/USDT"]
        settings.exchange.timeframes = ["1h"]
        self.storage.set_state("last_cycle_started", datetime.now(timezone.utc).isoformat())
        self.storage.set_state("last_cycle_completed", datetime.now(timezone.utc).isoformat())
        self.storage.set_state("last_cycle_status", "ok")
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "1h",
            [
                {
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1.0,
                }
            ],
        )
        checker = HealthChecker(self.storage, settings)
        status = checker.run()
        report = checker.render_report(status)
        self.assertIn(status.status, {"ok", "warming_up", "degraded"})
        self.assertIn("健康检查报告", report)
        self.assertEqual(status.stale_market_streams, 0)

    def test_health_checker_includes_market_data_failover_summary(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        self.storage.set_json_state(
            "market_data_last_route",
            {
                "operation": "measure_latency",
                "selected_provider": "binance",
                "fallback_used": True,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.storage.set_json_state(
            "market_data_failover_stats",
            {
                "measure_latency": {
                    "fallback_used": 1,
                    "primary_failures": 1,
                    "secondary_failures": 0,
                }
            },
        )

        checker = HealthChecker(self.storage, settings)
        status = checker.run()
        report = checker.render_report(status)

        self.assertEqual(status.latest_market_data_provider, "binance")
        self.assertTrue(status.market_data_failover_active)
        self.assertEqual(status.market_data_failover_count, 1)
        self.assertEqual(status.market_data_primary_failures, 1)
        self.assertIn("最近市场数据提供方: binance", report)
        self.assertIn("市场数据 Failover 激活: True", report)

    def test_performance_reporter_builds_symbol_edge_summary_from_expectancy(self):
        for idx, (symbol, is_correct, trade_net_return_pct) in enumerate(
            [
                ("BTC/USDT", True, 0.60),
                ("BTC/USDT", True, 0.30),
                ("BTC/USDT", False, -0.10),
                ("ETH/USDT", True, 0.20),
                ("ETH/USDT", False, -0.80),
                ("ETH/USDT", False, -0.40),
            ],
            start=1,
        ):
            self.storage.insert_prediction_evaluation(
                {
                    "symbol": symbol,
                    "timestamp": (
                        datetime.now(timezone.utc) - timedelta(hours=idx * 4)
                    ).isoformat(),
                    "evaluation_type": "execution",
                    "actual_up": bool(is_correct),
                    "predicted_up": True,
                    "is_correct": bool(is_correct),
                    "entry_close": 100.0,
                    "future_close": 101.0 if is_correct else 99.0,
                    "metadata": {
                        "trade_net_return_pct": trade_net_return_pct,
                        "cost_pct": 0.15,
                    },
                }
            )
        summary = PerformanceReporter(self.storage).build_symbol_edge_summary(limit=50)

        self.assertGreater(summary["BTC/USDT"]["expectancy_pct"], 0.0)
        self.assertGreater(summary["BTC/USDT"]["profit_factor"], 1.0)
        self.assertLess(summary["ETH/USDT"]["expectancy_pct"], 0.0)
        self.assertGreater(
            summary["BTC/USDT"]["objective_score"],
            summary["ETH/USDT"]["objective_score"],
        )

    def test_performance_reporter_builds_short_horizon_pipeline_summary(self):
        trader = PaperTrader(self.storage, initial_balance=10000.0)
        trader.execute_open(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.7,
            rationale="fast alpha entry",
            quantity=1.0,
            metadata={"pipeline_mode": "fast_alpha"},
        )
        trader.execute_close("BTC/USDT", 102.0, "take_profit")
        trader.execute_open(
            symbol="ETH/USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.7,
            rationale="paper canary entry",
            quantity=1.0,
            metadata={"pipeline_mode": "paper_canary"},
        )
        trader.execute_close("ETH/USDT", 99.0, "stop_loss")

        summary = PerformanceReporter(self.storage).build_pipeline_mode_summary(
            ["fast_alpha", "paper_canary"],
            limit=10,
        )

        self.assertEqual(summary["fast_alpha"]["closed_trade_count"], 1)
        self.assertEqual(summary["paper_canary"]["closed_trade_count"], 1)
        self.assertEqual(summary["_combined"]["closed_trade_count"], 2)
        self.assertEqual(summary["_combined"]["mode_counts"]["fast_alpha"], 1)
        self.assertEqual(summary["_combined"]["mode_counts"]["paper_canary"], 1)

    def test_pool_attribution_reporter_builds_trade_and_review_summary(self):
        self.storage.set_json_state("execution_symbols", ["BTC/USDT", "ETH/USDT"])
        trader = PaperTrader(self.storage, initial_balance=10000.0)
        trader.execute_open(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.7,
            rationale="paper canary entry",
            quantity=1.0,
            metadata={
                "pipeline_mode": "paper_canary",
                "paper_canary_mode": "soft_review",
                "reviewed_action": "HOLD",
                "review_score": 0.12,
                "decision_reason": "extreme_fear_offensive_setup",
            },
        )
        trader.execute_close(
            "BTC/USDT",
            102.0,
            "research_de_risk,research_exit_watch",
            close_qty=0.5,
        )
        trader.execute_close("BTC/USDT", 104.0, "research_exit")
        self.storage.insert_execution_event(
            "research_review",
            "ETH/USDT",
            {
                "reviewed_action": "CLOSE",
                "review_score": -0.63,
                "reasons": ["liquidity_weak", "news_coverage_thin"],
            },
        )

        result = PoolAttributionReporter(self.storage).build()
        report = PoolAttributionReporter.render(result)

        self.assertEqual(result["symbols"], ["BTC/USDT", "ETH/USDT"])
        self.assertEqual(result["closed_trade_count"], 1)
        self.assertGreater(result["total_net_pnl"], 0.0)
        btc_summary = next(
            row for row in result["symbol_summary"] if row["symbol"] == "BTC/USDT"
        )
        eth_summary = next(
            row for row in result["symbol_summary"] if row["symbol"] == "ETH/USDT"
        )
        self.assertEqual(btc_summary["trade_count"], 1)
        self.assertIn("research_exit", btc_summary["top_exit_reasons"])
        self.assertEqual(eth_summary["latest_reviewed_action"], "CLOSE")
        self.assertIn("交易池归因报告", report)
        self.assertIn("ETH/USDT", report)

    def test_pool_attribution_reporter_counts_partial_realized_pnl_for_open_trade(self):
        self.storage.set_json_state("execution_symbols", ["BTC/USDT"])
        trader = PaperTrader(self.storage, initial_balance=10000.0)
        trader.execute_open(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.7,
            rationale="paper canary runner",
            quantity=1.0,
            metadata={
                "pipeline_mode": "paper_canary",
                "paper_canary_mode": "soft_review",
                "reviewed_action": "HOLD",
                "review_score": 0.12,
                "decision_reason": "runner setup",
            },
        )
        trader.execute_close(
            "BTC/USDT",
            103.0,
            "research_de_risk,research_exit_watch",
            close_qty=0.5,
        )

        result = PoolAttributionReporter(self.storage).build()
        report = PoolAttributionReporter.render(result)

        self.assertEqual(result["closed_trade_count"], 0)
        self.assertEqual(result["open_trade_count"], 1)
        self.assertGreater(result["partial_realized_net_pnl"], 0.0)
        btc_summary = result["symbol_summary"][0]
        self.assertGreater(btc_summary["partial_realized_net_pnl"], 0.0)
        self.assertIn("未全平已实现净收益", report)

    def test_alpha_diagnostics_reporter_summarizes_mix_and_blocked_setups(self):
        self.storage.set_json_state("execution_symbols", ["BTC/USDT"])
        self.storage.set_json_state(
            "runtime_settings_learning_details",
            {
                "blocked_setups": [
                    {
                        "criteria": {"symbol": "BTC/USDT", "regime": "EXTREME_FEAR"},
                        "reason": "recent_realized_setup_negative_expectancy_-0.22",
                    }
                ],
                "stats": {
                    "recent_negative_setup_pause_count": 1,
                    "shadow_rehabilitated_setup_count": 0,
                },
                "reasons": ["recent_negative_setup_pauses_1"],
            },
        )
        self.storage.insert_execution_event(
            "paper_canary_open",
            "BTC/USDT",
            {"canary_mode": "soft_review"},
        )
        self.storage.insert_execution_event(
            "paper_canary_open",
            "BTC/USDT",
            {"canary_mode": "offensive_review"},
        )
        self.storage.insert_execution_event(
            "paper_canary_open",
            "ETH/USDT",
            {"canary_mode": "soft_review"},
        )
        self.storage.insert_execution_event(
            "position_review_watch",
            "BTC/USDT",
            {"trade_id": "t1", "research_exit_count": 2},
        )
        trader = PaperTrader(self.storage, initial_balance=10000.0)
        trader.execute_open(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.7,
            rationale="alpha diag entry",
            quantity=1.0,
            metadata={
                "pipeline_mode": "paper_canary",
                "paper_canary_mode": "soft_review",
                "reviewed_action": "HOLD",
                "review_score": -0.01,
                "decision_reason": "strong near miss",
            },
        )
        trader.execute_close("BTC/USDT", 99.2, "research_exit")

        trader.execute_open(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.8,
            rationale="paper canary strong open",
            quantity=1.0,
            metadata={
                "pipeline_mode": "paper_canary",
                "paper_canary_mode": "primary_review",
                "reviewed_action": "OPEN_LONG",
                "review_score": 0.20,
                "decision_reason": "primary review strong open",
            },
        )
        trader.execute_close(
            "BTC/USDT",
            99.7,
            "research_de_risk,research_exit_watch",
            close_qty=0.5,
        )
        trader.execute_close("BTC/USDT", 99.5, "research_exit")

        trader.execute_open(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.8,
            rationale="fast alpha strong open",
            quantity=1.0,
            metadata={
                "pipeline_mode": "fast_alpha",
                "reviewed_action": "OPEN_LONG",
                "review_score": 0.35,
                "decision_reason": "near miss strong open",
                "fast_alpha_review_policy_reason": "warming_up_soften|setup_avg_outcome=-0.14",
                "fast_alpha_review_policy_relaxed_reasons": ["setup_negative_expectancy"],
            },
        )
        self.storage.insert_execution_event(
            "fast_alpha_open",
            "BTC/USDT",
            {
                "review_policy_reason": "warming_up_soften|setup_avg_outcome=-0.14",
                "effective_review_score": 0.14,
            },
        )
        self.storage.insert_execution_event(
            "fast_alpha_blocked",
            "BTC/USDT",
            {
                "reason": "short_horizon_negative_expectancy_pause",
                "adaptive_reason": "negative_edge_pause",
            },
        )
        trader.execute_close(
            "BTC/USDT",
            99.6,
            "research_de_risk,research_exit_watch",
            close_qty=0.5,
        )
        trader.execute_close("BTC/USDT", 99.4, "research_exit")

        report_data = AlphaDiagnosticsReporter(self.storage, self.settings).build()
        report_text = AlphaDiagnosticsReporter.render(report_data)

        self.assertEqual(report_data["paper_canary_mix"]["total"]["soft_review"], 1)
        self.assertEqual(report_data["paper_canary_mix"]["total"]["offensive_review"], 1)
        self.assertEqual(report_data["learning"]["blocked_setup_count"], 1)
        self.assertEqual(report_data["review_watch"]["recent_24h_count"], 1)
        self.assertEqual(report_data["paper_canary_mix"]["strong_open_trade_count"], 1)
        self.assertEqual(report_data["paper_canary_mix"]["early_research_de_risk_trade_count"], 1)
        self.assertEqual(report_data["paper_canary_mix"]["early_research_exit_trade_count"], 1)
        self.assertEqual(report_data["fast_alpha"]["strong_open_trade_count"], 1)
        self.assertEqual(report_data["fast_alpha"]["early_research_de_risk_trade_count"], 1)
        self.assertEqual(report_data["fast_alpha"]["early_research_exit_trade_count"], 1)
        self.assertEqual(
            report_data["fast_alpha"]["short_horizon_policy"]["softened_open_count"],
            1,
        )
        self.assertEqual(
            report_data["fast_alpha"]["short_horizon_policy"]["warming_up_softened_open_count"],
            1,
        )
        self.assertEqual(
            report_data["fast_alpha"]["short_horizon_policy"]["negative_expectancy_pause_count"],
            1,
        )
        self.assertEqual(
            report_data["fast_alpha"]["short_horizon_policy"]["softened_closed_trade_count"],
            1,
        )
        self.assertEqual(
            report_data["fast_alpha"]["short_horizon_policy"]["softened_status"],
            "insufficient_samples",
        )
        self.assertEqual(
            report_data["fast_alpha"]["short_horizon_status"]["status"],
            "warming_up",
        )
        self.assertIn("Alpha 诊断日报", report_text)
        self.assertIn("soft占比", report_text)
        self.assertIn("recent_realized_setup_negative_expectancy", report_text)
        self.assertIn("Fast Alpha", report_text)
        self.assertIn("Short-horizon 当前状态", report_text)
        self.assertIn("Short-horizon 放行后状态", report_text)
        self.assertIn("Short-horizon 放行开仓", report_text)
        self.assertIn("负期望暂停次数", report_text)
        self.assertIn("Canary 强质量开仓数", report_text)
        self.assertIn("1h内 research_de_risk", report_text)

    def test_dashboard_exit_reason_breakdowns_group_by_symbol_and_setup(self):
        close_rows = pd.DataFrame(
            [
                {
                    "symbol": "BTC/USDT",
                    "net_pnl": -2.0,
                    "net_return_pct": -2.0,
                    "metadata_json": json.dumps(
                        {
                            "reason": "research_exit",
                            "pipeline_mode": "execution",
                            "entry_thesis": "high_conviction_long",
                            "setup_profile": {
                                "regime": "EXTREME_FEAR",
                                "liquidity_bucket": "strong",
                            },
                        }
                    ),
                },
                {
                    "symbol": "BTC/USDT",
                    "net_pnl": 1.0,
                    "net_return_pct": 1.0,
                    "metadata_json": json.dumps(
                        {
                            "reason": "research_de_risk,research_exit_watch",
                            "pipeline_mode": "paper_canary",
                            "entry_thesis": "paper_canary_primary",
                            "setup_profile": {
                                "regime": "EXTREME_FEAR",
                                "liquidity_bucket": "strong",
                            },
                        }
                    ),
                },
                {
                    "symbol": "ETH/USDT",
                    "net_pnl": 0.5,
                    "net_return_pct": 0.5,
                    "metadata_json": json.dumps(
                        {
                            "reason": "evidence_de_risk",
                            "pipeline_mode": "execution",
                            "entry_thesis": "generic_long",
                            "setup_profile": {
                                "regime": "RANGE",
                                "liquidity_bucket": "mid",
                            },
                        }
                    ),
                },
            ]
        )

        symbol_df, setup_df = _build_exit_reason_breakdowns(
            close_rows,
            load_json=lambda value, default=None: json.loads(value) if value else (default or {}),
        )

        btc_row = symbol_df[symbol_df["symbol"] == "BTC/USDT"].iloc[0].to_dict()
        self.assertEqual(btc_row["focus_close_count"], 2)
        self.assertEqual(btc_row["research_exit_count"], 1)
        self.assertEqual(btc_row["research_de_risk_count"], 1)

        eth_row = symbol_df[symbol_df["symbol"] == "ETH/USDT"].iloc[0].to_dict()
        self.assertEqual(eth_row["evidence_de_risk_count"], 1)

        self.assertTrue((setup_df["setup"].astype(str).str.contains("high_conviction_long")).any())
        self.assertTrue((setup_df["setup"].astype(str).str.contains("paper_canary_primary")).any())

    def test_daily_focus_reporter_builds_five_key_metrics(self):
        trader = PaperTrader(self.storage, initial_balance=10000.0)
        trader.execute_open(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.7,
            rationale="daily focus entry",
            quantity=1.0,
            metadata={"pipeline_mode": "execution"},
        )
        trader.execute_close("BTC/USDT", 105.0, "take_profit")
        trader.execute_open(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.7,
            rationale="daily focus fast alpha softened",
            quantity=1.0,
            metadata={
                "pipeline_mode": "fast_alpha",
                "fast_alpha_review_policy_reason": "warming_up_soften|setup_avg_outcome=-0.14",
            },
        )
        self.storage.insert_execution_event(
            "fast_alpha_open",
            "BTC/USDT",
            {
                "review_policy_reason": "warming_up_soften|setup_avg_outcome=-0.14",
            },
        )
        self.storage.insert_execution_event(
            "fast_alpha_blocked",
            "BTC/USDT",
            {
                "reason": "short_horizon_negative_expectancy_pause",
            },
        )
        trader.execute_close("BTC/USDT", 101.0, "take_profit")
        self.storage.insert_shadow_trade_run(
            {
                "symbol": "BTC/USDT",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "block_reason": "risk_block:insufficient_liquidity",
                "direction": "LONG",
                "entry_price": 100.0,
                "horizon_hours": 4,
                "status": "closed",
                "exit_price": 96.0,
                "pnl_pct": -4.0,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "setup_profile": {},
                "metadata": {},
            }
        )

        result = DailyFocusReporter(self.storage, self.settings).build(["BTC/USDT"])
        report = DailyFocusReporter.render(result)

        self.assertGreater(result["daily_net_pnl"], 0.0)
        self.assertGreater(result["daily_profit_factor"], 1.0)
        self.assertGreaterEqual(result["daily_max_drawdown_pct"], 0.0)
        self.assertGreater(result["avg_net_pnl_per_trade"], 0.0)
        self.assertAlmostEqual(result["blocked_avoided_loss_pct"], 4.0, places=6)
        self.assertEqual(result["fast_alpha_open_count"], 1)
        self.assertEqual(result["fast_alpha_short_horizon_open_count"], 1)
        self.assertEqual(result["fast_alpha_short_horizon_closed_trade_count"], 1)
        self.assertEqual(result["fast_alpha_negative_expectancy_pause_count"], 1)
        self.assertGreater(result["fast_alpha_short_horizon_net_pnl"], 0.0)
        self.assertIn("每日焦点复盘", report)
        self.assertIn("风控避免亏损", report)
        self.assertIn("Fast Alpha 净收益", report)
        self.assertIn("Short-horizon 放行开仓数", report)
        self.assertIn("Short-horizon 负期望暂停次数", report)

    def test_backtest_live_consistency_reporter_flags_extreme_walkforward_drift(self):
        now = datetime.now(timezone.utc)
        self.storage.set_json_state("execution_symbols", ["BTC/USDT"])
        self.storage.insert_backtest_run(
            symbol="BTC/USDT",
            engine="v2",
            summary={
                "total_return_pct": 1200.0,
                "profit_factor": 3.2,
                "max_drawdown_pct": 4.0,
            },
            trades=[],
        )
        self.storage.insert_walkforward_run(
            {
                "symbol": "BTC/USDT",
                "splits": [],
                "summary": {
                    "total_splits": 6,
                    "trade_count": 6,
                    "total_return_pct": 950.0,
                    "profit_factor": 2.4,
                },
            }
        )
        for idx in range(5):
            trade_id = f"live-loss-{idx}"
            self.storage.insert_trade(
                {
                    "id": trade_id,
                    "symbol": "BTC/USDT",
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "quantity": 1.0,
                    "entry_time": now.isoformat(),
                    "rationale": "live trade",
                    "confidence": 0.7,
                }
            )
            self.storage.update_trade_exit(
                trade_id,
                95.0,
                now.isoformat(),
                -5.0,
                -5.0,
            )
        self.storage.insert_execution_event("paper_canary_open", "BTC/USDT", {"canary_mode": "primary_review"})

        result = BacktestLiveConsistencyReporter(self.storage, self.settings).build(
            ["BTC/USDT"]
        )
        report = BacktestLiveConsistencyReporter.render(result)

        self.assertEqual(result["suspicious_symbol_count"], 1)
        self.assertIn("walkforward_live_pnl_divergence", result["rows"][0]["flags"])
        self.assertIn("回测/实盘一致性校验", report)

    def test_backtest_live_consistency_reporter_softens_sparse_walkforward_divergence(self):
        now = datetime.now(timezone.utc)
        self.storage.set_json_state("execution_symbols", ["ETH/USDT"])
        self.storage.insert_walkforward_run(
            {
                "symbol": "ETH/USDT",
                "splits": [],
                "summary": {
                    "total_splits": 7,
                    "trade_count": 2,
                    "total_return_pct": 830.0,
                    "profit_factor": 5.0,
                },
            }
        )
        for idx in range(5):
            trade_id = f"eth-live-loss-{idx}"
            self.storage.insert_trade(
                {
                    "id": trade_id,
                    "symbol": "ETH/USDT",
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "quantity": 1.0,
                    "entry_time": now.isoformat(),
                    "rationale": "live trade",
                    "confidence": 0.7,
                }
            )
            self.storage.update_trade_exit(
                trade_id,
                95.0,
                now.isoformat(),
                -5.0,
                -5.0,
            )

        result = BacktestLiveConsistencyReporter(self.storage, self.settings).build(
            ["ETH/USDT"]
        )

        self.assertNotIn("walkforward_live_pnl_divergence", result["rows"][0]["flags"])
        self.assertNotIn("profit_factor_divergence", result["rows"][0]["flags"])
        self.assertIn("walkforward_return_extreme_sparse", result["rows"][0]["flags"])

    def test_backtest_live_consistency_reporter_softens_sparse_extreme_walkforward_return(self):
        self.storage.set_json_state("execution_symbols", ["ETH/USDT"])
        self.storage.insert_backtest_run(
            symbol="ETH/USDT",
            engine="v2",
            summary={
                "total_return_pct": 140.0,
                "profit_factor": 1.3,
                "max_drawdown_pct": 5.0,
                "trade_count": 8,
            },
            trades=[],
        )
        self.storage.insert_walkforward_run(
            {
                "symbol": "ETH/USDT",
                "splits": [],
                "summary": {
                    "total_splits": 7,
                    "trade_count": 2,
                    "total_return_pct": 830.0,
                    "profit_factor": 5.0,
                },
            }
        )

        result = BacktestLiveConsistencyReporter(self.storage, self.settings).build(
            ["ETH/USDT"]
        )

        self.assertIn("walkforward_return_extreme_sparse", result["rows"][0]["flags"])
        self.assertNotIn("walkforward_return_extreme", result["rows"][0]["flags"])
        self.assertEqual(result["rows"][0]["walkforward_trade_count"], 2)

    def test_performance_reporter_merges_closed_paper_canary_feedback_into_edge_summary(self):
        trader = PaperTrader(self.storage, initial_balance=10000.0)
        trader.execute_open(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.6,
            rationale="paper canary",
            position_value=100.0,
            metadata={
                "pipeline_mode": "paper_canary",
                "prediction_timestamp": "2026-04-02T00:00:00+00:00",
            },
        )
        trader.execute_close("BTC/USDT", 106.0, reason="take_profit")

        summary = PerformanceReporter(self.storage).build_symbol_edge_summary(limit=20)

        self.assertIn("BTC/USDT", summary)
        self.assertEqual(summary["BTC/USDT"]["executed_count"], 1)
        self.assertEqual(summary["BTC/USDT"]["sample_count"], 1)
        self.assertGreater(summary["BTC/USDT"]["expectancy_pct"], 0.0)
        self.assertGreater(summary["BTC/USDT"]["objective_score"], 0.0)

    def test_performance_reporter_merges_partial_paper_canary_feedback_into_edge_summary(self):
        trader = PaperTrader(self.storage, initial_balance=10000.0)
        trader.execute_open(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            price=100.0,
            confidence=0.6,
            rationale="paper canary partial",
            quantity=1.0,
            metadata={
                "pipeline_mode": "paper_canary",
                "prediction_timestamp": "2026-04-02T01:00:00+00:00",
            },
        )
        trader.execute_close("BTC/USDT", 105.0, reason="take_profit", close_qty=0.5)

        summary = PerformanceReporter(self.storage).build_symbol_edge_summary(limit=20)

        self.assertIn("BTC/USDT", summary)
        self.assertEqual(summary["BTC/USDT"]["executed_count"], 1)
        self.assertGreater(summary["BTC/USDT"]["expectancy_pct"], 0.0)

    def test_health_checker_respects_analysis_interval_for_stale_streams(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.exchange.symbols = ["BTC/USDT"]
        settings.exchange.timeframes = ["1h"]
        settings.strategy.analysis_interval_seconds = 5400
        now = datetime.now(timezone.utc)
        self.storage.set_state("last_cycle_started", now.isoformat())
        self.storage.set_state("last_cycle_completed", now.isoformat())
        self.storage.set_state("last_cycle_status", "ok")
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "1h",
            [
                {
                    "timestamp": int((now - timedelta(minutes=65)).timestamp() * 1000),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1.0,
                }
            ],
        )
        checker = HealthChecker(self.storage, settings)
        status = checker.run()
        self.assertEqual(status.stale_market_streams, 0)

    def test_performance_reporter_aggregates_metrics(self):
        prediction_time = datetime.now(timezone.utc) - timedelta(hours=8)
        current_candle_ts = int((prediction_time - timedelta(minutes=1)).timestamp() * 1000)
        future_candle_ts = int((prediction_time + timedelta(hours=4, minutes=1)).timestamp() * 1000)
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "4h",
            [
                {
                    "timestamp": current_candle_ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1.0,
                },
                {
                    "timestamp": future_candle_ts,
                    "open": 101.0,
                    "high": 111.0,
                    "low": 100.0,
                    "close": 110.0,
                    "volume": 1.5,
                },
            ],
        )
        self.storage.insert_prediction_run(
            {
                "symbol": "BTC/USDT",
                "timestamp": prediction_time.isoformat(),
                "model_version": "test",
                "up_probability": 0.82,
                "feature_count": 56,
                "research": {
                    "symbol": "BTC/USDT",
                    "market_regime": "UPTREND",
                    "sentiment_score": 0.2,
                    "confidence": 0.6,
                    "risk_warning": [],
                    "key_reason": ["ok"],
                    "suggested_action": "OPEN_LONG",
                    "raw_content": "",
                    "timestamp": prediction_time.isoformat(),
                },
                "decision": {
                    "final_score": 0.91,
                    "regime": "UPTREND",
                    "suggested_action": "OPEN_LONG",
                },
            }
        )
        self.storage.insert_trade(
            {
                "id": "closed-1",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        self.storage.update_trade_exit(
            "closed-1",
            110.0,
            datetime.now(timezone.utc).isoformat(),
            10.0,
            10.0,
        )
        self.storage.insert_account_snapshot(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "equity": 10010.0,
                "realized_pnl": 10.0,
                "unrealized_pnl": 0.0,
                "daily_loss_pct": 0.0,
                "weekly_loss_pct": 0.0,
                "drawdown_pct": 0.01,
                "total_exposure_pct": 0.0,
                "open_positions": 0,
                "cooldown_until": None,
                "circuit_breaker_active": False,
            }
        )
        self.storage.insert_training_run(
            {
                "symbol": "BTC/USDT",
                "rows": 100,
                "feature_count": 10,
                "positives": 55,
                "negatives": 45,
                "model_path": "x.json",
                "trained_with_xgboost": True,
                "holdout_accuracy": 0.66,
            }
        )
        self.storage.insert_walkforward_run(
            {
                "symbol": "BTC/USDT",
                "splits": [],
                "summary": {"total_splits": 1, "total_return_pct": 5.5},
            }
        )
        snapshot = PerformanceReporter(self.storage).build()
        report = PerformanceReporter.render(snapshot)
        self.assertEqual(snapshot.total_closed_trades, 1)
        self.assertEqual(snapshot.prediction_eval_count, 1)
        self.assertEqual(
            snapshot.prediction_window_size,
            PerformanceReporter.RECENT_PREDICTION_EVALUATED_TARGET,
        )
        self.assertEqual(snapshot.current_prediction_eval_count, 1)
        self.assertEqual(
            snapshot.current_prediction_window_size,
            PerformanceReporter.RECENT_PREDICTION_EVALUATED_TARGET,
        )
        self.assertGreaterEqual(snapshot.xgboost_accuracy_pct, 100.0)
        self.assertGreaterEqual(snapshot.llm_accuracy_pct, 100.0)
        self.assertGreaterEqual(snapshot.fusion_accuracy_pct, 100.0)
        self.assertGreaterEqual(snapshot.current_xgboost_accuracy_pct, 100.0)
        self.assertEqual(snapshot.expanded_prediction_eval_count, 1)
        self.assertEqual(
            snapshot.expanded_prediction_window_size,
            PerformanceReporter.EXPANDED_PREDICTION_EVALUATED_TARGET,
        )
        self.assertGreaterEqual(snapshot.expanded_xgboost_accuracy_pct, 100.0)
        self.assertIsInstance(snapshot.llm_runtime_configured, bool)
        self.assertEqual(snapshot.research_fallback_count, 0)
        self.assertEqual(snapshot.research_total_count, 1)
        self.assertEqual(snapshot.degradation_status, "warming_up")
        self.assertIn("性能报告", report)
        self.assertIn(
            f"最近预测窗口样本: 1/{PerformanceReporter.RECENT_PREDICTION_EVALUATED_TARGET}",
            report,
        )
        self.assertIn(
            f"当前执行宇宙预测窗口样本: 1/{PerformanceReporter.RECENT_PREDICTION_EVALUATED_TARGET}",
            report,
        )
        self.assertIn("XGBoost 方向准确率: N/A", report)
        self.assertIn("当前执行宇宙 XGBoost 方向准确率: N/A", report)
        self.assertIn("扩展窗口 XGBoost 方向准确率: N/A", report)
        self.assertIn("执行闭环准确率: N/A", report)

    def test_performance_reporter_renders_execution_and_shadow_accuracy(self):
        now = datetime.now(timezone.utc).isoformat()
        self.storage.insert_prediction_evaluation(
            {
                "symbol": "BTC/USDT",
                "timestamp": now,
                "evaluation_type": "execution",
                "actual_up": True,
                "predicted_up": True,
                "is_correct": True,
                "entry_close": 100.0,
                "future_close": 104.0,
                "metadata": {"trade_net_return_pct": 4.0},
            }
        )
        self.storage.insert_prediction_evaluation(
            {
                "symbol": "ETH/USDT",
                "timestamp": now,
                "evaluation_type": "shadow_observation",
                "actual_up": True,
                "predicted_up": False,
                "is_correct": False,
                "entry_close": 100.0,
                "future_close": 98.0,
                "metadata": {"trade_net_return_pct": -2.0},
            }
        )

        snapshot = PerformanceReporter(self.storage).build()
        report = PerformanceReporter.render(snapshot)

        self.assertEqual(snapshot.execution_evaluation_count, 1)
        self.assertEqual(snapshot.shadow_evaluation_count, 1)
        self.assertAlmostEqual(snapshot.execution_accuracy_pct, 100.0, places=6)
        self.assertAlmostEqual(snapshot.shadow_accuracy_pct, 0.0, places=6)
        self.assertIn("执行闭环准确率: 100.00% (1)", report)
        self.assertIn("Shadow 观察准确率: 0.00% (1)", report)

    def test_performance_degradation_ignores_missing_fusion_samples(self):
        reporter = PerformanceReporter(self.storage)
        result = reporter._detect_degradation(
            xgboost_accuracy=60.0,
            fusion_accuracy=None,
            holdout_accuracy=60.0,
            prediction_eval_count=10,
        )
        self.assertEqual(result["status"], "healthy")

    def test_performance_degradation_tightens_thresholds_in_paper_mode(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "paper"
        reporter = PerformanceReporter(self.storage, settings)

        result = reporter._detect_degradation(
            xgboost_accuracy=49.0,
            fusion_accuracy=48.0,
            holdout_accuracy=54.0,
            prediction_eval_count=20,
        )

        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["reason"], "paper_mode_accuracy_guard")
        self.assertGreater(
            result["recommended_xgboost_threshold"],
            settings.model.xgboost_probability_threshold,
        )
        self.assertGreater(
            result["recommended_final_score_threshold"],
            settings.model.final_score_threshold,
        )

    def test_performance_degradation_uses_exploration_grace_after_canary_open(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "paper"
        reporter = PerformanceReporter(self.storage, settings)

        result = reporter._detect_degradation(
            xgboost_accuracy=49.0,
            fusion_accuracy=48.0,
            holdout_accuracy=54.0,
            prediction_eval_count=20,
            total_closed_trades=0,
            paper_canary_open_count=1,
        )

        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["reason"], "paper_mode_accuracy_guard_exploration_grace")
        self.assertTrue(result["paper_exploration_grace_active"])
        self.assertAlmostEqual(
            result["recommended_xgboost_threshold"],
            settings.model.xgboost_probability_threshold
            + settings.risk.paper_exploration_grace_threshold_tighten_pct,
            places=6,
        )

    def test_performance_degradation_prefers_positive_realized_edge_over_accuracy(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "paper"
        reporter = PerformanceReporter(self.storage, settings)

        result = reporter._detect_degradation(
            xgboost_accuracy=42.0,
            fusion_accuracy=41.0,
            holdout_accuracy=54.0,
            prediction_eval_count=40,
            recent_closed_trades=5,
            recent_expectancy_pct=0.35,
            recent_profit_factor=1.20,
            recent_max_drawdown_pct=2.0,
        )

        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["reason"], "realized_edge_override")

    def test_performance_degradation_flags_negative_realized_edge(self):
        settings = get_settings().model_copy(deep=True)
        settings.app.db_path = self.db_path
        settings.app.runtime_mode = "paper"
        reporter = PerformanceReporter(self.storage, settings)

        result = reporter._detect_degradation(
            xgboost_accuracy=61.0,
            fusion_accuracy=60.0,
            holdout_accuracy=61.0,
            prediction_eval_count=40,
            recent_closed_trades=5,
            recent_expectancy_pct=-0.20,
            recent_profit_factor=0.80,
            recent_max_drawdown_pct=6.0,
        )

        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["reason"], "realized_edge_negative")

    def test_performance_reporter_counts_partial_realized_pnl_from_open_trades(self):
        self.storage.insert_trade(
            {
                "id": "open-partial-1",
                "symbol": "BTC/USDT",
                "direction": "LONG",
                "entry_price": 100.0,
                "quantity": 1.0,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "rationale": "x",
                "confidence": 0.9,
            }
        )
        self.storage.upsert_trade_partial_close(
            "open-partial-1",
            closed_qty=0.5,
            exit_price=110.0,
            exit_time=datetime.now(timezone.utc).isoformat(),
            realized_pnl=5.0,
            realized_pnl_pct=5.0,
            remaining_qty=0.5,
        )
        snapshot = PerformanceReporter(self.storage).build()
        self.assertAlmostEqual(snapshot.total_realized_pnl, 5.0, places=6)

    def test_performance_reporter_computes_portfolio_heat_metrics(self):
        returns_pct = [1.0, -2.0, -1.0, 1.5, -1.0, -1.0]
        volatility_pct = PerformanceReporter._return_volatility_pct(returns_pct)
        loss_cluster_ratio_pct = PerformanceReporter._loss_cluster_ratio_pct(
            returns_pct
        )
        drawdown_velocity_pct = PerformanceReporter._drawdown_velocity_pct(
            PerformanceReporter._returns_max_drawdown_pct(returns_pct),
            len(returns_pct),
        )

        self.assertGreater(volatility_pct, 0.0)
        self.assertAlmostEqual(loss_cluster_ratio_pct, 66.6666667, places=4)
        self.assertGreater(drawdown_velocity_pct, 0.0)

    def test_performance_reporter_aggregates_pnl_ledger_metrics(self):
        from core.models import SignalDirection
        from execution.paper_trader import PaperTrader

        trader = PaperTrader(self.storage, initial_balance=1000.0)
        trader.execute_open(
            "BTC/USDT",
            SignalDirection.LONG,
            100.0,
            0.9,
            "ledger open",
            position_value=500.0,
            metadata={"model_id": "ledger-model"},
        )
        trader.execute_close("BTC/USDT", 108.0, "take_profit")
        self.storage.insert_account_snapshot(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "equity": trader.get_balance(),
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "daily_loss_pct": 0.0,
                "weekly_loss_pct": 0.0,
                "drawdown_pct": 0.0,
                "total_exposure_pct": 0.0,
                "open_positions": 0,
                "cooldown_until": None,
                "circuit_breaker_active": False,
            }
        )

        snapshot = PerformanceReporter(self.storage).build()
        report = PerformanceReporter.render(snapshot)

        self.assertEqual(snapshot.total_closed_trades, 1)
        self.assertGreater(snapshot.total_trade_cost, 0.0)
        self.assertGreaterEqual(snapshot.total_slippage_cost, 0.0)
        self.assertGreater(snapshot.recent_expectancy_pct, 0.0)
        self.assertGreater(snapshot.recent_profit_factor, 1.0)
        self.assertGreaterEqual(snapshot.avg_holding_hours, 0.0)
        self.assertIn("最近净期望收益", report)

    def test_performance_reporter_dedupes_same_symbol_timestamp_predictions(self):
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "4h",
            [
                {
                    "timestamp": int(timestamp.timestamp() * 1000),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1.0,
                },
                {
                    "timestamp": int((timestamp + timedelta(hours=4)).timestamp() * 1000),
                    "open": 101.0,
                    "high": 102.0,
                    "low": 100.0,
                    "close": 110.0,
                    "volume": 1.0,
                },
            ],
        )
        for prob in (0.6, 0.8):
            self.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": timestamp.isoformat(),
                    "model_version": "test",
                    "up_probability": prob,
                    "feature_count": 10,
                    "research": {
                        "symbol": "BTC/USDT",
                        "suggested_action": "OPEN_LONG",
                        "confidence": 0.7,
                    },
                    "decision": {
                        "final_score": 0.9,
                        "xgboost_threshold": 0.7,
                        "final_score_threshold": 0.8,
                    },
                }
            )
        snapshot = PerformanceReporter(self.storage).build()
        self.assertEqual(snapshot.prediction_eval_count, 1)

    def test_performance_reporter_keeps_distinct_pipeline_modes_for_same_timestamp(self):
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.storage.insert_ohlcv(
            "BTC/USDT:USDT",
            "4h",
            [
                {
                    "timestamp": int(timestamp.timestamp() * 1000),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1.0,
                },
                {
                    "timestamp": int((timestamp + timedelta(hours=4)).timestamp() * 1000),
                    "open": 101.0,
                    "high": 112.0,
                    "low": 100.0,
                    "close": 110.0,
                    "volume": 1.0,
                },
            ],
        )
        for pipeline_mode, prob in (("execution", 0.8), ("shadow_observation", 0.6)):
            self.storage.insert_prediction_run(
                {
                    "symbol": "BTC/USDT",
                    "timestamp": timestamp.isoformat(),
                    "model_version": pipeline_mode,
                    "up_probability": prob,
                    "feature_count": 10,
                    "research": {
                        "symbol": "BTC/USDT",
                        "suggested_action": "OPEN_LONG",
                        "confidence": 0.7,
                    },
                    "decision": {
                        "pipeline_mode": pipeline_mode,
                        "final_score": 0.9,
                        "xgboost_threshold": 0.7,
                        "final_score_threshold": 0.8,
                    },
                }
            )
        snapshot = PerformanceReporter(self.storage).build()
        self.assertEqual(snapshot.prediction_eval_count, 2)
        self.assertEqual(snapshot.research_total_count, 2)

    def test_performance_reporter_uses_runtime_thresholds_from_prediction_runs(self):
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.storage.insert_ohlcv(
            "BTC/USDT",
            "4h",
            [
                {
                    "timestamp": int(timestamp.timestamp() * 1000),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                },
                {
                    "timestamp": int((timestamp + timedelta(hours=4)).timestamp() * 1000),
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.5,
                    "close": 101.0,
                    "volume": 1100.0,
                },
            ],
        )
        self.storage.insert_prediction_run(
            {
                "symbol": "BTC/USDT",
                "timestamp": timestamp.isoformat(),
                "model_version": "runtime-test",
                "up_probability": 0.72,
                "feature_count": 12,
                "research": {
                    "suggested_action": "OPEN_LONG",
                    "confidence": 0.7,
                    "market_regime": "UPTREND",
                },
                "decision": {
                    "final_score": 0.83,
                    "xgboost_threshold": 0.70,
                    "final_score_threshold": 0.80,
                },
            }
        )

        reporter = PerformanceReporter(self.storage, self.settings)
        metrics = reporter.build()
        self.assertEqual(metrics.prediction_eval_count, 1)
        self.assertEqual(metrics.xgboost_accuracy_pct, 100.0)
        self.assertEqual(metrics.fusion_accuracy_pct, 100.0)

    def test_storage_json_state_round_trip(self):
        payload = {"xgboost_probability_threshold": 0.74, "take_profit_levels": [0.05, 0.08]}
        self.storage.set_json_state("runtime_settings_overrides", payload)
        self.assertEqual(
            self.storage.get_json_state("runtime_settings_overrides"),
            payload,
        )

    def test_storage_creates_runtime_indexes(self):
        expected = {
            "idx_prediction_runs_symbol_timestamp",
            "idx_prediction_evaluations_type_created_at",
            "idx_shadow_trade_runs_status_created_at",
            "idx_execution_events_type_created_at",
            "idx_report_artifacts_type_created_at",
            "idx_cycle_runs_status_started_at",
        }
        with self.storage._conn() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        names = {row["name"] for row in rows}
        self.assertTrue(expected.issubset(names))

    def test_dashboard_runtime_override_payload_keeps_only_differences(self):
        defaults = dashboard_module.runtime_setting_defaults()
        payload = build_runtime_override_payload(
            {
                "xgboost_probability_threshold": 0.78,
                "final_score_threshold": defaults["final_score_threshold"],
                "min_liquidity_ratio": defaults["min_liquidity_ratio"],
                "sentiment_weight": defaults["sentiment_weight"],
                "fixed_stop_loss_pct": 0.006,
                "take_profit_levels": "0.05, 0.09",
            }
        )
        self.assertEqual(
            payload,
            {
                "xgboost_probability_threshold": 0.78,
                "fixed_stop_loss_pct": 0.006,
                "take_profit_levels": [0.05, 0.09],
            },
        )
        self.assertEqual(normalize_take_profit_levels("0.08,0.05"), [0.05, 0.08])

    def test_drift_reporter_builds_summary(self):
        self.storage.insert_backtest_run(
            symbol="BTC/USDT",
            engine="v2",
            summary={
                "total_trades": 10,
                "win_rate": 60.0,
                "total_return_pct": 12.0,
                "max_drawdown_pct": 3.0,
                "profit_factor": 1.5,
                "sharpe_like": 1.0,
            },
            trades=[],
        )
        self.storage.insert_walkforward_run(
            {
                "symbol": "BTC/USDT",
                "splits": [],
                "summary": {
                    "total_splits": 3,
                    "avg_win_rate": 55.0,
                    "avg_trade_return_pct": 1.2,
                    "total_return_pct": 8.0,
                    "profit_factor": 1.3,
                    "sharpe_like": 0.8,
                },
            }
        )
        self.storage.insert_report_artifact(
            "performance",
            "\n".join(
                [
                    "# 性能报告",
                    "- Win Rate: 40.00%",
                    "- XGBoost Direction Accuracy: 48.00%",
                    "- Fusion Signal Accuracy: 45.00%",
                    "- Total Realized PnL: $5.00",
                    "- Latest Holdout Accuracy: 70.00%",
                ]
            ),
        )
        report_data = DriftReporter(self.storage).build()
        report_text = DriftReporter.render(report_data)
        self.assertIn("severity", report_data)
        self.assertIn("漂移报告", report_text)

    def test_drift_reporter_uses_runtime_return_percentage(self):
        self.storage.insert_backtest_run(
            symbol="BTC/USDT",
            engine="v2",
            summary={
                "total_trades": 1,
                "win_rate": 50.0,
                "total_return_pct": 10.0,
            },
            trades=[],
        )
        self.storage.insert_account_snapshot(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "equity": 10000.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "daily_loss_pct": 0.0,
                "weekly_loss_pct": 0.0,
                "drawdown_pct": 0.0,
                "total_exposure_pct": 0.0,
                "open_positions": 0,
                "cooldown_until": None,
                "circuit_breaker_active": False,
            }
        )
        self.storage.insert_account_snapshot(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "equity": 10500.0,
                "realized_pnl": 500.0,
                "unrealized_pnl": 0.0,
                "daily_loss_pct": 0.0,
                "weekly_loss_pct": 0.0,
                "drawdown_pct": 0.0,
                "total_exposure_pct": 0.0,
                "open_positions": 0,
                "cooldown_until": None,
                "circuit_breaker_active": False,
            }
        )
        self.storage.insert_report_artifact(
            "performance",
            "\n".join(
                [
                    "# 性能报告",
                    "- Win Rate: 40.00%",
                    "- XGBoost Direction Accuracy: 48.00%",
                    "- Fusion Signal Accuracy: 45.00%",
                    "- Total Realized PnL: $500.00",
                    "- Latest Holdout Accuracy: 70.00%",
                ]
            ),
        )
        report_data = DriftReporter(self.storage).build()
        self.assertAlmostEqual(report_data["runtime_return_pct"], 5.0, places=6)
        self.assertAlmostEqual(report_data["drift_return_gap"], 5.0, places=6)

    def test_drift_reporter_parses_chinese_performance_metrics(self):
        self.storage.insert_backtest_run(
            symbol="BTC/USDT",
            engine="v2",
            summary={"total_trades": 1, "win_rate": 60.0, "total_return_pct": 12.0},
            trades=[],
        )
        self.storage.insert_report_artifact(
            "performance",
            "\n".join(
                [
                    "# 性能报告",
                    "- 胜率: 40.00%",
                    "- XGBoost 方向准确率: 48.00%",
                    "- 最近 Holdout 准确率: 70.00%",
                ]
            ),
        )
        report_data = DriftReporter(self.storage).build()
        self.assertEqual(report_data["runtime_metrics"]["Win Rate"], "40.00%")
        self.assertEqual(report_data["runtime_metrics"]["XGBoost Direction Accuracy"], "48.00%")

    def test_ab_test_reporter_builds_summary(self):
        self.storage.insert_ab_test_run(
            {
                "symbol": "BTC/USDT",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "champion_model_version": "champion",
                "challenger_model_version": "challenger",
                "champion_probability": 0.60,
                "challenger_probability": 0.82,
                "champion_execute": False,
                "challenger_execute": True,
                "selected_variant": "challenger_shadow",
                "allocation_pct": 0.10,
                "notes": "test",
            }
        )
        report_data = ABTestReporter(self.storage).build()
        report_text = ABTestReporter.render(report_data)
        self.assertEqual(report_data["total_runs"], 1)
        self.assertIn("A/B 测试报告", report_text)
