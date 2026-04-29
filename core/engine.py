"""CryptoAI v3 engine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

from loguru import logger

from analysis.market_regime import MarketRegimeDetector
from analysis.dynamic_watchlist import DynamicWatchlistService
from analysis.cross_validation_service import CrossValidationService
from analysis.macro_service import MacroService
from analysis.news_service import NewsService
from analysis.onchain_service import OnchainService
from analysis.research_manager import ResearchManager
from analysis.research_llm import ResearchLLMAnalyzer
from backtest.v2_engine import V2BacktestEngine
from backtest.walkforward import WalkForwardBacktester
from config import Settings, get_settings, resolve_project_path
from core.i18n import get_runtime_language
from core.binance_market_data import BinanceMarketDataCollector
from core.cycle_runtime_service import CycleRuntimeService
from core.analysis_runtime_service import AnalysisRuntimeService
from core.engine_bootstrap import (
    build_core_services,
    build_executor,
    build_market_and_live_exchange,
    build_notifier,
    build_operations_services,
    configure_engine_paths,
)
from core.engine_runtime_builders import (
    build_analysis_runtime_service,
    build_execution_pool_runtime_service,
    build_guard_runtime_service,
    build_model_lifecycle_runtime_service,
    build_position_runtime_service,
    build_preflight_runtime_service,
    build_report_runtime_service,
    build_runtime_coordination_service,
    build_runtime_settings_service,
    build_snapshot_runtime_service,
)
from core.execution_pool_runtime_service import ExecutionPoolRuntimeService
from core.feature_pipeline import FeatureInput, FeaturePipeline
from core.guard_runtime_service import GuardRuntimeService
from core.models import MarketRegime, Position, RiskCheckResult, SuggestedAction
from core.model_lifecycle_runtime_service import ModelLifecycleRuntimeService
from core.okx_market_data import OKXMarketDataCollector
from core.position_runtime_service import PositionRuntimeService
from core.preflight_runtime_service import PreflightRuntimeService
from core.report_runtime_service import ReportRuntimeService
from core.runtime_coordination_service import RuntimeCoordinationService
from core.runtime_settings_service import RuntimeSettingsService
from core.scoring import (
    objective_score_from_metrics,
    objective_score_quality,
    objective_score_sample_factor,
)
from core.shadow_runtime_service import ShadowRuntimeService
from core.snapshot_runtime_service import SnapshotRuntimeService
from core.sentiment import SentimentCollector
from core.storage import Storage
from execution.exchange_adapter import BinanceExchangeAdapter, OKXExchangeAdapter, SlippageGuard
from execution.live_trader import LiveTrader
from execution.paper_trader import PaperTrader
from execution.reconciler import Reconciler
from learning.reflector import TradeReflector
from learning.experience_store import ExperienceStore
from learning.strategy_evolver import StrategyEvolver
from monitor.health_check import HealthChecker
from monitor.incident_report import IncidentReporter
from monitor.failure_report import FailureReporter
from monitor.ab_test_report import ABTestReporter
from monitor.drift_report import DriftReporter
from monitor.guard_report import GuardReporter
from monitor.maintenance_service import MaintenanceService
from monitor.system_data_service import SystemDataService
from monitor.notifier import (
    ConsoleChannel,
    CriticalWebhookChannel,
    CriticalFeishuWebhookChannel,
    FeishuWebhookChannel,
    FileChannel,
    Notifier,
    WebhookChannel,
)
from monitor.ops_overview import OpsOverviewService
from monitor.alpha_diagnostics_report import AlphaDiagnosticsReporter
from monitor.backtest_live_consistency_report import BacktestLiveConsistencyReporter
from monitor.daily_focus_report import DailyFocusReporter
from monitor.performance_report import PerformanceReporter
from monitor.pool_attribution_report import PoolAttributionReporter
from monitor.scheduler_service import SchedulerService
from strategy.decision_engine import DecisionEngine
from strategy.model_trainer import ModelTrainer, model_path_for_symbol
from strategy.risk_manager import RiskManager
from strategy.xgboost_predictor import XGBoostPredictor
from monitor.validation_sprint import ValidationSprintService


@dataclass
class LiveReadinessStatus:
    ready: bool
    reasons: list[str]
    metrics: dict[str, float | int | str | bool]


class LiveReadinessError(RuntimeError):
    pass


class CryptoAIV2Engine:
    """Main runtime engine for the v3 architecture."""

    RUNTIME_OVERRIDE_STATE_KEY = "runtime_settings_overrides"
    RUNTIME_LOCKED_FIELDS_STATE_KEY = "runtime_settings_locked_fields"
    RUNTIME_LEARNING_OVERRIDE_STATE_KEY = "runtime_settings_learning_overrides"
    RUNTIME_OVERRIDE_CONFLICT_STATE_KEY = "runtime_settings_override_conflicts"
    RUNTIME_LEARNING_DETAILS_STATE_KEY = "runtime_settings_learning_details"
    RUNTIME_EFFECTIVE_STATE_KEY = "runtime_settings_effective"
    EXECUTION_SYMBOLS_STATE_KEY = "execution_symbols"
    SHADOW_OBSERVATION_SYMBOLS_STATE_KEY = "shadow_observation_symbols"
    EXECUTION_POOL_LAST_REBUILD_AT_STATE_KEY = "execution_pool_last_rebuild_at"
    BROKEN_MODEL_SYMBOLS_STATE_KEY = "broken_model_symbols"
    MODEL_PROMOTION_CANDIDATES_STATE_KEY = "model_promotion_candidates"
    MODEL_PROMOTION_OBSERVATION_STATE_KEY = "model_promotion_observations"
    DAILY_REPORT_DATE_STATE_KEY = "last_daily_report_date"
    LOOP_MODEL_MAINTENANCE_LAST_RUN_AT_STATE_KEY = "last_loop_model_maintenance_at"
    POSITION_REVIEW_STATE_KEY = "position_review_state"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        db_path, self.report_dir = configure_engine_paths(
            settings=self.settings,
            resolve_project_path_fn=resolve_project_path,
            logger_obj=logger,
        )
        self.storage = Storage(str(db_path))
        self.market, live_exchange = build_market_and_live_exchange(
            settings=self.settings,
            storage=self.storage,
            okx_market_collector_cls=OKXMarketDataCollector,
            binance_market_collector_cls=BinanceMarketDataCollector,
            okx_exchange_adapter_cls=OKXExchangeAdapter,
            binance_exchange_adapter_cls=BinanceExchangeAdapter,
        )
        self.__dict__.update(
            vars(
                build_core_services(
                    settings=self.settings,
                    storage=self.storage,
                    market=self.market,
                    resolve_project_path_fn=resolve_project_path,
                    sentiment_collector_cls=SentimentCollector,
                    news_service_cls=NewsService,
                    macro_service_cls=MacroService,
                    onchain_service_cls=OnchainService,
                    dynamic_watchlist_service_cls=DynamicWatchlistService,
                    cross_validation_service_cls=CrossValidationService,
                    feature_pipeline_cls=FeaturePipeline,
                    market_regime_detector_cls=MarketRegimeDetector,
                    research_llm_analyzer_cls=ResearchLLMAnalyzer,
                    research_manager_cls=ResearchManager,
                    predictor_factory_cls=XGBoostPredictor,
                    model_trainer_cls=ModelTrainer,
                    risk_manager_cls=RiskManager,
                    walkforward_backtester_cls=WalkForwardBacktester,
                    backtest_engine_cls=V2BacktestEngine,
                    decision_engine_cls=DecisionEngine,
                )
            )
        )
        self.predictor = None
        self._predictors_by_symbol: dict[str, object] = {}
        self._predictor_signatures_by_symbol: dict[str, tuple[str, int | None, int | None]] = {}
        self.challenger_predictor = None
        self._challenger_predictors_by_symbol: dict[str, object] = {}
        self._challenger_predictor_signatures_by_symbol: dict[
            str,
            tuple[str, int | None, int | None],
        ] = {}
        self._runtime_settings_overrides: dict[str, float | list[float]] = {}
        self._runtime_settings_effective: dict[
            str,
            float | list[float] | dict[str, float | list[float]] | str,
        ] = {}
        self.executor = build_executor(
            settings=self.settings,
            storage=self.storage,
            live_exchange=live_exchange,
            live_trader_cls=LiveTrader,
            paper_trader_cls=PaperTrader,
            slippage_guard_cls=SlippageGuard,
        )
        self.__dict__.update(
            vars(
                build_operations_services(
                    storage=self.storage,
                    settings=self.settings,
                    executor=self.executor,
                    build_notifier_fn=build_notifier,
                    health_checker_cls=HealthChecker,
                    performance_reporter_cls=PerformanceReporter,
                    cycle_runtime_service_cls=CycleRuntimeService,
                    shadow_runtime_service_cls=ShadowRuntimeService,
                    reconciler_cls=Reconciler,
                    trade_reflector_cls=TradeReflector,
                    strategy_evolver_cls=StrategyEvolver,
                    maintenance_service_cls=MaintenanceService,
                    system_data_service_cls=SystemDataService,
                    failure_reporter_cls=FailureReporter,
                    ab_test_reporter_cls=ABTestReporter,
                    drift_reporter_cls=DriftReporter,
                    incident_reporter_cls=IncidentReporter,
                    guard_reporter_cls=GuardReporter,
                    ops_overview_service_cls=OpsOverviewService,
                    validation_sprint_service_cls=ValidationSprintService,
                    notifier_cls=Notifier,
                    console_channel_cls=ConsoleChannel,
                    file_channel_cls=FileChannel,
                    webhook_channel_cls=WebhookChannel,
                    critical_webhook_channel_cls=CriticalWebhookChannel,
                    feishu_webhook_channel_cls=FeishuWebhookChannel,
                    critical_feishu_webhook_channel_cls=CriticalFeishuWebhookChannel,
                )
            )
        )
        self.pool_attribution = PoolAttributionReporter(self.storage)
        self.alpha_diagnostics = AlphaDiagnosticsReporter(self.storage, self.settings)
        self.daily_focus = DailyFocusReporter(self.storage, self.settings)
        self.backtest_live_consistency = BacktestLiveConsistencyReporter(
            self.storage,
            self.settings,
        )
        self.runtime_settings_runtime = build_runtime_settings_service(
            engine=self,
            runtime_settings_service_cls=RuntimeSettingsService,
        )
        self.runtime_coordination_runtime = build_runtime_coordination_service(
            engine=self,
            runtime_coordination_service_cls=RuntimeCoordinationService,
        )
        self._apply_runtime_overrides()
        self._live_readiness_status: LiveReadinessStatus | None = None
        self._reset_runtime_state()
        self.snapshot_runtime = build_snapshot_runtime_service(
            engine=self,
            snapshot_runtime_service_cls=SnapshotRuntimeService,
        )
        self.position_runtime = build_position_runtime_service(
            engine=self,
            position_runtime_service_cls=PositionRuntimeService,
        )
        self.execution_pool_runtime = build_execution_pool_runtime_service(
            engine=self,
            execution_pool_runtime_service_cls=ExecutionPoolRuntimeService,
        )
        self.analysis_runtime = build_analysis_runtime_service(
            engine=self,
            analysis_runtime_service_cls=AnalysisRuntimeService,
        )
        self.preflight_runtime = build_preflight_runtime_service(
            engine=self,
            preflight_runtime_service_cls=PreflightRuntimeService,
        )
        self.guard_runtime = build_guard_runtime_service(
            engine=self,
            guard_runtime_service_cls=GuardRuntimeService,
        )
        self.report_runtime = build_report_runtime_service(
            engine=self,
            report_runtime_service_cls=ReportRuntimeService,
        )
        self.model_lifecycle_runtime = build_model_lifecycle_runtime_service(
            engine=self,
            model_lifecycle_runtime_service_cls=ModelLifecycleRuntimeService,
        )
        if (
            self.settings.app.runtime_mode == "live"
            and self.settings.app.allow_live_orders
        ):
            self._ensure_live_readiness()
        self.scheduler = SchedulerService(self)

    def run_once(self):
        now = datetime.now(timezone.utc)
        cycle_symbols = self.preflight_runtime.prepare_cycle_symbols(now)
        active_symbols = cycle_symbols["active_symbols"]
        shadow_symbols = cycle_symbols["shadow_symbols"]
        cycle_id = self.cycle_runtime.start_cycle(
            now=now,
            active_symbols=active_symbols,
            shadow_symbols=shadow_symbols,
            circuit_breaker_active=self._circuit_breaker_active,
        )
        opened_positions = 0
        closed_positions = 0
        logger.info(f"=== CryptoAI v3 cycle started {now.isoformat()} ===")
        preflight = self.preflight_runtime.run_preflight(
            now=now,
            cycle_id=cycle_id,
            active_symbols=active_symbols,
            opened_positions=opened_positions,
            closed_positions=closed_positions,
        )
        if preflight["abort"]:
            return
        positions = preflight["positions"]
        account = preflight["account"]
        reconciliation = preflight["reconciliation"]

        closed_positions += self._manage_open_positions(now, positions, account)
        recently_closed_symbols = self._recently_closed_symbols_since(now)
        active_symbols = self.get_active_symbols(force_refresh=False, now=now)
        if recently_closed_symbols:
            active_symbols = [
                symbol for symbol in active_symbols if symbol not in recently_closed_symbols
            ]
        positions = self.storage.get_positions()
        account = self._account_state(now, positions)
        active_pass = self.analysis_runtime.run_active_symbols(
            now=now,
            active_symbols=active_symbols,
            positions=positions,
            account=account,
            model_trading_disabled=self._model_trading_disabled,
            consecutive_wins=self._consecutive_wins,
            consecutive_losses=self._consecutive_losses,
        )
        opened_positions += int(active_pass["opened_positions"])
        positions = active_pass["positions"]
        account = active_pass["account"]

        self.analysis_runtime.run_shadow_symbols(
            now=now,
            shadow_symbols=shadow_symbols,
        )

        self._evaluate_matured_predictions(now)
        self._evaluate_shadow_trades(now)
        self._generate_reports(now)
        final_cycle_status = "failed" if self._circuit_breaker_active else "ok"
        self.cycle_runtime.complete_cycle(
            cycle_id,
            final_cycle_status=final_cycle_status,
            reconciliation_status=reconciliation.status,
            notes=self._circuit_breaker_reason,
            opened_positions=opened_positions,
            closed_positions=closed_positions,
            circuit_breaker_active=self._circuit_breaker_active,
        )
        try:
            self._run_loop_model_maintenance(now)
        except Exception as exc:
            logger.exception(f"Loop model maintenance failed: {exc}")
        logger.info("=== CryptoAI v3 cycle completed ===")

    def run_position_guard(self) -> dict[str, int | str]:
        now = datetime.now(timezone.utc)
        positions = self.storage.get_positions()
        if not positions:
            return {
                "checked_positions": 0,
                "closed_positions": 0,
                "status": "no_positions",
            }

        account = self._account_state(now, positions)
        closed_positions = self._manage_open_positions(now, positions, account)
        remaining_positions = self.storage.get_positions()
        if remaining_positions != positions:
            self._account_state(now, remaining_positions)
        return {
            "checked_positions": len(positions),
            "closed_positions": int(closed_positions),
            "status": "ok",
        }

    def run_entry_scan(self) -> dict[str, int | str | list[str]]:
        now = datetime.now(timezone.utc)
        active_symbols = self.get_active_symbols(force_refresh=False, now=now)
        if not active_symbols:
            return {
                "status": "no_active_symbols",
                "active_symbols": [],
                "opened_positions": 0,
            }
        if self._check_market_latency(now, symbols=active_symbols):
            return {
                "status": "market_latency_blocked",
                "active_symbols": active_symbols,
                "opened_positions": 0,
            }
        positions = self.storage.get_positions()
        account = self._account_state(now, positions)
        self._apply_model_degradation(now)
        self._persist_runtime_settings_effective()
        account = self._account_state(now, positions)
        if self._manual_recovery_blocked():
            return {
                "status": "manual_recovery_required",
                "active_symbols": active_symbols,
                "opened_positions": 0,
            }
        if account.circuit_breaker_active:
            return {
                "status": "circuit_breaker_active",
                "active_symbols": active_symbols,
                "opened_positions": 0,
            }

        active_pass = self.analysis_runtime.run_active_symbols(
            now=now,
            active_symbols=active_symbols,
            positions=positions,
            account=account,
            model_trading_disabled=self._model_trading_disabled,
            consecutive_wins=self._consecutive_wins,
            consecutive_losses=self._consecutive_losses,
        )
        return {
            "status": "ok",
            "active_symbols": active_symbols,
            "opened_positions": int(active_pass["opened_positions"]),
        }

    def _prepare_symbol_snapshot(
        self,
        symbol: str,
        now: datetime,
        include_blocked: bool = False,
    ):
        return self.snapshot_runtime.prepare_symbol_snapshot(
            symbol,
            now,
            include_blocked=include_blocked,
        )

    def _review_research_signal(
        self,
        symbol: str,
        insight,
        prediction,
        validation,
        features,
        fear_greed: float | None,
        news_summary: str,
        onchain_summary: str,
        news_sources: list[str],
        news_coverage_score: float,
        news_service_health_score: float,
    ):
        return self.snapshot_runtime.review_research_signal(
            symbol=symbol,
            insight=insight,
            prediction=prediction,
            validation=validation,
            features=features,
            fear_greed=fear_greed,
            news_summary=news_summary,
            onchain_summary=onchain_summary,
            news_sources=news_sources,
            news_coverage_score=news_coverage_score,
            news_service_health_score=news_service_health_score,
        )

    @staticmethod
    def _compose_trade_rationale(base_reason: str, review) -> str:
        setup_text = ExperienceStore.encode_setup_profile(
            getattr(review, "setup_profile", {}) or {}
        )
        if not setup_text:
            return base_reason
        return f"{base_reason}; {setup_text}"

    def get_shadow_observation_symbols(
        self,
        force_refresh: bool = False,
        now: datetime | None = None,
    ) -> list[str]:
        now = now or datetime.now(timezone.utc)
        if force_refresh:
            self.watchlist.refresh(force=True, now=now)
        execution_symbols = set(self.get_execution_symbols())
        summary = self._symbol_edge_summary(limit=1000)
        universe = self._execution_pool_candidate_universe(
            list(execution_symbols),
            summary,
        )
        ranked = self._rank_execution_pool_candidates(
            universe,
            list(execution_symbols),
            summary,
        )
        shadow_feedback = self.shadow_runtime.build_observation_feedback(limit=500)
        prioritized_ranked = self.shadow_runtime.prioritize_observation_candidates(
            ranked,
            shadow_feedback,
            floor_pct=float(self.settings.risk.execution_symbol_accuracy_floor_pct),
        )
        target_size = max(
            0,
            min(8, max(4, int(self.settings.exchange.max_active_symbols))),
        )
        shadow_symbols: list[str] = []
        for candidate in prioritized_ranked:
            symbol = str(candidate["symbol"])
            if symbol in execution_symbols:
                continue
            if not bool(candidate.get("has_model")):
                continue
            if symbol not in shadow_symbols:
                shadow_symbols.append(symbol)
            if len(shadow_symbols) >= target_size:
                break
        self.storage.set_json_state(
            self.SHADOW_OBSERVATION_SYMBOLS_STATE_KEY,
            shadow_symbols,
        )
        return shadow_symbols

    def _shadow_observation_feedback(
        self,
        limit: int = 500,
    ) -> dict[str, dict[str, float | int]]:
        return self.shadow_runtime.build_observation_feedback(limit=limit)

    def _prioritize_shadow_observation_candidates(
        self,
        ranked_candidates: list[dict[str, float | int | str | bool]],
        shadow_feedback: dict[str, dict[str, float | int]],
    ) -> list[dict[str, float | int | str | bool]]:
        return self.shadow_runtime.prioritize_observation_candidates(
            ranked_candidates,
            shadow_feedback,
            floor_pct=float(self.settings.risk.execution_symbol_accuracy_floor_pct),
        )

    def _evaluate_matured_predictions(
        self,
        now: datetime,
        limit: int = 1000,
    ) -> dict:
        return self.shadow_runtime.evaluate_matured_predictions(now, limit=limit)

    def _record_shadow_trade_if_blocked(
        self,
        symbol: str,
        features,
        prediction,
        decision,
        validation,
        review,
        risk_result,
        entry_price: float,
    ) -> None:
        self.shadow_runtime.record_blocked_shadow_trade(
            symbol=symbol,
            features=features,
            prediction=prediction,
            decision=decision,
            validation=validation,
            review=review,
            risk_result=risk_result,
            entry_price=entry_price,
            xgboost_threshold=float(self.decision_engine.xgboost_threshold),
            final_score_threshold=float(self.decision_engine.final_score_threshold),
        )

    def _evaluate_shadow_trades(
        self,
        now: datetime,
        limit: int = 500,
    ) -> dict:
        return self.shadow_runtime.evaluate_shadow_trades(now, limit=limit)

    def _manage_open_positions(self, now: datetime, positions: list[dict], account) -> int:
        return self.position_runtime.manage_open_positions(now, positions, account)

    def _reflect_closed_trade_result(self, symbol: str, result: dict) -> None:
        trade_id = str(result.get("trade_id") or "")
        if not trade_id:
            return
        metadata = result.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        pipeline_mode = str(metadata.get("pipeline_mode") or "").strip()
        reflection_source = "paper_canary" if pipeline_mode == "paper_canary" else "trade"
        try:
            self.reflector.reflect_trade(
                trade_id=trade_id,
                symbol=symbol,
                direction="LONG",
                confidence=float(result.get("confidence") or 0.0),
                rationale=str(result.get("rationale") or ""),
                entry_time=str(result.get("entry_time") or ""),
                entry_price=float(result.get("entry_price") or 0.0),
                exit_price=float(result.get("exit_price") or 0.0),
                exit_time=datetime.now(timezone.utc).isoformat(),
                pnl=float(result.get("pnl") or 0.0),
                pnl_pct=float(result.get("pnl_pct") or 0.0),
                source=reflection_source,
            )
        except Exception as exc:
            logger.exception(f"Reflection generation failed for {symbol}: {exc}")
        if (
            self.settings.app.runtime_mode == "paper"
            and pipeline_mode == "paper_canary"
        ):
            try:
                now = datetime.now(timezone.utc)
                self.runtime_coordination_runtime.refresh_after_paper_feedback(
                    now,
                    reason="paper_canary_trade_close",
                )
            except Exception as exc:
                logger.exception(
                    f"Execution pool refresh failed after paper canary close for {symbol}: {exc}"
                )

    def _handle_trade_close_feedback(self, symbol: str, result: dict) -> None:
        metadata = result.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        if self.settings.app.runtime_mode != "paper":
            return
        if str(metadata.get("pipeline_mode") or "").strip() != "paper_canary":
            return
        try:
            now = datetime.now(timezone.utc)
            self.runtime_coordination_runtime.refresh_after_paper_feedback(
                now,
                reason="paper_canary_partial_close",
            )
        except Exception as exc:
            logger.exception(
                f"Execution pool refresh failed after paper canary partial close for {symbol}: {exc}"
            )

    def _prepare_position_review(self, symbol: str, now: datetime) -> dict | None:
        return self.snapshot_runtime.prepare_position_review(symbol, now)

    def _position_review_exit_reasons(
        self,
        position: dict,
        current_price: float,
        hours_held: float,
        review_snapshot: dict | None,
    ) -> list[str]:
        return self.position_runtime.position_review_exit_reasons(
            position,
            current_price,
            hours_held,
            review_snapshot,
        )

    @staticmethod
    def _review_exit_close_quantity(
        initial_quantity: float,
        current_quantity: float,
        exit_reasons: list[str],
    ) -> float | None:
        return PositionRuntimeService.review_exit_close_quantity(
            initial_quantity,
            current_quantity,
            exit_reasons,
        )

    @staticmethod
    def _position_protection_reasons(
        position: dict,
        current_price: float,
        initial_quantity: float,
    ) -> list[str]:
        return PositionRuntimeService.position_protection_reasons(
            position,
            current_price,
            initial_quantity,
        )

    def _persist_analysis(
        self,
        symbol: str,
        insight,
        prediction,
        final_score: float,
        decision=None,
        analysis_timestamp: datetime | None = None,
        review=None,
        validation=None,
        pipeline_mode: str = "execution",
    ):
        timestamp = analysis_timestamp or datetime.now(timezone.utc)
        portfolio_rating = getattr(decision, "portfolio_rating", "HOLD")
        position_scale = float(getattr(decision, "position_scale", 0.0) or 0.0)
        self.storage.insert_signal(
            {
                "symbol": symbol,
                "source": "xgboost",
                "direction": "LONG" if prediction.up_probability >= self.decision_engine.xgboost_threshold else "FLAT",
                "confidence": prediction.up_probability,
                "rationale": f"model={prediction.model_version}",
                "risk_level": "MEDIUM",
            }
        )
        self.storage.insert_prediction_run(
            {
                "symbol": symbol,
                "timestamp": timestamp.isoformat(),
                "model_version": prediction.model_version,
                "up_probability": prediction.up_probability,
                "feature_count": prediction.feature_count,
                "research": insight.model_dump(mode="json"),
                "decision": {
                    "model_id": str(
                        getattr(prediction, "model_id", "") or prediction.model_version
                    ),
                    "final_score": final_score,
                    "should_execute": bool(getattr(decision, "should_execute", False)),
                    "decision_reason": str(getattr(decision, "reason", "") or ""),
                    "portfolio_rating": portfolio_rating,
                    "position_scale": position_scale,
                    "regime": insight.market_regime.value,
                    "suggested_action": insight.suggested_action.value,
                    "pipeline_mode": pipeline_mode,
                    "horizon_hours": int(
                        getattr(
                            decision,
                            "horizon_hours",
                            self.settings.strategy.max_hold_hours
                            or self.settings.training.prediction_horizon_hours,
                        )
                    ),
                    "validation_reason": getattr(validation, "reason", "ok") if validation is not None else "ok",
                    "setup_profile": getattr(review, "setup_profile", {}) if review is not None else {},
                    "setup_performance": getattr(review, "setup_performance", {}) if review is not None else {},
                    "xgboost_threshold": self.decision_engine.xgboost_threshold,
                    "final_score_threshold": self.decision_engine.final_score_threshold,
                    "min_liquidity_ratio": self.decision_engine.min_liquidity_ratio,
                    "sentiment_weight": self.decision_engine.sentiment_weight,
                    "fixed_stop_loss_pct": float(
                        getattr(
                            decision,
                            "stop_loss_pct",
                            self.settings.strategy.fixed_stop_loss_pct,
                        )
                        or self.settings.strategy.fixed_stop_loss_pct
                    ),
                    "take_profit_levels": list(
                        getattr(
                            decision,
                            "take_profit_levels",
                            self.settings.strategy.take_profit_levels,
                        )
                        or self.settings.strategy.take_profit_levels
                    ),
                },
            }
        )
        self.storage.insert_signal(
            {
                "symbol": symbol,
                "source": "llm_research",
                "direction": insight.suggested_action.value.replace("OPEN_", "") if insight.suggested_action.value.startswith("OPEN_") else "FLAT",
                "confidence": insight.confidence,
                "rationale": "; ".join(insight.key_reason),
                "risk_level": "HIGH" if insight.risk_warning else "MEDIUM",
            }
        )
        self.storage.insert_signal(
            {
                "symbol": symbol,
                "source": "decision_engine",
                "direction": "LONG" if final_score >= self.decision_engine.final_score_threshold else "FLAT",
                "confidence": final_score,
                "rationale": f"portfolio_rating={portfolio_rating}; final_score={final_score:.2f}; position_scale={position_scale:.2f}",
                "risk_level": "MEDIUM",
            }
        )

    def _record_trade_result(self, pnl: float):
        if pnl > 0:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_wins = 0
            self._consecutive_losses += 1
            loss_cooldown_until = self.risk.update_cooldown_after_losses(
                self._consecutive_losses
            )
            if loss_cooldown_until is not None:
                self._cooldown_until = max(
                    self._cooldown_until or loss_cooldown_until,
                    loss_cooldown_until,
                )

    def _evaluate_ab_test(
        self,
        now: datetime,
        symbol: str,
        features,
        insight,
        risk_result,
        champion_prediction,
        champion_decision,
        account,
        positions: list[dict],
    ) -> dict | None:
        challenger_predictor = self._challenger_predictor_for_symbol(symbol)
        if challenger_predictor is None:
            return None

        challenger_prediction = challenger_predictor.predict(features)
        challenger_context, challenger_decision = self.decision_engine.evaluate_entry(
            symbol=symbol,
            prediction=challenger_prediction,
            insight=insight,
            features=features,
            risk_result=risk_result,
        )
        candidate = self._promotion_candidate_for_symbol(symbol)
        candidate_stage = str(candidate.get("status") or "")
        analysis_pipeline_mode = (
            "challenger_live" if candidate_stage == "live" else "challenger_shadow"
        )
        allocation_pct = (
            float(candidate.get("live_allocation_pct", self._promotion_live_allocation_pct()) or 0.0)
            if candidate_stage == "live"
            else float(self.settings.ab_testing.challenger_allocation_pct or 0.0)
        )
        selected_variant = "none"
        if champion_decision.should_execute and challenger_decision.should_execute:
            selected_variant = "agreement"
        elif champion_decision.should_execute:
            selected_variant = "champion"
        elif challenger_decision.should_execute:
            selected_variant = "challenger_shadow"

        can_execute = (
            (self.settings.ab_testing.execute_challenger_live or candidate_stage == "live")
            and challenger_decision.should_execute
            and not champion_decision.should_execute
            and not self._model_trading_disabled
            and not any(position["symbol"] == symbol for position in positions)
        )
        position_cap = account.equity * allocation_pct
        evidence_adjustment = self._adjust_position_value_for_model_evidence(
            symbol=symbol,
            pipeline_mode=analysis_pipeline_mode,
            base_position_value=(
                min(challenger_decision.position_value, position_cap)
                if can_execute
                else 0.0
            ),
            now=now,
            prediction=challenger_prediction,
            decision=challenger_decision,
        )
        position_value = (
            float(evidence_adjustment.get("position_value", 0.0) or 0.0)
            if can_execute
            else 0.0
        )
        applied_allocation_pct = (
            allocation_pct * float(evidence_adjustment.get("scale", 1.0) or 1.0)
            if candidate_stage == "live"
            else allocation_pct
        )
        final_selected_variant = (
            "challenger_live"
            if can_execute and position_value > 0
            else selected_variant
        )
        self.storage.insert_ab_test_run(
            {
                "symbol": symbol,
                "timestamp": now.isoformat(),
                "champion_model_version": champion_prediction.model_version,
                "challenger_model_version": challenger_prediction.model_version,
                "champion_probability": champion_prediction.up_probability,
                "challenger_probability": challenger_prediction.up_probability,
                "champion_execute": champion_decision.should_execute,
                "challenger_execute": challenger_decision.should_execute,
                "selected_variant": final_selected_variant,
                "allocation_pct": applied_allocation_pct,
                "notes": (
                    champion_decision.reason
                    if champion_decision.should_execute
                    else (
                        f"{challenger_decision.reason}; evidence_scale="
                        f"{float(evidence_adjustment.get('scale', 1.0) or 1.0):.2f}"
                    )
                ),
            }
        )
        if not can_execute:
            return {
                "execute_live": False,
                "selected_variant": final_selected_variant,
                "analysis_pipeline_mode": analysis_pipeline_mode,
                "challenger_prediction": challenger_prediction,
                "challenger_decision": challenger_decision,
                "challenger_final_score": challenger_context.final_score,
                "evidence_scale": float(evidence_adjustment.get("scale", 1.0) or 1.0),
                "evidence_source": str(evidence_adjustment.get("source", "none") or "none"),
                "evidence_reason": str(evidence_adjustment.get("reason", "") or ""),
            }

        return {
            "execute_live": position_value > 0,
            "selected_variant": final_selected_variant,
            "position_value": position_value,
            "reason": challenger_decision.reason,
            "final_score": challenger_context.final_score,
            "analysis_pipeline_mode": analysis_pipeline_mode,
            "challenger_prediction": challenger_prediction,
            "challenger_decision": challenger_decision,
            "challenger_final_score": challenger_context.final_score,
            "evidence_scale": float(evidence_adjustment.get("scale", 1.0) or 1.0),
            "evidence_source": str(evidence_adjustment.get("source", "none") or "none"),
            "evidence_reason": str(evidence_adjustment.get("reason", "") or ""),
        }

    def _account_state(self, now: datetime, positions: list[dict]):
        return self.guard_runtime.account_state(now, positions)

    def _recently_closed_symbols_since(self, since: datetime) -> set[str]:
        cutoff = since.isoformat()
        with self.storage._conn() as conn:
            rows = conn.execute(
                """SELECT DISTINCT symbol FROM execution_events
                   WHERE event_type IN ('close', 'live_close')
                     AND created_at >= ?""",
                (cutoff,),
            ).fetchall()
        return {
            str(row["symbol"] or "").strip()
            for row in rows
            if str(row["symbol"] or "").strip()
        }

    def _priced_positions(self, positions: list[dict]) -> list[dict]:
        return self.guard_runtime.priced_positions(positions)

    def _portfolio_equity(self, positions: list[dict]) -> float:
        return self.guard_runtime.portfolio_equity(positions)

    def _enforce_accuracy_guard(self, now: datetime):
        self.guard_runtime.enforce_accuracy_guard(now)

    def _apply_model_degradation(self, now: datetime):
        updates = self.guard_runtime.apply_model_degradation(now)
        self.decision_engine.xgboost_threshold = updates["recommended_xgboost_threshold"]
        self.decision_engine.final_score_threshold = updates["recommended_final_score_threshold"]
        self._model_trading_disabled = bool(updates["model_trading_disabled"])

    def _reset_runtime_settings_to_base(self):
        self.runtime_settings_runtime.reset_runtime_settings_to_base()

    def _reset_runtime_state(self):
        self._peak_equity = (
            0.0
            if self.settings.app.runtime_mode == "live"
            else self.executor.initial_balance
        )
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._consecutive_market_data_failures = 0
        self._cooldown_until = None
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""
        self._model_degradation_status = "healthy"
        self._model_degradation_reason = ""
        self._model_trading_disabled = False

    def _apply_runtime_overrides(self):
        (
            self._runtime_settings_overrides,
            self._runtime_settings_effective,
        ) = self.runtime_settings_runtime.apply_runtime_overrides()

    def _sync_runtime_components(self):
        self.runtime_settings_runtime.sync_runtime_components()

    def _persist_runtime_settings_effective(self):
        self._runtime_settings_effective = (
            self.runtime_settings_runtime.persist_runtime_settings_effective(
                self._runtime_settings_overrides
            )
        )

    def _refresh_learning_runtime_overrides(self, now: datetime):
        return self.runtime_settings_runtime.refresh_learning_runtime_overrides(now)

    def _refresh_runtime_learning_feedback(self, now: datetime, *, reason: str) -> None:
        self._refresh_learning_runtime_overrides(now)
        self._apply_runtime_overrides()
        self._persist_runtime_settings_effective()
        self.storage.insert_execution_event(
            "learning_feedback_refresh",
            "SYSTEM",
            {
                "reason": reason,
                "updated_at": now.isoformat(),
            },
        )

    def _sanitize_runtime_override_payload(
        self,
        overrides: dict,
    ) -> dict[str, float | list[float]]:
        return self.runtime_settings_runtime.sanitize_runtime_override_payload(
            overrides
        )

    @staticmethod
    def _sanitize_runtime_override_value(
        key: str,
        raw_value,
    ) -> float | list[float] | None:
        return RuntimeSettingsService.sanitize_runtime_override_value(key, raw_value)

    def _register_market_data_failure(self, reason: str):
        self._consecutive_market_data_failures += 1
        if (
            self._consecutive_market_data_failures
            < self.settings.risk.api_failure_circuit_breaker_count
        ):
            return
        self._circuit_breaker_active = True
        self._circuit_breaker_reason = "api_failure_circuit_breaker"
        self._trigger_manual_recovery(
            "api_failure_circuit_breaker",
            f"{reason} | consecutive_failures={self._consecutive_market_data_failures}",
        )
        self.notifier.notify(
            "api_failure",
            "市场数据连续失败",
            f"{reason} | consecutive_failures={self._consecutive_market_data_failures}",
            level="critical",
        )

    def _extend_cooldown_until(self, cooldown_until: datetime) -> datetime:
        self._cooldown_until = max(
            self._cooldown_until or cooldown_until,
            cooldown_until,
        )
        return self._cooldown_until

    def _detect_abnormal_move(self, symbol: str, now: datetime) -> bool:
        market_symbol = f"{symbol}:USDT" if ":USDT" not in symbol else symbol
        candles_5m = self.market.fetch_historical_ohlcv(market_symbol, "5m", limit=2)
        if len(candles_5m) < 2:
            return False
        prev_close = float(candles_5m[-2]["close"])
        last_close = float(candles_5m[-1]["close"])
        if prev_close <= 0:
            return False
        move_pct = abs(last_close / prev_close - 1.0)
        if move_pct < self.settings.risk.abnormal_move_pct_5m:
            return False
        cooldown_until = now + timedelta(
            minutes=self.settings.risk.abnormal_move_cooldown_minutes
        )
        self._cooldown_until = max(
            self._cooldown_until or cooldown_until,
            cooldown_until,
        )
        self.storage.insert_execution_event(
            "abnormal_move",
            symbol,
            {
                "move_pct": move_pct,
                "cooldown_until": self._cooldown_until.isoformat(),
            },
        )
        self.notifier.notify(
            "abnormal_move",
            "异常波动暂停开仓",
            f"{symbol} 5m move={move_pct:.2%}, cooldown_until={self._cooldown_until.isoformat()}",
            level="warning",
        )
        return True

    def _manual_recovery_blocked(self) -> bool:
        return self.guard_runtime.manual_recovery_blocked()

    def _requires_manual_recovery(self, reason: str) -> bool:
        return GuardRuntimeService.requires_manual_recovery(reason)

    def _trigger_manual_recovery(self, reason: str, details: str):
        self.guard_runtime.trigger_manual_recovery(reason, details)

    def approve_manual_recovery(self) -> dict:
        return self.guard_runtime.approve_manual_recovery()

    def get_nextgen_autonomy_live_operator_request(self) -> dict:
        from nextgen_evolution.live_runtime import AutonomyLiveRuntime

        runtime = AutonomyLiveRuntime(self.storage, settings=self.settings)
        return runtime.load_operator_request()

    def set_nextgen_autonomy_live_operator_request(
        self,
        *,
        requested_live: bool | None = None,
        whitelist: tuple[str, ...] | list[str] | None = None,
        max_active_runtimes: int | None = None,
        reason: str = "engine",
    ) -> dict:
        from nextgen_evolution.live_runtime import AutonomyLiveRuntime

        runtime = AutonomyLiveRuntime(self.storage, settings=self.settings)
        return runtime.set_operator_request(
            requested_live=requested_live,
            whitelist=whitelist,
            max_active_runtimes=max_active_runtimes,
            reason=reason,
        )

    def run_nextgen_autonomy_live(
        self,
        *,
        requested_live: bool | None = None,
        trigger: str = "scheduler",
        trigger_reason: str = "",
        trigger_details: str = "",
        whitelist: tuple[str, ...] | list[str] | None = None,
        max_active_runtimes: int | None = None,
    ) -> dict:
        from nextgen_evolution.live_cycle import AutonomyLiveCycleRunner

        runner = AutonomyLiveCycleRunner(
            self.storage,
            settings=self.settings,
        )
        try:
            result = runner.run(
                requested_live=requested_live,
                trigger=trigger,
                trigger_reason=trigger_reason,
                trigger_details=trigger_details,
                whitelist=whitelist,
                max_active_runtimes=max_active_runtimes,
            )
        except Exception as exc:
            self.storage.insert_execution_event(
                "nextgen_autonomy_live_run_failed",
                "SYSTEM",
                {
                    "requested_live": (
                        None if requested_live is None else bool(requested_live)
                    ),
                    "trigger": trigger,
                    "trigger_reason": trigger_reason,
                    "trigger_details": trigger_details,
                    "error": str(exc),
                },
            )
            raise
        self.storage.insert_execution_event(
            "nextgen_autonomy_live_run",
            "SYSTEM",
            {
                "requested_live": result.get("requested_live"),
                "trigger": trigger,
                "trigger_reason": trigger_reason,
                "trigger_details": trigger_details,
                "status": result.get("status"),
                "reason": result.get("reason"),
                "operator_request": result.get("operator_request"),
                "effective_live": result.get("effective_live"),
                "force_flatten": result.get("force_flatten"),
                "autonomy_cycle_id": result.get("autonomy_cycle_id"),
                "intent_count": result.get("intent_count"),
                "action_counts": result.get("action_counts"),
                "intent_status_counts": result.get("intent_status_counts"),
                "repair_queue_requested_size": result.get(
                    "repair_queue_requested_size"
                ),
                "repair_queue_dropped_count": result.get(
                    "repair_queue_dropped_count"
                ),
                "repair_queue_dropped_runtime_ids": result.get(
                    "repair_queue_dropped_runtime_ids"
                ),
                "repair_queue_hold_priority_count": result.get(
                    "repair_queue_hold_priority_count"
                ),
                "repair_queue_postponed_rebuild_count": result.get(
                    "repair_queue_postponed_rebuild_count"
                ),
                "repair_queue_reprioritized_count": result.get(
                    "repair_queue_reprioritized_count"
                ),
                "repair_queue_hold_priority_active": result.get(
                    "repair_queue_hold_priority_active"
                ),
                "repair_queue_postponed_rebuild_active": result.get(
                    "repair_queue_postponed_rebuild_active"
                ),
                "repair_queue_dropped_active": result.get(
                    "repair_queue_dropped_active"
                ),
                "repair_queue_reprioritized_active": result.get(
                    "repair_queue_reprioritized_active"
                ),
            },
        )
        return result

    @staticmethod
    def _asset_aliases(symbol: str) -> set[str]:
        base = symbol.split("/", 1)[0].upper()
        aliases = {base, symbol.upper()}
        alias_map = {
            "BTC": {"BITCOIN", "比特币"},
            "ETH": {"ETHEREUM", "以太坊"},
            "SOL": {"SOLANA"},
            "DOGE": {"DOGECOIN"},
            "LINK": {"CHAINLINK"},
            "FIL": {"FILECOIN"},
            "AAVE": {"AAVE"},
            "ARB": {"ARBITRUM"},
            "OP": {"OPTIMISM"},
            "WLD": {"WORLDCOIN"},
            "SUI": {"SUI"},
            "AVAX": {"AVALANCHE"},
            "UNI": {"UNISWAP"},
            "NEAR": {"NEAR"},
            "INJ": {"INJECTIVE"},
            "ATOM": {"COSMOS"},
            "TIA": {"CELESTIA"},
            "ONDO": {"ONDO"},
            "RENDER": {"RNDR"},
            "POL": {"POLYGON", "MATIC"},
        }
        aliases.update(alias_map.get(base, set()))
        return {alias.upper() for alias in aliases}

    @staticmethod
    def _text_mentions_alias(text: str, alias: str) -> bool:
        alias = alias.lower()
        if not alias:
            return False
        if any("\u4e00" <= char <= "\u9fff" for char in alias):
            return alias in text
        return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None

    def _contains_bearish_news_risk(self, symbol: str, summary: str) -> bool:
        return self.snapshot_runtime.contains_bearish_news_risk(symbol, summary)

    @staticmethod
    def _model_file_signature(model_path: Path) -> tuple[str, int | None, int | None]:
        try:
            stat = model_path.stat()
        except OSError:
            return str(model_path), None, None
        return str(model_path), int(stat.st_mtime_ns), int(stat.st_size)

    def clear_symbol_models(self, symbol: str):
        self._predictors_by_symbol.pop(symbol, None)
        self._predictor_signatures_by_symbol.pop(symbol, None)
        self._challenger_predictors_by_symbol.pop(symbol, None)
        self._challenger_predictor_signatures_by_symbol.pop(symbol, None)

    def _broken_model_symbols(self) -> dict[str, dict]:
        state = self.storage.get_json_state(
            self.BROKEN_MODEL_SYMBOLS_STATE_KEY,
            {},
        )
        return state if isinstance(state, dict) else {}

    def _set_broken_model_symbols(self, state: dict[str, dict]):
        self.storage.set_json_state(self.BROKEN_MODEL_SYMBOLS_STATE_KEY, state)

    def _clear_broken_model_symbol(self, symbol: str):
        state = self._broken_model_symbols()
        if symbol not in state:
            return
        state.pop(symbol, None)
        self._set_broken_model_symbols(state)

    def _model_promotion_candidates(self) -> dict[str, dict]:
        state = self.storage.get_json_state(
            self.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            {},
        )
        return state if isinstance(state, dict) else {}

    def _set_model_promotion_candidates(self, state: dict[str, dict]) -> None:
        self.storage.set_json_state(
            self.MODEL_PROMOTION_CANDIDATES_STATE_KEY,
            state,
        )

    def _model_promotion_observations(self) -> dict[str, dict]:
        state = self.storage.get_json_state(
            self.MODEL_PROMOTION_OBSERVATION_STATE_KEY,
            {},
        )
        return state if isinstance(state, dict) else {}

    def _set_model_promotion_observations(self, state: dict[str, dict]) -> None:
        self.storage.set_json_state(
            self.MODEL_PROMOTION_OBSERVATION_STATE_KEY,
            state,
        )

    def _promotion_candidate_for_symbol(self, symbol: str) -> dict:
        candidate = self._model_promotion_candidates().get(symbol, {})
        return candidate if isinstance(candidate, dict) else {}

    def _promotion_candidate_stage(self, symbol: str) -> str:
        return str(self._promotion_candidate_for_symbol(symbol).get("status") or "")

    def _promotion_shadow_min_evaluations(self) -> int:
        return max(4, int(self.settings.risk.execution_symbol_min_samples))

    def _promotion_live_min_evaluations(self) -> int:
        shadow_min = self._promotion_shadow_min_evaluations()
        return max(3, (shadow_min + 1) // 2)

    @staticmethod
    def _bounded_int(value: float | int, lower: int, upper: int) -> int:
        return max(lower, min(int(round(float(value))), upper))

    def _promotion_recent_volatility_pct(self, symbol: str) -> float:
        for timeframe, limit in (("4h", 72), ("1h", 120), ("1d", 40)):
            candles = list(
                reversed(self.storage.get_ohlcv(symbol, timeframe, limit=limit))
            )
            if len(candles) < 10:
                continue
            close_change_pcts: list[float] = []
            range_pcts: list[float] = []
            for previous, current in zip(candles[:-1], candles[1:]):
                previous_close = float(previous.get("close") or 0.0)
                current_close = float(current.get("close") or 0.0)
                current_high = float(current.get("high") or 0.0)
                current_low = float(current.get("low") or 0.0)
                if previous_close > 0 and current_close > 0:
                    close_change_pcts.append(
                        abs(current_close / previous_close - 1.0) * 100.0
                    )
                if current_close > 0 and current_high > 0 and current_low > 0:
                    range_pcts.append(
                        max(0.0, (current_high - current_low) / current_close * 100.0)
                    )
            if close_change_pcts or range_pcts:
                close_change_avg = (
                    sum(close_change_pcts) / len(close_change_pcts)
                    if close_change_pcts
                    else 0.0
                )
                intrabar_range_avg = (
                    sum(range_pcts) / len(range_pcts) if range_pcts else 0.0
                )
                return close_change_avg * 0.55 + intrabar_range_avg * 0.45
        return 0.0

    def _promotion_reference_holding_hours(
        self,
        symbol: str,
        training_metadata: dict | None = None,
    ) -> float:
        payload = dict(training_metadata or {})
        for key in (
            "canary_live_net_avg_holding_hours",
            "live_net_avg_holding_hours",
            "avg_holding_hours",
            "candidate_avg_holding_hours",
        ):
            value = float(payload.get(key, 0.0) or 0.0)
            if value > 0:
                return value
        rows = self.storage.get_pnl_ledger(limit=200, symbol=symbol)
        holding_hours = [
            float(row.get("holding_hours") or 0.0)
            for row in rows
            if str(row.get("event_type") or "") == "close"
            and float(row.get("holding_hours") or 0.0) > 0
        ]
        if holding_hours:
            return sum(holding_hours) / len(holding_hours)
        return 0.0

    @staticmethod
    def _promotion_reference_trade_count(training_metadata: dict | None = None) -> int:
        payload = dict(training_metadata or {})
        candidate_wf = payload.get("candidate_walkforward_summary", {}) or {}
        recent_wf = payload.get("recent_walkforward_baseline_summary", {}) or {}
        return max(
            int(candidate_wf.get("trade_count", 0) or 0),
            int(recent_wf.get("avg_trade_count", 0) or 0),
        )

    def _promotion_adaptive_requirements(
        self,
        symbol: str,
        training_metadata: dict | None = None,
    ) -> dict[str, float | int | str]:
        base_shadow = self._promotion_shadow_min_evaluations()
        base_live = self._promotion_live_min_evaluations()
        base_observation = max(4, int(self.settings.risk.execution_symbol_min_samples))
        training_metadata = dict(training_metadata or {})
        volatility_pct = self._promotion_recent_volatility_pct(symbol)
        reference_holding_hours = self._promotion_reference_holding_hours(
            symbol,
            training_metadata,
        )
        reference_trade_count = self._promotion_reference_trade_count(
            training_metadata
        )

        scale = 1.0
        if volatility_pct >= 4.0:
            scale += 0.50
        elif volatility_pct >= 2.5:
            scale += 0.30
        elif volatility_pct >= 1.5:
            scale += 0.10
        elif 0 < volatility_pct <= 0.80:
            scale -= 0.15

        if reference_holding_hours >= 36:
            scale += 0.45
        elif reference_holding_hours >= 18:
            scale += 0.25
        elif reference_holding_hours >= 8:
            scale += 0.10
        elif 0 < reference_holding_hours <= 4:
            scale -= 0.15

        if 0 < reference_trade_count < 8:
            scale += 0.15
        elif reference_trade_count >= 24:
            scale -= 0.05

        scale = max(0.70, min(scale, 2.35))
        requirements_source = (
            "adaptive"
            if volatility_pct > 0
            or reference_holding_hours > 0
            or reference_trade_count > 0
            else "default"
        )
        return {
            "shadow_min_evaluations": self._bounded_int(
                base_shadow * scale,
                4,
                max(12, base_shadow * 3),
            ),
            "live_min_evaluations": self._bounded_int(
                base_live * scale,
                3,
                max(8, base_live * 3),
            ),
            "observation_min_evaluations": self._bounded_int(
                base_observation * max(0.90, scale),
                4,
                max(12, base_observation * 3),
            ),
            "shadow_max_age_hours": self._bounded_int(72.0 * scale, 48, 168),
            "live_max_age_hours": self._bounded_int(
                72.0 * max(0.65, min(scale * 0.90, 2.25)),
                48,
                168,
            ),
            "observation_max_age_hours": self._bounded_int(
                72.0 * max(0.70, min(scale * 1.05, 2.40)),
                48,
                168,
            ),
            "volatility_pct": volatility_pct,
            "reference_holding_hours": reference_holding_hours,
            "reference_trade_count": reference_trade_count,
            "requirement_scale": scale,
            "requirements_source": requirements_source,
        }

    def _promotion_live_allocation_pct(self) -> float:
        return min(float(self.settings.ab_testing.challenger_allocation_pct or 0.0), 0.03)

    def _prediction_accuracy_metrics(
        self,
        symbol: str,
        started_at: str,
        evaluation_type: str,
        limit: int = 200,
    ) -> dict[str, float | int]:
        with self.storage._conn() as conn:
            rows = conn.execute(
                """SELECT pe.is_correct
                   FROM prediction_evaluations pe
                   JOIN prediction_runs pr
                     ON pr.symbol = pe.symbol
                    AND pr.timestamp = pe.timestamp
                  WHERE pe.symbol = ?
                    AND pe.evaluation_type = ?
                    AND pr.timestamp >= ?
                  ORDER BY pe.created_at DESC
                  LIMIT ?""",
                (symbol, evaluation_type, started_at, limit),
            ).fetchall()
        eval_count = len(rows)
        accuracy = (
            sum(int(row["is_correct"] or 0) for row in rows) / eval_count
            if eval_count
            else 0.0
        )
        return {"eval_count": eval_count, "accuracy": accuracy}

    def _ab_variant_count(
        self,
        symbol: str,
        started_at: str,
        selected_variant: str,
    ) -> int:
        with self.storage._conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS c
                   FROM ab_test_runs
                  WHERE symbol = ?
                    AND timestamp >= ?
                    AND selected_variant = ?""",
                (symbol, started_at, selected_variant),
            ).fetchone()
        return int(row["c"] if row is not None else 0)

    @staticmethod
    def _model_version_from_path(path_str: str) -> str:
        text = str(path_str or "").strip()
        return Path(text).name if text else ""

    def _resolved_model_id(self, model_id: str, model_path: str) -> str:
        text = str(model_id or "").strip()
        if text:
            return text
        path_text = str(model_path or "").strip()
        if path_text:
            try:
                metadata = json.loads(
                    Path(path_text).with_suffix(".meta.json").read_text(encoding="utf-8")
                )
            except Exception:
                metadata = {}
            meta_model_id = str((metadata or {}).get("model_id") or "").strip()
            if meta_model_id:
                return meta_model_id
        return self._model_version_from_path(model_path)

    def _candidate_active_model_id(self, candidate: dict) -> str:
        metadata = dict(candidate.get("training_metadata", {}) or {})
        return self._resolved_model_id(
            str(
                candidate.get("active_model_id")
                or metadata.get("active_model_id")
                or metadata.get("incumbent_model_id")
                or ""
            ),
            str(candidate.get("active_model_path") or metadata.get("active_model_path") or ""),
        )

    def _candidate_challenger_model_id(self, candidate: dict) -> str:
        metadata = dict(candidate.get("training_metadata", {}) or {})
        return self._resolved_model_id(
            str(
                candidate.get("challenger_model_id")
                or metadata.get("challenger_model_id")
                or metadata.get("model_id")
                or ""
            ),
            str(
                candidate.get("challenger_model_path")
                or metadata.get("challenger_model_path")
                or metadata.get("model_path")
                or ""
            ),
        )

    def _observation_active_model_id(self, observation: dict) -> str:
        metadata = dict(observation.get("training_metadata", {}) or {})
        return self._resolved_model_id(
            str(
                observation.get("active_model_id")
                or metadata.get("active_model_id")
                or metadata.get("model_id")
                or ""
            ),
            str(observation.get("active_model_path") or metadata.get("active_model_path") or ""),
        )

    def _observation_backup_model_id(self, observation: dict) -> str:
        metadata = dict(observation.get("training_metadata", {}) or {})
        return self._resolved_model_id(
            str(
                observation.get("backup_model_id")
                or metadata.get("backup_model_id")
                or metadata.get("incumbent_model_id")
                or ""
            ),
            str(observation.get("backup_model_path") or ""),
        )

    def _upsert_model_registry_entry(
        self,
        *,
        symbol: str,
        model_id: str,
        model_path: str,
        role: str,
        stage: str,
        active: bool,
        metadata: dict | None = None,
        created_at: str | None = None,
    ) -> None:
        model_id = str(model_id or "").strip()
        if not model_id:
            return
        self.storage.upsert_model_registry(
            {
                "symbol": symbol,
                "model_id": model_id,
                "model_version": self._model_version_from_path(model_path),
                "model_path": str(model_path or ""),
                "role": role,
                "stage": stage,
                "active": active,
                "metadata": dict(metadata or {}),
                "created_at": created_at,
            }
        )

    @staticmethod
    def _objective_score_from_metrics(metrics: dict[str, float | int]) -> float:
        return objective_score_from_metrics(metrics)

    @staticmethod
    def _objective_score_sample_factor(metrics: dict[str, float | int]) -> float:
        return objective_score_sample_factor(metrics)

    def _objective_score_quality(self, metrics: dict[str, float | int]) -> float:
        return objective_score_quality(metrics)

    @staticmethod
    def _accuracy_safety_floor(
        baseline_accuracy: float,
        *,
        absolute_floor: float = 0.42,
        tolerance: float = 0.18,
    ) -> float:
        return max(absolute_floor, float(baseline_accuracy or 0.0) - tolerance)

    @staticmethod
    def _read_model_metadata(model_path: Path) -> dict:
        try:
            payload = json.loads(model_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _evidence_scale_from_metrics(
        self,
        metrics: dict[str, float | int],
        *,
        source: str,
    ) -> dict[str, float | int | str]:
        payload = dict(metrics)
        sample_factor = self._objective_score_sample_factor(payload)
        if "objective_score" not in payload:
            payload["objective_score"] = self._objective_score_from_metrics(payload)
        if "objective_quality" not in payload:
            payload["objective_quality"] = self._objective_score_quality(payload)
        expectancy_pct = float(
            payload.get("expectancy_pct", payload.get("avg_trade_return_pct", 0.0)) or 0.0
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

    def _active_model_evidence_scale(self, symbol: str) -> dict[str, float | int | str]:
        return self.model_lifecycle_runtime.active_model_evidence_scale(symbol)

    def _active_model_exit_policy(self, symbol: str) -> dict[str, float | int | str | bool]:
        return self.model_lifecycle_runtime.active_model_exit_policy(symbol)

    def _candidate_live_evidence_scale(
        self,
        symbol: str,
        candidate: dict,
    ) -> dict[str, float | int | str]:
        return self.model_lifecycle_runtime.candidate_live_evidence_scale(
            symbol,
            candidate,
        )

    def _adjust_position_value_for_model_evidence(
        self,
        *,
        symbol: str,
        pipeline_mode: str,
        base_position_value: float,
        now=None,
        prediction=None,
        decision=None,
    ) -> dict[str, float | int | str]:
        position_value = max(0.0, float(base_position_value or 0.0))
        mode = str(pipeline_mode or "").strip()
        if position_value <= 0:
            return {
                "position_value": 0.0,
                "scale": 1.0,
                "source": "none",
                "reason": "non_positive_position_value",
            }
        if mode in {"execution", "fast_alpha"}:
            evidence = self._active_model_evidence_scale(symbol)
        elif mode == "challenger_live":
            evidence = self._candidate_live_evidence_scale(
                symbol,
                self._promotion_candidate_for_symbol(symbol),
            )
        else:
            evidence = {
                "scale": 1.0,
                "source": "none",
                "reason": f"pipeline_mode_{mode or 'unknown'}_not_adjusted",
            }
        scale = max(0.0, min(float(evidence.get("scale", 1.0) or 1.0), 1.0))
        return {
            **evidence,
            "position_value": position_value * scale,
            "base_position_value": position_value,
        }

    def _training_objective_baseline(self, payload: dict) -> dict[str, float]:
        candidate_wf = payload.get("candidate_walkforward_summary", {}) or {}
        total_splits = int(candidate_wf.get("total_splits", 0) or 0)
        total_return_pct = float(candidate_wf.get("total_return_pct", 0.0) or 0.0)
        holdout_accuracy = float(payload.get("candidate_holdout_accuracy", 0.0) or 0.0)
        avg_trade_return_pct = (
            float(candidate_wf.get("avg_trade_return_pct", 0.0) or 0.0)
        )
        metrics = {
            "sample_count": max(total_splits, int(payload.get("holdout_rows", 0) or 0), 1),
            "executed_count": max(total_splits, 1),
            "accuracy": holdout_accuracy,
            "executed_precision": holdout_accuracy,
            "expectancy_pct": float(
                candidate_wf.get("expectancy_pct", avg_trade_return_pct) or 0.0
            ),
            "profit_factor": float(candidate_wf.get("profit_factor", 0.0) or 0.0),
            "max_drawdown_pct": float(
                candidate_wf.get("max_drawdown_pct", 0.0) or 0.0
            ),
            "trade_win_rate": float(candidate_wf.get("avg_win_rate", 0.0) or 0.0) / 100,
            "avg_cost_pct": 0.15,
            "avg_trade_return_pct": avg_trade_return_pct,
        }
        return {
            "baseline_objective_score": self._objective_score_from_metrics(metrics),
            "baseline_objective_quality": self._objective_score_quality(metrics),
            "baseline_expectancy_pct": float(metrics["expectancy_pct"]),
            "baseline_profit_factor": float(metrics["profit_factor"]),
            "baseline_max_drawdown_pct": float(metrics["max_drawdown_pct"]),
            "baseline_trade_win_rate": float(metrics["trade_win_rate"]),
            "baseline_avg_trade_return_pct": avg_trade_return_pct,
        }

    @staticmethod
    def _returns_max_drawdown_pct(returns_pct: list[float]) -> float:
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for return_pct in returns_pct:
            equity *= 1.0 + float(return_pct) / 100.0
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        return max_drawdown * 100

    @staticmethod
    def _profit_factor(values: list[float]) -> float:
        wins = [value for value in values if value > 0]
        losses = [value for value in values if value < 0]
        if losses and abs(sum(losses)) > 1e-12:
            return sum(wins) / abs(sum(losses))
        return 5.0 if wins else 0.0

    def _scorecard_metrics_from_rows(
        self,
        *,
        symbol: str,
        model_id: str,
        model_version: str,
        rows,
    ) -> dict[str, float | int | str]:
        sample_count = len(rows)
        accuracy = (
            sum(int(row["is_correct"] or 0) for row in rows) / sample_count
            if sample_count
            else 0.0
        )
        executed_records: list[dict[str, float | str]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = {}
            predicted_up = int(row["predicted_up"] or 0) == 1
            entry_close = float(row["entry_close"] or 0.0)
            future_close = float(row["future_close"] or 0.0)
            fallback_opportunity_return_pct = (
                ((future_close / entry_close) - 1.0) * 100
                if entry_close > 0
                else 0.0
            )
            estimated_cost_pct = float(
                metadata.get(
                    "estimated_cost_pct",
                    metadata.get("cost_pct", 0.15 if predicted_up else 0.0),
                )
                or 0.0
            )
            trade_net_return_pct = float(
                metadata.get(
                    "trade_net_return_pct",
                    fallback_opportunity_return_pct - estimated_cost_pct
                    if predicted_up
                    else 0.0,
                )
                or 0.0
            )
            if not predicted_up:
                continue
            executed_records.append(
                {
                    "timestamp": str(row["timestamp"] or ""),
                    "actual_up": float(row["actual_up"] or 0.0),
                    "trade_net_return_pct": trade_net_return_pct,
                    "cost_pct": float(metadata.get("cost_pct", estimated_cost_pct) or 0.0),
                    "favorable_excursion_pct": float(
                        metadata.get("favorable_excursion_pct", 0.0) or 0.0
                    ),
                    "adverse_excursion_pct": float(
                        metadata.get("adverse_excursion_pct", 0.0) or 0.0
                    ),
                }
            )
        executed_count = len(executed_records)
        executed_precision = (
            sum(record["actual_up"] for record in executed_records) / executed_count
            if executed_count
            else 0.0
        )
        trade_returns_pct = [
            float(record["trade_net_return_pct"]) for record in executed_records
        ]
        avg_trade_return_pct = (
            sum(trade_returns_pct) / executed_count if executed_count else 0.0
        )
        total_trade_return_pct = sum(trade_returns_pct) if trade_returns_pct else 0.0
        expectancy_pct = (
            total_trade_return_pct / sample_count if sample_count else 0.0
        )
        wins = [value for value in trade_returns_pct if value > 0]
        losses = [value for value in trade_returns_pct if value <= 0]
        profit_factor = (
            sum(wins) / abs(sum(losses))
            if losses and abs(sum(losses)) > 1e-12
            else (5.0 if wins else 0.0)
        )
        trade_win_rate = len(wins) / executed_count if executed_count else 0.0
        ordered_returns_pct = [
            float(record["trade_net_return_pct"])
            for record in sorted(executed_records, key=lambda item: item["timestamp"])
        ]
        max_drawdown_pct = self._returns_max_drawdown_pct(ordered_returns_pct)
        avg_cost_pct = (
            sum(float(record["cost_pct"]) for record in executed_records) / executed_count
            if executed_count
            else 0.0
        )
        avg_favorable_excursion_pct = (
            sum(float(record["favorable_excursion_pct"]) for record in executed_records)
            / executed_count
            if executed_count
            else 0.0
        )
        avg_adverse_excursion_pct = (
            sum(float(record["adverse_excursion_pct"]) for record in executed_records)
            / executed_count
            if executed_count
            else 0.0
        )
        metrics = {
            "symbol": symbol,
            "model_id": model_id,
            "model_version": model_version,
            "sample_count": sample_count,
            "executed_count": executed_count,
            "accuracy": accuracy,
            "executed_precision": executed_precision,
            "avg_trade_return_pct": avg_trade_return_pct,
            "total_trade_return_pct": total_trade_return_pct,
            "expectancy_pct": expectancy_pct,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown_pct,
            "trade_win_rate": trade_win_rate,
            "avg_cost_pct": avg_cost_pct,
            "avg_favorable_excursion_pct": avg_favorable_excursion_pct,
            "avg_adverse_excursion_pct": avg_adverse_excursion_pct,
        }
        metrics["objective_score"] = self._objective_score_from_metrics(metrics)
        return metrics

    def _build_model_scorecard(
        self,
        *,
        symbol: str,
        model_id: str,
        evaluation_type: str,
        started_at: str,
        limit: int = 200,
    ) -> dict[str, float | int | str]:
        model_id = str(model_id or "").strip()
        if not model_id:
            return {
                "symbol": symbol,
                "model_id": "",
                "model_version": "",
                "sample_count": 0,
                "executed_count": 0,
                "accuracy": 0.0,
                "executed_precision": 0.0,
                "avg_trade_return_pct": 0.0,
                "total_trade_return_pct": 0.0,
                "expectancy_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "trade_win_rate": 0.0,
                "avg_cost_pct": 0.0,
                "avg_favorable_excursion_pct": 0.0,
                "avg_adverse_excursion_pct": 0.0,
                "objective_score": 0.0,
            }
        with self.storage._conn() as conn:
            rows = conn.execute(
                """SELECT pe.is_correct, pe.predicted_up, pe.actual_up,
                          pe.entry_close, pe.future_close,
                          pe.timestamp, pe.metadata_json,
                          pr.model_version,
                          COALESCE(json_extract(pr.decision_json, '$.model_id'), pr.model_version)
                            AS runtime_model_id
                   FROM prediction_evaluations pe
                   JOIN prediction_runs pr
                     ON pr.symbol = pe.symbol
                    AND pr.timestamp = pe.timestamp
                  WHERE pe.symbol = ?
                    AND pe.evaluation_type = ?
                    AND pr.timestamp >= ?
                    AND COALESCE(json_extract(pr.decision_json, '$.model_id'), pr.model_version) = ?
                  ORDER BY pe.created_at DESC
                  LIMIT ?""",
                (symbol, evaluation_type, started_at, model_id, limit),
            ).fetchall()
        model_version = str(rows[0]["model_version"] or "") if rows else ""
        return self._scorecard_metrics_from_rows(
            symbol=symbol,
            model_id=model_id,
            model_version=model_version,
            rows=rows,
        )

    def _build_model_live_pnl_summary(
        self,
        *,
        symbol: str,
        model_id: str,
        started_at: str,
        limit: int = 5000,
    ) -> dict[str, float | int | str]:
        model_id = str(model_id or "").strip()
        summary: dict[str, float | int | str] = {
            "symbol": symbol,
            "model_id": model_id,
            "sample_count": 0,
            "executed_count": 0,
            "ledger_event_count": 0,
            "closed_trade_count": 0,
            "avg_trade_return_pct": 0.0,
            "total_trade_return_pct": 0.0,
            "expectancy_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "trade_win_rate": 0.0,
            "avg_cost_pct": 0.0,
            "avg_holding_hours": 0.0,
            "realized_net_pnl": 0.0,
            "total_trade_cost": 0.0,
            "total_slippage_cost": 0.0,
        }
        if not model_id:
            return summary
        rows = self.storage.get_pnl_ledger(
            limit=limit,
            symbol=symbol,
            model_id=model_id,
            since=started_at,
        )
        if not rows:
            return summary

        ordered_rows = sorted(
            rows,
            key=lambda row: (
                str(row.get("event_time") or ""),
                int(row.get("id") or 0),
            ),
        )
        summary["ledger_event_count"] = len(ordered_rows)
        by_trade: dict[str, list[dict]] = {}
        for row in ordered_rows:
            trade_id = str(row.get("trade_id") or "").strip()
            if not trade_id:
                continue
            by_trade.setdefault(trade_id, []).append(row)

        closed_trade_summaries: list[dict[str, float | str]] = []
        for trade_rows in by_trade.values():
            close_rows = [
                row for row in trade_rows if str(row.get("event_type") or "") == "close"
            ]
            if not close_rows:
                continue
            open_row = next(
                (
                    row
                    for row in trade_rows
                    if str(row.get("event_type") or "") == "open"
                ),
                None,
            )
            reference_notional = float(
                (open_row or {}).get("notional_value")
                or sum(float(row.get("notional_value") or 0.0) for row in close_rows)
                or 0.0
            )
            net_pnl = sum(float(row.get("net_pnl") or 0.0) for row in trade_rows)
            fee_cost = sum(float(row.get("fee_cost") or 0.0) for row in trade_rows)
            slippage_cost = sum(
                float(row.get("slippage_cost") or 0.0) for row in trade_rows
            )
            trade_return_pct = (
                net_pnl / reference_notional * 100.0
                if reference_notional > 0
                else sum(float(row.get("net_return_pct") or 0.0) for row in trade_rows)
            )
            closed_trade_summaries.append(
                {
                    "trade_return_pct": trade_return_pct,
                    "net_pnl": net_pnl,
                    "fee_cost": fee_cost,
                    "slippage_cost": slippage_cost,
                    "cost_pct": (
                        (fee_cost + slippage_cost) / reference_notional * 100.0
                        if reference_notional > 0
                        else 0.0
                    ),
                    "holding_hours": max(
                        float(row.get("holding_hours") or 0.0) for row in close_rows
                    ),
                    "event_time": str(close_rows[-1].get("event_time") or ""),
                }
            )

        closed_trade_count = len(closed_trade_summaries)
        summary["closed_trade_count"] = closed_trade_count
        summary["sample_count"] = closed_trade_count
        summary["executed_count"] = closed_trade_count
        if not closed_trade_summaries:
            return summary

        returns_pct = [
            float(item["trade_return_pct"] or 0.0)
            for item in sorted(
                closed_trade_summaries,
                key=lambda item: str(item["event_time"] or ""),
            )
        ]
        wins = [value for value in returns_pct if value > 0]
        summary["avg_trade_return_pct"] = sum(returns_pct) / closed_trade_count
        summary["total_trade_return_pct"] = sum(returns_pct)
        summary["expectancy_pct"] = summary["avg_trade_return_pct"]
        summary["profit_factor"] = self._profit_factor(returns_pct)
        summary["max_drawdown_pct"] = self._returns_max_drawdown_pct(returns_pct)
        summary["trade_win_rate"] = len(wins) / closed_trade_count
        summary["avg_cost_pct"] = sum(
            float(item["cost_pct"] or 0.0) for item in closed_trade_summaries
        ) / closed_trade_count
        summary["avg_holding_hours"] = sum(
            float(item["holding_hours"] or 0.0) for item in closed_trade_summaries
        ) / closed_trade_count
        summary["realized_net_pnl"] = sum(
            float(item["net_pnl"] or 0.0) for item in closed_trade_summaries
        )
        summary["total_trade_cost"] = sum(
            float(item["fee_cost"] or 0.0) for item in closed_trade_summaries
        )
        summary["total_slippage_cost"] = sum(
            float(item["slippage_cost"] or 0.0) for item in closed_trade_summaries
        )
        return summary

    def _record_model_scorecard(
        self,
        *,
        symbol: str,
        model_id: str,
        model_path: str,
        stage: str,
        evaluation_type: str,
        started_at: str,
        extra_metadata: dict | None = None,
    ) -> dict[str, float | int | str]:
        scorecard = self._build_model_scorecard(
            symbol=symbol,
            model_id=model_id,
            evaluation_type=evaluation_type,
            started_at=started_at,
        )
        if not scorecard.get("model_id"):
            return scorecard
        metadata = dict(extra_metadata or {})
        metadata.update(
            {
                "started_at": started_at,
                "model_path": model_path,
            }
        )
        self.storage.insert_model_scorecard(
            {
                "symbol": symbol,
                "model_id": scorecard["model_id"],
                "model_version": scorecard.get("model_version", "") or self._model_version_from_path(model_path),
                "stage": stage,
                "evaluation_type": evaluation_type,
                "sample_count": scorecard.get("sample_count", 0),
                "executed_count": scorecard.get("executed_count", 0),
                "accuracy": scorecard.get("accuracy", 0.0),
                "executed_precision": scorecard.get("executed_precision", 0.0),
                "avg_trade_return_pct": scorecard.get("avg_trade_return_pct", 0.0),
                "total_trade_return_pct": scorecard.get("total_trade_return_pct", 0.0),
                "expectancy_pct": scorecard.get("expectancy_pct", 0.0),
                "profit_factor": scorecard.get("profit_factor", 0.0),
                "max_drawdown_pct": scorecard.get("max_drawdown_pct", 0.0),
                "trade_win_rate": scorecard.get("trade_win_rate", 0.0),
                "avg_cost_pct": scorecard.get("avg_cost_pct", 0.0),
                "avg_favorable_excursion_pct": scorecard.get("avg_favorable_excursion_pct", 0.0),
                "avg_adverse_excursion_pct": scorecard.get("avg_adverse_excursion_pct", 0.0),
                "objective_score": scorecard.get("objective_score", 0.0),
                "metadata": metadata,
            }
        )
        return scorecard

    def _register_promotion_candidate(
        self,
        symbol: str,
        summary,
        now: datetime,
    ) -> None:
        self.model_lifecycle_runtime.register_promotion_candidate(
            symbol,
            summary,
            now,
        )

    def _sync_model_registry_from_training_summary(
        self,
        symbol: str,
        summary,
        now: datetime,
    ) -> None:
        self.model_lifecycle_runtime.sync_model_registry_from_training_summary(
            symbol,
            summary,
            now,
        )

    def _advance_promotion_candidate_to_live(
        self,
        symbol: str,
        candidate: dict,
        now: datetime,
        shadow_metrics: dict[str, float | int],
        champion_metrics: dict[str, float | int],
    ) -> dict:
        return self.model_lifecycle_runtime.advance_promotion_candidate_to_live(
            symbol,
            candidate,
            now,
            shadow_metrics,
            champion_metrics,
        )

    def _reject_promotion_candidate(
        self,
        symbol: str,
        candidate: dict,
        now: datetime,
        reason: str,
        metrics: dict[str, float | int],
    ) -> None:
        self.model_lifecycle_runtime.reject_promotion_candidate(
            symbol,
            candidate,
            now,
            reason,
            metrics,
        )

    def _promote_candidate_to_active(
        self,
        symbol: str,
        candidate: dict,
        now: datetime,
        metrics: dict[str, float | int],
    ) -> None:
        self.model_lifecycle_runtime.promote_candidate_to_active(
            symbol,
            candidate,
            now,
            metrics,
        )

    def _observe_promotion_candidates(self, now: datetime) -> dict[str, int]:
        return self.model_lifecycle_runtime.observe_promotion_candidates(now)

    def _register_promoted_model_observation(
        self,
        symbol: str,
        summary,
        now: datetime,
    ) -> None:
        self.model_lifecycle_runtime.register_promoted_model_observation(
            symbol,
            summary,
            now,
        )

    def _model_self_heal_interval_hours(self) -> int:
        return self.model_lifecycle_runtime.model_self_heal_interval_hours()

    def _runtime_model_path_for_symbol(self, symbol: str) -> Path:
        return self.model_lifecycle_runtime.runtime_model_path_for_symbol(symbol)

    def _post_promotion_execution_metrics(
        self,
        symbol: str,
        promoted_at: str,
        limit: int = 200,
    ) -> dict[str, float | int]:
        return self.model_lifecycle_runtime.post_promotion_execution_metrics(
            symbol,
            promoted_at,
            limit=limit,
        )

    @staticmethod
    def _remove_file_if_exists(path_str: str) -> None:
        ModelLifecycleRuntimeService.remove_file_if_exists(path_str)

    @staticmethod
    def _write_json_file_atomic(path: Path, payload: dict) -> None:
        ModelLifecycleRuntimeService.write_json_file_atomic(path, payload)

    def _accept_promoted_model(
        self,
        symbol: str,
        observation: dict,
        now: datetime,
        metrics: dict[str, float | int],
    ) -> None:
        self.model_lifecycle_runtime.accept_promoted_model(
            symbol,
            observation,
            now,
            metrics,
        )

    def _rollback_promoted_model(
        self,
        symbol: str,
        observation: dict,
        now: datetime,
        reason: str,
        metrics: dict[str, float | int],
    ) -> None:
        self.model_lifecycle_runtime.rollback_promoted_model(
            symbol,
            observation,
            now,
            reason,
            metrics,
        )

    def _observe_promoted_models(self, now: datetime) -> dict[str, int]:
        return self.model_lifecycle_runtime.observe_promoted_models(now)

    @staticmethod
    def _failure_kind_retrainable(failure_kind: str) -> bool:
        return failure_kind in {
            "missing_model_file",
            "empty_model_file",
            "model_load_failed",
        }

    def _handle_model_unavailable(
        self,
        *,
        symbol: str,
        predictor,
        prediction,
        now: datetime,
    ) -> dict:
        model_path = Path(getattr(predictor, "model_path", "") or "")
        signature = list(self._model_file_signature(model_path)) if str(model_path) else ["", None, None]
        failure_kind = str(getattr(predictor, "load_failure_kind", "") or "fallback_prediction")
        load_error = str(getattr(predictor, "load_error", "") or "")
        broken_state = self._broken_model_symbols()
        previous = broken_state.get(symbol) if isinstance(broken_state.get(symbol), dict) else {}
        last_attempt_at = self._parse_iso_datetime(previous.get("last_repair_attempt_at"))
        broken_state[symbol] = {
            "symbol": symbol,
            "failure_kind": failure_kind,
            "load_error": load_error,
            "model_path": str(model_path),
            "signature": signature,
            "model_version": str(getattr(prediction, "model_version", "") or ""),
            "first_detected_at": previous.get("first_detected_at") or now.isoformat(),
            "last_detected_at": now.isoformat(),
            "last_repair_attempt_at": previous.get("last_repair_attempt_at"),
            "last_repair_result": previous.get("last_repair_result", ""),
        }
        self._set_broken_model_symbols(broken_state)

        self.storage.insert_execution_event(
            "model_unavailable",
            symbol,
            {
                "failure_kind": failure_kind,
                "model_path": str(model_path),
                "signature": signature,
                "model_version": getattr(prediction, "model_version", ""),
                "load_error": load_error,
            },
        )

        if not self._failure_kind_retrainable(failure_kind):
            return {"status": "recorded", "retrained": False, "reason": failure_kind}

        if (
            last_attempt_at is not None
            and now - last_attempt_at < timedelta(hours=self._model_self_heal_interval_hours())
        ):
            return {"status": "cooldown", "retrained": False, "reason": failure_kind}

        return {
            "status": "queued",
            "retrained": False,
            "reason": failure_kind,
        }

    def _cached_symbol_predictor(
        self,
        *,
        symbol: str,
        cache: dict[str, object],
        signature_cache: dict[str, tuple[str, int | None, int | None]],
        model_path: Path,
    ):
        signature = self._model_file_signature(model_path)
        predictor = cache.get(symbol)
        if predictor is None or signature_cache.get(symbol) != signature:
            predictor = self._predictor_factory(
                str(model_path),
                enable_fallback=self.settings.model.enable_fallback_predictor,
            )
            cache[symbol] = predictor
            signature_cache[symbol] = signature
        return predictor

    def _predictor_for_symbol(self, symbol: str):
        if self.predictor is not None:
            return self.predictor
        return self._cached_symbol_predictor(
            symbol=symbol,
            cache=self._predictors_by_symbol,
            signature_cache=self._predictor_signatures_by_symbol,
            model_path=self._runtime_model_path_for_symbol(symbol),
        )

    def _challenger_predictor_for_symbol(self, symbol: str):
        if (
            not self.settings.ab_testing.enabled
            and not self._promotion_candidate_for_symbol(symbol)
        ):
            return None
        if self.challenger_predictor is not None:
            return self.challenger_predictor
        return self._cached_symbol_predictor(
            symbol=symbol,
            cache=self._challenger_predictors_by_symbol,
            signature_cache=self._challenger_predictor_signatures_by_symbol,
            model_path=model_path_for_symbol(self._challenger_predictor_base_path, symbol),
        )

    def _validate_market_data_quality(
        self,
        symbol: str,
        candles_by_timeframe: dict[str, list[dict]],
    ) -> dict:
        return self.snapshot_runtime.validate_market_data_quality(
            symbol,
            candles_by_timeframe,
        )

    @staticmethod
    def _storage_symbol_variants(symbol: str) -> list[str]:
        variants = [str(symbol)]
        if ":USDT" in symbol:
            variants.append(symbol.replace(":USDT", ""))
        else:
            variants.append(f"{symbol}:USDT")
        return list(dict.fromkeys(variants))

    def _stored_close_at_or_before(self, symbol: str, moment: datetime) -> float | None:
        return self.guard_runtime.stored_close_at_or_before(symbol, moment)

    def _period_baseline_price(
        self,
        position: dict,
        period_start: datetime,
        current_price: float,
    ) -> float:
        return self.guard_runtime.period_baseline_price(
            position,
            period_start,
            current_price,
        )

    def _check_market_latency(
        self,
        now: datetime,
        symbols: list[str] | None = None,
    ) -> bool:
        return self.guard_runtime.check_market_latency(now, symbols)

    def _generate_reports(self, now: datetime):
        report_date = now.date().isoformat()
        if self.storage.get_state(self.DAILY_REPORT_DATE_STATE_KEY) == report_date:
            return
        positions = self.storage.get_positions()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        closed_trades = self.storage.get_closed_trades(since=today_start.isoformat())
        equity = self._portfolio_equity(positions)
        self.notifier.notify_daily_report(positions, closed_trades, equity)
        self.storage.set_state(self.DAILY_REPORT_DATE_STATE_KEY, report_date)

    def train_models(self) -> list[dict]:
        return self._train_models_if_due(
            datetime.now(timezone.utc),
            force=True,
            reason="manual",
        )

    def _process_training_summary(
        self,
        *,
        symbol: str,
        summary,
        now: datetime,
        payload: dict | None = None,
    ) -> None:
        if getattr(summary, "trained_with_xgboost", False):
            self._sync_model_registry_from_training_summary(symbol, summary, now)
        if (
            getattr(summary, "promoted_to_active", False)
            and getattr(summary, "candidate_walkforward_summary", None)
        ):
            walkforward_result = {
                "symbol": symbol,
                "summary": dict(summary.candidate_walkforward_summary),
                "splits": list(
                    getattr(summary, "candidate_walkforward_splits", []) or []
                ),
            }
            self.storage.insert_walkforward_run(walkforward_result)
            walkforward_report = self.report_runtime.record_rendered_report(
                report_type="walkforward",
                symbol=symbol,
                renderer=self.walkforward.render_report,
                payload=walkforward_result,
            )
            self._register_promoted_model_observation(symbol, summary, now)
        elif getattr(summary, "promotion_status", "") == "canary_pending":
            self._register_promotion_candidate(symbol, summary, now)

    def _train_models_if_due(
        self,
        now: datetime,
        force: bool,
        reason: str,
        record_skips: bool = True,
    ) -> list[dict]:
        summaries = []
        skipped = []
        lang = self._current_language()
        symbols_to_consider = list(
            dict.fromkeys(
                self.get_execution_symbols()
                + list(self._broken_model_symbols().keys())
            )
        )
        for symbol in symbols_to_consider:
            should_train, skip_reason = self._should_retrain_symbol(
                symbol,
                now,
                force=force,
            )
            if not should_train:
                skipped.append({"symbol": symbol, "reason": skip_reason})
                continue
            summary = self.trainer.train_symbol(symbol)
            self.clear_symbol_models(symbol)
            payload = summary.__dict__.copy()
            payload["report"] = self.trainer.render_report(summary, lang=lang)
            self.storage.insert_training_run(payload)
            self.report_runtime.save_report_artifact(
                report_type="training",
                symbol=symbol,
                content=payload["report"],
                extension="md",
            )
            self._process_training_summary(
                symbol=symbol,
                summary=summary,
                now=now,
                payload=payload,
            )
            broken_state = self._broken_model_symbols()
            broken_entry = (
                broken_state.get(symbol)
                if isinstance(broken_state.get(symbol), dict)
                else {}
            )
            active_model_path = str(
                getattr(summary, "active_model_path", "")
                or payload.get("active_model_path")
                or summary.model_path
            )
            if broken_entry:
                broken_entry.update(
                    {
                        "last_repair_attempt_at": now.isoformat(),
                        "last_repair_result": summary.reason,
                        "model_path": active_model_path,
                        "signature": list(
                            self._model_file_signature(Path(active_model_path))
                        ),
                    }
                )
                if summary.trained_with_xgboost:
                    broken_state.pop(symbol, None)
                else:
                    broken_state[symbol] = broken_entry
                self._set_broken_model_symbols(broken_state)
                self.storage.insert_execution_event(
                    "model_self_heal",
                    symbol,
                    {
                        "trained_with_xgboost": summary.trained_with_xgboost,
                        "reason": summary.reason,
                        "rows": summary.rows,
                        "model_path": summary.model_path,
                    },
                )
            if summary.trained_with_xgboost:
                self._clear_broken_model_symbol(symbol)
            summaries.append(payload)
        if skipped and record_skips:
            self.storage.insert_execution_event(
                "model_retrain_skip",
                "SYSTEM",
                {
                    "reason": reason,
                    "force": force,
                    "skipped": skipped,
                },
            )
        return summaries

    def _loop_model_maintenance_interval_hours(self) -> int:
        scheduled = int(getattr(self.settings.scheduler, "training_cron_hours", 0) or 0)
        return max(1, min(24, scheduled if scheduled > 0 else 24))

    def _run_loop_model_maintenance(self, now: datetime) -> list[dict]:
        broken_symbols = self._broken_model_symbols()
        if not broken_symbols:
            last_run_at = self._parse_iso_datetime(
                self.storage.get_state(self.LOOP_MODEL_MAINTENANCE_LAST_RUN_AT_STATE_KEY)
            )
            if (
                last_run_at is not None
                and now - last_run_at
                < timedelta(hours=self._loop_model_maintenance_interval_hours())
            ):
                return []
        summaries = self._train_models_if_due(
            now,
            force=False,
            reason="loop_auto",
            record_skips=False,
        )
        self.storage.set_state(
            self.LOOP_MODEL_MAINTENANCE_LAST_RUN_AT_STATE_KEY,
            now.isoformat(),
        )
        return summaries

    def _current_training_data_signature(self, symbol: str) -> dict[str, int]:
        signature_getter = getattr(self.trainer, "training_data_signature", None)
        if callable(signature_getter):
            signature = signature_getter(symbol) or {}
            if isinstance(signature, dict):
                return {
                    "rows": int(signature.get("rows") or 0),
                    "start_timestamp": int(signature.get("start_timestamp") or 0),
                    "end_timestamp": int(signature.get("end_timestamp") or 0),
                }

        count_training_rows = getattr(self.trainer, "count_training_rows", None)
        if callable(count_training_rows):
            return {
                "rows": int(count_training_rows(symbol)),
                "start_timestamp": 0,
                "end_timestamp": 0,
            }

        dataset = self.trainer.build_dataset(symbol)
        timestamps = [int(ts) for ts in dataset.get("timestamps", []) if ts is not None]
        return {
            "rows": len(dataset.get("labels", [])),
            "start_timestamp": timestamps[0] if timestamps else 0,
            "end_timestamp": timestamps[-1] if timestamps else 0,
        }

    def _should_retrain_symbol(
        self,
        symbol: str,
        now: datetime,
        force: bool = False,
    ) -> tuple[bool, str]:
        if force:
            return True, "forced"

        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT metadata_json, created_at, model_path, trained_with_xgboost FROM training_runs "
                "WHERE symbol = ? ORDER BY created_at DESC LIMIT 1",
                (symbol,),
            ).fetchone()

        if row is None:
            return True, "missing_training_run"

        try:
            metadata = json.loads(row["metadata_json"])
        except Exception:
            metadata = {}

        created_at_raw = row["created_at"]
        try:
            created_at = datetime.fromisoformat(created_at_raw)
        except Exception:
            return True, "invalid_training_timestamp"

        broken_state = self._broken_model_symbols()
        broken_entry = (
            broken_state.get(symbol)
            if isinstance(broken_state.get(symbol), dict)
            else None
        )
        if broken_entry is not None:
            runtime_model_path = self._runtime_model_path_for_symbol(symbol)
            current_signature = list(self._model_file_signature(runtime_model_path))
            if (
                broken_entry.get("model_path") != str(runtime_model_path)
                or broken_entry.get("signature") != current_signature
            ):
                self._clear_broken_model_symbol(symbol)
            else:
                failure_kind = str(broken_entry.get("failure_kind") or "")
                if self._failure_kind_retrainable(failure_kind):
                    last_attempt_at = self._parse_iso_datetime(
                        broken_entry.get("last_repair_attempt_at")
                    )
                    if (
                        last_attempt_at is not None
                        and now - last_attempt_at
                        < timedelta(hours=self._model_self_heal_interval_hours())
                    ):
                        return False, "broken_model_repair_cooldown"
                    return True, f"broken_model:{failure_kind}"
                return False, f"broken_model_non_retrainable:{failure_kind or 'unknown'}"

        current_signature = self._current_training_data_signature(symbol)
        current_rows = int(current_signature.get("rows") or 0)
        previous_rows = int(metadata.get("rows") or 0)
        if current_rows > previous_rows:
            return True, f"new_rows:{previous_rows}->{current_rows}"
        current_start_timestamp = int(current_signature.get("start_timestamp") or 0)
        current_end_timestamp = int(current_signature.get("end_timestamp") or 0)
        previous_start_timestamp = int(metadata.get("dataset_start_timestamp_ms") or 0)
        previous_end_timestamp = int(metadata.get("dataset_end_timestamp_ms") or 0)
        if (
            current_rows >= self.settings.training.minimum_training_rows
            and current_end_timestamp
            and not previous_end_timestamp
        ):
            return True, "training_window_signature_missing"
        if (
            current_rows >= self.settings.training.minimum_training_rows
            and (
                current_end_timestamp > previous_end_timestamp
                or (
                    current_end_timestamp == previous_end_timestamp
                    and current_start_timestamp > previous_start_timestamp
                )
            )
        ):
            return True, (
                "new_training_window:"
                f"{previous_start_timestamp}-{previous_end_timestamp}"
                f"->{current_start_timestamp}-{current_end_timestamp}"
            )

        trained_with_xgboost = bool(
            metadata.get("trained_with_xgboost", row["trained_with_xgboost"])
        )
        model_path_raw = str(
            metadata.get("active_model_path")
            or metadata.get("model_path")
            or row["model_path"]
            or ""
        ).strip()
        if trained_with_xgboost:
            if not model_path_raw:
                return True, "missing_model_artifact"
            model_path = Path(model_path_raw)
            if not model_path.exists():
                return True, "missing_model_artifact"
            try:
                if model_path.stat().st_size <= 0:
                    return True, "invalid_model_artifact"
            except OSError:
                return True, "missing_model_artifact"

        if (
            not trained_with_xgboost
            and current_rows >= self.settings.training.minimum_training_rows
        ):
            return True, "previous_training_incomplete"

        if (
            now - created_at
            >= timedelta(days=self.settings.training.retrain_interval_days)
        ):
            return True, "retrain_interval_elapsed"

        return False, "no_new_training_data"

    def generate_reports(self) -> dict[str, str]:
        return self.report_runtime.generate_reports()

    def run_walkforward(self, symbol: str) -> dict:
        result = self.walkforward.run(symbol)
        self.storage.insert_walkforward_run(result)
        result["report"] = self.report_runtime.record_rendered_report(
            report_type="walkforward",
            symbol=symbol,
            renderer=self.walkforward.render_report,
            payload=result,
        )
        return result

    def run_backfill(self, days: int = 180) -> dict:
        return self._backfill_symbols(self.get_execution_symbols(), days=days)

    def run_reconciliation(self) -> dict:
        result = self.reconciler.run()
        return {
            "status": result.status,
            "mismatch_count": result.mismatch_count,
            "details": result.details,
        }

    def run_backtest(self, symbol: str) -> dict:
        result = self.backtester.run(symbol)
        self.storage.insert_backtest_run(
            symbol=symbol,
            engine="v2",
            summary=result["summary"],
            trades=result["trades"],
        )
        result["report"] = self.report_runtime.record_rendered_report(
            report_type="backtest",
            symbol=symbol,
            renderer=self.backtester.render_report,
            payload=result,
        )
        return result

    def run_health_check(self) -> dict:
        return self.report_runtime.run_health_check()

    def run_metrics(self) -> dict:
        return self.report_runtime.run_metrics()

    def evaluate_live_readiness(self) -> LiveReadinessStatus:
        readiness = self.guard_runtime.build_live_readiness()
        return LiveReadinessStatus(
            ready=readiness["ready"],
            reasons=readiness["reasons"],
            metrics=readiness["metrics"],
        )

    def render_live_readiness_report(
        self,
        readiness: LiveReadinessStatus | None = None,
    ) -> str:
        readiness = readiness or self.evaluate_live_readiness()
        return self.guard_runtime.render_live_readiness_report(
            {
                "ready": readiness.ready,
                "reasons": readiness.reasons,
                "metrics": readiness.metrics,
            }
        )

    def run_live_readiness_check(self) -> dict:
        return self.report_runtime.run_live_readiness_check()

    def _ensure_live_readiness(self) -> None:
        readiness = self.evaluate_live_readiness()
        self._live_readiness_status = readiness
        self.storage.set_state(
            "live_readiness_status",
            "ready" if readiness.ready else "blocked",
        )
        self.storage.set_state(
            "live_readiness_reason",
            ",".join(readiness.reasons) if readiness.reasons else "",
        )
        if readiness.ready:
            return
        self.storage.insert_execution_event(
            "live_readiness_block",
            "SYSTEM",
            {
                "reasons": readiness.reasons,
                "metrics": readiness.metrics,
            },
        )
        raise LiveReadinessError(self.render_live_readiness_report(readiness))

    def run_guard_report(self) -> dict:
        return self.report_runtime.run_guard_report()

    def run_ab_test_report(self) -> dict:
        return self.report_runtime.run_ab_test_report()

    def run_drift_report(self) -> dict:
        return self.report_runtime.run_drift_report()

    def run_maintenance(self) -> dict:
        return self.report_runtime.run_maintenance()

    def initialize_system_data(self) -> dict:
        result = self.system_data.initialize_system()
        self._reset_runtime_settings_to_base()
        self._runtime_settings_overrides = {}
        self._runtime_settings_effective = {}
        self._sync_runtime_components()
        self._reset_runtime_state()
        report = "\n".join(
            [
                "# System Initialization",
                f"- reset_tables: {result['reset_tables']}",
                f"- artifact_files_removed: {result['artifact_files_removed']}",
                f"- artifact_files_skipped: {result['artifact_files_skipped']}",
                f"- pycache_dirs_removed: {result['pycache_dirs_removed']}",
                f"- models_preserved: {result['models_preserved']}",
            ]
        )
        return {"summary": result, "report": report}

    def cleanup_runtime_data(self) -> dict:
        result = self.system_data.cleanup_data()
        report = "\n".join(
            [
                "# Data Cleanup",
                f"- artifact_files_removed: {result['artifact_files_removed']}",
                f"- artifact_files_skipped: {result['artifact_files_skipped']}",
                f"- pycache_dirs_removed: {result['pycache_dirs_removed']}",
                f"- models_preserved: {result['models_preserved']}",
            ]
        )
        return {"summary": result, "report": report}

    def run_failure_report(self) -> dict:
        return self.report_runtime.run_failure_report()

    def run_incident_report(self) -> dict:
        return self.report_runtime.run_incident_report()

    def run_ops_overview(self) -> dict:
        return self.report_runtime.run_ops_overview()

    def run_pool_attribution_report(self, symbols: list[str] | None = None) -> dict:
        selected_symbols = symbols or self.get_execution_symbols()
        result = self.pool_attribution.build(selected_symbols)
        report = self.report_runtime.record_rendered_report(
            report_type="pool_attribution",
            symbol=None,
            renderer=self.pool_attribution.render,
            payload=result,
        )
        return {**result, "report": report}

    def run_alpha_diagnostics_report(self, symbols: list[str] | None = None) -> dict:
        selected_symbols = symbols or self.get_execution_symbols()
        result = self.alpha_diagnostics.build(selected_symbols)
        report = self.report_runtime.record_rendered_report(
            report_type="alpha_diagnostics",
            symbol=None,
            renderer=self.alpha_diagnostics.render,
            payload=result,
        )
        return {**result, "report": report}

    def run_validation_sprint(self, symbols: list[str] | None = None) -> dict:
        selected_symbols = symbols or self.get_execution_symbols()
        result = self.validation.run(selected_symbols)
        report = self.report_runtime.record_rendered_report(
            report_type="validation_sprint",
            symbol=None,
            renderer=self.validation.render,
            payload=result,
        )
        return {**result, "report": report}

    def run_daily_focus_report(self, symbols: list[str] | None = None) -> dict:
        return self.report_runtime.run_daily_focus_report(symbols=symbols)

    def run_backtest_live_consistency_report(
        self,
        symbols: list[str] | None = None,
    ) -> dict:
        return self.report_runtime.run_backtest_live_consistency_report(
            symbols=symbols
        )

    def _current_language(self) -> str:
        return get_runtime_language(self.storage, self.settings)

    @staticmethod
    def _parse_iso_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _normalize_execution_symbols(self, symbols: list[str]) -> list[str]:
        return self.execution_pool_runtime.normalize_execution_symbols(symbols)

    def _model_ready_symbols(self, symbols: list[str]) -> list[str]:
        return self.execution_pool_runtime.model_ready_symbols(symbols)

    def _filter_active_symbols_by_model_readiness(
        self,
        symbols: list[str],
    ) -> list[str]:
        return self.execution_pool_runtime.filter_active_symbols_by_model_readiness(symbols)

    def _filter_symbols_by_recent_edge(self, symbols: list[str]) -> list[str]:
        return self.execution_pool_runtime.filter_symbols_by_recent_edge(symbols)

    def _symbol_accuracy_summary(self, limit: int = 500) -> dict[str, dict[str, float | int]]:
        return self.execution_pool_runtime.symbol_accuracy_summary(limit=limit)

    def _symbol_edge_summary(self, limit: int = 500) -> dict[str, dict[str, float | int]]:
        return self.execution_pool_runtime.symbol_edge_summary(limit=limit)

    def _execution_pool_target_size(self) -> int:
        return self.execution_pool_runtime.execution_pool_target_size()

    def _execution_pool_candidate_universe(
        self,
        current_symbols: list[str],
        summary: dict[str, dict[str, float | int]],
    ) -> list[str]:
        return self.execution_pool_runtime.execution_pool_candidate_universe(
            current_symbols,
            summary,
        )

    def _rank_execution_pool_candidates(
        self,
        symbols: list[str],
        current_symbols: list[str],
        summary: dict[str, dict[str, float | int]],
    ) -> list[dict[str, float | int | str | bool]]:
        return self.execution_pool_runtime.rank_execution_pool_candidates(
            symbols,
            current_symbols,
            summary,
        )

    @staticmethod
    def _select_rebuilt_execution_symbols(
        ranked_candidates: list[dict[str, float | int | str | bool]],
        target_size: int,
    ) -> list[str]:
        return ExecutionPoolRuntimeService.select_rebuilt_execution_symbols(
            ranked_candidates,
            target_size,
        )

    def rebuild_execution_symbols(
        self,
        force: bool = False,
        now: datetime | None = None,
        reason: str = "manual",
    ) -> dict:
        return self.execution_pool_runtime.rebuild_execution_symbols(
            force=force,
            now=now,
            reason=reason,
        )

    def _maybe_rebuild_execution_pool(
        self,
        now: datetime,
        active_symbols: list[str],
    ) -> dict | None:
        return self.execution_pool_runtime.maybe_rebuild_execution_pool(
            now,
            active_symbols,
        )

    def get_execution_symbols(self) -> list[str]:
        return self.execution_pool_runtime.get_execution_symbols()

    def _backfill_symbols(self, symbols: list[str], days: int = 180) -> dict[str, dict[str, int]]:
        return self.execution_pool_runtime.backfill_symbols(symbols, days=days)

    def set_execution_symbols(
        self,
        symbols: list[str],
        backfill_days: int = 180,
        train_models: bool = True,
        action: str = "set",
    ) -> dict:
        return self.execution_pool_runtime.set_execution_symbols(
            symbols,
            backfill_days=backfill_days,
            train_models=train_models,
            action=action,
        )

    def add_execution_symbols(
        self,
        symbols: list[str],
        backfill_days: int = 180,
    ) -> dict:
        return self.execution_pool_runtime.add_execution_symbols(
            symbols,
            backfill_days=backfill_days,
        )

    def remove_execution_symbols(self, symbols: list[str]) -> dict:
        return self.execution_pool_runtime.remove_execution_symbols(symbols)

    def get_active_symbols(
        self,
        force_refresh: bool = False,
        now: datetime | None = None,
    ) -> list[str]:
        return self.execution_pool_runtime.get_active_symbols(
            force_refresh=force_refresh,
            now=now,
        )

    def get_watchlist_snapshot(
        self,
        force_refresh: bool = False,
        now: datetime | None = None,
    ) -> dict:
        return self.execution_pool_runtime.get_watchlist_snapshot(
            force_refresh=force_refresh,
            now=now,
        )

    def run_watchlist_refresh(self) -> dict:
        return self.execution_pool_runtime.run_watchlist_refresh()


def build_engine(settings: Settings | None = None) -> CryptoAIV2Engine:
    return CryptoAIV2Engine(settings=settings)
