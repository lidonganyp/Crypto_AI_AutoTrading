"""Health checks for CryptoAI v3."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from config import Settings, resolve_project_path
from core.i18n import get_default_language, normalize_language, text_for
from core.storage import Storage
from strategy.model_trainer import load_xgboost, model_path_for_symbol


@dataclass
class HealthStatus:
    status: str
    runtime_mode: str
    db_exists: bool
    xgboost_available: bool
    model_file_exists: bool
    llm_runtime_configured: bool
    last_cycle_started: str | None
    last_cycle_completed: str | None
    last_cycle_status: str | None
    stale_market_streams: int
    latest_market_latency_seconds: float | None
    market_latency_status: str
    latest_account_snapshot: str | None
    latest_prediction_run: str | None
    latest_research_mode: str
    recent_research_fallback_ratio_pct: float
    latest_market_data_provider: str
    latest_market_data_operation: str
    market_data_failover_active: bool
    market_data_failover_count: int
    market_data_primary_failures: int
    market_data_secondary_failures: int
    latest_report_artifact: str | None
    latest_reconciliation: str | None


class HealthChecker:
    """Inspect core runtime dependencies and recent artifacts."""

    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings

    def run(self) -> HealthStatus:
        db_file = resolve_project_path(self.settings.app.db_path, self.settings)
        base_model_file = resolve_project_path(self.settings.model.xgboost_model_path, self.settings)
        active_symbols = self.storage.get_json_state("active_symbols", None)
        execution_symbols = self.storage.get_json_state("execution_symbols", None)
        if isinstance(active_symbols, list):
            symbols = [str(symbol) for symbol in active_symbols]
        elif isinstance(execution_symbols, list) and execution_symbols:
            symbols = [str(symbol) for symbol in execution_symbols]
        else:
            symbols = list(self.settings.exchange.symbols)
        model_candidates = [base_model_file] + [
            model_path_for_symbol(base_model_file, symbol)
            for symbol in symbols
        ]
        latest_account = self._latest_timestamp("account_snapshots")
        latest_prediction = self._latest_timestamp("prediction_runs")
        latest_report = self._latest_timestamp("report_artifacts")
        latest_reconciliation = self._latest_timestamp("reconciliation_runs")
        last_cycle_started = self.storage.get_state("last_cycle_started")
        last_cycle_completed = self.storage.get_state("last_cycle_completed")
        last_cycle_status = self.storage.get_state("last_cycle_status")
        stale_market_streams = self._count_stale_market_streams()
        market_latency = self._latest_market_latency()
        llm_runtime_configured = bool(
            self.settings.llm.deepseek_api_key.get_secret_value()
            or self.settings.llm.qwen_api_key.get_secret_value()
        )
        research_status = self._latest_research_status()
        market_data_failover = self._market_data_failover_status()

        status = "ok"
        if not db_file.exists():
            status = "degraded"
        elif latest_prediction is None or latest_account is None:
            status = "warming_up"
        elif stale_market_streams > 0:
            status = "degraded"
        elif market_latency["status"] == "degraded":
            status = "degraded"
        elif last_cycle_status == "failed":
            status = "degraded"
        elif research_status["latest_mode"] == "fallback":
            status = "degraded"

        return HealthStatus(
            status=status,
            runtime_mode=self.settings.app.runtime_mode,
            db_exists=db_file.exists(),
            xgboost_available=load_xgboost() is not None,
            model_file_exists=any(path.exists() for path in model_candidates),
            llm_runtime_configured=llm_runtime_configured,
            last_cycle_started=last_cycle_started,
            last_cycle_completed=last_cycle_completed,
            last_cycle_status=last_cycle_status,
            stale_market_streams=stale_market_streams,
            latest_market_latency_seconds=market_latency["latency_seconds"],
            market_latency_status=market_latency["status"],
            latest_account_snapshot=latest_account,
            latest_prediction_run=latest_prediction,
            latest_research_mode=research_status["latest_mode"],
            recent_research_fallback_ratio_pct=research_status["fallback_ratio_pct"],
            latest_market_data_provider=market_data_failover["latest_provider"],
            latest_market_data_operation=market_data_failover["latest_operation"],
            market_data_failover_active=market_data_failover["fallback_active"],
            market_data_failover_count=market_data_failover["fallback_count"],
            market_data_primary_failures=market_data_failover["primary_failures"],
            market_data_secondary_failures=market_data_failover["secondary_failures"],
            latest_report_artifact=latest_report,
            latest_reconciliation=latest_reconciliation,
        )

    def render_report(self, health: HealthStatus, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language(self.settings))
        return "\n".join(
            [
                text_for(lang, "# 健康检查报告", "# Health Report"),
                text_for(lang, f"- 状态: {health.status}", f"- Status: {health.status}"),
                text_for(lang, f"- 运行模式: {health.runtime_mode}", f"- Runtime Mode: {health.runtime_mode}"),
                text_for(lang, f"- 数据库存在: {health.db_exists}", f"- DB Exists: {health.db_exists}"),
                text_for(lang, f"- XGBoost 可用: {health.xgboost_available}", f"- XGBoost Available: {health.xgboost_available}"),
                text_for(lang, f"- 模型文件存在: {health.model_file_exists}", f"- Model File Exists: {health.model_file_exists}"),
                text_for(lang, f"- LLM 运行时已配置: {health.llm_runtime_configured}", f"- LLM Runtime Configured: {health.llm_runtime_configured}"),
                text_for(lang, f"- 最近一次周期开始: {health.last_cycle_started}", f"- Last Cycle Started: {health.last_cycle_started}"),
                text_for(lang, f"- 最近一次周期完成: {health.last_cycle_completed}", f"- Last Cycle Completed: {health.last_cycle_completed}"),
                text_for(lang, f"- 最近一次周期状态: {health.last_cycle_status}", f"- Last Cycle Status: {health.last_cycle_status}"),
                text_for(lang, f"- 过期市场流数量: {health.stale_market_streams}", f"- Stale Market Streams: {health.stale_market_streams}"),
                text_for(lang, f"- 最近市场延迟秒数: {health.latest_market_latency_seconds}", f"- Latest Market Latency Seconds: {health.latest_market_latency_seconds}"),
                text_for(lang, f"- 市场延迟状态: {health.market_latency_status}", f"- Market Latency Status: {health.market_latency_status}"),
                text_for(lang, f"- 最近账户快照: {health.latest_account_snapshot}", f"- Latest Account Snapshot: {health.latest_account_snapshot}"),
                text_for(lang, f"- 最近预测记录: {health.latest_prediction_run}", f"- Latest Prediction Run: {health.latest_prediction_run}"),
                text_for(lang, f"- 最近研究模式: {health.latest_research_mode}", f"- Latest Research Mode: {health.latest_research_mode}"),
                text_for(lang, f"- 最近研究回退占比: {health.recent_research_fallback_ratio_pct:.2f}%", f"- Recent Research Fallback Ratio: {health.recent_research_fallback_ratio_pct:.2f}%"),
                text_for(lang, f"- 最近市场数据提供方: {health.latest_market_data_provider}", f"- Latest Market Data Provider: {health.latest_market_data_provider}"),
                text_for(lang, f"- 最近市场数据操作: {health.latest_market_data_operation}", f"- Latest Market Data Operation: {health.latest_market_data_operation}"),
                text_for(lang, f"- 市场数据 Failover 激活: {health.market_data_failover_active}", f"- Market Data Failover Active: {health.market_data_failover_active}"),
                text_for(lang, f"- 市场数据 Failover 次数: {health.market_data_failover_count}", f"- Market Data Failover Count: {health.market_data_failover_count}"),
                text_for(lang, f"- 主市场数据失败次数: {health.market_data_primary_failures}", f"- Primary Market Data Failures: {health.market_data_primary_failures}"),
                text_for(lang, f"- 次市场数据失败次数: {health.market_data_secondary_failures}", f"- Secondary Market Data Failures: {health.market_data_secondary_failures}"),
                text_for(lang, f"- 最近报告产物: {health.latest_report_artifact}", f"- Latest Report Artifact: {health.latest_report_artifact}"),
                text_for(lang, f"- 最近对账记录: {health.latest_reconciliation}", f"- Latest Reconciliation: {health.latest_reconciliation}"),
            ]
        )

    def _latest_timestamp(self, table: str) -> str | None:
        with self.storage._conn() as conn:
            row = conn.execute(
                f"SELECT created_at FROM {table} ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return row["created_at"] if row else None

    def _count_stale_market_streams(self) -> int:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        stale = 0
        active_symbols = self.storage.get_json_state("active_symbols", None)
        execution_symbols = self.storage.get_json_state("execution_symbols", None)
        if isinstance(active_symbols, list):
            symbols = [str(symbol) for symbol in active_symbols]
        elif isinstance(execution_symbols, list) and execution_symbols:
            symbols = [str(symbol) for symbol in execution_symbols]
        else:
            symbols = self.settings.exchange.symbols
        with self.storage._conn() as conn:
            for symbol in symbols:
                market_symbol = f"{symbol}:USDT" if ":USDT" not in symbol else symbol
                for timeframe in self.settings.exchange.timeframes:
                    row = conn.execute(
                        "SELECT MAX(timestamp) AS ts FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                        (market_symbol, timeframe),
                    ).fetchone()
                    ts = row["ts"] if row else None
                    if ts is None:
                        stale += 1
                    elif (now_ms - ts) > self._market_stream_stale_threshold_ms(timeframe):
                        stale += 1
        return stale

    def _market_stream_stale_threshold_ms(self, timeframe: str) -> int:
        candle_span_ms = {
            "1h": 3600000,
            "4h": 4 * 3600000,
            "1d": 24 * 3600000,
        }.get(timeframe, 3600000)
        analysis_interval_ms = max(
            int(self.settings.strategy.analysis_interval_seconds * 1000),
            0,
        )
        allowed_delay_ms = max(int(self.settings.exchange.data_delay_seconds * 1000), 0)
        # A candle can legitimately be as old as one full candle window plus the
        # configured analysis interval before the next cycle refreshes it.
        return candle_span_ms + max(analysis_interval_ms, allowed_delay_ms)

    def _latest_market_latency(self) -> dict:
        latest_value = self.storage.get_state("latest_market_latency_seconds")
        if latest_value is None:
            return {"latency_seconds": None, "status": "unknown"}
        latency = float(latest_value)
        if latency >= self.settings.exchange.data_latency_circuit_breaker_seconds:
            status = "degraded"
        elif latency >= self.settings.exchange.data_latency_warning_seconds:
            status = "warning"
        else:
            status = "ok"
        return {"latency_seconds": latency, "status": status}

    def _latest_research_status(self) -> dict:
        with self.storage._conn() as conn:
            rows = conn.execute(
                "SELECT research_json FROM prediction_runs ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
        if not rows:
            return {"latest_mode": "unknown", "fallback_ratio_pct": 0.0}
        modes = []
        for row in rows:
            raw = str(row["research_json"] or "")
            modes.append("fallback" if "fallback_research_model" in raw else "llm")
        fallback_count = sum(1 for mode in modes if mode == "fallback")
        return {
            "latest_mode": modes[0],
            "fallback_ratio_pct": fallback_count / len(modes) * 100,
        }

    def _market_data_failover_status(self) -> dict[str, str | bool | int]:
        route = self.storage.get_json_state("market_data_last_route", {}) or {}
        stats = self.storage.get_json_state("market_data_failover_stats", {}) or {}
        fallback_count = 0
        primary_failures = 0
        secondary_failures = 0
        if isinstance(stats, dict):
            for payload in stats.values():
                payload = payload if isinstance(payload, dict) else {}
                fallback_count += int(payload.get("fallback_used", 0) or 0)
                primary_failures += int(payload.get("primary_failures", 0) or 0)
                secondary_failures += int(payload.get("secondary_failures", 0) or 0)
        return {
            "latest_provider": str(route.get("selected_provider") or "unknown"),
            "latest_operation": str(route.get("operation") or "unknown"),
            "fallback_active": bool(route.get("fallback_used")),
            "fallback_count": fallback_count,
            "primary_failures": primary_failures,
            "secondary_failures": secondary_failures,
        }
