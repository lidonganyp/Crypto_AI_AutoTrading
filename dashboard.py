"""CryptoAI v3 dashboard."""
from __future__ import annotations

from contextlib import nullcontext
import inspect
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from config import get_settings
from core.dashboard_data_service import (
    DashboardRuntimeStateView,
    execute_sql as dashboard_execute_sql,
    get_state_json as dashboard_get_state_json,
    get_state_row as dashboard_get_state_row,
    load_json as dashboard_load_json,
    parse_report_history as dashboard_parse_report_history,
    query_df as dashboard_query_df,
    query_one as dashboard_query_one,
    resolve_dashboard_db_path,
    set_state_json as dashboard_set_state_json,
)
from core.dashboard_page_context import DashboardPageContext
from core.dashboard_page_renderers import (
    render_backtest_page,
    render_overview_page,
    render_reports_page,
    render_settings_page,
    render_ops_page,
    render_predictions_page,
    render_training_page,
    render_walkforward_page,
    render_watchlist_page,
)
from core.i18n import LANGUAGE_STATE_KEY, get_default_language, normalize_language
from core.report_metrics import (
    METRIC_KEY_ALIASES,
    canonical_metric_key,
    parse_markdown_metrics as parse_report_metrics,
)


st.set_page_config(
    page_title="CryptoAI v3",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

RUNTIME_OVERRIDE_STATE_KEY = "runtime_settings_overrides"
RUNTIME_LOCKED_FIELDS_STATE_KEY = "runtime_settings_locked_fields"
RUNTIME_LEARNING_OVERRIDE_STATE_KEY = "runtime_settings_learning_overrides"
RUNTIME_OVERRIDE_CONFLICT_STATE_KEY = "runtime_settings_override_conflicts"
RUNTIME_EFFECTIVE_STATE_KEY = "runtime_settings_effective"
WATCHLIST_WHITELIST_KEY = "watchlist_whitelist"
WATCHLIST_BLACKLIST_KEY = "watchlist_blacklist"
WATCHLIST_SNAPSHOT_KEY = "active_watchlist_snapshot"
EXECUTION_SYMBOLS_KEY = "execution_symbols"

ZH_UI = {
    "Language": "\u8bed\u8a00",
    "Chinese": "\u4e2d\u6587",
    "English": "\u82f1\u6587",
    "Pages": "\u9875\u9762",
    "Overview": "\u603b\u89c8",
    "Settings": "\u8bbe\u7f6e",
    "Predictions": "\u9884\u6d4b",
    "Watchlist": "\u4ea4\u6613\u6c60",
    "Ops": "\u8fd0\u7ef4",
    "Training": "\u8bad\u7ec3",
    "Backtest": "\u56de\u6d4b",
    "Walk-Forward": "\u6eda\u52a8\u9a8c\u8bc1",
    "Reports": "\u62a5\u544a",
    "Execution": "\u6267\u884c",
    "Health": "\u5065\u5eb7\u68c0\u67e5",
    "Guards": "\u98ce\u63a7",
    "AB Test": "A/B \u6d4b\u8bd5",
    "Drift": "\u504f\u5dee",
    "Metrics": "\u6307\u6807",
    "Failures": "\u6545\u969c",
    "Scheduler": "\u8c03\u5ea6",
    "Logs": "\u65e5\u5fd7",
    "Operations": "\u64cd\u4f5c",
    "CryptoAI v3 Overview": "CryptoAI v3 \u603b\u89c8",
    "Runtime Settings": "\u8fd0\u884c\u65f6\u8bbe\u7f6e",
    "Ops Overview": "\u8fd0\u7ef4\u603b\u89c8",
    "Predictions & Features": "\u9884\u6d4b\u4e0e\u7279\u5f81",
    "Dynamic Watchlist": "\u52a8\u6001\u4ea4\u6613\u6c60",
    "Training": "\u8bad\u7ec3",
    "Walk-Forward Evaluation": "\u6eda\u52a8\u9a8c\u8bc1",
    "V3 Backtest": "V3 \u56de\u6d4b",
    "Reports": "\u62a5\u544a",
    "Execution Audit": "\u6267\u884c\u5ba1\u8ba1",
    "Health": "\u5065\u5eb7\u68c0\u67e5",
    "Performance Metrics": "\u6027\u80fd\u6307\u6807",
    "Drift Report": "\u504f\u5dee\u62a5\u544a",
    "Guardrails": "\u98ce\u63a7",
    "Failures & Maintenance": "\u6545\u969c\u4e0e\u7ef4\u62a4",
    "Log Viewer": "\u65e5\u5fd7\u67e5\u770b",
    "Operations": "\u64cd\u4f5c",
    "Open Positions": "\u5f53\u524d\u6301\u4ed3",
    "Equity": "\u6743\u76ca",
    "Drawdown": "\u56de\u64a4",
    "Circuit Breaker": "\u7194\u65ad\u72b6\u6001",
    "Reconciliation": "\u5bf9\u8d26",
    "XGB Accuracy": "XGB \u51c6\u786e\u7387",
    "Ops Report": "\u8fd0\u7ef4\u62a5\u544a",
    "Guard Report": "\u98ce\u63a7\u62a5\u544a",
    "ACTIVE": "\u5df2\u89e6\u53d1",
    "OFF": "\u5173\u95ed",
    "READY": "\u5df2\u751f\u6210",
    "MISSING": "\u7f3a\u5931",
    "N/A": "\u6682\u65e0",
    "(no output)": "\uff08\u65e0\u8f93\u51fa\uff09",
    "yes": "\u662f",
    "no": "\u5426",
    "true": "\u662f",
    "false": "\u5426",
    "default": "\u9ed8\u8ba4",
    "Accuracy Trend": "\u51c6\u786e\u7387\u8d8b\u52bf",
    "Source": "\u6765\u6e90",
    "Macro": "\u5b8f\u89c2",
    "fallback": "\u56de\u9000",
    "SYSTEM": "\u7cfb\u7edf",
    "ALL": "\u5168\u90e8",
    "ok": "\u6b63\u5e38",
    "running": "\u8fd0\u884c\u4e2d",
    "success": "\u6210\u529f",
    "failed": "\u5931\u8d25",
    "error": "\u9519\u8bef",
    "warning": "\u8b66\u544a",
    "execution": "\u6267\u884c",
    "shadow_observation": "\u5f71\u5b50\u89c2\u5bdf",
    "challenger_shadow": "\u6311\u6218\u8005\u5f71\u5b50",
    "challenger_live": "\u6311\u6218\u8005\u5b9e\u76d8",
    "paper_canary_open": "\u7eb8\u9762 Canary \u5f00\u4ed3",
    "research_review": "\u7814\u7a76\u590d\u6838",
    "prediction_evaluation": "\u9884\u6d4b\u8bc4\u4f30",
    "shadow_trade_evaluation": "\u5f71\u5b50\u4ea4\u6613\u8bc4\u4f30",
    "execution_pool_update": "\u6267\u884c\u6c60\u66f4\u65b0",
    "execution_pool_rebuild": "\u6267\u884c\u6c60\u91cd\u5efa",
    "learning_runtime_override": "\u5b66\u4e60\u8fd0\u884c\u65f6\u8986\u76d6",
    "model_degradation": "\u6a21\u578b\u8870\u51cf",
    "model_retrain_skip": "\u8df3\u8fc7\u6a21\u578b\u91cd\u8bad",
    "edge_filter": "\u8fb9\u9645\u8fc7\u6ee4",
    "model_observation_started": "\u6a21\u578b\u89c2\u5bdf\u5f00\u59cb",
    "model_promotion_candidate_started": "\u6a21\u578b\u664b\u7ea7\u5019\u9009\u5f00\u59cb",
    "model_self_heal": "\u6a21\u578b\u81ea\u6108",
    "model_unavailable": "\u6a21\u578b\u4e0d\u53ef\u7528",
    "cross_validation": "\u4ea4\u53c9\u9a8c\u8bc1",
    "training": "\u8bad\u7ec3",
    "ops_overview": "\u8fd0\u7ef4\u603b\u89c8",
    "performance": "\u6027\u80fd",
    "health": "\u5065\u5eb7\u68c0\u67e5",
    "walkforward": "\u6eda\u52a8\u9a8c\u8bc1",
    "failure": "\u6545\u969c",
    "guard": "\u98ce\u63a7",
    "drift": "\u504f\u5dee",
    "ab_test": "A/B \u6d4b\u8bd5",
    "incident": "\u4e8b\u6545",
    "alpha_diagnostics": "Alpha \u8bca\u65ad",
    "pool_attribution": "\u6c60\u5f52\u56e0",
    "warming_up": "\u9884\u70ed\u4e2d",
    "positive_edge": "\u6b63\u8fb9\u9645",
    "neutral": "\u4e2d\u6027",
    "disabled_negative_edge": "\u653e\u884c\u540e\u8d1f\u8fb9\u9645\u5df2\u7981\u7528",
    "degraded": "\u5df2\u964d\u7ea7",
    "healthy": "\u5065\u5eb7",
    "insufficient_samples": "\u6837\u672c\u4e0d\u8db3",
    "realized_edge_negative": "\u5df2\u5b9e\u73b0\u8fb9\u9645\u4e3a\u8d1f",
    "realized_edge_override": "\u5df2\u5b9e\u73b0\u8fb9\u9645\u8986\u76d6",
    "paper_mode_accuracy_guard": "\u7eb8\u9762\u6a21\u5f0f\u51c6\u786e\u7387\u4fdd\u62a4",
    "paper_mode_accuracy_guard_exploration_grace": "\u7eb8\u9762\u6a21\u5f0f\u51c6\u786e\u7387\u4fdd\u62a4\uff08\u63a2\u7d22\u7f13\u51b2\uff09",
    "live_accuracy_below_disable_floor": "\u5b9e\u76d8\u51c6\u786e\u7387\u4f4e\u4e8e\u505c\u7528\u9608\u503c",
    "live_accuracy_below_degrade_floor": "\u5b9e\u76d8\u51c6\u786e\u7387\u4f4e\u4e8e\u964d\u7ea7\u9608\u503c",
    "accuracy_gap_vs_holdout": "\u4e0e Holdout \u51c6\u786e\u7387\u504f\u5dee\u8fc7\u5927",
    "HOLD": "\u89c2\u671b",
    "OPEN_LONG": "\u5f00\u591a",
    "CLOSE": "\u5e73\u4ed3",
    "UNKNOWN": "\u672a\u77e5",
    "UPTREND": "\u4e0a\u5347\u8d8b\u52bf",
    "DOWNTREND": "\u4e0b\u964d\u8d8b\u52bf",
    "RANGE": "\u9707\u8361",
    "EXTREME_FEAR": "\u6781\u7aef\u6050\u60e7",
    "These overrides apply on the next engine cycle without editing .env. Model degradation logic may still tighten thresholds at runtime.": "\u8fd9\u4e9b\u8986\u76d6\u53c2\u6570\u4f1a\u5728\u4e0b\u4e00\u4e2a\u5f15\u64ce\u5468\u671f\u751f\u6548\uff0c\u65e0\u9700\u4fee\u6539 .env\u3002\u6a21\u578b\u8870\u51cf\u903b\u8f91\u4ecd\u53ef\u80fd\u5728\u8fd0\u884c\u65f6\u8fdb\u4e00\u6b65\u6536\u7d27\u9608\u503c\u3002",
    "Override Status": "\u8986\u76d6\u72b6\u6001",
    "updated_at": "\u66f4\u65b0\u65f6\u95f4",
    "XGBoost Probability Threshold": "XGBoost \u6982\u7387\u9608\u503c",
    "Min Liquidity Ratio": "\u6700\u4f4e\u6d41\u52a8\u6027\u6bd4\u4f8b",
    "Fixed Stop Loss Pct": "\u56fa\u5b9a\u6b62\u635f\u6bd4\u4f8b",
    "Final Score Threshold": "\u6700\u7ec8\u5f97\u5206\u9608\u503c",
    "Sentiment Weight": "\u60c5\u7eea\u6743\u91cd",
    "Take Profit Levels": "\u6b62\u76c8\u6863\u4f4d",
    "Comma-separated values, for example: 0.05, 0.08": "\u7528\u9017\u53f7\u5206\u9694\uff0c\u4f8b\u5982\uff1a0.05, 0.08",
    "Save Runtime Overrides": "\u4fdd\u5b58\u8fd0\u884c\u65f6\u8986\u76d6",
    "Reset To Defaults": "\u6062\u590d\u9ed8\u8ba4",
    "Runtime overrides saved.": "\u8fd0\u884c\u65f6\u8986\u76d6\u5df2\u4fdd\u5b58\u3002",
    "Runtime overrides reset to defaults.": "\u8fd0\u884c\u65f6\u8986\u76d6\u5df2\u6062\u590d\u9ed8\u8ba4\u503c\u3002",
    "Base Defaults": "\u57fa\u7840\u9ed8\u8ba4\u503c",
    "Saved Overrides": "\u5df2\u4fdd\u5b58\u8986\u76d6",
    "Current Effective Runtime": "\u5f53\u524d\u751f\u6548\u914d\u7f6e",
    "Recent Closed Trades": "\u6700\u8fd1\u5e73\u4ed3\u4ea4\u6613",
    "Exit Reason Drag By Symbol": "\u6309\u6807\u7684\u51fa\u573a\u539f\u56e0\u78e8\u635f",
    "Exit Reason Drag By Setup": "\u6309 setup \u7684\u51fa\u573a\u539f\u56e0\u78e8\u635f",
    "No focused exit-reason samples.": "\u6682\u65e0\u91cd\u70b9\u51fa\u573a\u539f\u56e0\u6837\u672c\u3002",
    "No focused setup-level exit samples.": "\u6682\u65e0\u91cd\u70b9 setup \u5c42\u7ea7\u7684\u51fa\u573a\u6837\u672c\u3002",
    "No open positions.": "\u5f53\u524d\u6ca1\u6709\u6301\u4ed3\u3002",
    "No closed trades.": "\u6700\u8fd1\u6ca1\u6709\u5e73\u4ed3\u4ea4\u6613\u3002",
    "Current Active Universe": "\u5f53\u524d\u4ea4\u6613\u6c60",
    "No active watchlist is available.": "\u5f53\u524d\u6ca1\u6709\u53ef\u7528\u4ea4\u6613\u6c60\u3002",
    "Recent Prediction Runs": "\u6700\u8fd1\u9884\u6d4b\u8bb0\u5f55",
    "No prediction runs recorded.": "\u6682\u65e0\u9884\u6d4b\u8bb0\u5f55\u3002",
    "Recent Feature Snapshots": "\u6700\u65b0\u7279\u5f81\u5feb\u7167",
    "Recent Research Inputs": "\u6700\u8fd1\u7814\u7a76\u8f93\u5165",
    "Current Active Universe": "\u5f53\u524d\u4ea4\u6613\u6c60",
    "Added Symbols": "\u672c\u8f6e\u65b0\u589e\u5e01\u79cd",
    "Removed Symbols": "\u672c\u8f6e\u79fb\u9664\u5e01\u79cd",
    "Manual Overrides": "\u624b\u52a8\u8986\u76d6",
    "Learning Overrides": "\u5b66\u4e60\u8986\u76d6",
    "Blocked Learning Overrides": "\u88ab\u624b\u52a8\u62e6\u622a\u7684\u5b66\u4e60\u8986\u76d6",
    "Locked Fields": "\u9501\u5b9a\u5b57\u6bb5",
    "Override Sources": "\u8986\u76d6\u6765\u6e90",
    "Automation Summary": "\u81ea\u52a8\u5316\u6458\u8981",
    "Active Automation Rules": "\u751f\u6548\u4e2d\u7684\u81ea\u52a8\u89c4\u5219",
    "Advanced Manual Controls": "\u9ad8\u7ea7\u624b\u52a8\u63a7\u5236",
    "Whitelist": "\u767d\u540d\u5355",
    "Blacklist": "\u9ed1\u540d\u5355",
    "Save Lists": "\u4fdd\u5b58\u540d\u5355",
    "Refresh Watchlist Now": "\u7acb\u5373\u5237\u65b0\u4ea4\u6613\u6c60",
    "Candidate Scores": "\u5019\u9009\u6c60\u8bc4\u5206",
    "No candidate scoring available. Refresh the watchlist first.": "\u6682\u65e0\u5019\u9009\u6c60\u8bc4\u5206\uff0c\u8bf7\u5148\u5237\u65b0\u4ea4\u6613\u6c60\u3002",
    "Generate Reports": "\u751f\u6210\u65e5\u62a5/\u5468\u62a5",
    "Generate Ops Overview": "\u751f\u6210\u8fd0\u7ef4\u603b\u89c8",
    "Run Reconciliation": "\u6267\u884c\u5bf9\u8d26",
    "Run Health Check": "\u6267\u884c\u5065\u5eb7\u68c0\u67e5",
    "Generate Metrics": "\u751f\u6210\u6307\u6807",
    "Generate Guard Report": "\u751f\u6210\u98ce\u63a7\u62a5\u544a",
    "Approve Recovery": "\u6279\u51c6\u6062\u590d",
    "Run Job": "\u8fd0\u884c\u4efb\u52a1",
    "Refresh": "\u5237\u65b0",
    "Log File": "\u65e5\u5fd7\u6587\u4ef6",
    "Tail Lines": "\u663e\u793a\u884c\u6570",
    "LLM-enhanced systematic crypto trading": "LLM \u589e\u5f3a\u7684\u7cfb\u7edf\u5316\u52a0\u5bc6\u4ea4\u6613",
    "Last Cycle": "\u6700\u8fd1\u5468\u671f",
    "Last Scheduler Job": "\u6700\u8fd1\u8c03\u5ea6\u4efb\u52a1",
    "Latest Reports": "\u6700\u8fd1\u62a5\u544a",
    "Failure report available.": "\u5df2\u751f\u6210\u6545\u969c\u62a5\u544a\u3002",
    "Model Degradation": "\u6a21\u578b\u8870\u51cf",
    "Last Accuracy Guard": "\u6700\u8fd1\u51c6\u786e\u7387\u4fdd\u62a4",
    "Latest Model Event": "\u6700\u8fd1\u6a21\u578b\u4e8b\u4ef6",
    "Model Lifecycle Summary": "\u6a21\u578b\u751f\u547d\u5468\u671f\u6458\u8981",
    "Active Model": "\u5f53\u524d Active \u6a21\u578b",
    "Challenger Model": "\u5f53\u524d Challenger \u6a21\u578b",
    "Observation State": "\u89c2\u5bdf\u72b6\u6001",
    "Latest Rollback": "\u6700\u65b0\u56de\u6eda",
    "Model Lifecycle": "\u6a21\u578b\u751f\u547d\u5468\u671f",
    "No model lifecycle data.": "\u6682\u65e0\u6a21\u578b\u751f\u547d\u5468\u671f\u6570\u636e\u3002",
    "Current Promotion Observations": "\u5f53\u524d\u664b\u7ea7\u89c2\u5bdf\u4efb\u52a1",
    "No promoted models under observation.": "\u6682\u65e0\u5904\u4e8e\u89c2\u5bdf\u671f\u7684\u6a21\u578b\u3002",
    "Recent Model Events": "\u6700\u8fd1\u6a21\u578b\u4e8b\u4ef6",
    "No model lifecycle events recorded.": "\u6682\u65e0\u6a21\u578b\u751f\u547d\u5468\u671f\u4e8b\u4ef6\u3002",
    "Latest Model Rollback Report": "\u6700\u65b0\u6a21\u578b\u56de\u6eda\u62a5\u544a",
    "No rollback reports.": "\u6682\u65e0\u6a21\u578b\u56de\u6eda\u62a5\u544a\u3002",
    "Model Registry": "\u6a21\u578b\u6ce8\u518c\u8868",
    "No model registry records.": "\u6682\u65e0\u6a21\u578b\u6ce8\u518c\u8bb0\u5f55\u3002",
    "Recent Model Scorecards": "\u6700\u8fd1\u6a21\u578b\u8bc4\u5206\u5361",
    "No model scorecards.": "\u6682\u65e0\u6a21\u578b\u8bc4\u5206\u5361\u3002",
    "Ops overview generated.": "\u8fd0\u7ef4\u603b\u89c8\u5df2\u751f\u6210\u3002",
    "Ops overview failed.": "\u8fd0\u7ef4\u603b\u89c8\u751f\u6210\u5931\u8d25\u3002",
    "Reconciliation finished.": "\u5bf9\u8d26\u5df2\u5b8c\u6210\u3002",
    "Reconciliation failed.": "\u5bf9\u8d26\u5931\u8d25\u3002",
    "No ops overview reports.": "\u6682\u65e0\u8fd0\u7ef4\u603b\u89c8\u62a5\u544a\u3002",
    "Nextgen Live Queue": "Nextgen Live 队列",
    "No nextgen live queue history.": "暂无 Nextgen Live 队列历史。",
    "Queue Filter": "队列过滤",
    "No nextgen live queue rows match filter.": "没有匹配过滤条件的 Nextgen Live 队列项。",
    "Nextgen Autonomy Cycle Details": "Nextgen 自治周期明细",
    "Inspect Nextgen Autonomy Cycle": "查看 Nextgen 自治周期",
    "No nextgen autonomy cycle details.": "暂无 Nextgen 自治周期明细。",
    "Cycle ID": "周期 ID",
    "Cycle Notes": "周期 Notes",
    "Execution Directives": "执行指令",
    "No execution directives for this cycle.": "该周期暂无执行指令。",
    "Repair Plans": "修复计划",
    "No repair plans for this cycle.": "该周期暂无修复计划。",
    "Selected Queue Trigger": "选中队列触发源",
    "Selected Queue Reason": "选中队列原因",
    "Selected Queue Latest Issue": "选中队列最近问题",
    "Selected Queue Latest Issue Reason": "选中队列问题原因",
    "Cycle Runtime Filter": "周期运行时过滤",
    "Queue Focus": "队列焦点",
    "Queue Focus Runtimes": "队列焦点运行时",
    "Queue Focus Actions": "队列焦点动作",
    "Queue Focus Priorities": "队列焦点优先级",
    "Focused": "焦点",
    "Full": "完整",
    "Focused View Mode": "焦点视图模式",
    "All Focus Rows": "全部焦点项",
    "Execution Only": "仅执行指令",
    "Repair Only": "仅修复计划",
    "Focused Execution Directives": "焦点执行指令",
    "No focused execution directives for this cycle.": "该周期暂无焦点执行指令。",
    "Focused Repair Plans": "焦点修复计划",
    "No focused repair plans for this cycle.": "该周期暂无焦点修复计划。",
    "No queue-focused rows for this cycle.": "该周期暂无队列焦点项。",
    "Additional Focused Execution Hidden": "未展示的焦点执行条数",
    "Additional Focused Repair Hidden": "未展示的焦点修复条数",
    "Recent Scheduler Runs": "\u6700\u8fd1\u8c03\u5ea6\u8bb0\u5f55",
    "No scheduler runs.": "\u6682\u65e0\u8c03\u5ea6\u8bb0\u5f55\u3002",
    "Recent Reconciliation Runs": "\u6700\u8fd1\u5bf9\u8d26\u8bb0\u5f55",
    "No reconciliation runs.": "\u6682\u65e0\u5bf9\u8d26\u8bb0\u5f55\u3002",
    "idle": "\u7a7a\u95f2",
    "observing": "\u89c2\u5bdf\u4e2d",
    "shadow": "\u5f71\u5b50\u89c2\u5bdf",
    "live": "\u5c0f\u6d41\u91cf\u5b9e\u76d8",
    "Inspect Reconciliation Run": "\u67e5\u770b\u5bf9\u8d26\u8bb0\u5f55",
    "Status": "\u72b6\u6001",
    "Mismatch Count": "\u5dee\u5f02\u6570\u91cf",
    "Mismatch Ratio": "\u5dee\u5f02\u6bd4\u4f8b",
    "Reconciliation Details": "\u5bf9\u8d26\u8be6\u60c5",
    "Inspect Prediction Run": "\u67e5\u770b\u9884\u6d4b\u8bb0\u5f55",
    "LLM Decision Chain": "LLM \u51b3\u7b56\u94fe\u8def",
    "Symbol": "\u5e01\u79cd",
    "Up Probability": "\u4e0a\u6da8\u6982\u7387",
    "Suggested Action": "\u5efa\u8bae\u52a8\u4f5c",
    "Final Score": "\u6700\u7ec8\u5f97\u5206",
    "LLM Confidence": "LLM \u7f6e\u4fe1\u5ea6",
    "Market Regime": "\u5e02\u573a\u72b6\u6001",
    "XGB Threshold": "XGB \u9608\u503c",
    "Final Threshold": "\u6700\u7ec8\u9608\u503c",
    "Key Reasons": "\u5173\u952e\u539f\u56e0",
    "No key reasons recorded.": "\u6682\u65e0\u5173\u952e\u539f\u56e0\u3002",
    "Risk Warnings": "\u98ce\u9669\u63d0\u793a",
    "No risk warnings recorded.": "\u6682\u65e0\u98ce\u9669\u63d0\u793a\u3002",
    "Decision Runtime Parameters": "\u51b3\u7b56\u8fd0\u884c\u53c2\u6570",
    "LLM Raw Content": "LLM \u539f\u59cb\u5185\u5bb9",
    "Latest Intelligence Factors": "\u6700\u65b0\u667a\u80fd\u56e0\u5b50",
    "LLM Sentiment": "LLM \u60c5\u7eea",
    "Onchain Netflow": "\u94fe\u4e0a\u51c0\u6d41\u5165",
    "Onchain Whale": "\u9cb8\u9c7c\u5206\u5e03",
    "Black Swan Index": "\u9ed1\u5929\u9e45\u6307\u6570",
    "Trend Strength": "\u8d8b\u52bf\u5f3a\u5ea6",
    "ATR Percentile": "ATR \u5206\u4f4d",
    "Price Structure": "\u4ef7\u683c\u7ed3\u6784",
    "Latest Research Digest": "\u6700\u65b0\u7814\u7a76\u6458\u8981",
    "News Summary": "\u65b0\u95fb\u6458\u8981",
    "Macro Summary": "\u5b8f\u89c2\u6458\u8981",
    "Onchain Summary": "\u94fe\u4e0a\u6458\u8981",
    "Fear & Greed": "\u6050\u60e7\u4e0e\u8d2a\u5a6a",
    "Latest Feature Snapshots": "\u6700\u65b0\u7279\u5f81\u5feb\u7167",
    "No feature snapshots recorded.": "\u6682\u65e0\u7279\u5f81\u5feb\u7167\u3002",
    "No research inputs recorded.": "\u6682\u65e0\u7814\u7a76\u8f93\u5165\u3002",
    "Net Expectancy": "\u51c0\u671f\u671b\u6536\u76ca",
    "Net Profit Factor": "\u51c0\u76c8\u4e8f\u6bd4",
    "Equity Return": "\u6743\u76ca\u7d2f\u8ba1\u6536\u76ca",
    "Total Trade Cost": "\u7d2f\u8ba1\u4ea4\u6613\u6210\u672c",
    "Total Slippage Drag": "\u7d2f\u8ba1\u6ed1\u70b9\u62d6\u7d2f",
    "Recent Closed Trades": "\u6700\u8fd1\u95ed\u73af\u4ea4\u6613\u6570",
    "Recent Net Expectancy": "\u6700\u8fd1\u51c0\u671f\u671b\u6536\u76ca",
    "Recent Net Profit Factor": "\u6700\u8fd1\u51c0\u76c8\u4e8f\u6bd4",
    "Recent Net Max Drawdown": "\u6700\u8fd1\u51c0\u6700\u5927\u56de\u64a4",
    "Recent Net Sharpe": "\u6700\u8fd1\u51c0 Sharpe",
    "Recent Net Sortino": "\u6700\u8fd1\u51c0 Sortino",
    "Average Holding Hours": "\u5e73\u5747\u6301\u6709\u65f6\u957f",
    "Equity Curve": "\u6743\u76ca\u66f2\u7ebf",
    "Recent PnL Ledger": "\u6700\u8fd1 PnL \u53f0\u8d26",
    "No PnL ledger entries.": "\u6682\u65e0 PnL \u53f0\u8d26\u8bb0\u5f55\u3002",
    "Shadow Runtime Coverage": "\u5f71\u5b50\u8fd0\u884c\u8986\u76d6",
    "Execution Evaluations": "\u6267\u884c\u8bc4\u4f30\u6570",
    "Shadow Evaluations": "\u89c2\u5bdf\u8bc4\u4f30\u6570",
    "Open Shadow Trades": "\u672a\u5230\u671f\u5f71\u5b50\u4ea4\u6613",
    "Evaluated Shadow Trades": "\u5df2\u8bc4\u4f30\u5f71\u5b50\u4ea4\u6613",
    "Execution Accuracy": "\u6267\u884c\u51c6\u786e\u7387",
    "Shadow Accuracy": "\u89c2\u5bdf\u51c6\u786e\u7387",
    "Execution Expectancy": "\u6267\u884c\u671f\u671b\u6536\u76ca",
    "Shadow Expectancy": "\u5f71\u5b50\u671f\u671b\u6536\u76ca",
    "Shadow Avg PnL": "\u5f71\u5b50\u5e73\u5747\u6536\u76ca",
    "Recent Prediction Evaluations": "\u6700\u8fd1\u6210\u719f\u8bc4\u4f30",
    "No prediction evaluations recorded.": "\u6682\u65e0\u6210\u719f\u8bc4\u4f30\u8bb0\u5f55\u3002",
    "Blocked Shadow Trades": "\u88ab\u62e6\u5f71\u5b50\u4ea4\u6613",
    "No blocked shadow trades recorded.": "\u6682\u65e0\u88ab\u62e6\u5f71\u5b50\u4ea4\u6613\u8bb0\u5f55\u3002",
    "Exploration Slots": "\u63a2\u7d22\u5e2d\u4f4d",
    "Paper Canary Opens": "Paper Canary \u5f00\u4ed3\u6570",
    "Soft Canary Opens": "\u8f6f\u5ba1\u6279 Canary \u5f00\u4ed3\u6570",
    "Market Data Provider": "\u5e02\u573a\u6570\u636e\u63d0\u4f9b\u65b9",
    "Short-Horizon Opens": "Short-Horizon \u653e\u884c\u5f00\u4ed3",
    "Short-Horizon Net PnL": "Short-Horizon \u653e\u884c\u51c0\u6536\u76ca",
    "Short-Horizon Closed Trades": "Short-Horizon \u653e\u884c\u5e73\u4ed3\u6570",
    "Short-Horizon Pauses": "Short-Horizon \u8d1f\u671f\u671b\u6682\u505c",
    "Short-Horizon Status": "Short-Horizon \u5f53\u524d\u72b6\u6001",
    "Short-Horizon Samples": "Short-Horizon \u6700\u8fd1\u6837\u672c",
    "Short-Horizon Expectancy": "Short-Horizon \u6700\u8fd1\u51c0\u671f\u671b",
    "Short-Horizon Profit Factor": "Short-Horizon \u6700\u8fd1\u51c0\u76c8\u4e8f\u6bd4",
    "Market Data Operation": "\u5e02\u573a\u6570\u636e\u64cd\u4f5c",
    "Market Data Failover": "\u5e02\u573a\u6570\u636e\u6545\u969c\u5207\u6362",
    "Market Data Failover Count": "\u5e02\u573a\u6570\u636e\u6545\u969c\u5207\u6362\u6b21\u6570",
    "Primary Market Data Failures": "\u4e3b\u5e02\u573a\u6570\u636e\u5931\u8d25\u6b21\u6570",
    "Secondary Market Data Failures": "\u6b21\u5e02\u573a\u6570\u636e\u5931\u8d25\u6b21\u6570",
    "No market data failover route recorded.": "\u6682\u65e0\u5e02\u573a\u6570\u636e\u6545\u969c\u5207\u6362\u8def\u7531\u8bb0\u5f55\u3002",
    "Performance Report": "\u6027\u80fd\u62a5\u544a",
    "Health Report": "\u5065\u5eb7\u68c0\u67e5\u62a5\u544a",
    "Pool Attribution Report": "\u4ea4\u6613\u6c60\u5f52\u56e0\u62a5\u544a",
    "Incident Report": "\u4e8b\u6545\u62a5\u544a",
    "Failure Report": "\u6545\u969c\u62a5\u544a",
    "AB Test Report": "A/B \u6d4b\u8bd5\u62a5\u544a",
    "Validation Sprint Report": "\u5feb\u901f\u9a8c\u8bc1\u62a5\u544a",
    "Guard Report": "\u98ce\u63a7\u544a\u8b66\u62a5\u544a",
    "Promotion Funnel": "\u664b\u7ea7\u6f0f\u6597",
    "Shadow Observation": "\u5f71\u5b50\u89c2\u5bdf",
    "Shadow Candidates": "\u5f71\u5b50\u5019\u9009",
    "Live Canary": "\u5b9e\u76d8 Canary",
    "Promoted Observing": "\u664b\u7ea7\u89c2\u5bdf\u4e2d",
    "Paper Canary Opened": "\u5df2\u89e6\u53d1 Paper Canary",
    "Promoted": "\u5df2\u664b\u7ea7",
    "Accepted": "\u5df2\u63a5\u53d7",
    "Rolled Back": "\u5df2\u56de\u6eda",
    "No promotion funnel data.": "\u6682\u65e0\u664b\u7ea7\u6f0f\u6597\u6570\u636e\u3002",
    "Current Promotion Candidates": "\u5f53\u524d\u664b\u7ea7\u5019\u9009",
    "No promotion candidates pending.": "\u6682\u65e0\u5f85\u5904\u7406\u7684\u664b\u7ea7\u5019\u9009\u3002",
    "Pending Shadow Candidates": "\u5f85\u5904\u7406\u5f71\u5b50\u5019\u9009",
    "Live Canary Candidates": "\u5b9e\u76d8 Canary \u5019\u9009",
    "Promoted Models Under Observation": "\u89c2\u5bdf\u4e2d\u7684\u664b\u7ea7\u6a21\u578b",
    "Latest Prediction Eval": "\u6700\u65b0\u9884\u6d4b\u8bc4\u4f30",
    "Recent Research Reviews": "\u6700\u8fd1\u7814\u7a76\u590d\u6838",
    "No data.": "\u6682\u65e0\u6570\u636e\u3002",
    "Prediction Flow": "\u9884\u6d4b\u6d41\u7a0b",
    "Prediction Evaluations": "\u9884\u6d4b\u8bc4\u4f30",
    "Shadow Trades": "\u5f71\u5b50\u4ea4\u6613",
    "Active Symbols": "\u6d3b\u8dc3\u5e01\u79cd\u6570",
    "Added": "\u65b0\u589e",
    "Removed": "\u79fb\u9664",
    "Last Refresh": "\u6700\u8fd1\u5237\u65b0",
    "Refresh Reason": "\u5237\u65b0\u539f\u56e0",
    "none": "\u65e0",
    "Whitelist/blacklist saved.": "\u767d\u540d\u5355/\u9ed1\u540d\u5355\u5df2\u4fdd\u5b58\u3002",
    "Watchlist refreshed.": "\u4ea4\u6613\u6c60\u5df2\u5237\u65b0\u3002",
    "Watchlist refresh failed.": "\u4ea4\u6613\u6c60\u5237\u65b0\u5931\u8d25\u3002",
    "Execution Pool": "\u6267\u884c\u6c60",
    "Model-Ready Pool": "\u6a21\u578b\u5c31\u7eea\u6c60",
    "Observation Pool": "\u89c2\u5bdf\u6c60",
    "Current Execution Pool": "\u5f53\u524d\u6267\u884c\u6c60",
    "Current Model-Ready Pool": "\u5f53\u524d\u6a21\u578b\u5c31\u7eea\u6c60",
    "Consistency-Blocked Symbols": "\u4e00\u81f4\u6027\u62e6\u622a\u6807\u7684",
    "No consistency-blocked symbols.": "\u6682\u65e0\u88ab\u4e00\u81f4\u6027\u95e8\u69db\u62e6\u4e0b\u7684\u6807\u7684\u3002",
    "Current Observation Pool": "\u5f53\u524d\u89c2\u5bdf\u6c60",
    "Paper Exploration Slot": "Paper \u63a2\u7d22\u5e2d\u4f4d",
    "Apply Execution Pool": "\u5e94\u7528\u6267\u884c\u6c60",
    "Add To Execution Pool": "\u52a0\u5165\u6267\u884c\u6c60",
    "Remove From Execution Pool": "\u79fb\u51fa\u6267\u884c\u6c60",
    "Execution symbols, comma-separated.": "\u6267\u884c\u5e01\u79cd\uff0c\u9017\u53f7\u5206\u9694\u3002",
    "Symbols to add into the execution pool, comma-separated.": "\u52a0\u5165\u6267\u884c\u6c60\u7684\u5e01\u79cd\uff0c\u9017\u53f7\u5206\u9694\u3002",
    "Symbols to remove from the execution pool, comma-separated.": "\u4ece\u6267\u884c\u6c60\u79fb\u51fa\u7684\u5e01\u79cd\uff0c\u9017\u53f7\u5206\u9694\u3002",
    "Execution pool updated.": "\u6267\u884c\u6c60\u5df2\u66f4\u65b0\u3002",
    "Execution pool update failed.": "\u6267\u884c\u6c60\u66f4\u65b0\u5931\u8d25\u3002",
    "Execution pool addition finished.": "\u5df2\u5b8c\u6210\u52a0\u5165\u6267\u884c\u6c60\u3002",
    "Execution pool addition failed.": "\u52a0\u5165\u6267\u884c\u6c60\u5931\u8d25\u3002",
    "Execution pool removal finished.": "\u5df2\u5b8c\u6210\u79fb\u51fa\u6267\u884c\u6c60\u3002",
    "Execution pool removal failed.": "\u79fb\u51fa\u6267\u884c\u6c60\u5931\u8d25\u3002",
    "Execution Readiness": "\u6267\u884c\u5c31\u7eea\u60c5\u51b5",
    "Execution Pool Changes": "\u6267\u884c\u6c60\u53d8\u66f4\u8bb0\u5f55",
    "No execution pool changes recorded.": "\u6682\u65e0\u6267\u884c\u6c60\u53d8\u66f4\u8bb0\u5f55\u3002",
    "No execution readiness data.": "\u6682\u65e0\u6267\u884c\u5c31\u7eea\u6570\u636e\u3002",
    "in_execution_pool": "\u5728\u6267\u884c\u6c60",
    "model_ready": "\u6a21\u578b\u5c31\u7eea",
    "training_reason": "\u8bad\u7ec3\u539f\u56e0",
    "training_rows": "\u8bad\u7ec3\u884c\u6570",
    "holdout_accuracy": "Holdout \u51c6\u786e\u7387",
    "status_reason": "\u72b6\u6001\u539f\u56e0",
    "consistency_flags": "\u4e00\u81f4\u6027\u6807\u8bb0",
    "ready": "\u5c31\u7eea",
    "consistency_blocked": "\u4e00\u81f4\u6027\u62e6\u622a",
    "not_in_execution_pool": "\u672a\u52a0\u5165\u6267\u884c\u6c60",
    "not_trained": "\u672a\u8bad\u7ec3",
    "insufficient_rows": "\u6837\u672c\u4e0d\u8db3",
    "model_file_missing": "\u6a21\u578b\u6587\u4ef6\u7f3a\u5931",
    "training_failed_or_disabled": "\u8bad\u7ec3\u672a\u751f\u6210\u53ef\u7528\u6a21\u578b",
    "Force include symbols, comma-separated.": "\u5f3a\u5236\u7eb3\u5165\u4ea4\u6613\u6c60\uff0c\u9017\u53f7\u5206\u9694\u3002",
    "Force exclude symbols, comma-separated.": "\u5f3a\u5236\u6392\u9664\u4ea4\u6613\u6c60\uff0c\u9017\u53f7\u5206\u9694\u3002",
    "Run Training": "\u6267\u884c\u8bad\u7ec3",
    "Training finished.": "\u8bad\u7ec3\u5df2\u5b8c\u6210\u3002",
    "Training failed.": "\u8bad\u7ec3\u5931\u8d25\u3002",
    "Low resource mode is enabled. Training is disabled in the dashboard on this node.": "\u5f53\u524d\u5df2\u5f00\u542f\u4f4e\u8d44\u6e90\u6a21\u5f0f\uff0c\u672c\u8282\u70b9\u9762\u677f\u4e0a\u5df2\u7981\u7528\u8bad\u7ec3\u3002",
    "Recent Training Runs": "\u6700\u8fd1\u8bad\u7ec3\u8bb0\u5f55",
    "Latest Training Metadata": "\u6700\u65b0\u8bad\u7ec3\u5143\u6570\u636e",
    "No training runs recorded.": "\u6682\u65e0\u8bad\u7ec3\u8bb0\u5f55\u3002",
    "Run Walk-Forward": "\u6267\u884c\u6eda\u52a8\u9a8c\u8bc1",
    "Walk-forward finished.": "\u6eda\u52a8\u9a8c\u8bc1\u5df2\u5b8c\u6210\u3002",
    "Walk-forward failed.": "\u6eda\u52a8\u9a8c\u8bc1\u5931\u8d25\u3002",
    "Low resource mode is enabled. Walk-forward is disabled in the dashboard on this node.": "\u5f53\u524d\u5df2\u5f00\u542f\u4f4e\u8d44\u6e90\u6a21\u5f0f\uff0c\u672c\u8282\u70b9\u9762\u677f\u4e0a\u5df2\u7981\u7528\u6eda\u52a8\u9a8c\u8bc1\u3002",
    "No walk-forward runs recorded.": "\u6682\u65e0\u6eda\u52a8\u9a8c\u8bc1\u8bb0\u5f55\u3002",
    "Recent Walk-Forward Runs": "\u6700\u8fd1\u6eda\u52a8\u9a8c\u8bc1\u8bb0\u5f55",
    "Backtest Symbol": "\u56de\u6d4b\u5e01\u79cd",
    "Run Backtest": "\u6267\u884c\u56de\u6d4b",
    "Backtest finished.": "\u56de\u6d4b\u5df2\u5b8c\u6210\u3002",
    "Backtest failed.": "\u56de\u6d4b\u5931\u8d25\u3002",
    "Low resource mode is enabled. Backtest is disabled in the dashboard on this node.": "\u5f53\u524d\u5df2\u5f00\u542f\u4f4e\u8d44\u6e90\u6a21\u5f0f\uff0c\u672c\u8282\u70b9\u9762\u677f\u4e0a\u5df2\u7981\u7528\u56de\u6d4b\u3002",
    "No backtest runs recorded.": "\u6682\u65e0\u56de\u6d4b\u8bb0\u5f55\u3002",
    "Report generation failed.": "\u62a5\u544a\u751f\u6210\u5931\u8d25\u3002",
    "No training metadata.": "\u6682\u65e0\u8bad\u7ec3\u5143\u6570\u636e\u3002",
    "Latest Walk-Forward Summary": "\u6700\u65b0\u6eda\u52a8\u9a8c\u8bc1\u6458\u8981",
    "No walk-forward summary.": "\u6682\u65e0\u6eda\u52a8\u9a8c\u8bc1\u6458\u8981\u3002",
    "Latest Drift Report": "\u6700\u65b0\u504f\u5dee\u62a5\u544a",
    "Latest AB Test Report": "\u6700\u65b0 A/B \u6d4b\u8bd5\u62a5\u544a",
    "Latest Execution Events": "\u6700\u65b0\u6267\u884c\u4e8b\u4ef6",
    "No execution events.": "\u6682\u65e0\u6267\u884c\u4e8b\u4ef6\u3002",
    "Report Artifacts": "\u62a5\u544a\u4ea7\u7269",
    "View Artifact": "\u67e5\u770b\u62a5\u544a\u4ea7\u7269",
    "Orders": "\u8ba2\u5355",
    "No orders recorded.": "\u6682\u65e0\u8ba2\u5355\u8bb0\u5f55\u3002",
    "No execution events recorded.": "\u6682\u65e0\u6267\u884c\u4e8b\u4ef6\u8bb0\u5f55\u3002",
    "Execution Events": "\u6267\u884c\u4e8b\u4ef6",
    "No reconciliation runs recorded.": "\u6682\u65e0\u5bf9\u8d26\u8bb0\u5f55\u3002",
    "Reconciliation Runs": "\u5bf9\u8d26\u8bb0\u5f55",
    "Cycle Runs": "\u5468\u671f\u8bb0\u5f55",
    "No cycle runs recorded.": "\u6682\u65e0\u5468\u671f\u8bb0\u5f55\u3002",
    "Latest Risk State": "\u6700\u65b0\u98ce\u9669\u72b6\u6001",
    "Health check finished.": "\u5065\u5eb7\u68c0\u67e5\u5df2\u5b8c\u6210\u3002",
    "Health check failed.": "\u5065\u5eb7\u68c0\u67e5\u5931\u8d25\u3002",
    "No health reports recorded.": "\u6682\u65e0\u5065\u5eb7\u68c0\u67e5\u62a5\u544a\u3002",
    "Metrics generated.": "\u6307\u6807\u5df2\u751f\u6210\u3002",
    "Metrics failed.": "\u6307\u6807\u751f\u6210\u5931\u8d25\u3002",
    "Holdout Accuracy": "\u7559\u51fa\u96c6\u51c6\u786e\u7387",
    "Latest WF Return": "\u6700\u65b0 WF \u6536\u76ca",
    "WF Return": "WF \u6536\u76ca",
    "Evaluated Predictions": "\u5df2\u8bc4\u4f30\u9884\u6d4b\u6570",
    "LLM Accuracy": "LLM \u51c6\u786e\u7387",
    "Fusion Accuracy": "\u878d\u5408\u51c6\u786e\u7387",
    "Degradation Status": "\u8870\u51cf\u72b6\u6001",
    "Runtime XGB Threshold": "\u8fd0\u884c\u65f6 XGB \u9608\u503c",
    "Runtime Final Threshold": "\u8fd0\u884c\u65f6\u6700\u7ec8\u9608\u503c",
    "No performance reports recorded.": "\u6682\u65e0\u6027\u80fd\u62a5\u544a\u3002",
    "No walk-forward runs.": "\u6682\u65e0\u6eda\u52a8\u9a8c\u8bc1\u8bb0\u5f55\u3002",
    "Performance History": "\u6027\u80fd\u5386\u53f2",
    "Generate Drift Report": "\u751f\u6210\u504f\u5dee\u62a5\u544a",
    "Drift report generated.": "\u504f\u5dee\u62a5\u544a\u5df2\u751f\u6210\u3002",
    "Drift report failed.": "\u504f\u5dee\u62a5\u544a\u751f\u6210\u5931\u8d25\u3002",
    "Low resource mode is enabled. Drift generation is disabled in the dashboard on this node.": "\u5f53\u524d\u5df2\u5f00\u542f\u4f4e\u8d44\u6e90\u6a21\u5f0f\uff0c\u672c\u8282\u70b9\u9762\u677f\u4e0a\u5df2\u7981\u7528\u504f\u5dee\u62a5\u544a\u751f\u6210\u3002",
    "No drift reports recorded.": "\u6682\u65e0\u504f\u5dee\u62a5\u544a\u3002",
    "Recent Backtests": "\u6700\u8fd1\u56de\u6d4b",
    "No backtest runs.": "\u6682\u65e0\u56de\u6d4b\u8bb0\u5f55\u3002",
    "Generate AB Test Report": "\u751f\u6210 A/B \u6d4b\u8bd5\u62a5\u544a",
    "AB test report generated.": "A/B \u6d4b\u8bd5\u62a5\u544a\u5df2\u751f\u6210\u3002",
    "AB test report failed.": "A/B \u6d4b\u8bd5\u62a5\u544a\u751f\u6210\u5931\u8d25\u3002",
    "Low resource mode is enabled. AB test reporting is disabled in the dashboard on this node.": "\u5f53\u524d\u5df2\u5f00\u542f\u4f4e\u8d44\u6e90\u6a21\u5f0f\uff0c\u672c\u8282\u70b9\u9762\u677f\u4e0a\u5df2\u7981\u7528 A/B \u6d4b\u8bd5\u62a5\u544a\u3002",
    "No A/B test runs recorded.": "\u6682\u65e0 A/B \u6d4b\u8bd5\u8bb0\u5f55\u3002",
    "Guard report generated.": "\u98ce\u63a7\u62a5\u544a\u5df2\u751f\u6210\u3002",
    "Guard report failed.": "\u98ce\u63a7\u62a5\u544a\u751f\u6210\u5931\u8d25\u3002",
    "Manual recovery approved.": "\u5df2\u6279\u51c6\u4eba\u5de5\u6062\u590d\u3002",
    "Manual recovery approval failed.": "\u4eba\u5de5\u6062\u590d\u6279\u51c6\u5931\u8d25\u3002",
    "Recent Guard Events": "\u6700\u8fd1\u98ce\u63a7\u4e8b\u4ef6",
    "Cooldown": "\u51b7\u5374\u671f",
    "Guard Reason": "\u98ce\u63a7\u539f\u56e0",
    "Degradation Reason": "\u8870\u51cf\u539f\u56e0",
    "Manual Recovery": "\u4eba\u5de5\u6062\u590d",
    "Recovery Reason": "\u6062\u590d\u539f\u56e0",
    "Guard Report Artifacts": "\u98ce\u63a7\u62a5\u544a\u4ea7\u7269",
    "No guard reports recorded.": "\u6682\u65e0\u98ce\u63a7\u62a5\u544a\u3002",
    "Guard Execution Events": "\u98ce\u63a7\u6267\u884c\u4e8b\u4ef6",
    "No guard execution events.": "\u6682\u65e0\u98ce\u63a7\u6267\u884c\u4e8b\u4ef6\u3002",
    "Guard Event Trend": "\u98ce\u63a7\u4e8b\u4ef6\u8d8b\u52bf",
    "Generate Failure Report": "\u751f\u6210\u6545\u969c\u62a5\u544a",
    "Failure report generated.": "\u6545\u969c\u62a5\u544a\u5df2\u751f\u6210\u3002",
    "Failure report failed.": "\u6545\u969c\u62a5\u544a\u751f\u6210\u5931\u8d25\u3002",
    "Run Maintenance": "\u6267\u884c\u7ef4\u62a4",
    "Maintenance completed.": "\u7ef4\u62a4\u5df2\u5b8c\u6210\u3002",
    "Maintenance failed.": "\u7ef4\u62a4\u5931\u8d25\u3002",
    "Generate Incident Report": "\u751f\u6210\u4e8b\u6545\u62a5\u544a",
    "Incident report generated.": "\u4e8b\u6545\u62a5\u544a\u5df2\u751f\u6210\u3002",
    "Incident report failed.": "\u4e8b\u6545\u62a5\u544a\u751f\u6210\u5931\u8d25\u3002",
    "No failure or maintenance reports.": "\u6682\u65e0\u6545\u969c\u6216\u7ef4\u62a4\u62a5\u544a\u3002",
    "Run Scheduled Job": "\u8fd0\u884c\u8c03\u5ea6\u4efb\u52a1",
    "Scheduled job finished.": "\u8c03\u5ea6\u4efb\u52a1\u5df2\u5b8c\u6210\u3002",
    "Scheduled job failed.": "\u8c03\u5ea6\u4efb\u52a1\u5931\u8d25\u3002",
    "Low resource mode is enabled. Heavy jobs are hidden from the scheduler UI on this node.": "\u5f53\u524d\u5df2\u5f00\u542f\u4f4e\u8d44\u6e90\u6a21\u5f0f\uff0c\u91cd\u4efb\u52a1\u5df2\u5728\u8c03\u5ea6\u754c\u9762\u4e2d\u9690\u85cf\u3002",
    "No scheduler runs recorded.": "\u6682\u65e0\u8c03\u5ea6\u8bb0\u5f55\u3002",
    "No log files found in data/logs.": "data/logs \u4e0b\u6682\u65e0\u65e5\u5fd7\u6587\u4ef6\u3002",
    "No report artifacts.": "\u6682\u65e0\u62a5\u544a\u4ea7\u7269\u3002",
    "(empty log)": "\u7a7a\u65e5\u5fd7",
    "Run Analysis Once": "\u6267\u884c\u4e00\u6b21\u5206\u6790",
    "Analysis finished.": "\u5206\u6790\u5df2\u5b8c\u6210\u3002",
    "Analysis failed.": "\u5206\u6790\u5931\u8d25\u3002",
    "Run Backfill 180d": "\u6267\u884c 180 \u5929\u56de\u586b",
    "Backfill finished.": "\u56de\u586b\u5df2\u5b8c\u6210\u3002",
    "Backfill failed.": "\u56de\u586b\u5931\u8d25\u3002",
    "Loop mode should be started from terminal or process manager.\nDashboard only supports one-shot commands.": "\u5faa\u73af\u6a21\u5f0f\u8bf7\u4ece\u7ec8\u7aef\u6216\u8fdb\u7a0b\u7ba1\u7406\u5668\u542f\u52a8\u3002\nDashboard \u53ea\u652f\u6301\u5355\u6b21\u547d\u4ee4\u3002",
    "Low resource mode is enabled. Backfill is disabled in the operations page on this node.": "\u5f53\u524d\u5df2\u5f00\u542f\u4f4e\u8d44\u6e90\u6a21\u5f0f\uff0c\u672c\u8282\u70b9\u64cd\u4f5c\u9875\u9762\u5df2\u7981\u7528\u56de\u586b\u3002",
    "Recent Account Snapshots": "\u6700\u8fd1\u8d26\u6237\u5feb\u7167",
    "report": "\u62a5\u544a",
    "ops": "\u8fd0\u7ef4",
    "metrics": "\u6307\u6807",
    "guards": "\u98ce\u63a7",
    "reconcile": "\u5bf9\u8d26",
    "schedule": "\u8c03\u5ea6",
    "train": "\u8bad\u7ec3",
    "backtest": "\u56de\u6d4b",
    "backfill": "\u56de\u586b",
    "once": "\u5355\u6b21\u5206\u6790",
    "entries": "\u5165\u573a\u626b\u63cf",
    "manual_refresh": "\u624b\u52a8\u5237\u65b0",
    "auto_rebuild": "\u81ea\u52a8\u91cd\u5efa",
    "manual": "\u624b\u52a8",
    "primary": "\u4e3b\u7b56\u7565",
    "soft_review": "\u8f6f\u5ba1\u6279",
    "offensive_review": "\u8fdb\u653b\u5ba1\u67e5",
    "paper": "\u7eb8\u9762",
    "value": "\u503c",
    "manager_not_approved": "\u672a\u901a\u8fc7\u7ba1\u7406\u5668\u5ba1\u6279",
    "risk_warning_present": "\u5b58\u5728\u98ce\u9669\u63d0\u793a",
    "regime_extreme_fear": "\u5e02\u573a\u5904\u4e8e\u6781\u7aef\u6050\u60e7",
    "fear_greed_extreme_fear": "\u6050\u60e7\u8d2a\u5a6a\u6307\u6570\u663e\u793a\u6781\u7aef\u6050\u60e7",
    "news_coverage_thin": "\u65b0\u95fb\u8986\u76d6\u7a00\u8584",
    "setup_negative_expectancy": "\u5f62\u6001\u671f\u671b\u6536\u76ca\u4e3a\u8d1f",
    "trend_against": "\u8d8b\u52bf\u4e0d\u5229",
    "trend_supportive": "\u8d8b\u52bf\u652f\u6301",
    "onchain_neutral": "\u94fe\u4e0a\u4fe1\u53f7\u4e2d\u6027",
    "liquidity_weak": "\u6d41\u52a8\u6027\u504f\u5f31",
    "liquidity_supportive": "\u6d41\u52a8\u6027\u652f\u6301",
    "xgb_weak": "XGB \u4fe1\u53f7\u504f\u5f31",
    "xgb_pass": "XGB \u901a\u8fc7\u57fa\u7840\u9608\u503c",
    "xgb_strong": "XGB \u4fe1\u53f7\u8f83\u5f3a",
    "setup_auto_pause": "\u5f62\u6001\u88ab\u81ea\u52a8\u6682\u505c",
    "review_score": "\u590d\u6838\u8bc4\u5206",
    "setup_avg_outcome": "\u5f62\u6001\u5e73\u5747\u7ed3\u679c",
    "recent_realized_setup_negative_expectancy": "\u8fd1\u671f\u5df2\u5b9e\u73b0\u5f62\u6001\u8d1f\u671f\u671b",
    "realized_setup_negative_expectancy": "\u5df2\u5b9e\u73b0\u5f62\u6001\u8d1f\u671f\u671b",
    "fallback_research_model": "\u7814\u7a76\u6a21\u578b\u56de\u9000",
    "fallback_liquidity_supportive": "\u56de\u9000\u903b\u8f91\u8ba4\u4e3a\u6d41\u52a8\u6027\u652f\u6301",
    "fallback_liquidity_weak": "\u56de\u9000\u903b\u8f91\u8ba4\u4e3a\u6d41\u52a8\u6027\u504f\u5f31",
    "fallback_trend_supportive": "\u56de\u9000\u903b\u8f91\u8ba4\u4e3a\u8d8b\u52bf\u652f\u6301",
    "fallback_trend_against": "\u56de\u9000\u903b\u8f91\u8ba4\u4e3a\u8d8b\u52bf\u4e0d\u5229",
    "fallback_oversold_reversal": "\u56de\u9000\u903b\u8f91\u8ba4\u4e3a\u8d85\u8dcc\u53cd\u8f6c",
    "fallback_momentum_breakdown": "\u56de\u9000\u903b\u8f91\u8ba4\u4e3a\u52a8\u80fd\u7834\u574f",
    "fallback_funding_overheated": "\u56de\u9000\u903b\u8f91\u8ba4\u4e3a\u8d44\u91d1\u8d39\u8fc7\u70ed",
    "fallback_short_crowding_support": "\u56de\u9000\u903b\u8f91\u8ba4\u4e3a\u7a7a\u5934\u62e5\u6324\u652f\u6301\u53cd\u5f39",
    "extreme fear": "\u6781\u7aef\u6050\u60e7",
    "extreme greed": "\u6781\u7aef\u8d2a\u5a6a",
    "liquidity weak": "\u6d41\u52a8\u6027\u504f\u5f31",
    "trend against": "\u8d8b\u52bf\u4e0d\u5229",
    "funding overheated": "\u8d44\u91d1\u8d39\u8fc7\u70ed",
    "high downside volatility": "\u4e0b\u8dcc\u6ce2\u52a8\u8f83\u5927",
}


def t(text: str) -> str:
    if current_language() == "zh":
        return ZH_UI.get(text, text)
    return text


def current_language() -> str:
    row = get_state_row(LANGUAGE_STATE_KEY)
    if row and row.get("value"):
        return normalize_language(row["value"])
    return get_default_language()


def set_language(language: str) -> None:
    execute_sql(
        """INSERT OR REPLACE INTO system_state (key, value, updated_at)
           VALUES (?, ?, ?)""",
        (LANGUAGE_STATE_KEY, normalize_language(language), datetime.now(timezone.utc).isoformat()),
    )


COLUMN_ZH = {
    "id": "ID",
    "symbol": "币种",
    "direction": "方向",
    "entry_price": "开仓价",
    "exit_price": "平仓价",
    "quantity": "数量",
    "entry_time": "开仓时间",
    "exit_time": "平仓时间",
    "pnl": "盈亏",
    "pnl_pct": "收益率",
    "updated_at": "更新时间",
    "created_at": "创建时间",
    "timestamp": "时间",
    "timeframe": "周期",
    "valid": "有效",
    "report_type": "报告类型",
    "job_name": "任务名",
    "status": "状态",
    "started_at": "开始时间",
    "completed_at": "完成时间",
    "mismatch_count": "差异数量",
    "order_id": "订单号",
    "side": "买卖方向",
    "order_type": "订单类型",
    "price": "价格",
    "reason": "原因",
    "event_type": "事件类型",
    "exchange_order_id": "交易所订单号",
    "execution_type": "执行类型",
    "cooldown_until": "冷却至",
    "model_version": "模型版本",
    "up_probability": "上涨概率",
    "feature_count": "特征数",
    "fear_greed": "恐惧贪婪",
    "news_summary": "新闻摘要",
    "macro_summary": "宏观摘要",
    "onchain_summary": "链上摘要",
    "report_type": "报告类型",
    "open_positions": "持仓数",
    "equity": "权益",
    "drawdown_pct": "回撤",
    "daily_loss_pct": "日亏损",
    "weekly_loss_pct": "周亏损",
    "realized_pnl": "已实现盈亏",
    "unrealized_pnl": "未实现盈亏",
    "total_exposure_pct": "总仓位",
    "circuit_breaker_active": "熔断触发",
    "notes": "备注",
    "source": "来源",
    "score": "评分",
    "sector": "赛道",
    "quote_volume_24h": "24小时成交额",
    "change_pct_24h": "24小时涨跌幅",
    "is_core": "核心币",
    "champion_model_version": "冠军模型版本",
    "challenger_model_version": "挑战者模型版本",
    "champion_probability": "冠军概率",
    "challenger_probability": "挑战者概率",
    "selected_variant": "选中版本",
    "allocation_pct": "分配比例",
    "feature_count": "特征数量",
    "pipeline_mode": "流程",
    "evaluation_type": "评估类型",
    "is_correct": "是否正确",
    "entry_close": "入场收盘价",
    "future_close": "到期收盘价",
    "block_reason": "拦截原因",
    "trade_net_return_pct": "交易净收益率",
    "opportunity_return_pct": "机会收益率",
    "favorable_excursion_pct": "最大有利波动",
    "adverse_excursion_pct": "最大不利波动",
    "rows": "样本行数",
    "positives": "正样本",
    "negatives": "负样本",
    "model_path": "模型路径",
    "engine": "引擎",
    "trade_id": "交易ID",
    "event_time": "事件时间",
    "notional_value": "名义价值",
    "reference_price": "参考价格",
    "fill_price": "成交价格",
    "gross_pnl": "毛收益",
    "fee_cost": "手续费",
    "slippage_cost": "滑点拖累",
    "net_pnl": "净收益",
    "net_return_pct": "净收益率",
    "holding_hours": "持有时长(小时)",
    "model_id": "模型ID",
    "execution_expectancy_pct": "执行期望收益",
    "shadow_expectancy_pct": "影子期望收益",
    "expectancy_pct": "期望收益率",
    "profit_factor": "盈亏比",
    "max_drawdown_pct": "最大回撤",
    "trade_win_rate": "交易胜率",
    "avg_cost_pct": "平均成本",
    "avg_favorable_excursion_pct": "平均有利波动",
    "avg_adverse_excursion_pct": "平均不利波动",
    "active_model_path": "Active模型路径",
    "active_symbols": "活跃币种",
    "added_symbols": "新增币种",
    "avg_expectancy_pct": "平均期望收益",
    "avg_max_drawdown_pct": "平均最大回撤",
    "avg_pnl_pct": "平均收益率",
    "avg_profit_factor": "平均盈亏比",
    "backup_model_path": "备份模型路径",
    "baseline_expectancy_pct": "基线期望收益",
    "baseline_holdout_accuracy": "基线Holdout准确率",
    "baseline_max_drawdown_pct": "基线最大回撤",
    "baseline_profit_factor": "基线盈亏比",
    "blacklist": "黑名单",
    "candidate_holdout_accuracy": "候选Holdout准确率",
    "candidate_walkforward_summary": "候选滚动验证摘要",
    "candidates": "候选列表",
    "challenger_model_path": "挑战者模型路径",
    "content": "内容",
    "decision_json": "决策JSON",
    "details_json": "详情JSON",
    "effective_mode": "生效模式",
    "evaluated_count": "已评估数量",
    "execution_accuracy": "执行准确率",
    "execution_count": "执行数量",
    "funnel_stage": "漏斗阶段",
    "key_reason": "关键原因",
    "live_allocation_pct": "实盘分配比例",
    "metadata_json": "元数据JSON",
    "metrics": "指标",
    "missing_positions": "缺失持仓",
    "missing_trades": "缺失交易",
    "model_ready_symbols": "模型就绪币种",
    "open_count": "未平数量",
    "payload_json": "载荷JSON",
    "promotion_reason": "晋级原因",
    "quantity_mismatches": "数量不一致",
    "raw_active_symbols": "原始观察币种",
    "removed_symbols": "移除币种",
    "research_json": "研究JSON",
    "restored_from": "恢复来源",
    "risk_warning": "风险提示",
    "shadow_accuracy": "影子准确率",
    "shadow_avg_trade_return_pct": "影子平均交易收益",
    "shadow_champion_accuracy": "影子冠军准确率",
    "shadow_count": "影子数量",
    "shadow_max_drawdown_pct": "影子最大回撤",
    "shadow_profit_factor": "影子盈亏比",
    "soft_count": "软审批数量",
    "total_count": "总数量",
    "total_return_pct": "总收益率",
    "trained_with_xgboost": "已使用XGBoost训练",
    "training_metadata": "训练元数据",
    "whitelist": "白名单",
}

CANONICAL_TO_ZH_METRIC = {
    canonical: local
    for local, canonical in METRIC_KEY_ALIASES.items()
    if any(ord(ch) > 127 for ch in local)
}

REPORT_TITLE_ZH = {
    "Performance Report": "性能报告",
    "Health Report": "健康检查报告",
    "Ops Overview": "运维总览",
    "Pool Attribution Report": "交易池归因报告",
    "Incident Report": "事故报告",
    "Drift Report": "漂移报告",
    "Failure Report": "故障报告",
    "Alpha Diagnostics": "Alpha 诊断日报",
    "Guard Report": "风控告警报告",
    "AB Test Report": "A/B 测试报告",
    "Validation Sprint Report": "快速验证报告",
}


def display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or current_language() != "zh":
        return df
    displayed = df.copy()
    for column in displayed.columns:
        if pd.api.types.is_bool_dtype(displayed[column]):
            displayed[column] = displayed[column].map(display_value)
            continue
        if pd.api.types.is_object_dtype(displayed[column]) or pd.api.types.is_string_dtype(displayed[column]):
            displayed[column] = displayed[column].apply(display_value)
    mapping = {column: COLUMN_ZH.get(column, column) for column in df.columns}
    return displayed.rename(columns=mapping)


def _translate_ui_text(text: str) -> str:
    raw = str(text)
    stripped = raw.strip()
    candidates = [raw, stripped]
    if stripped:
        candidates.extend([stripped.lower(), stripped.upper()])
    seen = set()
    for candidate in candidates:
        if candidate in seen or candidate == "":
            continue
        seen.add(candidate)
        if candidate in ZH_UI:
            return ZH_UI[candidate]
        if candidate in COLUMN_ZH:
            return COLUMN_ZH[candidate]
    return raw


def display_value(value):
    if current_language() != "zh":
        return value
    if value is None:
        return ZH_UI["N/A"]
    if isinstance(value, bool):
        return ZH_UI["yes"] if value else ZH_UI["no"]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if pd.isna(value):
                return ZH_UI["N/A"]
        except Exception:
            pass
        return value
    if isinstance(value, list):
        return [display_value(item) for item in value]
    if isinstance(value, dict):
        return {display_value(key): display_value(item) for key, item in value.items()}
    text = str(value)
    if not text:
        return text
    match = re.match(r"^([A-Za-z0-9_./-]+)\s+\(([^()]*)\)$", text)
    if match:
        head = str(display_value(match.group(1)))
        tail = str(display_value(match.group(2)))
        if head != match.group(1) or tail != match.group(2):
            return f"{head} ({tail})"
    match = re.match(r"^([A-Za-z0-9_./-]+)=(.+)$", text)
    if match:
        left = str(display_value(match.group(1)))
        right = str(display_value(match.group(2)))
        if left != match.group(1) or right != match.group(2):
            return f"{left}={right}"
    translated = _translate_ui_text(text)
    if translated != text:
        return translated
    match = re.match(r"^(review_score|setup_avg_outcome|recent_realized_setup_negative_expectancy|realized_setup_negative_expectancy)_(.+)$", text)
    if match:
        return f"{display_value(match.group(1))}_{match.group(2)}"
    for separator in (" | ", ", ", "; ", ","):
        if separator not in text:
            continue
        parts = text.split(separator)
        translated_parts = [display_value(part) for part in parts]
        if any(str(new) != old for new, old in zip(translated_parts, parts)):
            return separator.join(str(part) for part in translated_parts)
    return text


def display_json(value):
    if current_language() != "zh":
        return value
    if isinstance(value, dict):
        translated = {}
        for key, item in value.items():
            translated_key = COLUMN_ZH.get(str(key), ZH_UI.get(str(key), str(key)))
            translated[translated_key] = display_json(item)
        return translated
    if isinstance(value, list):
        return [display_json(item) for item in value]
    return display_value(value)


def display_kv_rows(data: dict, preferred_keys: list[str] | None = None) -> pd.DataFrame:
    if not isinstance(data, dict):
        return pd.DataFrame()
    keys = preferred_keys or list(data.keys())
    rows = []
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, (dict, list)):
            value = json.dumps(display_json(value), ensure_ascii=False)
        else:
            value = display_value(value)
        rows.append(
            {
                "field": COLUMN_ZH.get(str(key), ZH_UI.get(str(key), str(key))),
                "value": value,
            }
        )
    return pd.DataFrame(rows)


def display_research_text(value: str) -> str:
    text = (value or "").strip()
    if current_language() != "zh":
        return text
    known = {
        "No external news feed available, using neutral news context.": "当前未获取到外部新闻，使用中性新闻上下文。",
        "On-chain data unavailable, using neutral on-chain context.": "当前未获取到链上数据，使用中性链上上下文。",
        "Macro context neutral.": "宏观环境中性。",
        "Risk appetite elevated; avoid aggressive sizing.": "市场风险偏好抬升，避免激进加仓。",
        "Market stress elevated; keep size defensive.": "市场压力较大，仓位保持防守。",
    }
    prefix_map = {
        "CoinDesk headlines: ": "CoinDesk 头条：",
        "Cointelegraph headlines: ": "Cointelegraph 头条：",
        "Jin10 headlines: ": "金十头条：",
        "CryptoPanic headlines: ": "CryptoPanic 头条：",
        "Glassnode exchange balance=": "Glassnode 交易所余额=",
        "CoinMetrics ": "CoinMetrics ",
    }
    translated_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        translated = known.get(stripped, stripped)
        if translated == stripped:
            for prefix, zh_prefix in prefix_map.items():
                if stripped.startswith(prefix):
                    translated = zh_prefix + stripped[len(prefix) :]
                    break
        translated_lines.append(str(display_value(translated)))
    return "\n".join(translated_lines)


def _translate_report_label(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return text
    if text in REPORT_TITLE_ZH:
        return REPORT_TITLE_ZH[text]
    canonical = canonical_metric_key(text)
    if canonical in CANONICAL_TO_ZH_METRIC:
        return CANONICAL_TO_ZH_METRIC[canonical]
    translated = t(text)
    return translated if translated != text else text


def localize_report_text(content: str) -> str:
    text = str(content or "")
    if current_language() != "zh" or not text.strip():
        return text
    localized_lines: list[str] = []
    for line in text.splitlines():
        indent = line[: len(line) - len(line.lstrip())]
        body = line[len(indent) :]
        if body.startswith("# "):
            localized_lines.append(f"{indent}# {_translate_report_label(body[2:])}")
            continue
        if body.startswith("## "):
            localized_lines.append(f"{indent}## {_translate_report_label(body[3:])}")
            continue
        if body.startswith("- ") and ": " in body:
            key, value = body[2:].split(": ", 1)
            localized_lines.append(
                f"{indent}- {_translate_report_label(key)}: {display_value(value)}"
            )
            continue
        localized_lines.append(indent + display_research_text(body))
    return "\n".join(localized_lines)


def detect_summary_source(text: str, kind: str) -> tuple[str, bool]:
    content = (text or "").strip()
    lower = content.lower()
    if kind == "news":
        if lower.startswith("cointelegraph headlines:"):
            return "Cointelegraph", False
        if lower.startswith("coindesk headlines:"):
            return "CoinDesk", False
        if lower.startswith("jin10 headlines:"):
            return "Jin10", False
        if lower.startswith("cryptopanic headlines:"):
            return "CryptoPanic", False
        return "fallback", True
    if kind == "onchain":
        if lower.startswith("coinmetrics "):
            return "CoinMetrics", False
        if lower.startswith("glassnode "):
            return "Glassnode", False
        return "fallback", True
    if kind == "macro":
        return "Macro", False
    return "-", True


def db_path() -> str:
    settings = get_settings()
    path = Path(settings.app.db_path)
    if path.is_absolute():
        return str(path)
    return str(Path(settings.app.project_root) / path)


def python_executable() -> str:
    project_root = Path(get_settings().app.project_root)
    # Check for both Windows and Linux venv paths
    windows_venv = project_root / ".venv" / "Scripts" / "python.exe"
    linux_venv = project_root / ".venv" / "bin" / "python"
    
    if windows_venv.exists():
        return str(windows_venv)
    if linux_venv.exists():
        return str(linux_venv)
    return sys.executable


def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    return dashboard_query_df(sql, params, db_path=db_path())


def query_one(sql: str, params: tuple = ()) -> dict | None:
    return dashboard_query_one(sql, params, db_path=db_path())


def execute_sql(sql: str, params: tuple = ()) -> None:
    dashboard_execute_sql(sql, params, db_path=db_path())


def load_json(value: str | None, default=None):
    return dashboard_load_json(value, default)


def get_state_row(key: str) -> dict | None:
    return dashboard_get_state_row(key, db_path=db_path())


def get_state_json(key: str, default=None):
    return dashboard_get_state_json(key, default, db_path=db_path())


def runtime_state_snapshot():
    return DashboardRuntimeStateView(db_path()).snapshot()


def page_context() -> DashboardPageContext:
    return DashboardPageContext(
        st=st,
        t=t,
        query_df=query_df,
        query_one=query_one,
        get_state_json=get_state_json,
        get_state_row=get_state_row,
        set_state_json=set_state_json,
        load_json=load_json,
        parse_markdown_metrics=parse_markdown_metrics,
        parse_report_history=parse_report_history,
        to_numeric_percent=to_numeric_percent,
        display_df=display_df,
        display_json=display_json,
        display_kv_rows=display_kv_rows,
        display_value=display_value,
        display_research_text=display_research_text,
        detect_summary_source=detect_summary_source,
        current_language=current_language,
        localize_report_text=localize_report_text,
        run_command=run_command,
        parse_symbol_text=parse_symbol_text,
        get_settings_fn=get_settings,
        runtime_setting_defaults=runtime_setting_defaults,
        build_runtime_override_payload=build_runtime_override_payload,
        runtime_state_snapshot=runtime_state_snapshot,
    )


def set_state_json(key: str, value) -> None:
    dashboard_set_state_json(key, value, db_path=db_path())


def parse_symbol_text(value: str) -> list[str]:
    parsed: list[str] = []
    for raw in (value or "").replace("\n", ",").split(","):
        symbol = raw.strip().upper().replace(" ", "")
        if not symbol:
            continue
        if "-" in symbol and "/" not in symbol:
            parts = symbol.split("-")
            if len(parts) >= 2:
                symbol = f"{parts[0]}/{parts[1]}"
        if symbol.endswith("USDT") and "/" not in symbol and len(symbol) > 4:
            symbol = f"{symbol[:-4]}/USDT"
        if symbol not in parsed:
            parsed.append(symbol)
    return parsed


def parse_markdown_metrics(content: str) -> dict[str, str]:
    return parse_report_metrics(content)


def parse_report_history(report_type: str, limit: int = 30) -> pd.DataFrame:
    return dashboard_parse_report_history(report_type, limit=limit, db_path=db_path())


def to_numeric_percent(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    )


def parse_event_payloads(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "payload_json" not in events.columns:
        return events
    parsed = events.copy()
    payloads = parsed["payload_json"].apply(
        lambda value: json.loads(value) if value else {}
    )
    parsed["reason"] = payloads.apply(lambda payload: payload.get("reason", ""))
    parsed["exchange_order_id"] = payloads.apply(
        lambda payload: payload.get("exchange_order_id", "")
    )
    parsed["execution_type"] = payloads.apply(
        lambda payload: payload.get("execution_type", "")
    )
    parsed["cooldown_until"] = payloads.apply(
        lambda payload: payload.get("cooldown_until", "")
    )
    return parsed


def runtime_setting_defaults() -> dict[str, float | list[float]]:
    settings = get_settings()
    return {
        "xgboost_probability_threshold": float(
            settings.model.xgboost_probability_threshold
        ),
        "final_score_threshold": float(settings.model.final_score_threshold),
        "min_liquidity_ratio": float(settings.strategy.min_liquidity_ratio),
        "sentiment_weight": float(settings.strategy.sentiment_weight),
        "fixed_stop_loss_pct": float(settings.strategy.fixed_stop_loss_pct),
        "take_profit_levels": list(settings.strategy.take_profit_levels),
    }


def is_low_resource_mode() -> bool:
    return bool(get_settings().app.low_resource_mode)


def scheduler_job_options(low_resource_mode: bool) -> list[str]:
    options = [
        "once",
        "train",
        "walkforward",
        "report",
        "health",
        "guards",
        "abtest",
        "drift",
        "metrics",
        "reconcile",
        "ops",
    ]
    if not low_resource_mode:
        return options
    disabled_jobs = {"train", "walkforward", "abtest", "drift"}
    return [job for job in options if job not in disabled_jobs]


def normalize_take_profit_levels(value: str | list[float] | None) -> list[float]:
    if isinstance(value, list):
        items = value
    else:
        items = [part.strip() for part in str(value or "").split(",")]
    levels: list[float] = []
    for item in items:
        if item in ("", None):
            continue
        levels.append(float(item))
    normalized = sorted(level for level in levels if level > 0)
    if not normalized:
        raise ValueError("take_profit_levels cannot be empty")
    return normalized


def build_runtime_override_payload(values: dict[str, float | list[float]]) -> dict[str, float | list[float]]:
    defaults = runtime_setting_defaults()
    payload: dict[str, float | list[float]] = {}
    for key, value in values.items():
        if key == "take_profit_levels":
            normalized = normalize_take_profit_levels(value)
            if normalized != defaults[key]:
                payload[key] = normalized
            continue
        numeric = float(value)
        if abs(numeric - float(defaults[key])) > 1e-9:
            payload[key] = numeric
    return payload


def log_files() -> list[Path]:
    log_dir = Path(get_settings().app.project_root) / "data" / "logs"
    if not log_dir.exists():
        return []
    return sorted(
        log_dir.glob("*.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def tail_text(path: Path, line_count: int) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def command_timeout_seconds(command_name: str) -> int:
    timeout_map = {
        "train": 3600,
        "walkforward": 3600,
        "backfill": 3600,
        "backtest": 3600,
        "execution-set": 3600,
        "execution-add": 3600,
        "execution-remove": 900,
        "report": 900,
        "ops": 900,
        "metrics": 900,
        "guards": 900,
        "health": 600,
        "reconcile": 600,
        "schedule": 3600,
    }
    return timeout_map.get(command_name, 300)


def run_command(*args: str) -> tuple[bool, str]:
    command = [python_executable(), "main.py", *args]
    timeout_seconds = command_timeout_seconds(args[0]) if args else 300
    try:
        result = subprocess.run(
            command,
            cwd=str(get_settings().app.project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return False, str(exc)

    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return result.returncode == 0, output.strip()


def render_overview():
    _call_page_renderer(
        render_overview_page,
        page_ctx=page_context(),
    )


def render_settings():
    _call_page_renderer(
        render_settings_page,
        page_ctx=page_context(),
    )


def render_ops():
    _call_page_renderer(
        render_ops_page,
        page_ctx=page_context(),
    )


def render_predictions():
    _call_page_renderer(
        render_predictions_page,
        page_ctx=page_context(),
    )


def render_watchlist():
    _call_page_renderer(
        render_watchlist_page,
        page_ctx=page_context(),
        parse_event_payloads=parse_event_payloads,
        watchlist_snapshot_key=WATCHLIST_SNAPSHOT_KEY,
        execution_symbols_key=EXECUTION_SYMBOLS_KEY,
        watchlist_whitelist_key=WATCHLIST_WHITELIST_KEY,
        watchlist_blacklist_key=WATCHLIST_BLACKLIST_KEY,
    )


def render_training():
    _call_page_renderer(
        render_training_page,
        st=st,
        t=t,
        is_low_resource_mode=is_low_resource_mode,
        query_df=query_df,
        run_command=run_command,
        display_df=display_df,
        display_json=display_json,
        get_settings_fn=get_settings,
    )


def render_walkforward():
    _call_page_renderer(
        render_walkforward_page,
        st=st,
        t=t,
        is_low_resource_mode=is_low_resource_mode,
        run_command=run_command,
        query_df=query_df,
        display_df=display_df,
        get_settings_fn=get_settings,
    )


def render_backtest():
    _call_page_renderer(
        render_backtest_page,
        st=st,
        t=t,
        is_low_resource_mode=is_low_resource_mode,
        run_command=run_command,
        query_df=query_df,
        display_df=display_df,
        get_settings_fn=get_settings,
    )


def render_reports():
    _call_page_renderer(
        render_reports_page,
        st=st,
        t=t,
        run_command=run_command,
        query_df=query_df,
        query_one=query_one,
        display_df=display_df,
        display_json=display_json,
        load_json=load_json,
        localize_report_text=localize_report_text,
    )


def _call_page_renderer(renderer, /, **kwargs):
    signature = inspect.signature(renderer)
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return renderer(**kwargs)
    expanded = dict(kwargs)
    page_ctx = expanded.get("page_ctx")
    if page_ctx is not None and "page_ctx" not in signature.parameters:
        for name in signature.parameters:
            if name in expanded:
                continue
            if hasattr(page_ctx, name):
                expanded[name] = getattr(page_ctx, name)
    filtered = {
        key: value
        for key, value in expanded.items()
        if key in signature.parameters
    }
    return renderer(**filtered)


def render_execution():
    st.title(t("Execution Audit"))
    orders = query_df(
        "SELECT * FROM orders ORDER BY updated_at DESC LIMIT 100"
    )
    events = query_df(
        "SELECT * FROM execution_events ORDER BY created_at DESC LIMIT 100"
    )
    reconciliations = query_df(
        "SELECT * FROM reconciliation_runs ORDER BY created_at DESC LIMIT 50"
    )
    cycles = query_df(
        "SELECT * FROM cycle_runs ORDER BY created_at DESC LIMIT 50"
    )
    st.subheader(t("Orders"))
    if orders.empty:
        st.info(t("No orders recorded."))
    else:
        st.dataframe(display_df(orders), use_container_width=True)

    st.subheader(t("Execution Events"))
    if events.empty:
        st.info(t("No execution events recorded."))
    else:
        st.dataframe(display_df(events), use_container_width=True)

    st.subheader(t("Reconciliation Runs"))
    if reconciliations.empty:
        st.info(t("No reconciliation runs recorded."))
    else:
        st.dataframe(display_df(reconciliations), use_container_width=True)

    st.subheader(t("Cycle Runs"))
    if cycles.empty:
        st.info(t("No cycle runs recorded."))
    else:
        st.dataframe(display_df(cycles), use_container_width=True)

    latest_account = query_one(
        "SELECT * FROM account_snapshots ORDER BY created_at DESC LIMIT 1"
    )
    if latest_account:
        st.subheader(t("Latest Risk State"))
        st.json(display_json(latest_account))


def render_health():
    st.title(t("Health"))
    if st.button(t("Run Health Check"), type="primary", use_container_width=True):
        ok, output = run_command("health")
        st.code(output or t("(no output)"))
        if ok:
            st.success(t("Health check finished."))
        else:
            st.error(t("Health check failed."))

    latest_health = query_df(
        "SELECT * FROM report_artifacts WHERE report_type='health' ORDER BY created_at DESC LIMIT 10"
    )
    if latest_health.empty:
        st.info(t("No health reports recorded."))
    else:
        st.dataframe(
            display_df(latest_health[["report_type", "created_at"]]),
            use_container_width=True,
        )
        st.code(localize_report_text(latest_health.iloc[0]["content"]))


def render_metrics():
    st.title(t("Performance Metrics"))
    if st.button(t("Generate Metrics"), type="primary", use_container_width=True):
        ok, output = run_command("metrics")
        st.code(output or t("(no output)"))
        if ok:
            st.success(t("Metrics generated."))
        else:
            st.error(t("Metrics failed."))

    latest_performance = query_df(
        "SELECT * FROM report_artifacts WHERE report_type='performance' ORDER BY created_at DESC LIMIT 10"
    )
    latest_account = query_one(
        "SELECT * FROM account_snapshots ORDER BY created_at DESC LIMIT 1"
    )
    latest_training = query_one(
        "SELECT metadata_json FROM training_runs ORDER BY created_at DESC LIMIT 1"
    )
    latest_walkforward = query_one(
        "SELECT summary_json FROM walkforward_runs ORDER BY created_at DESC LIMIT 1"
    )
    latest_performance_row = latest_performance.iloc[0].to_dict() if not latest_performance.empty else None
    performance_metrics = parse_markdown_metrics(latest_performance_row["content"]) if latest_performance_row else {}
    performance_history = parse_report_history("performance", limit=30)
    history_df = pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        t("Equity"),
        f"${latest_account['equity']:,.2f}" if latest_account else display_value("N/A"),
    )
    c2.metric(
        t("Drawdown"),
        f"{latest_account['drawdown_pct'] * 100:.2f}%"
        if latest_account else display_value("N/A"),
    )
    c3.metric(
        t("Holdout Accuracy"),
        f"{json.loads(latest_training['metadata_json']).get('holdout_accuracy', 0.0) * 100:.2f}%"
        if latest_training else display_value("N/A"),
    )
    c4.metric(
        t("Latest WF Return"),
        f"{json.loads(latest_walkforward['summary_json']).get('total_return_pct', 0.0):.2f}%"
        if latest_walkforward else display_value("N/A"),
    )
    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        t("Evaluated Predictions"),
        performance_metrics.get("Evaluated Predictions", "0"),
    )
    c6.metric(
        t("XGB Accuracy"),
        display_value(performance_metrics.get("XGBoost Direction Accuracy", "N/A")),
    )
    c7.metric(
        t("LLM Accuracy"),
        display_value(performance_metrics.get("LLM Action Accuracy", "N/A")),
    )
    c8.metric(
        t("Fusion Accuracy"),
        display_value(performance_metrics.get("Fusion Signal Accuracy", "N/A")),
    )
    c9, c10, c11 = st.columns(3)
    c9.metric(
        t("Degradation Status"),
        display_value(performance_metrics.get("Degradation Status", "N/A")),
    )
    c10.metric(
        t("Runtime XGB Threshold"),
        display_value(performance_metrics.get("Recommended XGB Threshold", "N/A")),
    )
    c11.metric(
        t("Runtime Final Threshold"),
        display_value(performance_metrics.get("Recommended Final Threshold", "N/A")),
    )

    if latest_performance.empty:
        st.info(t("No performance reports recorded."))
    else:
        st.dataframe(
            display_df(latest_performance[["report_type", "created_at"]]),
            use_container_width=True,
        )
        st.code(localize_report_text(latest_performance.iloc[0]["content"]))
        if not performance_history.empty:
            st.subheader(t("Performance History"))
            history_df = performance_history[["created_at"]].copy()
            for source_label, metric_key in (
                (t("XGB Accuracy"), "XGBoost Direction Accuracy"),
                (t("LLM Accuracy"), "LLM Action Accuracy"),
                (t("Fusion Accuracy"), "Fusion Signal Accuracy"),
            ):
                if metric_key in performance_history:
                    history_df[source_label] = to_numeric_percent(
                        performance_history[metric_key]
                    )
            if "Latest Walk-Forward Return" in performance_history:
                history_df[t("WF Return")] = to_numeric_percent(
                    performance_history["Latest Walk-Forward Return"]
                )
            history_df = history_df.set_index("created_at").dropna(how="all")
        if not history_df.empty:
            st.line_chart(history_df, use_container_width=True)


def render_drift():
    st.title(t("Drift Report"))
    low_resource_mode = is_low_resource_mode()
    if st.button(
        t("Generate Drift Report"),
        type="primary",
        use_container_width=True,
        disabled=low_resource_mode,
    ):
        ok, output = run_command("drift")
        st.code(output or t("(no output)"))
        if ok:
            st.success(t("Drift report generated."))
        else:
            st.error(t("Drift report failed."))
    if low_resource_mode:
        st.warning(t("Low resource mode is enabled. Drift generation is disabled in the dashboard on this node."))

    latest_drift = query_df(
        "SELECT * FROM report_artifacts WHERE report_type='drift' ORDER BY created_at DESC LIMIT 10"
    )
    latest_backtest = query_df(
        "SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT 10"
    )
    latest_walkforward = query_df(
        "SELECT * FROM walkforward_runs ORDER BY created_at DESC LIMIT 10"
    )

    if latest_drift.empty:
        st.info(t("No drift reports recorded."))
    else:
        st.dataframe(display_df(latest_drift[["report_type", "created_at"]]), use_container_width=True)
        st.code(localize_report_text(latest_drift.iloc[0]["content"]))

    left, right = st.columns(2)
    with left:
        st.subheader(t("Recent Backtests"))
        if latest_backtest.empty:
            st.info(t("No backtest runs."))
        else:
            st.dataframe(display_df(latest_backtest[["symbol", "created_at"]]), use_container_width=True)
    with right:
        st.subheader(t("Recent Walk-Forward Runs"))
        if latest_walkforward.empty:
            st.info(t("No walk-forward runs."))
        else:
            st.dataframe(display_df(latest_walkforward[["symbol", "created_at"]]), use_container_width=True)


def render_abtest():
    st.title(t("AB Test"))
    low_resource_mode = is_low_resource_mode()
    if st.button(
        t("Generate AB Test Report"),
        type="primary",
        use_container_width=True,
        disabled=low_resource_mode,
    ):
        ok, output = run_command("abtest")
        st.code(output or t("(no output)"))
        if ok:
            st.success(t("AB test report generated."))
        else:
            st.error(t("AB test report failed."))
    if low_resource_mode:
        st.warning(t("Low resource mode is enabled. AB test reporting is disabled in the dashboard on this node."))

    ab_runs = query_df(
        "SELECT * FROM ab_test_runs ORDER BY created_at DESC LIMIT 50"
    )
    ab_reports = query_df(
        "SELECT * FROM report_artifacts WHERE report_type='ab_test' ORDER BY created_at DESC LIMIT 10"
    )
    if ab_runs.empty:
        st.info(t("No A/B test runs recorded."))
    else:
        st.dataframe(
            display_df(ab_runs[
                [
                    "symbol",
                    "champion_model_version",
                    "challenger_model_version",
                    "champion_probability",
                    "challenger_probability",
                    "selected_variant",
                    "allocation_pct",
                    "created_at",
                ]
            ]),
            use_container_width=True,
        )

    if not ab_reports.empty:
        st.subheader(t("Latest AB Test Report"))
        st.code(localize_report_text(ab_reports.iloc[0]["content"]))


def render_guards():
    st.title(t("Guards"))
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button(t("Generate Guard Report"), type="primary", use_container_width=True):
            ok, output = run_command("guards")
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Guard report generated."))
            else:
                st.error(t("Guard report failed."))
    with action_col2:
        if st.button(t("Approve Recovery"), use_container_width=True):
            ok, output = run_command("approve-recovery")
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Manual recovery approved."))
            else:
                st.error(t("Manual recovery approval failed."))

    latest_guard = query_df(
        "SELECT * FROM report_artifacts WHERE report_type='guard' ORDER BY created_at DESC LIMIT 10"
    )
    guard_events = query_df(
        "SELECT * FROM execution_events WHERE event_type IN ('model_accuracy_guard','model_degradation','api_failure','abnormal_move','live_order_timeout','live_open_limit_timeout','live_close_limit_timeout') ORDER BY created_at DESC LIMIT 50"
    )
    guard_events = parse_event_payloads(guard_events)
    latest_account = query_one(
        "SELECT * FROM account_snapshots ORDER BY created_at DESC LIMIT 1"
    )
    model_degradation_status = query_one(
        "SELECT value, updated_at FROM system_state WHERE key='model_degradation_status'"
    )
    model_degradation_reason = query_one(
        "SELECT value, updated_at FROM system_state WHERE key='model_degradation_reason'"
    )
    last_accuracy_guard_triggered = query_one(
        "SELECT value, updated_at FROM system_state WHERE key='last_accuracy_guard_triggered'"
    )
    last_accuracy_guard_reason = query_one(
        "SELECT value, updated_at FROM system_state WHERE key='last_accuracy_guard_reason'"
    )
    manual_recovery_required = query_one(
        "SELECT value, updated_at FROM system_state WHERE key='manual_recovery_required'"
    )
    manual_recovery_reason = query_one(
        "SELECT value, updated_at FROM system_state WHERE key='manual_recovery_reason'"
    )
    guard_history = query_df(
        "SELECT event_type, created_at FROM execution_events WHERE event_type IN ('model_accuracy_guard','model_degradation','api_failure','abnormal_move','live_order_timeout','live_open_limit_timeout','live_close_limit_timeout') ORDER BY created_at DESC LIMIT 200"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        t("Recent Guard Events"),
        str(len(guard_events.index)) if not guard_events.empty else "0",
    )
    c2.metric(
        t("Circuit Breaker"),
        t("ACTIVE") if latest_account and latest_account["circuit_breaker_active"] else t("OFF"),
    )
    c3.metric(
        t("Cooldown"),
        latest_account.get("cooldown_until", "none") if latest_account else "none",
    )
    c4, c5 = st.columns(2)
    c4.metric(
        t("Last Accuracy Guard"),
        last_accuracy_guard_triggered["value"] if last_accuracy_guard_triggered else "none",
    )
    c5.metric(
        t("Guard Reason"),
        last_accuracy_guard_reason["value"] if last_accuracy_guard_reason else "none",
    )
    c6, c7 = st.columns(2)
    c6.metric(
        t("Model Degradation"),
        model_degradation_status["value"] if model_degradation_status else "healthy",
    )
    c7.metric(
        t("Degradation Reason"),
        model_degradation_reason["value"] if model_degradation_reason else "none",
    )
    c8, c9 = st.columns(2)
    c8.metric(
        t("Manual Recovery"),
        manual_recovery_required["value"] if manual_recovery_required else "false",
    )
    c9.metric(
        t("Recovery Reason"),
        manual_recovery_reason["value"] if manual_recovery_reason else "none",
    )

    st.subheader(t("Guard Report Artifacts"))
    if latest_guard.empty:
        st.info(t("No guard reports recorded."))
    else:
        st.dataframe(display_df(latest_guard[["report_type", "created_at"]]), use_container_width=True)
        st.code(localize_report_text(latest_guard.iloc[0]["content"]))

    st.subheader(t("Guard Execution Events"))
    if guard_events.empty:
        st.info(t("No guard execution events."))
    else:
        st.dataframe(
            display_df(guard_events[
                ["event_type", "symbol", "created_at", "reason", "exchange_order_id", "execution_type"]
            ]),
            use_container_width=True,
        )
    if not guard_history.empty:
        st.subheader(t("Guard Event Trend"))
        guard_history["created_at"] = pd.to_datetime(guard_history["created_at"], errors="coerce")
        pivot = (
            guard_history.assign(count=1)
            .pivot_table(
                index="created_at",
                columns="event_type",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            .sort_index()
        )
        if not pivot.empty:
            st.bar_chart(pivot, use_container_width=True)


def render_failures():
    st.title(t("Failures"))
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(t("Generate Failure Report"), type="primary", use_container_width=True):
            ok, output = run_command("failures")
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Failure report generated."))
            else:
                st.error(t("Failure report failed."))
    with col2:
        if st.button(t("Run Maintenance"), type="primary", use_container_width=True):
            ok, output = run_command("maintenance")
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Maintenance completed."))
            else:
                st.error(t("Maintenance failed."))
    with col3:
        if st.button(t("Generate Incident Report"), type="primary", use_container_width=True):
            ok, output = run_command("incidents")
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Incident report generated."))
            else:
                st.error(t("Incident report failed."))

    failure_reports = query_df(
        "SELECT * FROM report_artifacts WHERE report_type IN ('failure', 'maintenance', 'incident') ORDER BY created_at DESC LIMIT 20"
    )
    if failure_reports.empty:
        st.info(t("No failure or maintenance reports."))
    else:
        st.dataframe(
            display_df(failure_reports[["report_type", "created_at"]]),
            use_container_width=True,
        )
        st.code(localize_report_text(failure_reports.iloc[0]["content"]))


def render_scheduler():
    st.title(t("Scheduler"))
    low_resource_mode = is_low_resource_mode()
    job_options = scheduler_job_options(low_resource_mode)
    col1, col2 = st.columns(2)
    with col1:
        job_name = st.selectbox(
            t("Run Scheduled Job"),
            job_options,
        )
    with col2:
        if st.button(t("Run Job"), type="primary", use_container_width=True):
            ok, output = run_command("schedule", job_name)
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Scheduled job finished."))
            else:
                st.error(t("Scheduled job failed."))
    if low_resource_mode:
        st.warning(t("Low resource mode is enabled. Heavy jobs are hidden from the scheduler UI on this node."))

    runs = query_df(
        "SELECT * FROM scheduler_runs ORDER BY created_at DESC LIMIT 50"
    )
    if runs.empty:
        st.info(t("No scheduler runs recorded."))
    else:
        st.dataframe(display_df(runs), use_container_width=True)


def render_logs():
    st.title(t("Log Viewer"))
    files = log_files()
    if not files:
        st.info(t("No log files found in data/logs."))
        return

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        selected_log = st.selectbox(
            t("Log File"),
            options=files,
            format_func=lambda path: path.name,
        )
    with col2:
        line_count = st.number_input(
            t("Tail Lines"),
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
        )
    with col3:
        if st.button(t("Refresh"), use_container_width=True):
            st.rerun()

    st.caption(
        f"{t('Tail Lines')}: {int(line_count)} | {selected_log.name} | size={selected_log.stat().st_size} bytes"
    )
    content = tail_text(selected_log, int(line_count)) or t("(empty log)")
    st.text_area(
        t("Log File"),
        value=content,
        height=520,
        label_visibility="collapsed",
    )


def render_operations():
    st.title(t("Operations"))
    low_resource_mode = is_low_resource_mode()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button(t("Run Analysis Once"), use_container_width=True):
            ok, output = run_command("once")
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Analysis finished."))
            else:
                st.error(t("Analysis failed."))
    with col2:
        if st.button(
            t("Run Backfill 180d"),
            use_container_width=True,
            disabled=low_resource_mode,
        ):
            ok, output = run_command("backfill", "180")
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Backfill finished."))
            else:
                st.error(t("Backfill failed."))
    with col3:
        if st.button(t("Run Reconciliation"), use_container_width=True):
            ok, output = run_command("reconcile")
            st.code(output or t("(no output)"))
            if ok:
                st.success(t("Reconciliation finished."))
            else:
                st.error(t("Reconciliation failed."))
    with col4:
        st.info(t("Loop mode should be started from terminal or process manager.\nDashboard only supports one-shot commands."))
    if low_resource_mode:
        st.warning(t("Low resource mode is enabled. Backfill is disabled in the operations page on this node."))

    account_history = query_df(
        "SELECT * FROM account_snapshots ORDER BY created_at DESC LIMIT 50"
    )
    if not account_history.empty:
        st.subheader(t("Recent Account Snapshots"))
        st.dataframe(display_df(account_history), use_container_width=True)


def main():
    with st.sidebar:
        st.title("CryptoAI v3")
        st.caption(t("LLM-enhanced systematic crypto trading"))
        language = st.selectbox(
            t("Language"),
            ["zh", "en"],
            index=0 if current_language() == "zh" else 1,
            format_func=lambda value: t("Chinese") if value == "zh" else t("English"),
        )
        if language != current_language():
            set_language(language)
            st.rerun()
        page = st.radio(
            t("Pages"),
            [
                "Overview",
                "Settings",
                "Predictions",
                "Ops",
            ],
            format_func=t,
        )
        st.divider()
        st.caption(
            f"UTC {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        st.caption(f"DB: {db_path()}")

    if page == "Overview":
        render_overview()
    elif page == "Settings":
        render_settings()
    elif page == "Predictions":
        render_predictions()
    elif page == "Ops":
        render_ops()


if __name__ == "__main__":
    main()
