"""Shared dashboard page context."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DashboardPageContext:
    st: Any
    t: Callable[[str], str]
    query_df: Callable[..., Any]
    query_one: Callable[..., Any]
    get_state_json: Callable[..., Any]
    get_state_row: Callable[..., Any]
    set_state_json: Callable[..., Any]
    load_json: Callable[..., Any]
    parse_markdown_metrics: Callable[..., Any]
    parse_report_history: Callable[..., Any]
    to_numeric_percent: Callable[..., Any]
    display_df: Callable[..., Any]
    display_json: Callable[..., Any]
    display_kv_rows: Callable[..., Any]
    display_value: Callable[..., Any]
    display_research_text: Callable[..., Any]
    detect_summary_source: Callable[..., Any]
    current_language: Callable[..., Any]
    localize_report_text: Callable[..., Any]
    run_command: Callable[..., Any]
    parse_symbol_text: Callable[..., Any]
    get_settings_fn: Callable[..., Any]
    runtime_setting_defaults: Callable[..., Any]
    build_runtime_override_payload: Callable[..., Any]
    runtime_state_snapshot: Callable[..., Any]
