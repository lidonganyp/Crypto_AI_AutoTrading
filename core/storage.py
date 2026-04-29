"""Database storage layer using SQLite"""
from __future__ import annotations

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Any

from loguru import logger


class Storage:
    RUNTIME_TABLES = [
        "ohlcv",
        "trades",
        "orders",
        "signals",
        "sentiment",
        "reflections",
        "positions",
        "feature_snapshots",
        "prediction_runs",
        "prediction_evaluations",
        "shadow_trade_runs",
        "pnl_ledger",
        "account_snapshots",
        "training_runs",
        "walkforward_runs",
        "execution_events",
        "report_artifacts",
        "reconciliation_runs",
        "research_inputs",
        "backtest_runs",
        "cycle_runs",
        "scheduler_runs",
        "ab_test_runs",
        "model_registry",
        "model_scorecards",
    ]

    def __init__(self, db_path: str = "data/cryptoai.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    initial_quantity REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    pnl REAL,
                    pnl_pct REAL,
                    rationale TEXT,
                    confidence REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT DEFAULT 'open'
                );

                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    rationale TEXT,
                    risk_level TEXT,
                    raw_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    value REAL,
                    label TEXT,
                    summary TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reflections (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    direction TEXT,
                    confidence REAL,
                    rationale TEXT,
                    source TEXT,
                    experience_weight REAL,
                    realized_return_pct REAL,
                    outcome_24h REAL,
                    outcome_7d REAL,
                    correct_signals TEXT,
                    wrong_signals TEXT,
                    lesson TEXT,
                    market_regime TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    valid INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prediction_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    up_probability REAL NOT NULL,
                    feature_count INTEGER NOT NULL,
                    research_json TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prediction_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    evaluation_type TEXT NOT NULL,
                    actual_up INTEGER NOT NULL,
                    predicted_up INTEGER NOT NULL,
                    is_correct INTEGER NOT NULL,
                    entry_close REAL NOT NULL,
                    future_close REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (symbol, timestamp, evaluation_type)
                );

                CREATE TABLE IF NOT EXISTS shadow_trade_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    block_reason TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    horizon_hours INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    exit_price REAL,
                    pnl_pct REAL,
                    setup_profile_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    evaluated_at TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE (symbol, timestamp, block_reason)
                );

                CREATE TABLE IF NOT EXISTS pnl_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    notional_value REAL NOT NULL,
                    reference_price REAL NOT NULL,
                    fill_price REAL NOT NULL,
                    gross_pnl REAL NOT NULL,
                    fee_cost REAL NOT NULL,
                    slippage_cost REAL NOT NULL,
                    net_pnl REAL NOT NULL,
                    net_return_pct REAL NOT NULL,
                    holding_hours REAL NOT NULL,
                    model_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS account_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    daily_loss_pct REAL NOT NULL,
                    weekly_loss_pct REAL NOT NULL,
                    drawdown_pct REAL NOT NULL,
                    total_exposure_pct REAL NOT NULL,
                    open_positions INTEGER NOT NULL,
                    cooldown_until TEXT,
                    circuit_breaker_active INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    rows INTEGER NOT NULL,
                    feature_count INTEGER NOT NULL,
                    positives INTEGER NOT NULL,
                    negatives INTEGER NOT NULL,
                    model_path TEXT NOT NULL,
                    trained_with_xgboost INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS walkforward_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    total_splits INTEGER NOT NULL,
                    summary_json TEXT NOT NULL,
                    splits_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS execution_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS report_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_type TEXT NOT NULL,
                    symbol TEXT,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reconciliation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    mismatch_count INTEGER NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_inputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    news_summary TEXT NOT NULL,
                    macro_summary TEXT NOT NULL,
                    fear_greed REAL,
                    onchain_summary TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    trades_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cycle_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    symbols_json TEXT NOT NULL,
                    opened_positions INTEGER NOT NULL,
                    closed_positions INTEGER NOT NULL,
                    circuit_breaker_active INTEGER NOT NULL,
                    reconciliation_status TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scheduler_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ab_test_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    champion_model_version TEXT NOT NULL,
                    challenger_model_version TEXT NOT NULL,
                    champion_probability REAL NOT NULL,
                    challenger_probability REAL NOT NULL,
                    champion_execute INTEGER NOT NULL,
                    challenger_execute INTEGER NOT NULL,
                    selected_variant TEXT NOT NULL,
                    allocation_pct REAL NOT NULL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_id TEXT NOT NULL UNIQUE,
                    model_version TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    role TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    active INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_scorecards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    evaluation_type TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    executed_count INTEGER NOT NULL,
                    accuracy REAL NOT NULL,
                    executed_precision REAL NOT NULL,
                    avg_trade_return_pct REAL NOT NULL,
                    total_trade_return_pct REAL NOT NULL,
                    expectancy_pct REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    max_drawdown_pct REAL NOT NULL,
                    trade_win_rate REAL NOT NULL,
                    avg_cost_pct REAL NOT NULL,
                    avg_favorable_excursion_pct REAL NOT NULL,
                    avg_adverse_excursion_pct REAL NOT NULL,
                    objective_score REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(trades)").fetchall()
            }
            if "initial_quantity" not in columns:
                conn.execute("ALTER TABLE trades ADD COLUMN initial_quantity REAL")
            if "metadata_json" not in columns:
                conn.execute(
                    "ALTER TABLE trades ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                )
            conn.execute(
                "UPDATE trades SET initial_quantity = quantity WHERE initial_quantity IS NULL"
            )
            conn.execute(
                "UPDATE trades SET metadata_json = '{}' "
                "WHERE metadata_json IS NULL OR TRIM(metadata_json) = ''"
            )
            reflection_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(reflections)").fetchall()
            }
            if "source" not in reflection_columns:
                conn.execute("ALTER TABLE reflections ADD COLUMN source TEXT")
            if "experience_weight" not in reflection_columns:
                conn.execute("ALTER TABLE reflections ADD COLUMN experience_weight REAL")
            if "realized_return_pct" not in reflection_columns:
                conn.execute("ALTER TABLE reflections ADD COLUMN realized_return_pct REAL")
                conn.execute(
                    "UPDATE reflections SET realized_return_pct = outcome_24h "
                    "WHERE realized_return_pct IS NULL"
                )
            model_scorecard_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(model_scorecards)").fetchall()
            }
            if "expectancy_pct" not in model_scorecard_columns:
                conn.execute(
                    "ALTER TABLE model_scorecards ADD COLUMN expectancy_pct REAL NOT NULL DEFAULT 0"
                )
            if "profit_factor" not in model_scorecard_columns:
                conn.execute(
                    "ALTER TABLE model_scorecards ADD COLUMN profit_factor REAL NOT NULL DEFAULT 0"
                )
            if "max_drawdown_pct" not in model_scorecard_columns:
                conn.execute(
                    "ALTER TABLE model_scorecards ADD COLUMN max_drawdown_pct REAL NOT NULL DEFAULT 0"
                )
            if "trade_win_rate" not in model_scorecard_columns:
                conn.execute(
                    "ALTER TABLE model_scorecards ADD COLUMN trade_win_rate REAL NOT NULL DEFAULT 0"
                )
            if "avg_cost_pct" not in model_scorecard_columns:
                conn.execute(
                    "ALTER TABLE model_scorecards ADD COLUMN avg_cost_pct REAL NOT NULL DEFAULT 0"
                )
            if "avg_favorable_excursion_pct" not in model_scorecard_columns:
                conn.execute(
                    "ALTER TABLE model_scorecards ADD COLUMN avg_favorable_excursion_pct REAL NOT NULL DEFAULT 0"
                )
            if "avg_adverse_excursion_pct" not in model_scorecard_columns:
                conn.execute(
                    "ALTER TABLE model_scorecards ADD COLUMN avg_adverse_excursion_pct REAL NOT NULL DEFAULT 0"
                )
            conn.execute(
                "UPDATE reflections SET source = COALESCE(source, CASE "
                "WHEN trade_id LIKE 'shadow:%' THEN 'shadow_observation' "
                "ELSE 'trade' END)"
            )
            conn.execute(
                "UPDATE reflections SET experience_weight = COALESCE(experience_weight, CASE "
                "WHEN trade_id LIKE 'shadow:%' THEN 0.35 "
                "ELSE 1.0 END)"
            )
            # The composite PRIMARY KEY on ohlcv already provides the lookup index.
            conn.execute("DROP INDEX IF EXISTS idx_ohlcv_symbol_tf")
            self._init_indexes(conn)
            logger.info(f"Database initialized: {self.db_path}")

    @staticmethod
    def _init_indexes(conn: sqlite3.Connection) -> None:
        """Create read-heavy indexes used by the runtime loop, dashboard, and reports."""
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_trades_status_exit_time
            ON trades (status, exit_time);

            CREATE INDEX IF NOT EXISTS idx_trades_symbol_status
            ON trades (symbol, status);

            CREATE INDEX IF NOT EXISTS idx_trades_symbol_exit_time
            ON trades (symbol, exit_time);

            CREATE INDEX IF NOT EXISTS idx_orders_status_updated_at
            ON orders (status, updated_at);

            CREATE INDEX IF NOT EXISTS idx_orders_symbol_updated_at
            ON orders (symbol, updated_at);

            CREATE INDEX IF NOT EXISTS idx_signals_symbol_created_at
            ON signals (symbol, created_at);

            CREATE INDEX IF NOT EXISTS idx_sentiment_source_created_at
            ON sentiment (source, created_at);

            CREATE INDEX IF NOT EXISTS idx_feature_snapshots_symbol_timestamp
            ON feature_snapshots (symbol, timestamp);

            CREATE INDEX IF NOT EXISTS idx_feature_snapshots_created_at
            ON feature_snapshots (created_at);

            CREATE INDEX IF NOT EXISTS idx_prediction_runs_symbol_timestamp
            ON prediction_runs (symbol, timestamp);

            CREATE INDEX IF NOT EXISTS idx_prediction_runs_created_at
            ON prediction_runs (created_at);

            CREATE INDEX IF NOT EXISTS idx_prediction_evaluations_type_created_at
            ON prediction_evaluations (evaluation_type, created_at);

            CREATE INDEX IF NOT EXISTS idx_prediction_evaluations_symbol_created_at
            ON prediction_evaluations (symbol, created_at);

            CREATE INDEX IF NOT EXISTS idx_shadow_trade_runs_status_created_at
            ON shadow_trade_runs (status, created_at);

            CREATE INDEX IF NOT EXISTS idx_shadow_trade_runs_symbol_status_timestamp
            ON shadow_trade_runs (symbol, status, timestamp);

            CREATE INDEX IF NOT EXISTS idx_pnl_ledger_trade_id
            ON pnl_ledger (trade_id);

            CREATE INDEX IF NOT EXISTS idx_pnl_ledger_symbol_event_time
            ON pnl_ledger (symbol, event_time);

            CREATE INDEX IF NOT EXISTS idx_pnl_ledger_model_event_time
            ON pnl_ledger (model_id, event_time);

            CREATE INDEX IF NOT EXISTS idx_account_snapshots_created_at
            ON account_snapshots (created_at);

            CREATE INDEX IF NOT EXISTS idx_training_runs_symbol_created_at
            ON training_runs (symbol, created_at);

            CREATE INDEX IF NOT EXISTS idx_walkforward_runs_symbol_created_at
            ON walkforward_runs (symbol, created_at);

            CREATE INDEX IF NOT EXISTS idx_execution_events_type_created_at
            ON execution_events (event_type, created_at);

            CREATE INDEX IF NOT EXISTS idx_execution_events_symbol_created_at
            ON execution_events (symbol, created_at);

            CREATE INDEX IF NOT EXISTS idx_report_artifacts_type_created_at
            ON report_artifacts (report_type, created_at);

            CREATE INDEX IF NOT EXISTS idx_report_artifacts_symbol_type_created_at
            ON report_artifacts (symbol, report_type, created_at);

            CREATE INDEX IF NOT EXISTS idx_reconciliation_runs_status_created_at
            ON reconciliation_runs (status, created_at);

            CREATE INDEX IF NOT EXISTS idx_research_inputs_symbol_timestamp
            ON research_inputs (symbol, timestamp);

            CREATE INDEX IF NOT EXISTS idx_backtest_runs_symbol_engine_created_at
            ON backtest_runs (symbol, engine, created_at);

            CREATE INDEX IF NOT EXISTS idx_cycle_runs_status_started_at
            ON cycle_runs (status, started_at);

            CREATE INDEX IF NOT EXISTS idx_scheduler_runs_job_status_started_at
            ON scheduler_runs (job_name, status, started_at);

            CREATE INDEX IF NOT EXISTS idx_ab_test_runs_symbol_timestamp
            ON ab_test_runs (symbol, timestamp);

            CREATE INDEX IF NOT EXISTS idx_model_registry_symbol_updated_at
            ON model_registry (symbol, updated_at);

            CREATE INDEX IF NOT EXISTS idx_model_registry_stage_updated_at
            ON model_registry (stage, updated_at);

            CREATE INDEX IF NOT EXISTS idx_model_scorecards_model_created_at
            ON model_scorecards (model_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_model_scorecards_symbol_stage_created_at
            ON model_scorecards (symbol, stage, created_at);
        """)

    # ── OHLCV ───────────────────────────────────────────

    def insert_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        candles: list[dict],
    ) -> int:
        """批量插入 K 线数据，返回插入条数"""
        rows = []
        for c in candles:
            rows.append((
                symbol, timeframe,
                c["timestamp"], c["open"], c["high"],
                c["low"], c["close"], c["volume"],
            ))
        with self._conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO ohlcv
                   (symbol, timeframe, timestamp, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            return len(rows)

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int = 500,
    ) -> list[dict]:
        with self._conn() as conn:
            q = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
            """
            params: list[Any] = [symbol, timeframe]
            if since:
                q += " AND timestamp >= ?"
                params.append(since)
            q += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(q, params).fetchall()
            return [dict(r) for r in rows]

    # ── Trades ──────────────────────────────────────────

    def insert_trade(self, trade: dict) -> str:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO trades
                   (id, symbol, direction, entry_price, quantity, initial_quantity,
                    entry_time, rationale, confidence, metadata_json, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade["id"], trade["symbol"], trade["direction"],
                    trade["entry_price"], trade["quantity"],
                    trade.get("initial_quantity", trade["quantity"]),
                    trade["entry_time"], trade.get("rationale", ""),
                    trade.get("confidence", 0),
                    json.dumps(trade.get("metadata", {}), default=str),
                    "open",
                ),
            )
            return trade["id"]

    def insert_order(self, order: dict) -> str:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO orders
                   (order_id, symbol, side, order_type, status, price, quantity,
                    reason, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order["order_id"],
                    order["symbol"],
                    order["side"],
                    order["order_type"],
                    order["status"],
                    order["price"],
                    order["quantity"],
                    order.get("reason", ""),
                    order["created_at"],
                    order["updated_at"],
                ),
            )
            return order["order_id"]

    def update_order_status(self, order_id: str, status: str, reason: str = ""):
        with self._conn() as conn:
            conn.execute(
                """UPDATE orders
                   SET status = ?, reason = ?, updated_at = ?
                   WHERE order_id = ?""",
                (
                    status,
                    reason,
                    datetime.now(timezone.utc).isoformat(),
                    order_id,
                ),
            )

    def get_orders(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM orders ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_trade_exit(self, trade_id: str, exit_price: float,
                          exit_time: str, pnl: float, pnl_pct: float):
        with self._conn() as conn:
            conn.execute(
                """UPDATE trades SET exit_price=?, exit_time=?,
                   pnl=?, pnl_pct=?, quantity=COALESCE(initial_quantity, quantity), status='closed'
                   WHERE id=?""",
                (exit_price, exit_time, pnl, pnl_pct, trade_id),
            )

    def upsert_trade_partial_close(
        self,
        trade_id: str,
        closed_qty: float,
        exit_price: float,
        exit_time: str,
        realized_pnl: float,
        realized_pnl_pct: float,
        remaining_qty: float,
    ):
        """部分平仓：累加已实现盈亏，更新剩余数量"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT pnl, quantity, initial_quantity, entry_price FROM trades WHERE id=?",
                (trade_id,),
            ).fetchone()
            if not row:
                return
            prev_pnl = row["pnl"] or 0
            entry_price = row["entry_price"]
            initial_quantity = row["initial_quantity"] or row["quantity"]
            total_pnl = prev_pnl + realized_pnl
            total_pnl_pct = (
                total_pnl / (initial_quantity * entry_price) * 100
                if initial_quantity > 0 and entry_price > 0
                else realized_pnl_pct
            )
            conn.execute(
                """UPDATE trades SET
                   quantity=?, initial_quantity=COALESCE(initial_quantity, ?),
                   pnl=?, pnl_pct=?
                   WHERE id=?""",
                (
                    remaining_qty,
                    initial_quantity,
                    total_pnl,
                    total_pnl_pct,
                    trade_id,
                ),
            )

    def update_open_trade_position(
        self,
        trade_id: str,
        *,
        entry_price: float,
        quantity: float,
        initial_quantity: float,
        rationale: str,
        confidence: float,
        metadata: dict | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE trades SET
                   entry_price=?,
                   quantity=?,
                   initial_quantity=?,
                   rationale=?,
                   confidence=?,
                   metadata_json=?
                   WHERE id=? AND status='open'""",
                (
                    float(entry_price),
                    float(quantity),
                    float(initial_quantity),
                    str(rationale or ""),
                    float(confidence or 0.0),
                    json.dumps(metadata or {}, default=str),
                    trade_id,
                ),
            )

    def get_open_trades(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='open'"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_closed_trades(self, since: str | None = None) -> list[dict]:
        """获取已平仓交易，可按时间过滤"""
        with self._conn() as conn:
            if since:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE status='closed' AND exit_time >= ? "
                    "ORDER BY exit_time DESC",
                    (since,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE status='closed' "
                    "ORDER BY exit_time DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def insert_pnl_ledger_entry(self, payload: dict) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO pnl_ledger
                   (trade_id, symbol, direction, event_type, event_time,
                    quantity, notional_value, reference_price, fill_price,
                    gross_pnl, fee_cost, slippage_cost, net_pnl, net_return_pct,
                    holding_hours, model_id, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload["trade_id"],
                    payload["symbol"],
                    payload.get("direction", "LONG"),
                    payload["event_type"],
                    payload.get("event_time")
                    or datetime.now(timezone.utc).isoformat(),
                    float(payload.get("quantity", 0.0) or 0.0),
                    float(payload.get("notional_value", 0.0) or 0.0),
                    float(payload.get("reference_price", 0.0) or 0.0),
                    float(payload.get("fill_price", 0.0) or 0.0),
                    float(payload.get("gross_pnl", 0.0) or 0.0),
                    float(payload.get("fee_cost", 0.0) or 0.0),
                    float(payload.get("slippage_cost", 0.0) or 0.0),
                    float(payload.get("net_pnl", 0.0) or 0.0),
                    float(payload.get("net_return_pct", 0.0) or 0.0),
                    float(payload.get("holding_hours", 0.0) or 0.0),
                    str(payload.get("model_id", "") or ""),
                    json.dumps(payload.get("metadata", {}), default=str),
                    payload.get("created_at") or datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def get_pnl_ledger(
        self,
        *,
        limit: int = 100,
        symbol: str | None = None,
        event_type: str | None = None,
        model_id: str | None = None,
        since: str | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM pnl_ledger WHERE 1=1"
        params: list[Any] = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        if since:
            query += " AND event_time >= ?"
            params.append(since)
        query += " ORDER BY event_time DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ── Signals ─────────────────────────────────────────

    def insert_signal(self, signal: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO signals
                   (symbol, source, direction, confidence,
                    rationale, risk_level, raw_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal["symbol"], signal["source"],
                    signal["direction"], signal["confidence"],
                    signal.get("rationale", ""),
                    signal.get("risk_level", "MEDIUM"),
                    json.dumps(signal, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def insert_feature_snapshot(self, snapshot: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO feature_snapshots
                   (symbol, timeframe, timestamp, features_json, valid, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    snapshot["symbol"],
                    snapshot["timeframe"],
                    snapshot["timestamp"],
                    json.dumps(snapshot["features"], default=str),
                    1 if snapshot.get("valid", True) else 0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def insert_prediction_run(self, run: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO prediction_runs
                   (symbol, timestamp, model_version, up_probability, feature_count,
                    research_json, decision_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run["symbol"],
                    run["timestamp"],
                    run["model_version"],
                    run["up_probability"],
                    run["feature_count"],
                    json.dumps(run["research"], default=str),
                    json.dumps(run["decision"], default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def insert_prediction_evaluation(self, evaluation: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO prediction_evaluations
                   (symbol, timestamp, evaluation_type, actual_up, predicted_up, is_correct,
                    entry_close, future_close, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    evaluation["symbol"],
                    evaluation["timestamp"],
                    evaluation["evaluation_type"],
                    1 if evaluation.get("actual_up") else 0,
                    1 if evaluation.get("predicted_up") else 0,
                    1 if evaluation.get("is_correct") else 0,
                    evaluation["entry_close"],
                    evaluation["future_close"],
                    json.dumps(evaluation.get("metadata", {}), default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def has_prediction_evaluation(
        self,
        symbol: str,
        timestamp: str,
        evaluation_type: str,
    ) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT 1 FROM prediction_evaluations
                   WHERE symbol = ? AND timestamp = ? AND evaluation_type = ?
                   LIMIT 1""",
                (symbol, timestamp, evaluation_type),
            ).fetchone()
            return row is not None

    def insert_shadow_trade_run(self, payload: dict) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO shadow_trade_runs
                   (symbol, timestamp, block_reason, direction, entry_price, horizon_hours,
                    status, exit_price, pnl_pct, setup_profile_json, metadata_json,
                    evaluated_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload["symbol"],
                    payload["timestamp"],
                    payload["block_reason"],
                    payload.get("direction", "LONG"),
                    payload["entry_price"],
                    payload["horizon_hours"],
                    payload.get("status", "open"),
                    payload.get("exit_price"),
                    payload.get("pnl_pct"),
                    json.dumps(payload.get("setup_profile", {}), default=str),
                    json.dumps(payload.get("metadata", {}), default=str),
                    payload.get("evaluated_at"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid or 0)

    def get_open_shadow_trade_runs(self, limit: int = 500) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM shadow_trade_runs
                   WHERE status = 'open'
                   ORDER BY created_at ASC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def update_shadow_trade_run(self, run_id: int, payload: dict):
        with self._conn() as conn:
            conn.execute(
                """UPDATE shadow_trade_runs
                   SET status = ?, exit_price = ?, pnl_pct = ?, metadata_json = ?,
                       evaluated_at = ?
                   WHERE id = ?""",
                (
                    payload["status"],
                    payload.get("exit_price"),
                    payload.get("pnl_pct"),
                    json.dumps(payload.get("metadata", {}), default=str),
                    payload.get("evaluated_at"),
                    run_id,
                ),
            )

    def insert_account_snapshot(self, snapshot: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO account_snapshots
                   (timestamp, equity, realized_pnl, unrealized_pnl, daily_loss_pct,
                    weekly_loss_pct, drawdown_pct, total_exposure_pct, open_positions,
                    cooldown_until, circuit_breaker_active, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot["timestamp"],
                    snapshot["equity"],
                    snapshot["realized_pnl"],
                    snapshot["unrealized_pnl"],
                    snapshot["daily_loss_pct"],
                    snapshot["weekly_loss_pct"],
                    snapshot["drawdown_pct"],
                    snapshot["total_exposure_pct"],
                    snapshot["open_positions"],
                    snapshot.get("cooldown_until"),
                    1 if snapshot.get("circuit_breaker_active") else 0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def insert_training_run(self, summary: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO training_runs
                   (symbol, rows, feature_count, positives, negatives, model_path,
                    trained_with_xgboost, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    summary["symbol"],
                    summary["rows"],
                    summary["feature_count"],
                    summary["positives"],
                    summary["negatives"],
                    summary["model_path"],
                    1 if summary.get("trained_with_xgboost") else 0,
                    json.dumps(summary, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def insert_walkforward_run(self, result: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO walkforward_runs
                   (symbol, total_splits, summary_json, splits_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    result["symbol"],
                    result["summary"]["total_splits"],
                    json.dumps(result["summary"], default=str),
                    json.dumps(result["splits"], default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def insert_execution_event(self, event_type: str, symbol: str, payload: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO execution_events
                   (event_type, symbol, payload_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    event_type,
                    symbol,
                    json.dumps(payload, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def insert_report_artifact(
        self,
        report_type: str,
        content: str,
        symbol: str | None = None,
    ):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO report_artifacts
                   (report_type, symbol, content, created_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    report_type,
                    symbol,
                    content,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def insert_reconciliation_run(self, status: str, mismatch_count: int, details: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO reconciliation_runs
                   (status, mismatch_count, details_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    status,
                    mismatch_count,
                    json.dumps(details, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def insert_research_input(self, payload: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO research_inputs
                   (symbol, timestamp, news_summary, macro_summary, fear_greed,
                    onchain_summary, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload["symbol"],
                    payload["timestamp"],
                    payload["news_summary"],
                    payload["macro_summary"],
                    payload.get("fear_greed"),
                    payload.get("onchain_summary", ""),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def insert_backtest_run(
        self,
        symbol: str,
        engine: str,
        summary: dict,
        trades: list[dict],
    ):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO backtest_runs
                   (symbol, engine, summary_json, trades_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    symbol,
                    engine,
                    json.dumps(summary, default=str),
                    json.dumps(trades, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def insert_cycle_run(self, payload: dict) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO cycle_runs
                   (started_at, completed_at, status, symbols_json, opened_positions,
                    closed_positions, circuit_breaker_active, reconciliation_status,
                    notes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload["started_at"],
                    payload.get("completed_at"),
                    payload["status"],
                    json.dumps(payload["symbols"], default=str),
                    payload.get("opened_positions", 0),
                    payload.get("closed_positions", 0),
                    1 if payload.get("circuit_breaker_active") else 0,
                    payload.get("reconciliation_status", "unknown"),
                    payload.get("notes", ""),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def update_cycle_run(self, cycle_id: int, payload: dict):
        with self._conn() as conn:
            conn.execute(
                """UPDATE cycle_runs
                   SET completed_at = ?, status = ?, opened_positions = ?,
                       closed_positions = ?, circuit_breaker_active = ?,
                       reconciliation_status = ?, notes = ?
                   WHERE id = ?""",
                (
                    payload.get("completed_at"),
                    payload["status"],
                    payload.get("opened_positions", 0),
                    payload.get("closed_positions", 0),
                    1 if payload.get("circuit_breaker_active") else 0,
                    payload.get("reconciliation_status", "unknown"),
                    payload.get("notes", ""),
                    cycle_id,
                ),
            )

    def insert_scheduler_run(self, payload: dict) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO scheduler_runs
                   (job_name, status, output, started_at, completed_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    payload["job_name"],
                    payload["status"],
                    payload.get("output", ""),
                    payload["started_at"],
                    payload.get("completed_at"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def update_scheduler_run(self, run_id: int, payload: dict):
        with self._conn() as conn:
            conn.execute(
                """UPDATE scheduler_runs
                   SET status = ?, output = ?, completed_at = ?
                   WHERE id = ?""",
                (
                    payload["status"],
                    payload.get("output", ""),
                    payload.get("completed_at"),
                    run_id,
                ),
            )

    def insert_ab_test_run(self, payload: dict) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO ab_test_runs
                   (symbol, timestamp, champion_model_version, challenger_model_version,
                    champion_probability, challenger_probability, champion_execute,
                    challenger_execute, selected_variant, allocation_pct, notes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload["symbol"],
                    payload["timestamp"],
                    payload["champion_model_version"],
                    payload["challenger_model_version"],
                    payload["champion_probability"],
                    payload["challenger_probability"],
                    1 if payload.get("champion_execute") else 0,
                    1 if payload.get("challenger_execute") else 0,
                    payload["selected_variant"],
                    payload.get("allocation_pct", 0.0),
                    payload.get("notes", ""),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def upsert_model_registry(self, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        metadata = payload.get("metadata", {})
        created_at = payload.get("created_at") or now
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO model_registry
                   (symbol, model_id, model_version, model_path, role, stage, active,
                    metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(model_id) DO UPDATE SET
                       symbol=excluded.symbol,
                       model_version=excluded.model_version,
                       model_path=excluded.model_path,
                       role=excluded.role,
                       stage=excluded.stage,
                       active=excluded.active,
                       metadata_json=excluded.metadata_json,
                       updated_at=excluded.updated_at""",
                (
                    payload["symbol"],
                    payload["model_id"],
                    payload.get("model_version", ""),
                    payload.get("model_path", ""),
                    payload.get("role", ""),
                    payload.get("stage", ""),
                    1 if payload.get("active") else 0,
                    json.dumps(metadata, default=str),
                    created_at,
                    now,
                ),
            )

    def insert_model_scorecard(self, payload: dict) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO model_scorecards
                   (symbol, model_id, model_version, stage, evaluation_type,
                   sample_count, executed_count, accuracy, executed_precision,
                   avg_trade_return_pct, total_trade_return_pct, expectancy_pct,
                   profit_factor, max_drawdown_pct, trade_win_rate, avg_cost_pct,
                   avg_favorable_excursion_pct, avg_adverse_excursion_pct, objective_score,
                   metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload["symbol"],
                    payload["model_id"],
                    payload.get("model_version", ""),
                    payload.get("stage", ""),
                    payload.get("evaluation_type", ""),
                    int(payload.get("sample_count", 0) or 0),
                    int(payload.get("executed_count", 0) or 0),
                    float(payload.get("accuracy", 0.0) or 0.0),
                    float(payload.get("executed_precision", 0.0) or 0.0),
                    float(payload.get("avg_trade_return_pct", 0.0) or 0.0),
                    float(payload.get("total_trade_return_pct", 0.0) or 0.0),
                    float(payload.get("expectancy_pct", 0.0) or 0.0),
                    float(payload.get("profit_factor", 0.0) or 0.0),
                    float(payload.get("max_drawdown_pct", 0.0) or 0.0),
                    float(payload.get("trade_win_rate", 0.0) or 0.0),
                    float(payload.get("avg_cost_pct", 0.0) or 0.0),
                    float(payload.get("avg_favorable_excursion_pct", 0.0) or 0.0),
                    float(payload.get("avg_adverse_excursion_pct", 0.0) or 0.0),
                    float(payload.get("objective_score", 0.0) or 0.0),
                    json.dumps(payload.get("metadata", {}), default=str),
                    payload.get("created_at") or datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def delete_older_than(self, table: str, column: str, cutoff_iso: str) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE {column} < ?",
                (cutoff_iso,),
            )
            return cursor.rowcount

    def reset_runtime_data(self, preserve_state_keys: list[str] | None = None) -> dict[str, int]:
        preserve_state_keys = preserve_state_keys or []
        summary: dict[str, int] = {}
        with self._conn() as conn:
            for table in self.RUNTIME_TABLES:
                cursor = conn.execute(f"DELETE FROM {table}")
                summary[table] = cursor.rowcount
            if preserve_state_keys:
                placeholders = ",".join(["?"] * len(preserve_state_keys))
                cursor = conn.execute(
                    f"DELETE FROM system_state WHERE key NOT IN ({placeholders})",
                    tuple(preserve_state_keys),
                )
            else:
                cursor = conn.execute("DELETE FROM system_state")
            summary["system_state"] = cursor.rowcount
        return summary

    def vacuum(self) -> None:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()

    # ── Sentiment ───────────────────────────────────────

    def insert_sentiment(self, data: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO sentiment (source, value, label, summary, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    data["source"], data.get("value"),
                    data.get("label", ""), data.get("summary", ""),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    # ── Positions ───────────────────────────────────────

    def upsert_position(self, pos: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO positions
                   (symbol, direction, entry_price, quantity,
                    entry_time, stop_loss, take_profit, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pos["symbol"], pos["direction"],
                    pos["entry_price"], pos["quantity"],
                    pos["entry_time"],
                    pos.get("stop_loss"), pos.get("take_profit"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_positions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM positions").fetchall()
            return [dict(r) for r in rows]

    def delete_position(self, symbol: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))

    # ── State ───────────────────────────────────────────

    def get_state(self, key: str, default: str | None = None) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key=?", (key,)
            ).fetchone()
            return row["value"] if row else default

    def set_state(self, key: str, value: str):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO system_state (key, value, updated_at)
                   VALUES (?, ?, ?)""",
                (key, value, datetime.now(timezone.utc).isoformat()),
            )

    def get_json_state(self, key: str, default: Any | None = None) -> Any | None:
        raw = self.get_state(key)
        if raw in (None, ""):
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON system_state payload for key={key}")
            return default

    def set_json_state(self, key: str, value: Any):
        self.set_state(key, json.dumps(value, default=str))

    # ── Reflections ─────────────────────────────────────

    def _insert_reflection(self, reflection) -> str:
        """插入交易反思记录"""
        trade_id = str(reflection.trade_id)
        source = str(getattr(reflection, "source", "") or "").strip()
        if not source:
            source = "shadow_observation" if trade_id.startswith("shadow:") else "trade"
        raw_weight = getattr(reflection, "experience_weight", 0.0)
        try:
            experience_weight = float(raw_weight or 0.0)
        except (TypeError, ValueError):
            experience_weight = 0.0
        if experience_weight <= 0:
            experience_weight = 0.35 if source == "shadow_observation" else 1.0
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO reflections
                   (trade_id, symbol, direction, confidence, rationale, source,
                    experience_weight,
                    realized_return_pct, outcome_24h, outcome_7d, correct_signals,
                    wrong_signals, lesson, market_regime, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade_id, reflection.symbol,
                    reflection.direction, reflection.confidence,
                    reflection.rationale,
                    source,
                    experience_weight,
                    (
                        reflection.realized_return_pct
                        if getattr(reflection, "realized_return_pct", None) is not None
                        else reflection.outcome_24h
                    ),
                    reflection.outcome_24h,
                    reflection.outcome_7d,
                    json.dumps(reflection.correct_signals),
                    json.dumps(reflection.wrong_signals),
                    reflection.lesson,
                    reflection.market_regime.value,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return reflection.trade_id

    def get_similar_lessons(
        self, symbol: str, direction: str, limit: int = 5
    ) -> list[str]:
        """获取类似交易的经验"""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT lesson FROM reflections
                   WHERE symbol = ? AND direction = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (symbol, direction, limit),
            ).fetchall()
            return [r["lesson"] for r in rows if r["lesson"]]
