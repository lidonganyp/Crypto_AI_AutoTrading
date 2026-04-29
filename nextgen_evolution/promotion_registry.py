"""Persistence layer for next-generation experiment and promotion history."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone

from .experiment_lab import ExperimentResult
from .models import (
    AutonomyDirective,
    ExecutionAction,
    ExecutionDirective,
    ExecutionIntent,
    PortfolioAllocation,
    PortfolioPerformanceSnapshot,
    PromotionStage,
    RepairPlan,
    RuntimeEvidenceSnapshot,
    RuntimeLifecycleState,
    RuntimeState,
    StrategyGenome,
)
from .runtime_override_policy import (
    hydrate_runtime_policy_notes,
    strip_legacy_runtime_policy_notes,
)


class PromotionRegistry:
    """Persist experiment runs, candidate scorecards, and capital allocations."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_tables()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_tables(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nextgen_experiment_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    candle_count INTEGER NOT NULL,
                    evaluated_candidates INTEGER NOT NULL,
                    promoted_candidates INTEGER NOT NULL,
                    allocated_candidates INTEGER NOT NULL,
                    notes_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS nextgen_candidate_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    strategy_id TEXT NOT NULL,
                    family TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    edge_score REAL NOT NULL,
                    robustness_score REAL NOT NULL,
                    deployment_score REAL NOT NULL,
                    total_score REAL NOT NULL,
                    mutation_of TEXT,
                    params_json TEXT NOT NULL DEFAULT '{}',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES nextgen_experiment_runs(id)
                );

                CREATE TABLE IF NOT EXISTS nextgen_allocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    strategy_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    allocated_capital REAL NOT NULL,
                    weight REAL NOT NULL,
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES nextgen_experiment_runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_nextgen_runs_symbol_timeframe
                ON nextgen_experiment_runs(symbol, timeframe, created_at);

                CREATE INDEX IF NOT EXISTS idx_nextgen_scores_run_id
                ON nextgen_candidate_scores(run_id, total_score DESC);

                CREATE INDEX IF NOT EXISTS idx_nextgen_allocations_run_id
                ON nextgen_allocations(run_id, weight DESC);

                CREATE TABLE IF NOT EXISTS nextgen_portfolio_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_capital REAL NOT NULL,
                    allocated_capital REAL NOT NULL,
                    reserve_capital REAL NOT NULL,
                    symbol_count INTEGER NOT NULL,
                    allocation_count INTEGER NOT NULL,
                    notes_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS nextgen_portfolio_allocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_run_id INTEGER NOT NULL,
                    experiment_run_id INTEGER,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL DEFAULT '',
                    strategy_id TEXT NOT NULL,
                    family TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    allocated_capital REAL NOT NULL,
                    weight REAL NOT NULL,
                    score REAL NOT NULL,
                    entry_price REAL NOT NULL DEFAULT 0.0,
                    last_price REAL NOT NULL DEFAULT 0.0,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    marked_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (portfolio_run_id) REFERENCES nextgen_portfolio_runs(id),
                    FOREIGN KEY (experiment_run_id) REFERENCES nextgen_experiment_runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_nextgen_portfolio_runs_created_at
                ON nextgen_portfolio_runs(created_at, id);

                CREATE INDEX IF NOT EXISTS idx_nextgen_portfolio_allocations_run_id
                ON nextgen_portfolio_allocations(portfolio_run_id, weight DESC);

                CREATE TABLE IF NOT EXISTS nextgen_portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_run_id INTEGER NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
                    equity REAL NOT NULL DEFAULT 0.0,
                    gross_exposure REAL NOT NULL DEFAULT 0.0,
                    net_exposure REAL NOT NULL DEFAULT 0.0,
                    open_positions INTEGER NOT NULL DEFAULT 0,
                    closed_positions INTEGER NOT NULL DEFAULT 0,
                    win_rate REAL NOT NULL DEFAULT 0.0,
                    max_drawdown_pct REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'active',
                    notes_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (portfolio_run_id) REFERENCES nextgen_portfolio_runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_nextgen_portfolio_snapshots_run_id
                ON nextgen_portfolio_snapshots(portfolio_run_id, created_at DESC, id DESC);

                CREATE TABLE IF NOT EXISTS nextgen_autonomy_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_run_id INTEGER,
                    strategy_count INTEGER NOT NULL,
                    execution_count INTEGER NOT NULL,
                    repair_count INTEGER NOT NULL,
                    quarantine_count INTEGER NOT NULL,
                    retire_count INTEGER NOT NULL,
                    quarantined_json TEXT NOT NULL DEFAULT '[]',
                    retired_json TEXT NOT NULL DEFAULT '[]',
                    notes_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (portfolio_run_id) REFERENCES nextgen_portfolio_runs(id)
                );

                CREATE TABLE IF NOT EXISTS nextgen_execution_directives (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    autonomy_cycle_id INTEGER NOT NULL,
                    strategy_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    from_stage TEXT NOT NULL,
                    target_stage TEXT NOT NULL,
                    capital_multiplier REAL NOT NULL DEFAULT 1.0,
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (autonomy_cycle_id) REFERENCES nextgen_autonomy_cycles(id)
                );

                CREATE TABLE IF NOT EXISTS nextgen_repair_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    autonomy_cycle_id INTEGER NOT NULL,
                    strategy_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    validation_stage TEXT NOT NULL,
                    capital_multiplier REAL NOT NULL DEFAULT 1.0,
                    runtime_overrides_json TEXT NOT NULL DEFAULT '{}',
                    candidate_strategy_id TEXT,
                    candidate_family TEXT,
                    candidate_params_json TEXT NOT NULL DEFAULT '{}',
                    candidate_tags_json TEXT NOT NULL DEFAULT '[]',
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (autonomy_cycle_id) REFERENCES nextgen_autonomy_cycles(id)
                );

                CREATE INDEX IF NOT EXISTS idx_nextgen_autonomy_cycles_created_at
                ON nextgen_autonomy_cycles(created_at DESC, id DESC);

                CREATE INDEX IF NOT EXISTS idx_nextgen_execution_directives_cycle_id
                ON nextgen_execution_directives(autonomy_cycle_id, id DESC);

                CREATE INDEX IF NOT EXISTS idx_nextgen_repair_plans_cycle_id
                ON nextgen_repair_plans(autonomy_cycle_id, priority DESC, id DESC);

                CREATE TABLE IF NOT EXISTS nextgen_repair_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    autonomy_cycle_id INTEGER,
                    source_strategy_id TEXT NOT NULL,
                    source_runtime_id TEXT NOT NULL DEFAULT '',
                    candidate_strategy_id TEXT NOT NULL,
                    candidate_family TEXT NOT NULL DEFAULT '',
                    action TEXT NOT NULL,
                    validation_stage TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    experiment_run_id INTEGER,
                    status TEXT NOT NULL DEFAULT '',
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    notes_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (autonomy_cycle_id) REFERENCES nextgen_autonomy_cycles(id),
                    FOREIGN KEY (experiment_run_id) REFERENCES nextgen_experiment_runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_nextgen_repair_executions_cycle_id
                ON nextgen_repair_executions(autonomy_cycle_id, id DESC);

                CREATE INDEX IF NOT EXISTS idx_nextgen_repair_executions_run_id
                ON nextgen_repair_executions(experiment_run_id, id DESC);

                CREATE TABLE IF NOT EXISTS nextgen_runtime_states (
                    runtime_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    family TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL,
                    promotion_stage TEXT NOT NULL,
                    target_stage TEXT NOT NULL,
                    last_directive_action TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0.0,
                    allocated_capital REAL NOT NULL DEFAULT 0.0,
                    desired_capital REAL NOT NULL DEFAULT 0.0,
                    current_capital REAL NOT NULL DEFAULT 0.0,
                    current_weight REAL NOT NULL DEFAULT 0.0,
                    capital_multiplier REAL NOT NULL DEFAULT 1.0,
                    limited_live_cycles INTEGER NOT NULL DEFAULT 0,
                    notes_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_nextgen_runtime_states_symbol
                ON nextgen_runtime_states(symbol, updated_at DESC);

                CREATE TABLE IF NOT EXISTS nextgen_execution_intents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    autonomy_cycle_id INTEGER,
                    runtime_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    family TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL,
                    action TEXT NOT NULL,
                    desired_capital REAL NOT NULL DEFAULT 0.0,
                    current_capital REAL NOT NULL DEFAULT 0.0,
                    price REAL NOT NULL DEFAULT 0.0,
                    quantity REAL NOT NULL DEFAULT 0.0,
                    close_quantity REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'planned',
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    notes_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (autonomy_cycle_id) REFERENCES nextgen_autonomy_cycles(id)
                );

                CREATE INDEX IF NOT EXISTS idx_nextgen_execution_intents_cycle_id
                ON nextgen_execution_intents(autonomy_cycle_id, id DESC);

                CREATE INDEX IF NOT EXISTS idx_nextgen_execution_intents_runtime_id
                ON nextgen_execution_intents(runtime_id, id DESC);

                CREATE TABLE IF NOT EXISTS nextgen_runtime_evidence_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    runtime_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    family TEXT NOT NULL,
                    open_position INTEGER NOT NULL DEFAULT 0,
                    current_capital REAL NOT NULL DEFAULT 0.0,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
                    total_net_pnl REAL NOT NULL DEFAULT 0.0,
                    current_drawdown_pct REAL NOT NULL DEFAULT 0.0,
                    max_drawdown_pct REAL NOT NULL DEFAULT 0.0,
                    closed_trade_count INTEGER NOT NULL DEFAULT 0,
                    win_rate REAL NOT NULL DEFAULT 0.0,
                    consecutive_losses INTEGER NOT NULL DEFAULT 0,
                    health_status TEXT NOT NULL DEFAULT 'unproven',
                    notes_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_nextgen_runtime_evidence_runtime_id
                ON nextgen_runtime_evidence_snapshots(runtime_id, id DESC);
                """
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_runs",
                "status TEXT NOT NULL DEFAULT 'active'",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_runs",
                "latest_realized_pnl REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_runs",
                "latest_unrealized_pnl REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_runs",
                "latest_equity REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_runs",
                "latest_gross_exposure REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_runs",
                "latest_net_exposure REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_runs",
                "latest_open_positions INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_runs",
                "latest_closed_positions INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_runs",
                "latest_win_rate REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_runs",
                "latest_max_drawdown_pct REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_runs",
                "updated_at TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_allocations",
                "timeframe TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_allocations",
                "entry_price REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_allocations",
                "last_price REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_allocations",
                "realized_pnl REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_allocations",
                "unrealized_pnl REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                conn,
                "nextgen_portfolio_allocations",
                "marked_at TEXT NOT NULL DEFAULT ''",
            )

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection,
        table_name: str,
        definition: str,
    ) -> None:
        column_name = definition.split()[0]
        existing = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name in existing:
            return
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {definition}")

    @staticmethod
    def _parse_runtime_id(runtime_id: str) -> tuple[str, str, str]:
        parts = str(runtime_id).split("|", 2)
        if len(parts) != 3:
            return "", "", str(runtime_id)
        return parts[0], parts[1], parts[2]

    def persist_experiment(
        self,
        result: ExperimentResult,
        *,
        notes: dict | None = None,
    ) -> ExperimentResult:
        created_at = datetime.now(timezone.utc).isoformat()
        merged_notes = dict(result.notes)
        merged_notes.update(notes or {})
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO nextgen_experiment_runs
                   (symbol, timeframe, candle_count, evaluated_candidates,
                    promoted_candidates, allocated_candidates, notes_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.symbol,
                    result.timeframe,
                    int(result.candle_count),
                    len(result.scorecards),
                    len(result.promoted),
                    len(result.allocations),
                    json.dumps(merged_notes, default=str),
                    created_at,
                ),
            )
            run_id = int(cursor.lastrowid)
            conn.executemany(
                """INSERT INTO nextgen_candidate_scores
                   (run_id, strategy_id, family, stage, edge_score, robustness_score,
                    deployment_score, total_score, mutation_of, params_json, tags_json,
                    reasons_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        run_id,
                        card.genome.strategy_id,
                        card.genome.family,
                        card.stage.value,
                        card.edge_score,
                        card.robustness_score,
                        card.deployment_score,
                        card.total_score,
                        card.genome.mutation_of,
                        json.dumps(card.genome.params, default=str),
                        json.dumps(list(card.genome.tags), default=str),
                        json.dumps(card.reasons, default=str),
                        created_at,
                    )
                    for card in result.scorecards
                ],
            )
            conn.executemany(
                """INSERT INTO nextgen_allocations
                   (run_id, strategy_id, stage, allocated_capital, weight,
                    reasons_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        run_id,
                        item.strategy_id,
                        item.stage.value,
                        item.allocated_capital,
                        item.weight,
                        json.dumps(item.reasons, default=str),
                        created_at,
                    )
                    for item in result.allocations
                ],
            )
        return replace(result, registry_run_id=run_id)

    def latest_run(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM nextgen_experiment_runs
                   ORDER BY id DESC
                   LIMIT 1"""
            ).fetchone()
        return dict(row) if row else None

    def latest_scores(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM nextgen_candidate_scores
                   ORDER BY id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def persist_autonomy_cycle(
        self,
        directive: AutonomyDirective,
        *,
        portfolio_run_id: int | None = None,
        notes: dict | None = None,
    ) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        merged_notes = dict(directive.notes or {})
        merged_notes.update(notes or {})
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO nextgen_autonomy_cycles
                   (portfolio_run_id, strategy_count, execution_count, repair_count,
                    quarantine_count, retire_count, quarantined_json, retired_json,
                    notes_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    int(portfolio_run_id) if portfolio_run_id is not None else None,
                    int(merged_notes.get("strategy_count", len(directive.execution))),
                    len(directive.execution),
                    len(directive.repairs),
                    len(directive.quarantined),
                    len(directive.retired),
                    json.dumps(directive.quarantined, default=str),
                    json.dumps(directive.retired, default=str),
                    json.dumps(merged_notes, default=str),
                    created_at,
                ),
            )
            autonomy_cycle_id = int(cursor.lastrowid)
            if directive.execution:
                conn.executemany(
                    """INSERT INTO nextgen_execution_directives
                       (autonomy_cycle_id, strategy_id, action, from_stage, target_stage,
                        capital_multiplier, reasons_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            autonomy_cycle_id,
                            item.strategy_id,
                            item.action.value,
                            item.from_stage.value,
                            item.target_stage.value,
                            float(item.capital_multiplier),
                            json.dumps(item.reasons, default=str),
                            created_at,
                        )
                        for item in directive.execution
                    ],
                )
            if directive.repairs:
                conn.executemany(
                    """INSERT INTO nextgen_repair_plans
                       (autonomy_cycle_id, strategy_id, action, priority, validation_stage,
                        capital_multiplier, runtime_overrides_json, candidate_strategy_id,
                        candidate_family, candidate_params_json, candidate_tags_json,
                        reasons_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            autonomy_cycle_id,
                            item.strategy_id,
                            item.action.value,
                            int(item.priority),
                            item.validation_stage.value,
                            float(item.capital_multiplier),
                            json.dumps(item.runtime_overrides or {}, default=str),
                            (
                                item.candidate_genome.strategy_id
                                if item.candidate_genome is not None
                                else None
                            ),
                            (
                                item.candidate_genome.family
                                if item.candidate_genome is not None
                                else None
                            ),
                            json.dumps(
                                item.candidate_genome.params
                                if item.candidate_genome is not None
                                else {},
                                default=str,
                            ),
                            json.dumps(
                                list(item.candidate_genome.tags)
                                if item.candidate_genome is not None
                                else [],
                                default=str,
                            ),
                            json.dumps(item.reasons, default=str),
                            created_at,
                        )
                        for item in directive.repairs
                    ],
                )
        return autonomy_cycle_id

    def latest_autonomy_cycle(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM nextgen_autonomy_cycles
                   ORDER BY id DESC
                   LIMIT 1"""
            ).fetchone()
        return dict(row) if row else None

    def latest_execution_directives(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM nextgen_execution_directives
                   ORDER BY id DESC
                   LIMIT ?""",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def append_execution_directives(
        self,
        autonomy_cycle_id: int,
        directives: list[ExecutionDirective],
        *,
        strategy_count_delta: int = 0,
        notes: dict | None = None,
    ) -> None:
        if not directives:
            return
        created_at = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO nextgen_execution_directives
                   (autonomy_cycle_id, strategy_id, action, from_stage, target_stage,
                    capital_multiplier, reasons_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        int(autonomy_cycle_id),
                        item.strategy_id,
                        item.action.value,
                        item.from_stage.value,
                        item.target_stage.value,
                        float(item.capital_multiplier),
                        json.dumps(item.reasons, default=str),
                        created_at,
                    )
                    for item in directives
                ],
            )
            cycle = conn.execute(
                """SELECT strategy_count, execution_count, notes_json
                   FROM nextgen_autonomy_cycles
                   WHERE id = ?""",
                (int(autonomy_cycle_id),),
            ).fetchone()
            if cycle is None:
                return
            try:
                merged_notes = json.loads(cycle["notes_json"] or "{}")
            except Exception:
                merged_notes = {}
            if not isinstance(merged_notes, dict):
                merged_notes = {}
            merged_notes.update(notes or {})
            conn.execute(
                """UPDATE nextgen_autonomy_cycles
                   SET strategy_count = ?,
                       execution_count = ?,
                       notes_json = ?
                   WHERE id = ?""",
                (
                    int(cycle["strategy_count"] or 0) + int(strategy_count_delta),
                    int(cycle["execution_count"] or 0) + len(directives),
                    json.dumps(merged_notes, default=str),
                    int(autonomy_cycle_id),
                ),
            )

    def latest_repair_plans(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM nextgen_repair_plans
                   ORDER BY id DESC
                   LIMIT ?""",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def persist_repair_execution(
        self,
        plan: RepairPlan,
        result: ExperimentResult,
        *,
        source_runtime_id: str = "",
        autonomy_cycle_id: int | None = None,
        notes: dict | None = None,
    ) -> int:
        candidate = plan.candidate_genome
        if candidate is None:
            raise ValueError("repair execution requires candidate_genome")
        _, _, runtime_strategy_id = self._parse_runtime_id(source_runtime_id or plan.strategy_id)
        source_strategy_id = runtime_strategy_id or candidate.mutation_of or str(plan.strategy_id)
        repair_notes = {}
        if isinstance(result.notes.get("repair_validation"), dict):
            repair_notes.update(result.notes["repair_validation"])
        repair_notes.update(notes or {})
        outcome = result.scorecards[0].stage.value if result.scorecards else "no_score"
        created_at = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO nextgen_repair_executions
                   (autonomy_cycle_id, source_strategy_id, source_runtime_id,
                    candidate_strategy_id, candidate_family, action, validation_stage,
                    symbol, timeframe, experiment_run_id, status, reasons_json,
                    notes_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    int(autonomy_cycle_id) if autonomy_cycle_id is not None else None,
                    source_strategy_id,
                    str(source_runtime_id or plan.strategy_id),
                    candidate.strategy_id,
                    candidate.family,
                    plan.action.value,
                    plan.validation_stage.value,
                    result.symbol,
                    result.timeframe,
                    int(result.registry_run_id) if result.registry_run_id is not None else None,
                    outcome,
                    json.dumps(plan.reasons, default=str),
                    json.dumps(repair_notes, default=str),
                    created_at,
                ),
            )
        return int(cursor.lastrowid)

    def latest_repair_executions(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM nextgen_repair_executions
                   ORDER BY id DESC
                   LIMIT ?""",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_pnl_ledger(self, limit: int = 200) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM pnl_ledger
                   ORDER BY event_time DESC, id DESC
                   LIMIT ?""",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_candidate_genomes(
        self,
        strategy_ids: list[str],
    ) -> dict[str, StrategyGenome]:
        normalized_ids = [
            str(item).strip()
            for item in strategy_ids
            if str(item).strip()
        ]
        if not normalized_ids:
            return {}
        placeholders = ",".join("?" for _ in normalized_ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT strategy_id, family, mutation_of, params_json, tags_json, id
                    FROM nextgen_candidate_scores
                    WHERE strategy_id IN ({placeholders})
                    ORDER BY strategy_id ASC, id DESC""",
                tuple(normalized_ids),
            ).fetchall()
        genomes: dict[str, StrategyGenome] = {}
        for row in rows:
            strategy_id = str(row["strategy_id"] or "")
            if not strategy_id or strategy_id in genomes:
                continue
            try:
                params = json.loads(row["params_json"] or "{}")
            except Exception:
                params = {}
            try:
                tags = json.loads(row["tags_json"] or "[]")
            except Exception:
                tags = []
            genomes[strategy_id] = StrategyGenome(
                strategy_id=strategy_id,
                family=str(row["family"] or ""),
                params=(
                    {str(key): float(value) for key, value in params.items()}
                    if isinstance(params, dict)
                    else {}
                ),
                mutation_of=(
                    str(row["mutation_of"])
                    if row["mutation_of"] not in {None, ""}
                    else None
                ),
                tags=tuple(
                    str(item)
                    for item in tags
                    if str(item).strip()
                ),
            )
        return genomes

    def persist_runtime_states(self, states: list[RuntimeState]) -> None:
        if not states:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO nextgen_runtime_states
                   (runtime_id, symbol, timeframe, strategy_id, family, lifecycle_state,
                    promotion_stage, target_stage, last_directive_action, score,
                    allocated_capital, desired_capital, current_capital, current_weight,
                    capital_multiplier, limited_live_cycles, notes_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(runtime_id) DO UPDATE SET
                       symbol = excluded.symbol,
                       timeframe = excluded.timeframe,
                       strategy_id = excluded.strategy_id,
                       family = excluded.family,
                       lifecycle_state = excluded.lifecycle_state,
                       promotion_stage = excluded.promotion_stage,
                       target_stage = excluded.target_stage,
                       last_directive_action = excluded.last_directive_action,
                       score = excluded.score,
                       allocated_capital = excluded.allocated_capital,
                       desired_capital = excluded.desired_capital,
                       current_capital = excluded.current_capital,
                       current_weight = excluded.current_weight,
                       capital_multiplier = excluded.capital_multiplier,
                       limited_live_cycles = excluded.limited_live_cycles,
                       notes_json = excluded.notes_json,
                       updated_at = excluded.updated_at""",
                [
                    (
                        item.runtime_id,
                        item.symbol,
                        item.timeframe,
                        item.strategy_id,
                        item.family,
                        item.lifecycle_state.value,
                        item.promotion_stage.value,
                        item.target_stage.value,
                        item.last_directive_action.value,
                        float(item.score),
                        float(item.allocated_capital),
                        float(item.desired_capital),
                        float(item.current_capital),
                        float(item.current_weight),
                        float(item.capital_multiplier),
                        int(item.limited_live_cycles),
                        json.dumps(
                            strip_legacy_runtime_policy_notes(item.notes or {}),
                            default=str,
                        ),
                        now,
                        now,
                    )
                    for item in states
                ],
            )

    def latest_runtime_states(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM nextgen_runtime_states
                   ORDER BY updated_at DESC, runtime_id ASC
                   LIMIT ?""",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_runtime_states(self, *, hydrate_legacy: bool = True) -> list[RuntimeState]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM nextgen_runtime_states
                   ORDER BY updated_at DESC, runtime_id ASC"""
            ).fetchall()
        states: list[RuntimeState] = []
        for row in rows:
            try:
                notes = json.loads(row["notes_json"] or "{}")
            except Exception:
                notes = {}
            if hydrate_legacy and isinstance(notes, dict):
                notes = hydrate_runtime_policy_notes(notes)
            states.append(
                RuntimeState(
                    runtime_id=str(row["runtime_id"]),
                    symbol=str(row["symbol"]),
                    timeframe=str(row["timeframe"]),
                    strategy_id=str(row["strategy_id"]),
                    family=str(row["family"]),
                    lifecycle_state=RuntimeLifecycleState(str(row["lifecycle_state"])),
                    promotion_stage=PromotionStage(str(row["promotion_stage"])),
                    target_stage=PromotionStage(str(row["target_stage"])),
                    last_directive_action=ExecutionAction(str(row["last_directive_action"])),
                    score=float(row["score"] or 0.0),
                    allocated_capital=float(row["allocated_capital"] or 0.0),
                    desired_capital=float(row["desired_capital"] or 0.0),
                    current_capital=float(row["current_capital"] or 0.0),
                    current_weight=float(row["current_weight"] or 0.0),
                    capital_multiplier=float(row["capital_multiplier"] or 1.0),
                    limited_live_cycles=int(row["limited_live_cycles"] or 0),
                    notes=notes if isinstance(notes, dict) else {},
                )
            )
        return states

    def persist_execution_intents(
        self,
        intents: list[ExecutionIntent],
        *,
        autonomy_cycle_id: int | None = None,
    ) -> None:
        if not intents:
            return
        created_at = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO nextgen_execution_intents
                   (autonomy_cycle_id, runtime_id, symbol, timeframe, strategy_id, family,
                    lifecycle_state, action, desired_capital, current_capital, price,
                    quantity, close_quantity, status, reasons_json, notes_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        int(autonomy_cycle_id) if autonomy_cycle_id is not None else None,
                        item.runtime_id,
                        item.symbol,
                        item.timeframe,
                        item.strategy_id,
                        item.family,
                        item.lifecycle_state.value,
                        item.action.value,
                        float(item.desired_capital),
                        float(item.current_capital),
                        float(item.price),
                        float(item.quantity),
                        float(item.close_quantity),
                        item.status,
                        json.dumps(item.reasons, default=str),
                        json.dumps(item.notes or {}, default=str),
                        created_at,
                    )
                    for item in intents
                ],
            )

    def latest_execution_intents(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM nextgen_execution_intents
                   ORDER BY id DESC
                   LIMIT ?""",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def persist_runtime_evidence(
        self,
        snapshots: list[RuntimeEvidenceSnapshot],
    ) -> None:
        if not snapshots:
            return
        created_at = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO nextgen_runtime_evidence_snapshots
                   (runtime_id, symbol, timeframe, strategy_id, family, open_position,
                    current_capital, realized_pnl, unrealized_pnl, total_net_pnl,
                    current_drawdown_pct, max_drawdown_pct, closed_trade_count, win_rate,
                    consecutive_losses, health_status, notes_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        item.runtime_id,
                        item.symbol,
                        item.timeframe,
                        item.strategy_id,
                        item.family,
                        1 if item.open_position else 0,
                        float(item.current_capital),
                        float(item.realized_pnl),
                        float(item.unrealized_pnl),
                        float(item.total_net_pnl),
                        float(item.current_drawdown_pct),
                        float(item.max_drawdown_pct),
                        int(item.closed_trade_count),
                        float(item.win_rate),
                        int(item.consecutive_losses),
                        item.health_status,
                        json.dumps(item.notes or {}, default=str),
                        created_at,
                    )
                    for item in snapshots
                ],
            )

    def latest_runtime_evidence(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM nextgen_runtime_evidence_snapshots
                   ORDER BY id DESC
                   LIMIT ?""",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def persist_portfolio(
        self,
        allocations: list[PortfolioAllocation],
        *,
        total_capital: float,
        experiment_results: list[ExperimentResult] | None = None,
        price_by_symbol: dict[str, float] | None = None,
        notes: dict | None = None,
    ) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        allocated_capital = round(sum(item.allocated_capital for item in allocations), 2)
        reserve_capital = round(max(0.0, total_capital - allocated_capital), 2)
        results = experiment_results or []
        run_id_by_symbol = {
            result.symbol: result.registry_run_id
            for result in results
            if result.registry_run_id is not None
        }
        timeframe_by_symbol = {
            result.symbol: result.timeframe
            for result in results
        }
        entry_prices = dict(price_by_symbol or {})
        merged_notes = dict(notes or {})
        if results:
            merged_notes.setdefault(
                "experiment_run_ids",
                [result.registry_run_id for result in results if result.registry_run_id is not None],
            )
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO nextgen_portfolio_runs
                   (total_capital, allocated_capital, reserve_capital, symbol_count,
                    allocation_count, notes_json, created_at, status,
                    latest_realized_pnl, latest_unrealized_pnl, latest_equity,
                    latest_gross_exposure, latest_net_exposure, latest_open_positions,
                    latest_closed_positions, latest_win_rate, latest_max_drawdown_pct, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    float(total_capital),
                    float(allocated_capital),
                    float(reserve_capital),
                    len({item.symbol for item in allocations}),
                    len(allocations),
                    json.dumps(merged_notes, default=str),
                    created_at,
                    "active",
                    0.0,
                    0.0,
                    float(total_capital),
                    float(allocated_capital),
                    float(allocated_capital),
                    len(allocations),
                    0,
                    0.0,
                    0.0,
                    created_at,
                ),
            )
            portfolio_run_id = int(cursor.lastrowid)
            conn.executemany(
                """INSERT INTO nextgen_portfolio_allocations
                   (portfolio_run_id, experiment_run_id, symbol, timeframe, strategy_id, family, stage,
                    allocated_capital, weight, score, entry_price, last_price, realized_pnl,
                    unrealized_pnl, reasons_json, marked_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        portfolio_run_id,
                        run_id_by_symbol.get(item.symbol),
                        item.symbol,
                        item.timeframe or timeframe_by_symbol.get(item.symbol, ""),
                        item.strategy_id,
                        item.family,
                        item.stage.value,
                        item.allocated_capital,
                        item.weight,
                        item.score,
                        float(entry_prices.get(item.symbol, item.entry_price)),
                        float(entry_prices.get(item.symbol, item.mark_price)),
                        float(item.realized_pnl),
                        float(item.unrealized_pnl),
                        json.dumps(item.reasons, default=str),
                        created_at,
                        created_at,
                    )
                    for item in allocations
                ],
            )
        self.persist_portfolio_snapshot(
            portfolio_run_id,
            PortfolioPerformanceSnapshot(
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                equity=float(total_capital),
                gross_exposure=float(allocated_capital),
                net_exposure=float(allocated_capital),
                open_positions=len(allocations),
                closed_positions=0,
                win_rate=0.0,
                max_drawdown_pct=0.0,
                status="active",
                notes={"source": "portfolio_init"},
            ),
        )
        return portfolio_run_id

    def latest_portfolio_run(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM nextgen_portfolio_runs
                   ORDER BY id DESC
                   LIMIT 1"""
            ).fetchone()
        return dict(row) if row else None

    def latest_portfolio_allocations(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM nextgen_portfolio_allocations
                   ORDER BY id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def portfolio_run(self, portfolio_run_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM nextgen_portfolio_runs
                   WHERE id = ?""",
                (int(portfolio_run_id),),
            ).fetchone()
        return dict(row) if row else None

    def portfolio_allocations(self, portfolio_run_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM nextgen_portfolio_allocations
                   WHERE portfolio_run_id = ?
                   ORDER BY weight DESC, id ASC""",
                (int(portfolio_run_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_portfolio_allocation_marks(self, marks: list[dict]) -> None:
        if not marks:
            return
        marked_at = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.executemany(
                """UPDATE nextgen_portfolio_allocations
                   SET last_price = ?,
                       realized_pnl = ?,
                       unrealized_pnl = ?,
                       marked_at = ?
                   WHERE id = ?""",
                [
                    (
                        float(item["mark_price"]),
                        float(item.get("realized_pnl", 0.0)),
                        float(item.get("unrealized_pnl", 0.0)),
                        marked_at,
                        int(item["portfolio_allocation_id"]),
                    )
                    for item in marks
                ],
            )

    def persist_portfolio_snapshot(
        self,
        portfolio_run_id: int,
        snapshot: PortfolioPerformanceSnapshot,
    ) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        notes_json = json.dumps(snapshot.notes or {}, default=str)
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO nextgen_portfolio_snapshots
                   (portfolio_run_id, realized_pnl, unrealized_pnl, equity, gross_exposure,
                    net_exposure, open_positions, closed_positions, win_rate,
                    max_drawdown_pct, status, notes_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    int(portfolio_run_id),
                    float(snapshot.realized_pnl),
                    float(snapshot.unrealized_pnl),
                    float(snapshot.equity),
                    float(snapshot.gross_exposure),
                    float(snapshot.net_exposure),
                    int(snapshot.open_positions),
                    int(snapshot.closed_positions),
                    float(snapshot.win_rate),
                    float(snapshot.max_drawdown_pct),
                    str(snapshot.status),
                    notes_json,
                    created_at,
                ),
            )
            snapshot_id = int(cursor.lastrowid)
            conn.execute(
                """UPDATE nextgen_portfolio_runs
                   SET status = ?,
                       latest_realized_pnl = ?,
                       latest_unrealized_pnl = ?,
                       latest_equity = ?,
                       latest_gross_exposure = ?,
                       latest_net_exposure = ?,
                       latest_open_positions = ?,
                       latest_closed_positions = ?,
                       latest_win_rate = ?,
                       latest_max_drawdown_pct = ?,
                       updated_at = ?
                   WHERE id = ?""",
                (
                    str(snapshot.status),
                    float(snapshot.realized_pnl),
                    float(snapshot.unrealized_pnl),
                    float(snapshot.equity),
                    float(snapshot.gross_exposure),
                    float(snapshot.net_exposure),
                    int(snapshot.open_positions),
                    int(snapshot.closed_positions),
                    float(snapshot.win_rate),
                    float(snapshot.max_drawdown_pct),
                    created_at,
                    int(portfolio_run_id),
                ),
            )
        return snapshot_id

    def latest_portfolio_snapshots(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM nextgen_portfolio_snapshots
                   ORDER BY id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def portfolio_snapshots(self, portfolio_run_id: int, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM nextgen_portfolio_snapshots
                   WHERE portfolio_run_id = ?
                   ORDER BY id DESC
                   LIMIT ?""",
                (int(portfolio_run_id), int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]
