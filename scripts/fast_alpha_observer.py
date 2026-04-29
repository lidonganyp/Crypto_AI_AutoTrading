from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path("data/cryptoai.db")
LOG_PATH = Path("data/reports/fast_alpha_observation.log")
STATE_PATH = Path("data/runtime/fast_alpha_observer_state.json")
EVENT_TYPES = (
    "fast_alpha_open",
    "fast_alpha_blocked",
    "close",
    "position_review_watch",
)
POLL_SECONDS = 300


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"last_id": 0}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"last_id": 0}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False),
        encoding="utf-8",
    )


def append(text: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")


def current_positions(conn: sqlite3.Connection) -> list[dict]:
    return [
        dict(row)
        for row in conn.execute(
            "SELECT symbol, direction, quantity, entry_price, entry_time, updated_at "
            "FROM positions ORDER BY updated_at DESC"
        ).fetchall()
    ]


def format_row(row: sqlite3.Row) -> str:
    payload = json.loads(row["payload_json"] or "{}")
    return json.dumps(
        {
            "id": row["id"],
            "ts": row["created_at"],
            "event_type": row["event_type"],
            "symbol": row["symbol"],
            "reason": payload.get("reason"),
            "pnl_pct": payload.get("pnl_pct"),
            "position_value": payload.get("position_value"),
            "review_score": payload.get("review_score"),
            "model_evidence_scale": payload.get("model_evidence_scale"),
            "model_evidence_source": payload.get("model_evidence_source"),
            "model_evidence_reason": payload.get("model_evidence_reason"),
            "research_exit_count": payload.get("research_exit_count"),
        },
        ensure_ascii=False,
    )


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = load_state()
    append(
        f"\n=== observer_resume {utc_now()} last_id={int(state.get('last_id', 0) or 0)} ==="
    )
    while True:
        try:
            state = load_state()
            last_id = int(state.get("last_id", 0) or 0)
            placeholders = ",".join("?" for _ in EVENT_TYPES)
            with connect() as conn:
                rows = conn.execute(
                    f"SELECT id, event_type, symbol, payload_json, created_at "
                    f"FROM execution_events WHERE id > ? "
                    f"AND event_type IN ({placeholders}) "
                    f"ORDER BY id ASC",
                    (last_id, *EVENT_TYPES),
                ).fetchall()
                if rows:
                    max_id = max(int(row["id"]) for row in rows)
                    append(
                        f"\n=== observation_tick {utc_now()} rows={len(rows)} last_id={max_id} ==="
                    )
                    for row in rows:
                        append(format_row(row))
                    append(json.dumps({"positions": current_positions(conn)}, ensure_ascii=False))
                    state["last_id"] = max_id
                    save_state(state)
        except Exception as exc:
            append(f"observer_error {utc_now()} {exc}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
